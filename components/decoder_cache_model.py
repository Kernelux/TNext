"""
Decoder-Only CNN + Cache Model for Mini-ARC
===========================================

A GPT-style decoder-only model using CNN as compute blocks with selective cache memory.
No refinement passes — single forward pass like modern LLMs.

Architecture:
    Input → [Embedding] → [Decoder Layers] → [Output Proj]
    
Each Decoder Layer:
    1. CACHE READ: Query global cache for context
    2. CAUSAL CNN: 1D convolutions with causal masking
    3. CACHE WRITE: Selectively store important patterns

Key Design Decisions:
- Fully causal processing (position i sees ≤ i) — simplest attention pattern
- Single forward pass (no refinement) — efficient like GPT
- Cache provides "long-term memory" beyond CNN receptive field
- Teacher forcing during training (fast, memory-efficient)

Input Format:
    [demo_in₁, demo_out₁, ..., demo_inₙ, demo_outₙ, test_in, test_out]
    
    All tokens processed causally. Loss computed only on test_out.

Usage:
    model = DecoderCacheModel(vocab_size=14, d_model=64)
    logits, cache, aux = model(demo_inputs, demo_outputs, test_input, test_output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

from .config import TrainingConfig, FeatureFlags, SLOT_DIMS
from .memory_controller import MemoryController
from .modules import CacheSelfAttention

# Reuse CNN building blocks from cnn_cache_model
from .cnn_cache_model import CausalConv1d, ConvBlock, DilatedConvStack


# ============================================================================
# Decoder Layer (Single-Pass, No Refinement)
# ============================================================================

class DecoderCacheLayer(nn.Module):
    """
    Decoder layer with causal CNN + cache memory.
    
    Simplified from CNNCacheLayer:
    - No iteration embedding (single pass)
    - Same Read → Compute → Write flow
    - Causal CNN ensures autoregressive generation
    
    Flow:
        Input x
            │
            ▼
        ┌─────────────────┐
        │ CACHE READ      │ ← Query global cache for context
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ CAUSAL CNN      │ ← Local patterns (position i sees ≤ i)
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ CACHE WRITE     │ ← Store important patterns locally
        └─────────────────┘
    """
    
    def __init__(
        self,
        d_model: int,
        d_cache: int,
        num_slots: int,
        num_layers: int,
        layer_idx: int,
        kernel_size: int = 5,
        num_conv_layers: int = 2,
        dropout: float = 0.1,
        soft_eviction: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.layer_idx = layer_idx
        
        # Memory Controller (handles cache read/write)
        self.memory = MemoryController(
            d_model=d_model,
            d_cache=d_cache,
            num_slots=num_slots,
            num_layers=num_layers,
            layer_idx=layer_idx,
            dropout=dropout,
            use_fixed_threshold=True,
            fixed_threshold=0.5,
            soft_eviction=soft_eviction,
        )
        
        # Causal CNN compute
        self.compute = DilatedConvStack(
            d_model=d_model,
            num_layers=num_conv_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        
        # Layer norm for post-compute
        self.post_norm = nn.LayerNorm(d_model)
        
        # Cache self-attention for memory consolidation
        d_slot = SLOT_DIMS.d_slot(d_cache)
        
        # Find valid num_heads that divides d_slot evenly
        cache_attn_heads = 1
        for h in [5, 13]:
            if d_slot % h == 0:
                cache_attn_heads = h
                break
        
        self.cache_self_attn = CacheSelfAttention(
            d_cache=d_slot,
            num_heads=cache_attn_heads,
            dropout=dropout,
            use_linear=False,
        )
    
    def forward(
        self,
        x: torch.Tensor,                # [B, S, D]
        cache: torch.Tensor,            # [B, L*K, D_slot]
        temperature: float = 1.0,
        hard: bool = False,
        cache_layer_embed: Optional[nn.Embedding] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass: Read → Compute → Write
        
        Returns:
            output: [B, S, D] processed features
            updated_cache: [B, L*K, D_slot]
            aux: auxiliary info (gates for monitoring)
        """
        # === CACHE READ: Get global context ===
        read_result = self.memory.read(
            x=x,
            cache=cache,
            use_global=True,
            temperature=temperature,
            hard=hard,
        )
        x_enhanced = read_result['x_enhanced']
        
        # === CAUSAL CNN COMPUTE ===
        output = self.compute(x_enhanced)
        output = self.post_norm(output)

        # === CACHE WRITE: Store important patterns ===
        write_result = self.memory.write(
            output=output,
            cache=cache,
            temperature=temperature,
            hard=hard,
            iteration_idx=0,  # Single pass, always 0
            pass_idx=0,
            cache_layer_embed=cache_layer_embed,
            cache_iter_embed=None,
            cache_pass_embed=None,
            mask=mask,
        )
        updated_cache = write_result['updated_cache']
        
        # === CACHE SELF-ATTENTION ===
        updated_cache = self.cache_self_attn(updated_cache)
        
        # Auxiliary info for monitoring
        aux = {
            'read_gate': read_result['read_gate'].detach().mean().item(),
            'write_gate': write_result['write_gate'].detach().mean().item(),
            'read_gate_tensor': read_result['read_gate'],
            'write_gate_tensor': write_result['write_gate'],
        }
        
        return output, updated_cache, aux


# ============================================================================
# Full Decoder-Only Model
# ============================================================================

class DecoderCacheModel(nn.Module):
    """
    Decoder-only CNN + Cache model for Mini-ARC.
    
    GPT-style architecture:
    - Single forward pass (no refinement)
    - Fully causal processing
    - Cache provides global memory
    
    Training (teacher forcing):
        Input: [demos..., test_in, test_out_shifted]
        Loss:  Only on test_out tokens
        
    Inference (autoregressive):
        1. Process prefix [demos..., test_in]
        2. Generate test_out token-by-token
    """
    
    def __init__(
        self,
        vocab_size: int = 14,
        d_model: int = 64,
        d_cache: int = 48,
        num_layers: int = 4,
        num_slots: int = 16,
        kernel_size: int = 5,
        num_conv_layers_per_block: int = 2,
        max_seq_len: int = 256,  # Total sequence length
        dropout: float = 0.1,
        soft_eviction: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.total_slots = num_layers * num_slots
        self.soft_eviction = soft_eviction
        
        # Slot dimensions from config
        self.d_meta = SLOT_DIMS.d_meta
        self.d_slot = SLOT_DIMS.d_slot(d_cache)
        self.max_seq_len = max_seq_len
        
        # === Embeddings ===
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Factored position embedding (for grid data):
        # Instead of 7200 positions (921K params), use:
        # - row_embed: 30 positions (30 × d_model)
        # - col_embed: 30 positions (30 × d_model)  
        # - grid_embed: 8 grids (8 × d_model) - which grid in sequence
        # Total: 68 × d_model vs 7200 × d_model = 106x reduction!
        self.grid_size = 30  # MAX_GRID_SIZE
        self.max_grids = 8   # 3 demos × 2 (in/out) + test_in + test_out
        self.row_embed = nn.Embedding(self.grid_size, d_model)
        self.col_embed = nn.Embedding(self.grid_size, d_model)
        self.grid_embed = nn.Embedding(self.max_grids, d_model)
        
        # Segment embedding: 0=context (demos + test_in), 1=generation (test_out)
        self.segment_embed = nn.Embedding(2, d_model)
        
        # Learned slot embeddings (initialize cache)
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_layers, num_slots, d_cache) * 0.02
        )
        
        # Layer-ID embeddings for cache
        self.layer_id_embeddings = nn.Parameter(
            torch.randn(num_layers, SLOT_DIMS.d_layer_embed) * 0.02
        )
        
        # Cache layer embedding (shared across layers for write)
        self.cache_layer_embed = nn.Embedding(num_layers, SLOT_DIMS.d_layer_embed)
        
        # === Decoder Layers ===
        self.layers = nn.ModuleList([
            DecoderCacheLayer(
                d_model=d_model,
                d_cache=d_cache,
                num_slots=num_slots,
                num_layers=num_layers,
                layer_idx=i,
                kernel_size=kernel_size,
                num_conv_layers=num_conv_layers_per_block,
                dropout=dropout,
                soft_eviction=soft_eviction,
            )
            for i in range(num_layers)
        ])
        
        # === Output ===
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)
        nn.init.normal_(self.grid_embed.weight, std=0.02)
        nn.init.normal_(self.segment_embed.weight, std=0.02)
        nn.init.normal_(self.cache_layer_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def get_initial_cache(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize cache with learned slot embeddings."""
        # Content from learned embeddings
        content = self.slot_embeddings.view(self.total_slots, self.d_cache)
        content = content.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Confidence: 0 for empty slots
        confidence = torch.zeros(batch_size, self.total_slots, 1, device=device)
        
        # Layer identity
        layer_ids = self.layer_id_embeddings.unsqueeze(1).expand(
            -1, self.num_slots, -1
        ).reshape(self.total_slots, SLOT_DIMS.d_layer_embed)
        layer_ids = layer_ids.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Iteration and pass: zeros (not used in decoder-only)
        iter_pass_dim = SLOT_DIMS.d_iter_embed + SLOT_DIMS.d_pass_embed
        iter_pass = torch.zeros(batch_size, self.total_slots, iter_pass_dim, device=device)
        
        # Combine: [content, confidence, layer_id, iter_embed, pass_embed]
        temporal = torch.cat([layer_ids, iter_pass], dim=-1)
        metadata = torch.cat([confidence, temporal], dim=-1)
        cache = torch.cat([content, metadata], dim=-1)
        
        return cache
    
    def embed_sequence(
        self,
        tokens: torch.Tensor,       # [B, S] where S = grid_size^2 = 900
        grid_idx: int = 0,          # Which grid (0-7)
        segment_id: int = 0,        # 0=context, 1=generation
        start_pos: int = 0,         # Starting position within grid (for generation)
    ) -> torch.Tensor:
        """
        Embed a token sequence with factored position and segment info.
        
        For a 30x30 grid flattened to 900 tokens:
        - Position i maps to row = i // 30, col = i % 30
        - grid_idx identifies which grid in the sequence (0-7)
        - start_pos: offset for position calculation (used in autoregressive generation)
        """
        B, S = tokens.shape
        device = tokens.device
        
        # Token embedding
        tok_emb = self.token_embed(tokens)
        
        # Factored position embedding
        # For flattened grid: position i → (row=i//30, col=i%30)
        # start_pos allows correct positioning during autoregressive generation
        positions = torch.arange(start_pos, start_pos + S, device=device)
        rows = (positions // self.grid_size).clamp(0, self.grid_size - 1)
        cols = (positions % self.grid_size).clamp(0, self.grid_size - 1)
        
        row_emb = self.row_embed(rows)  # [S, D]
        col_emb = self.col_embed(cols)  # [S, D]
        
        # Grid index embedding (which grid in sequence)
        grid_emb = self.grid_embed(torch.tensor(min(grid_idx, self.max_grids - 1), device=device))
        
        # Segment embedding
        seg_emb = self.segment_embed(torch.tensor(segment_id, device=device))
        
        # Combine: token + row + col + grid + segment
        return tok_emb + row_emb + col_emb + grid_emb + seg_emb
    
    def forward(
        self,
        demo_inputs: torch.Tensor,      # [B, num_demos, S]
        demo_outputs: torch.Tensor,     # [B, num_demos, S]
        test_input: torch.Tensor,       # [B, S]
        test_output: Optional[torch.Tensor] = None,  # [B, S] for training
        temperature: float = 1.0,
        hard: bool = False,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with teacher forcing.
        
        Training: Include test_output, loss computed on these positions
        Inference: Omit test_output, use generate() for autoregressive
        
        Args:
            demo_inputs: Demo input grids [B, num_demos, S]
            demo_outputs: Demo output grids [B, num_demos, S]
            test_input: Test input grid [B, S]
            test_output: Test output grid [B, S] (optional, for teacher forcing)
            temperature: Gumbel-softmax temperature
            hard: Use hard decisions
            return_aux: Return auxiliary info
        
        Returns:
            logits: [B, S_gen, vocab] logits for generation positions
            cache: Final cache state
            aux: Auxiliary info (gates, etc.)
        """
        B = test_input.shape[0]
        device = test_input.device
        num_demos = demo_inputs.shape[1]
        S = test_input.shape[1]  # Sequence length per segment (900 for 30x30)
        
        # === Build full sequence ===
        # [demo_in_0, demo_out_0, ..., demo_in_N, demo_out_N, test_in, (test_out)]
        # Each grid gets a unique grid_idx (0-7)
        all_parts = []
        all_masks = []  # For cache write masking (1=valid, 0=pad)
        grid_idx = 0
        
        # Demos (segment_id=0 for context)
        for d in range(num_demos):
            demo_in_emb = self.embed_sequence(demo_inputs[:, d], grid_idx=grid_idx, segment_id=0)
            grid_idx += 1
            all_parts.append(demo_in_emb)
            all_masks.append((demo_inputs[:, d] != 0).float().unsqueeze(-1))
            
            demo_out_emb = self.embed_sequence(demo_outputs[:, d], grid_idx=grid_idx, segment_id=0)
            grid_idx += 1
            all_parts.append(demo_out_emb)
            all_masks.append((demo_outputs[:, d] != 0).float().unsqueeze(-1))
        
        # Test input (segment_id=0, still context)
        test_in_emb = self.embed_sequence(test_input, grid_idx=grid_idx, segment_id=0)
        test_in_grid_idx = grid_idx
        grid_idx += 1
        all_parts.append(test_in_emb)
        all_masks.append((test_input != 0).float().unsqueeze(-1))
        
        # === ADD OUTPUT_MARKER as generation trigger ===
        # This single token signals "now start predicting the output grid"
        # Without this, the model sees [...PAD, PAD, PAD] and must suddenly predict colors
        OUTPUT_MARKER = 3  # From dataset.py vocab
        output_marker_token = torch.full((B, 1), OUTPUT_MARKER, dtype=torch.long, device=device)
        # Embed with segment_id=1 to signal "generation mode"
        output_marker_emb = self.embed_sequence(output_marker_token, grid_idx=test_in_grid_idx, segment_id=1, start_pos=0)
        all_parts.append(output_marker_emb)
        all_masks.append(torch.ones((B, 1, 1), device=device))
        
        # Track where generation starts (count tokens so far, AFTER the output marker)
        gen_start_pos = sum(part.shape[1] for part in all_parts)
        
        # Test output (segment_id=1 for generation, if provided)
        # CRITICAL: For proper teacher forcing, we predict token i from tokens 0..i-1
        # So we include test_output[:-1] as input to predict test_output[1:]
        # BUT: We want to predict ALL of test_output, so:
        #   - The OUTPUT_MARKER predicts test_output[0]
        #   - test_output[i] predicts test_output[i+1]
        if test_output is not None:
            # Shift: include test_output[:-1] as input, predict test_output[:]
            # The model at position p should predict token at position p+1
            test_out_shifted = test_output[:, :-1].clone()  # All but last token
            
            # CRITICAL: Replace IGNORE_LABEL (-100) with PAD_TOKEN (0) for embedding
            # The dataset uses -100 for padding in test_output to mask loss,
            # but we need valid tokens (0) for the input embedding.
            test_out_shifted[test_out_shifted == -100] = 0
            
            # CRITICAL: Use SAME grid_idx as test_input since they share spatial positions!
            # The aligned sequence encoding means position i in test_output corresponds to
            # the same (row, col) as position i in test_input. Only segment_id differs.
            # Start positions from 0 since OUTPUT_MARKER was at start_pos=0 (conceptually position -1)
            test_out_emb = self.embed_sequence(test_out_shifted, grid_idx=test_in_grid_idx, segment_id=1, start_pos=0)
            all_parts.append(test_out_emb)
            all_masks.append((test_out_shifted != 0).float().unsqueeze(-1))
        
        # Concatenate all
        h = torch.cat(all_parts, dim=1)  # [B, total_len, D]
        write_mask = torch.cat(all_masks, dim=1)  # [B, total_len, 1]
        
        # === Initialize cache ===
        cache = self.get_initial_cache(B, device)
        
        # === Process through decoder layers ===
        aux = {
            'read_gates': [],
            'write_gates': [],
        }
        
        for layer_idx, layer in enumerate(self.layers):
            h, cache, layer_aux = layer(
                x=h,
                cache=cache,
                temperature=temperature,
                hard=hard,
                cache_layer_embed=self.cache_layer_embed,
                mask=write_mask,
            )
            aux['read_gates'].append(layer_aux['read_gate_tensor'])
            aux['write_gates'].append(layer_aux['write_gate_tensor'])
        
        # === Output projection ===
        h_out = self.output_norm(h)
        logits = self.output_proj(h_out)  # [B, total_len, vocab]
        
        # Extract generation logits with proper teacher forcing:
        # - Logit at position gen_start_pos-1 predicts test_output[0] (from test_input only)
        # - Logit at position gen_start_pos+i-1 predicts test_output[i] (from test_input + test_output[:i])
        # So we need logits at positions [gen_start_pos-1, gen_start_pos-1+S) to predict test_output[:S]
        if test_output is not None:
            # Logits at positions [gen_start_pos-1, gen_start_pos-1+S) predict test_output[:S]
            gen_logits = logits[:, gen_start_pos-1:gen_start_pos-1+S, :]
        else:
            # No test_output: return logits at last position (for generation)
            gen_logits = logits[:, -1:, :]
        
        return gen_logits, cache, aux
    
    @torch.no_grad()
    def generate(
        self,
        demo_inputs: torch.Tensor,      # [B, num_demos, S]
        demo_outputs: torch.Tensor,     # [B, num_demos, S]
        test_input: torch.Tensor,       # [B, S]
        max_len: int = 25,              # Maximum generation length
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation for inference.
        
        Args:
            demo_inputs, demo_outputs, test_input: Same as forward
            max_len: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = greedy via argmax)
        
        Returns:
            generated: [B, max_len] generated token indices
        """
        B = test_input.shape[0]
        device = test_input.device
        num_demos = demo_inputs.shape[1]
        S = test_input.shape[1]
        
        # Build prefix sequence with grid indices
        all_parts = []
        grid_idx = 0
        
        for d in range(num_demos):
            demo_in_emb = self.embed_sequence(demo_inputs[:, d], grid_idx=grid_idx, segment_id=0)
            grid_idx += 1
            all_parts.append(demo_in_emb)
            
            demo_out_emb = self.embed_sequence(demo_outputs[:, d], grid_idx=grid_idx, segment_id=0)
            grid_idx += 1
            all_parts.append(demo_out_emb)
        
        test_in_emb = self.embed_sequence(test_input, grid_idx=grid_idx, segment_id=0)
        test_in_grid_idx = grid_idx  # Save for generation
        grid_idx += 1
        all_parts.append(test_in_emb)
        
        # Grid index for generated output - SAME as test_input since they share spatial positions!
        gen_grid_idx = test_in_grid_idx
        
        # === ADD OUTPUT_MARKER as generation trigger (same as forward) ===
        OUTPUT_MARKER = 3
        output_marker_token = torch.full((B, 1), OUTPUT_MARKER, dtype=torch.long, device=device)
        output_marker_emb = self.embed_sequence(output_marker_token, grid_idx=gen_grid_idx, segment_id=1, start_pos=0)
        all_parts.append(output_marker_emb)
        
        h = torch.cat(all_parts, dim=1)  # [B, prefix_len + 1, D]
        
        # Generate tokens autoregressively
        # For aligned sequences, we generate a full 30x30 grid (900 tokens)
        # The model learns to output PAD/EOS tokens in appropriate positions
        generated = []
        
        # EOS token for early stopping (optional, but useful for variable-size outputs)
        EOS_TOKEN = 1
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Current input embeddings (starts with prefix + output_marker)
        current_input_emb = h
        
        for step in range(max_len):
            # 1. Run forward pass on current_input_emb
            # We must re-compute from scratch because the cache is updated based on the full sequence
            # and the causal convolutions need the full context.
            # (Optimization: Implement KV-cache and causal conv cache for O(N) generation)
            
            cache = self.get_initial_cache(B, device)
            h_step = current_input_emb
            
            for layer in self.layers:
                h_step, cache, _ = layer(h_step, cache, temperature=temperature, hard=True)
            
            # 2. Predict next token from last position
            h_out = self.output_norm(h_step[:, -1:, :])
            logits = self.output_proj(h_out)  # [B, 1, vocab]
            
            # Sample or argmax
            if temperature <= 0.01:
                next_token = logits.argmax(dim=-1)  # [B, 1]
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), 1)  # [B, 1]
            
            generated.append(next_token)
            
            # 3. Embed next token and append to input
            # step=0 → position 0 (row=0, col=0)
            next_emb = self.embed_sequence(
                next_token, 
                grid_idx=gen_grid_idx, 
                segment_id=1,
                start_pos=step,  # Critical: position within output grid
            )
            
            current_input_emb = torch.cat([current_input_emb, next_emb], dim=1)
        
        return torch.cat(generated, dim=1)  # [B, max_len]


# ============================================================================
# Convenience Functions
# ============================================================================

def create_decoder_cache_model(
    preset: str = "fast",
    vocab_size: int = 14,
    max_seq_len: int = 256,
) -> DecoderCacheModel:
    """
    Create a Decoder+Cache model with preset configurations.
    
    Presets (optimized for laptop training):
        debug: Minimal for testing
        fast: Quick training iteration
        medium: Balanced performance
        full: Best performance (may be slow on laptop)
    """
    configs = {
        "debug": dict(
            d_model=32, d_cache=24, num_layers=2, num_slots=8,
            kernel_size=3, num_conv_layers_per_block=1,
        ),
        "fast": dict(
            d_model=64, d_cache=48, num_layers=3, num_slots=16,
            kernel_size=5, num_conv_layers_per_block=2,
        ),
        "medium": dict(
            d_model=96, d_cache=64, num_layers=4, num_slots=24,
            kernel_size=5, num_conv_layers_per_block=2,
        ),
        "full": dict(
            d_model=128, d_cache=64, num_layers=6, num_slots=32,
            kernel_size=5, num_conv_layers_per_block=3,
        ),
    }
    
    cfg = configs.get(preset, configs["fast"])
    
    return DecoderCacheModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **cfg,
    )
