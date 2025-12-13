"""
CNN + Cache Model for Mini-ARC
==============================

A 1D CNN-based model that uses the DLSMN cache system to achieve
transformer-like global reasoning capabilities.

Key Insight:
- CNNs excel at local pattern recognition (spatial inductive bias)
- Transformers excel at global context (attention)
- Cache provides global context to CNNs without O(N²) attention

Architecture:
    Input → [Embedding] → [CNN + Cache Layers] → [Output Proj]

Each CNN + Cache Layer:
    1. CACHE READ: Query cache for global context
    2. CNN COMPUTE: 1D convolutions (local patterns)
    3. CACHE WRITE: Store important features selectively

Refinement Structure (simple, no adaptive halting):
    - M model-level passes (re-process entire input M times)
    - K layer-level iterations (each layer refines K times per pass)
    - Cache persists across all iterations and passes

This makes CNNs "behave" like transformers by:
- Content-based retrieval (cache read ≈ attention over curated memory)
- Long-range dependencies (any layer can read any other layer's cache)
- Selective memory (only important patterns are stored)

Usage:
    model = CNNCacheModel(vocab_size=14, d_model=64, max_passes=3, max_layer_iters=2)
    logits, cache, aux = model(demo_inputs, demo_outputs, test_input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

from .config import TrainingConfig, FeatureFlags, SLOT_DIMS
from .memory_controller import MemoryController
from .modules import CacheSelfAttention


# ============================================================================
# CNN Building Blocks
# ============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution (no future leakage).

    For autoregressive tasks, we need to ensure position i only sees positions <= i.
    This is done by left-padding the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Padding = (kernel_size - 1) * dilation for causal conv
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,  # We'll pad manually
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, S] input
        Returns:
            [B, C_out, S] output (same sequence length)
        """
        # Left-pad for causal convolution
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ConvBlock(nn.Module):
    """
    Sandglass-style convolutional block with residual connection.
    
    Based on MobileNeXt (2020): https://arxiv.org/abs/2007.02269
    
    Structure: LayerNorm → DW Conv (high-D) → GELU → Compress → Expand → Dropout + Residual
    
    Key improvements over Inverted Bottleneck:
    1. Depthwise conv on HIGH-dimensional space (richer spatial features)
    2. HIGH-dimensional shortcut (better gradient flow)
    3. Compression bottleneck in the MIDDLE (parameter efficiency)
    
    Flow: d → DW(d) → d/r → d (where r=2 is reduction ratio)
    
    This achieves +1.7% accuracy over MobileNetV2 with same params/FLOPs.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 5,
        reduction: int = 2,  # Sandglass: compress by this factor
        dropout: float = 0.1,
        dilation: int = 1,
    ):
        super().__init__()
        d_bottleneck = d_model // reduction

        self.norm = nn.LayerNorm(d_model)

        # === SANDGLASS STRUCTURE ===
        # 1. Depthwise conv on HIGH-D space (spatial features on rich representation)
        self.dw_conv = CausalConv1d(
            d_model, d_model, kernel_size, 
            dilation=dilation, groups=d_model  # Depthwise: each channel independently
        )
        
        # 2. Compress to bottleneck (1x1 pointwise)
        self.compress = nn.Conv1d(d_model, d_bottleneck, 1)
        
        # 3. Expand back to original dim (1x1 pointwise) 
        self.expand = nn.Conv1d(d_bottleneck, d_model, 1)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] input (sequence-last format)
        Returns:
            [B, S, D] output
        """
        residual = x  # HIGH-D shortcut (key for gradient flow)

        # LayerNorm expects [B, S, D]
        x = self.norm(x)

        # Conv expects [B, D, S]
        x = x.transpose(1, 2)
        
        # Sandglass: DW on high-D → compress → expand
        x = self.dw_conv(x)      # Spatial features on rich representation
        x = self.act(x)
        x = self.compress(x)     # Bottleneck (information compression)
        x = self.act(x)
        x = self.expand(x)       # Restore dimensionality
        x = self.dropout(x)

        # Back to [B, S, D]
        x = x.transpose(1, 2)

        return x + residual  # HIGH-D residual connection


class DilatedConvStack(nn.Module):
    """
    Stack of dilated convolutions with exponentially increasing dilation.

    Dilations: [1, 2, 4, 8, ...] to capture multi-scale patterns.

    This gives CNNs a larger receptive field without huge kernels.
    With dilation [1,2,4,8] and kernel=3, receptive field = 2*(1+2+4+8) + 1 = 31
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            ConvBlock(
                d_model=d_model,
                kernel_size=kernel_size,
                dilation=2**i,  # Exponential dilation
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] input
        Returns:
            [B, S, D] output with large receptive field
        """
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================================
# CNN + Cache Layer
# ============================================================================

class CNNCacheLayer(nn.Module):
    """
    CNN layer augmented with selective cache memory.

    This is the core innovation: combining CNN's local processing
    with transformer-like global context via cache.

    Flow:
        Input x
            │
            ▼
        ┌─────────────────┐
        │ CACHE READ      │ ← Query: "What global context do I need?"
        │ (attention over │   Returns: relevant cached features
        │  K slots)       │
        └────────┬────────┘
                 │
                 ▼ x_enhanced (local + global)
        ┌─────────────────┐
        │ CNN COMPUTE     │ ← Dilated convolutions for local patterns
        │ (local patterns)│   Large receptive field via dilation
        └────────┬────────┘
                 │
                 ▼ output
        ┌─────────────────┐
        │ CACHE WRITE     │ ← "Is this pattern important?"
        │ (selective)     │   If yes: store in appropriate slot
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
        )

        # CNN compute (dilated stack for large receptive field)
        self.compute = DilatedConvStack(
            d_model=d_model,
            num_layers=num_conv_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Layer norm for post-compute
        self.post_norm = nn.LayerNorm(d_model)

        # === Cache Self-Attention (Memory Consolidation) ===
        # After each write, allow cache slots to attend to each other.
        # This enables:
        # 1. Integration of new info with existing knowledge
        # 2. Transitive reasoning (A→B, B→C ⟹ A→C)
        # 3. Deduplication and consolidation
        d_slot = SLOT_DIMS.d_slot(d_cache)

        # Find valid num_heads that divides d_slot evenly
        cache_attn_heads = 1
        for h in [5, 13]:  # Common divisors for d_slot = 65 (48 + 17)
            if d_slot % h == 0:
                cache_attn_heads = h
                break

        self.cache_self_attn = CacheSelfAttention(
            d_cache=d_slot,  # Full slot dimension (content + metadata)
            num_heads=cache_attn_heads,
            dropout=dropout,
            use_linear=False,
        )

        # Iteration embedding (for recursive refinement)
        self.iter_embed = nn.Embedding(8, d_model)
        nn.init.normal_(self.iter_embed.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                # [B, S, D]
        cache: torch.Tensor,            # [B, L*K, D_slot]
        iteration: int = 0,
        pass_idx: int = 0,
        temperature: float = 1.0,
        hard: bool = False,
        # Temporal embeddings from model
        cache_layer_embed: Optional[nn.Embedding] = None,
        cache_iter_embed: Optional[nn.Embedding] = None,
        cache_pass_embed: Optional[nn.Embedding] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass: Read → Compute → Write

        Returns:
            output: [B, S, D] processed features
            updated_cache: [B, L*K, D_slot]
            aux: auxiliary info
        """
        B, S, D = x.shape
        device = x.device

        # Add iteration embedding
        iter_emb = self.iter_embed(torch.tensor(iteration, device=device))
        x = x + iter_emb.unsqueeze(0).unsqueeze(0)

        # === CACHE READ: Get global context ===
        read_result = self.memory.read(
            x=x,
            cache=cache,
            use_global=True,  # Read from all layers (cross-layer communication)
            temperature=temperature,
            hard=hard,
        )
        x_enhanced = read_result['x_enhanced']  # [B, S, D]

        # === CNN COMPUTE: Local pattern processing ===
        output = self.compute(x_enhanced)
        output = self.post_norm(output)

        # === CACHE WRITE: Selectively store important features ===
        write_result = self.memory.write(
            output=output,
            cache=cache,
            temperature=temperature,
            hard=hard,
            iteration_idx=iteration,
            pass_idx=pass_idx,
            cache_layer_embed=cache_layer_embed,
            cache_iter_embed=cache_iter_embed,
            cache_pass_embed=cache_pass_embed,
        )
        updated_cache = write_result['updated_cache']

        # === CACHE SELF-ATTENTION: Memory consolidation ===
        # Allow cache slots to attend to each other for:
        # - Integration of new info with existing knowledge
        # - Transitive reasoning across slots
        # - Deduplication and consolidation
        updated_cache = self.cache_self_attn(updated_cache)

        # Auxiliary info
        aux = {
            'read_gate': read_result['read_gate'].detach().mean().item(),  # For display during training
            'write_gate': write_result['write_gate'].detach().mean().item(),  # For display during training
            'read_gate_tensor': read_result['read_gate'],  # For loss computation
            'write_gate_tensor': write_result['write_gate'],  # For loss computation
            'soft_read_gate': read_result.get('soft_read_gate'),
            'soft_write_gate': write_result.get('soft_write_gate'),
        }

        return output, updated_cache, aux


# ============================================================================
# Full CNN + Cache Model
# ============================================================================

class CNNCacheModel(nn.Module):
    """
    Full CNN + Cache model for Mini-ARC.

    Architecture:
        Embedding → [CNN+Cache Layer 0] → [CNN+Cache Layer 1] → ... → Output

    Simple refinement (no adaptive halting):
        - M model-level passes: re-process input M times, cache persists
        - K layer-level iterations: each layer refines K times per pass
        
    The cache enables:
    1. Cross-layer communication (lower layers inform higher)
    2. Multi-pass refinement (cache persists across passes)
    3. Selective memory (only important patterns stored)
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
        max_seq_len: int = 25,
        max_passes: int = 3,        # M: model-level passes
        max_layer_iters: int = 1,   # K: layer-level iterations
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.total_slots = num_layers * num_slots
        self.max_passes = max_passes
        self.max_layer_iters = max_layer_iters

        # Slot dimensions from config
        self.d_meta = SLOT_DIMS.d_meta
        self.d_slot = SLOT_DIMS.d_slot(d_cache)

        # === Embeddings ===
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len * 10, d_model)  # Support long sequences
        self.segment_embed = nn.Embedding(2, d_model)  # context=0, answer=1

        # Learned slot embeddings (initialize cache)
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_layers, num_slots, d_cache) * 0.02
        )

        # Layer-ID embeddings
        self.layer_id_embeddings = nn.Parameter(
            torch.randn(num_layers, SLOT_DIMS.d_layer_embed) * 0.02
        )

        # Pass embeddings (model knows which pass it's on)
        self.pass_embed = nn.Embedding(max_passes, d_model)

        # Temporal embeddings for cache metadata (shared across layers)
        self.cache_layer_embed = nn.Embedding(num_layers, SLOT_DIMS.d_layer_embed)
        self.cache_iter_embed = nn.Embedding(8, SLOT_DIMS.d_iter_embed)
        self.cache_pass_embed = nn.Embedding(max_passes, SLOT_DIMS.d_pass_embed)

        # === CNN + Cache Layers ===
        self.layers = nn.ModuleList([
            CNNCacheLayer(
                d_model=d_model,
                d_cache=d_cache,
                num_slots=num_slots,
                num_layers=num_layers,
                layer_idx=i,
                kernel_size=kernel_size,
                num_conv_layers=num_conv_layers_per_block,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # === Output ===
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Answer feedback (simple additive, no gate needed)
        self.prev_answer_embed = nn.Embedding(vocab_size, d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.segment_embed.weight, std=0.02)
        nn.init.normal_(self.pass_embed.weight, std=0.02)
        nn.init.normal_(self.cache_layer_embed.weight, std=0.02)
        nn.init.normal_(self.cache_iter_embed.weight, std=0.02)
        nn.init.normal_(self.cache_pass_embed.weight, std=0.02)
        nn.init.normal_(self.prev_answer_embed.weight, std=0.02)
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

        # Iteration and pass: zeros
        iter_pass_dim = SLOT_DIMS.d_iter_embed + SLOT_DIMS.d_pass_embed
        iter_pass = torch.zeros(batch_size, self.total_slots, iter_pass_dim, device=device)

        # Combine
        temporal = torch.cat([layer_ids, iter_pass], dim=-1)
        metadata = torch.cat([confidence, temporal], dim=-1)
        cache = torch.cat([content, metadata], dim=-1)

        return cache

    def embed_sequence(self, seq: torch.Tensor, start_pos: int = 0, is_answer: bool = False) -> torch.Tensor:
        """Embed a sequence with position and segment information."""
        B, S = seq.shape
        device = seq.device

        # Token embedding
        tok_emb = self.token_embed(seq)

        # Position embedding
        positions = torch.arange(start_pos, start_pos + S, device=device)
        pos_emb = self.pos_embed(positions)

        # Segment embedding
        seg_id = 1 if is_answer else 0
        seg_emb = self.segment_embed(torch.tensor(seg_id, device=device))

        return tok_emb + pos_emb + seg_emb

    def forward(
        self,
        demo_inputs: torch.Tensor,    # [B, num_demos, S]
        demo_outputs: torch.Tensor,   # [B, num_demos, S]
        test_input: torch.Tensor,     # [B, S]
        config: Optional[TrainingConfig] = None,
        step: int = 0,
        return_aux: bool = True,
        max_passes: Optional[int] = None,      # Override M
        max_layer_iters: Optional[int] = None, # Override K
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Simple multi-pass forward with CNN + Cache.
        
        Refinement structure (fixed, no adaptive halting):
            - M model-level passes: re-process entire sequence M times
            - K layer-level iterations: each layer refines K times per pass
            - Cache persists across all iterations and passes

        Process:
        1. Embed all inputs (demos + test)
        2. For each pass m in [0, M):
           a. Add pass embedding
           b. For each layer:
              - Run K iterations of read → compute → write
           c. Generate predictions
           d. Feed predictions back for next pass
        3. Return final predictions
        """
        if config is None:
            config = TrainingConfig()

        B = test_input.shape[0]
        device = test_input.device
        num_demos = demo_inputs.shape[1]
        S = test_input.shape[1]

        # Use provided values or defaults
        M = max_passes if max_passes is not None else self.max_passes
        K = max_layer_iters if max_layer_iters is not None else self.max_layer_iters

        temperature = config.get_temperature(step)
        hard = (temperature < 0.2)

        # === Build combined sequence ===
        # [demo_in_0, demo_out_0, ..., demo_in_N, demo_out_N, test_in]
        all_parts = []
        pos = 0

        for d in range(num_demos):
            demo_in_emb = self.embed_sequence(demo_inputs[:, d], start_pos=pos, is_answer=False)
            pos += demo_in_emb.shape[1]
            all_parts.append(demo_in_emb)

            demo_out_emb = self.embed_sequence(demo_outputs[:, d], start_pos=pos, is_answer=True)
            pos += demo_out_emb.shape[1]
            all_parts.append(demo_out_emb)

        test_in_emb = self.embed_sequence(test_input, start_pos=pos, is_answer=False)
        all_parts.append(test_in_emb)

        context_seq = torch.cat(all_parts, dim=1)  # [B, total_ctx_len, D]

        # Initialize cache
        cache = self.get_initial_cache(B, device)

        # Auxiliary tracking
        aux = {
            'passes_run': M,
            'layer_iters': K,
            'pass_logits': [],      # For deep supervision
            'read_gates': [],       # For loss computation
            'write_gates': [],      # For loss computation
        }

        # Previous answer (for feedback)
        prev_answer = None
        test_logits = None

        # === MODEL-LEVEL PASSES (M iterations) ===
        for pass_idx in range(M):
            # Start with original context sequence
            h = context_seq.clone()

            # Add pass embedding (model knows which pass it's on)
            pass_emb = self.pass_embed(torch.tensor(pass_idx, device=device))
            h = h + pass_emb.unsqueeze(0).unsqueeze(0)

            # Answer feedback (from previous pass) - simple additive
            if prev_answer is not None and config.features.use_answer_feedback:
                ans_emb = self.prev_answer_embed(prev_answer)  # [B, S, D]
                # Add to test region only (last S positions)
                h_test = h[:, -S:, :] + ans_emb
                h = torch.cat([h[:, :-S, :], h_test], dim=1)

            # Track gates for this pass
            pass_read_gates = []
            pass_write_gates = []

            # === LAYER PROCESSING ===
            for layer_idx, layer in enumerate(self.layers):
                # === LAYER-LEVEL ITERATIONS (K iterations per layer) ===
                for iter_idx in range(K):
                    h, cache, layer_aux = layer(
                        x=h,
                        cache=cache,
                        iteration=iter_idx,
                        pass_idx=pass_idx,
                        temperature=temperature,
                        hard=hard,
                        cache_layer_embed=self.cache_layer_embed,
                        cache_iter_embed=self.cache_iter_embed,
                        cache_pass_embed=self.cache_pass_embed,
                    )
                    # Collect gates from last iteration only
                    if iter_idx == K - 1:
                        pass_read_gates.append(layer_aux['read_gate_tensor'])
                        pass_write_gates.append(layer_aux['write_gate_tensor'])

            # Aggregate gates for this pass
            aux['read_gates'].append(torch.stack(pass_read_gates).mean(dim=0))
            aux['write_gates'].append(torch.stack(pass_write_gates).mean(dim=0))

            # === Output projection ===
            h_out = self.output_norm(h)
            logits = self.output_proj(h_out)  # [B, ctx_len, vocab]

            # Extract test output logits (last S positions)
            test_logits = logits[:, -S:, :]  # [B, S, vocab]
            aux['pass_logits'].append(test_logits.detach())

            # Store prediction for next pass feedback
            prev_answer = test_logits.argmax(dim=-1)  # [B, S]

        # Final output
        assert test_logits is not None, "No passes were run"
        return test_logits, cache, aux


# ============================================================================
# Convenience Functions
# ============================================================================

def create_cnn_cache_model(
    preset: str = "fast",
    vocab_size: int = 14,
    max_seq_len: int = 25,
) -> CNNCacheModel:
    """
    Create a CNN+Cache model with preset configurations.

    Presets:
        debug: Minimal for testing (M=2, K=1)
        fast: Quick training (M=3, K=1)
        full: Best performance (M=4, K=2)
    """
    configs = {
        "debug": dict(
            d_model=32, d_cache=24, num_layers=2, num_slots=8,
            kernel_size=3, num_conv_layers_per_block=1, 
            max_passes=2, max_layer_iters=1,
        ),
        "fast": dict(
            d_model=64, d_cache=48, num_layers=3, num_slots=16,
            kernel_size=5, num_conv_layers_per_block=2, 
            max_passes=3, max_layer_iters=1,
        ),
        "full": dict(
            d_model=128, d_cache=64, num_layers=4, num_slots=32,
            kernel_size=5, num_conv_layers_per_block=3, 
            max_passes=4, max_layer_iters=2,
        ),
    }

    cfg = configs.get(preset, configs["fast"])

    return CNNCacheModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **cfg,
    )
