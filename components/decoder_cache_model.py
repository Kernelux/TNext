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
from .shared_memory import (
    SharedLTMProjections,
    SharedWMProjections,
    EfficientMemoryController,
    EfficientVolatileMemory,
)
from .modules import CacheSelfAttention, UnifiedMemoryConsolidator, SinusoidalPositionalEmbedding

# Reuse CNN building blocks from cnn_cache_model
from .cnn_cache_model import CausalConv1d, ConvBlock, DilatedConvStack


# ============================================================================
# Decoder Layer (Single-Pass, No Refinement)
# ============================================================================

class DecoderCacheLayer(nn.Module):
    """
    Decoder layer with causal CNN + dual memory system.
    
    NOW USES SHARED MODULES for efficiency:
    - SharedLTMProjections: Model-level, shared across all layers (LTM read/write)
    - SharedWMProjections: Model-level, shared across all layers (WM read/write)
    - Shared CacheSelfAttention: Model-level, shared consolidation modules
    - EfficientMemoryController: Per-layer lightweight ops for LTM
    - EfficientVolatileMemory: Per-layer lightweight ops for WM

    Write behavior toggles:
    - ltm_wta_write: winner-take-all slot assignment vs blended writes
    - wm_wta_write: winner-take-all (default) vs blended writes
    
    Memory Systems:
    - LTM (Long-Term Memory): Persistent cache for rules/patterns
    - WM (Working Memory): Volatile clipboard for intermediate values
    
    Flow:
        Input x
            │
            ▼
        ┌─────────────────┐
        │ LTM READ        │ ← Query global cache for patterns/rules
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ WM READ         │ ← Query working memory (clipboard)
        └────────┬────────┘   (decays slot after read)
                 │
                 ▼
        ┌─────────────────┐
        │ CAUSAL CNN      │ ← Local patterns (position i sees ≤ i)
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ WM WRITE        │ ← Store to clipboard (hard overwrite)
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ LTM WRITE       │ ← Store important patterns (soft blend)
        └─────────────────┘
    """
    
    def __init__(
        self,
        shared_ltm: SharedLTMProjections,
        shared_wm: SharedWMProjections,
        d_model: int,
        d_cache: int,
        num_slots: int,
        num_layers: int,
        layer_idx: int,
        consolidator: Optional['UnifiedMemoryConsolidator'] = None,  # Shared consolidator
        kernel_size: int = 5,
        num_conv_layers: int = 2,
        dropout: float = 0.1,
        soft_eviction: bool = False,
        num_latent_slots: int = 32,  # For CausalLatentBank
        num_wm_slots: int = 8,  # Working memory slots (smaller than LTM)
        ltm_wta_write: bool = False,  # WTA collision resolution for LTM
        wm_wta_write: bool = True,   # WTA collision resolution for WM (default: True)
        use_ltm: bool = True,
        use_wm: bool = True,
        use_consolidation: bool = True,
        use_causal_latent: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.num_wm_slots = num_wm_slots
        self.layer_idx = layer_idx
        self.consolidator = consolidator  # Shared across layers, called after writes
        self.use_ltm = use_ltm
        self.use_wm = use_wm
        self.use_consolidation = use_consolidation
        self.use_causal_latent = use_causal_latent
        
        # Efficient Long-Term Memory Controller (uses shared projections)
        self.memory = EfficientMemoryController(
            shared=shared_ltm,
            layer_idx=layer_idx,
            num_slots=num_slots,
            num_layers=num_layers,
            d_model=d_model,
            d_cache=d_cache,
            dropout=dropout,
            soft_eviction=soft_eviction,
            wta_write=ltm_wta_write,
        )
        
        # Efficient Working Memory Controller (uses shared projections)
        self.working_memory = EfficientVolatileMemory(
            shared=shared_wm,
            layer_idx=layer_idx,
            num_slots=num_wm_slots,
            d_model=d_model,
            d_cache=d_cache,
            read_decay=0.0,  # Hard invalidation: read once, then slot is stale
            dropout=dropout,
            wta_write=wm_wta_write,
        )
        
        # Causal CNN compute with Latent Bank for global context
        self.compute = DilatedConvStack(
            d_model=d_model,
            num_layers=num_conv_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            num_latent_slots=num_latent_slots,
            use_causal_latent=use_causal_latent,
        )
        
        # Layer norm for post-compute
        self.post_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,                # [B, S, D]
        cache: torch.Tensor,            # [B, L*K, D_slot] - Long-term memory
        wm: torch.Tensor,               # [B, K_wm, D_wm_slot] - Working memory
        temperature: float = 1.0,
        hard: bool = False,
        cache_layer_embed: Optional[nn.Embedding] = None,  # Unused now (shared handles it)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass: LTM Read → WM Read → Compute → WM Write → LTM Write
        
        Returns:
            output: [B, S, D] processed features
            updated_cache: [B, L*K, D_slot] - Updated long-term memory
            updated_wm: [B, K_wm, D_wm_slot] - Updated working memory
            aux: auxiliary info (gates for monitoring)
        """
        # === LTM READ: Get global context from long-term memory ===
        if self.use_ltm:
            ltm_read_result = self.memory.read(
                x=x,
                cache=cache,
                use_global=True,
                temperature=temperature,
                hard=hard,
            )
            x_ltm = ltm_read_result['x_enhanced']
        else:
            x_ltm = x
            ltm_read_result = {'read_gate': torch.zeros(1, device=x.device), 'x_enhanced': x}
        
        # === WM READ: Get context from working memory (clipboard) ===
        if self.use_wm:
            wm_read_result = self.working_memory.read(
                x=x_ltm,
                wm=wm,
                hard=hard,
            )
            x_enhanced = wm_read_result['x_enhanced']
            wm_after_read = wm_read_result['wm_updated']  # Freshness decayed (slots invalidated)
        else:
            x_enhanced = x_ltm
            wm_after_read = wm
            wm_read_result = {'read_gate': torch.zeros(1, device=x.device), 'x_enhanced': x_ltm, 'wm_updated': wm}
        
        # === CONSOLIDATION (after WM read, before compute) ===
        # WM is read-once: slots are invalidated after read.
        # Consolidate NOW so LTM can absorb clipboard content before it's lost.
        # This transfers volatile→persistent before the content disappears.
        if self.use_consolidation and self.consolidator is not None:
            cache, wm_after_read = self.consolidator(cache, wm_after_read)
        
        # === CAUSAL CNN COMPUTE ===
        output = self.compute(x_enhanced)
        output = self.post_norm(output)

        # === WM WRITE: Store to clipboard (hard overwrite, use-once) ===
        if self.use_wm:
            wm_write_result = self.working_memory.write(
                x=output,
                wm=wm_after_read,
                hard=hard,
                mask=mask,
            )
            updated_wm = wm_write_result['wm_updated']
        else:
            updated_wm = wm_after_read
            wm_write_result = {'write_gate': torch.zeros(1, device=x.device), 'wm_updated': wm_after_read}

        # === LTM WRITE: Store important patterns to long-term memory ===
        if self.use_ltm:
            ltm_write_result = self.memory.write(
                output=output,
                cache=cache,
                temperature=temperature,
                hard=hard,
                iteration_idx=0,  # Single pass, always 0
                pass_idx=0,
                mask=mask,
            )
            updated_cache = ltm_write_result['updated_cache']
        else:
            updated_cache = cache
            ltm_write_result = {'write_gate': torch.zeros(1, device=x.device), 'updated_cache': cache}
        
        # === CONSOLIDATION (after all writes) ===
        # Full sync: next layer sees both new WM and LTM writes integrated
        if self.use_consolidation and self.consolidator is not None:
            updated_cache, updated_wm = self.consolidator(updated_cache, updated_wm)
        
        # Calculate WM fullness (validity of slots)
        # updated_wm is [B, K_wm, D_wm_slot] with structure [content | validity | temporal]
        # validity is at position d_cache (single value)
        wm_validity = updated_wm[..., self.d_cache]  # [B, K_wm]
        
        # Auxiliary info for monitoring
        aux = {
            'ltm_read_gate': ltm_read_result['read_gate'].detach().mean().item(),
            'ltm_write_gate': ltm_write_result['write_gate'].detach().mean().item(),
            'wm_read_gate': wm_read_result['read_gate'].detach().mean().item(),
            'wm_write_gate': wm_write_result['write_gate'].detach().mean().item(),
            'ltm_read_gate_tensor': ltm_read_result['read_gate'],
            'ltm_write_gate_tensor': ltm_write_result['write_gate'],
            'wm_read_gate_tensor': wm_read_result['read_gate'],
            'wm_write_gate_tensor': wm_write_result['write_gate'],
            'wm_validity': wm_validity,
        }
        
        return output, updated_cache, updated_wm, aux

    def forward_chunk(
        self,
        x: torch.Tensor,                # [B, S, D]
        cache: torch.Tensor,            # [B, L*K, D_slot]
        wm: torch.Tensor,               # [B, K_wm, D_wm_slot]
        cnn_state: Optional[Dict] = None,
        temperature: float = 1.0,
        hard: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
        """
        Forward pass for a chunk.
        
        Returns:
            output: [B, S, D]
            updated_cache: [B, L*K, D_slot]
            updated_wm: [B, K_wm, D_wm_slot]
            aux: auxiliary info
            new_cnn_state: Dict
        """
        # === LTM READ ===
        if self.use_ltm:
            ltm_read_result = self.memory.read(
                x=x,
                cache=cache,
                use_global=True,
                temperature=temperature,
                hard=hard,
            )
            x_ltm = ltm_read_result['x_enhanced']
        else:
            x_ltm = x
            ltm_read_result = {'read_gate': torch.zeros(1, device=x.device), 'x_enhanced': x}
        
        # === WM READ ===
        if self.use_wm:
            wm_read_result = self.working_memory.read(
                x=x_ltm,
                wm=wm,
                hard=hard,
            )
            x_enhanced = wm_read_result['x_enhanced']
            wm_after_read = wm_read_result['wm_updated']
        else:
            x_enhanced = x_ltm
            wm_after_read = wm
            wm_read_result = {'read_gate': torch.zeros(1, device=x.device), 'x_enhanced': x_ltm, 'wm_updated': wm}
        
        # === CONSOLIDATION (after WM read - preserve clipboard before invalidation) ===
        if self.use_consolidation and self.consolidator is not None:
            cache, wm_after_read = self.consolidator(cache, wm_after_read)
        
        # === CAUSAL CNN COMPUTE (Chunked) ===
        output, new_cnn_state = self.compute.forward_chunk(x_enhanced, cnn_state)
        output = self.post_norm(output)

        # === WM WRITE ===
        if self.use_wm:
            wm_write_result = self.working_memory.write(
                x=output,
                wm=wm_after_read,
                hard=hard,
                mask=mask,
            )
            updated_wm = wm_write_result['wm_updated']
        else:
            updated_wm = wm_after_read
            wm_write_result = {'write_gate': torch.zeros(1, device=x.device), 'wm_updated': wm_after_read}

        # === LTM WRITE ===
        if self.use_ltm:
            ltm_write_result = self.memory.write(
                output=output,
                cache=cache,
                temperature=temperature,
                hard=hard,
                iteration_idx=0,
                pass_idx=0,
                mask=mask,
            )
            updated_cache = ltm_write_result['updated_cache']
        else:
            updated_cache = cache
            ltm_write_result = {'write_gate': torch.zeros(1, device=x.device), 'updated_cache': cache}
        
        # === CONSOLIDATION (after all writes) ===
        if self.use_consolidation and self.consolidator is not None:
            updated_cache, updated_wm = self.consolidator(updated_cache, updated_wm)
        
        # Aux - extract validity (at d_cache position)
        wm_validity = updated_wm[..., self.d_cache]
        aux = {
            'ltm_read_gate': ltm_read_result['read_gate'].detach().mean().item(),
            'ltm_write_gate': ltm_write_result['write_gate'].detach().mean().item(),
            'wm_read_gate': wm_read_result['read_gate'].detach().mean().item(),
            'wm_write_gate': wm_write_result['write_gate'].detach().mean().item(),
            'ltm_read_gate_tensor': ltm_read_result['read_gate'],
            'ltm_write_gate_tensor': ltm_write_result['write_gate'],
            'wm_read_gate_tensor': wm_read_result['read_gate'],
            'wm_write_gate_tensor': wm_write_result['write_gate'],
            'wm_validity': wm_validity,
        }
        
        return output, updated_cache, updated_wm, aux, new_cnn_state

    def forward_step(
        self,
        x: torch.Tensor,                # [B, 1, D]
        cache: torch.Tensor,            # [B, L*K, D_slot]
        wm: torch.Tensor,               # [B, K_wm, D_wm_slot]
        cnn_states: Optional[List[torch.Tensor]] = None,
        global_state: Optional[Tuple] = None,
        temperature: float = 1.0,
        hard: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], Tuple, Dict]:
        """
        Incremental forward step with caching.
        """
        # === LTM READ ===
        if self.use_ltm:
            ltm_read_result = self.memory.read(
                x=x,
                cache=cache,
                use_global=True,
                temperature=temperature,
                hard=hard,
            )
            x_ltm = ltm_read_result['x_enhanced']
        else:
            x_ltm = x
            ltm_read_result = {'read_gate': torch.zeros(1, device=x.device), 'x_enhanced': x}
        
        # === WM READ ===
        if self.use_wm:
            wm_read_result = self.working_memory.read(
                x=x_ltm,
                wm=wm,
                hard=hard,
            )
            x_enhanced = wm_read_result['x_enhanced']
            wm_after_read = wm_read_result['wm_updated']
        else:
            x_enhanced = x_ltm
            wm_after_read = wm
            wm_read_result = {'read_gate': torch.zeros(1, device=x.device), 'x_enhanced': x_ltm, 'wm_updated': wm}
        
        # === CONSOLIDATION (after WM read - preserve clipboard before invalidation) ===
        if self.use_consolidation and self.consolidator is not None:
            cache, wm_after_read = self.consolidator(cache, wm_after_read)
        
        # === CAUSAL CNN COMPUTE (Incremental) ===
        output, new_cnn_states, new_global_state = self.compute.forward_step(
            x_enhanced, cnn_states, global_state
        )
        output = self.post_norm(output)

        # === WM WRITE ===
        if self.use_wm:
            wm_write_result = self.working_memory.write(
                x=output,
                wm=wm_after_read,
                hard=hard,
                mask=mask,
            )
            updated_wm = wm_write_result['wm_updated']
        else:
            updated_wm = wm_after_read
            wm_write_result = {'write_gate': torch.zeros(1, device=x.device), 'wm_updated': wm_after_read}

        # === LTM WRITE ===
        if self.use_ltm:
            ltm_write_result = self.memory.write(
                output=output,
                cache=cache,
                temperature=temperature,
                hard=hard,
                iteration_idx=0,
                pass_idx=0,
                mask=mask,
            )
            updated_cache = ltm_write_result['updated_cache']
        else:
            updated_cache = cache
            ltm_write_result = {'write_gate': torch.zeros(1, device=x.device), 'updated_cache': cache}
        
        # === CONSOLIDATION (after all writes) ===
        if self.use_consolidation and self.consolidator is not None:
            updated_cache, updated_wm = self.consolidator(updated_cache, updated_wm)
        
        # Aux info - extract validity (at d_cache position)
        wm_validity = updated_wm[..., self.d_cache]
        aux = {
            'ltm_read_gate': ltm_read_result['read_gate'].detach().mean().item(),
            'ltm_write_gate': ltm_write_result['write_gate'].detach().mean().item(),
            'wm_read_gate': wm_read_result['read_gate'].detach().mean().item(),
            'wm_write_gate': wm_write_result['write_gate'].detach().mean().item(),
            'wm_validity': wm_validity,
        }
        
        return output, updated_cache, updated_wm, new_cnn_states, new_global_state, aux


# ============================================================================
# Full Decoder-Only Model
# ============================================================================

class DecoderCacheModel(nn.Module):
    """
    Decoder-only CNN + Cache model for Mini-ARC.
    
    NOW WITH SHARED MEMORY PROJECTIONS:
    - SharedLTMProjections (LTM) + SharedWMProjections (WM) shared across layers
    - ~90% reduction in memory-related parameters
    - Factored slot initialization for additional savings
    
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
        num_latent_slots: int = 32,  # For CausalLatentBank
        num_wm_slots: int = 8,  # Working memory slots per layer
        ltm_wta_write: bool = False,  # WTA collision resolution for LTM (default: blend)
        wm_wta_write: bool = True,    # WTA collision resolution for WM (default: WTA)
        use_ltm: bool = True,
        use_wm: bool = True,
        use_consolidation: bool = True,
        use_causal_latent: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.num_wm_slots = num_wm_slots
        self.total_slots = num_layers * num_slots
        self.soft_eviction = soft_eviction
        self.num_latent_slots = num_latent_slots
        self.ltm_wta_write = ltm_wta_write
        self.wm_wta_write = wm_wta_write
        self.use_ltm = use_ltm
        self.use_wm = use_wm
        self.use_consolidation = use_consolidation
        self.use_causal_latent = use_causal_latent
        
        # Slot dimensions from config (WM now aligned with LTM)
        # WM slot: [content (d_cache) | validity (1) | temporal (16)] = d_cache + 17
        self.d_meta = SLOT_DIMS.d_meta
        self.d_slot = SLOT_DIMS.d_slot(d_cache)
        self.max_seq_len = max_seq_len
        
        # === SHARED MEMORY PROJECTIONS ===
        # Separate shared modules for LTM and WM
        self.shared_ltm = SharedLTMProjections(
            d_model=d_model,
            d_cache=d_cache,
            num_slots=num_slots,  # Per-layer slots
            num_layers=num_layers,
            dropout=dropout,
        )
        self.shared_wm = SharedWMProjections(
            d_model=d_model,
            d_cache=d_cache,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # WM slot dimension (aligned with LTM: content + validity + temporal)
        self.d_wm_slot = d_cache + self.d_meta  # Same as LTM slot
        
        # === UNIFIED MEMORY CONSOLIDATION (once per forward, after all layers) ===
        # Cross-consolidates LTM and WM in a unified workspace:
        # - LTM can attend to WM (absorb new patterns)
        # - WM can attend to LTM (retrieve global rules)
        total_ltm_slots = num_layers * num_slots
        self.memory_consolidator = UnifiedMemoryConsolidator(
            d_cache=d_cache,
            d_model=d_model,  # Use d_model for attention (common dim)
            num_ltm_slots=total_ltm_slots,
            num_wm_slots=num_wm_slots,
            num_heads=4,
            dropout=dropout,
        )
        
        # === Embeddings ===
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Factored position embedding (for grid data):
        # Using Sinusoidal Embeddings (better generalization than learned)
        # - row_embed: 30 positions
        # - col_embed: 30 positions
        # - grid_embed: 8 grids
        self.grid_size = 30  # MAX_GRID_SIZE
        self.max_grids = 8   # 3 demos × 2 (in/out) + test_in + test_out
        
        # Shared sinusoidal embedding for rows, cols, and grid index
        # Max length needs to cover max(grid_size, max_grids)
        self.pos_embed = SinusoidalPositionalEmbedding(d_model, max_len=max(self.grid_size, self.max_grids))
        
        # Segment embedding: 0=context (demos + test_in), 1=generation (test_out)
        self.segment_embed = nn.Embedding(2, d_model)
        
        # === Decoder Layers (with shared consolidator for per-layer consolidation) ===
        self.layers = nn.ModuleList([
            DecoderCacheLayer(
                shared_ltm=self.shared_ltm,
                shared_wm=self.shared_wm,
                d_model=d_model,
                d_cache=d_cache,
                num_slots=num_slots,
                num_layers=num_layers,
                layer_idx=i,
                consolidator=self.memory_consolidator,  # Shared, called after each layer's writes
                kernel_size=kernel_size,
                num_conv_layers=num_conv_layers_per_block,
                dropout=dropout,
                soft_eviction=soft_eviction,
                num_latent_slots=num_latent_slots,
                num_wm_slots=num_wm_slots,
                ltm_wta_write=ltm_wta_write,
                wm_wta_write=wm_wta_write,
                use_ltm=use_ltm,
                use_wm=use_wm,
                use_consolidation=use_consolidation,
                use_causal_latent=use_causal_latent,
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
        # pos_embed is fixed, no init needed
        nn.init.normal_(self.segment_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def get_initial_cache(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize cache using factored slot embeddings from shared memory.
        
        Uses SharedLTMProjections.get_slot_init() for efficient initialization:
        - Factored: (num_slots × d_cache/4) expanded to (num_slots × d_cache)
        - Per-layer bias added for layer-specific patterns
        """
        all_content = []
        all_temporal = []
        
        for layer_idx in range(self.num_layers):
            # Get factored slot initialization for this layer
            layer_content = self.shared_ltm.get_slot_init(layer_idx)  # [num_slots, d_cache]
            all_content.append(layer_content)
            
            # Get temporal embedding for this layer (from shared memory)
            layer_temporal = self.shared_ltm.get_temporal_embedding(
                layer_idx, iter_idx=0, pass_idx=0, device=device
            )  # [1, d_temporal]
            layer_temporal = layer_temporal.expand(self.num_slots, -1)  # [num_slots, d_temporal]
            all_temporal.append(layer_temporal)
        
        # Stack all layers: [total_slots, d_cache]
        content = torch.cat(all_content, dim=0)
        content = content.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Stack temporal: [total_slots, d_temporal]
        temporal = torch.cat(all_temporal, dim=0)
        temporal = temporal.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Confidence: 0 for empty slots
        confidence = torch.zeros(batch_size, self.total_slots, 1, device=device)
        
        # Combine: [content, confidence, temporal]
        metadata = torch.cat([confidence, temporal], dim=-1)
        cache = torch.cat([content, metadata], dim=-1)
        
        return cache
    
    def get_initial_wm(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize global working memory.
        
        WM slot structure (aligned with LTM):
            [content (d_cache) | validity (1) | temporal (d_temporal)]
        
        Returns:
            [B, num_wm_slots, d_wm_slot] tensor.
            Working memory starts empty (all zeros, validity=0).
        """
        return torch.zeros(batch_size, self.num_wm_slots, self.d_wm_slot, device=device)
    
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
        
        row_emb = self.pos_embed(rows)  # [S, D]
        col_emb = self.pos_embed(cols)  # [S, D]
        
        # Grid index embedding (which grid in sequence)
        grid_idx_clamped = min(grid_idx, self.max_grids - 1)
        grid_emb = self.pos_embed(torch.tensor(grid_idx_clamped, device=device))
        
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
        Forward pass with teacher forcing (Single Pass - Fast).
        
        Concatenates all grids into one sequence and processes through layers once.
        Much faster than chunked processing (~9x speedup).
        
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
        
        # === Build full sequence (concatenate all grids) ===
        all_embeddings = []
        all_masks = []
        grid_idx = 0
        
        # Demos
        for d in range(num_demos):
            demo_in_emb = self.embed_sequence(demo_inputs[:, d], grid_idx=grid_idx, segment_id=0)
            grid_idx += 1
            all_embeddings.append(demo_in_emb)
            all_masks.append((demo_inputs[:, d] != 0).float().unsqueeze(-1))
            
            demo_out_emb = self.embed_sequence(demo_outputs[:, d], grid_idx=grid_idx, segment_id=0)
            grid_idx += 1
            all_embeddings.append(demo_out_emb)
            all_masks.append((demo_outputs[:, d] != 0).float().unsqueeze(-1))
        
        # Test input
        test_in_emb = self.embed_sequence(test_input, grid_idx=grid_idx, segment_id=0)
        test_in_grid_idx = grid_idx
        grid_idx += 1
        all_embeddings.append(test_in_emb)
        all_masks.append((test_input != 0).float().unsqueeze(-1))
        
        # Output marker
        OUTPUT_MARKER = 3
        output_marker_token = torch.full((B, 1), OUTPUT_MARKER, dtype=torch.long, device=device)
        output_marker_emb = self.embed_sequence(output_marker_token, grid_idx=test_in_grid_idx, segment_id=1, start_pos=0)
        all_embeddings.append(output_marker_emb)
        all_masks.append(torch.ones((B, 1, 1), device=device))
        
        # Track where generation starts (for extracting logits later)
        context_len = sum(e.shape[1] for e in all_embeddings)
        
        # Test output (teacher forcing)
        if test_output is not None:
            test_out_shifted = test_output[:, :-1].clone()
            test_out_shifted[test_out_shifted == -100] = 0
            test_out_emb = self.embed_sequence(test_out_shifted, grid_idx=test_in_grid_idx, segment_id=1, start_pos=0)
            all_embeddings.append(test_out_emb)
            all_masks.append((test_out_shifted != 0).float().unsqueeze(-1))
        
        # Concatenate into single sequence
        h = torch.cat(all_embeddings, dim=1)  # [B, total_len, D]
        mask = torch.cat(all_masks, dim=1)    # [B, total_len, 1]
        
        # === Initialize memories ===
        cache = self.get_initial_cache(B, device)
        wm = self.get_initial_wm(B, device)
        
        # === Single pass through all layers ===
        # Each layer now consolidates after writes (self-contained for recursive refinement)
        aux = {
            'ltm_read_gates': [],
            'ltm_write_gates': [],
            'wm_read_gates': [],
            'wm_write_gates': [],
            'wm_validity': [],  # Now validity, not freshness
        }
        
        for layer_idx, layer in enumerate(self.layers):
            h, cache, wm, layer_aux = layer.forward(
                x=h,
                cache=cache,
                wm=wm,
                temperature=temperature,
                hard=hard,
                mask=mask,
            )
            
            # Accumulate aux info
            aux['ltm_read_gates'].append(layer_aux['ltm_read_gate_tensor'])
            aux['ltm_write_gates'].append(layer_aux['ltm_write_gate_tensor'])
            aux['wm_read_gates'].append(layer_aux['wm_read_gate_tensor'])
            aux['wm_write_gates'].append(layer_aux['wm_write_gate_tensor'])
            aux['wm_validity'].append(layer_aux['wm_validity'])
        
        # === Output projection ===
        h_out = self.output_norm(h)
        logits = self.output_proj(h_out)
        
        # Extract generation logits (from output_marker onwards)
        # context_len points to end of context (after marker)
        # We want logits from marker position to predict test_out
        gen_start = context_len - 1  # marker position
        gen_logits = logits[:, gen_start:, :]
            
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
        Autoregressive generation for inference (Optimized with Caching).
        
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
        
        # === Build prefix sequence ===
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
        
        # Full prefix embedding
        prefix_emb = torch.cat(all_parts, dim=1)  # [B, prefix_len, D]
        prefix_len = prefix_emb.shape[1]
        
        # === Initialize States ===
        cache = self.get_initial_cache(B, device)
        wm = self.get_initial_wm(B, device)
        
        # CNN states: list of lists (one list of states per layer)
        cnn_states = [None] * self.num_layers
        
        # Global states: list of tuples (one per layer)
        global_states = [None] * self.num_layers
        
        # === Process Prefix (Step-by-Step to prime states) ===
        # This is O(prefix_len) but avoids O(prefix_len^2) recomputation
        # We process the prefix token by token to build up the correct cache and CNN states
        x_step = None
        for t in range(prefix_len):
            x_step = prefix_emb[:, t:t+1, :]  # [B, 1, D]
            
            # Pass through layers
            for layer_idx, layer in enumerate(self.layers):
                x_step, cache, wm, cnn_states[layer_idx], global_states[layer_idx], _ = layer.forward_step(
                    x=x_step,
                    cache=cache,
                    wm=wm,
                    cnn_states=cnn_states[layer_idx],
                    global_state=global_states[layer_idx],
                    temperature=temperature,
                    hard=True, # Use hard decisions during generation
                )
            # Each layer consolidates after its writes, no need for model-level consolidation
        
        # x_step now contains the output of the last layer for the last prefix token (OUTPUT_MARKER)
        
        # === Generate ===
        generated = []
        
        for step in range(max_len):
            # 1. Predict next token from last position
            h_out = self.output_norm(x_step)
            logits = self.output_proj(h_out)  # [B, 1, vocab]
            
            # Sample or argmax
            if temperature <= 0.01:
                next_token = logits.argmax(dim=-1)  # [B, 1]
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), 1)  # [B, 1]
            
            generated.append(next_token)
            
            # 2. Embed next token
            # step=0 → position 0 (row=0, col=0)
            next_emb = self.embed_sequence(
                next_token, 
                grid_idx=gen_grid_idx, 
                segment_id=1,
                start_pos=step,  # Critical: position within output grid
            )
            
            # 3. Run forward step
            x_step = next_emb
            for layer_idx, layer in enumerate(self.layers):
                x_step, cache, wm, cnn_states[layer_idx], global_states[layer_idx], _ = layer.forward_step(
                    x=x_step,
                    cache=cache,
                    wm=wm,
                    cnn_states=cnn_states[layer_idx],
                    global_state=global_states[layer_idx],
                    temperature=temperature,
                    hard=True,
                )
            # Each layer consolidates after its writes
        
        return torch.cat(generated, dim=1)  # [B, max_len]


# ============================================================================
# Convenience Functions
# ============================================================================

def create_decoder_cache_model(
    preset: str = "fast",
    vocab_size: int = 14,
    max_seq_len: int = 256,
    ltm_wta_write: bool = False,  # WTA collision resolution for LTM (default: blend)
    wm_wta_write: bool = True,    # WTA collision resolution for WM (default: WTA)
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
            kernel_size=3, num_conv_layers_per_block=1, num_latent_slots=16,
        ),
        "fast": dict(
            d_model=64, d_cache=48, num_layers=3, num_slots=16,
            kernel_size=5, num_conv_layers_per_block=2, num_latent_slots=32,
        ),
        "medium": dict(
            d_model=96, d_cache=64, num_layers=4, num_slots=24,
            kernel_size=5, num_conv_layers_per_block=2, num_latent_slots=48,
        ),
        "full": dict(
            d_model=128, d_cache=64, num_layers=6, num_slots=32,
            kernel_size=5, num_conv_layers_per_block=3, num_latent_slots=64,
        ),
    }
    
    cfg = configs.get(preset, configs["fast"])
    
    return DecoderCacheModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        ltm_wta_write=ltm_wta_write,
        wm_wta_write=wm_wta_write,
        **cfg,
    )
