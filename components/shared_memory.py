"""
Shared Memory Projections
=========================

Consolidates memory-related projections into two shared modules (LTM + WM).
This eliminates redundant parameters across layers while keeping LTM/WM distinct.

Parameter Savings:
-----------------
Before: Each layer had its own projections (~80K × 6 layers × 2 memory types = ~960K)
After:  Two shared projection sets (~2×80K) + lightweight per-layer ops (~2K × 6 = ~12K)
Total savings: ~790K parameters (~82% reduction in memory-related params)

Design:
-------
    SharedLTMProjections (MODEL-LEVEL)
    ├── to_cache/from_cache/query_proj/fusion
    ├── importance + write decision heads
    └── slot_init + temporal embeddings

    SharedWMProjections (MODEL-LEVEL)
    ├── to_wm/from_wm/query_wm/fusion
    ├── importance + write decision heads (WM-specific)
    └── read gate
    
    LightweightMemoryOps (PER-LAYER, minimal params)
    ├── read_gate_bias:  Learnable scalar bias for read gating
    ├── write_gate_bias: Learnable scalar bias for write gating
    └── layer_scale:     Per-layer output scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List

from .config import SLOT_DIMS
from .utils import gumbel_softmax


class SharedLTMProjections(nn.Module):
    """
    Shared projection layers for Long-Term Memory (LTM) only.
    Keeps LTM ops consolidated across layers while remaining separate from WM.
    """
    
    def __init__(
        self,
        d_model: int,
        d_cache: int,
        num_slots: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.num_layers = num_layers
        
        # Slot dimensions from config
        self.d_meta = SLOT_DIMS.d_meta
        self.d_slot = SLOT_DIMS.d_slot(d_cache)
        
        # === Core Projections (shared across all layers) ===

        # LTM projections (content + metadata space)
        self.to_cache = nn.Linear(d_model, d_cache)
        self.from_cache = nn.Linear(self.d_slot, d_model)
        self.query_proj = nn.Linear(d_model, d_cache)

        # Fusion layer (used for LTM reads)
        self.fusion = nn.Linear(d_model * 2, d_model)

        # === Gating Networks (shared, but per-layer biases added externally) ===

        # Read gate: "Do I need memory context here?"
        # Outputs logit, layer-specific bias added by LightweightMemoryOps
        self.read_gate_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

        # LTM importance scorer (xLSTM-style: raw logit, exp() applied externally)
        self.importance_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

        # Write decision: "Should this token write to LTM?"
        # Input: [query, context_from_cache]
        self.write_decision_net = nn.Sequential(
            nn.LayerNorm(d_cache * 2),
            nn.Linear(d_cache * 2, d_cache // 2),
            nn.SiLU(),
            nn.Linear(d_cache // 2, 1),
        )
        
        # === Factored Slot Initialization ===
        # Instead of num_layers × num_slots × d_cache params,
        # use factored: (num_slots × d_cache/4) + (d_cache/4 × d_cache)
        # For 6 layers, 192 slots, d_cache=64:
        #   Before: 6 × 192 × 64 = 73,728 params
        #   After:  192 × 16 + 16 × 64 = 3,072 + 1,024 = 4,096 params (18x reduction)
        d_slot_init = max(d_cache // 4, 8)
        self.slot_init_base = nn.Parameter(torch.randn(num_slots, d_slot_init) * 0.02)
        self.slot_init_expand = nn.Linear(d_slot_init, d_cache, bias=False)
        
        # Layer-specific slot modulation (small addition per layer)
        self.layer_slot_bias = nn.Parameter(torch.zeros(num_layers, 1, d_cache) * 0.01)
        
        # === Layer Embeddings ===
        self.layer_embed = nn.Embedding(num_layers, SLOT_DIMS.d_layer_embed)
        
        # === Temporal Embeddings (for iterative models) ===
        self.iter_embed = nn.Embedding(8, SLOT_DIMS.d_iter_embed)  # Max 8 iterations
        self.pass_embed = nn.Embedding(8, SLOT_DIMS.d_pass_embed)  # Max 8 passes
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with appropriate strategies."""
        # Xavier for projection layers
        for module in [self.to_cache, self.from_cache, self.query_proj, self.fusion]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.slot_init_expand.weight)
        
        # Sequential modules
        for seq in [self.read_gate_net, self.importance_net, self.write_decision_net]:
            for m in seq:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        # Embeddings
        nn.init.normal_(self.layer_embed.weight, std=0.02)
        nn.init.normal_(self.iter_embed.weight, std=0.02)
        nn.init.normal_(self.pass_embed.weight, std=0.02)
    
    def get_slot_init(self, layer_idx: int) -> torch.Tensor:
        """
        Get initialized slot content for a specific layer.
        
        Args:
            layer_idx: Which layer's slots to initialize
            
        Returns:
            slot_content: [num_slots, d_cache] initialized slot content
        """
        # Expand factored representation
        base = self.slot_init_expand(self.slot_init_base)  # [num_slots, d_cache]
        
        # Add layer-specific bias
        layer_bias = self.layer_slot_bias[layer_idx]  # [1, d_cache]
        
        return base + layer_bias
    
    def project_to_cache(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden state to cache space."""
        return self.to_cache(x)
    
    def project_from_cache(self, cache: torch.Tensor) -> torch.Tensor:
        """Project cache content (with metadata) to hidden space."""
        return self.from_cache(cache)

    def generate_query(self, x: torch.Tensor) -> torch.Tensor:
        """Generate attention query from hidden state."""
        return self.query_proj(x)
    
    def fuse_context(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Fuse hidden state with retrieved context."""
        combined = torch.cat([x, context], dim=-1)
        return self.fusion(combined)
    
    def compute_read_gate_logit(self, x: torch.Tensor) -> torch.Tensor:
        """Compute read gate logit (add per-layer bias externally)."""
        return self.read_gate_net(x)
    
    def compute_importance_logit(self, x: torch.Tensor) -> torch.Tensor:
        """Compute importance logit (apply exp() externally)."""
        return self.importance_net(x)

    
    def compute_write_decision_logit(
        self, 
        query: torch.Tensor, 
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute write decision logit."""
        combined = torch.cat([query, context], dim=-1)
        return self.write_decision_net(combined)

    
    def get_temporal_embedding(
        self,
        layer_idx: int,
        iter_idx: int = 0,
        pass_idx: int = 0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Get temporal embedding for cache slot metadata."""
        if device is None:
            device = self.layer_embed.weight.device
            
        layer_emb = self.layer_embed(torch.tensor([layer_idx], device=device))
        iter_emb = self.iter_embed(torch.tensor([iter_idx], device=device))
        pass_emb = self.pass_embed(torch.tensor([pass_idx], device=device))
        
        return torch.cat([layer_emb, iter_emb, pass_emb], dim=-1)  # [1, d_temporal]


class SharedWMProjections(nn.Module):
    """
    Shared projection layers for Working Memory (WM) only.
    Keeps WM ops consolidated across layers while remaining separate from LTM.
    
    WM slot structure (aligned with LTM):
        [content (d_cache) | validity (1) | temporal (d_temporal)]
        where temporal = [layer_embed (8) | iter_embed (4) | pass_embed (4)]
    
    WM semantics (different from LTM):
        - Read clears slot (validity → 0) - one-time use buffer
        - Write overwrites slot (WTA) - not blending
        - Temporal encodes WHO wrote (layer_idx) and WHEN (iter, pass)
    """

    def __init__(
        self,
        d_model: int,
        d_cache: int,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        
        # Import slot dimensions
        from .config import SLOT_DIMS
        self.d_meta = SLOT_DIMS.d_meta  # 17: validity(1) + temporal(16)
        self.d_temporal = SLOT_DIMS.d_temporal  # 16
        self.d_layer_embed = SLOT_DIMS.d_layer_embed  # 8
        self.d_iter_embed = SLOT_DIMS.d_iter_embed  # 4
        self.d_pass_embed = SLOT_DIMS.d_pass_embed  # 4
        
        # Full WM slot dimension (aligned with LTM)
        self.d_wm_slot = d_cache + self.d_meta

        # Core projections for WM space
        self.to_wm = nn.Linear(d_model, d_cache)
        self.from_wm = nn.Linear(d_cache, d_model)
        self.query_wm = nn.Linear(d_model, d_cache)

        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)

        # Gating and importance heads
        self.read_gate_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

        self.importance_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

        self.write_decision_net = nn.Sequential(
            nn.LayerNorm(d_cache * 2),
            nn.Linear(d_cache * 2, d_cache // 2),
            nn.SiLU(),
            nn.Linear(d_cache // 2, 1),
        )
        
        # Temporal embeddings (same structure as LTM)
        self.layer_embed = nn.Embedding(num_layers, self.d_layer_embed)
        self.iter_embed = nn.Embedding(16, self.d_iter_embed)  # Up to 16 iterations
        self.pass_embed = nn.Embedding(8, self.d_pass_embed)   # Up to 8 passes

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in [self.to_wm, self.from_wm, self.query_wm, self.fusion]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        for seq in [self.read_gate_net, self.importance_net, self.write_decision_net]:
            for m in seq:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        # Temporal embeddings
        nn.init.normal_(self.layer_embed.weight, std=0.02)
        nn.init.normal_(self.iter_embed.weight, std=0.02)
        nn.init.normal_(self.pass_embed.weight, std=0.02)
    
    def get_temporal_embedding(
        self, 
        layer_idx: int, 
        iter_idx: int = 0, 
        pass_idx: int = 0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Get temporal embedding for a write operation."""
        layer_idx = min(layer_idx, self.num_layers - 1)
        iter_idx = min(iter_idx, 15)
        pass_idx = min(pass_idx, 7)
        
        layer_emb = self.layer_embed.weight[layer_idx]  # [d_layer]
        iter_emb = self.iter_embed.weight[iter_idx]     # [d_iter]
        pass_emb = self.pass_embed.weight[pass_idx]     # [d_pass]
        
        temporal = torch.cat([layer_emb, iter_emb, pass_emb], dim=-1)  # [d_temporal]
        return temporal.unsqueeze(0)  # [1, d_temporal]

    # Projections
    def project_to_wm(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_wm(x)

    def project_from_wm(self, slot: torch.Tensor) -> torch.Tensor:
        return self.from_wm(slot)

    def generate_wm_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_wm(x)

    def fuse_context(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, context], dim=-1)
        return self.fusion(combined)

    # Heads
    def compute_read_gate_logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.read_gate_net(x)

    def compute_importance_logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.importance_net(x)

    def compute_write_decision_logit(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([query, context], dim=-1)
        return self.write_decision_net(combined)


class LightweightMemoryOps(nn.Module):
    """
    Per-layer lightweight memory operations.
    
    Contains only layer-specific learnable parameters (biases, scales).
    All heavy projections come from SharedLTMProjections/SharedWMProjections.
    
    This allows each layer to have slightly different read/write behavior
    while sharing 95%+ of the parameters.
    
    Args:
        layer_idx: This layer's index
        d_model: Model hidden dimension (for scale initialization)
    """
    
    def __init__(self, layer_idx: int, d_model: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Per-layer gate biases (learned offsets to shared gate networks)
        self.read_gate_bias = nn.Parameter(torch.zeros(1))
        self.write_gate_bias = nn.Parameter(torch.tensor(0.5))  # Slightly encourage writes
        
        # Per-layer output scale (like LayerScale in CaiT)
        self.read_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.write_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Temperature for this layer's write decisions
        self.write_temperature = nn.Parameter(torch.tensor(1.0))
        
        # Base retention for read gate (Griffin-style)
        self.read_log_decay = nn.Parameter(torch.tensor(2.0))  # σ(2) ≈ 0.88
    
    def apply_read_gate(
        self, 
        gate_logit: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Apply layer-specific read gating with Griffin-style exponential decay.
        
        Args:
            gate_logit: Raw logit from shared network [B, S, 1]
            hard: Use hard gating
            
        Returns:
            read_gate: [B, S, 1] in (0, 1)
        """
        # Add layer-specific bias
        r_t = torch.sigmoid(gate_logit + self.read_gate_bias)
        
        # Griffin-style exponential decay
        # Clamp sigmoid output to ensure stable log computation
        base_retention = torch.sigmoid(self.read_log_decay).clamp(min=0.1, max=0.99)
        log_base = torch.log(base_retention)  # Safe since base_retention >= 0.1
        
        # Clamp the exponent to prevent extreme values
        exponent = (8.0 * r_t * log_base).clamp(min=-20.0, max=0.0)
        read_gate = torch.exp(exponent)
        
        # Flip: gate=1 means use context, gate=0 means keep input
        read_gate = 1.0 - read_gate
        
        if hard:
            read_gate_hard = (read_gate > 0.5).float()
            read_gate = read_gate_hard - read_gate.detach() + read_gate
            
        return read_gate
    
    def apply_write_gate(
        self,
        decision_logit: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Apply layer-specific write gating with xLSTM-style exp.
        
        Args:
            decision_logit: Raw logit from shared network [B, S, 1]
            hard: Use hard gating
            
        Returns:
            write_gate: [B, S, 1] in (0, 1)
        """
        # Add layer-specific bias and apply temperature
        temp = self.write_temperature.clamp(min=0.1)
        adjusted_logit = (decision_logit + self.write_gate_bias) / temp
        
        # xLSTM-style exponential gating
        decision_exp = torch.exp(adjusted_logit.clamp(max=10.0))
        write_gate = decision_exp / (1.0 + decision_exp)
        
        if hard:
            write_gate_hard = (write_gate > 0.5).float()
            write_gate = write_gate_hard - write_gate.detach() + write_gate
            
        return write_gate
    
    def scale_read_output(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-layer scaling to read output."""
        return x * self.read_scale
    
    def scale_write_importance(self, importance: torch.Tensor) -> torch.Tensor:
        """Apply per-layer scaling to write importance."""
        return importance * self.write_scale.abs()


class EfficientMemoryController(nn.Module):
    """
    Efficient Memory Controller using shared projections (DLSM v0.1.1 style).
    
    READ (input-driven):
        1. g_read: "Should I read?" (per-token gate)
        2. Attend to cache slots via softmax attention
        3. Blend relevant slots (standard attention - no WTA needed)
        4. Gated fusion with input
    
    WRITE (output-driven):
        1. g_write: "Should I write?" (per-token gate)
        2. s_imp: "How important?" (importance score)
        3. p_write: "Which slot?" (soft routing via attention to cache)
        4. Collision resolution: multiple tokens → same slot
           - wta_write=True: winner-take-all (highest importance wins)
           - wta_write=False: importance-weighted blend (DLSM default)
    
    Args:
        shared: SharedLTMProjections instance (from model level)
        layer_idx: This layer's index
        num_slots: Number of slots for this layer
        num_layers: Total number of layers
        d_model: Model hidden dimension
        d_cache: Cache content dimension
        dropout: Dropout probability
        soft_eviction: Use soft blending for slot update vs hard replacement
        wta_write: Winner-take-all for write collisions (default: False = blend)
    """
    
    def __init__(
        self,
        shared: SharedLTMProjections,
        layer_idx: int,
        num_slots: int,
        num_layers: int,
        d_model: int,
        d_cache: int,
        dropout: float = 0.1,
        soft_eviction: bool = False,
        wta_write: bool = False,
    ):
        super().__init__()
        self.shared = shared
        self.layer_idx = layer_idx
        self.num_slots = num_slots
        self.num_layers = num_layers
        self.total_slots = num_layers * num_slots
        self.d_model = d_model
        self.d_cache = d_cache
        self.soft_eviction = soft_eviction
        self.wta_write = wta_write
        
        # Slot dimensions
        self.d_meta = SLOT_DIMS.d_meta
        self.d_slot = SLOT_DIMS.d_slot(d_cache)
        
        # Lightweight per-layer operations
        self.ops = LightweightMemoryOps(layer_idx, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _get_local_cache(self, cache: torch.Tensor) -> torch.Tensor:
        """Extract this layer's slots from global cache."""
        start = self.layer_idx * self.num_slots
        end = start + self.num_slots
        return cache[:, start:end, :]
    
    def _split_cache(self, cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split cache into content, confidence, temporal."""
        content = cache[..., :self.d_cache]
        confidence = cache[..., self.d_cache:self.d_cache+1]
        temporal = cache[..., self.d_cache+1:]
        return content, confidence, temporal
    
    def _merge_cache(
        self, 
        content: torch.Tensor, 
        confidence: torch.Tensor, 
        temporal: torch.Tensor,
    ) -> torch.Tensor:
        """Merge content and metadata back into cache tensor."""
        return torch.cat([content, confidence, temporal], dim=-1)
    
    def _update_local_cache(self, cache: torch.Tensor, new_local: torch.Tensor) -> torch.Tensor:
        """Update this layer's slots in global cache."""
        start = self.layer_idx * self.num_slots
        end = start + self.num_slots
        
        if start == 0:
            if end == cache.shape[1]:
                return new_local
            return torch.cat([new_local, cache[:, end:, :]], dim=1)
        elif end == cache.shape[1]:
            return torch.cat([cache[:, :start, :], new_local], dim=1)
        else:
            return torch.cat([cache[:, :start, :], new_local, cache[:, end:, :]], dim=1)
    
    def _attend_to_cache(
        self,
        query: torch.Tensor,
        cache_content: torch.Tensor,
        full_cache: Optional[torch.Tensor] = None,
        wta: bool = False,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention to cache slots.
        
        Args:
            wta: If True, each query attends to single best slot (WTA)
                 If False, blend across all slots weighted by attention
        """
        scores = torch.matmul(query, cache_content.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_cache)
        scores = scores.clamp(min=-20.0, max=20.0)
        
        if wta:
            # WTA: pick single best slot per query position
            if hard or not self.training:
                # Hard WTA: argmax
                best_idx = scores.argmax(dim=-1)  # [B, S]
                attn_weights = F.one_hot(best_idx, num_classes=scores.shape[-1]).float()
            else:
                # Soft WTA: Gumbel-softmax for gradients
                attn_weights = F.gumbel_softmax(scores, tau=0.5, hard=True, dim=-1)
        else:
            # Blend: standard softmax attention
            attn_weights = F.softmax(scores, dim=-1)
        
        attn_weights = self.dropout(attn_weights)
        
        retrieve_from = full_cache if full_cache is not None else cache_content
        context = torch.matmul(attn_weights, retrieve_from)
        
        return context, attn_weights
    
    def read(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        use_global: bool = False,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        READ Phase: Query memory and fuse with input (DLSM v0.1.1 style).
        
        Flow:
            1. g_read: "Should I read?" (per-token gate)
            2. Generate query, attend to cache slots via softmax attention
            3. Blend all relevant slots (standard weighted attention)
            4. Gated fusion: (1 - g_read) * x + g_read * fused
        
        Note: No WTA on reads - the model learns to attend to what's relevant,
        and multiple relevant slots naturally blend together.
        """
        B, S, _ = x.shape
        
        # Compute read gate using shared network + layer-specific bias
        gate_logit = self.shared.compute_read_gate_logit(x)
        read_gate = self.ops.apply_read_gate(gate_logit, hard=hard or not self.training)
        
        # Generate query using shared projection
        query = self.shared.generate_query(x)
        
        # Select cache to read from
        read_cache = cache if use_global else self._get_local_cache(cache)
        cache_content, cache_confidence, cache_temporal = self._split_cache(read_cache)
        
        # Attend to cache (always blend - standard softmax attention)
        context_full, attn_weights = self._attend_to_cache(
            query, cache_content, full_cache=read_cache,
            wta=False, hard=hard  # Always blend for reads
        )
        
        # Project context using shared projection
        context = self.shared.project_from_cache(context_full)
        
        # Apply layer-specific scaling
        context = self.ops.scale_read_output(context)
        
        # Gated fusion using shared fusion layer
        fused = self.shared.fuse_context(x, context)
        x_enhanced = (1 - read_gate) * x + read_gate * fused
        
        return {
            'x_enhanced': x_enhanced,
            'context': context,
            'read_gate': read_gate,
            'attn_weights': attn_weights,
        }
    
    def write(
        self,
        output: torch.Tensor,
        cache: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
        iteration_idx: int = 0,
        pass_idx: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        WRITE Phase: Selectively update cache with output content (DLSM v0.1.1 style).
        
        Flow:
            1. g_write: "Should I write?" (per-token write decision)
            2. s_imp: "How important?" (importance score for collision resolution)
            3. p_write: "Which slot?" (soft routing via attention to cache slots)
            4. Collision resolution: when multiple tokens target same slot
               - wta_write=True: winner-take-all (highest importance wins)
               - wta_write=False: importance-weighted blend (DLSM default)
        
        Note: Soft slot routing (softmax over slots) naturally handles "writing to
        multiple slots" - each token has a probability distribution over slots.
        No need for top-k multi-slot selection.
        """
        B, S, _ = output.shape
        K = self.num_slots
        device = output.device
        
        # Get local cache
        local_cache = self._get_local_cache(cache)
        old_content, old_confidence, old_temporal = self._split_cache(local_cache)
        
        # Project to cache space using shared projection
        tokens_cache = self.shared.project_to_cache(output)
        
        # === g_write: "Should I write?" ===
        # Compute write decision using shared network
        context_cache, _ = self._attend_to_cache(tokens_cache, old_content)
        decision_logit = self.shared.compute_write_decision_logit(tokens_cache, context_cache)
        write_gate = self.ops.apply_write_gate(decision_logit, hard=hard or not self.training)
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            write_gate = write_gate * mask
        
        # === s_imp: "How important?" ===
        # Compute importance using shared network
        importance_logit = self.shared.compute_importance_logit(output)
        temp = self.ops.write_temperature.clamp(min=0.1)
        # Clamp logit to prevent exp overflow
        importance_logit_clamped = (importance_logit / temp).clamp(min=-10.0, max=10.0)
        importance_raw = torch.exp(importance_logit_clamped)
        importance = importance_raw / (importance_raw.sum(dim=1, keepdim=True) + 1e-8)
        importance = importance * S
        
        # Apply layer-specific scaling
        importance = self.ops.scale_write_importance(importance)
        importance_weights = importance.clamp(min=1e-6)  # [B, S, 1]
        
        # === p_write: "Which slot?" ===
        # Soft slot routing via attention to cache slots
        write_query = self.shared.generate_query(output)
        slot_logits = torch.matmul(write_query, old_content.transpose(-2, -1))
        slot_logits = slot_logits / math.sqrt(self.d_cache)
        slot_logits = slot_logits.clamp(min=-10.0, max=10.0)
        
        # Soft slot selection: probability distribution over slots
        if hard or not self.training:
            slot_probs = gumbel_softmax(slot_logits, temperature, hard=True)
        else:
            slot_probs = gumbel_softmax(slot_logits, temperature, hard=False)
        
        # === COLLISION RESOLUTION ===
        # Compute effective write strength per (token, slot) pair
        # write_strength[b, s, k] = write_gate * importance * slot_prob
        masked_slot_probs = slot_probs * write_gate  # [B, S, K]
        write_strength = masked_slot_probs * importance_weights  # [B, S, K]
        
        if self.wta_write:
            # WTA collision resolution: for each slot, only the strongest token wins
            # Handles case where multiple tokens route to same slot
            write_strength_t = write_strength.transpose(1, 2)  # [B, K, S]
            
            # Stable WTA: use softmax for selection (avoids Gumbel instability)
            # Temperature controls selection sharpness
            wta_tau = 1.0
            
            # Clamp write strength to prevent extreme values
            write_strength_safe = write_strength_t.clamp(min=1e-8)
            
            if hard or not self.training:
                # Hard WTA: argmax winner per slot
                winner_indices = write_strength_t.argmax(dim=-1)  # [B, K]
                winner_mask_t = F.one_hot(winner_indices, num_classes=S).float()  # [B, K, S]
            else:
                # Soft WTA: Use softmax with temperature (more stable than Gumbel)
                # This gives differentiable winner selection
                selection_logits = torch.log(write_strength_safe) / wta_tau
                selection_logits = selection_logits.clamp(min=-20.0, max=20.0)
                winner_mask_t = F.softmax(selection_logits, dim=-1)  # [B, K, S]
            
            # Apply WTA mask: weight each token's contribution by selection probability
            final_weights_t = winner_mask_t * write_strength_t  # [B, K, S]
        else:
            # Blended (DLSM default): all tokens contribute weighted by importance
            final_weights_t = write_strength.transpose(1, 2)  # [B, K, S]
        
        # Compute new slot content
        slot_write_total = final_weights_t.sum(dim=-1, keepdim=True)  # [B, K, 1]
        slot_weights_safe = slot_write_total.clamp(min=0.1)
        
        new_content = torch.matmul(final_weights_t, tokens_cache)  # [B, K, d_cache]
        new_content = new_content / slot_weights_safe
        
        new_confidence = slot_write_total / slot_weights_safe
        
        # has_writes with Straight-Through Estimator for gradient flow
        has_writes_hard = (slot_write_total.squeeze(-1) > 0.1).unsqueeze(-1).float()
        has_writes_soft = torch.sigmoid((slot_write_total.squeeze(-1) - 0.1) * 10.0).unsqueeze(-1)
        has_writes = has_writes_hard - has_writes_soft.detach() + has_writes_soft
        
        # Get temporal embedding from shared
        new_temporal = self.shared.get_temporal_embedding(
            self.layer_idx, iteration_idx, pass_idx, device
        )
        new_temporal = new_temporal.unsqueeze(0).expand(B, K, -1)
        
        if self.soft_eviction:
            update_strength = torch.tanh(slot_write_total.squeeze(-1)).unsqueeze(-1)
            final_content = update_strength * new_content + (1 - update_strength) * old_content
            final_confidence = update_strength * new_confidence + (1 - update_strength) * old_confidence
            is_significant = (update_strength > 0.5).float()
            final_temporal = is_significant * new_temporal + (1 - is_significant) * old_temporal
        else:
            final_content = has_writes * new_content + (1 - has_writes) * old_content
            final_confidence = has_writes * new_confidence + (1 - has_writes) * old_confidence
            final_temporal = has_writes * new_temporal + (1 - has_writes) * old_temporal
        
        new_local_cache = self._merge_cache(final_content, final_confidence, final_temporal)
        updated_cache = self._update_local_cache(cache, new_local_cache)
        
        return {
            'updated_cache': updated_cache,
            'write_gate': write_gate,
            'importance': importance,
            'slot_probs': slot_probs,
            'num_writes': write_gate.sum(dim=1).squeeze(-1),
        }
    
    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = False,
        iteration_idx: int = 0,
        pass_idx: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full read-then-write cycle for LTM.
        
        Args:
            x: Input for read phase [B, S, D_model]
            cache: Global cache state [B, L*K, D_slot]
            output: Output for write phase (defaults to x if None)
            temperature: Gumbel-softmax temperature
            hard: Use hard decisions
            iteration_idx: Current iteration (for temporal embedding)
            pass_idx: Current pass (for temporal embedding)
            mask: Optional mask for valid tokens
            
        Returns:
            Combined results from read and write operations.
        """
        # Read phase
        read_result = self.read(x, cache, hard=hard)
        
        # Write phase (use output if provided, else use x)
        write_input = output if output is not None else x
        write_result = self.write(
            write_input, cache,
            temperature=temperature,
            hard=hard,
            iteration_idx=iteration_idx,
            pass_idx=pass_idx,
            mask=mask,
        )
        
        return {
            'x_enhanced': read_result['x_enhanced'],
            'updated_cache': write_result['updated_cache'],
            'read_gate': read_result['read_gate'],
            'write_gate': write_result['write_gate'],
            'read_attn': read_result['attn_weights'],
            'slot_probs': write_result['slot_probs'],
            'importance': write_result['importance'],
        }


class EfficientVolatileMemory(nn.Module):
    """
    Efficient Working Memory using shared projections (buffer/swap semantics).
    
    WM slot structure (aligned with LTM):
        [content (d_cache) | validity (1) | temporal (d_temporal)]
        where temporal = [layer_embed (8) | iter_embed (4) | pass_embed (4)]
    
    Buffer/Swap Semantics:
    - Read CLEARS slot (validity → 0) - one-time use
    - Write OVERWRITES slot (WTA) - winner takes all
    - Temporal metadata encodes WHO (layer) and WHEN (iter, pass)
    - No decay - content persists until explicitly read
    
    READ Flow:
        1. g_read: "Should I read?" (per-token gate)
        2. Attend to VALID slots (validity > 0.5)
        3. Blend relevant slots via attention
        4. CLEAR read slots (set validity = 0)
    
    WRITE Flow:
        1. g_write: "Should I write?" (per-token gate)
        2. s_imp: "How important?" (importance score)
        3. p_write: "Which slot?" (prefer invalid/empty slots)
        4. WTA collision resolution: highest importance wins
        5. Set validity = 1, store temporal metadata
    
    Args:
        shared: SharedWMProjections instance
        layer_idx: This layer's index
        num_slots: Number of working memory slots (typically 4-8)
        d_model: Model hidden dimension
        d_cache: Slot content dimension
        dropout: Dropout probability
        wta_write: Winner-take-all for write collisions (always True for WM)
    """
    
    def __init__(
        self,
        shared: SharedWMProjections,
        layer_idx: int,
        num_slots: int = 8,
        d_model: int = 64,
        d_cache: int = 48,
        read_decay: float = 0.0,  # DEPRECATED - kept for API compat, ignored
        dropout: float = 0.1,
        freshness_threshold: float = 0.1,  # DEPRECATED - now uses validity
        wta_write: bool = True,
    ):
        super().__init__()
        self.shared = shared
        self.layer_idx = layer_idx
        self.num_slots = num_slots
        self.d_model = d_model
        self.d_cache = d_cache
        self.wta_write = True  # Always WTA for buffer semantics
        
        # Import slot dimensions (aligned with LTM)
        from .config import SLOT_DIMS
        self.d_meta = SLOT_DIMS.d_meta  # 17
        self.d_temporal = SLOT_DIMS.d_temporal  # 16
        
        # Full slot: content + validity + temporal (same as LTM)
        self.d_full_slot = d_cache + self.d_meta
        
        # Lightweight per-layer operations
        self.ops = LightweightMemoryOps(layer_idx, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize empty working memory (all slots invalid)."""
        return torch.zeros(batch_size, self.num_slots, self.d_full_slot, device=device)
    
    def _split_slot(self, wm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split slot into content, validity, and temporal."""
        content = wm[..., :self.d_cache]
        validity = wm[..., self.d_cache:self.d_cache+1]
        temporal = wm[..., self.d_cache+1:]
        return content, validity, temporal
    
    def _merge_slot(
        self, 
        content: torch.Tensor, 
        validity: torch.Tensor, 
        temporal: torch.Tensor
    ) -> torch.Tensor:
        """Merge content, validity, and temporal into full slot."""
        return torch.cat([content, validity, temporal], dim=-1)
    
    def read(
        self,
        x: torch.Tensor,
        wm: torch.Tensor,
        hard: bool = False,
        iter_idx: int = 0,
        pass_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Read from working memory with CLEAR-on-read semantics.
        
        Buffer semantics: reading a slot clears it (validity → 0).
        This makes WM a one-time-use swap buffer.
        
        Flow:
            1. g_read: "Should I read?" (per-token gate)
            2. Attend to VALID slots only (validity > 0.5)
            3. Blend relevant slots via attention
            4. CLEAR slots that were read (validity → 0)
        """
        B, S, _ = x.shape
        
        content, validity, temporal = self._split_slot(wm)
        
        # Compute read gate using shared network + layer-specific bias
        gate_logit = self.shared.compute_read_gate_logit(x)
        gate = self.ops.apply_read_gate(gate_logit, hard=hard)
        
        # Generate query using shared projection
        query = self.shared.generate_wm_query(x)
        
        # Attention scores
        scores = torch.matmul(query, content.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_cache)
        
        # Mask INVALID slots (validity < 0.5)
        validity_t = validity.transpose(-2, -1)  # [B, 1, K]
        invalid_mask = (validity_t < 0.5)
        scores = scores.masked_fill(invalid_mask, -1e9)
        
        # Check for empty memory (all slots invalid)
        has_valid_slots_hard = (~invalid_mask).any(dim=-1, keepdim=True).float()  # [B, S, 1]
        
        # Soft approximation for gradients
        soft_valid = torch.sigmoid((validity_t - 0.5).sum(dim=-1, keepdim=True) * 10.0)
        has_valid_slots = has_valid_slots_hard - soft_valid.detach() + soft_valid
        
        # Attention over valid slots
        attn_weights = F.softmax(scores, dim=-1)  # [B, S, K]
        attn_weights = attn_weights * has_valid_slots
        attn_weights = self.dropout(attn_weights)
        
        # Retrieve context
        context = torch.matmul(attn_weights, content)
        context = self.shared.project_from_wm(context)
        
        # Apply layer-specific scaling
        context = self.ops.scale_read_output(context)
        
        # Gated fusion
        fused = self.shared.fuse_context(x, context)
        
        # Only read if we have valid slots
        effective_gate = gate * has_valid_slots
        x_enhanced = (1 - effective_gate) * x + effective_gate * fused
        
        # === CLEAR-on-read: invalidate slots that were read ===
        # Sum attention across all tokens to get per-slot read pressure
        read_pressure = attn_weights.sum(dim=1, keepdim=True).transpose(-2, -1)  # [B, K, 1]
        
        # Slots with significant read pressure get cleared
        if hard:
            was_read = (read_pressure > 0.1).float()
            # STE for gradients
            was_read_soft = torch.sigmoid((read_pressure - 0.1) * 20.0)
            was_read = was_read - was_read_soft.detach() + was_read_soft
        else:
            was_read = torch.sigmoid((read_pressure - 0.1) * 20.0)
        
        # Clear validity for read slots (validity → 0)
        new_validity = validity * (1.0 - was_read)
        
        # Content and temporal stay the same (just marked invalid)
        wm_updated = self._merge_slot(content, new_validity, temporal)
        
        return {
            'x_enhanced': x_enhanced,
            'wm_updated': wm_updated,
            'read_gate': gate,
            'attn_weights': attn_weights,
            'context': context,
        }
    
    def write(
        self,
        x: torch.Tensor,
        wm: torch.Tensor,
        hard: bool = False,
        mask: Optional[torch.Tensor] = None,
        iter_idx: int = 0,
        pass_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Write to working memory with WTA overwrite semantics.
        
        Buffer semantics: writes OVERWRITE slots (not blend).
        Prefers invalid/empty slots. Stores temporal metadata.
        
        Flow:
            1. g_write: "Should I write?" (per-token gate)
            2. s_imp: "How important?" (importance score)
            3. p_write: "Which slot?" (prefer invalid slots)
            4. WTA: highest importance token wins each slot
            5. Store content + validity=1 + temporal metadata
        """
        B, S, _ = x.shape
        K = self.num_slots
        device = x.device
        
        old_content, old_validity, old_temporal = self._split_slot(wm)
        
        # Project to slot space
        x_slot = self.shared.project_to_wm(x)
        
        # === g_write: "Should I write?" ===
        # Context from existing slots (for novelty assessment)
        wm_context = torch.matmul(
            F.softmax(torch.matmul(x_slot, old_content.transpose(-2, -1)) / math.sqrt(self.d_cache), dim=-1),
            old_content
        )
        decision_logit = self.shared.compute_write_decision_logit(x_slot, wm_context)
        write_gate = self.ops.apply_write_gate(decision_logit, hard=hard)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            write_gate = write_gate * mask
        
        # === s_imp: "How important?" ===
        importance_logit = self.shared.compute_importance_logit(x)
        temp = self.ops.write_temperature.clamp(min=0.1)
        importance_logit_clamped = (importance_logit / temp).clamp(min=-10.0, max=10.0)
        importance = torch.exp(importance_logit_clamped)
        importance = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)
        importance = importance * S
        importance = self.ops.scale_write_importance(importance)
        importance_weights = importance.clamp(min=1e-6)  # [B, S, 1]
        
        # === p_write: "Which slot?" ===
        # Prefer INVALID slots (via validity penalty on valid slots)
        write_query = self.shared.generate_wm_query(x)
        content_scores = torch.matmul(write_query, old_content.transpose(-2, -1))
        content_scores = content_scores / math.sqrt(self.d_cache)
        
        # Strongly prefer invalid slots: penalize valid slots
        validity_penalty = old_validity.transpose(-2, -1) * 5.0  # [B, 1, K]
        slot_scores = content_scores - validity_penalty  # [B, S, K]
        
        # Soft slot selection
        slot_selection = F.softmax(slot_scores, dim=-1)  # [B, S, K]
        
        # === WTA COLLISION RESOLUTION ===
        write_strength = slot_selection * write_gate * importance_weights  # [B, S, K]
        write_strength_t = write_strength.transpose(1, 2)  # [B, K, S]
        
        # WTA: for each slot, only the strongest token wins
        write_strength_safe = write_strength_t.clamp(min=1e-8)
        
        if hard or not self.training:
            # Hard WTA: argmax winner per slot
            winner_indices = write_strength_t.argmax(dim=-1)  # [B, K]
            winner_mask_t = F.one_hot(winner_indices, num_classes=S).float()  # [B, K, S]
        else:
            # Soft WTA via softmax
            selection_logits = torch.log(write_strength_safe).clamp(min=-20.0, max=20.0)
            winner_mask_t = F.softmax(selection_logits, dim=-1)  # [B, K, S]
        
        final_weights_t = winner_mask_t * write_strength_t  # [B, K, S]
        
        # === UPDATE SLOTS ===
        slot_write_total = final_weights_t.sum(dim=-1, keepdim=True)  # [B, K, 1]
        slot_write_safe = slot_write_total.clamp(min=1e-8)
        
        # New content: weighted average of writing tokens
        new_content = torch.matmul(final_weights_t, x_slot)  # [B, K, d_cache]
        new_content = new_content / slot_write_safe
        
        # Determine which slots received writes
        if hard:
            has_writes = (slot_write_total > 0.1).float()
            has_writes_soft = torch.sigmoid((slot_write_total - 0.1) * 10.0)
            has_writes = has_writes - has_writes_soft.detach() + has_writes_soft
        else:
            has_writes = torch.sigmoid((slot_write_total - 0.1) * 10.0)
        
        # Get temporal embedding for this write
        temporal_emb = self.shared.get_temporal_embedding(
            self.layer_idx, iter_idx, pass_idx, device
        )  # [1, d_temporal]
        temporal_emb = temporal_emb.unsqueeze(0).expand(B, K, -1)  # [B, K, d_temporal]
        
        # Update: written slots get new content + validity=1 + new temporal
        final_content = has_writes * new_content + (1 - has_writes) * old_content
        final_validity = has_writes * 1.0 + (1 - has_writes) * old_validity
        final_temporal = has_writes * temporal_emb + (1 - has_writes) * old_temporal
        
        wm_updated = self._merge_slot(final_content, final_validity, final_temporal)
        
        return {
            'wm_updated': wm_updated,
            'write_gate': write_gate,
            'importance': importance_weights,
            'slot_selection': slot_selection,
            'num_writes': write_gate.sum(dim=1).squeeze(-1),
        }
    
    def forward(
        self,
        x: torch.Tensor,
        wm: torch.Tensor,
        hard: bool = False,
        mask: Optional[torch.Tensor] = None,
        iter_idx: int = 0,
        pass_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Combined read then write."""
        read_result = self.read(x, wm, hard=hard, iter_idx=iter_idx, pass_idx=pass_idx)
        write_result = self.write(
            read_result['x_enhanced'], 
            read_result['wm_updated'], 
            hard=hard, 
            mask=mask,
            iter_idx=iter_idx,
            pass_idx=pass_idx,
        )
        
        return {
            'x_enhanced': read_result['x_enhanced'],
            'wm_updated': write_result['wm_updated'],
            'read_gate': read_result['read_gate'],
            'write_gate': write_result['write_gate'],
            'read_attn': read_result['attn_weights'],
            'write_slot_selection': write_result['slot_selection'],
        }
