"""
Unified Memory Controller (v0.2.0)
==================================

A modular, pluggable memory controller following the proposal.md design.

Key Design Principles:
----------------------
1. **Separation of Concerns**: Memory operations are independent from computation
2. **Pluggable Interface**: Can be added to any architecture (Transformer, CNN, etc.)
3. **Unified Read/Write**: Single controller handles both directions
4. **Output-Based Decisions**: All decisions based on output quality, not cache state

Architecture:
-------------
    Input x ──────────────────────────────────────
        │
        ▼
    ┌─────────────────────────────────────────┐
    │  Memory Controller (READ)               │
    │  • Query generation from input          │
    │  • Attention over cache slots           │
    │  • Gated fusion with input              │
    └─────────────────────────────────────────┘
        │
        ▼ (enhanced_input)
    ┌─────────────────────────────────────────┐
    │  Base Computation Block                 │
    │  (Transformer, MLP, etc.)               │
    └─────────────────────────────────────────┘
        │
        ▼ (output)
    ┌─────────────────────────────────────────┐
    │  Memory Controller (WRITE)              │
    │  • Importance scoring                   │
    │  • Slot selection                       │
    │  • Selective cache update               │
    └─────────────────────────────────────────┘
        │
        ▼ (output, updated_cache)

Usage:
------
```python
# Standalone usage
controller = MemoryController(d_model=128, d_cache=64, num_slots=16)

# In a custom layer
class MyLayer(nn.Module):
    def __init__(self):
        self.memory = MemoryController(...)
        self.compute = TransformerBlock(...)

    def forward(self, x, cache):
        # Read phase
        x_enhanced, context = self.memory.read(x, cache)

        # Compute
        output = self.compute(x_enhanced)

        # Write phase
        updated_cache = self.memory.write(output, cache)

        return output, updated_cache
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

from .utils import gumbel_softmax
from .config import SLOT_DIMS


class MemoryController(nn.Module):
    """
    Unified Memory Controller - Modular and Pluggable.

    This controller can be attached to ANY computation block to add
    selective memory capabilities. It handles:

    1. READ: Query cache based on input content, gate fusion
    2. WRITE: Score output importance, select slots, update cache

    The controller does NOT include computation - that's the job of
    the base architecture block it's attached to.

    Args:
        d_model:    Model hidden dimension
        d_cache:    Cache slot dimension
        num_slots:  Number of slots per layer
        num_layers: Total layers (for global cache indexing)
        layer_idx:  This layer's index (for local cache access)
        dropout:    Dropout probability
        use_fixed_threshold: Use fixed 0.5 threshold instead of learned
        fixed_threshold: Value for fixed threshold (default 0.5)
    """

    def __init__(
        self,
        d_model: int,
        d_cache: int,
        num_slots: int,
        num_layers: int = 1,
        layer_idx: int = 0,
        dropout: float = 0.1,
        max_iterations: int = 4,  # For temporal embeddings
        max_passes: int = 6,      # For temporal embeddings
        use_fixed_threshold: bool = True,
        fixed_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        self.total_slots = num_layers * num_slots
        self.use_fixed_threshold = use_fixed_threshold
        self.fixed_threshold = fixed_threshold
        
        # === Slot Metadata Dimensions (from centralized config) ===
        self.d_layer_embed = SLOT_DIMS.d_layer_embed
        self.d_iter_embed = SLOT_DIMS.d_iter_embed
        self.d_pass_embed = SLOT_DIMS.d_pass_embed
        self.d_meta = SLOT_DIMS.d_meta
        self.d_slot = SLOT_DIMS.d_slot(d_cache)
        
        # === Temporal Embedding Tables ===
        self.layer_embed = nn.Embedding(num_layers, self.d_layer_embed)
        self.iter_embed = nn.Embedding(max_iterations, self.d_iter_embed)
        self.pass_embed = nn.Embedding(max_passes, self.d_pass_embed)

        # === Space Projections ===
        self.to_cache = nn.Linear(d_model, d_cache)
        # from_cache now takes content + metadata (confidence as feature)
        self.from_cache = nn.Linear(self.d_slot, d_model)

        # === READ Components ===
        # Query generator: input → cache space query
        self.read_query = nn.Linear(d_model, d_cache)

        # Read gate: INPUT-ONLY decision on whether this token needs context
        # Per spec: "σ(W_r · x) - Does this token need context?"
        # Decision happens BEFORE attending to cache - pure input-driven
        self.read_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        # === WRITE Components ===
        # Write query: output → cache space for slot selection
        self.write_query = nn.Linear(d_model, d_cache)

        # Importance scorer: how valuable is this content intrinsically?
        # Used for weighting token contributions during collision (multi-token → same slot)
        self.importance_scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # === Unified Write Decision Gate ===
        # Single gate that decides whether to write by comparing output to cache.
        # Input: [output_in_cache_space, retrieved_cache_context] → binary decision
        # This replaces separate write_gate (output-only) + novelty_gate (comparison)
        # The comparison IS the decision: "Is this worth writing given what's already stored?"
        self.write_decision = nn.Sequential(
            nn.LayerNorm(d_cache * 2),
            nn.Linear(d_cache * 2, d_cache),
            nn.SiLU(),
            nn.Linear(d_cache, 1),
        )
        
        # === Learned Thresholds (context-dependent) ===
        # Instead of hardcoded 0.5, learn context-dependent thresholds.
        # This allows stricter/looser gating based on the situation.
        
        # Write threshold: based on cache context (stricter if cache has good info)
        self.write_threshold = nn.Sequential(
            nn.Linear(d_cache, d_cache // 2),
            nn.SiLU(),
            nn.Linear(d_cache // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Read threshold: based on input (stricter if input is self-sufficient)
        self.read_threshold = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # NOTE: Token collision handling uses importance-weighted averaging.
        # When multiple tokens write to the same slot, their content is merged
        # weighted by their importance scores. This is differentiable and lets
        # the model learn to combine different concepts into a single slot.
        # Slot eviction is HARD - when a slot receives a write, old content
        # is fully replaced (no blending with previous slot content).

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with appropriate strategies."""
        # Xavier for query/projection layers
        for module in [self.to_cache, self.from_cache, self.read_query, self.write_query]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Sequential modules
        for seq in [self.read_gate, self.importance_scorer, self.write_decision,
                    self.write_threshold, self.read_threshold]:
            for m in seq:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        # Bias threshold networks to start near 0.5
        # This gives neutral initial behavior
        if self.write_threshold[-2].bias is not None:
            nn.init.zeros_(self.write_threshold[-2].bias)  # sigmoid(0) = 0.5
        if self.read_threshold[-2].bias is not None:
            nn.init.zeros_(self.read_threshold[-2].bias)  # sigmoid(0) = 0.5
        
        # Initialize temporal embeddings
        nn.init.normal_(self.layer_embed.weight, std=0.02)
        nn.init.normal_(self.iter_embed.weight, std=0.02)
        nn.init.normal_(self.pass_embed.weight, std=0.02)

    def _get_local_cache(self, cache: torch.Tensor) -> torch.Tensor:
        """Extract this layer's slots from global cache (content + metadata)."""
        start = self.layer_idx * self.num_slots
        end = start + self.num_slots
        return cache[:, start:end, :]  # [B, K, D_slot]
    
    def _split_cache(self, cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split cache into content and metadata components.
        
        Args:
            cache: [B, K, D_slot] where D_slot = D_cache + D_meta
        
        Returns:
            content: [B, K, D_cache] - the actual cached representations
            confidence: [B, K, 1] - slot confidence scores
            temporal: [B, K, D_layer + D_iter + D_pass] - temporal embeddings
        """
        content = cache[..., :self.d_cache]  # [B, K, D_cache]
        confidence = cache[..., self.d_cache:self.d_cache+1]  # [B, K, 1]
        temporal = cache[..., self.d_cache+1:]  # [B, K, D_layer + D_iter + D_pass]
        return content, confidence, temporal
    
    def _merge_cache(self, content: torch.Tensor, confidence: torch.Tensor, 
                     temporal: torch.Tensor) -> torch.Tensor:
        """
        Merge content and metadata back into cache tensor.
        
        Args:
            content: [B, K, D_cache]
            confidence: [B, K, 1]
            temporal: [B, K, D_layer + D_iter + D_pass]
        
        Returns:
            cache: [B, K, D_slot]
        """
        return torch.cat([content, confidence, temporal], dim=-1)

    def _update_local_cache(
        self,
        cache: torch.Tensor,
        new_local: torch.Tensor
    ) -> torch.Tensor:
        """Update this layer's slots in global cache (in-place safe)."""
        # Use index_copy_ style update to avoid full clone
        # This creates a new tensor only for the updated region
        start = self.layer_idx * self.num_slots
        end = start + self.num_slots
        
        # Create output by concatenating unchanged + new + unchanged
        if start == 0:
            if end == cache.shape[1]:
                return new_local  # Cache is just this layer
            return torch.cat([new_local, cache[:, end:, :]], dim=1)
        elif end == cache.shape[1]:
            return torch.cat([cache[:, :start, :], new_local], dim=1)
        else:
            return torch.cat([
                cache[:, :start, :],
                new_local,
                cache[:, end:, :]
            ], dim=1)

    def _attend_to_cache(
        self,
        query: torch.Tensor,      # [B, S, D_cache]
        cache_content: torch.Tensor,  # [B, K, D_cache] - content only, not metadata
        full_cache: Optional[torch.Tensor] = None,  # [B, K, D_slot] - if provided, retrieve from this
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention to cache slots.
        
        Query matching is done against cache_content (d_cache dims).
        If full_cache is provided, retrieval is from full_cache (includes metadata).
        This allows confidence and temporal info to be "features" during read.

        Returns:
            context: [B, S, D_cache] or [B, S, D_slot] - retrieved information
            attn_weights: [B, S, K] - attention distribution
        """
        # Attention scores (query against content only)
        scores = torch.matmul(query, cache_content.transpose(-2, -1))  # [B, S, K]
        scores = scores / math.sqrt(self.d_cache)
        scores = scores.clamp(min=-20.0, max=20.0)  # Numerical stability

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Retrieve context (from full cache if provided, else just content)
        retrieve_from = full_cache if full_cache is not None else cache_content
        context = torch.matmul(attn_weights, retrieve_from)

        return context, attn_weights

    def read(
        self,
        x: torch.Tensor,                       # [B, S, D_model]
        cache: torch.Tensor,                   # [B, L*K, D_cache]
        use_global: bool = False,              # Read from all layers or just local
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        READ Phase: Query memory and fuse with input.

        Process:
        1. Generate query from input content
        2. Attend to cache slots (local or global)
        3. Gate fusion based on relevance

        Args:
            x: Input tensor [B, S, D_model]
            cache: Global cache [B, L*K, D_cache]
            use_global: If True, attend to entire cache; else local only
            temperature: Softmax temperature
            hard: Use hard gating (inference)

        Returns:
            Dict with:
                - 'x_enhanced': Input fused with memory [B, S, D_model]
                - 'context': Retrieved context [B, S, D_model]
                - 'read_gate': Gate values [B, S, 1]
                - 'attn_weights': Attention over slots [B, S, K]
        """
        B, S, _ = x.shape

        # === Step 1: Compute read gate from INPUT ONLY (before attending) ===
        # Per spec: "σ(W_r · x) - Does this token need context?"
        gate_logit = self.read_gate(x)  # [B, S, 1]
        soft_gate = torch.sigmoid(gate_logit)
        
        # Threshold: fixed or learned
        if self.use_fixed_threshold:
            # Fixed threshold - forces gate VALUES to be decisive
            # Polarization loss will push soft_gate to 0 or 1
            read_thresh = torch.full_like(soft_gate, self.fixed_threshold)
        else:
            # Learned context-dependent threshold
            read_thresh = self.read_threshold(x)  # [B, S, 1] in [0, 1]

        if hard or not self.training:
            read_gate = (soft_gate > read_thresh).float()
        else:
            # STE: hard forward, soft backward
            hard_gate = (soft_gate > read_thresh).float()
            read_gate = hard_gate - soft_gate.detach() + soft_gate

        # === Step 2: Only attend if gate is open (sparse optimization possible) ===
        # Generate read query from input
        query = self.read_query(x)  # [B, S, D_cache]

        # Select cache to read from (full slot including metadata)
        read_cache = cache if use_global else self._get_local_cache(cache)
        
        # Split cache to get content for query matching
        cache_content, cache_confidence, cache_temporal = self._split_cache(read_cache)

        # Attend to cache: query matches content, but retrieve full slot (content + metadata)
        # This way confidence and temporal info become "features" the model can use
        context_full, attn_weights = self._attend_to_cache(
            query, cache_content, full_cache=read_cache
        )  # [B, S, D_slot]

        # Project full context (content + metadata as features) to model space
        context = self.from_cache(context_full)  # [B, S, D_model]

        # === Step 3: Gated fusion ===
        # Only incorporate context where read_gate is open
        x_enhanced = x + read_gate * context

        return {
            'x_enhanced': x_enhanced,
            'context': context,
            'read_gate': read_gate,
            'soft_read_gate': soft_gate,  # For polarization loss (pre-threshold)
            'read_threshold': read_thresh,  # Include threshold tensor for gradient flow
            'attn_weights': attn_weights,
        }

    def write(
        self,
        output: torch.Tensor,                  # [B, S, D_model]
        cache: torch.Tensor,                   # [B, L*K, D_slot]
        temperature: float = 1.0,
        hard: bool = False,

        iteration_idx: int = 0,                # Current layer iteration
        pass_idx: int = 0,                     # Current model pass
        # Centralized temporal embeddings (from model) - use these if provided
        cache_layer_embed: Optional[nn.Embedding] = None,
        cache_iter_embed: Optional[nn.Embedding] = None,
        cache_pass_embed: Optional[nn.Embedding] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        WRITE Phase: Selectively update cache with output content.

        Process:
        1. Score token importance
        2. Gate which tokens should write
        3. Select target slots via attention
        4. Update cache with learned blend gate

        Args:
            output: Computation output [B, S, D_model]
            cache: Global cache [B, L*K, D_slot] (content + metadata)
            temperature: Gumbel-softmax temperature
            hard: Use hard slot selection

            iteration_idx: Current iteration within layer (for temporal embedding)
            pass_idx: Current model-level pass (for temporal embedding)

        Returns:
            Dict with:
                - 'updated_cache': New cache [B, L*K, D_slot]
                - 'write_gate': Which tokens wrote [B, T, 1]
                - 'importance': Token importance scores [B, T, 1]
                - 'slot_probs': Slot selection probs [B, T, K]
        """
        B, S, _ = output.shape
        K = self.num_slots
        device = output.device

        tokens = output  # [B, S, D_model] - all tokens are candidates

        # Get local cache and split into content/metadata
        local_cache = self._get_local_cache(cache)  # [B, K, D_slot]
        old_content, old_confidence, old_temporal = self._split_cache(local_cache)
        # old_content: [B, K, D_cache], old_confidence: [B, K, 1]

        # Project tokens to cache space
        tokens_cache = self.to_cache(tokens)  # [B, T, D_cache]

        # === Step 1: Importance Scoring (for token collision weighting) ===
        importance = self.importance_scorer(tokens)  # [B, T, 1]

        # === Step 2: Write Decision (unified gate comparing output to cache) ===
        # Single gate that answers: "Is this worth writing given what's already stored?"
        # Compare output to what it would retrieve from cache
        context_cache, _ = self._attend_to_cache(tokens_cache, old_content)
        combined = torch.cat([tokens_cache, context_cache], dim=-1)  # [B, T, 2*D_cache]
        decision_logit = self.write_decision(combined)  # [B, T, 1]
        soft_decision = torch.sigmoid(decision_logit)
        
        # Threshold: fixed or learned
        if self.use_fixed_threshold:
            # Fixed threshold - forces gate VALUES to be decisive
            # Polarization loss will push soft_decision to 0 or 1
            write_thresh = torch.full_like(soft_decision, self.fixed_threshold)
        else:
            # Learned context-dependent threshold (based on cache context)
            # Stricter threshold if cache already has relevant info
            write_thresh = self.write_threshold(context_cache)  # [B, T, 1] in [0, 1]

        if hard or not self.training:
            write_gate = (soft_decision > write_thresh).float()
        else:
            # STE: hard forward, soft backward
            hard_gate = (soft_decision > write_thresh).float()
            write_gate = hard_gate - soft_decision.detach() + soft_decision

        # === Step 3: Slot Selection ===
        # Query for which slots to write to (match against content only)
        write_query = self.write_query(tokens)  # [B, T, D_cache]
        slot_logits = torch.matmul(write_query, old_content.transpose(-2, -1))
        slot_logits = slot_logits / math.sqrt(self.d_cache)
        slot_logits = slot_logits.clamp(min=-10.0, max=10.0)

        # Gumbel-softmax for differentiable selection
        if hard or not self.training:
            slot_probs = gumbel_softmax(slot_logits, temperature, hard=True)
        else:
            slot_probs = gumbel_softmax(slot_logits, temperature, hard=False)

        # === Step 6: Cache Update with HARD Eviction ===
        # Token collision: multiple tokens writing to same slot are merged via
        # importance-weighted averaging. This is the "learned blend" for collision.
        # Slot eviction: slots that receive writes are FULLY REPLACED (hard eviction).
        
        # Mask by write gate
        masked_slot_probs = slot_probs * write_gate  # [B, T, K]
        importance_weights = importance.clamp(min=1e-6)  # [B, T, 1]
        
        # === Token Collision Blending ===
        # When multiple tokens target the same slot, blend their content
        # weighted by importance. This lets the model merge concepts.
        weighted_tokens = tokens_cache * importance_weights  # [B, T, D_cache]

        # Aggregate to slots: [B, K, T] @ [B, T, D_cache] → [B, K, D_cache]
        slot_probs_t = masked_slot_probs.transpose(1, 2)  # [B, K, T]
        new_content = torch.matmul(slot_probs_t, weighted_tokens)  # [B, K, D_cache]

        # Normalize by total weight per slot (importance-weighted average)
        slot_weights = torch.matmul(slot_probs_t, importance_weights.squeeze(-1).unsqueeze(-1))
        slot_weights_safe = slot_weights.clamp(min=0.1)
        new_content = new_content / slot_weights_safe
        
        # Compute new slot confidence as mean importance of writing tokens
        new_confidence = torch.matmul(slot_probs_t, importance)  # [B, K, 1]
        new_confidence = new_confidence / slot_weights_safe

        # Determine which slots received writes
        has_writes = (slot_weights.squeeze(-1) > 0.1).unsqueeze(-1).float()  # [B, K, 1]
        
        # Build temporal embedding for newly written content
        # Use centralized embeddings if provided, otherwise fall back to local
        layer_embed_table = cache_layer_embed if cache_layer_embed is not None else self.layer_embed
        iter_embed_table = cache_iter_embed if cache_iter_embed is not None else self.iter_embed
        pass_embed_table = cache_pass_embed if cache_pass_embed is not None else self.pass_embed
        
        layer_emb = layer_embed_table(torch.tensor([self.layer_idx], device=device))  # [1, D_layer]
        iter_emb = iter_embed_table(torch.tensor([iteration_idx], device=device))    # [1, D_iter]
        pass_emb = pass_embed_table(torch.tensor([pass_idx], device=device))         # [1, D_pass]
        new_temporal = torch.cat([layer_emb, iter_emb, pass_emb], dim=-1)  # [1, D_temporal]
        new_temporal = new_temporal.unsqueeze(0).expand(B, K, -1)  # [B, K, D_temporal]
        
        # === HARD EVICTION ===
        # Slots that received writes get fully replaced with new content.
        # Slots that didn't receive writes keep their old content unchanged.
        final_content = has_writes * new_content + (1 - has_writes) * old_content
        final_confidence = has_writes * new_confidence + (1 - has_writes) * old_confidence
        final_temporal = has_writes * new_temporal + (1 - has_writes) * old_temporal
        
        # Merge back into full cache format
        new_local_cache = self._merge_cache(final_content, final_confidence, final_temporal)

        # Update global cache
        updated_cache = self._update_local_cache(cache, new_local_cache)

        return {
            'updated_cache': updated_cache,
            'write_gate': write_gate,
            'soft_write_gate': soft_decision,  # For polarization loss (pre-threshold)
            'write_threshold': write_thresh,  # Include threshold tensor for gradient flow
            'write_decision': soft_decision,  # For debugging/analysis (alias)
            'importance': importance,
            'slot_probs': slot_probs,
            'num_writes': write_gate.sum(dim=1).squeeze(-1),  # [B]
        }

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full read-write cycle.

        If output is None, only performs READ.
        If output is provided, performs READ then WRITE.

        This is a convenience method - you can also call read() and write()
        separately for more control.
        """
        # Read phase
        read_result = self.read(x, cache, temperature=temperature, hard=hard)

        result = {
            'x_enhanced': read_result['x_enhanced'],
            'read_context': read_result['context'],
            'read_gate': read_result['read_gate'],
            'read_attn': read_result['attn_weights'],
            'updated_cache': cache,  # Default: unchanged
        }

        # Write phase (if output provided)
        if output is not None:
            write_result = self.write(output, cache, temperature=temperature, hard=hard)
            result['updated_cache'] = write_result['updated_cache']
            result['write_gate'] = write_result['write_gate']
            result['write_importance'] = write_result['importance']
            result['write_slot_probs'] = write_result['slot_probs']

        return result


class LayerHaltEstimator(nn.Module):
    """
    Per-layer halting estimator based on info gain and budget.
    
    Each layer gets its own estimator because:
    1. Different layers have different roles (feature extraction vs reasoning)
    2. Layers may need different amounts of refinement
    3. Layer-specific patterns in when to halt
    
    Halt criterion: Combines info gain with remaining budget
    - High info gain = making progress → continue
    - Low info gain = diminishing returns → halt
    - Low budget = save compute → halt even if making some progress
    
    The threshold network learns to balance these factors.
    """
    
    def __init__(self, d_model: int, layer_idx: int, max_iterations: int = 8):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.max_iterations = max_iterations
        
        # Lightweight analyzer (per-layer, so keep small)
        self.h_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model // 8),
        )
        
        # Halt threshold: decides when THIS layer should stop iterating
        # Input: h_features + info_gain + budget
        # Now uses info_gain instead of stability
        self.halt_threshold = nn.Sequential(
            nn.Linear(d_model // 8 + 2, d_model // 16),  # +2 for info_gain, budget
            nn.SiLU(),
            nn.Linear(d_model // 16, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize threshold to encourage more iterations initially
        # Low threshold = need low info_gain to halt = iterate more
        with torch.no_grad():
            for layer in self.halt_threshold:
                if isinstance(layer, nn.Linear) and layer.out_features == 1:
                    layer.bias.fill_(-1.0)  # sigmoid(-1) ≈ 0.27 (low threshold)
    
    def compute_info_gain(
        self, 
        h: torch.Tensor,                 # [B, S, D]
        h_prev: Optional[torch.Tensor],  # [B, S, D]
    ) -> torch.Tensor:
        """
        Compute information gain via entropy change of hidden state.
        
        Info gain = |H(h_prev) - H(h)| (absolute entropy change)
        High info gain = iteration made meaningful change
        Low info gain = representation settled, can halt
        """
        if h_prev is None:
            # First iteration - return high info gain to encourage continuation
            return torch.ones(h.shape[0], device=h.device) * 0.5
        
        B = h.shape[0]
        
        # Compute entropy of hidden state distributions
        # Treat hidden dim as distribution via softmax
        p_curr = F.softmax(h, dim=-1)  # [B, S, D]
        p_prev = F.softmax(h_prev, dim=-1)  # [B, S, D]
        
        # Entropy: -sum(p * log(p))
        entropy_curr = -(p_curr * torch.log(p_curr + 1e-8)).sum(dim=-1).mean(dim=-1)  # [B]
        entropy_prev = -(p_prev * torch.log(p_prev + 1e-8)).sum(dim=-1).mean(dim=-1)  # [B]
        
        # Info gain = absolute entropy change (direction doesn't matter)
        info_gain = (entropy_prev - entropy_curr).abs()  # [B]
        
        # Normalize to roughly [0, 1] range (entropy is in [0, log(D)])
        max_entropy = math.log(self.d_model)
        info_gain = (info_gain / max_entropy).clamp(0, 1)
        
        return info_gain
    
    # def compute_stability(
    #     self, 
    #     h: torch.Tensor,           # [B, S, D]
    #     h_prev: Optional[torch.Tensor],  # [B, S, D]
    # ) -> torch.Tensor:
    #     """
    #     Compute hidden state stability via cosine similarity.
    #     DEPRECATED: Kept for backward compatibility, but info_gain is preferred.
    #     """
    #     if h_prev is None:
    #         return torch.zeros(h.shape[0], device=h.device)
        
    #     B = h.shape[0]
    #     h_flat = h.view(B, -1)
    #     h_prev_flat = h_prev.view(B, -1)
    #     stability = F.cosine_similarity(h_flat, h_prev_flat, dim=-1)
    #     return stability.clamp(0, 1)
    
    def forward(
        self,
        h: torch.Tensor,                      # [B, S, D]
        h_prev: Optional[torch.Tensor] = None,  # [B, S, D]
        current_iter: int = 0,
        max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute info gain and halt threshold.
        
        Halt decision: halt if info_gain < threshold
        - Low info_gain = not making progress → halt
        - Threshold adapts based on budget and hidden state
        
        Returns:
            info_gain: [B] information gain (replaces stability)
            threshold: [B] learned halt threshold
        """
        max_iter = max_iter or self.max_iterations
        B = h.shape[0]
        device = h.device
        
        # Compute info gain (replaces stability)
        info_gain = self.compute_info_gain(h, h_prev)  # [B]
        
        # Hidden state features
        h_pooled = h.mean(dim=1)  # [B, D]
        h_features = self.h_analyzer(h_pooled)  # [B, D//8]
        
        # Budget feature: how much compute budget remains
        # Low budget → lower threshold (more willing to halt)
        budget_remaining = (max_iter - 1 - current_iter) / max(max_iter - 1, 1)
        budget_tensor = torch.full((B, 1), budget_remaining, device=device, dtype=h.dtype)
        
        # Combine features: h_features + info_gain + budget
        info_gain_tensor = info_gain.unsqueeze(-1)  # [B, 1]
        features = torch.cat([h_features, info_gain_tensor, budget_tensor], dim=-1)
        
        # Compute threshold
        # Threshold adjusts based on budget:
        # - High budget: higher threshold (require more info_gain to continue)
        # - Low budget: lower threshold (halt even with some info_gain)
        threshold = self.halt_threshold(features).squeeze(-1)  # [B]
        
        return info_gain, threshold
        threshold = self.halt_threshold(features).squeeze(-1)  # [B]
        
        return stability, threshold


class ModelHaltEstimator(nn.Module):
    """
    Model-level halting estimator using entropy AND correctness prediction.
    
    Key insight from TRM: Use Q-learning for halting!
    - q_halt: "Is the current answer correct?" (trained with BCE against actual correctness)
    - Halt when: q_halt > 0 (model believes answer is correct)
    
    This is smarter than entropy alone because:
    1. High confidence ≠ correct answer (entropy can be misleading)
    2. Direct supervision: we know if answer is right during training
    3. Learns to recognize what "wrong answers look like"
    
    Also uses entropy as auxiliary signal:
    - Low entropy = confident prediction (peaked distribution)
    - Information gain = entropy reduction between passes
    """
    
    def __init__(self, d_model: int, vocab_size: int, max_passes: int = 8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_passes = max_passes
        self.max_entropy = math.log(vocab_size)
        
        # Hidden state analyzer
        self.h_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 4),
        )
        
        # === Q-HALT HEAD (TRM-inspired) ===
        # Predicts: "Is the current answer correct?"
        # Output: logit (>0 means "yes, halt", <0 means "no, continue")
        # Trained with: BCE against actual correctness
        # 
        # Now also takes info_gain as input so it can learn:
        # - High info_gain + low confidence → keep going (making progress)
        # - Low info_gain + high confidence → halt (converged)
        # - Low info_gain + low confidence → halt (stuck, save compute)
        self.q_halt_head = nn.Sequential(
            nn.Linear(d_model // 4 + 3, d_model // 8),  # +3 for entropy, info_gain, budget
            nn.SiLU(),
            nn.Linear(d_model // 8, 1),
        )
        
        # Legacy: Learned threshold (for backward compatibility / ablation)
        # Can be used instead of q_halt if use_q_halt=False
        self.halt_threshold = nn.Sequential(
            nn.Linear(d_model // 4 + 3, d_model // 8),
            nn.SiLU(),
            nn.Linear(d_model // 8, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize q_halt to output ~0 (uncertain) initially
        # This encourages exploration early in training
        with torch.no_grad():
            for layer in self.q_halt_head:
                if isinstance(layer, nn.Linear) and layer.out_features == 1:
                    layer.bias.fill_(-5.0)  # sigmoid(-5) ≈ 0.007, very unlikely to halt early
        
        # Initialize threshold HIGH to encourage exploration
        with torch.no_grad():
            for layer in self.halt_threshold:
                if isinstance(layer, nn.Linear) and layer.out_features == 1:
                    layer.bias.fill_(1.7)  # sigmoid(1.7) ≈ 0.85
    
    @staticmethod
    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy: H(p) = -Σ p_i log(p_i)"""
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, S]
        return entropy
    
    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy-based confidence: 1 - normalized_entropy"""
        entropy = self.compute_entropy(logits).mean(dim=1)  # [B]
        normalized_entropy = entropy / self.max_entropy
        confidence = 1.0 - normalized_entropy.clamp(0, 1)
        return confidence  # [B]
    
    def compute_information_gain(
        self,
        logits: torch.Tensor,                 # [B, S, V] current
        prev_logits: Optional[torch.Tensor],  # [B, S, V] previous
    ) -> torch.Tensor:
        """
        Compute information gain (entropy reduction).
        
        IG = H(prev) - H(curr)
        Positive = model became more confident
        """
        if prev_logits is None:
            return torch.zeros(logits.shape[0], device=logits.device)
        
        curr_entropy = self.compute_entropy(logits).mean(dim=1)
        prev_entropy = self.compute_entropy(prev_logits).mean(dim=1)
        return prev_entropy - curr_entropy  # [B]
    
    def forward(
        self,
        h: torch.Tensor,                        # [B, S, D]
        logits: torch.Tensor,                   # [B, S, V]
        prev_logits: Optional[torch.Tensor] = None,
        current_pass: int = 0,
        max_pass: Optional[int] = None,
        use_q_halt: bool = True,               # Use Q-halt (TRM) vs entropy threshold
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute confidence, halt decision, and auxiliary info.
        
        Returns:
            confidence: [B] entropy-based confidence (for logging/legacy)
            q_halt_logits: [B] halt logits (>0 = halt, <0 = continue)
            aux: dict with entropy, info_gain, threshold for logging
        """
        max_pass = max_pass or self.max_passes
        B = h.shape[0]
        device = h.device
        
        # Entropy-based confidence
        confidence = self.compute_confidence(logits)  # [B]
        
        # Information gain
        info_gain = self.compute_information_gain(logits, prev_logits)  # [B]
        
        # Hidden state features
        h_pooled = h.mean(dim=1)  # [B, D]
        h_features = self.h_analyzer(h_pooled)  # [B, D//4]
        
        # Budget feature
        budget_remaining = (max_pass - 1 - current_pass) / max(max_pass - 1, 1)
        budget_tensor = torch.full((B, 1), budget_remaining, device=device, dtype=h.dtype)
        
        # Entropy feature (normalized)
        entropy = self.compute_entropy(logits).mean(dim=1)  # [B]
        entropy_normalized = (entropy / self.max_entropy).unsqueeze(-1)  # [B, 1]
        
        # Info gain feature (normalized) - helps Q-halt decide if we're making progress
        info_gain_normalized = (info_gain / self.max_entropy).unsqueeze(-1).clamp(-1, 1)  # [B, 1]
        
        # Q-halt: "Is the answer correct?" 
        # Now includes info_gain so it can learn to consider progress
        q_features = torch.cat([h_features, entropy_normalized, info_gain_normalized, budget_tensor], dim=-1)
        q_halt_logits = self.q_halt_head(q_features).squeeze(-1)  # [B]
        
        # Legacy threshold (for ablation / backward compat)
        confidence_tensor = confidence.unsqueeze(-1)
        info_gain_tensor = info_gain.unsqueeze(-1)
        threshold_features = torch.cat([h_features, confidence_tensor, info_gain_tensor, budget_tensor], dim=-1)
        threshold = self.halt_threshold(threshold_features).squeeze(-1)  # [B]
        
        # Auxiliary info for logging and loss computation
        # Keep info_gain with gradients for maximization loss!
        aux = {
            'entropy': entropy.detach(),        # [B] for logging
            'info_gain': info_gain,             # [B] WITH gradients for loss!
            'threshold': threshold.detach(),    # [B] legacy threshold
            'q_halt_logits': q_halt_logits.detach(),  # [B] for logging
        }
        
        return confidence, q_halt_logits, aux

