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
    """

    def __init__(
        self,
        d_model: int,
        d_cache: int,
        num_slots: int,
        num_layers: int = 1,
        layer_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        self.total_slots = num_layers * num_slots

        # === Space Projections ===
        self.to_cache = nn.Linear(d_model, d_cache)
        self.from_cache = nn.Linear(d_cache, d_model)

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
        self.importance_scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Write gate: OUTPUT-ONLY decision on whether to write
        # Per spec: "σ(W_w · P) - Is this pattern worth saving?"
        # This is intrinsic value - "should I write this at all?"
        self.write_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Novelty gate: modulates write based on comparison to existing cache
        # This answers "is this new/different from what's already stored?"
        self.novelty_gate = nn.Sequential(
            nn.LayerNorm(d_cache * 2),
            nn.Linear(d_cache * 2, d_cache),
            nn.SiLU(),
            nn.Linear(d_cache, 1),
        )

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
        for seq in [self.read_gate, self.importance_scorer, self.write_gate]:
            for m in seq:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def _get_local_cache(self, cache: torch.Tensor) -> torch.Tensor:
        """Extract this layer's slots from global cache."""
        start = self.layer_idx * self.num_slots
        end = start + self.num_slots
        return cache[:, start:end, :]  # [B, K, D_cache]

    def _update_local_cache(
        self,
        cache: torch.Tensor,
        new_local: torch.Tensor
    ) -> torch.Tensor:
        """Update this layer's slots in global cache."""
        updated = cache.clone()
        start = self.layer_idx * self.num_slots
        end = start + self.num_slots
        updated[:, start:end, :] = new_local
        return updated

    def _attend_to_cache(
        self,
        query: torch.Tensor,   # [B, S, D_cache]
        cache: torch.Tensor,   # [B, K, D_cache]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention to cache slots.

        Returns:
            context: [B, S, D_cache] - retrieved information
            attn_weights: [B, S, K] - attention distribution
        """
        # Attention scores
        scores = torch.matmul(query, cache.transpose(-2, -1))  # [B, S, K]
        scores = scores / math.sqrt(self.d_cache)
        scores = scores.clamp(min=-20.0, max=20.0)  # Numerical stability

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Retrieve context
        context = torch.matmul(attn_weights, cache)  # [B, S, D_cache]

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

        if hard or not self.training:
            read_gate = (soft_gate > 0.5).float()
        else:
            # STE: hard forward, soft backward
            hard_gate = (soft_gate > 0.5).float()
            read_gate = hard_gate - soft_gate.detach() + soft_gate

        # === Step 2: Only attend if gate is open (sparse optimization possible) ===
        # Generate read query from input
        query = self.read_query(x)  # [B, S, D_cache]

        # Select cache to read from
        read_cache = cache if use_global else self._get_local_cache(cache)

        # Attend to cache
        context_cache, attn_weights = self._attend_to_cache(query, read_cache)

        # Project context to model space
        context = self.from_cache(context_cache)  # [B, S, D_model]

        # === Step 3: Gated fusion ===
        # Only incorporate context where read_gate is open
        x_enhanced = x + read_gate * context

        return {
            'x_enhanced': x_enhanced,
            'context': context,
            'read_gate': read_gate,
            'attn_weights': attn_weights,
        }

    def write(
        self,
        output: torch.Tensor,                  # [B, S, D_model]
        cache: torch.Tensor,                   # [B, L*K, D_cache]
        temperature: float = 1.0,
        hard: bool = False,
        max_write_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        WRITE Phase: Selectively update cache with output content.

        Process:
        1. Score token importance
        2. Gate which tokens should write
        3. Select target slots via attention
        4. Update cache with soft-WTA

        Args:
            output: Computation output [B, S, D_model]
            cache: Global cache [B, L*K, D_cache]
            temperature: Gumbel-softmax temperature
            hard: Use hard slot selection
            max_write_tokens: Limit tokens considered for writing

        Returns:
            Dict with:
                - 'updated_cache': New cache [B, L*K, D_cache]
                - 'write_gate': Which tokens wrote [B, T, 1]
                - 'importance': Token importance scores [B, T, 1]
                - 'slot_probs': Slot selection probs [B, T, K]
        """
        B, S, _ = output.shape
        K = self.num_slots

        # Limit tokens for efficiency
        T = min(S, max_write_tokens) if max_write_tokens else S
        tokens = output[:, :T, :]  # [B, T, D_model]

        # Get local cache
        local_cache = self._get_local_cache(cache)  # [B, K, D_cache]

        # Project tokens to cache space
        tokens_cache = self.to_cache(tokens)  # [B, T, D_cache]

        # === Step 1: Importance Scoring (intrinsic value of content) ===
        importance = self.importance_scorer(tokens)  # [B, T, 1]

        # === Step 2: Write Gate (OUTPUT-ONLY: should this token write at all?) ===
        # Per spec: "σ(W_w · P) - Is this pattern worth saving?"
        gate_logit = self.write_gate(tokens)  # [B, T, 1]
        soft_gate = torch.sigmoid(gate_logit)

        # === Step 3: Novelty Gate (compare to cache: is this new?) ===
        # This modulates the write gate based on how different the content is
        context_cache, _ = self._attend_to_cache(tokens_cache, local_cache)
        combined = torch.cat([tokens_cache, context_cache], dim=-1)  # [B, T, 2*D_cache]
        novelty_logit = self.novelty_gate(combined)  # [B, T, 1]
        novelty = torch.sigmoid(novelty_logit)

        # === Step 4: Combined write decision ===
        # Write if: (worth saving) AND (novel enough)
        # Both signals matter: don't write boring content, don't write duplicates
        combined_gate = soft_gate * novelty  # [B, T, 1]

        if hard or not self.training:
            write_gate = (combined_gate > 0.5).float()
        else:
            hard_gate = (combined_gate > 0.5).float()
            write_gate = hard_gate - combined_gate.detach() + combined_gate

        # === Step 5: Slot Selection ===
        # Query for which slots to write to
        write_query = self.write_query(tokens)  # [B, T, D_cache]
        slot_logits = torch.matmul(write_query, local_cache.transpose(-2, -1))
        slot_logits = slot_logits / math.sqrt(self.d_cache)
        slot_logits = slot_logits.clamp(min=-10.0, max=10.0)

        # Gumbel-softmax for differentiable selection
        if hard or not self.training:
            slot_probs = gumbel_softmax(slot_logits, temperature, hard=True)
        else:
            slot_probs = gumbel_softmax(slot_logits, temperature, hard=False)

        # === Step 6: Cache Update ===
        # Mask by write gate and importance
        masked_slot_probs = slot_probs * write_gate  # [B, T, K]
        importance_weights = importance.clamp(min=1e-6)  # [B, T, 1]
        weighted_tokens = tokens_cache * importance_weights  # [B, T, D_cache]

        # Aggregate to slots: [B, K, T] @ [B, T, D_cache] → [B, K, D_cache]
        slot_probs_t = masked_slot_probs.transpose(1, 2)  # [B, K, T]
        slot_writes = torch.matmul(slot_probs_t, weighted_tokens)  # [B, K, D_cache]

        # Normalize by total weight per slot
        slot_weights = torch.matmul(slot_probs_t, importance_weights.squeeze(-1).unsqueeze(-1))
        slot_weights_safe = slot_weights.clamp(min=0.1)
        slot_writes = slot_writes / slot_weights_safe

        # Determine which slots received writes
        has_writes = (slot_weights.squeeze(-1) > 0.1).unsqueeze(-1).float()

        # True eviction: replace written slots
        new_local_cache = has_writes * slot_writes + (1 - has_writes) * local_cache

        # Update global cache
        updated_cache = self._update_local_cache(cache, new_local_cache)

        return {
            'updated_cache': updated_cache,
            'write_gate': write_gate,
            'novelty': novelty,  # Added for debugging/analysis
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


class ConfidenceEstimator(nn.Module):
    """
    Estimates confidence/quality of model output for halting decisions.

    This complements the Q-halt mechanism by providing a direct
    assessment of output quality, not just hidden state pooling.

    Signals considered:
    1. Output entropy (low = confident)
    2. Hidden state coherence (how "settled" is the representation)
    3. Pass-to-pass agreement (optional, if prev_output provided)
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Hidden state analyzer
        self.h_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Combine signals
        self.confidence_combiner = nn.Sequential(
            nn.Linear(3, 16),  # 3 signals: h_score, entropy, agreement
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h: torch.Tensor,                        # [B, S, D_model]
        logits: Optional[torch.Tensor] = None,  # [B, S, V]
        prev_logits: Optional[torch.Tensor] = None,  # [B, S, V] from previous pass
    ) -> torch.Tensor:
        """
        Compute confidence score in [0, 1].

        Returns:
            confidence: [B] confidence score per batch item
        """
        B = h.shape[0]
        device = h.device

        # Signal 1: Hidden state quality
        h_pooled = h.mean(dim=1)  # [B, D_model]
        h_score = torch.sigmoid(self.h_analyzer(h_pooled))  # [B, 1]

        # Signal 2: Output entropy (if logits provided)
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B, S]
            # Normalize: max entropy = log(vocab_size)
            max_entropy = math.log(self.vocab_size)
            normalized_entropy = entropy.mean(dim=1, keepdim=True) / max_entropy  # [B, 1]
            # Low entropy = high confidence
            entropy_confidence = 1 - normalized_entropy
        else:
            entropy_confidence = torch.ones(B, 1, device=device) * 0.5

        # Signal 3: Agreement with previous pass (if provided)
        if prev_logits is not None and logits is not None:
            # KL divergence between current and previous
            curr_probs = F.softmax(logits, dim=-1)
            prev_probs = F.softmax(prev_logits, dim=-1)
            kl = (curr_probs * (torch.log(curr_probs + 1e-8) - torch.log(prev_probs + 1e-8))).sum(dim=-1)
            agreement = torch.exp(-kl.mean(dim=1, keepdim=True))  # [B, 1] - high when similar
        else:
            agreement = torch.ones(B, 1, device=device) * 0.5

        # Combine signals
        signals = torch.cat([h_score, entropy_confidence, agreement], dim=-1)  # [B, 3]
        confidence = self.confidence_combiner(signals).squeeze(-1)  # [B]

        return confidence
