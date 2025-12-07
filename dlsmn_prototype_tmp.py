"""
DLSMN Prototype: Dual-Head Layered Selective Memory Network
============================================================
A minimal implementation to validate the architecture on associative recall.

Task: Given pairs like "A→7, B→3, C→9", answer queries like "A?" → "7"
This tests:
  - Selective caching (store mappings, ignore noise)
  - Cross-layer reasoning (lower layers encode keys, higher layers decode)
  - Fixed memory (cache size independent of sequence length)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# ============================================================================
# Training Configuration (Section 8.4, 9.5)
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration following DLSM_V0.1.md recommendations."""
    # Temperature annealing (Section 8.2, 8.3)
    tau_start: float = 1.0       # Initial temperature (soft routing)
    tau_min: float = 0.1         # Final temperature (near-hard routing)
    anneal_rate: float = 0.0003  # Exponential decay rate

    # Training phases (Section 8.4)
    warmup_steps: int = 5000
    transition_steps: int = 20000  # End of transition phase

    # Auxiliary loss weights (Section 9.5)
    lambda_predict: float = 0.1
    lambda_consistency: float = 0.05
    lambda_diversity: float = 0.01
    lambda_sparsity: float = 0.01
    lambda_balance: float = 0.01

    # Importance threshold
    write_threshold: float = 0.5

    # Hybrid routing (Section 7.8)
    alpha_learned: float = 0.5  # Balance between learned and content-based routing

    # ACT (Section 10.1)
    max_passes: int = 3
    lambda_ponder: float = 0.01

    def get_temperature(self, step: int) -> float:
        """Gumbel-Softmax temperature schedule (Section 8.2)."""
        return max(self.tau_min, self.tau_start * math.exp(-self.anneal_rate * step))

    def get_phase(self, step: int) -> str:
        """Get current training phase (Section 8.4)."""
        if step < self.warmup_steps:
            return "warmup"
        elif step < self.transition_steps:
            return "transition"
        else:
            return "hard"


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Gumbel-Softmax with optional hard mode (Section 8.2).

    During training with high temperature: soft routing (all slots receive updates)
    As temperature → 0: converges to hard one-hot selection
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-through: hard forward, soft backward
        idx = y_soft.argmax(dim=-1)
        y_hard = F.one_hot(idx, logits.shape[-1]).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class SelectionHead(nn.Module):
    """
    Head B: The Gatekeeper (Section 2.2, 7.7, 7.8)

    Decides what to cache and where using:
    - Importance score (gated sigmoid)
    - Hybrid slot routing (learned + content-based similarity to slot anchors)
    - Gumbel-Softmax for differentiable slot selection
    """

    def __init__(self, d_model: int, d_cache: int, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        self.d_cache = d_cache

        # Importance gate (Section 2.2)
        self.gate = nn.Linear(d_model, 1)

        # Learned slot routing (Section 7.2)
        self.slot_selector = nn.Linear(d_model, num_slots)

        # Content-based routing queries (Section 7.8)
        self.slot_query = nn.Linear(d_model, d_cache)
        self.slot_key = nn.Linear(d_cache, d_cache)

        # Dynamic alpha for hybrid routing (Section 7.8)
        self.alpha_net = nn.Linear(d_model, 1)

    def forward(
        self,
        y: torch.Tensor,                    # [B, S, D] - patterns to potentially cache
        slot_embeddings: torch.Tensor,      # [K, D_cache] - learned slot anchors
        temperature: float = 1.0,
        hard: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            scores: [B, S] importance scores
            slot_probs: [B, S, K] slot assignment probabilities
            soft_probs: [B, S, K] soft probabilities for entropy computation
            alpha: [B, S] routing balance per token
        """
        B, S, D = y.shape
        K = self.num_slots

        # Importance score: σ(W_gate · y) ∈ [0, 1]
        scores = torch.sigmoid(self.gate(y)).squeeze(-1)  # [B, S]

        # Learned routing logits: W_slot · y
        learned_logits = self.slot_selector(y)  # [B, S, K]

        # Content-based similarity to slot anchors (Section 7.7, 7.8)
        # slot_sim_i = (W_q · y)^T · S[i]
        query = self.slot_query(y)  # [B, S, D_cache]
        keys = self.slot_key(slot_embeddings)  # [K, D_cache]
        content_logits = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.d_cache)  # [B, S, K]

        # Dynamic alpha: learned per-token balance (Section 7.8)
        alpha = torch.sigmoid(self.alpha_net(y)).squeeze(-1)  # [B, S]

        # Hybrid routing: α · learned + (1-α) · content-based
        alpha_expanded = alpha.unsqueeze(-1)  # [B, S, 1]
        combined_logits = alpha_expanded * learned_logits + (1 - alpha_expanded) * content_logits

        # Gumbel-Softmax for slot selection (Section 8.2)
        slot_probs = gumbel_softmax(combined_logits, temperature, hard=hard)  # [B, S, K]

        # Soft probs for entropy/diversity computation
        soft_probs = F.softmax(combined_logits, dim=-1)

        return {
            'scores': scores,
            'slot_probs': slot_probs,
            'soft_probs': soft_probs,
            'alpha': alpha,
        }


class CacheSelfAttention(nn.Module):
    """
    Cache-to-Cache Attention (Section 10.2)

    Allows memory-only computation between passes:
    C^{p+0.5} = C^{p} + SelfAttn(C^{p})

    Enables symbolic reasoning, graph propagation, and iterative
    refinement without expensive full forward passes.
    """

    def __init__(self, d_cache: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_cache, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_cache)

    def forward(self, cache: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cache: [B, L*K, D_cache] - full global cache
        Returns:
            Updated cache with inter-slot reasoning
        """
        attn_out, _ = self.attn(cache, cache, cache)
        return self.norm(cache + attn_out)


class DLSMNLayer(nn.Module):
    """
    DLSMN Layer following DLSM_V0.1.md specification.

    Implements the Compute-Select-Cache cycle (Section 3.1):
    1. READ: Query cache for relevant context
    2. COMPUTE: Head A processes input (± context)
    3. SELECT: Head B decides what to cache and where
    """

    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        d_cache: int,
        num_slots: int,
        num_layers: int,
        num_patterns: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.num_patterns = num_patterns

        # Layer-ID embedding (Section 7.6)
        self.layer_embed = nn.Parameter(torch.randn(1, 1, d_cache // 4) * 0.02)
        self.d_layer = d_cache // 4

        # Head A: Computation (Section 2.2)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Pattern pooling: learnable queries that extract pattern summaries
        self.pattern_queries = nn.Parameter(torch.randn(num_patterns, d_model) * 0.02)
        self.pattern_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Head B: Selection (Section 2.2)
        self.selection_head = SelectionHead(d_model, d_cache, num_slots)

        # Cache projections (Section 2.1)
        # W_compress: D_model → D_cache (per-layer)
        self.W_compress = nn.Linear(d_model, d_cache)
        # W_decompress: D_cache → D_model (per-layer)
        self.W_decompress = nn.Linear(d_cache + self.d_layer, d_model)  # +layer_id

        # Cache read attention (Section 3.1 Step 1)
        self.cache_query = nn.Linear(d_model, d_cache)
        self.cache_key = nn.Linear(d_cache + self.d_layer, d_cache)  # +layer_id
        self.cache_value = nn.Linear(d_cache + self.d_layer, d_cache)  # +layer_id

        # Cache-to-cache attention for memory-only reasoning
        self.cache_self_attn = CacheSelfAttention(d_cache, num_heads, dropout)

        # Gated fusion (Section 3.2 - recommended)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)

    def read_cache(
        self,
        x: torch.Tensor,           # [B, S, D_model]
        cache: torch.Tensor,       # [B, L*K, D_cache]
        layer_ids: torch.Tensor,   # [L*K, D_layer] - layer embeddings for each slot
        cache_mask: Optional[torch.Tensor] = None  # [B, L*K] True=blocked
    ) -> torch.Tensor:
        """
        Step 1: READ - Retrieve context from cache (Section 3.1)

        q = W_Q · x
        α = Softmax(q · (W_K · [C; layer_id])^T / √d + M)
        raw_context = α · (W_V · [C; layer_id])
        context = W_decompress · raw_context
        """
        B, S, _ = x.shape
        _, total_slots, _ = cache.shape

        # Concatenate layer-ID embeddings to cache entries (Section 7.6)
        cache_with_id = torch.cat([cache, layer_ids.unsqueeze(0).expand(B, -1, -1)], dim=-1)

        q = self.cache_query(x)  # [B, S, D_cache]
        k = self.cache_key(cache_with_id)  # [B, L*K, D_cache]
        v = self.cache_value(cache_with_id)  # [B, L*K, D_cache]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_cache)

        # Apply pass-aware mask (Section 2.3)
        if cache_mask is not None:
            mask = cache_mask.unsqueeze(1).expand(-1, S, -1)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        raw_context = torch.matmul(attn, v)  # [B, S, D_cache]

        # Decompress with layer info
        raw_context_with_id = torch.cat([
            raw_context,
            self.layer_embed.expand(B, S, -1)
        ], dim=-1)
        context = self.W_decompress(raw_context_with_id)  # [B, S, D_model]

        return context

    def fuse(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Gated fusion (Section 3.2 - recommended strategy):
        g = σ(W[x, context])
        output = g·x + (1-g)·context
        """
        combined = torch.cat([x, context], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        return gate * x + (1 - gate) * context

    def forward(
        self,
        x: torch.Tensor,                           # [B, S, D_model]
        cache: torch.Tensor,                       # [B, L*K, D_cache]
        slot_embeddings: torch.Tensor,             # [K, D_cache] - this layer's slot anchors
        layer_ids: torch.Tensor,                   # [L*K, D_layer] - all layer embeddings
        cache_mask: Optional[torch.Tensor] = None, # [B, L*K]
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full Compute-Select-Cache cycle (Section 3.1).
        """
        B, S, _ = x.shape

        # Step 1: READ from cache
        context = self.read_cache(x, cache, layer_ids, cache_mask)
        x_fused = self.fuse(x, context)

        # Step 2: COMPUTE (Head A)
        attn_out, _ = self.self_attn(x_fused, x_fused, x_fused)
        x_fused = self.norm1(x_fused + attn_out)
        ffn_out = self.ffn(x_fused)
        y = self.norm2(x_fused + ffn_out)

        # Pattern pooling: extract pattern-level summaries
        queries = self.pattern_queries.unsqueeze(0).expand(B, -1, -1)
        patterns, _ = self.pattern_attn(queries, y, y)  # [B, num_patterns, D_model]

        # Step 3: SELECT (Head B)
        selection = self.selection_head(patterns, slot_embeddings, temperature, hard)

        # Compress patterns to cache space
        patterns_cache = self.W_compress(patterns)  # [B, num_patterns, D_cache]

        # Compute entropy for diversity loss (Section 9.3)
        soft_probs = selection['soft_probs']
        entropy = -(soft_probs * torch.log(soft_probs + 1e-8)).sum(dim=-1).mean()

        # Compute slot usage distribution for balance loss (Section 8.5)
        slot_counts = selection['slot_probs'].sum(dim=1)  # [B, K]

        cache_updates = {
            'y_cache': patterns_cache,
            'scores': selection['scores'],
            'slot_probs': selection['slot_probs'],
            'soft_probs': soft_probs,
            'slot_counts': slot_counts,
            'entropy': entropy,
            'alpha': selection['alpha'],
            'patterns': patterns,  # For consistency loss
        }

        return y, cache_updates


class DLSMN(nn.Module):
    """Full DLSMN model with improved auxiliary losses and ACT."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        d_cache: int = 64,
        num_layers: int = 3,
        num_slots: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.total_slots = num_layers * num_slots

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # DLSMN layers with layer_idx and num_layers arguments
        self.layers = nn.ModuleList([
            DLSMNLayer(i, d_model, d_cache, num_slots, num_layers, num_heads=num_heads, dropout=dropout)
            for i in range(num_layers)
        ])

        # Output head
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Learned cache initialization
        self.cache_init = nn.Parameter(torch.randn(1, self.total_slots, d_cache) * 0.02)

        # Learned slot embeddings for each layer (Section 7.7)
        self.slot_embeddings = nn.Parameter(torch.randn(num_layers, num_slots, d_cache) * 0.02)

        # Layer-ID embeddings for each slot (Section 7.6)
        self.layer_id_embeddings = nn.Parameter(torch.randn(self.total_slots, d_cache // 4) * 0.02)

        # Halting unit for ACT (Section 10.1)
        self.halt_predictor = nn.Linear(d_model, 1)

    def get_cache_mask(self, batch_size: int, layer_idx: int, pass_num: int = 1):
        """
        Create pass-aware cache mask.

        Pass 1: layer j can only read from layers 0 to j-1
        Pass 2+: full access
        """
        mask = torch.zeros(batch_size, self.total_slots, dtype=torch.bool)

        if pass_num == 1:
            # Mask out current and higher layers
            accessible_slots = layer_idx * self.num_slots
            mask[:, accessible_slots:] = True
        # else: all False = all accessible

        return mask

    def apply_cache_updates(self, cache: torch.Tensor, updates: dict):
        """Apply aggregated writes to cache using soft addressing for gradients."""
        if updates is None:
            return cache

        B = cache.shape[0]
        y_cache = updates['y_cache']  # [B, S, d_cache]
        scores = updates['scores']  # [B, S]
        slot_probs = updates['slot_probs']  # [B, S, K]
        threshold = updates['threshold']
        start_idx = updates['start_idx']
        K = slot_probs.shape[-1]

        # Mask by threshold
        write_mask = (scores > threshold).float().unsqueeze(-1)  # [B, S, 1]

        # Weighted contribution: score * slot_prob * y_cache
        # This gives gradient flow through both score and slot selection
        weighted_y = write_mask * scores.unsqueeze(-1) * y_cache  # [B, S, d_cache]

        # Distribute to slots via soft addressing
        # slot_probs: [B, S, K], weighted_y: [B, S, d_cache]
        # Result: [B, K, d_cache]
        slot_writes = torch.einsum('bsk,bsd->bkd', slot_probs * write_mask, weighted_y)
        slot_weights = torch.einsum('bsk,bs->bk', slot_probs, write_mask.squeeze(-1) * scores)

        # Normalize
        slot_weights = slot_weights.unsqueeze(-1).clamp(min=1e-8)
        slot_writes = slot_writes / slot_weights

        # Write to appropriate slice
        new_cache = cache.clone()
        end_idx = start_idx + K

        # Blend with existing (hard overwrite for now)
        has_writes = (slot_weights.squeeze(-1) > 1e-6).unsqueeze(-1).float()
        new_cache[:, start_idx:end_idx] = (
            has_writes * slot_writes + (1 - has_writes) * cache[:, start_idx:end_idx]
        )

        return new_cache

    def forward(self, x: torch.Tensor, max_steps: int = None, temperature: float = 1.0, halt_threshold: float = 0.99):
        """
        Forward pass with ACT (Adaptive Computation Time).

        Args:
            x: [batch, seq_len] - token indices
            max_steps: maximum computation steps (if None, use self.num_layers * 2)
            temperature: Gumbel-Softmax temperature
            halt_threshold: cumulative probability threshold for halting
        """
        B, S = x.shape
        device = x.device

        # Set default max steps if not provided
        if max_steps is None:
            max_steps = self.num_layers * 2

        # Initialize
        h = self.embedding(x) + self.pos_encoding[:, :S, :]
        cache = self.cache_init.expand(B, -1, -1).clone()

        # For ACT: we'll run through steps, not layers
        remaining_steps = torch.ones(B, S, device=device, dtype=torch.bool)  # [B, S] - which positions still need computation
        cumulative_halt = torch.zeros(B, S, device=device)  # [B, S] - cumulative halt probability
        total_ponder_loss = torch.tensor(0.0, device=device)  # Total ponder loss for regularization

        # Gather auxiliary losses
        all_entropies = []
        all_scores = []
        all_slot_counts = []

        step = 0
        while step < max_steps and remaining_steps.any():
            # Select a random layer for this step
            layer_idx = step % self.num_layers
            layer = self.layers[layer_idx]

            # Compute halting probability for each position in the sequence
            halt_probs = torch.sigmoid(self.halt_predictor(h)).squeeze(-1)  # [B, S]

            # Update cumulative halt probability only for remaining positions
            new_cumulative = cumulative_halt + halt_probs * remaining_steps.float()
            cumulative_halt = torch.min(new_cumulative, torch.ones_like(new_cumulative))

            # Check which positions should halt now
            should_halt = (cumulative_halt >= halt_threshold) & remaining_steps
            remaining_steps = remaining_steps & ~should_halt  # Update remaining steps

            # Compute ponder cost for regularization (how much each position computed)
            step_ponders = halt_probs * remaining_steps.float()
            total_ponder_loss += step_ponders.mean()

            # Only process positions that are still active
            cache_mask = self.get_cache_mask(B, layer_idx, 1).to(device)  # Use pass 1 style masking

            # Get this layer's slot embeddings and layer-id embeddings
            layer_slot_embeddings = self.slot_embeddings[layer_idx]  # [num_slots, d_cache]
            layer_start_idx = layer_idx * self.num_slots
            layer_end_idx = layer_start_idx + self.num_slots
            layer_ids = self.layer_id_embeddings[layer_start_idx:layer_end_idx]  # [num_slots, d_layer]

            h, updates = layer(
                h, cache, layer_slot_embeddings, layer_ids,
                cache_mask=cache_mask,
                temperature=temperature,
                hard=False  # Don't use hard selection during training
            )

            if updates is not None:
                # Collect auxiliary losses
                all_entropies.append(updates['entropy'])
                all_scores.append(updates['scores'])
                all_slot_counts.append(updates['slot_counts'])

                # Apply updates
                cache_updates = {
                    'y_cache': updates['y_cache'],
                    'scores': updates['scores'],
                    'slot_probs': updates['slot_probs'],
                    'start_idx': layer_start_idx,
                    'threshold': 0.5,  # Use default threshold
                }
                cache = self.apply_cache_updates(cache, cache_updates)

            step += 1

        logits = self.output_proj(h)

        # Compute auxiliary losses (Section 9.5)
        aux_loss = 0.0

        # Sparsity loss: encourage low scores (less writing)
        if all_scores:
            all_scores_flat = torch.cat([s.flatten() for s in all_scores])
            sparsity_loss = all_scores_flat.mean()
            aux_loss += 0.01 * sparsity_loss

        # Diversity loss: maximize entropy of slot assignments
        if all_entropies:
            entropy_loss = torch.stack(all_entropies).mean()
            aux_loss -= 0.01 * entropy_loss  # Negative because we want to maximize entropy

        # Balance loss: encourage uniform slot usage
        if all_slot_counts:
            all_slot_counts_tensor = torch.cat(all_slot_counts, dim=0)  # [B*K]
            balance_loss = torch.var(all_slot_counts_tensor.float(), dim=-1)
            aux_loss += 0.01 * balance_loss

        # ACT Ponder loss: encourage efficient computation
        aux_loss += 0.01 * total_ponder_loss

        return logits, aux_loss, cache, step  # Return step count to see how many steps were taken


# ============================================================================
# Associative Recall Dataset
# ============================================================================

class AssociativeRecallDataset(Dataset):
    """
    Task: Given "A→7, B→3, C→9, query:A" → answer "7"

    Format: [key1, val1, key2, val2, ..., SEP, query_key, MASK]
    Target: value corresponding to query_key

    This tests:
    - Selective memory: cache the key-value pairs, ignore structure tokens
    - Retrieval: find the right value for the query
    """

    def __init__(
        self,
        num_samples: int = 10000,
        num_pairs: int = 5,
        num_keys: int = 26,  # A-Z
        num_values: int = 10,  # 0-9
        num_distractors: int = 3,  # noise tokens between pairs
    ):
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_distractors = num_distractors

        # Vocabulary:
        # 0: PAD, 1: SEP, 2: MASK, 3: ARROW
        # 4-29: keys (A-Z)
        # 30-39: values (0-9)
        # 40-49: distractor tokens
        self.PAD = 0
        self.SEP = 1
        self.MASK = 2
        self.ARROW = 3
        self.KEY_START = 4
        self.VAL_START = 4 + num_keys
        self.DIST_START = self.VAL_START + num_values
        self.vocab_size = self.DIST_START + 10

        self.data = [self._generate_sample() for _ in range(num_samples)]

    def _generate_sample(self):
        # Generate random key-value pairs
        keys = random.sample(range(self.num_keys), self.num_pairs)
        values = [random.randint(0, self.num_values - 1) for _ in range(self.num_pairs)]

        # Build sequence: key1 → val1, distractor, key2 → val2, ...
        seq = []
        for i, (k, v) in enumerate(zip(keys, values)):
            seq.append(self.KEY_START + k)
            seq.append(self.ARROW)
            seq.append(self.VAL_START + v)

            # Add distractors (except after last pair)
            if i < self.num_pairs - 1:
                for _ in range(random.randint(0, self.num_distractors)):
                    seq.append(self.DIST_START + random.randint(0, 9))

        # Add separator and query
        seq.append(self.SEP)
        query_idx = random.randint(0, self.num_pairs - 1)
        seq.append(self.KEY_START + keys[query_idx])
        seq.append(self.MASK)

        target = self.VAL_START + values[query_idx]

        return torch.tensor(seq), target, len(seq) - 1  # position of MASK

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq, target, mask_pos = self.data[idx]
        return seq, target, mask_pos


def collate_fn(batch):
    """Pad sequences to same length."""
    seqs, targets, mask_positions = zip(*batch)
    max_len = max(len(s) for s in seqs)

    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s

    return padded, torch.tensor(targets), torch.tensor(mask_positions)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, num_steps=1, epoch=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:2d}", leave=False)
    for batch_idx, (seqs, targets, mask_pos) in enumerate(pbar):
        seqs = seqs.to(device)
        targets = targets.to(device)
        mask_pos = mask_pos.to(device)

        optimizer.zero_grad()

        logits, aux_loss, _, steps = model(seqs, max_steps=num_steps)

        # Get predictions at mask positions
        B = seqs.shape[0]
        mask_logits = logits[torch.arange(B), mask_pos]  # [B, vocab]

        loss = F.cross_entropy(mask_logits, targets) + aux_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.detach().item()
        preds = mask_logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += B

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.3f}',
            'acc': f'{correct / total:.3f}',
            'steps': f'{steps}'
        })

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device, num_steps=1, desc="Eval"):
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for seqs, targets, mask_pos in pbar:
        seqs = seqs.to(device)
        targets = targets.to(device)
        mask_pos = mask_pos.to(device)

        logits, _, _, steps = model(seqs, max_steps=num_steps)

        B = seqs.shape[0]
        mask_logits = logits[torch.arange(B), mask_pos]
        preds = mask_logits.argmax(dim=-1)

        correct += (preds == targets).sum().item()
        total += B

        pbar.set_postfix({'acc': f'{correct / total:.3f}', 'steps': f'{steps}'})

    return correct / total


def visualize_cache(model, sample_seq, device, dataset):
    """Visualize what gets cached for a sample sequence."""
    model.eval()

    seq = sample_seq.unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, cache = model(seq, num_passes=2)

    # Decode sequence for display
    tokens = []
    for t in sample_seq.tolist():
        if t == dataset.PAD:
            tokens.append('[PAD]')
        elif t == dataset.SEP:
            tokens.append('[SEP]')
        elif t == dataset.MASK:
            tokens.append('[MASK]')
        elif t == dataset.ARROW:
            tokens.append('→')
        elif t >= dataset.KEY_START and t < dataset.VAL_START:
            tokens.append(chr(ord('A') + t - dataset.KEY_START))
        elif t >= dataset.VAL_START and t < dataset.DIST_START:
            tokens.append(str(t - dataset.VAL_START))
        else:
            tokens.append(f'd{t - dataset.DIST_START}')

    print(f"\nSequence: {' '.join(tokens)}")
    print(f"\nCache shape: {cache.shape}")
    print(f"Cache L2 norms per slot:")

    norms = cache[0].norm(dim=-1)  # [total_slots]
    for layer in range(model.num_layers):
        start = layer * model.num_slots
        layer_norms = norms[start:start + model.num_slots]
        print(f"  Layer {layer}: {layer_norms.cpu().numpy().round(2)}")


# ============================================================================
# Main
# ============================================================================

def main():
    # Config
    device = torch.device('mps')#'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration
    config = TrainingConfig()

    # Dataset
    train_dataset = AssociativeRecallDataset(num_samples=10000, num_pairs=5)
    test_dataset = AssociativeRecallDataset(num_samples=1000, num_pairs=5)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    print(f"Vocab size: {train_dataset.vocab_size}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Model
    model = DLSMN(
        vocab_size=train_dataset.vocab_size,
        d_model=128,
        d_cache=64,
        num_layers=3,
        num_slots=8,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print("\n" + "="*60)
    print("Training DLSMN on Associative Recall")
    print("="*60 + "\n")

    best_acc = 0
    global_step = 0
    for epoch in range(50):
        # Use config for num_steps based on phase
        phase = config.get_phase(global_step)
        if phase == "warmup":
            num_steps = 1
        elif phase == "transition":
            num_steps = min(3, 1 + epoch // 10)  # Gradually increase
        else:  # hard
            num_steps = 3

        # Get current temperature
        current_temp = config.get_temperature(global_step)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, num_steps, epoch)
        test_acc = evaluate(model, test_loader, device, num_steps, desc="Test")
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            marker = " ★"
        else:
            marker = ""

        print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | Steps: {num_steps} | Temp: {current_temp:.3f}{marker}")
        global_step += len(train_loader)  # Update global step

    print(f"\nBest test accuracy: {best_acc:.3f}")

    # Visualize cache for a sample
    print("\n" + "="*60)
    print("Cache Visualization")
    print("="*60)
    sample_seq, target, mask_pos = test_dataset[0]
    visualize_cache(model, sample_seq, device, test_dataset)

    # Test with different number of steps
    print("\n" + "="*60)
    print("Accuracy vs Number of Steps")
    print("="*60)
    for steps in [1, 2, 3]:
        acc = evaluate(model, test_loader, device, num_steps=steps)
        print(f"  {steps} step(s): {acc:.3f}")


if __name__ == "__main__":
    main()
