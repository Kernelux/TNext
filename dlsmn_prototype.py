"""
DLSMN Prototype: Dual-Head Layered Selective Memory Network
============================================================
A minimal implementation to validate the architecture on associative recall.

Task: Given pairs like "A→7, B→3, C→9", answer queries like "A?" → "7"
This tests:
  - Selective caching (store mappings, ignore noise)
  - Cross-layer reasoning (lower layers encode keys, higher layers decode)
  - Fixed memory (cache size independent of sequence length)

Key features from DLSM_V0.1.md spec:
- Gumbel-Softmax annealing (soft→hard routing) [Section 8.2]
- Learned slot embeddings as concept anchors [Section 7.7]
- Hybrid slot routing (learned + content-based) [Section 7.8]
- Layer-ID embeddings for representational separation [Section 7.6]
- Full auxiliary losses: predictive, consistency, diversity, sparsity, balance [Section 9]
- Training curriculum: warm-up → transition → hard [Section 8.4]
- Cache-to-cache attention for memory-only reasoning [Section 10.2]
- Adaptive Computation Time (ACT) for efficient multi-pass [Section 10.1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# ============================================================================
# Feature Flags for Ablation Study
# ============================================================================

@dataclass
class FeatureFlags:
    """
    Feature flags for ablation studies.
    
    CORE FEATURES (essential to DLSMN architecture):
    - use_cache: Global hierarchical cache C ∈ ℝ^{(L×K) × D_cache} [Section 2.1]
    - use_selection_head: Head B for importance scoring and slot routing [Section 2.2]
    - use_multi_pass: Multiple forward passes with cache refinement [Section 2.3]
    
    IMPROVEMENTS (optional enhancements from spec):
    - use_gumbel_softmax: Differentiable slot selection with annealing [Section 8.2]
    - use_slot_embeddings: Learned concept anchors for cache init [Section 7.7]
    - use_hybrid_routing: Combined learned + content-based routing [Section 7.8]
    - use_layer_id: Layer-ID embeddings for representational separation [Section 7.6]
    - use_cache_self_attn: Cache-to-cache attention between passes [Section 10.2]
    - use_act_halting: Adaptive Computation Time for early stopping [Section 10.1]
    - use_gated_fusion: Gated combination of input and cache context [Section 3.2]
    
    AUXILIARY LOSSES (regularization techniques):
    - use_diversity_loss: Encourage uniform slot usage [Section 9.3]
    - use_balance_loss: Penalize uneven slot distribution [Section 8.5]
    - use_sparsity_loss: Encourage decisive routing [Section 9.4]
    - use_consistency_loss: Cross-layer cache decodability [Section 9.2]
    - use_ponder_loss: Penalize excessive computation [Section 10.1]
    """
    
    # === CORE FEATURES (disable to break DLSMN) ===
    use_cache: bool = True              # Without this, it's just a transformer
    use_selection_head: bool = True     # Without this, no selective caching
    use_multi_pass: bool = True         # Without this, single-pass only
    
    # === IMPROVEMENTS (disable for ablation) ===
    # Routing improvements
    use_gumbel_softmax: bool = True     # False = argmax routing (non-differentiable)
    use_slot_embeddings: bool = True    # False = zero-init cache
    use_hybrid_routing: bool = True     # False = learned routing only
    use_layer_id: bool = True           # False = no layer separation in cache
    
    # Architecture improvements
    use_cache_self_attn: bool = True    # False = no inter-pass cache reasoning
    use_act_halting: bool = True        # False = fixed number of passes
    use_gated_fusion: bool = True       # False = additive fusion
    
    # Pattern pooling
    use_pattern_pooling: bool = True    # False = direct token caching
    
    # === AUXILIARY LOSSES ===
    use_diversity_loss: bool = True
    use_balance_loss: bool = True
    use_sparsity_loss: bool = True
    use_consistency_loss: bool = True
    use_ponder_loss: bool = True
    
    def describe(self) -> str:
        """Return a string describing enabled features."""
        core = []
        if self.use_cache: core.append("cache")
        if self.use_selection_head: core.append("selection")
        if self.use_multi_pass: core.append("multi-pass")
        
        improvements = []
        if self.use_gumbel_softmax: improvements.append("gumbel")
        if self.use_slot_embeddings: improvements.append("slot-emb")
        if self.use_hybrid_routing: improvements.append("hybrid")
        if self.use_layer_id: improvements.append("layer-id")
        if self.use_cache_self_attn: improvements.append("cache-attn")
        if self.use_act_halting: improvements.append("ACT")
        if self.use_gated_fusion: improvements.append("gated")
        if self.use_pattern_pooling: improvements.append("patterns")
        
        losses = []
        if self.use_diversity_loss: losses.append("div")
        if self.use_balance_loss: losses.append("bal")
        if self.use_sparsity_loss: losses.append("sparse")
        if self.use_consistency_loss: losses.append("consist")
        if self.use_ponder_loss: losses.append("ponder")
        
        return (f"Core: [{', '.join(core)}] | "
                f"Improvements: [{', '.join(improvements)}] | "
                f"Losses: [{', '.join(losses)}]")


# Preset configurations for common ablation experiments
FEATURE_PRESETS = {
    "full": FeatureFlags(),
    
    "core_only": FeatureFlags(
        use_gumbel_softmax=False,
        use_slot_embeddings=False,
        use_hybrid_routing=False,
        use_layer_id=False,
        use_cache_self_attn=False,
        use_act_halting=False,
        use_gated_fusion=False,
        use_pattern_pooling=False,
        use_diversity_loss=False,
        use_balance_loss=False,
        use_sparsity_loss=False,
        use_consistency_loss=False,
        use_ponder_loss=False,
    ),
    
    "single_pass": FeatureFlags(
        use_multi_pass=False,
        use_cache_self_attn=False,
        use_act_halting=False,
        use_ponder_loss=False,
    ),
    
    "no_gumbel": FeatureFlags(use_gumbel_softmax=False),
    "no_slot_emb": FeatureFlags(use_slot_embeddings=False),
    "no_hybrid": FeatureFlags(use_hybrid_routing=False),
    "no_cache_attn": FeatureFlags(use_cache_self_attn=False),
    
    "no_aux_loss": FeatureFlags(
        use_diversity_loss=False,
        use_balance_loss=False,
        use_sparsity_loss=False,
        use_consistency_loss=False,
        use_ponder_loss=False,
    ),
    
    "fast": FeatureFlags(
        use_multi_pass=False,
        use_cache_self_attn=False,
        use_act_halting=False,
        use_consistency_loss=False,
        use_ponder_loss=False,
        use_pattern_pooling=False,
    ),
}


# ============================================================================
# Training Configuration (Section 8.4, 9.5)
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration following DLSM_V0.1.md recommendations."""
    tau_start: float = 1.0
    tau_min: float = 0.1
    anneal_rate: float = 0.0003
    warmup_steps: int = 5000
    transition_steps: int = 20000
    lambda_predict: float = 0.1
    lambda_consistency: float = 0.05
    lambda_diversity: float = 0.01
    lambda_sparsity: float = 0.01
    lambda_balance: float = 0.01
    write_threshold: float = 0.5
    alpha_learned: float = 0.5
    max_passes: int = 3
    lambda_ponder: float = 0.01
    features: FeatureFlags = field(default_factory=FeatureFlags)

    def get_temperature(self, step: int) -> float:
        return max(self.tau_min, self.tau_start * math.exp(-self.anneal_rate * step))

    def get_phase(self, step: int) -> str:
        if step < self.warmup_steps:
            return "warmup"
        elif step < self.transition_steps:
            return "transition"
        else:
            return "hard"


# ============================================================================
# DLSMN Components
# ============================================================================

def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """Gumbel-Softmax with optional hard mode (Section 8.2)."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    if hard:
        idx = y_soft.argmax(dim=-1)
        y_hard = F.one_hot(idx, logits.shape[-1]).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class SelectionHead(nn.Module):
    """Head B: The Gatekeeper (Section 2.2, 7.7, 7.8)"""

    def __init__(self, d_model: int, d_cache: int, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        self.d_cache = d_cache
        self.gate = nn.Linear(d_model, 1)
        self.slot_selector = nn.Linear(d_model, num_slots)
        self.slot_query = nn.Linear(d_model, d_cache)
        self.slot_key = nn.Linear(d_cache, d_cache)
        self.alpha_net = nn.Linear(d_model, 1)

    def forward(
        self,
        y: torch.Tensor,
        slot_embeddings: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
        features: Optional[FeatureFlags] = None,
    ) -> Dict[str, torch.Tensor]:
        if features is None:
            features = FeatureFlags()
            
        B, S, D = y.shape
        K = self.num_slots

        scores = torch.sigmoid(self.gate(y)).squeeze(-1)
        learned_logits = self.slot_selector(y)

        if features.use_hybrid_routing:
            query = self.slot_query(y)
            keys = self.slot_key(slot_embeddings)
            content_logits = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.d_cache)
            alpha = torch.sigmoid(self.alpha_net(y)).squeeze(-1)
            alpha_expanded = alpha.unsqueeze(-1)
            combined_logits = alpha_expanded * learned_logits + (1 - alpha_expanded) * content_logits
        else:
            combined_logits = learned_logits
            alpha = torch.ones(B, S, device=y.device)

        if features.use_gumbel_softmax:
            slot_probs = gumbel_softmax(combined_logits, temperature, hard=hard)
        else:
            idx = combined_logits.argmax(dim=-1)
            slot_probs = F.one_hot(idx, K).float()

        soft_probs = F.softmax(combined_logits, dim=-1)

        return {
            'scores': scores,
            'slot_probs': slot_probs,
            'soft_probs': soft_probs,
            'alpha': alpha,
        }


class CacheSelfAttention(nn.Module):
    """Cache-to-Cache Attention (Section 10.2)"""

    def __init__(self, d_cache: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_cache, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_cache)

    def forward(self, cache: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(cache, cache, cache)
        return self.norm(cache + attn_out)


class DLSMNLayer(nn.Module):
    """DLSMN Layer following DLSM_V0.1.md specification."""

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
        self.d_layer = d_cache // 4

        self.layer_embed = nn.Parameter(torch.randn(1, 1, self.d_layer) * 0.02)

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.pattern_queries = nn.Parameter(torch.randn(num_patterns, d_model) * 0.02)
        self.pattern_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.selection_head = SelectionHead(d_model, d_cache, num_slots)
        self.W_compress = nn.Linear(d_model, d_cache)
        self.W_decompress = nn.Linear(d_cache + self.d_layer, d_model)
        self.cache_query = nn.Linear(d_model, d_cache)
        self.cache_key = nn.Linear(d_cache + self.d_layer, d_cache)
        self.cache_value = nn.Linear(d_cache + self.d_layer, d_cache)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)

    def read_cache(self, x, cache, layer_ids, cache_mask=None):
        B, S, _ = x.shape
        cache_with_id = torch.cat([cache, layer_ids.unsqueeze(0).expand(B, -1, -1)], dim=-1)

        q = self.cache_query(x)
        k = self.cache_key(cache_with_id)
        v = self.cache_value(cache_with_id)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_cache)
        if cache_mask is not None:
            mask = cache_mask.unsqueeze(1).expand(-1, S, -1)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        raw_context = torch.matmul(attn, v)

        raw_context_with_id = torch.cat([raw_context, self.layer_embed.expand(B, S, -1)], dim=-1)
        return self.W_decompress(raw_context_with_id)

    def fuse(self, x, context):
        combined = torch.cat([x, context], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        return gate * x + (1 - gate) * context

    def forward(self, x, cache, slot_embeddings, layer_ids, cache_mask=None,
                temperature=1.0, hard=False, features=None):
        if features is None:
            features = FeatureFlags()
            
        B, S, _ = x.shape

        if features.use_cache:
            context = self.read_cache(x, cache, layer_ids, cache_mask)
            x_fused = self.fuse(x, context) if features.use_gated_fusion else x + context
        else:
            x_fused = x

        attn_out, _ = self.self_attn(x_fused, x_fused, x_fused)
        x_fused = self.norm1(x_fused + attn_out)
        ffn_out = self.ffn(x_fused)
        y = self.norm2(x_fused + ffn_out)

        if features.use_pattern_pooling:
            queries = self.pattern_queries.unsqueeze(0).expand(B, -1, -1)
            patterns, _ = self.pattern_attn(queries, y, y)
        else:
            if S <= self.num_patterns:
                patterns = F.pad(y, (0, 0, 0, self.num_patterns - S))
            else:
                patterns = y[:, :self.num_patterns, :]

        if features.use_selection_head:
            selection = self.selection_head(patterns, slot_embeddings, temperature, hard, features)
        else:
            num_p = patterns.shape[1]
            K = self.num_slots
            selection = {
                'scores': torch.ones(B, num_p, device=x.device),
                'slot_probs': torch.ones(B, num_p, K, device=x.device) / K,
                'soft_probs': torch.ones(B, num_p, K, device=x.device) / K,
                'alpha': torch.ones(B, num_p, device=x.device),
            }

        patterns_cache = self.W_compress(patterns)
        soft_probs = selection['soft_probs']
        entropy = -(soft_probs * torch.log(soft_probs + 1e-8)).sum(dim=-1).mean()
        slot_counts = selection['slot_probs'].sum(dim=1)

        return y, {
            'y_cache': patterns_cache,
            'scores': selection['scores'],
            'slot_probs': selection['slot_probs'],
            'soft_probs': soft_probs,
            'slot_counts': slot_counts,
            'entropy': entropy,
            'alpha': selection['alpha'],
            'patterns': patterns,
        }


class DLSMN(nn.Module):
    """Full DLSMN model for sequence tasks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        d_cache: int = 64,
        num_layers: int = 3,
        num_slots: int = 8,
        num_patterns: int = 16,
        num_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.num_patterns = num_patterns
        self.total_slots = num_layers * num_slots
        self.d_layer = d_cache // 4

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        self.slot_embeddings = nn.Parameter(torch.randn(num_layers, num_slots, d_cache) * 0.02)
        self.layer_id_embeddings = nn.Parameter(torch.randn(num_layers, self.d_layer) * 0.02)

        self.layers = nn.ModuleList([
            DLSMNLayer(i, d_model, d_cache, num_slots, num_layers, num_patterns, num_heads, dropout)
            for i in range(num_layers)
        ])

        self.cache_self_attn = CacheSelfAttention(d_cache, num_heads, dropout)
        self.halt_net = nn.Sequential(
            nn.Linear(d_cache, d_cache // 2),
            nn.ReLU(),
            nn.Linear(d_cache // 2, 1),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def get_initial_cache(self, batch_size, device, features=None):
        if features is None:
            features = FeatureFlags()
        if features.use_slot_embeddings:
            cache = self.slot_embeddings.view(self.total_slots, self.d_cache)
            return cache.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return torch.zeros(batch_size, self.total_slots, self.d_cache, device=device)

    def get_layer_ids(self, features=None):
        if features is None:
            features = FeatureFlags()
        if features.use_layer_id:
            layer_ids = self.layer_id_embeddings.unsqueeze(1).expand(-1, self.num_slots, -1)
            return layer_ids.reshape(self.total_slots, self.d_layer)
        return torch.zeros(self.total_slots, self.d_layer, device=self.layer_id_embeddings.device)

    def get_cache_mask(self, batch_size, layer_idx, pass_num, device):
        mask = torch.zeros(batch_size, self.total_slots, dtype=torch.bool, device=device)
        if pass_num == 1:
            mask[:, layer_idx * self.num_slots:] = True
        return mask

    def apply_cache_updates(self, cache, updates, layer_idx, threshold):
        B = cache.shape[0]
        y_cache = updates['y_cache']
        scores = updates['scores']
        slot_probs = updates['slot_probs']
        K = self.num_slots

        write_mask = (scores > threshold).float().unsqueeze(-1)
        weights = write_mask * scores.unsqueeze(-1)
        weighted_y = weights * y_cache

        slot_writes = torch.einsum('bsk,bsd->bkd', slot_probs * write_mask, weighted_y)
        slot_weights = torch.einsum('bsk,bs->bk', slot_probs, write_mask.squeeze(-1) * scores)
        slot_weights = slot_weights.unsqueeze(-1).clamp(min=1e-8)
        slot_writes = slot_writes / slot_weights

        new_cache = cache.clone()
        start_idx = layer_idx * K
        end_idx = start_idx + K
        has_writes = (slot_weights.squeeze(-1) > 1e-6).unsqueeze(-1).float()
        new_cache[:, start_idx:end_idx] = has_writes * slot_writes + (1 - has_writes) * cache[:, start_idx:end_idx]
        return new_cache

    def compute_halt_prob(self, cache):
        return self.halt_net(cache.mean(dim=1)).squeeze(-1)

    def forward(self, x, config=None, step=0, return_aux=True):
        if config is None:
            config = TrainingConfig()

        B, S = x.shape
        device = x.device
        phase = config.get_phase(step)
        temperature = config.get_temperature(step)
        hard = (phase == "hard")
        threshold = config.write_threshold
        features = config.features
        num_passes = config.max_passes if features.use_multi_pass else 1

        h = self.embedding(x) + self.pos_encoding[:, :S, :]
        cache = self.get_initial_cache(B, device, features)
        layer_ids = self.get_layer_ids(features).to(device)

        aux_data = {'entropy': [], 'slot_counts': [], 'patterns': [], 'cache_states': [], 'halt_probs': []}
        cumulative_halt = torch.zeros(B, device=device)
        total_ponder_cost = 0.0

        pass_num = 1
        for pass_num in range(1, num_passes + 1):
            for layer_idx, layer in enumerate(self.layers):
                cache_mask = self.get_cache_mask(B, layer_idx, pass_num, device)
                slot_emb = self.slot_embeddings[layer_idx]
                h, updates = layer(h, cache, slot_emb, layer_ids, cache_mask, temperature, hard, features)

                if features.use_cache:
                    cache = self.apply_cache_updates(cache, updates, layer_idx, threshold)

                if return_aux:
                    aux_data['entropy'].append(updates['entropy'])
                    aux_data['slot_counts'].append(updates['slot_counts'])
                    aux_data['patterns'].append(updates['patterns'])

            if pass_num < num_passes and features.use_cache_self_attn:
                cache = self.cache_self_attn(cache)

            if features.use_act_halting:
                halt_prob = self.compute_halt_prob(cache)
                aux_data['halt_probs'].append(halt_prob)
                remaining = 1 - cumulative_halt
                total_ponder_cost += remaining.mean()
                cumulative_halt = cumulative_halt + (1 - cumulative_halt) * halt_prob
                if not self.training and halt_prob.mean() > 0.5:
                    break
            else:
                aux_data['halt_probs'].append(torch.zeros(B, device=device))

            if return_aux:
                aux_data['cache_states'].append(cache.clone())

        logits = self.output_proj(h)

        aux_info = {'temperature': temperature, 'phase': phase, 'ponder_cost': total_ponder_cost, 'num_passes': pass_num}
        if return_aux and aux_data['entropy']:
            aux_info['avg_entropy'] = torch.stack(aux_data['entropy']).mean()
            aux_info['slot_counts'] = aux_data['slot_counts']
            aux_info['patterns'] = aux_data['patterns']
            aux_info['cache_states'] = aux_data['cache_states']
            aux_info['halt_probs'] = aux_data['halt_probs']

        return logits, cache, aux_info


# ============================================================================
# Auxiliary Losses (Section 9)
# ============================================================================

def compute_diversity_loss(slot_counts_list):
    if not slot_counts_list:
        return torch.tensor(0.0)
    total_counts = torch.stack(slot_counts_list).sum(dim=0) + 1e-8
    probs = total_counts / total_counts.sum(dim=-1, keepdim=True)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    return -entropy


def compute_balance_loss(slot_counts_list):
    if not slot_counts_list:
        return torch.tensor(0.0)
    total_counts = torch.stack(slot_counts_list).sum(dim=0)
    mean = total_counts.mean(dim=-1, keepdim=True)
    std = total_counts.std(dim=-1, keepdim=True)
    cv = std / (mean + 1e-8)
    return (cv ** 2).mean()


def compute_consistency_loss(cache_states, patterns_list, model):
    if len(cache_states) < 2 or len(patterns_list) < 2:
        return torch.tensor(0.0)
    total_loss = torch.tensor(0.0, device=cache_states[0].device)
    K = model.num_slots
    for layer_idx in range(model.num_layers - 1):
        start_idx = layer_idx * K
        cache_j = cache_states[-1][:, start_idx:start_idx + K]
        if layer_idx + 1 < len(model.layers):
            next_layer = model.layers[layer_idx + 1]
            B = cache_j.shape[0]
            layer_id = model.layer_id_embeddings[layer_idx + 1].unsqueeze(0).unsqueeze(0).expand(B, K, -1)
            cache_with_id = torch.cat([cache_j, layer_id], dim=-1)
            decoded = next_layer.W_decompress(cache_with_id)
            if layer_idx + 1 < len(patterns_list):
                target = patterns_list[layer_idx + 1].detach()
                total_loss = total_loss + F.mse_loss(decoded.mean(dim=1), target.mean(dim=1))
    return total_loss / max(model.num_layers - 1, 1)


# ============================================================================
# Associative Recall Dataset
# ============================================================================

class AssociativeRecallDataset(Dataset):
    """Task: Given "A→7, B→3, C→9, query:A" → answer "7" """

    def __init__(self, num_samples=10000, num_pairs=5, num_keys=26, num_values=10, num_distractors=3):
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_distractors = num_distractors

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
        keys = random.sample(range(self.num_keys), self.num_pairs)
        values = [random.randint(0, self.num_values - 1) for _ in range(self.num_pairs)]

        seq = []
        for i, (k, v) in enumerate(zip(keys, values)):
            seq.extend([self.KEY_START + k, self.ARROW, self.VAL_START + v])
            if i < self.num_pairs - 1:
                seq.extend([self.DIST_START + random.randint(0, 9) for _ in range(random.randint(0, self.num_distractors))])

        query_idx = random.randint(0, self.num_pairs - 1)
        seq.extend([self.SEP, self.KEY_START + keys[query_idx], self.MASK])
        return torch.tensor(seq), self.VAL_START + values[query_idx], len(seq) - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    seqs, targets, mask_positions = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    return padded, torch.tensor(targets), torch.tensor(mask_positions)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, config, global_step=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    features = config.features

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for seqs, targets, mask_pos in pbar:
        seqs, targets, mask_pos = seqs.to(device), targets.to(device), mask_pos.to(device)
        optimizer.zero_grad()

        logits, cache, aux_info = model(seqs, config=config, step=global_step, return_aux=True)
        B = seqs.shape[0]
        mask_logits = logits[torch.arange(B, device=device), mask_pos]
        task_loss = F.cross_entropy(mask_logits, targets)

        phase = config.get_phase(global_step)

        div_loss = compute_diversity_loss(aux_info.get('slot_counts', [])) if features.use_diversity_loss else 0
        bal_loss = compute_balance_loss(aux_info.get('slot_counts', [])) if features.use_balance_loss else 0
        sparse_loss = aux_info.get('avg_entropy', 0) if features.use_sparsity_loss else 0
        consist_loss = compute_consistency_loss(aux_info.get('cache_states', []), aux_info.get('patterns', []), model) if features.use_consistency_loss else 0
        ponder_loss = aux_info.get('ponder_cost', 0) if features.use_ponder_loss else 0

        if phase == "warmup":
            loss = task_loss + config.lambda_diversity * div_loss + config.lambda_balance * bal_loss
        elif phase == "transition":
            loss = task_loss + config.lambda_consistency * consist_loss + config.lambda_sparsity * sparse_loss
        else:
            loss = task_loss + config.lambda_consistency * consist_loss + 2 * config.lambda_sparsity * sparse_loss + config.lambda_ponder * ponder_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.detach().item()
        global_step += 1
        preds = mask_logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += B

        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.3f}', 'τ': f'{aux_info["temperature"]:.2f}', 'phase': phase[:4]})

    return total_loss / len(dataloader), correct / total, global_step


@torch.no_grad()
def evaluate(model, dataloader, device, config):
    model.eval()
    correct = 0
    total = 0

    eval_config = TrainingConfig(tau_min=0.1, tau_start=0.1, max_passes=config.max_passes, features=config.features)

    for seqs, targets, mask_pos in tqdm(dataloader, desc="Eval", leave=False):
        seqs, targets, mask_pos = seqs.to(device), targets.to(device), mask_pos.to(device)
        logits, _, _ = model(seqs, config=eval_config, step=config.transition_steps + 1, return_aux=False)
        B = seqs.shape[0]
        mask_logits = logits[torch.arange(B, device=device), mask_pos]
        correct += (mask_logits.argmax(dim=-1) == targets).sum().item()
        total += B

    return correct / total


def visualize_cache(model, sample_seq, device, dataset, config):
    model.eval()
    seq = sample_seq.unsqueeze(0).to(device)
    with torch.no_grad():
        _, cache, _ = model(seq, config=config, step=0, return_aux=False)

    tokens = []
    for t in sample_seq.tolist():
        if t == dataset.PAD: tokens.append('[PAD]')
        elif t == dataset.SEP: tokens.append('[SEP]')
        elif t == dataset.MASK: tokens.append('[MASK]')
        elif t == dataset.ARROW: tokens.append('→')
        elif dataset.KEY_START <= t < dataset.VAL_START: tokens.append(chr(ord('A') + t - dataset.KEY_START))
        elif dataset.VAL_START <= t < dataset.DIST_START: tokens.append(str(t - dataset.VAL_START))
        else: tokens.append(f'd{t - dataset.DIST_START}')

    print(f"\nSequence: {' '.join(tokens)}")
    print(f"Cache shape: {cache.shape}")
    print("Cache L2 norms per slot:")
    norms = cache[0].norm(dim=-1)
    for layer in range(model.num_layers):
        start = layer * model.num_slots
        print(f"  Layer {layer}: {norms[start:start + model.num_slots].cpu().numpy().round(2)}")


# ============================================================================
# Main
# ============================================================================

def main():
    import sys

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = AssociativeRecallDataset(num_samples=10000, num_pairs=5)
    test_dataset = AssociativeRecallDataset(num_samples=1000, num_pairs=5)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    print(f"Vocab size: {train_dataset.vocab_size}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Feature preset selection
    preset_name = sys.argv[1] if len(sys.argv) > 1 else "fast"
    features = FEATURE_PRESETS.get(preset_name, FEATURE_PRESETS["fast"])
    print(f"\nUsing feature preset: '{preset_name}'")
    print(f"  {features.describe()}")

    config = TrainingConfig(
        warmup_steps=len(train_loader) * 5,
        transition_steps=len(train_loader) * 20,
        max_passes=2 if features.use_multi_pass else 1,
        features=features,
    )

    model = DLSMN(
        vocab_size=train_dataset.vocab_size,
        d_model=128,
        d_cache=64,
        num_layers=3,
        num_slots=8,
        num_patterns=16,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print("\n" + "=" * 60)
    print("Training DLSMN on Associative Recall")
    print("=" * 60)
    print(f"Feature preset: {preset_name}")
    print(f"Training phases: warmup({config.warmup_steps}) → transition({config.transition_steps}) → hard")
    print(f"Temperature: {config.tau_start} → {config.tau_min} (anneal_rate={config.anneal_rate})")
    print(f"Max passes: {config.max_passes}")
    print("=" * 60 + "\n")

    best_acc = 0
    global_step = 0

    for epoch in range(50):
        train_loss, train_acc, global_step = train_epoch(model, train_loader, optimizer, device, config, global_step)

        if (epoch + 1) % 10 == 0:
            test_acc = evaluate(model, test_loader, device, config)
            marker = " ★" if test_acc > best_acc else ""
            best_acc = max(best_acc, test_acc)
            phase = config.get_phase(global_step)
            temp = config.get_temperature(global_step)
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | τ={temp:.2f} [{phase}]{marker}")
        else:
            phase = config.get_phase(global_step)
            temp = config.get_temperature(global_step)
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train: {train_acc:.3f} | τ={temp:.2f} [{phase}]")

        scheduler.step()

    print(f"\nBest test accuracy: {best_acc:.3f}")

    print("\n" + "=" * 60)
    print("Cache Visualization")
    print("=" * 60)
    sample_seq, _, _ = test_dataset[0]
    visualize_cache(model, sample_seq, device, test_dataset, config)

    print("\n" + "=" * 60)
    print("Accuracy vs Number of Passes")
    print("=" * 60)
    for passes in [1, 2, 3]:
        test_config = TrainingConfig(max_passes=passes, features=FeatureFlags(use_multi_pass=(passes > 1)))
        acc = evaluate(model, test_loader, device, test_config)
        print(f"  {passes} pass(es): {acc:.3f}")


if __name__ == "__main__":
    main()
