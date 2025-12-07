"""
DLSMN on ARC-AGI-2
==================
Faithful implementation of DLSM_V0.1.md for the Abstraction and Reasoning Corpus.

Key features from the spec:
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
import json
import os
import glob
import random
import math
from pathlib import Path
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
    use_hybrid_routing: bool = False    # False = learned routing only (RECOMMENDED DEFAULT)
                                        # True = add content-based similarity (only if needed)
    use_layer_id: bool = True           # False = no layer separation in cache
    
    # Architecture improvements
    use_cache_self_attn: bool = True    # False = no inter-pass cache reasoning
    use_act_halting: bool = True        # False = fixed number of passes
    use_gated_fusion: bool = True       # False = additive fusion
    
    # Pattern pooling
    use_pattern_pooling: bool = True    # False = direct token caching
    
    # === AUXILIARY LOSSES (simplified) ===
    use_diversity_loss: bool = True     # Prevent slot collapse (warmup only)
    use_ponder_loss: bool = True        # Penalize excessive passes (ACT only)
    # REMOVED: balance (redundant with diversity), sparsity (redundant with Gumbel),
    #          consistency (over-engineered, no evidence it helps)
    
    # === REFINEMENTS (from review) ===
    use_write_count_masking: bool = True # Mask unwritten slots with -inf
    use_temporal_decay: bool = True      # Store age per slot
    use_noise_injection: bool = True     # Add noise to router logits early in training
    use_soft_wta_update: bool = True     # Use exponential weighting for updates
    
    # === TRM INSIGHTS (ref/2510.04871v1.pdf) ===
    use_deep_supervision: bool = True    # Train on every pass
    use_explicit_q_head: bool = True     # Train halt_net as Q-Head (correctness predictor)
    use_answer_feedback: bool = True     # Feed previous pass's answer back (TRM's key insight)
    
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
        if self.use_ponder_loss: losses.append("ponder")
        
        refinements = []
        if self.use_write_count_masking: refinements.append("masking")
        if self.use_temporal_decay: refinements.append("decay")
        if self.use_noise_injection: refinements.append("noise")
        if self.use_soft_wta_update: refinements.append("soft-wta")
        
        trm = []
        if self.use_deep_supervision: trm.append("deep-sup")
        if self.use_explicit_q_head: trm.append("q-head")
        if self.use_answer_feedback: trm.append("ans-fb")
        
        return (f"Core: [{', '.join(core)}] | "
                f"Improvements: [{', '.join(improvements)}] | "
                f"Refinements: [{', '.join(refinements)}] | "
                f"TRM: [{', '.join(trm)}] | "
                f"Losses: [{', '.join(losses)}]")


# Preset configurations for common ablation experiments
FEATURE_PRESETS = {
    # Full model with all features
    "full": FeatureFlags(),
    
    # Core only - minimal DLSMN
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
        use_ponder_loss=False,
    ),
    
    # No multi-pass (single pass baseline)
    "single_pass": FeatureFlags(
        use_multi_pass=False,
        use_cache_self_attn=False,
        use_act_halting=False,
        use_ponder_loss=False,
    ),
    
    # No Gumbel-Softmax (hard routing from start)
    "no_gumbel": FeatureFlags(use_gumbel_softmax=False),
    
    # No slot embeddings (zero-init cache)
    "no_slot_emb": FeatureFlags(use_slot_embeddings=False),
    
    # With hybrid routing (learned + content-based, only if needed)
    "with_hybrid": FeatureFlags(use_hybrid_routing=True),
    
    # No cache-to-cache attention
    "no_cache_attn": FeatureFlags(use_cache_self_attn=False),
    
    # No auxiliary losses
    "no_aux_loss": FeatureFlags(
        use_diversity_loss=False,
        use_ponder_loss=False,
    ),
    
    # Fast training (minimal overhead)
    "fast": FeatureFlags(
        use_multi_pass=False,
        use_cache_self_attn=False,
        use_act_halting=False,
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
    # Temperature annealing (Section 8.2, 8.3)
    tau_start: float = 1.0       # Initial temperature (soft routing)
    tau_min: float = 0.1         # Final temperature (near-hard routing)
    anneal_rate: float = 0.0003  # Exponential decay rate
    
    # Training phases (Section 8.4)
    warmup_steps: int = 5000
    transition_steps: int = 20000  # End of transition phase
    
    # Loss weights (simplified - only what's needed)
    lambda_diversity: float = 0.01  # Prevents slot collapse (warmup only)
    lambda_ponder: float = 0.01     # Penalizes excessive passes (ACT only)
    lambda_q_head: float = 0.1      # Q-head correctness predictor (TRM)
    
    # Importance threshold
    write_threshold: float = 0.5
    
    # Hybrid routing (Section 8.8 - Optional)
    # Default α=1.0 means pure learned routing. Only set <1.0 if model struggles to organize memory.
    alpha_learned: float = 1.0  # Balance between learned (1.0) and content-based (0.0) routing
    
    # ACT (Section 10.1)
    max_passes: int = 3
    
    # Exploration (Section 9.2.1)
    exploration_steps: int = 2000
    exploration_start: float = 1.0
    
    # Feature flags for ablation
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
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


# ============================================================================
# ARC Dataset
# ============================================================================

class ARCDataset(Dataset):
    """
    ARC-AGI dataset loader (lazy loading).
    
    Each task has:
    - train: list of {input: grid, output: grid} demo pairs
    - test: list of {input: grid, output: grid} test pairs
    
    Files are loaded on-the-fly to avoid slow startup.
    """
    
    def __init__(self, data_dir: str, split: str = "training", max_grid_size: int = 30):
        self.max_grid_size = max_grid_size
        self.task_dir = Path(data_dir) / split
        
        # Just get file list (fast)
        self.task_files = sorted(self.task_dir.glob("*.json"))
        print(f"Found {len(self.task_files)} tasks in {self.task_dir}")
        
        # Build index: (file_idx, test_idx) for each sample
        # We need to know how many test cases per file - do a quick scan
        self.index = []
        for file_idx, task_file in enumerate(self.task_files):
            with open(task_file) as f:
                task = json.load(f)
            num_tests = len(task["test"])
            for test_idx in range(num_tests):
                self.index.append((file_idx, test_idx))
        
        print(f"Created {len(self.index)} samples")
        
        # Cache for recently loaded files
        self._cache = {}
        self._cache_size = 100
    
    def __len__(self):
        return len(self.index)
    
    def _load_task(self, file_idx):
        """Load task file with simple caching."""
        if file_idx not in self._cache:
            # Evict old entries if cache is full
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            
            with open(self.task_files[file_idx]) as f:
                self._cache[file_idx] = json.load(f)
        
        return self._cache[file_idx]
    
    def pad_grid(self, grid, size):
        """Pad grid to fixed size, return grid and mask."""
        h, w = len(grid), len(grid[0]) if grid else 0
        padded = torch.zeros(size, size, dtype=torch.long)
        mask = torch.ones(size, size, dtype=torch.bool)  # True = padding
        
        for i in range(min(h, size)):
            for j in range(min(w, size)):
                padded[i, j] = grid[i][j]
                mask[i, j] = False
        
        return padded, mask, h, w
    
    def __getitem__(self, idx):
        # Get file and test indices from our index
        file_idx, test_idx = self.index[idx]
        task = self._load_task(file_idx)
        
        size = self.max_grid_size
        
        # Encode demo pairs (from "train" in the JSON)
        demo_inputs = []
        demo_outputs = []
        demo_masks = []
        
        for pair in task["train"]:
            inp, inp_mask, _, _ = self.pad_grid(pair["input"], size)
            out, out_mask, _, _ = self.pad_grid(pair["output"], size)
            demo_inputs.append(inp)
            demo_outputs.append(out)
            demo_masks.append(inp_mask)
        
        # Pad to fixed number of demos (3)
        while len(demo_inputs) < 3:
            demo_inputs.append(torch.zeros(size, size, dtype=torch.long))
            demo_outputs.append(torch.zeros(size, size, dtype=torch.long))
            demo_masks.append(torch.ones(size, size, dtype=torch.bool))
        
        demo_inputs = torch.stack(demo_inputs[:3])  # [3, H, W]
        demo_outputs = torch.stack(demo_outputs[:3])  # [3, H, W]
        demo_masks = torch.stack(demo_masks[:3])  # [3, H, W]
        
        # Encode test (use specific test_idx)
        test_pair = task["test"][test_idx]
        test_input, test_mask, _, _ = self.pad_grid(test_pair["input"], size)
        test_output, output_mask, out_h, out_w = self.pad_grid(test_pair["output"], size)
        
        # Task ID from filename
        task_id = self.task_files[file_idx].stem
        
        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "demo_masks": demo_masks,
            "test_input": test_input,
            "test_mask": test_mask,
            "test_output": test_output,
            "output_mask": output_mask,
            "output_size": torch.tensor([out_h, out_w]),
            "task_id": task_id,
        }


# ============================================================================
# DLSMN Components (Following DLSM_V0.1.md faithfully)
# ============================================================================

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
    Head B: The Gatekeeper (Section 2.2, 8.2, 8.8)
    
    Decides what to cache and where using:
    - Importance score (gated sigmoid)
    - Attention sharpening γ (learnable, controls slot selection precision)
    - Learned slot routing (default) with optional hybrid content-based routing
    - Gumbel-Softmax for differentiable slot selection
    
    The model learns:
    - Which concepts deserve caching (importance score)
    - Where to store them (slot selection via W_slot)
    - When to overwrite stale content (implicitly through task loss)
    """
    
    def __init__(self, d_model: int, d_cache: int, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        self.d_cache = d_cache
        
        # Importance gate (Section 2.2)
        self.gate = nn.Linear(d_model, 1)
        
        # Learned slot routing (Section 8.2 - default)
        self.slot_selector = nn.Linear(d_model, num_slots)
        
        # Attention sharpening γ (Section 2.2): controls slot selection precision
        # High γ → sharper selection (more confident routing)
        # Low γ → softer selection (more exploration)
        # Initialize to 1.0
        self.gamma = nn.Parameter(torch.ones(1))
        
        # Content-based routing queries (Section 8.8 - optional hybrid)
        self.slot_query = nn.Linear(d_model, d_cache)
        self.slot_key = nn.Linear(d_cache, d_cache)
        
        # Dynamic alpha for hybrid routing (Section 8.8)
        # Initialized with bias toward α=1 (pure learned routing)
        self.alpha_net = nn.Linear(d_model, 1)
        nn.init.constant_(self.alpha_net.bias, 2.0)  # sigmoid(2.0) ≈ 0.88 → mostly learned
        
    def forward(
        self, 
        y: torch.Tensor,                    # [B, S, D] - patterns to potentially cache
        slot_embeddings: torch.Tensor,      # [K, D_cache] - learned slot anchors
        temperature: float = 1.0,
        hard: bool = False,
        features: Optional[FeatureFlags] = None,
        step: int = 0,
        exploration_steps: int = 0,
        exploration_start: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            scores: [B, S] importance scores
            slot_probs: [B, S, K] slot assignment probabilities
            soft_probs: [B, S, K] soft probabilities for entropy computation
            alpha: [B, S] routing balance per token
        """
        if features is None:
            features = FeatureFlags()
            
        B, S, D = y.shape
        K = self.num_slots
        
        # Importance score: σ(W_gate · y) ∈ [0, 1]
        scores = torch.sigmoid(self.gate(y)).squeeze(-1)  # [B, S]
        
        # Learned routing logits: W_slot · y
        learned_logits = self.slot_selector(y)  # [B, S, K]
        
        # Apply attention sharpening γ (Section 2.2)
        # slot_logits = γ * slot_logits
        learned_logits = self.gamma * learned_logits
        
        # [REFINEMENT: use_noise_injection]
        # Section 9.2.1: Random Noise Injection for Cold Start Exploration
        if features.use_noise_injection and step < exploration_steps:
            noise_scale = exploration_start * (1 - step / exploration_steps)
            # Add noise to logits to encourage exploring all slots
            learned_logits = learned_logits + noise_scale * torch.randn_like(learned_logits)
        
        # [ABLATION: use_hybrid_routing]
        # Section 8.8: Hybrid routing is OPTIONAL. Default is pure learned (α=1).
        # Only use content-based if model struggles to organize memory.
        if features.use_hybrid_routing:
            # Content-based similarity to slot anchors (Section 8.7, 8.8)
            query = self.slot_query(y)  # [B, S, D_cache]
            keys = self.slot_key(slot_embeddings)  # [K, D_cache]
            content_logits = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.d_cache)
            
            # Dynamic alpha: learned per-token balance (Section 8.8)
            # Initialized to favor learned routing (α ≈ 0.88)
            alpha = torch.sigmoid(self.alpha_net(y)).squeeze(-1)  # [B, S]
            
            # Hybrid routing: α · learned + (1-α) · content-based
            # When α=1: pure learned routing (model decides staleness implicitly)
            # When α=0: pure content-based (similar content clusters together)
            alpha_expanded = alpha.unsqueeze(-1)  # [B, S, 1]
            combined_logits = alpha_expanded * learned_logits + (1 - alpha_expanded) * content_logits
        else:
            # Pure learned routing (recommended default)
            combined_logits = learned_logits
            alpha = torch.ones(B, S, device=y.device)  # α=1 means pure learned
        
        # [ABLATION: use_gumbel_softmax]
        if features.use_gumbel_softmax:
            slot_probs = gumbel_softmax(combined_logits, temperature, hard=hard)
        else:
            # Hard argmax routing (non-differentiable, use STE)
            idx = combined_logits.argmax(dim=-1)
            slot_probs = F.one_hot(idx, K).float()
        
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
        # +layer_id + age (1 dim)
        self.cache_key = nn.Linear(d_cache + self.d_layer + 1, d_cache)  
        self.cache_value = nn.Linear(d_cache + self.d_layer + 1, d_cache)  # +layer_id + age
        
        # Gated fusion (Section 3.2 - recommended)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        
        # Layer-Selective Cache Injection (Section 3.2)
        # Per-layer gate controlling how much cache context to use
        # g_j = σ(W_inject · x) ∈ [0, 1]
        # Early layers can learn to ignore cache, later layers use it heavily
        self.inject_gate = nn.Linear(d_model, 1)
        # Initialize bias to allow gradual cache usage (sigmoid(0) = 0.5)
        nn.init.constant_(self.inject_gate.bias, 0.0)
        
    def read_cache(
        self, 
        x: torch.Tensor,           # [B, S, D_model]
        cache: torch.Tensor,       # [B, L*K, D_cache]
        layer_ids: torch.Tensor,   # [L*K, D_layer] - layer embeddings for each slot
        cache_mask: Optional[torch.Tensor] = None,  # [B, L*K] True=blocked
        write_counts: Optional[torch.Tensor] = None, # [B, L*K] number of writes per slot
        slot_ages: Optional[torch.Tensor] = None,    # [B, L*K] age since last write
        features: Optional[FeatureFlags] = None,
    ) -> torch.Tensor:
        """
        Step 1: READ - Retrieve context from cache (Section 3.1)
        
        q = W_Q · x
        α = Softmax(q · (W_K · [C; layer_id])^T / √d + M)
        raw_context = α · (W_V · [C; layer_id])
        context = W_decompress · raw_context
        """
        if features is None:
            features = FeatureFlags()
            
        B, S, _ = x.shape
        _, total_slots, _ = cache.shape
        
        # Concatenate layer-ID embeddings to cache entries (Section 7.6)
        cache_with_id = torch.cat([cache, layer_ids.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        
        # [REFINEMENT: use_temporal_decay]
        # Section 8.4: Append age to cache key projection
        if features.use_temporal_decay and slot_ages is not None:
            # Normalize age (simple scaling)
            normalized_age = slot_ages.unsqueeze(-1) / 100.0  # Scale down
            cache_with_id = torch.cat([cache_with_id, normalized_age], dim=-1)
        else:
            # Append zero age if feature disabled or ages not provided
            zeros = torch.zeros(B, total_slots, 1, device=x.device)
            cache_with_id = torch.cat([cache_with_id, zeros], dim=-1)
        
        q = self.cache_query(x)  # [B, S, D_cache]
        k = self.cache_key(cache_with_id)  # [B, L*K, D_cache]
        v = self.cache_value(cache_with_id)  # [B, L*K, D_cache]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_cache)
        
        # Apply pass-aware mask (Section 2.3)
        if cache_mask is not None:
            mask = cache_mask.unsqueeze(1).expand(-1, S, -1)
            scores = scores.masked_fill(mask, float('-inf'))
            
        # [REFINEMENT: use_write_count_masking]
        # Section 7 & 8.1: Mask unwritten slots to mitigate Cold Start
        if features.use_write_count_masking and write_counts is not None:
            # Mask where write_count == 0
            unwritten = (write_counts == 0).unsqueeze(1).expand(-1, S, -1)
            scores = scores.masked_fill(unwritten, float('-inf'))
        
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
    
    def fuse(self, x: torch.Tensor, context: torch.Tensor, use_layer_selective: bool = False) -> torch.Tensor:
        """
        Cache fusion strategies (Section 3.2):
        
        Gated fusion (default):
            g = σ(W[x, context])
            output = g·x + (1-g)·context
        
        Layer-Selective injection:
            g_j = σ(W_inject · x) ∈ [0, 1]
            output = x + g_j · context
            Allows each layer to learn how much cache context to incorporate.
        """
        if use_layer_selective:
            # Layer-Selective Cache Injection (Section 3.2)
            # g_j controls per-layer cache influence
            g_j = torch.sigmoid(self.inject_gate(x))  # [B, S, 1]
            return x + g_j * context
        else:
            # Standard gated fusion
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
        write_counts: Optional[torch.Tensor] = None,
        slot_ages: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = False,
        features: Optional[FeatureFlags] = None,
        step: int = 0,
        exploration_steps: int = 0,
        exploration_start: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full Compute-Select-Cache cycle (Section 3.1).
        """
        if features is None:
            features = FeatureFlags()
            
        B, S, _ = x.shape
        
        # Step 1: READ from cache [ABLATION: use_cache]
        if features.use_cache:
            context = self.read_cache(
                x, cache, layer_ids, cache_mask, 
                write_counts=write_counts, slot_ages=slot_ages, features=features
            )
            # [ABLATION: use_gated_fusion]
            if features.use_gated_fusion:
                # Use layer-selective injection for middle/later layers
                # Early layers (idx 0) use standard gated fusion
                use_layer_selective = (self.layer_idx > 0)
                x_fused = self.fuse(x, context, use_layer_selective=use_layer_selective)
            else:
                x_fused = x + context  # Simple additive fusion
        else:
            x_fused = x  # No cache read
        
        # Step 2: COMPUTE (Head A)
        attn_out, _ = self.self_attn(x_fused, x_fused, x_fused)
        x_fused = self.norm1(x_fused + attn_out)
        ffn_out = self.ffn(x_fused)
        y = self.norm2(x_fused + ffn_out)
        
        # [ABLATION: use_pattern_pooling]
        if features.use_pattern_pooling:
            # Pattern pooling: extract pattern-level summaries
            queries = self.pattern_queries.unsqueeze(0).expand(B, -1, -1)
            patterns, _ = self.pattern_attn(queries, y, y)  # [B, num_patterns, D_model]
        else:
            # Direct token selection (sample or pool)
            if S <= self.num_patterns:
                # Pad if fewer tokens than patterns
                patterns = F.pad(y, (0, 0, 0, self.num_patterns - S))
            else:
                # Simple pooling to reduce to num_patterns
                patterns = y[:, :self.num_patterns, :]  # Take first N tokens
        
        # Step 3: SELECT (Head B) [ABLATION: use_selection_head]
        if features.use_selection_head:
            selection = self.selection_head(
                patterns, slot_embeddings, temperature, hard, features,
                step=step,
                exploration_steps=exploration_steps,
                exploration_start=exploration_start
            )
        else:
            # Uniform selection (all patterns equally important, uniform slot distribution)
            num_p = patterns.shape[1]
            K = self.num_slots
            selection = {
                'scores': torch.ones(B, num_p, device=x.device),
                'slot_probs': torch.ones(B, num_p, K, device=x.device) / K,
                'soft_probs': torch.ones(B, num_p, K, device=x.device) / K,
                'alpha': torch.ones(B, num_p, device=x.device),
            }
        
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


class DLSMN_ARC(nn.Module):
    """
    DLSMN model for ARC-AGI tasks.
    
    Faithful implementation of DLSM_V0.1.md:
    - Global cache: C ∈ ℝ^{(L×K) × D_cache} (Section 2.1)
    - Learned slot embeddings as concept anchors (Section 7.7)
    - Layer-ID embeddings for representational separation (Section 7.6)
    - Per-layer W_compress and W_decompress projections (Section 2.1)
    - Pass-aware masking (Section 2.3)
    - Cache-to-cache attention (Section 10.2)
    - ACT halting (Section 10.1)
    """
    
    def __init__(
        self,
        num_colors: int = 10,
        d_model: int = 128,
        d_cache: int = 64,
        num_layers: int = 3,
        num_slots: int = 16,
        num_patterns: int = 16,
        num_heads: int = 4,
        max_grid_size: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_colors = num_colors
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.num_patterns = num_patterns
        self.total_slots = num_layers * num_slots
        self.max_grid_size = max_grid_size
        self.d_layer = d_cache // 4  # Layer-ID embedding dimension
        
        # Embeddings
        self.color_embed = nn.Embedding(num_colors + 1, d_model)
        self.pos_embed_h = nn.Embedding(max_grid_size, d_model // 2)
        self.pos_embed_w = nn.Embedding(max_grid_size, d_model // 2)
        self.type_embed = nn.Embedding(4, d_model)
        
        # Learned slot embeddings (Section 7.7) - concept anchors
        # S ∈ ℝ^{(L×K) × D_cache}
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_layers, num_slots, d_cache) * 0.02
        )
        
        # Layer-ID embeddings (Section 7.6) - for representational separation
        self.layer_id_embeddings = nn.Parameter(
            torch.randn(num_layers, self.d_layer) * 0.02
        )
        
        # DLSMN layers with proper signatures
        self.layers = nn.ModuleList([
            DLSMNLayer(
                layer_idx=i,
                d_model=d_model,
                d_cache=d_cache,
                num_slots=num_slots,
                num_layers=num_layers,
                num_patterns=num_patterns,
                num_heads=num_heads,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        
        # Cache-to-cache attention (Section 10.2)
        self.cache_self_attn = CacheSelfAttention(d_cache, num_heads, dropout)
        
        # ACT halting mechanism (Section 10.1)
        # h^{(p)} = σ(W_h · pool(C^{(p)}))
        self.halt_net = nn.Sequential(
            nn.Linear(d_cache, d_cache // 2),
            nn.ReLU(),
            nn.Linear(d_cache // 2, 1),
            nn.Sigmoid(),
        )
        
        # Predictive head for auxiliary loss (Section 9.1)
        self.predictor = nn.Sequential(
            nn.Linear(d_cache, d_cache),
            nn.ReLU(),
            nn.Linear(d_cache, d_model),
        )
        
        # Output head
        self.output_proj = nn.Linear(d_model, num_colors)
        
        # [TRM INSIGHT: Answer Feedback]
        # Project previous answer back to embedding space for next pass
        # This allows the model to "see" its previous attempt and refine it
        self.answer_embed = nn.Embedding(num_colors, d_model)
        # Gate to control how much previous answer influences next pass
        self.answer_gate = nn.Linear(d_model * 2, d_model)
        
        # Size prediction head
        self.size_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, max_grid_size * 2),
        )
        
        # Cache initialization with slot embeddings (Section 7.7)
        # C^{(0)} = S
    
    def get_initial_cache(
        self, 
        batch_size: int, 
        device: torch.device,
        features: Optional[FeatureFlags] = None,
    ) -> torch.Tensor:
        """
        Initialize cache with learned slot embeddings (Section 7.7).
        C^{(0)} = S
        """
        if features is None:
            features = FeatureFlags()
        
        # [ABLATION: use_slot_embeddings]
        if features.use_slot_embeddings:
            # Reshape slot embeddings from [L, K, D_cache] to [L*K, D_cache]
            cache = self.slot_embeddings.view(self.total_slots, self.d_cache)
            return cache.unsqueeze(0).expand(batch_size, -1, -1).clone()
        else:
            # Zero initialization
            return torch.zeros(batch_size, self.total_slots, self.d_cache, device=device)
    
    def get_layer_ids(self, features: Optional[FeatureFlags] = None) -> torch.Tensor:
        """
        Get layer-ID embeddings for all slots (Section 7.6).
        Returns [L*K, D_layer] tensor where each slot has its layer's embedding.
        """
        if features is None:
            features = FeatureFlags()
        
        # [ABLATION: use_layer_id]
        if features.use_layer_id:
            # Repeat each layer's embedding K times
            layer_ids = self.layer_id_embeddings.unsqueeze(1).expand(-1, self.num_slots, -1)
            return layer_ids.reshape(self.total_slots, self.d_layer)
        else:
            # Zero layer IDs (no layer separation)
            return torch.zeros(self.total_slots, self.d_layer, device=self.layer_id_embeddings.device)
    
    def embed_grid(self, grid: torch.Tensor, grid_type: int) -> torch.Tensor:
        """Embed a grid with positional and type information."""
        B, H, W = grid.shape
        device = grid.device
        
        color_emb = self.color_embed(grid)
        
        h_pos = torch.arange(H, device=device)
        w_pos = torch.arange(W, device=device)
        h_emb = self.pos_embed_h(h_pos)
        w_emb = self.pos_embed_w(w_pos)
        
        pos_emb = torch.cat([
            h_emb.unsqueeze(1).expand(-1, W, -1),
            w_emb.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1)
        
        type_emb = self.type_embed(torch.tensor(grid_type, device=device))
        emb = color_emb + pos_emb.unsqueeze(0) + type_emb
        
        return emb.view(B, H * W, -1)
    
    def get_cache_mask(
        self, 
        batch_size: int, 
        layer_idx: int, 
        pass_num: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Pass-aware read mask (Section 2.3):
        - Pass 1: Layer j reads C[0:j*K] (earlier layers only)
        - Pass 2+: Layer j reads C[:] (full cache)
        
        Returns mask where True = BLOCKED.
        """
        mask = torch.zeros(batch_size, self.total_slots, dtype=torch.bool, device=device)
        if pass_num == 1:
            first_blocked_slot = layer_idx * self.num_slots
            mask[:, first_blocked_slot:] = True
        return mask
    
    def apply_cache_updates(
        self, 
        cache: torch.Tensor, 
        updates: Dict,
        layer_idx: int,
        threshold: float,
        features: Optional[FeatureFlags] = None,
    ) -> torch.Tensor:
        """
        Apply weighted cache writes (Section 3.1 Step 3, 7.3).
        
        Weighted mean aggregation:
        C[i] = Σ_t (score_t · y_cache_t · 1[slot_t = i]) / Σ_t (score_t · 1[slot_t = i])
        
        [REFINEMENT: use_soft_wta_update]
        Soft Winner-Take-All (Section 8.3):
        Weight by e^score instead of score to let most important token dominate.
        """
        if features is None:
            features = FeatureFlags()
            
        B = cache.shape[0]
        y_cache = updates['y_cache']         # [B, num_patterns, D_cache]
        scores = updates['scores']           # [B, num_patterns]
        slot_probs = updates['slot_probs']   # [B, num_patterns, K]
        K = self.num_slots
        
        # Write mask based on importance threshold
        write_mask = (scores > threshold).float().unsqueeze(-1)  # [B, num_patterns, 1]
        
        # [REFINEMENT: use_soft_wta_update]
        if features.use_soft_wta_update:
            # Exponential weighting: e^score
            # We subtract max score for numerical stability if needed, but scores are [0,1] so it's fine.
            # However, we only want to weight the *written* tokens.
            # If we just do exp(score), unwritten tokens (score < threshold) might still have non-zero weight if we don't mask.
            # But we multiply by write_mask later.
            
            # Use exp(score * scale) to sharpen? Spec says just e^score.
            weights = torch.exp(scores).unsqueeze(-1) * write_mask
        else:
            # Linear weighting (Original)
            weights = scores.unsqueeze(-1) * write_mask
            
        weighted_y = weights * y_cache  # [B, num_patterns, D_cache]
        
        # Aggregate to slots: slot_writes[b, k, d] = Σ_s slot_probs[b,s,k] * weighted_y[b,s,d]
        slot_writes = torch.einsum('bsk,bsd->bkd', slot_probs, weighted_y)
        
        # Sum of weights for normalization
        # slot_weights[b, k] = Σ_s slot_probs[b,s,k] * weights[b,s]
        slot_weights = torch.einsum('bsk,bs->bk', slot_probs, weights.squeeze(-1))
        
        # Normalize
        slot_weights = slot_weights.unsqueeze(-1).clamp(min=1e-8)
        slot_writes = slot_writes / slot_weights
        
        # Update cache at this layer's slots
        new_cache = cache.clone()
        start_idx = layer_idx * K
        end_idx = start_idx + K
        
        # Only update slots that received writes
        has_writes = (slot_weights.squeeze(-1) > 1e-6).unsqueeze(-1).float()
        
        # [REFINEMENT: Soft update interpolation?]
        # Spec says: C[i] <- (1-alpha)C[i] + alpha * new_val
        # But for multi-pass, we usually overwrite or accumulate.
        # The code currently does hard overwrite if has_writes.
        # Let's stick to hard overwrite for now as per "Across time/passes: Hard overwrite" in spec Section 8.3
        # unless we are in RNN mode (which we are adding support for via temporal decay, but this function is general).
        
        new_cache[:, start_idx:end_idx] = (
            has_writes * slot_writes + (1 - has_writes) * cache[:, start_idx:end_idx]
        )
        
        return new_cache
    
    def compute_halt_prob(self, cache: torch.Tensor) -> torch.Tensor:
        """
        ACT halting probability (Section 10.1):
        h^{(p)} = σ(W_h · pool(C^{(p)}))
        """
        pooled = cache.mean(dim=1)  # [B, D_cache]
        return self.halt_net(pooled).squeeze(-1)  # [B]
    
    def forward(
        self,
        demo_inputs: torch.Tensor,    # [B, num_demos, H, W]
        demo_outputs: torch.Tensor,   # [B, num_demos, H, W]
        test_input: torch.Tensor,     # [B, H, W]
        config: Optional[TrainingConfig] = None,
        step: int = 0,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Unified forward pass following DLSM_V0.1.md (Section 11).
        
        Architecture at a Glance:
        1. (Optional) Read: Attend to cache
        2. Compute: Head A processes (with fused context)
        3. Select: Head B decides importance + slot via hybrid routing
        4. Aggregate: Write to cache
        5. (Optional) Cache-to-Cache: Memory-only reasoning between passes
        6. (Optional) ACT halt: Decide whether to continue
        """
        if config is None:
            config = TrainingConfig()
        
        B = test_input.shape[0]
        device = test_input.device
        num_demos = demo_inputs.shape[1]
        H, W = test_input.shape[1], test_input.shape[2]
        
        # Get training phase and temperature
        phase = config.get_phase(step)
        temperature = config.get_temperature(step)
        hard = (phase == "hard")
        threshold = config.write_threshold
        features = config.features
        
        # Determine number of passes [ABLATION: use_multi_pass]
        num_passes = config.max_passes if features.use_multi_pass else 1
        
        # Build unified sequence
        seq_parts = []
        for demo_idx in range(num_demos):
            demo_in = self.embed_grid(demo_inputs[:, demo_idx], grid_type=0)
            demo_out = self.embed_grid(demo_outputs[:, demo_idx], grid_type=1)
            seq_parts.extend([demo_in, demo_out])
        
        test_emb = self.embed_grid(test_input, grid_type=2)
        seq_parts.append(test_emb)
        
        full_seq = torch.cat(seq_parts, dim=1)
        test_start_idx = 2 * num_demos * H * W
        
        # Initialize cache with slot embeddings (Section 7.7) [ABLATION: use_slot_embeddings]
        cache = self.get_initial_cache(B, device, features)
        layer_ids = self.get_layer_ids(features).to(device)
        
        # Track auxiliary losses
        aux_data = {
            'entropy': [],
            'slot_counts': [],
            'patterns': [],
            'cache_states': [],
            'halt_probs': [],
        }
        
        # Adaptive multi-pass (Section 10.1)
        cumulative_halt = torch.zeros(B, device=device)
        total_ponder_cost = 0.0
        
        # [REFINEMENT: State tracking]
        # Initialize write counts and slot ages
        write_counts = torch.zeros(B, self.total_slots, device=device)
        slot_ages = torch.zeros(B, self.total_slots, device=device)
        
        h = full_seq
        pass_num = 1  # Initialize pass_num
        pass_logits = None  # Initialize to handle case where num_passes=0
        prev_answer = None  # [TRM] Store previous pass's answer for feedback
        
        for pass_num in range(1, num_passes + 1):
            # [TRM INSIGHT: Answer Feedback + Gradient Detachment]
            # Key idea from TRM: feed the previous answer back to help refinement
            # But detach gradients to prevent unstable gradient flow across passes
            
            if pass_num > 1 and features.use_answer_feedback and prev_answer is not None:
                # Embed the previous answer (detached to block gradients)
                # prev_answer: [B, H, W] -> embed -> [B, H*W, D]
                prev_answer_emb = self.answer_embed(prev_answer.detach())  # [B, H, W, D]
                prev_answer_emb = prev_answer_emb.view(B, H * W, -1)  # [B, H*W, D]
                
                # Get the test portion of full_seq
                test_emb_orig = full_seq[:, test_start_idx:]  # [B, H*W, D]
                
                # Gated fusion: let model learn how much to use previous answer
                combined = torch.cat([test_emb_orig, prev_answer_emb], dim=-1)
                gate = torch.sigmoid(self.answer_gate(combined))
                test_emb_refined = gate * test_emb_orig + (1 - gate) * prev_answer_emb
                
                # Replace test portion with answer-informed embedding
                h = torch.cat([full_seq[:, :test_start_idx], test_emb_refined], dim=1)
            else:
                h = full_seq
            
            for layer_idx, layer in enumerate(self.layers):
                # Pass-aware masking (Section 2.3)
                cache_mask = self.get_cache_mask(B, layer_idx, pass_num, device)
                
                # Get this layer's slot embeddings
                slot_emb = self.slot_embeddings[layer_idx]  # [K, D_cache]
                
                # Forward through layer
                h, updates = layer(
                    h, cache, slot_emb, layer_ids, cache_mask,
                    write_counts=write_counts,
                    slot_ages=slot_ages,
                    temperature=temperature, hard=hard, features=features,
                    step=step,
                    exploration_steps=config.exploration_steps,
                    exploration_start=config.exploration_start,
                )
                
                # Apply cache updates [ABLATION: use_cache]
                if features.use_cache:
                    cache = self.apply_cache_updates(cache, updates, layer_idx, threshold, features=features)
                    
                    # [REFINEMENT: Update state]
                    # Update write counts
                    start_idx = layer_idx * self.num_slots
                    end_idx = start_idx + self.num_slots
                    
                    # Sum writes for this batch (soft counts or hard counts?)
                    # Using sum of slot_probs for this layer
                    layer_slot_counts = updates['slot_probs'].sum(dim=1)  # [B, K]
                    write_counts[:, start_idx:end_idx] += layer_slot_counts
                    
                    # Update ages
                    # Increment age for all slots
                    # (In multi-pass, "time" is pass number or just step count? 
                    # Spec says "Each time step, increment age... Reset age when writing")
                    # Here we treat each layer execution as a "step" or maybe each pass?
                    # Let's increment per layer execution to be safe, or just once per pass?
                    # If we increment per layer, later layers see older cache from earlier layers.
                    # Let's increment age for ALL slots by 1.
                    slot_ages += 1
                    
                    # Reset age for slots that were written to (soft reset?)
                    # If write_weight > threshold, reset?
                    # Let's use soft reset based on write intensity if possible, or hard reset if significant write.
                    # Simple approach: if layer_slot_counts > 0.1 (arbitrary small threshold), reset.
                    written_mask = (layer_slot_counts > 1e-3).float()
                    slot_ages[:, start_idx:end_idx] = slot_ages[:, start_idx:end_idx] * (1 - written_mask)
                
                # Collect auxiliary data
                if return_aux:
                    aux_data['entropy'].append(updates['entropy'])
                    aux_data['slot_counts'].append(updates['slot_counts'])
                    aux_data['patterns'].append(updates['patterns'])
            
            # Cache-to-cache attention between passes (Section 10.2) [ABLATION: use_cache_self_attn]
            if pass_num < num_passes and features.use_cache_self_attn:
                cache = self.cache_self_attn(cache)
            
            # ACT halting decision (Section 10.1) [ABLATION: use_act_halting]
            # Or Q-Head prediction (TRM Insight)
            halt_prob = self.compute_halt_prob(cache)
            aux_data['halt_probs'].append(halt_prob)
            
            # [TRM INSIGHT: Deep Supervision]
            # Compute output for this pass - only for the TEST portion
            # Note: 'h' contains [demos..., test], we only want test output
            test_h = h[:, test_start_idx:]  # [B, H*W, D]
            pass_logits = self.output_proj(test_h).view(B, H, W, -1)  # [B, H, W, num_colors]
            
            # Store pass outputs for deep supervision loss
            if 'pass_logits' not in aux_data:
                aux_data['pass_logits'] = []
            aux_data['pass_logits'].append(pass_logits)
            
            # [TRM INSIGHT: Store answer for next pass feedback]
            # Detach to prevent gradient flow, argmax for discrete answer
            prev_answer = pass_logits.detach().argmax(dim=-1)  # [B, H, W]
            
            if features.use_act_halting and not features.use_explicit_q_head:
                # Standard ACT halting logic (stop if confident)
                remaining = 1 - cumulative_halt
                total_ponder_cost += remaining.mean()
                cumulative_halt = cumulative_halt + (1 - cumulative_halt) * halt_prob
                
                # If we are confident enough, we can stop (during inference)
                if not self.training and halt_prob.mean() > 0.5:
                    break
            
            # Store cache state for consistency loss
            if return_aux:
                aux_data['cache_states'].append(cache.clone())
            
            # Update sequence for next pass (if not using pure cache reasoning)
            # In DLSMN, h is updated layer-by-layer. For next pass, we start with original input?
            # Or do we feed the output of last pass?
            # Spec Section 2.3: "Pass 2..."
            # Usually we re-process the input sequence but with updated cache.
            # So h should be reset to initial embedding?
            # The code currently does `h = full_seq` at start of loop. Correct.
            
        # Final output is from the last executed pass
        # Safety check in case no passes were executed
        if pass_logits is None:
            test_h = h[:, test_start_idx:]  # [B, H*W, D]
            pass_logits = self.output_proj(test_h).view(B, H, W, -1)  # [B, H, W, num_colors]
        logits = pass_logits
        
        # Size prediction (optional, usually just on final state)
        # For simplicity, we just use the final state h for size prediction (test portion only)
        # Output: [B, 2, max_grid_size] for cross_entropy loss
        test_h = h[:, test_start_idx:]  # [B, H*W, D]
        size_logits_flat = self.size_proj(test_h.mean(dim=1))  # [B, max_grid_size * 2]
        size_logits = size_logits_flat.view(-1, 2, self.max_grid_size)  # [B, 2, max_grid_size]
        
        aux_info = {
            'temperature': temperature, 
            'phase': phase, 
            'ponder_cost': total_ponder_cost, 
            'num_passes': pass_num,
            'pass_logits': aux_data.get('pass_logits', []) # For Deep Supervision
        }
        
        if return_aux and aux_data['entropy']:
            aux_info['avg_entropy'] = torch.stack(aux_data['entropy']).mean()
            aux_info['slot_counts'] = aux_data['slot_counts']
            aux_info['patterns'] = aux_data['patterns']
            aux_info['cache_states'] = aux_data['cache_states']
            aux_info['halt_probs'] = aux_data['halt_probs']
        
        return logits, size_logits, cache, aux_info


# ============================================================================
# Auxiliary Losses (Section 9)
# ============================================================================

def compute_diversity_loss(slot_counts_list: list) -> torch.Tensor:
    """
    Slot Diversity Loss (Section 9.3 - Solution A):
    L_diversity = -λ_D · H(1/T · Σ_t slot_probs_t)
    
    Encourages uniform slot usage across the sequence.
    """
    if not slot_counts_list:
        return torch.tensor(0.0)
    
    # Aggregate slot counts across all layers
    total_counts = torch.stack(slot_counts_list).sum(dim=0)  # [B, K]
    
    # Normalize to distribution
    total_counts = total_counts + 1e-8
    probs = total_counts / total_counts.sum(dim=-1, keepdim=True)
    
    # Entropy (higher = more uniform = better)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    
    # We want to maximize entropy, so return negative
    return -entropy


# REMOVED: compute_balance_loss - redundant with diversity
# REMOVED: compute_sparsity_loss - redundant with Gumbel annealing  
# REMOVED: compute_consistency_loss - over-engineered, no evidence it helps


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: DLSMN_ARC, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    config: TrainingConfig,
    global_step: int = 0
) -> Tuple[float, float, float, int]:
    """
    Training epoch with simplified loss.
    
    Loss = L_task + λ·L_diversity (warmup) + λ·L_ponder (ACT) + λ·L_q_head (TRM)
    
    Removed (redundant/over-engineered):
    - L_predict: size implicit in task loss
    - L_consistency: no evidence it helps, task loss sufficient  
    - L_sparsity: redundant with Gumbel annealing
    - L_balance: redundant with diversity
    """
    model.train()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_tasks = 0
    total_tasks = 0
    cell_acc = 0.0
    task_acc = 0.0
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        output_mask = batch["output_mask"].to(device)
        output_size = batch["output_size"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with training config
        logits, size_logits, cache, aux_info = model(
            demo_inputs, demo_outputs, test_input,
            config=config,
            step=global_step,
            return_aux=True,
        )
        
        # === Task Loss ===
        
        # [TRM INSIGHT: Deep Supervision]
        # Calculate loss for every pass and average/sum them
        pass_logits_list = aux_info.get('pass_logits', [logits])
        total_color_loss = 0.0
        
        # Target preparation
        # demo_outputs: [B, num_demos, H, W] -> flatten to [B*num_demos*H*W]
        targets = test_output.view(-1)
        
        # Get dimensions from test_output
        B, H, W = test_output.shape
        
        for p_logits in pass_logits_list:
            # p_logits: [B, H, W, num_colors] -> [B*H*W, num_colors]
            p_logits_flat = p_logits.reshape(-1, model.num_colors)
            total_color_loss += F.cross_entropy(p_logits_flat, targets, ignore_index=-1)
            
        color_loss = total_color_loss / len(pass_logits_list)
        
        # Size loss (only on final pass for simplicity)
        # size_logits: [B, 2, max_grid_size]
        # test_output size? We need ground truth size.
        # Assuming test_output is [B, H, W], we can get H and W.
        true_h = output_size[:, 0] # Corrected: output_size already has H, W
        true_w = output_size[:, 1] # Corrected: output_size already has H, W
        
        size_h = F.cross_entropy(size_logits[:, 0], true_h)
        size_w = F.cross_entropy(size_logits[:, 1], true_w)
        size_loss = size_h + size_w
        
        # [TRM INSIGHT: Explicit Q-Head Loss]
        # Train halt_net to predict whether current answer is correct
        # This provides a learned "confidence" signal for when to stop refining
        q_head_loss = torch.tensor(0.0, device=device)
        features = config.features
        if features.use_explicit_q_head and 'halt_probs' in aux_info and aux_info['halt_probs']:
            halt_probs = aux_info['halt_probs']  # List of [B] tensors
            
            # Ensure we have matching lengths
            num_passes_with_probs = min(len(halt_probs), len(pass_logits_list))
            
            for i in range(num_passes_with_probs):
                p_logits = pass_logits_list[i]
                # p_logits: [B, H, W, C]
                # Use detached predictions to avoid Q-head gradient affecting main model
                preds = p_logits.detach().argmax(dim=-1)  # [B, H, W]
                
                # TRM uses full solution correctness for halting decision
                # is_correct[b] = 1 if sample b is fully correct, 0 otherwise
                is_correct = (preds == test_output).all(dim=-1).all(dim=-1).float()  # [B]
                
                # Halt prob for this pass
                p_halt = halt_probs[i]  # [B]
                
                # BCE Loss: halt_prob should be 1 if correct (should stop), 0 if wrong (should continue)
                q_head_loss += F.binary_cross_entropy(p_halt, is_correct.detach())
            
            if num_passes_with_probs > 0:
                q_head_loss = q_head_loss / num_passes_with_probs
        
        # Task loss: color + size + q_head
        # Size loss weight = 1.0 (not separate hyperparameter - it's part of the task)
        task_loss = color_loss + size_loss + config.lambda_q_head * q_head_loss
        
        # === Simplified Auxiliary Losses ===
        phase = config.get_phase(global_step)
        features = config.features
        
        # Diversity loss - ONLY during warmup to prevent slot collapse
        diversity_loss = torch.tensor(0.0, device=device)
        if phase == "warmup" and features.use_diversity_loss and 'slot_counts' in aux_info and aux_info['slot_counts']:
            diversity_loss = compute_diversity_loss(aux_info['slot_counts'])
        
        # Ponder cost - ONLY if using ACT halting
        ponder_loss = torch.tensor(0.0, device=device)
        if features.use_ponder_loss and features.use_act_halting and 'ponder_cost' in aux_info:
            ponder_loss = aux_info['ponder_cost']
        
        # === Final Loss (Simple!) ===
        loss = task_loss + config.lambda_diversity * diversity_loss + config.lambda_ponder * ponder_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.detach().item()
        global_step += 1
        
        # Accuracy metrics
        preds = logits.argmax(dim=-1)
        correct = ((preds == test_output) & ~output_mask).sum().item()
        total = (~output_mask).sum().item()
        correct_cells += correct
        total_cells += total
        
        for i in range(preds.shape[0]):
            mask = ~output_mask[i]
            if (preds[i][mask] == test_output[i][mask]).all():
                correct_tasks += 1
            total_tasks += 1
        
        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        pbar.set_postfix({
            'loss': f'{loss.detach().item():.3f}',
            'cell': f'{cell_acc:.3f}',
            'task': f'{task_acc:.3f}',
            'τ': f'{aux_info["temperature"]:.2f}',
            'phase': phase[:4],
        })
    
    return total_loss / len(dataloader), cell_acc, task_acc, global_step


@torch.no_grad()
def evaluate(
    model: DLSMN_ARC, 
    dataloader: DataLoader, 
    device: torch.device, 
    config: TrainingConfig
) -> Tuple[float, float]:
    """Evaluation with adaptive computation."""
    model.eval()
    correct_tasks = 0
    total_tasks = 0
    correct_cells = 0
    total_cells = 0
    cell_acc = 0.0
    task_acc = 0.0
    
    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for batch in pbar:
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        output_mask = batch["output_mask"].to(device)
        
        # Use hard routing during evaluation
        eval_config = TrainingConfig(
            tau_min=0.1,
            tau_start=0.1,  # Hard routing
            max_passes=config.max_passes,
        )
        
        logits, _, _, _ = model(
            demo_inputs, demo_outputs, test_input,
            config=eval_config,
            step=config.transition_steps + 1,  # Force hard phase
            return_aux=False,
        )
        
        preds = logits.argmax(dim=-1)
        
        for i in range(preds.shape[0]):
            mask = ~output_mask[i]
            correct_cells += (preds[i][mask] == test_output[i][mask]).sum().item()
            total_cells += mask.sum().item()
            if (preds[i][mask] == test_output[i][mask]).all():
                correct_tasks += 1
            total_tasks += 1
        
        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        pbar.set_postfix({'cell': f'{cell_acc:.3f}', 'task': f'{task_acc:.3f}'})
    
    return cell_acc, task_acc


# ============================================================================
# Main
# ============================================================================

def main():
    """
    Main training loop following DLSM_V0.1.md training recipe (Section 11).
    
    Training Phases (Section 8.4):
    - Phase 1 (Warm-up): Soft routing, all slots active
    - Phase 2 (Transition): Gumbel-softmax annealing
    - Phase 3 (Hard): Near-hard routing with STE
    """
    # Check if data exists
    data_dir = Path("./ARC-AGI-2/data")
    if not data_dir.exists():
        print("Downloading ARC-AGI-2 dataset...")
        os.system("git clone https://github.com/arcprize/ARC-AGI-2.git")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_dataset = ARCDataset(str(data_dir), split="training")
    eval_dataset = ARCDataset(str(data_dir), split="evaluation")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # =========================================================================
    # Feature flags for ablation study
    # Options: "full", "core_only", "single_pass", "no_gumbel", "no_slot_emb",
    #          "no_hybrid", "no_cache_attn", "no_aux_loss", "fast"
    # =========================================================================
    import sys
    preset_name = sys.argv[1] if len(sys.argv) > 1 else "fast"  # Default to fast for quick iteration
    features = FEATURE_PRESETS.get(preset_name, FEATURE_PRESETS["fast"])
    print(f"\nUsing feature preset: '{preset_name}'")
    print(f"  {features.describe()}")
    
    # Training configuration (simplified)
    config = TrainingConfig(
        tau_start=1.0,
        tau_min=0.1,
        anneal_rate=0.0003,
        warmup_steps=len(train_loader) * 5,      # 5 epochs warm-up
        transition_steps=len(train_loader) * 20,  # 20 epochs transition
        lambda_diversity=0.01,
        lambda_ponder=0.01,
        max_passes=2 if features.use_multi_pass else 1,
        features=features,
    )
    
    # Model with full DLSM_V0.1.md features
    model = DLSMN_ARC(
        num_colors=10,
        d_model=64,
        d_cache=64,
        num_layers=3,
        num_slots=8,
        num_patterns=16,
        num_heads=1,
        max_grid_size=30,
        dropout=0.1,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    print("\n" + "="*70)
    print("Training DLSMN on ARC-AGI-2 (following DLSM_V0.1.md specification)")
    print("="*70)
    print(f"Feature preset: {preset_name}")
    print(f"Training phases: warmup({config.warmup_steps}) → transition({config.transition_steps}) → hard")
    print(f"Temperature: {config.tau_start} → {config.tau_min} (anneal_rate={config.anneal_rate})")
    print(f"Max passes: {config.max_passes}")
    print("="*70 + "\n")
    
    best_task_acc = 0
    global_step = 0
    
    for epoch in range(50):
        train_loss, train_cell_acc, train_task_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, config, global_step
        )
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            eval_cell_acc, eval_task_acc = evaluate(model, eval_loader, device, config)
            marker = " ★" if eval_task_acc > best_task_acc else ""
            if eval_task_acc > best_task_acc:
                best_task_acc = eval_task_acc
            
            phase = config.get_phase(global_step)
            temp = config.get_temperature(global_step)
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Train Cell: {train_cell_acc:.3f} Task: {train_task_acc:.3f} | "
                  f"Eval Cell: {eval_cell_acc:.3f} Task: {eval_task_acc:.3f} | "
                  f"τ={temp:.2f} [{phase}]{marker}")
        else:
            phase = config.get_phase(global_step)
            temp = config.get_temperature(global_step)
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Train Cell: {train_cell_acc:.3f} Task: {train_task_acc:.3f} | "
                  f"τ={temp:.2f} [{phase}]")
        
        scheduler.step()
    
    print(f"\nBest evaluation task accuracy: {best_task_acc:.3f}")
    print("(TRM achieves ~7.8% task accuracy on ARC-AGI-2)")


if __name__ == "__main__":
    main()
