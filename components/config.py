from dataclasses import dataclass, field
from typing import Optional, Dict
import math

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
    use_act_halting: bool = True        # False = fixed number of passes (model-level)
    use_layer_act: bool = True          # False = fixed recurrent steps (layer-level)
    use_gated_fusion: bool = True       # False = additive fusion
    use_moe_memory: bool = True         # MoE-style memory routing (should I read? should I write?)
    
    # Memory optimization
    detach_cache_between_passes: bool = True  # Detach cache between passes to reduce memory
    use_linear_attention: bool = True   # Use O(S) linear attention instead of O(S²) softmax attention

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
    use_deep_supervision: bool = False   # Train on every pass (memory intensive - disabled by default)
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

    # Fast-full: All features enabled, but model hyperparams adjusted for speed in main()
    # Use this when you want all features but need faster iteration
    "fast_full": FeatureFlags(),

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

    # Runpod optimized - balanced for ARC evaluation with memory constraints
    "runpod": FeatureFlags(
        use_diversity_loss=True,     # Keep diversity to prevent slot collapse
        use_ponder_loss=True,        # Keep ponder for efficiency
        use_explicit_q_head=True,    # Keep Q-head for halting prediction
        use_answer_feedback=True,    # Keep answer feedback mechanism
    ),
}


# ============================================================================
# Training Configuration (Section 8.4, 9.5)
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Training configuration - simplified direct training.

    Philosophy:
    - Start at max capacity (full passes, full recurrent steps)
    - Efficiency losses push model to use fewer when it can
    - No warmup/transition phases - train directly
    """
    # Temperature for Gumbel-Softmax routing
    tau_start: float = 1.0       # Initial temperature (soft routing)
    tau_min: float = 0.5         # Final temperature (keep softer to prevent NaN)
    anneal_rate: float = 0.0003  # Slower annealing for stability

    # Loss weights
    lambda_diversity: float = 0.01   # Prevents slot collapse
    lambda_ponder: float = 0.05      # Penalizes excessive passes (increased - key for efficiency)
    lambda_q_head: float = 0.1       # Q-head correctness predictor
    lambda_step_efficiency: float = 0.05  # Layer-level step efficiency (increased)

    # Importance threshold
    write_threshold: float = 0.5

    # Hybrid routing
    alpha_learned: float = 1.0  # 1.0 = pure learned routing

    # Compute budget (model starts at max, learns to reduce)
    max_passes: int = 8           # Maximum thinking passes
    max_recurrent_steps: int = 4  # Maximum refinement steps per layer

    # EMA for training stability
    use_ema: bool = True
    ema_decay: float = 0.999

    # Feature flags for ablation
    features: FeatureFlags = field(default_factory=FeatureFlags)

    def get_temperature(self, step: int) -> float:
        """Gumbel-Softmax temperature schedule."""
        return max(self.tau_min, self.tau_start * math.exp(-self.anneal_rate * step))
    def get_pass_mode(self, step: int, training: bool = True) -> str:
        """
        Pass mode for adaptive computation.

        Simplified: Always adaptive - let efficiency losses drive reduction.
        Model starts at max capacity and learns to reduce.
        """
        # Always use adaptive with ponder penalty
        # The ponder loss will push model to use fewer passes when it can
        return 'adaptive'
