from dataclasses import dataclass, field
from typing import Optional, Dict
import math


# ============================================================================
# Slot Dimension Constants
# ============================================================================
# Centralized dimensions for cache slot metadata structure.
# These MUST be consistent across MemoryController, UnifiedMemoryLayer, and RecursiveRefinementModel.

@dataclass(frozen=True)
class SlotDimensions:
    """
    Slot metadata dimensions - frozen to prevent accidental modification.
    
    Cache slot structure: [content (d_cache) | confidence (1) | temporal (16)]
    Temporal breakdown: [layer_id (8) | iter_embed (4) | pass_embed (4)]
    
    Total metadata: d_meta = 1 + 8 + 4 + 4 = 17
    """
    d_layer_embed: int = 8   # Layer identity embedding dimension
    d_iter_embed: int = 4    # Iteration embedding dimension  
    d_pass_embed: int = 4    # Pass embedding dimension
    
    @property
    def d_temporal(self) -> int:
        """Total temporal embedding dimension."""
        return self.d_layer_embed + self.d_iter_embed + self.d_pass_embed  # 16
    
    @property
    def d_meta(self) -> int:
        """Total metadata dimension (confidence + temporal)."""
        return 1 + self.d_temporal  # 17
    
    def d_slot(self, d_cache: int) -> int:
        """Total slot dimension for a given cache dimension."""
        return d_cache + self.d_meta


# Global instance - use this everywhere instead of hardcoding 17, 8, 4, etc.
SLOT_DIMS = SlotDimensions()


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

    IMPROVEMENTS (optional enhancements):
    - use_gumbel_softmax: Differentiable slot selection with annealing [Section 8.2]
    - use_slot_embeddings: Learned concept anchors for cache init [Section 7.7]
    - use_layer_id: Layer-ID embeddings for representational separation [Section 7.6]
    - use_cache_self_attn: Cache-to-cache attention after writes [Section 10.2]
    - use_act_halting: Adaptive Computation Time for model-level early stopping [Section 10.1]
    - use_layer_act: Per-layer ACT iterations
    - use_gated_fusion: Gated combination of input and cache context [Section 3.2]
    - use_linear_attention: O(S) linear attention instead of O(S²)

    AUXILIARY LOSSES:
    - use_diversity_loss: Encourage uniform slot usage [Section 9.3]
    - use_ponder_cost: Penalize excessive computation

    TRM INSIGHTS (from TinyRecursiveModels paper):
    - use_deep_supervision: Train on every pass with weighted loss
    - use_answer_feedback: Feed previous pass's answer back
    """

    # === CORE FEATURES (disable to break architecture) ===
    use_cache: bool = True              # Without this, it's just a transformer
    use_selection_head: bool = True     # Without this, no selective caching
    use_multi_pass: bool = True         # Without this, single-pass only

    # === ARCHITECTURE IMPROVEMENTS ===
    use_gumbel_softmax: bool = True     # False = argmax routing (non-differentiable)
    use_slot_embeddings: bool = True    # False = zero-init cache
    use_layer_id: bool = True           # False = no layer separation in cache
    use_cache_self_attn: bool = True    # Cache-to-cache attention after writes
    use_act_halting: bool = True        # Model-level adaptive halting
    use_layer_act: bool = True          # Per-layer ACT iterations
    use_gated_fusion: bool = True       # Gated fusion of input and cache context
    use_linear_attention: bool = True   # O(S) linear attention

    # === AUXILIARY LOSSES ===
    use_diversity_loss: bool = True     # Prevent slot collapse
    use_ponder_cost: bool = True        # Penalize excessive passes
    ponder_cost_weight: float = 0.01    # Weight of ponder cost

    # === TRM INSIGHTS ===
    use_deep_supervision: bool = True   # Train on every pass with weighted loss
    use_answer_feedback: bool = True    # Feed previous pass's answer back
    deep_supervision_decay: float = 0.8 # Weight decay for earlier passes

    # === POSITION ENCODING ===
    pos_encoding: str = "rope"          # "rope" (zero params), "learned", or "none"

    def describe(self) -> str:
        """Return a string describing enabled features."""
        core = []
        if self.use_cache: core.append("cache")
        if self.use_selection_head: core.append("selection")
        if self.use_multi_pass: core.append("multi-pass")

        arch = []
        if self.use_gumbel_softmax: arch.append("gumbel")
        if self.use_slot_embeddings: arch.append("slot-emb")
        if self.use_layer_id: arch.append("layer-id")
        if self.use_cache_self_attn: arch.append("cache-attn")
        if self.use_act_halting: arch.append("model-ACT")
        if self.use_layer_act: arch.append("layer-ACT")
        if self.use_gated_fusion: arch.append("gated")
        if self.use_linear_attention: arch.append("linear-attn")

        losses = []
        if self.use_diversity_loss: losses.append("div")
        if self.use_ponder_cost: losses.append("ponder")

        trm = []
        if self.use_deep_supervision: trm.append("deep-sup")
        if self.use_answer_feedback: trm.append("ans-fb")

        return (f"Core: [{', '.join(core)}] | "
                f"Arch: [{', '.join(arch)}] | "
                f"Losses: [{', '.join(losses)}] | "
                f"TRM: [{', '.join(trm)}]")


# ============================================================================
# Feature Presets
# ============================================================================

FEATURE_PRESETS = {
    # Full model with all features
    "full": FeatureFlags(),

    # Fast-full: All features enabled (model hyperparams adjusted in main())
    "fast_full": FeatureFlags(),

    # Core only - minimal DLSMN
    "core_only": FeatureFlags(
        use_gumbel_softmax=False,
        use_slot_embeddings=False,
        use_layer_id=False,
        use_cache_self_attn=False,
        use_act_halting=False,
        use_layer_act=False,
        use_gated_fusion=False,
        use_diversity_loss=False,
        use_ponder_cost=False,
    ),

    # No multi-pass (single pass baseline)
    "single_pass": FeatureFlags(
        use_multi_pass=False,
        use_cache_self_attn=False,
        use_act_halting=False,
    ),

    # Fast training (minimal overhead)
    "fast": FeatureFlags(
        use_multi_pass=False,
        use_cache_self_attn=False,
        use_act_halting=False,
    ),

    # TRM-style: Model passes + Memory Cache, NO layer iterations
    "trm": FeatureFlags(
        use_cache=True,
        use_selection_head=True,
        use_multi_pass=True,
        use_act_halting=True,        # Model-level halting: YES
        use_layer_act=False,         # Layer iterations: NO (TRM-style)
        use_gumbel_softmax=True,
        use_slot_embeddings=True,
        use_layer_id=True,
        use_cache_self_attn=True,
        use_gated_fusion=True,
        use_linear_attention=True,
        use_deep_supervision=True,
        use_answer_feedback=True,
        use_diversity_loss=True,
        use_ponder_cost=True,
        pos_encoding="rope",
    ),

    # TRM-minimal: Minimal features for comparison
    "trm_minimal": FeatureFlags(
        use_cache=True,
        use_selection_head=True,
        use_multi_pass=True,
        use_act_halting=True,
        use_layer_act=False,
        use_gumbel_softmax=True,
        use_slot_embeddings=True,
        use_layer_id=False,
        use_cache_self_attn=False,
        use_gated_fusion=False,
        use_linear_attention=True,
        use_deep_supervision=True,
        use_answer_feedback=True,
        use_diversity_loss=False,
        use_ponder_cost=True,
        pos_encoding="rope",
    ),
}


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Training configuration.

    Philosophy:
    - Start at max capacity (full passes, full recurrent steps)
    - Efficiency losses push model to use fewer when it can
    """
    # === TEMPERATURE ANNEALING (Gumbel-Softmax) ===
    tau_start: float = 1.0       # Initial temperature (soft routing)
    tau_min: float = 0.5         # Final temperature
    anneal_rate: float = 0.0003  # Annealing rate

    # === LOSS WEIGHTS ===
    # Task losses
    lambda_diversity: float = 0.01       # Slot diversity
    lambda_q_head: float = 0.1           # Q-head correctness predictor
    lambda_step_efficiency: float = 0.5  # Layer-level step efficiency
    
    # Confidence losses
    lambda_confidence_calibration: float = 1.0   # Confidence vs correctness alignment
    lambda_halt_prediction: float = 0.3          # Halt predicts sequence correctness
    lambda_confidence_monotonicity: float = 0.1  # Later passes = higher confidence
    
    # Gate regularization
    lambda_gate_polar: float = 0.1       # Push gates away from 0.5
    lambda_gate_sparsity: float = 0.1    # Target gate activation ratios
    lambda_feedback_polar: float = 0.1   # Feedback gate polarization

    # === GATE THRESHOLDS ===
    use_fixed_gate_threshold: bool = True   # True = fixed 0.5, False = learned
    gate_threshold: float = 0.5             # Fixed threshold value

    # === HALT THRESHOLDS ===
    confidence_threshold: float = 0.8       # Stop when confidence > this

    # === COMPUTE BUDGET ===
    max_passes: int = 6                     # Maximum model-level passes
    max_recurrent_steps: int = 4            # Maximum per-layer iterations

    # === CURRICULUM LEARNING ===
    use_curriculum: bool = False
    curriculum_warmup_steps: int = 135
    curriculum_increase_every: int = 135
    
    # === WARMUP (force max passes to stabilize Q-halt) ===
    warmup_epochs: int = 1  # Epochs to force max passes (no early halt)

    # === EMA ===
    use_ema: bool = True
    ema_decay: float = 0.999

    # === FEATURE FLAGS ===
    features: FeatureFlags = field(default_factory=FeatureFlags)

    def get_temperature(self, step: int) -> float:
        """Gumbel-Softmax temperature schedule."""
        return max(self.tau_min, self.tau_start * math.exp(-self.anneal_rate * step))
    
    def get_curriculum_passes(self, step: int) -> int:
        """Get effective passes based on curriculum (1 → max_passes)."""
        if not self.use_curriculum:
            return self.max_passes
        if step < self.curriculum_warmup_steps:
            return 1
        steps_after = step - self.curriculum_warmup_steps
        increases = steps_after // self.curriculum_increase_every
        return min(1 + increases, self.max_passes)
    
    def get_curriculum_recurrence(self, step: int) -> int:
        """Get effective recurrence based on curriculum (1 → max_recurrent_steps)."""
        if not self.use_curriculum:
            return self.max_recurrent_steps
        if step < self.curriculum_warmup_steps:
            return 1
        steps_after = step - self.curriculum_warmup_steps
        increases = steps_after // self.curriculum_increase_every
        return min(1 + increases, self.max_recurrent_steps)