from .config import FeatureFlags, TrainingConfig, FEATURE_PRESETS, SLOT_DIMS, SlotDimensions
from .dataset import ARCDataset
from .model import DLSMN_ARC
from .utils import (
    EMA, grid_to_rgb,
    make_hippo_legs_matrix, make_hippo_legs_b,
    init_hippo_legs, init_hippo_diagonal, init_hippo_embedding,
)
from .losses import compute_total_loss
from .optimizers import AdamAtan2, AdamAtan2Foreach
from .memory_controller import (
    MemoryController, 
    LayerHaltEstimator,   # Per-layer halting (hidden state stability)
    ModelHaltEstimator,   # Model-level halting (entropy-based)
)
from .unified_layer import UnifiedMemoryLayer, ComputeBlock
from .recursive_refinement_model import RecursiveRefinementModel
from .cnn_cache_model import CNNCacheModel, create_cnn_cache_model
from .decoder_cache_model import DecoderCacheModel, create_decoder_cache_model

