import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Weight Initialization Utilities
# ============================================================================

def init_linear_kaiming(layer: nn.Linear, nonlinearity: str = 'relu'):
    """
    Kaiming/He initialization for linear layers followed by ReLU/SiLU/GELU.
    
    Good for: FFN layers, attention projections with nonlinear activations.
    """
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity=nonlinearity)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_linear_xavier(layer: nn.Linear):
    """
    Xavier/Glorot initialization for linear layers.
    
    Good for: Layers followed by Sigmoid/Tanh, or general purpose.
    """
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_linear_normal(layer: nn.Linear, std: float = 0.02):
    """
    Normal initialization with small std for stable training.
    
    Good for: Output projections, embeddings, residual branches.
    """
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_embedding(embedding: nn.Embedding, std: float = 0.02):
    """
    Initialize embedding with small normal values.
    """
    nn.init.normal_(embedding.weight, mean=0.0, std=std)


def init_layer_norm(layer_norm: nn.LayerNorm):
    """
    Initialize LayerNorm to identity (weight=1, bias=0).
    """
    nn.init.ones_(layer_norm.weight)
    nn.init.zeros_(layer_norm.bias)


def init_gate_bias(layer: nn.Linear, initial_value: float = 0.0):
    """
    Initialize gate output bias to control initial gate behavior.
    
    - bias=0: gates start at sigmoid(0) = 0.5 (50% gating)
    - bias=-2: gates start at sigmoid(-2) ≈ 0.12 (mostly closed)
    - bias=+2: gates start at sigmoid(+2) ≈ 0.88 (mostly open)
    """
    if layer.bias is not None:
        nn.init.constant_(layer.bias, initial_value)


def init_sequential(seq: nn.Sequential, final_is_gate: bool = False, gate_bias: float = 0.0):
    """
    Initialize a Sequential module with appropriate methods per layer type.
    
    Args:
        seq: nn.Sequential to initialize
        final_is_gate: If True, the final Linear outputs a gate (sigmoid)
        gate_bias: Initial bias for gate output (if final_is_gate=True)
    """
    layers = list(seq.children())
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            is_last = (i == len(layers) - 1) or (i == len(layers) - 2 and isinstance(layers[-1], nn.Sigmoid))
            
            if is_last and final_is_gate:
                # Gate output: Xavier + custom bias
                init_linear_xavier(layer)
                init_gate_bias(layer, gate_bias)
            elif i < len(layers) - 1:
                # Hidden layer followed by activation
                next_layer = layers[i + 1] if i + 1 < len(layers) else None
                if isinstance(next_layer, (nn.ReLU, nn.SiLU, nn.GELU)):
                    init_linear_kaiming(layer, nonlinearity='relu')  # SiLU/GELU similar to ReLU
                else:
                    init_linear_xavier(layer)
            else:
                # Final non-gate layer
                init_linear_normal(layer, std=0.02)
        elif isinstance(layer, nn.LayerNorm):
            init_layer_norm(layer)


# ============================================================================
# Gumbel-Softmax
# ============================================================================

def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Gumbel-Softmax with optional hard mode (Section 8.2).
    
    During training with high temperature: soft routing (all slots receive updates)
    As temperature → 0: converges to hard one-hot selection
    """
    # Clamp logits to prevent numerical issues in softmax
    logits = logits.clamp(min=-20.0, max=20.0)
    
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-8, max=1.0-1e-8)) + 1e-8)
    y_soft = F.softmax((logits + gumbel_noise) / max(temperature, 0.01), dim=-1)
    
    if hard:
        # Straight-through: hard forward, soft backward
        idx = y_soft.argmax(dim=-1)
        y_hard = F.one_hot(idx, logits.shape[-1]).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


def grid_to_rgb(grid_idx: torch.Tensor) -> torch.Tensor:
    """
    Convert ARC grid indices (0-9) to RGB tensor for visualization.
    
    Args:
        grid_idx: [H, W] int tensor with values 0-9
        
    Returns:
        [3, H*4, W*4] float tensor (0.0-1.0) upsmpaled 4x
    """
    device = grid_idx.device
    
    # Simple ARC palette
    palette = torch.tensor([
        [0, 0, 0],       # 0: Black
        [0, 116, 217],   # 1: Blue
        [255, 65, 54],   # 2: Red
        [46, 204, 64],   # 3: Green
        [255, 220, 0],   # 4: Yellow
        [170, 170, 170], # 5: Grey
        [240, 18, 190],  # 6: Fuschia
        [255, 133, 27],  # 7: Orange
        [127, 219, 255], # 8: Teal
        [135, 12, 37],   # 9: Maroon
    ], dtype=torch.uint8, device=device)
    
    # Clamp to 0-9 for safety
    grid_clamped = grid_idx.clamp(0, 9)
    rgb = palette[grid_clamped].permute(2, 0, 1) # [3, H, W]
    
    # Upscale by 4x for visibility
    return F.interpolate(rgb.unsqueeze(0).float(), scale_factor=4, mode='nearest').squeeze(0) / 255.0


class EMA:
    """
    Exponential Moving Average of model weights (TRM Section 4.7).
    
    Helps prevent overfitting and improves stability on small datasets.
    At test time, use ema_model for better generalization.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights after each optimization step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
