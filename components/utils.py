import torch
import torch.nn as nn
import torch.nn.functional as F

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
