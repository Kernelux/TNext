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
    
    Uses log-softmax based normalization for numerical stability instead of raw clamping.
    """
    # Normalize logits using log_softmax for numerical stability (better than raw clamping)
    # This centers the logits around 0 and prevents extreme values
    log_probs = F.log_softmax(logits, dim=-1)  # Numerically stable
    
    # Sample Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
    # Use clamping only for the uniform samples to avoid log(0)
    uniform = torch.rand_like(logits).clamp(min=1e-10, max=1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(uniform))
    
    # Add noise to log_probs (equivalent to adding to normalized logits)
    # Then apply temperature scaling and softmax
    y_soft = F.softmax((log_probs + gumbel_noise) / max(temperature, 0.01), dim=-1)
    
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


# ============================================================================
# HiPPO Initialization
# ============================================================================
# 
# HiPPO (High-order Polynomial Projection Operators) provides mathematically
# grounded initialization for state space models and memory systems.
# 
# Key insight: Initialize state transition matrices to compress input history
# optimally, with recent inputs weighted more than older ones.
#
# Reference: "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (2020)
# Also used in S4, S4D, and Mamba architectures.
# ============================================================================

def make_hippo_legs_matrix(N: int) -> torch.Tensor:
    """
    Create the HiPPO-LegS (Scaled Legendre) matrix.
    
    This is the "HiPPO matrix" used in S4 and Mamba. It creates a state
    transition matrix that optimally compresses the input history using
    Legendre polynomial basis functions.
    
    Properties:
    - Lower triangular (causal)
    - Diagonal elements: n+1 (decay rate increases with state index)
    - Below diagonal: sqrt((2n+1)(2k+1)) for n > k
    
    The matrix captures all past inputs with recent inputs weighted more
    strongly (decaying memory with Legendre polynomial basis).
    
    Args:
        N: State dimension (number of coefficients to track)
        
    Returns:
        A: [N, N] HiPPO-LegS matrix
    """
    # Build the A matrix
    # A_nk = (2n+1)^0.5 * (2k+1)^0.5  if n > k
    #      = n + 1                    if n = k  
    #      = 0                        if n < k
    
    A = torch.zeros(N, N)
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = math.sqrt((2*n + 1) * (2*k + 1))
            elif n == k:
                A[n, k] = n + 1
            # else: 0 (upper triangular part)
    
    return A


def make_hippo_legs_b(N: int) -> torch.Tensor:
    """
    Create the HiPPO-LegS input matrix B.
    
    B_n = sqrt(2n + 1)
    
    This weights how much each Legendre coefficient is influenced by new input.
    Higher-order coefficients (capturing faster variations) get larger weights.
    
    Args:
        N: State dimension
        
    Returns:
        B: [N, 1] input weight vector
    """
    B = torch.zeros(N, 1)
    for n in range(N):
        B[n, 0] = math.sqrt(2*n + 1)
    return B


def init_hippo_legs(weight: torch.Tensor, dt: float = 1.0):
    """
    Initialize a weight matrix using discretized HiPPO-LegS.
    
    For a state update: h_{k+1} = A_bar @ h_k + B_bar @ x_k
    
    Using forward Euler discretization:
        A_bar = I - dt * A
        B_bar = dt * B
    
    This initialization is ideal for:
    - Cache/memory slot state transitions
    - Recurrent state matrices
    - Any matrix that should "remember" history optimally
    
    Args:
        weight: [N, N] weight tensor to initialize in-place
        dt: Discretization step size (smaller = finer granularity)
    """
    N = weight.shape[0]
    assert weight.shape[1] == N, "HiPPO init requires square matrix"
    
    A = make_hippo_legs_matrix(N)
    
    # Forward Euler discretization: A_bar = I - dt * A
    # Note: We use negative A because HiPPO defines dh/dt = -A @ h + B @ x
    A_bar = torch.eye(N) - dt * A
    
    with torch.no_grad():
        weight.copy_(A_bar)


def init_hippo_diagonal(weight: torch.Tensor, dt: float = 1.0):
    """
    Initialize a weight matrix using HiPPO diagonal approximation (S4D-style).
    
    S4D showed that using just the diagonal of HiPPO works surprisingly well
    and is much more efficient. This is what Mamba uses.
    
    Diagonal elements: -1, -2, -3, ..., -N (after discretization: 1-dt, 1-2dt, ...)
    
    Args:
        weight: [N, N] weight tensor to initialize (only diagonal set)
        dt: Discretization step size
    """
    N = weight.shape[0]
    
    with torch.no_grad():
        weight.zero_()
        for n in range(N):
            # Diagonal of HiPPO-LegS is -(n+1)
            # Discretized: 1 - dt * (n+1)
            weight[n, n] = 1.0 - dt * (n + 1)


def init_hippo_embedding(embedding: torch.Tensor, dt: float = 0.1):
    """
    Initialize embeddings using HiPPO-inspired basis functions.
    
    Each row of the embedding is initialized to capture different
    "frequencies" of the input, similar to how Legendre polynomials
    capture different scales of variation.
    
    This is useful for:
    - Slot embeddings (memory cache initialization)
    - Position encodings
    - Any embedding that should represent temporal/sequential structure
    
    Args:
        embedding: [num_embeddings, dim] tensor to initialize
        dt: Effective "time scale" for the basis functions
    """
    num_embeddings, dim = embedding.shape
    
    # Use HiPPO B vector scaled appropriately
    # Each embedding row corresponds to a different "time" in the past
    B = make_hippo_legs_b(dim)  # [dim, 1]
    
    with torch.no_grad():
        for i in range(num_embeddings):
            # Scale factor: later embeddings (higher i) represent "older" memories
            # Older memories get exponentially decayed
            scale = math.exp(-dt * i)
            # Each embedding is a scaled version of the B vector plus small noise
            embedding[i] = B.squeeze() * scale + torch.randn(dim) * 0.01

