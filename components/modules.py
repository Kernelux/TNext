"""
DLSMN Memory Modules (v0.1.1)
=============================

This module implements the memory components for the Dual-Head Layered Selective 
Memory Network as specified in DLSM_V0.1.md and DLSM_V0.1.1.md.

Key Components:
---------------
1. LinearAttention        - O(S) memory-efficient attention (utility)
2. LinearCrossAttention   - Cross-attention variant for memory reading
3. MemoryRouter          - MoE-style bi-directional memory controller (v0.1.1 §2.3)
4. CacheSelfAttention    - Inter-slot reasoning between passes (v0.1.1 §9.1)

Architecture Overview (from DLSM_V0.1.1.md §3):
-----------------------------------------------
The Dual-Gated Cycle per layer:

    Input x
        │
        ▼
    ┌─────────────────────────────────────────┐
    │  Step 1: ACTIVE READ                     │
    │  ───────────────────                     │
    │  1. x → cache space: x_c = W_to · x     │
    │  2. Attend to global cache → context     │
    │  3. Gate: g_read = σ(MLP([x; ctx]))     │
    │  4. Fuse: x' = g·ctx + (1-g)·x          │
    └─────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────┐
    │  Step 2: COMPUTE                         │
    │  ───────────────                         │
    │  y = FFN(SelfAttn(x'))                  │
    └─────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────┐
    │  Step 3: PATTERN POOLING                 │
    │  ───────────────────────                 │
    │  P = CrossAttn(Q_patterns, y, y)        │
    │  Bottleneck: S tokens → P patterns      │
    └─────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────┐
    │  Step 4: ACTIVE WRITE (TRUE EVICTION)    │
    │  ────────────────────────────────────    │
    │  1. Novelty: P attends to global cache  │
    │  2. Score = g_write × importance        │
    │  3. Slot selection: P attends to local  │
    │     cache → attention-based routing     │
    │  4. EVICT: new content REPLACES old     │
    │     (no blending - temporal info in     │
    │      cache handles age tracking)        │
    └─────────────────────────────────────────┘

Memory Router Design (v0.1.1 §2.3):
-----------------------------------
Replaces simple "Head B" from v0.1 with bi-directional MoE control:

    READ Phase:
    - x attends to GLOBAL cache (all layers)
    - g_read gates how much context to use
    - Returns fused input for downstream compute

    WRITE Phase:
    - Patterns attend to GLOBAL cache for novelty assessment
    - g_write × importance = write score
    - Patterns attend to LOCAL cache for slot selection
    - TRUE EVICTION: winning pattern REPLACES slot content
    - Soft WTA resolves conflicts when multiple patterns target same slot

Key Design Decisions:
--------------------
1. Attention-based slot selection (not learned scorer)
   - Pattern-to-slot dot-product attention decides WHERE to write
   - High similarity = "this slot should hold my content"

2. True eviction (not blending)
   - New content fully replaces old slot content
   - Temporal/layer embeddings in cache track "age"
   - No need for explicit decay or merge

3. Soft WTA for conflict resolution
   - Multiple patterns may target same slot
   - Weight by write_scores, normalize per slot
   - Differentiable discrete selection via Gumbel-Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

from .utils import gumbel_softmax
from .config import FeatureFlags


# ============================================================================
# Rotary Position Embedding (RoPE)
# ============================================================================
# Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
#            https://arxiv.org/abs/2104.09864
# Clean explanation: https://krasserm.github.io/2022/12/13/rotary-position-embedding/
#
# Key insight: RoPE encodes RELATIVE position by rotating Q and K by angles
# proportional to ABSOLUTE position. The inner product Q·K then depends only
# on relative position (n-m), not absolute positions m and n.
#
# Benefits over learned positional embeddings:
# - Zero learnable parameters (saves ~1.6M params for our config!)
# - Generalizes to longer sequences than trained
# - Compatible with linear attention (operates on Q/K, not attention matrix)
# ============================================================================

CosSin = Tuple[torch.Tensor, torch.Tensor]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.
    
    For input [u1, u2, u3, u4, ...], returns [-u2, u1, -u4, u3, ...]
    This is needed for the RoPE rotation formula: x * cos + rotate_half(x) * sin
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to Q and K.
    
    Formula: q_rot = q * cos + rotate_half(q) * sin
    
    Args:
        q, k: [B, num_heads, S, head_dim]
        cos, sin: [S, head_dim] precomputed frequency encodings
    
    Returns:
        q_rot, k_rot with relative position information encoded
    """
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    
    # cos/sin shape: [S, head_dim] → need to broadcast to [1, 1, S, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation: x_rot = x * cos + rotate_half(x) * sin
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot.to(orig_dtype), k_rot.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Precomputes frequency position encodings up to max_seq_len.
    Uses θ_i = 10000^(-2(i-1)/d) as inverse frequencies.
    
    Zero learnable parameters - all computed from predefined function.
    """
    
    def __init__(
        self, 
        head_dim: int, 
        max_seq_len: int = 8192, 
        base: float = 10000.0,
        device=None
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Inverse frequencies: θ_i = base^(-2(i-1)/d) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        
        # Outer product: pos × inv_freq → [max_seq_len, head_dim/2]
        freqs = torch.outer(t, inv_freq)
        
        # Duplicate along frequency dimension: [max_seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin (not learnable, hence register_buffer with persistent=False)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, seq_len: Optional[int] = None) -> CosSin:
        """
        Return precomputed cos/sin, sliced to seq_len.
        
        Args:
            seq_len: If provided, slice to this length
        
        Returns:
            (cos, sin): Each [seq_len, head_dim]
        """
        if seq_len is None:
            return self.cos_cached, self.sin_cached
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# ============================================================================
# Efficient Attention Utilities
# ============================================================================

def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map for linear attention kernel.
    
    Clamp to prevent near-zero values that cause gradient explosion in division.
    """
    return (F.elu(x) + 1).clamp(min=0.01)


class LinearAttention(nn.Module):
    """
    Linear Attention with O(S) time and memory complexity.
    
    Standard attention:  softmax(QK^T / √d) @ V  → O(S²) memory
    Linear attention:    φ(Q) @ (φ(K)^T @ V)    → O(S·d) memory
    
    Where φ is the ELU+1 feature map. This avoids materializing the S×S 
    attention matrix, enabling longer sequences and larger batch sizes.
    
    RoPE compatible: Apply rotation to Q, K BEFORE feature map.
    
    Reference: "Transformers are RNNs" (Katharopoulos et al., 2020)
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights with Xavier."""
        from .utils import init_linear_xavier, init_linear_normal
        
        # Q, K, V projections: Xavier (no strong nonlinearity follows directly)
        init_linear_xavier(self.q_proj)
        init_linear_xavier(self.k_proj)
        init_linear_xavier(self.v_proj)
        
        # Output projection: small normal for residual-friendly init
        # Use smaller std for stability in deep networks
        init_linear_normal(self.out_proj, std=0.01)
        
    def forward(
        self, 
        query: torch.Tensor,   # [B, S_q, D]
        key: torch.Tensor,     # [B, S_k, D]
        value: torch.Tensor,   # [B, S_k, D]
        cos_sin: Optional[CosSin] = None,  # RoPE (cos, sin) - NEW
        need_weights: bool = False,
    ) -> tuple:
        B, S_q, _ = query.shape
        _, S_k, _ = key.shape
        
        # Project and reshape: [B, S, D] → [B, H, S, head_dim]
        Q = self.q_proj(query).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE BEFORE feature map (position info preserved through rotation)
        if cos_sin is not None:
            cos, sin = cos_sin
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        
        # Apply feature map φ(x) = ELU(x) + 1
        Q = elu_feature_map(Q)
        K = elu_feature_map(K)
        
        # Linear attention: φ(Q) @ (φ(K)^T @ V)
        # Step 1: K^T @ V → [B, H, head_dim, head_dim]
        KV = torch.einsum('bhsd,bhse->bhde', K, V)
        
        # Step 2: Q @ KV → [B, H, S_q, head_dim]
        out = torch.einsum('bhqd,bhde->bhqe', Q, KV)
        
        # Normalize by sum of keys (with better numerical stability)
        K_sum = K.sum(dim=2, keepdim=True)  # [B, H, 1, head_dim]
        normalizer = torch.einsum('bhqd,bhkd->bhq', Q, K_sum)
        # Clamp denominator to prevent division by near-zero
        normalizer = normalizer.clamp(min=1.0) + 1e-6
        out = out / normalizer.unsqueeze(-1)
        # NOTE: Activation clamping removed - AdamAtan2 handles gradient magnitude
        
        # Reshape back: [B, H, S_q, head_dim] → [B, S_q, D]
        out = out.transpose(1, 2).contiguous().view(B, S_q, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out, None  # No attention weights in linear attention


class StandardAttention(nn.Module):
    """
    Standard Scaled Dot-Product Attention with RoPE support.
    
    Uses PyTorch's scaled_dot_product_attention which leverages
    FlashAttention/Memory-Efficient Attention when available.
    
    Unlike LinearAttention, this supports:
    - RoPE (Rotary Position Embeddings) for relative position
    - Causal masking (for autoregressive)
    - Full attention matrix (O(S²) but exact)
    
    Reference: TRM's Attention + krasserm's RoPE article
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 4, 
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.dropout_p = dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Separate Q, K, V projections (cleaner for RoPE which only affects Q, K)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with truncated normal (like TRM)."""
        from .utils import init_linear_xavier, init_linear_normal
        init_linear_xavier(self.q_proj)
        init_linear_xavier(self.k_proj)
        init_linear_xavier(self.v_proj)
        init_linear_normal(self.out_proj, std=0.02)
        
    def forward(
        self, 
        x: torch.Tensor,                           # [B, S, D]
        cos_sin: Optional[CosSin] = None,          # RoPE (cos, sin) from RotaryEmbedding
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, _ = x.shape
        
        # Project: [B, S, D] → [B, S, H, head_dim] → [B, H, S, head_dim]
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K (V is position-independent)
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Efficient attention (uses FlashAttn/MemoryEfficient when available)
        dropout_p = self.dropout_p if self.training else 0.0
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=self.causal,
        )
        
        # Reshape: [B, H, S, head_dim] → [B, S, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.out_proj(attn_output)
        
        return out, None


class LinearCrossAttention(nn.Module):
    """
    Linear Cross-Attention for memory reading.
    
    Q comes from input tokens, K/V come from memory slots.
    Same O(S) complexity as LinearAttention.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = LinearAttention(d_model, num_heads, dropout)
        
    def forward(
        self,
        query: torch.Tensor,   # [B, S_q, D] - from input
        memory: torch.Tensor,  # [B, S_m, D] - from cache
        need_weights: bool = False,
    ) -> tuple:
        return self.attn(query, memory, memory, need_weights)


# ============================================================================
# MoE Memory Router (DLSM v0.1.1 §2.3)
# ============================================================================

class MemoryRouter(nn.Module):
    """
    MoE-style Bi-Directional Memory Router with ACTUAL read/write operations.
    
    From DLSM_V0.1.1.md §2.3:
    "The core controller of DLSMN v0.1.1 is the MoE Router, which replaces
    the simple 'Head B'. It outputs decisions for both Reading (Input-driven)
    and Writing (Output-driven)."
    
    This module PERFORMS the memory operations, not just routes them:
    
    READ Phase (v0.1.1 §3 Step 1-2):
    ┌────────────────────────────────────────────────────────┐
    │  1. x attends to global cache → context                │
    │  2. g_read = σ(W · [x; context])  "Should I read?"    │
    │  3. x_fused = g_read · context + (1 - g_read) · x     │
    │                                                        │
    │  Output: x_fused [B,S,D], read_gate [B,S,1]           │
    └────────────────────────────────────────────────────────┘
    
    WRITE Phase (v0.1.1 §3 Step 4) - TRUE EVICTION:
    ┌────────────────────────────────────────────────────────┐
    │  1. Pattern attends to GLOBAL cache → novelty score    │
    │  2. write_score = g_write × importance                 │
    │  3. Pattern attends to LOCAL cache → slot selection    │
    │     (attention similarity determines target slot)      │
    │  4. EVICT: slot content REPLACED (not blended)         │
    │     Soft WTA resolves multi-pattern conflicts          │
    │                                                        │
    │  Output: new_local_cache [B,K,D], write_scores [B,P]  │
    └────────────────────────────────────────────────────────┘
    
    Key Design:
    -----------
    - READ uses GLOBAL cache, returns FUSED output
    - WRITE assesses novelty via GLOBAL, selects slot via LOCAL
    - Slot selection via attention (pattern @ local_cache^T)
    - TRUE EVICTION: new content fully replaces old (no merge)
    - Temporal/layer embeddings in cache handle age tracking
    """
    
    def __init__(self, d_model: int, d_cache: int, num_slots: int, 
                 max_recurrent_steps: int = 4, max_passes: int = 4):
        """
        Args:
            d_model:   Hidden dimension of the model
            d_cache:   Dimension of cache entries  
            num_slots: Number of slots per layer (K in the spec)
            max_recurrent_steps: Max recurrent steps per layer (for scaling)
            max_passes: Max passes through the model (for scaling)
        """
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        
        # NOTE: Manual scaling removed - AdamAtan2 handles gradient magnitude
        
        # === Projections ===
        self.to_cache = nn.Linear(d_model, d_cache)
        self.from_cache = nn.Linear(d_cache, d_model)
        
        # Note: Pre-projection LayerNorms removed - AdamAtan2 handles gradient magnitude
        # and removing them preserves more information in the representations
        
        # === READ Controls ===
        # g_read: "Should this token read from memory?"
        self.read_gate_mlp = nn.Sequential(
            nn.LayerNorm(d_cache * 2),
            nn.Linear(d_cache * 2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # === WRITE Controls ===
        # g_write: "Should this token be written to memory?"
        self.write_gate_mlp = nn.Sequential(
            nn.LayerNorm(d_model + d_cache),
            nn.Linear(d_model + d_cache, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # s_imp: "How important/novel is this pattern?" (for WTA when competing)
        self.importance_net = nn.Sequential(
            nn.Linear(d_model + d_cache, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # Note: External input norms removed - MLPs have internal LayerNorms
        # and AdamAtan2 handles gradient magnitude
        
        # Note: Slot selection uses direct attention (pattern @ local_cache^T)
        # No learned eviction_scorer needed - attention similarity determines
        # which slot's content is most "replaceable" by the new pattern
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MemoryRouter weights with appropriate strategies."""
        from .utils import (
            init_linear_xavier, init_linear_normal, init_sequential, init_gate_bias
        )
        
        # Space projections: Xavier for general purpose
        init_linear_xavier(self.to_cache)
        init_linear_xavier(self.from_cache)
        
        # Gate networks: Initialize to start ~0.5 (neutral)
        init_sequential(self.read_gate_mlp, final_is_gate=False, gate_bias=0.0)
        init_sequential(self.write_gate_mlp, final_is_gate=False, gate_bias=0.0)
        init_sequential(self.importance_net, final_is_gate=True, gate_bias=0.0)
    
    def _attend_to_cache(
        self, 
        query: torch.Tensor,   # [B, S, D_cache]
        cache: torch.Tensor,   # [B, N, D_cache]
    ) -> tuple:
        """
        Scaled dot-product attention to cache.
        Returns (context, attention_weights).
        """
        scores = torch.matmul(query, cache.transpose(-2, -1))  # [B, S, N]
        scores = scores / math.sqrt(self.d_cache)
        # Clamp scores to prevent softmax saturation (gradient vanishing)
        scores = scores.clamp(min=-20.0, max=20.0)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, cache)  # [B, S, D_cache]
        return context, attn_weights
    
    def read(
        self,
        x: torch.Tensor,                       # [B, S, D_model]
        cache: Optional[torch.Tensor] = None,  # [B, L*K, D_cache]
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Active Read Phase following DLSM spec (v0.1.2).
        
        Three-stage decision process (mirrors write):
        1. "Should I read?" - Hard gate per token (do I need context?)
        2. "What to read?" - Attention over cache slots  
        3. "How to fuse?" - Additive fusion with gating
        
        Args:
            x:     Input tokens [B, S, D_model]
            cache: Global cache [B, L*K, D_cache]
            temperature: For soft gating during training
            hard: Force hard decisions (for inference)
        
        Returns:
            x_fused:    [B, S, D_model] - output (fused or original)
            read_gate:  [B, S, 1] - gate values
            context:    [B, S, D_model] - retrieved context
        """
        B, S, D = x.shape
        device = x.device
        
        # Project input to cache space (no pre-norm - AdamAtan2 handles gradients)
        x_cache = self.to_cache(x)  # [B, S, D_cache]
        
        # === STEP 1: "Should I read?" - Hard Gate ===
        if cache is not None and cache.shape[1] > 0:
            # Full gradient flow - AdamAtan2 handles gradient magnitude
            context_cache, attn_weights = self._attend_to_cache(x_cache, cache)
        else:
            context_cache = torch.zeros_like(x_cache)
            attn_weights = None
        
        # Gate decision: does this token need cache context?
        # Input: [token_repr, cache_context] → gate logit
        combined = torch.cat([x_cache, context_cache], dim=-1)  # [B, S, 2*D_cache]
        # MLP has internal LayerNorm as first layer
        read_gate_logit = self.read_gate_mlp(combined)  # [B, S, 1]
        
        # Soft gate with STE for hard decisions
        # No noise - let the network learn clean gate signals
        soft_gate = torch.sigmoid(read_gate_logit)
        if hard or not self.training:
            read_gate = (soft_gate > 0.5).float()
        else:
            # STE: hard forward, soft backward
            hard_gate = (soft_gate > 0.5).float()
            read_gate = hard_gate - soft_gate.detach() + soft_gate
        
        # === STEP 2: "What to read?" - Already computed above ===
        # context_cache contains the attended values from cache
        
        # === STEP 3: "How to fuse?" - Gated additive fusion ===
        # Project context back to model space (no pre-norm - AdamAtan2 handles gradients)
        context = self.from_cache(context_cache)  # [B, S, D_model]
        
        # Additive fusion: x_fused = x + read_gate * context
        # - If read_gate = 1: x + context (use cache)
        # - If read_gate = 0: x (skip cache)
        # Gradients always flow through x
        x_fused = x + read_gate * context
        
        return {
            'x_fused': x_fused,
            'read_gate': read_gate,
            'context': context,
            'attn_weights': attn_weights,
        }
    
    def write(
        self,
        tokens: torch.Tensor,                       # [B, T, D_model] - candidate tokens
        cache: Optional[torch.Tensor] = None,       # [B, L*K, D_cache] global
        local_cache: Optional[torch.Tensor] = None, # [B, K, D_cache] this layer
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Active Write Phase following DLSM spec (v0.1.1 §3, Step 4).
        
        Three-stage decision process:
        1. "Should I write?" - Hard gate per token (binary decision)
        2. "What to write?" - Importance scoring among gated tokens
        3. "Where to write?" - Slot selection via attention
        
        Key constraint: At most K tokens write (one per slot).
        Tokens compete via importance-weighted soft-WTA.
        
        Args:
            tokens:      Candidate tokens [B, T, D_model]
            cache:       Global cache [B, L*K, D_cache] for novelty assessment
            local_cache: This layer's slots [B, K, D_cache]
            temperature: Gumbel-softmax temperature
            hard:        Use hard decisions (for inference)
        
        Returns:
            new_local_cache: [B, K, D_cache] - UPDATED cache
            write_scores:    [B, T] - gate × importance
            write_gate:      [B, T, 1] - which tokens passed gate
            slot_probs:      [B, T, K] - slot assignment probabilities
        """
        B, T, D = tokens.shape
        K = self.num_slots
        device = tokens.device
        
        # Project tokens to cache space (no pre-norm - AdamAtan2 handles gradients)
        t_cache = self.to_cache(tokens)  # [B, T, D_cache]
        
        # === STEP 1: "Should I write?" - Hard Gate ===
        # Assess novelty by comparing to global cache
        if cache is not None and cache.shape[1] > 0:
            # Full gradient flow - AdamAtan2 handles gradient magnitude
            context_cache, _ = self._attend_to_cache(t_cache, cache)
        else:
            context_cache = torch.zeros_like(t_cache)
        
        # Gate decision: is this token worth writing?
        if tokens.shape[1] != context_cache.shape[1]:
            # Handle potential shape mismatch (e.g. if _attend_to_cache behavior changes)
            if context_cache.shape[1] < tokens.shape[1]:
                # Pad context_cache with zeros
                pad_len = tokens.shape[1] - context_cache.shape[1]
                zeros = torch.zeros(B, pad_len, context_cache.shape[2], device=device)
                context_cache = torch.cat([context_cache, zeros], dim=1)
            elif context_cache.shape[1] > tokens.shape[1]:
                # Slice context_cache
                context_cache = context_cache[:, :tokens.shape[1], :]
                
        combined = torch.cat([tokens, context_cache], dim=-1)  # [B, T, D + D_cache]
        # MLP has internal LayerNorm as first layer
        write_gate_logit = self.write_gate_mlp(combined)  # [B, T, 1]
        
        # Soft gate with STE for hard decisions
        # No noise - let the network learn clean gate signals
        soft_gate = torch.sigmoid(write_gate_logit)
        if hard or not self.training:
            write_gate = (soft_gate > 0.5).float()
        else:
            # STE: hard forward, soft backward
            hard_gate = (soft_gate > 0.5).float()
            write_gate = hard_gate - soft_gate.detach() + soft_gate
        
        # === STEP 2: "What to write?" - Importance Scoring ===
        # Only tokens that pass the gate are considered
        importance = self.importance_net(combined)  # [B, T, 1]
        
        # Masked importance: tokens not passing gate have 0 importance
        masked_importance = importance * write_gate  # [B, T, 1]
        
        # Combined write score for logging
        write_scores = masked_importance.squeeze(-1)  # [B, T]
        
        # === STEP 3: "Where to write?" - Slot Selection ===
        if local_cache is not None:
            # Full gradient flow - AdamAtan2 handles gradient magnitude
            slot_logits = torch.matmul(t_cache, local_cache.transpose(-2, -1))
            slot_logits = slot_logits / math.sqrt(self.d_cache)
            slot_logits = slot_logits.clamp(min=-10.0, max=10.0)  # Numerical stability
        else:
            slot_logits = torch.zeros(B, T, K, device=device)
            local_cache = torch.zeros(B, K, self.d_cache, device=device)
        
        # Gumbel-Softmax for differentiable slot selection
        if hard or not self.training:
            slot_probs = gumbel_softmax(slot_logits, temperature, hard=True)
        else:
            slot_probs = gumbel_softmax(slot_logits, temperature, hard=False)
        
        # === WRITE AGGREGATION ===
        # Mask slot_probs by write gate: only gated tokens contribute
        masked_slot_probs = slot_probs * write_gate  # [B, T, K]
        
        # Weight by importance for soft-WTA (most important token wins)
        importance_weights = masked_importance.clamp(min=1e-6)  # [B, T, 1]
        weighted_tokens = t_cache * importance_weights  # [B, T, D_cache]
        
        # Aggregate to slots: [B, K, T] @ [B, T, D] → [B, K, D]
        slot_probs_t = masked_slot_probs.transpose(1, 2)  # [B, K, T]
        slot_writes = torch.matmul(slot_probs_t, weighted_tokens)  # [B, K, D_cache]
        
        # Normalize by total weight per slot
        slot_weights = torch.matmul(slot_probs_t, importance_weights.squeeze(-1).unsqueeze(-1))
        # Clamp denominator to prevent division by near-zero (numerical stability)
        slot_weights_safe = slot_weights.clamp(min=0.1)
        slot_writes = slot_writes / slot_weights_safe
        # NOTE: Output clamping removed - AdamAtan2 handles gradient magnitude
        
        # Which slots received writes?
        has_writes = (slot_weights.squeeze(-1) > 0.1).unsqueeze(-1).float()  # Match clamp threshold
        
        # TRUE EVICTION: written slots get new content, others unchanged
        new_local_cache = has_writes * slot_writes + (1 - has_writes) * local_cache
        
        return {
            'new_local_cache': new_local_cache,
            'write_scores': write_scores,
            'write_gate': write_gate,
            'slot_probs': slot_probs,
            'num_gated': write_gate.sum(dim=1).squeeze(-1),  # [B] tokens passing gate
        }
    
    def forward(
        self, 
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        local_cache: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = False,
        mode: str = 'read',
    ) -> Dict[str, torch.Tensor]:
        """
        Unified interface for read/write operations.
        
        Args:
            x:           Input [B, S, D] for read, patterns [B, P, D] for write
            cache:       Global cache [B, L*K, D_cache]
            local_cache: This layer's slots [B, K, D_cache] (write only)
            temperature: Softmax temperature
            hard:        Use hard slot selection (write only)
            mode:        'read' or 'write'
        
        Returns:
            READ mode:  {'x_fused', 'read_gate', 'context', 'attn_weights'}
            WRITE mode: {'new_local_cache', 'write_scores', 'write_gate', 'slot_probs'}
        """
        if mode == 'read':
            return self.read(x, cache)
        elif mode == 'write':
            return self.write(x, cache, local_cache, temperature, hard)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'read' or 'write'.")



# ============================================================================
# Cache-to-Cache Attention (DLSM v0.1.1 §9.1)
# ============================================================================

class CacheSelfAttention(nn.Module):
    """
    Cache-to-Cache Self-Attention for inter-slot reasoning.
    
    From DLSM_V0.1.1.md §9.1:
    "To enable 'sleep-like' consolidation, we can run passes of Inter-Cache 
    Attention between computation passes: C' = SelfAttn(C)
    
    This allows the memory to deduplicate entries and propagate graph-like 
    relationships (A->B, B->C implies A->C) without standard layer compute."
    
    Use Cases:
    - Symbolic reasoning across memory slots
    - Graph propagation / transitive inference  
    - Memory consolidation / deduplication
    
    Pass-Aware Masking (v0.1.1 §2.3):
    - After each layer writes, cache slots share context
    - Pass 1: Layer N's slots can only attend to layers 0..N-1 (causal)
    - Pass 2+: All slots attend to all slots (full global context)
    
    Uses standard attention (not linear) when masking is required.
    """
    
    def __init__(
        self, 
        d_cache: int, 
        num_heads: int = 4, 
        dropout: float = 0.1, 
        use_linear: bool = True
    ):
        super().__init__()
        self.d_cache = d_cache
        self.num_heads = num_heads
        self.use_linear = use_linear
        
        if use_linear:
            self.attn = LinearAttention(d_cache, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(
                d_cache, num_heads, dropout=dropout, batch_first=True
            )
        
        # Standard attention for masked operation (linear attention doesn't support masks)
        self.masked_attn = nn.MultiheadAttention(
            d_cache, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_cache)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize CacheSelfAttention weights."""
        from .utils import init_layer_norm
        
        # LayerNorm
        init_layer_norm(self.norm)
        
        # Note: LinearAttention and nn.MultiheadAttention handle their own init
        # nn.MultiheadAttention uses xavier_uniform_ by default
        
    def forward(
        self, 
        cache: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply self-attention over cache slots with optional masking.
        
        Args:
            cache:     [B, L*K, D_cache] - full global cache
            attn_mask: [L*K, L*K] - attention mask where True = BLOCKED
                       If None, uses unmasked (linear) attention
        
        Returns:
            Updated cache with inter-slot information flow
        """
        if attn_mask is not None:
            # Use standard attention when masking is required
            # Convert bool mask to float: True (blocked) → -inf
            float_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))
            attn_out, _ = self.masked_attn(cache, cache, cache, attn_mask=float_mask)
        elif self.use_linear:
            attn_out, _ = self.attn(cache, cache, cache)
        else:
            attn_out, _ = self.attn(cache, cache, cache)
        
        # Residual connection without LayerNorm - AdamAtan2 handles gradient magnitude
        # and this preserves cache magnitude information for slot diversity
        return cache + attn_out