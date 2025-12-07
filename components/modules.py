import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict

from .utils import gumbel_softmax
from .config import FeatureFlags


# ============================================================================
# Memory-Efficient Attention Alternatives
# ============================================================================

def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map for linear attention."""
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    """
    Linear Attention: O(S) time and memory instead of O(S²).
    
    Instead of: softmax(QK^T / √d) @ V  [O(S²)]
    Uses:       φ(Q) @ (φ(K)^T @ V)     [O(S·d)]
    
    Where φ is ELU+1 feature map.
    This avoids materializing the S×S attention matrix.
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
        
    def forward(
        self, 
        query: torch.Tensor,  # [B, S_q, D]
        key: torch.Tensor,    # [B, S_k, D]
        value: torch.Tensor,  # [B, S_k, D]
        need_weights: bool = False,
    ) -> tuple:
        B, S_q, _ = query.shape
        _, S_k, _ = key.shape
        
        # Project and reshape to heads
        Q = self.q_proj(query).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [B, H, S, head_dim]
        
        # Apply feature map (ELU+1)
        Q = elu_feature_map(Q)
        K = elu_feature_map(K)
        
        # Linear attention: φ(Q) @ (φ(K)^T @ V)
        # First compute K^T @ V: [B, H, head_dim, head_dim]
        KV = torch.einsum('bhsd,bhse->bhde', K, V)
        
        # Then Q @ KV: [B, H, S_q, head_dim]
        out = torch.einsum('bhqd,bhde->bhqe', Q, KV)
        
        # Normalize by sum of keys (for numerical stability)
        K_sum = K.sum(dim=2, keepdim=True)  # [B, H, 1, head_dim]
        normalizer = torch.einsum('bhqd,bhkd->bhq', Q, K_sum) + 1e-6  # [B, H, S_q]
        out = out / normalizer.unsqueeze(-1)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, S_q, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out, None  # No attention weights in linear attention


class LinearCrossAttention(nn.Module):
    """
    Linear Cross-Attention for memory reading.
    Same O(S) complexity, but Q and K/V come from different sources.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = LinearAttention(d_model, num_heads, dropout)
        
    def forward(
        self,
        query: torch.Tensor,   # [B, S_q, D] - queries from input
        memory: torch.Tensor,  # [B, S_m, D] - keys/values from memory
        need_weights: bool = False,
    ) -> tuple:
        return self.attn(query, memory, memory, need_weights)

class MemoryRouter(nn.Module):
    """
    MoE-style Memory Router for sequential read/write operations.
    
    Flow:
      input x
        │
        ├─► READ PHASE (on input x):
        │     1. Should I read?      → read_gate [B,S,1]
        │     2. What slots to read? → read_slot_probs [B,S,K]
        │
        │   context = weighted_read(cache, read_slot_probs) * read_gate
        │   x_fused = fuse(x, context)
        │   y = process(x_fused)
        │
        └─► WRITE PHASE (on output y/patterns):
              3. Should I write?     → write_gate [B,S,1]    (coarse: is this cacheable?)
              4. What to write?      → write_importance [B,S,1] (fine: how valuable?)
              5. Where to write?     → write_slot_probs [B,S,K]
              
              write_score = write_gate × write_importance
              cache = update(cache, patterns, write_score, write_slot_probs)
    
    Note: READ routing uses input x, WRITE routing uses processed patterns.
    """
    
    def __init__(self, d_model: int, d_cache: int, num_slots: int):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        
        # === READ (applied to input x) ===
        # 1. Should I read from cache?
        self.read_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # 2. What slots to read from?
        self.read_slot_router = nn.Linear(d_model, num_slots)
        
        # === WRITE (applied to patterns/y) ===
        # 3. Should I write to cache? (coarse filter)
        self.write_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # 4. What to write? (importance/priority score)
        self.write_importance = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # 5. Where to write? (slot selection)
        self.write_slot_router = nn.Linear(d_model, num_slots)
        
        # Attention sharpening γ for slot selection
        self.gamma = nn.Parameter(torch.ones(1))
        
    def _read(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Read phase routing."""
        read_gate = self.read_gate(x)  # [B, S, 1] - should I read?
        read_slot_logits = self.read_slot_router(x)  # [B, S, K]
        read_slot_probs = F.softmax(read_slot_logits / temperature, dim=-1)  # what slots to read?
        
        return {
            'read_gate': read_gate,
            'read_slot_probs': read_slot_probs,
        }

    def _write(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Write phase routing."""
        B, S, D = x.shape
        K = self.num_slots
        
        write_gate = self.write_gate(x)  # [B, S, 1] - should I write?
        write_imp = self.write_importance(x)  # [B, S, 1] - how important?
        
        # Where to write? (with Gumbel-Softmax for differentiable routing)
        write_slot_logits = self.gamma * self.write_slot_router(x)  # [B, S, K]
        if hard:
            write_slot_probs = gumbel_softmax(write_slot_logits, temperature, hard=True)
        else:
            write_slot_probs = gumbel_softmax(write_slot_logits, temperature, hard=False)
        
        # Combined write score = gate × importance
        write_scores = (write_gate * write_imp).squeeze(-1)  # [B, S]
        
        return {
            'write_gate': write_gate,
            'write_scores': write_scores,
            'write_slot_probs': write_slot_probs,
        }

    def forward(
        self, 
        x: torch.Tensor,           # [B, S, D] input tokens
        temperature: float = 1.0,
        hard: bool = False,
        mode: str = 'read',
    ) -> Dict[str, torch.Tensor]:
        """
        Returns routing decisions for read OR write operations.
        
        Args:
            x: Input tensor [B, S, D]
            mode: 'read' (on input tokens) or 'write' (on patterns)
        """
        if mode == 'read':
            return self._read(x, temperature, hard)
        elif mode == 'write':
            return self._write(x, temperature, hard)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'read' or 'write'.")


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
    
    Uses LinearAttention by default for O(S) memory efficiency.
    """
    
    def __init__(self, d_cache: int, num_heads: int = 4, dropout: float = 0.1, use_linear: bool = True):
        super().__init__()
        self.use_linear = use_linear
        if use_linear:
            self.attn = LinearAttention(d_cache, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(d_cache, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_cache)
        
    def forward(self, cache: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cache: [B, L*K, D_cache] - full global cache
        Returns:
            Updated cache with inter-slot reasoning
        """
        if self.use_linear:
            attn_out, _ = self.attn(cache, cache, cache)
        else:
            attn_out, _ = self.attn(cache, cache, cache)
        return self.norm(cache + attn_out)
