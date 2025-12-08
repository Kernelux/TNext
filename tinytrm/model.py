"""
TinyTRM Model for ARC-AGI-2
===========================

Implementation based on "Less is More: Recursive Reasoning with Tiny Networks"
(Jolicoeur-Martineau, 2025) - arXiv:2510.04871
Official repo: https://github.com/SamsungSAILMontreal/TinyRecursiveModels

Key insights from the paper:
1. Single tiny network (2 layers) beats HRM's two 4-layer networks
2. Two features: z_H (answer), z_L (latent reasoning) 
3. Deep recursion: L_cycles latent recursions, H_cycles times (H-1 without gradients)
4. Deep supervision: reuse (z_H, z_L) across ACT supervision steps
5. ACT halting: Q-head predicts halt/continue with Q-learning
6. EMA: prevents overfitting on small data

TRM achieves 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with only 7M params.

Our enhancement: Add slot-based memory to help with pattern storage/retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from pydantic import BaseModel


class TRMConfig(BaseModel):
    """Configuration for TinyTRM - aligned with official repo."""
    # Batch/sequence
    batch_size: int = 32
    seq_len: int = 900  # 30x30 max
    
    # Model dimensions
    vocab_size: int = 11  # 10 colors + padding
    hidden_size: int = 512  # Paper: 512
    expansion: float = 4.0  # SwiGLU expansion (scaled by 2/3 internally)
    num_heads: int = 8
    
    # Architecture
    L_layers: int = 2  # Number of layers in reasoning module (paper: 2)
    H_cycles: int = 3  # Outer cycles (T), H-1 without grad (paper: 3)
    L_cycles: int = 4  # Inner latent recursions (n) (paper: 4-6 for ARC)
    
    # Halting (ACT)
    halt_max_steps: int = 16  # Max deep supervision steps
    halt_exploration_prob: float = 0.1  # Epsilon-greedy exploration
    
    # Slot memory (our enhancement)
    use_slot_memory: bool = True
    num_slots: int = 16
    d_slot: int = 128
    
    # Training
    use_mlp_t: bool = False  # Use MLP on L instead of attention (paper uses attention for ARC)
    pos_encodings: str = "rope"  # rope, learned, or none
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "float32"  # or bfloat16 for H100
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Grid settings
    max_grid_size: int = 30
    num_puzzle_identifiers: int = 1000  # For puzzle embeddings (ARC task IDs)
    puzzle_emb_ndim: int = 512  # Puzzle embedding dimension
    puzzle_emb_len: int = 16  # Number of puzzle embedding tokens


@dataclass
class TRMInnerCarry:
    """Carry state for recursive model - holds z_H (answer) and z_L (latent)."""
    z_H: torch.Tensor  # [B, L, D] - answer embedding
    z_L: torch.Tensor  # [B, L, D] - latent reasoning embedding


@dataclass 
class TRMCarry:
    """Full carry state including ACT tracking."""
    inner_carry: TRMInnerCarry
    steps: torch.Tensor  # [B] - number of steps taken
    halted: torch.Tensor  # [B] - whether sequence has halted
    current_data: Dict[str, torch.Tensor]


def rms_norm(x: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    """RMS normalization (functional)."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + variance_epsilon)


def _find_multiple(a: int, b: int) -> int:
    """Make a a multiple of b, rounding up."""
    return (-(a // -b)) * b


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated normal initialization."""
    nn.init.trunc_normal_(tensor, std=std)
    return tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation (Shazeer, 2020) - matches official repo.
    Includes the 2/3 scaling and multiple-of-256 alignment used in the reference code.
    """
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        # Reference logic: Scale by 2/3 to keep params comparable to FFN, round to 256
        intermediate_size = _find_multiple(int(hidden_size * expansion * 2 / 3), 256)
        
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) - dynamic length."""
    def __init__(self, dim: int, max_position_embeddings: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos/sin
        self._cached_seq_len = 0
        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
    
    def forward(self, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) for rotary embeddings."""
        if seq_len is None:
            seq_len = self.max_position_embeddings
        
        # Use cache if available and sufficient
        if self._cached_seq_len >= seq_len and self._cached_cos is not None:
            return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
        
        # Compute for at least max(seq_len, max_position_embeddings)
        compute_len = max(seq_len, self.max_position_embeddings)
        t = torch.arange(compute_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache
        self._cached_seq_len = compute_len
        self._cached_cos = emb.cos()
        self._cached_sin = emb.sin()
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    cos = cos[:q.shape[2], :].unsqueeze(0).unsqueeze(0)  # [1, 1, L, D]
    sin = sin[:q.shape[2], :].unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention - matches official repo."""
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int,
        head_dim: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.causal = causal
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat KV for GQA if needed
        if self.num_key_value_heads != self.num_heads:
            repeat = self.num_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if self.causal:
            mask = torch.triu(torch.ones(L, L, device=hidden_states.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)


class TRMBlock(nn.Module):
    """
    TRM Block - matches official repo structure.
    Post-norm architecture with RMS normalization.
    Can use either attention or MLP for sequence mixing.
    """
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps
        
        if config.use_mlp_t:
            # MLP for sequence mixing (transpose)
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + config.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            # Self-attention for sequence mixing
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        
        # Channel MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward with post-norm (official repo style)."""
        # Sequence mixing
        if self.config.use_mlp_t:
            hidden_states = hidden_states.transpose(1, 2)  # [B, D, L]
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)  # [B, L, D]
        else:
            out = self.self_attn(hidden_states, cos_sin=cos_sin)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        
        # Channel mixing
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        
        return hidden_states


class TRMReasoningModule(nn.Module):
    """
    Stack of TRM blocks with input injection - matches official repo.
    """
    def __init__(self, layers: List[TRMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward with input injection (additive)."""
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)
        return hidden_states


class MixerBlock(nn.Module):
    """Legacy MLP-Mixer block - kept for backward compatibility."""
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len * 4),
            nn.SiLU(),
            nn.Linear(seq_len * 4, seq_len),
            nn.Dropout(dropout),
        )
        self.channel_mix = SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(-1, -2)
        y = self.token_mix(y).transpose(-1, -2)
        x = x + self.dropout(y)
        x = x + self.dropout(self.channel_mix(self.norm2(x)))
        return x


class SlotMemory(nn.Module):
    """
    Enhanced Slot-based Memory for TinyTRM.
    
    Combines ideas from DLSMN's MemoryRouter and SelectionHead:
    - Multi-slot competitive storage
    - Gated read/write with importance scoring
    - Attention sharpening γ for slot selection precision
    - Slot usage tracking to prevent overwriting important content
    - Optional cache self-attention for inter-slot reasoning
    
    Operations:
    - Read: Query slots with learned routing + content similarity
    - Write: Gated update with importance scoring and slot competition
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_slots: int, 
        d_slot: int,
        use_cache_self_attn: bool = True,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.d_slot = d_slot
        self.d_model = d_model
        
        # Slot embeddings (learned initialization)
        self.slot_embed = nn.Parameter(torch.randn(num_slots, d_slot) * 0.02)
        
        # === READ PHASE ===
        # 1. Should I read? (read gate)
        self.read_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # 2. What slots to read from? (learned routing)
        self.read_slot_router = nn.Linear(d_model, num_slots)
        
        # 3. Content-based read (query-key matching)
        self.read_query = nn.Linear(d_model, d_slot)
        self.read_key = nn.Linear(d_slot, d_slot)
        self.read_value = nn.Linear(d_slot, d_model)
        
        # === WRITE PHASE ===
        # 1. Should I write? (write gate - coarse filter)
        self.write_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # 2. How important is this? (importance score)
        self.write_importance = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # 3. Where to write? (slot selection)
        self.write_slot_router = nn.Linear(d_model, num_slots)
        self.write_content = nn.Linear(d_model, d_slot)
        
        # Attention sharpening γ (controls slot selection precision)
        # High γ → sharper selection, Low γ → softer exploration
        self.gamma = nn.Parameter(torch.ones(1))
        
        # === FUSION ===
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        # === CACHE SELF-ATTENTION (inter-slot reasoning) ===
        self.use_cache_self_attn = use_cache_self_attn
        if use_cache_self_attn:
            self.cache_attn = nn.MultiheadAttention(
                d_slot, num_heads, dropout=0.1, batch_first=True
            )
            self.cache_norm = nn.LayerNorm(d_slot)
        
        # Slot usage tracking (soft, decays over time)
        self.register_buffer('usage_decay', torch.tensor(0.99))
    
    def read(
        self, 
        x: torch.Tensor,       # [B, L, D]
        slots: torch.Tensor,   # [B, K, d_slot]
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from slots with gated attention.
        
        Returns: (context, read_weights)
        """
        B, L, D = x.shape
        
        # Should I read?
        read_gate = self.read_gate(x)  # [B, L, 1]
        
        # Learned routing (what slots to read)
        route_logits = self.read_slot_router(x) * self.gamma  # [B, L, K]
        
        # Content-based routing (query-key similarity)
        q = self.read_query(x)  # [B, L, d_slot]
        k = self.read_key(slots)  # [B, K, d_slot]
        content_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_slot)
        
        # Combine learned + content routing (hybrid)
        combined_logits = 0.7 * route_logits + 0.3 * content_logits
        read_weights = F.softmax(combined_logits / temperature, dim=-1)  # [B, L, K]
        
        # Read values
        v = self.read_value(slots)  # [B, K, D]
        context = torch.matmul(read_weights, v)  # [B, L, D]
        
        # Apply read gate
        context = context * read_gate
        
        return context, read_weights
    
    def write(
        self, 
        x: torch.Tensor,       # [B, L, D] - content to potentially write
        slots: torch.Tensor,   # [B, K, d_slot]
        usage: torch.Tensor,   # [B, K] - slot usage scores
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write to slots with gated importance scoring.
        
        Returns: (updated_slots, updated_usage)
        """
        B, L, D = x.shape
        K = self.num_slots
        
        # Pool over sequence
        x_pool = x.mean(dim=1)  # [B, D]
        
        # Should I write? (coarse gate)
        write_gate = self.write_gate(x_pool)  # [B, 1]
        
        # How important? (fine-grained)
        importance = self.write_importance(x_pool)  # [B, 1]
        
        # Combined write score
        write_score = write_gate * importance  # [B, 1]
        
        # Where to write? (slot selection with usage penalty)
        slot_logits = self.write_slot_router(x_pool) * self.gamma  # [B, K]
        
        # Penalize highly-used slots (encourage spreading)
        slot_logits = slot_logits - 0.5 * usage
        
        write_probs = F.softmax(slot_logits / temperature, dim=-1)  # [B, K]
        
        # Content to write
        content = self.write_content(x_pool)  # [B, d_slot]
        
        # Weighted update: write_score [B, 1] * write_probs [B, K] = [B, K]
        update_weight = write_score * write_probs  # [B, K] via broadcast
        update_weight = update_weight.unsqueeze(-1)  # [B, K, 1]
        content = content.unsqueeze(1).expand(-1, K, -1)  # [B, K, d_slot]
        
        # Soft update: blend old and new
        new_slots = (1 - update_weight) * slots + update_weight * content
        
        # Update usage (decay old + add new)
        # write_score: [B, 1], write_probs: [B, K] -> broadcast correctly
        new_usage = self.usage_decay * usage + write_score * write_probs
        
        return new_slots, new_usage
    
    def cache_reasoning(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Inter-slot self-attention for reasoning between stored patterns.
        Allows slots to "talk to each other" and propagate information.
        """
        if not self.use_cache_self_attn:
            return slots
        
        attn_out, _ = self.cache_attn(slots, slots, slots)
        return self.cache_norm(slots + attn_out)
    
    def forward(
        self, 
        x: torch.Tensor,               # [B, L, D]
        slots: torch.Tensor,           # [B, K, d_slot]
        usage: Optional[torch.Tensor] = None,  # [B, K]
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full read-then-write cycle.
        
        Returns: (x_with_memory, updated_slots, updated_usage)
        """
        B = x.shape[0]
        
        # Initialize usage if not provided
        if usage is None:
            usage = torch.zeros(B, self.num_slots, device=x.device)
        
        # Inter-slot reasoning first (optional)
        slots = self.cache_reasoning(slots)
        
        # Read from slots
        context, read_weights = self.read(x, slots, temperature)
        
        # Fuse with input
        gate = self.fusion_gate(torch.cat([x, context], dim=-1))
        x_fused = gate * context + (1 - gate) * x
        
        # Write to slots
        new_slots, new_usage = self.write(x_fused, slots, usage, temperature)
        
        return x_fused, new_slots, new_usage


class TinyTRM_Inner(nn.Module):
    """
    Inner model for TinyTRM - matches official repo structure.
    
    This handles the core recursion logic:
    - Input embedding
    - L_level reasoning module (2 layers)
    - H/L cycle recursion with gradient truncation
    - Output heads (LM + Q)
    """
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # === I/O Embeddings ===
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        # Token embedding (colors)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, std=embed_init_std)
        
        # Output heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = nn.Linear(config.hidden_size, 2, bias=True)  # halt, continue
        
        # Puzzle embedding length
        self.puzzle_emb_len = config.puzzle_emb_len
        
        # Puzzle embeddings (task identifier)
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = nn.Embedding(
                config.num_puzzle_identifiers, 
                config.puzzle_emb_ndim
            )
            nn.init.zeros_(self.puzzle_emb.weight)
        
        # Position encodings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + config.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = nn.Embedding(
                config.seq_len + config.puzzle_emb_len, 
                config.hidden_size
            )
            nn.init.normal_(self.embed_pos.weight, std=embed_init_std)
        
        # === Reasoning Module (L_layers blocks) ===
        self.L_level = TRMReasoningModule(
            layers=[TRMBlock(config) for _ in range(config.L_layers)]
        )
        
        # === Slot Memory (our enhancement) ===
        if config.use_slot_memory:
            self.slot_memory = SlotMemory(
                config.hidden_size,
                config.num_slots,
                config.d_slot,
            )
        
        # === Initial states for z_H and z_L ===
        self.H_init = nn.Parameter(
            trunc_normal_init_(torch.empty(config.hidden_size), std=1)
        )
        self.L_init = nn.Parameter(
            trunc_normal_init_(torch.empty(config.hidden_size), std=1)
        )
        
        # Initialize Q-head to near-zero for faster bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
    
    def _input_embeddings(
        self, 
        inputs: torch.Tensor,  # [B, L]
        puzzle_identifiers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create input embeddings with puzzle prefix."""
        B, L = inputs.shape
        
        # Token embeddings
        token_emb = self.embed_tokens(inputs) * self.embed_scale  # [B, L, D]
        
        # Add puzzle embedding prefix
        if hasattr(self, 'puzzle_emb') and puzzle_identifiers is not None:
            puzzle_emb = self.puzzle_emb(puzzle_identifiers)  # [B, puzzle_emb_ndim]
            # Reshape to [B, puzzle_emb_len, D]
            puzzle_emb = puzzle_emb.view(B, self.puzzle_emb_len, -1)
            # Pad if needed
            if puzzle_emb.shape[-1] < self.config.hidden_size:
                pad = torch.zeros(
                    B, self.puzzle_emb_len, 
                    self.config.hidden_size - puzzle_emb.shape[-1],
                    device=inputs.device, dtype=puzzle_emb.dtype
                )
                puzzle_emb = torch.cat([puzzle_emb, pad], dim=-1)
            token_emb = torch.cat([puzzle_emb, token_emb], dim=1)
        
        # Add position embeddings
        if hasattr(self, 'embed_pos'):
            pos_ids = torch.arange(token_emb.shape[1], device=inputs.device)
            token_emb = token_emb + self.embed_pos(pos_ids)
        
        return token_emb
    
    def empty_carry(self, batch_size: int, seq_len: Optional[int] = None) -> TRMInnerCarry:
        """Create empty carry state."""
        L = seq_len if seq_len is not None else (self.config.seq_len + self.puzzle_emb_len)
        return TRMInnerCarry(
            z_H=torch.empty(batch_size, L, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, L, self.config.hidden_size, dtype=self.forward_dtype),
        )
    
    def reset_carry(
        self, 
        reset_flag: torch.Tensor, 
        carry: TRMInnerCarry,
    ) -> TRMInnerCarry:
        """Reset carry for halted sequences."""
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )
    
    def forward(
        self,
        carry: TRMInnerCarry,
        batch: Dict[str, torch.Tensor],
        slots: Optional[torch.Tensor] = None,
        usage: Optional[torch.Tensor] = None,
    ) -> Tuple[TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with H_cycles × L_cycles recursion.
        
        Cache Usage Pattern:
        1. LATENT PHASE: z_L reads/writes to slots during reasoning
        2. REFINEMENT PHASE: z_H reads from slots before final output
        
        This makes the cache GLOBAL - information stored during latent
        reasoning is available for answer refinement.
        
        Returns: (new_carry, logits, (q_halt, q_continue), new_slots, new_usage)
        """
        # Input encoding (this may prepend puzzle_emb tokens)
        input_embeddings = self._input_embeddings(
            batch["inputs"], 
            batch.get("puzzle_identifiers"),
        )
        
        # Get current states
        z_H, z_L = carry.z_H, carry.z_L
        B = z_H.shape[0]
        actual_seq_len = input_embeddings.shape[1]
        
        # Resize carry states if sequence length changed
        if z_H.shape[1] != actual_seq_len:
            # Reinitialize with correct sequence length
            z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(B, actual_seq_len, -1).clone()
            z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(B, actual_seq_len, -1).clone()
        
        # Get rotary embeddings if using RoPE - use actual embedding length
        cos_sin = self.rotary_emb(actual_seq_len) if hasattr(self, 'rotary_emb') else None
        
        # === Deep recursion: H_cycles-1 without grad, then 1 with grad ===
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                # L_cycles of latent updates (REASONING PHASE)
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
                    
                    # Slot memory: latent writes patterns it discovers
                    if self.config.use_slot_memory and slots is not None:
                        z_L, slots, usage = self.slot_memory(z_L, slots, usage)
                
                # Update answer z_H from z_L
                z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)
        
        # === Last cycle with gradients ===
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
            
            # Slot memory during latent reasoning
            if self.config.use_slot_memory and slots is not None:
                z_L, slots, usage = self.slot_memory(z_L, slots, usage)
        
        # === REFINEMENT: z_H reads from accumulated slot memory ===
        # This is the key change - answer gets to read what latent discovered
        if self.config.use_slot_memory and slots is not None:
            # Read-only from slots (no write during refinement)
            context, _ = self.slot_memory.read(z_H, slots)
            gate = self.slot_memory.fusion_gate(torch.cat([z_H, context], dim=-1))
            z_H = gate * context + (1 - gate) * z_H
        
        # Final answer update with latent knowledge
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)
        
        # === Outputs ===
        # New carry (detached for next step)
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        
        # LM output (skip puzzle prefix)
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        
        # Q-head (from first position, like official repo)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), slots, usage


class TinyTRM(nn.Module):
    """
    Tiny Recursive Model for ARC-AGI - ACT wrapper.
    
    Matches official repo structure with our slot memory enhancement.
    
    Architecture:
    - Inner model handles recursion
    - Outer ACT wrapper handles halting logic
    - Deep supervision over multiple steps
    """
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.inner = TinyTRM_Inner(config)
    
    @property
    def puzzle_emb(self):
        """Access puzzle embeddings from inner model."""
        return self.inner.puzzle_emb if hasattr(self.inner, 'puzzle_emb') else None
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRMCarry:
        """Create initial carry state for a batch."""
        batch_size = batch["inputs"].shape[0]
        seq_len = batch["inputs"].shape[1]
        
        # Account for puzzle embedding prefix if it will be added
        if hasattr(self.inner, 'puzzle_emb') and self.config.puzzle_emb_ndim > 0:
            seq_len = seq_len + self.config.puzzle_emb_len
        
        return TRMCarry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Start halted to trigger reset
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )
    
    def forward(
        self,
        carry: TRMCarry,
        batch: Dict[str, torch.Tensor],
        slots: Optional[torch.Tensor] = None,
        usage: Optional[torch.Tensor] = None,
    ) -> Tuple[TRMCarry, Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with ACT halting logic.
        
        Returns: (new_carry, outputs_dict, new_slots, new_usage)
        """
        device = batch["inputs"].device
        
        # Reset carry for halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        # Reset step counter for halted sequences
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        
        # Update current data for halted sequences
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }
        
        # Initialize slots if needed
        if self.config.use_slot_memory and slots is None:
            slots = self.inner.slot_memory.slot_embed.unsqueeze(0).expand(
                batch["inputs"].shape[0], -1, -1
            ).clone()
            usage = torch.zeros(batch["inputs"].shape[0], self.config.num_slots, device=device)
        
        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), new_slots, new_usage = self.inner(
            new_inner_carry, new_current_data, slots, usage
        )
        
        # Increment steps
        new_steps = new_steps + 1
        
        # Halting decision (epsilon-greedy during training)
        is_last_step = new_steps >= self.config.halt_max_steps
        
        if self.training:
            # Epsilon-greedy exploration
            explore = torch.rand(carry.halted.shape, device=device) < self.config.halt_exploration_prob
            halt_prob = torch.sigmoid(q_halt_logits)
            halt_decision = torch.where(
                explore,
                torch.rand_like(halt_prob) < 0.5,
                halt_prob > 0.5
            )
            halted = halt_decision | is_last_step
        else:
            # Greedy at inference
            halted = (torch.sigmoid(q_halt_logits) > 0.5) | is_last_step
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "halted": halted,
            "steps": new_steps,
        }
        
        # Q-learning target (for training)
        if self.training and not is_last_step.all():
            with torch.no_grad():
                # Get next step Q values
                _, _, (next_q_halt, next_q_continue), _, _ = self.inner(
                    new_inner_carry, new_current_data, new_slots, new_usage
                )
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt,
                        torch.maximum(next_q_halt, next_q_continue)
                    )
                )
        
        new_carry = TRMCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )
        
        return new_carry, outputs, new_slots, new_usage


# === Simplified interface for ARC tasks ===

class TinyTRMForARC(nn.Module):
    """
    Simplified TinyTRM interface for ARC tasks.
    
    Handles:
    - Grid embedding (colors + positions)
    - Task formatting (demos + test)
    - Deep supervision training loop
    """
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.model = TinyTRM(config)
        
        # Additional embeddings for ARC
        self.color_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_h = nn.Embedding(config.max_grid_size, config.hidden_size // 2)
        self.pos_w = nn.Embedding(config.max_grid_size, config.hidden_size // 2)
        self.type_embed = nn.Embedding(4, config.hidden_size)
        
        # Size prediction
        self.size_head = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, config.max_grid_size * 2),
        )
    
    def embed_grid(
        self,
        grid: torch.Tensor,  # [B, H, W]
        grid_type: int,
    ) -> torch.Tensor:
        """Embed a single grid."""
        B, H, W = grid.shape
        device = grid.device
        
        # Color embedding
        color_emb = self.color_embed(grid)  # [B, H, W, D]
        
        # Position embeddings
        h_pos = torch.arange(H, device=device)
        w_pos = torch.arange(W, device=device)
        h_emb = self.pos_h(h_pos)
        w_emb = self.pos_w(w_pos)
        
        pos_emb = torch.cat([
            h_emb.unsqueeze(1).expand(-1, W, -1),
            w_emb.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1)
        
        type_emb = self.type_embed(torch.tensor(grid_type, device=device))
        
        emb = color_emb + pos_emb.unsqueeze(0) + type_emb
        return emb.view(B, H * W, -1)
    
    def encode_task(
        self,
        demo_inputs: torch.Tensor,   # [B, N_demo, H, W]
        demo_outputs: torch.Tensor,  # [B, N_demo, H, W]
        test_input: torch.Tensor,    # [B, H, W]
    ) -> torch.Tensor:
        """Encode full ARC task to token sequence."""
        B, N_demo, H, W = demo_inputs.shape
        
        # Flatten all grids to sequence
        tokens = []
        for i in range(N_demo):
            tokens.append(demo_inputs[:, i].view(B, -1))
            tokens.append(demo_outputs[:, i].view(B, -1))
        tokens.append(test_input.view(B, -1))
        
        return torch.cat(tokens, dim=1)  # [B, total_tokens]
    
    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        test_input: torch.Tensor,
        n_supervision: Optional[int] = None,
        return_all_steps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with deep supervision.
        """
        B = demo_inputs.shape[0]
        H, W = test_input.shape[1], test_input.shape[2]
        device = demo_inputs.device
        
        n_sup = n_supervision or self.config.halt_max_steps
        
        # Encode task to token sequence
        inputs = self.encode_task(demo_inputs, demo_outputs, test_input)
        
        # Create batch dict
        batch = {
            "inputs": inputs,
            "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device=device),
        }
        
        # Initialize carry
        carry = self.model.initial_carry(batch)
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        carry.steps = carry.steps.to(device)
        carry.halted = carry.halted.to(device)
        
        # Initialize slots and usage
        slots = None
        usage = None
        if self.config.use_slot_memory:
            slots = self.model.inner.slot_memory.slot_embed.unsqueeze(0).expand(B, -1, -1).clone().to(device)
            usage = torch.zeros(B, self.config.num_slots, device=device)
        
        # Deep supervision loop
        all_logits = []
        all_q_halt = []
        all_q_continue = []
        
        for step in range(n_sup):
            carry, outputs, slots, usage = self.model(carry, batch, slots, usage)
            
            # Reshape logits to grid
            logits = outputs["logits"]  # [B, L, vocab]
            output_len = H * W
            grid_logits = logits[:, -output_len:, :].view(B, H, W, -1)
            
            all_logits.append(grid_logits)
            all_q_halt.append(outputs["q_halt_logits"])
            all_q_continue.append(outputs["q_continue_logits"])
            
            # GLOBAL CACHE: Don't detach during training to allow 
            # long-term credit assignment. Only detach at inference
            # to save memory.
            if not self.training:
                if slots is not None:
                    slots = slots.detach()
                if usage is not None:
                    usage = usage.detach()
        
        # Size prediction from final state
        final_hidden = carry.inner_carry.z_H.mean(dim=1)
        size_logits = self.size_head(final_hidden)
        
        result = {
            "logits": all_logits[-1],
            "size_logits": size_logits,
            "q_halt": torch.stack(all_q_halt, dim=1),
            "q_continue": torch.stack(all_q_continue, dim=1),
        }
        
        if return_all_steps:
            result["all_logits"] = all_logits
        
        return result


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
