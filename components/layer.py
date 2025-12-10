"""
DLSMN Layer (v0.1.2 - Token-Level Caching)
==========================================

Implements the Dual-Gated Cycle with TOKEN-LEVEL caching:

    Input x ─────────────────────────────────────────────────
        │
        ▼
    ┌─────────────────────────────────────────┐
    │  Step 1: ACTIVE READ                     │
    │  ───────────────────                     │
    │  • "Should I read?" → g_read gate       │
    │  • "Which tokens to read?" → attention  │
    │  • x' = x + g_read · context            │
    │                                          │
    │  Output: x_fused (additive fusion)       │
    └─────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────┐
    │  Step 2: COMPUTE                         │
    │  ───────────────                         │
    │  • attn = SelfAttn(x_fused)             │
    │  • y = FFN(LayerNorm(x + attn))         │
    │                                          │
    │  Output: y [B, S, D]                     │
    └─────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────┐
    │  Step 3: ACTIVE WRITE (3-stage)         │
    │  ───────────────────────────────────────│
    │  For each token in y:                   │
    │                                          │
    │  (a) "Should I write?" → hard gate      │
    │      Binary decision per token          │
    │                                          │
    │  (b) "What to write?" → importance      │
    │      Score gated tokens by priority     │
    │                                          │
    │  (c) "Where to write?" → slot attn      │
    │      Route to K slots via soft-WTA      │
    │                                          │
    │  Output: updated local_cache [B, K, D]   │
    └─────────────────────────────────────────┘
        │
        ▼
    Output y, updated global cache ──────────────────────────

Key Design (Token-Level):
-------------------------
• NO pattern pooling - cache stores ACTUAL token representations
• 3-stage write process per DLSM spec:
  1. "Should I write?" (hard gate) - filters unworthy tokens
  2. "What to write?" (importance) - prioritizes among gated tokens  
  3. "Where to write?" (slot selection) - soft-WTA for slot assignment
• Importance-weighted soft-WTA ensures most important tokens win slots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

from .config import FeatureFlags
from .modules import MemoryRouter, LinearAttention, CosSin


class DLSMNLayer(nn.Module):
    """
    DLSMN Layer implementing the Dual-Gated Cycle (v0.1.1 §3).
    
    The layer delegates memory operations to MemoryRouter:
    - READ: Router queries cache, computes gate, returns fused x
    - WRITE: Router evaluates patterns, selects slots, returns updated cache
    """
    
    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        d_cache: int,
        num_slots: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_recurrent_steps: int = 4,
        max_passes: int = 4,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.num_layers = num_layers
        self.max_recurrent_steps = max_recurrent_steps
        self.max_passes = max_passes
        
        # MoE Memory Router handles READ and WRITE
        self.memory_router = MemoryRouter(
            d_model, d_cache, num_slots,
            max_recurrent_steps=max_recurrent_steps,
            max_passes=max_passes
        )
        
        # Computation block (Pre-LN transformer)
        self.self_attn = LinearAttention(d_model, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Max tokens to consider for writing (efficiency limit)
        # The router's gate decides which actually write
        self.max_write_tokens = num_slots * 4  # Consider more, gate filters
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights with appropriate strategies."""
        from .utils import (
            init_linear_kaiming, init_linear_normal, init_layer_norm, 
            init_sequential, init_linear_xavier
        )
        
        # FFN: LayerNorm at [0], Linear at [1] and [3]
        # Kaiming for hidden layer, scaled normal for output (smaller for deep nets)
        init_layer_norm(self.ffn[0])  # Pre-FFN LayerNorm
        init_linear_kaiming(self.ffn[1], nonlinearity='relu')  # GELU ≈ ReLU
        init_linear_normal(self.ffn[3], std=0.02 / math.sqrt(2.0 * self.num_layers))
        
        # LayerNorms
        init_layer_norm(self.norm1)
        init_layer_norm(self.norm2)
    
    def compute(
        self, 
        x: torch.Tensor, 
        cos_sin: Optional[CosSin] = None,
    ) -> torch.Tensor:
        """
        Step 2: COMPUTE - Self-attention + FFN.
        
        Args:
            x: Input (potentially fused with memory) [B, S, D]
            cos_sin: RoPE (cos, sin) for positional encoding in attention
        
        Returns:
            y: Output features [B, S, D]
        """
        # Self-attention with RoPE
        attn_out, _ = self.self_attn(x, x, x, cos_sin=cos_sin)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        y = self.norm2(x + ffn_out)
        
        return y
    
    def forward(
        self, 
        x: torch.Tensor,                           # [B, S, D_model]
        cache: torch.Tensor,                       # [B, L*K, D_cache]
        cos_sin: Optional[CosSin] = None,          # RoPE positional encoding
        input_injection: Optional[torch.Tensor] = None,  # [B, S, D_model] - TRM-style embedding injection
        temperature: float = 1.0,
        hard: bool = False,
        features: Optional[FeatureFlags] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Full Dual-Gated Cycle (v0.1.2 - Token-Level Caching).
        
        The write process follows DLSM spec:
        1. "Should I write?" - Hard gate per token
        2. "What to write?" - Importance scoring among gated tokens
        3. "Where to write?" - Slot selection via attention
        
        TRM-style input injection: if input_injection is provided, it's added to x
        at the start of the layer. This provides a direct gradient path from loss
        to embeddings, even with gradient-free passes.
        
        Args:
            x:               Input tokens [B, S, D_model]
            cache:           Global cache [B, L*K, D_cache]
            cos_sin:         RoPE (cos, sin) for positional encoding
            input_injection: Optional embedding injection (added to x at start)
            temperature:     Gumbel-softmax temperature
            hard:            Use hard decisions
            features:        Feature flags
        
        Returns:
            y:           Output features [B, S, D_model]
            new_cache:   Updated global cache [B, L*K, D_cache]
            aux:         Auxiliary info for logging/losses
        """
        if features is None:
            features = FeatureFlags()
        
        # === TRM-STYLE INPUT INJECTION ===
        # Add input_injection to provide direct gradient path from loss → embeddings
        if input_injection is not None:
            x = x + input_injection
            
        B, S, _ = x.shape
        K = self.num_slots
        
        # Extract this layer's local cache
        start_idx = self.layer_idx * K
        end_idx = start_idx + K
        local_cache = cache[:, start_idx:end_idx, :]  # [B, K, D_cache]
        
        # === STEP 1: ACTIVE READ ===
        # Router decides: should read? what to read? then fuses
        if features.use_cache and features.use_moe_memory:
            read_result = self.memory_router.read(
                x, cache, 
                temperature=temperature,
                hard=hard,
            )
            x_fused = read_result['x_fused']  # x + gate * context (additive)
            read_gate = read_result['read_gate']
            read_context = read_result['context']
        else:
            x_fused = x
            read_gate = torch.zeros(B, S, 1, device=x.device)
            read_context = None
        
        # Step 2: COMPUTE
        y = self.compute(x_fused, cos_sin=cos_sin)
        
        # === STEP 3: ACTIVE WRITE (Token-Level) ===
        # Pass ALL tokens to router - it handles the 3-stage decision:
        # 1. Should write? (gate)
        # 2. What to write? (importance)
        # 3. Where to write? (slot selection)
        #
        # For efficiency, we can limit to max_write_tokens candidates
        # but the router's gate decides which actually write
        T = min(S, self.max_write_tokens)
        write_candidates = y[:, :T, :]  # [B, T, D] - first T tokens as candidates
        
        if features.use_cache and features.use_moe_memory:
            write_result = self.memory_router.write(
                write_candidates,
                cache=cache,              # Global for novelty assessment
                local_cache=local_cache,  # Local for slot selection
                temperature=temperature,
                hard=hard,
            )
            new_local_cache = write_result['new_local_cache']
            write_scores = write_result['write_scores']
            write_gate = write_result['write_gate']
            slot_probs = write_result['slot_probs']
        else:
            new_local_cache = local_cache
            write_scores = torch.zeros(B, T, device=x.device)
            write_gate = torch.zeros(B, T, 1, device=x.device)
            slot_probs = torch.zeros(B, T, K, device=x.device)
        
        # === UPDATE GLOBAL CACHE ===
        new_cache = cache.clone()
        new_cache[:, start_idx:end_idx, :] = new_local_cache
        
        # === AUXILIARY INFO ===
        aux = {
            'read_gate': read_gate,
            'read_context': read_context,
            'write_gate': write_gate,
            'write_scores': write_scores,
            'slot_probs': slot_probs,
            # Entropy for diversity loss
            'entropy': -(slot_probs * torch.log(slot_probs + 1e-8)).sum(dim=-1).mean(),
            # Slot usage distribution
            'slot_counts': slot_probs.sum(dim=1),  # [B, K]
        }
        
        return y, new_cache, aux
