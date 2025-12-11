"""
Unified Memory Layer (v0.2.0)
=============================

A new layer design that cleanly separates:
1. Memory operations (via MemoryController)
2. Computation (via any compute block)
3. Halting/confidence assessment

This layer supports true recursive refinement within each layer:
- Multiple internal iterations until confidence threshold met
- Each iteration: read → compute → write → assess
- TRM-style latent state refinement

Usage:
------
```python
layer = UnifiedMemoryLayer(
    d_model=128,
    d_cache=64,
    num_slots=16,
    layer_idx=0,
    num_layers=3,
)

# Single forward pass
output, cache, aux = layer(x, cache)

# With recursive refinement (TRM-style)
output, cache, aux = layer(x, cache, max_iterations=3, confidence_threshold=0.8)
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from contextlib import nullcontext

from .memory_controller import MemoryController, ConfidenceEstimator
from .modules import LinearAttention, CosSin


class ComputeBlock(nn.Module):
    """
    Standard computation block (Transformer-style).
    
    This is the "base architecture" that the MemoryController wraps around.
    Can be replaced with any computation mechanism.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_linear_attention: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Self-attention (linear for efficiency)
        if use_linear_attention:
            self.self_attn = LinearAttention(d_model, num_heads, dropout)
        else:
            # Standard attention
            self.self_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        self.use_linear = use_linear_attention
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer norms (Pre-LN style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
    ) -> torch.Tensor:
        """
        Standard Transformer computation.
        
        Args:
            x: Input [B, S, D]
            cos_sin: RoPE embeddings (optional)
        
        Returns:
            Output [B, S, D]
        """
        # Self-attention
        if self.use_linear:
            attn_out, _ = self.self_attn(x, x, x, cos_sin=cos_sin)
        else:
            x_norm = self.norm1(x)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class UnifiedMemoryLayer(nn.Module):
    """
    Unified Memory Layer with clean separation of concerns.
    
    Architecture per forward pass:
    1. MEMORY READ: Query cache, fuse with input
    2. COMPUTE: Self-attention + FFN
    3. MEMORY WRITE: Score importance, update cache
    4. ASSESS: Confidence/halting estimation (optional)
    
    Supports recursive refinement within the layer:
    - Multiple iterations of the above cycle
    - Halts early when confidence threshold met
    - TRM-style latent state (cache) refinement
    """
    
    def __init__(
        self,
        d_model: int,
        d_cache: int,
        num_slots: int,
        num_layers: int,
        layer_idx: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_write_tokens: int = 64,
        use_linear_attention: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.layer_idx = layer_idx
        self.max_write_tokens = max_write_tokens
        
        # === Core Components ===
        
        # Memory Controller (handles all memory operations)
        self.memory = MemoryController(
            d_model=d_model,
            d_cache=d_cache,
            num_slots=num_slots,
            num_layers=num_layers,
            layer_idx=layer_idx,
            dropout=dropout,
        )
        
        # Compute Block (handles all computation)
        self.compute = ComputeBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_linear_attention=use_linear_attention,
        )
        
        # === Layer-Level Iteration Feedback (Thought Injection) ===
        # Unlike model-level answer feedback, this injects the *previous iteration's
        # hidden state* back into the current iteration. This is needed because:
        # 1. Cache writes are gated/selective - not all information gets stored
        # 2. Read/write don't happen every time (gating can be zero)
        # 3. Provides direct gradient path for iteration-over-iteration learning
        self.iteration_feedback_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Optional: Confidence estimator for recursive refinement
        # (Can be shared across layers or per-layer)
        self.confidence_estimator = None  # Set externally if needed
    
    def forward(
        self,
        x: torch.Tensor,                       # [B, S, D_model]
        cache: torch.Tensor,                   # [B, L*K, D_cache]
        cos_sin: Optional[CosSin] = None,      # RoPE embeddings
        input_injection: Optional[torch.Tensor] = None,  # TRM-style injection
        temperature: float = 1.0,
        hard: bool = False,
        max_iterations: int = 1,               # For recursive refinement
        confidence_threshold: float = 0.9,     # Early stopping threshold
        return_all_iterations: bool = False,   # Return intermediate states
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with optional recursive refinement.
        
        Args:
            x: Input tensor [B, S, D_model]
            cache: Global cache [B, L*K, D_cache]
            cos_sin: RoPE embeddings for attention
            input_injection: Added to input (TRM-style gradient path)
            temperature: Gumbel-softmax temperature
            hard: Use hard routing decisions
            max_iterations: Max refinement iterations within layer
            confidence_threshold: Stop if confidence exceeds this
            return_all_iterations: Return all intermediate outputs
        
        Returns:
            output: Final output [B, S, D_model]
            updated_cache: Final cache [B, L*K, D_cache]
            aux: Auxiliary information dict
        """
        # TRM-style input injection
        if input_injection is not None:
            x = x + input_injection
        
        B, S, _ = x.shape
        
        # Track auxiliary information
        aux = {
            'read_gates': [],
            'write_gates': [],
            'confidences': [],
            'iteration_feedback_gates': [],  # Track feedback gating
            'iterations_run': 0,
        }
        
        if return_all_iterations:
            aux['all_outputs'] = []
        
        # Current state
        h = x
        current_cache = cache
        h_prev = None  # Previous iteration's output for thought injection
        
        # === Recursive Refinement Loop ===
        for iteration in range(max_iterations):
            aux['iterations_run'] += 1
            
            # --- Step 0: THOUGHT INJECTION (from previous iteration) ---
            # Inject previous iteration's output to provide direct feedback path.
            # This complements cache (which is selective/gated) by providing
            # full output continuity across iterations.
            if iteration > 0 and h_prev is not None:
                # Concatenate current state with previous, let gate decide how much to blend
                gate_input = torch.cat([h, h_prev], dim=-1)
                feedback_gate = self.iteration_feedback_gate(gate_input)
                h = h + feedback_gate * h_prev  # Gated residual from previous iteration
                aux['iteration_feedback_gates'].append(feedback_gate.mean().detach())
            
            # --- Step 1: MEMORY READ ---
            read_result = self.memory.read(
                h, current_cache,
                temperature=temperature,
                hard=hard,
            )
            h_enhanced = read_result['x_enhanced']
            aux['read_gates'].append(read_result['read_gate'].detach())
            
            # --- Step 2: COMPUTE ---
            h_computed = self.compute(h_enhanced, cos_sin=cos_sin)
            
            # --- Step 3: MEMORY WRITE ---
            # Only write first max_write_tokens for efficiency
            write_result = self.memory.write(
                h_computed, current_cache,
                temperature=temperature,
                hard=hard,
                max_write_tokens=self.max_write_tokens,
            )
            current_cache = write_result['updated_cache']
            aux['write_gates'].append(write_result['write_gate'].detach())
            
            # Update state - save for next iteration's thought injection
            h = h_computed
            h_prev = h_computed.detach()  # Detach to prevent gradient explosion
            
            if return_all_iterations:
                aux['all_outputs'].append(h.detach())
            
            # --- Step 4: CONFIDENCE CHECK (for early stopping) ---
            confidence_est = self.confidence_estimator
            if max_iterations > 1 and confidence_est is not None:
                confidence = confidence_est(h)
                aux['confidences'].append(confidence.detach())
                
                # Early stopping if confident enough
                if not self.training and confidence.mean() > confidence_threshold:
                    break
        
        # Aggregate auxiliary stats
        if aux['read_gates']:
            aux['avg_read_gate'] = torch.stack(aux['read_gates']).mean()
        if aux['write_gates']:
            aux['avg_write_gate'] = torch.stack(aux['write_gates']).mean()
        if aux['iteration_feedback_gates']:
            aux['avg_iteration_feedback_gate'] = torch.stack(aux['iteration_feedback_gates']).mean()
        
        return h, current_cache, aux



# Note: RecursiveMemoryModel was removed - use RecursiveRefinementModel from 
# recursive_refinement_model.py for the full ARC-compatible implementation.
# This file provides the building blocks: ComputeBlock and UnifiedMemoryLayer.
