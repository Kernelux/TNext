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
from torch.utils.checkpoint import checkpoint

from .memory_controller import MemoryController, ConfidenceEstimator
from .modules import LinearAttention, CosSin, CacheSelfAttention


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
        use_checkpoint: bool = False,  # Gradient checkpointing for memory savings
        use_fixed_threshold: bool = True,  # Use fixed 0.5 threshold for gates
        fixed_threshold: float = 0.5,       # Fixed threshold value
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.layer_idx = layer_idx
        self.max_write_tokens = max_write_tokens
        self.use_checkpoint = use_checkpoint
        
        # === Core Components ===
        
        # Memory Controller (handles all memory operations)
        self.memory = MemoryController(
            d_model=d_model,
            d_cache=d_cache,
            num_slots=num_slots,
            num_layers=num_layers,
            layer_idx=layer_idx,
            dropout=dropout,
            use_fixed_threshold=use_fixed_threshold,
            fixed_threshold=fixed_threshold,
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
        
        # === Cache Self-Attention (Memory Consolidation) ===
        # After each write, allow cache slots to attend to each other.
        # This enables:
        # 1. Integration of new info with existing knowledge
        # 2. Transitive reasoning (A→B, B→C ⟹ A→C)
        # 3. Deduplication and consolidation
        # 
        # Operates on FULL SLOT (content + metadata) so that:
        # - Confidence scores influence attention weights
        # - Layer ID enables cross-layer slot reasoning
        # - Temporal info (iter/pass) provides recency context
        d_meta = 17  # Must match MemoryController: confidence(1) + temporal(16)
        d_slot = d_cache + d_meta
        
        # Find largest valid num_heads that divides d_slot evenly
        # d_slot = 65 (48 + 17), so we need a divisor of 65: 1, 5, 13, 65
        cache_attn_heads = 1  # Default to single head
        for h in [5, 13]:  # Try common divisors (65 = 5 * 13)
            if d_slot % h == 0:
                cache_attn_heads = h
                break
        
        self.cache_self_attn = CacheSelfAttention(
            d_cache=d_slot,  # Full slot dimension, not just content
            num_heads=cache_attn_heads,
            dropout=dropout,
            use_linear=True,  # Linear attention for efficiency
        )
    
    def forward(
        self,
        x: torch.Tensor,                       # [B, S, D_model]
        cache: torch.Tensor,                   # [B, L*K, D_slot]
        cos_sin: Optional[CosSin] = None,      # RoPE embeddings
        input_injection: Optional[torch.Tensor] = None,  # TRM-style injection
        temperature: float = 1.0,
        hard: bool = False,
        max_iterations: int = 1,               # For recursive refinement
        confidence_threshold: float = 0.9,     # Early stopping threshold
        return_all_iterations: bool = False,   # Return intermediate states
        pass_idx: int = 0,                     # Current model-level pass (for temporal embedding)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with optional recursive refinement.
        
        Args:
            x: Input tensor [B, S, D_model]
            cache: Global cache [B, L*K, D_slot] (content + metadata)
            cos_sin: RoPE embeddings for attention
            input_injection: Added to input (TRM-style gradient path)
            temperature: Gumbel-softmax temperature
            hard: Use hard routing decisions
            max_iterations: Max refinement iterations within layer
            confidence_threshold: Stop if confidence exceeds this
            return_all_iterations: Return all intermediate outputs
            pass_idx: Current model-level pass (for temporal embedding in write)
        
        Returns:
            output: Final output [B, S, D_model]
            updated_cache: Final cache [B, L*K, D_slot]
            aux: Auxiliary information dict
        """
        # TRM-style input injection
        if input_injection is not None:
            x = x + input_injection
        
        B, S, _ = x.shape
        
        # Track auxiliary information - use scalars to save memory
        aux = {
            'read_gate_sum': 0.0,
            'read_gate_count': 0,
            'write_gate_sum': 0.0,
            'write_gate_count': 0,
            'confidence_sum': 0.0,
            'confidence_count': 0,
            'iteration_feedback_sum': 0.0,
            'iteration_feedback_count': 0,
            'iteration_feedback_gates': [],  # Tensor list for polarization loss
            'iterations_run': 0,
            # Gate tensors for polarization loss (stored with gradients)
            'read_gates': [],
            'write_gates': [],
        }
        
        # Only store tensors if explicitly requested (for debugging)
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
                aux['iteration_feedback_sum'] += feedback_gate.mean().detach().item()
                aux['iteration_feedback_count'] += 1
                aux['iteration_feedback_gates'].append(feedback_gate.mean(dim=(1,2)))  # [B] avg gate
            
            # --- Step 1: MEMORY READ ---
            read_result = self.memory.read(
                h, current_cache,
                temperature=temperature,
                hard=hard,
            )
            h_enhanced = read_result['x_enhanced']
            aux['read_gate_sum'] += read_result['read_gate'].mean().detach().item()
            aux['read_gate_count'] += 1
            # Store SOFT gate tensor for polarization loss (pre-threshold, with gradients)
            aux['read_gates'].append(read_result['soft_read_gate'])
            
            # Store read threshold tensor (for gradient flow to threshold network)
            if 'read_thresholds' not in aux:
                aux['read_thresholds'] = []
            aux['read_thresholds'].append(read_result.get('read_threshold'))
            
            # --- Step 2: COMPUTE ---
            if self.use_checkpoint and self.training:
                h_computed = checkpoint(
                    self.compute, h_enhanced, cos_sin,
                    use_reentrant=False
                )
            else:
                h_computed = self.compute(h_enhanced, cos_sin=cos_sin)
            
            # --- Step 3: MEMORY WRITE ---
            # Only write first max_write_tokens for efficiency
            write_result = self.memory.write(
                h_computed, current_cache,
                temperature=temperature,
                hard=hard,
                max_write_tokens=self.max_write_tokens,
                iteration_idx=iteration,  # Current iteration within this layer
                pass_idx=pass_idx,         # Current model-level pass
            )
            current_cache = write_result['updated_cache']
            aux['write_gate_sum'] += write_result['write_gate'].mean().detach().item()
            aux['write_gate_count'] += 1
            # Store SOFT gate tensor for polarization loss (pre-threshold, with gradients)
            aux['write_gates'].append(write_result['soft_write_gate'])
            
            # Store write threshold tensor (for gradient flow to threshold network)
            if 'write_thresholds' not in aux:
                aux['write_thresholds'] = []
            aux['write_thresholds'].append(write_result.get('write_threshold'))
            
            # --- Step 3b: CACHE CONSOLIDATION (Self-Attention) ---
            # After writing, let cache slots attend to each other to:
            # 1. Integrate new info with existing knowledge
            # 2. Enable transitive reasoning across slots
            # 3. Consolidate and deduplicate information
            # 
            # Operates on FULL SLOT (content + metadata) so that attention
            # can leverage confidence, layer_id, and temporal information
            # when deciding how slots should influence each other.
            current_cache = self.cache_self_attn(current_cache)
            
            # Update state - save for next iteration's thought injection
            h = h_computed
            h_prev = h_computed.detach()  # Detach to prevent gradient explosion
            
            if return_all_iterations:
                aux['all_outputs'].append(h.detach())
            
            # --- Step 4: CONFIDENCE CHECK (for early stopping) ---
            confidence_est = self.confidence_estimator
            if max_iterations > 1 and confidence_est is not None:
                confidence = confidence_est(h)
                conf_val = confidence.mean().detach().item()
                aux['confidence_sum'] += conf_val
                aux['confidence_count'] += 1
                
                # Get learned halt threshold (budget + confidence aware)
                # Threshold sees: h_pooled, confidence, iter_ratio, budget_remaining
                # Low confidence + high budget → higher threshold (keep going)
                # High confidence OR low budget → lower threshold (halt)
                halt_threshold = confidence_est.get_layer_halt_threshold(
                    h,
                    confidence=confidence,  # No detach - allow gradient flow for learning
                    current_iter=iteration,
                    max_iter=max_iterations,
                )
                threshold_val = halt_threshold.mean().detach().item()
                aux['halt_threshold'] = threshold_val  # Scalar for debugging
                
                # Store threshold TENSOR (with gradients) for loss computation
                if 'layer_halt_thresholds' not in aux:
                    aux['layer_halt_thresholds'] = []
                aux['layer_halt_thresholds'].append(halt_threshold)  # [B] with gradients
                
                # Also store confidence tensor for layer-level threshold supervision
                if 'layer_confidences' not in aux:
                    aux['layer_confidences'] = []
                aux['layer_confidences'].append(confidence.detach())
                
                # Early stopping if confident enough (using learned threshold)
                # During inference: always respect confidence
                # During training: still allow halting (gradient flows through halt_threshold)
                if conf_val > threshold_val:
                    break
        
        # Aggregate auxiliary stats (already scalars)
        if aux['read_gate_count'] > 0:
            aux['avg_read_gate'] = aux['read_gate_sum'] / aux['read_gate_count']
        if aux['write_gate_count'] > 0:
            aux['avg_write_gate'] = aux['write_gate_sum'] / aux['write_gate_count']
        if aux['iteration_feedback_count'] > 0:
            aux['avg_iteration_feedback_gate'] = aux['iteration_feedback_sum'] / aux['iteration_feedback_count']
        if aux['confidence_count'] > 0:
            aux['avg_confidence'] = aux['confidence_sum'] / aux['confidence_count']
        
        return h, current_cache, aux



# Note: RecursiveMemoryModel was removed - use RecursiveRefinementModel from 
# recursive_refinement_model.py for the full ARC-compatible implementation.
# This file provides the building blocks: ComputeBlock and UnifiedMemoryLayer.
