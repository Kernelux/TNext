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

from .config import SLOT_DIMS
from .memory_controller import MemoryController, LayerHaltEstimator
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
        use_linear_attention: bool = True,
        use_checkpoint: bool = False,  # Gradient checkpointing for memory savings
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.layer_idx = layer_idx
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
        )
        
        # Compute Block (handles all computation)
        self.compute = ComputeBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_linear_attention=use_linear_attention,
        )
        
        # NOTE: TRM uses simple additive injection (h = h + h_prev), no gate needed.
        # The iteration feedback gate was removed as it adds complexity without benefit.
        # Skip connections + simple addition provide sufficient gradient flow.
        
        # === Iteration Embeddings (Layer-level iteration awareness) ===
        # Let the layer know which iteration it's on (how many times it's refined)
        # max_iterations is typically 4-8
        self.iteration_embeddings = nn.Embedding(16, d_model)  # Support up to 16 iterations
        nn.init.normal_(self.iteration_embeddings.weight, mean=0.0, std=0.02)
        
        # === Per-Layer Halt Estimator ===
        # Each layer has its own halting estimator because:
        # 1. Different layers learn different representations (early=features, late=reasoning)
        # 2. Each layer may need different amounts of refinement
        # 3. Layer-specific patterns in when to halt
        # Uses hidden state stability (no logits at layer level)
        self.layer_halt_estimator = LayerHaltEstimator(
            d_model=d_model,
            layer_idx=layer_idx,
            max_iterations=8,  # Will be overridden by max_iterations in forward
        )
        
        # DEPRECATED: Old shared confidence estimator
        # Keep for backward compatibility but mark for removal
        self.confidence_estimator = None  # Set externally if using old API
        
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
        d_slot = SLOT_DIMS.d_slot(d_cache)
        
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
    
    def _compute_representation_entropy(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of hidden state distribution.
        
        Treats the hidden dimension as a distribution (via softmax) and 
        computes entropy. High entropy = spread activations, low entropy = peaked.
        
        Changes in entropy between iterations indicate meaningful computation.
        
        Args:
            h: Hidden state [B, S, D]
            
        Returns:
            entropy: [B] mean entropy per sample
        """
        # Softmax over hidden dim to get "probability distribution"
        p = F.softmax(h, dim=-1)  # [B, S, D]
        
        # Entropy: -sum(p * log(p))
        log_p = torch.log(p + 1e-8)
        entropy_per_position = -(p * log_p).sum(dim=-1)  # [B, S]
        
        # Mean over sequence positions
        return entropy_per_position.mean(dim=-1)  # [B]
    
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
        # Centralized temporal embeddings (from model)
        cache_layer_embed: Optional[nn.Embedding] = None,
        cache_iter_embed: Optional[nn.Embedding] = None,
        cache_pass_embed: Optional[nn.Embedding] = None,
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
            'iterations_run': 0,
            # Gate tensors for polarization loss (stored with gradients)
            'read_gates': [],
            'write_gates': [],
            # Layer-level entropy tracking for info gain
            'layer_entropies': [],  # Entropy at each iteration
            'layer_info_gains': [],  # Info gain between iterations
        }
        
        # Only store tensors if explicitly requested (for debugging)
        if return_all_iterations:
            aux['all_outputs'] = []
        
        # Current state
        h = x
        current_cache = cache
        h_prev = None  # Previous iteration's output for thought injection
        h_prev_for_stability = None  # Previous h state for stability measurement
        
        # === INPUT INJECTION (TRM-style) ===
        # Save original layer input for re-injection at each iteration
        # This keeps the computation grounded to the original input
        layer_input = x  # [B, S, D] - preserved for input injection + final skip
        
        # === Recursive Refinement Loop ===
        for iteration in range(max_iterations):
            aux['iterations_run'] += 1
            
            # --- Step 0: ITERATION EMBEDDING + INPUT + THOUGHT INJECTION ---
            # Add iteration embedding so layer knows which refinement step it's on
            iter_emb = self.iteration_embeddings(
                torch.tensor(iteration, device=x.device)
            )  # [D_model]
            h = h + iter_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, D_model] broadcast
            
            # TRM-style input injection: add original input AND previous iteration output
            if iteration > 0:
                # Re-inject original input (like TRM's input_embeddings)
                h = h + layer_input
                # Also add previous iteration's output for iterative refinement
                if h_prev is not None:
                    h = h + h_prev
                aux['iteration_feedback_sum'] += 1.0  # Track that injection happened
                aux['iteration_feedback_count'] += 1
            
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
            write_result = self.memory.write(
                h_computed, current_cache,
                temperature=temperature,
                hard=hard,
                iteration_idx=iteration,  # Current iteration within this layer
                pass_idx=pass_idx,         # Current model-level pass
                # Centralized temporal embeddings (from model)
                cache_layer_embed=cache_layer_embed,
                cache_iter_embed=cache_iter_embed,
                cache_pass_embed=cache_pass_embed,
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
            # IMPORTANT: Save h_prev BEFORE updating h, so stability measures actual change
            h_prev_for_stability = h.detach()  # Previous h (before this iteration's compute)
            h = h_computed
            h_prev = h_computed.detach()  # For thought injection in NEXT iteration
            
            # --- Track Layer Entropy & Info Gain ---
            # Compute entropy of hidden state distribution (treats h as logits over features)
            # This measures "spread" of activations - productive iterations should change this
            with torch.no_grad():
                h_entropy = self._compute_representation_entropy(h)  # [B]
                aux['layer_entropies'].append(h_entropy)
                
                # Compute info gain if we have previous entropy
                if len(aux['layer_entropies']) > 1:
                    prev_entropy = aux['layer_entropies'][-2]
                    # Info gain = entropy reduction (becoming more certain)
                    # Can also be negative if becoming less certain (exploring)
                    info_gain = prev_entropy - h_entropy  # [B]
                    aux['layer_info_gains'].append(info_gain)
            
            if return_all_iterations:
                aux['all_outputs'].append(h.detach())
            
            # --- Step 4: HALT CHECK (for early stopping) ---
            # Uses info gain to decide if this layer should stop iterating.
            # Halt when info_gain < threshold (not making progress)
            # Each layer has its own halt estimator that learns layer-specific patterns.
            if max_iterations > 1:
                # Per-layer halt estimator (uses info gain + budget)
                # Use h_prev_for_stability (state BEFORE this iteration) to measure change
                info_gain, threshold = self.layer_halt_estimator(
                    h,
                    h_prev=h_prev_for_stability,  # Compare to state BEFORE this iteration
                    current_iter=iteration,
                    max_iter=max_iterations,
                )
                
                info_gain_val = info_gain.mean().detach().item()
                threshold_val = threshold.mean().detach().item()
                
                aux['confidence_sum'] += info_gain_val  # info_gain = "confidence" for layers
                aux['confidence_count'] += 1
                aux['halt_threshold'] = threshold_val  # Scalar for debugging
                
                # Store tensors (with gradients) for loss computation
                if 'layer_halt_thresholds' not in aux:
                    aux['layer_halt_thresholds'] = []
                aux['layer_halt_thresholds'].append(threshold)
                
                if 'layer_confidences' not in aux:
                    aux['layer_confidences'] = []
                aux['layer_confidences'].append(info_gain.detach())
                
                # Store info gains WITH gradients for divergence loss
                # This allows backprop to encourage productive iterations
                if 'layer_stabilities' not in aux:
                    aux['layer_stabilities'] = []
                aux['layer_stabilities'].append(info_gain)  # Keep gradients!
                
                # Early stopping: halt when info_gain < threshold (not making progress)
                # Different from old logic which was stability > threshold
                should_halt = info_gain_val < threshold_val
                
                if should_halt:
                    if self.training:
                        # ε-greedy exploration during training (30% chance to continue)
                        # This ensures the model sees what happens with more iterations
                        explore = torch.rand(1, device=h.device).item() < 0.3
                        if not explore:
                            break
                    else:
                        break
        
        # === SKIP CONNECTION: Add layer input to final output ===
        # This ensures gradient flow even if iterations don't produce useful changes
        # h = iteration_output + layer_input (dense residual)
        h = h + layer_input
        
        # Aggregate auxiliary stats (already scalars)
        if aux['read_gate_count'] > 0:
            aux['avg_read_gate'] = aux['read_gate_sum'] / aux['read_gate_count']
        if aux['write_gate_count'] > 0:
            aux['avg_write_gate'] = aux['write_gate_sum'] / aux['write_gate_count']
        if aux['iteration_feedback_count'] > 0:
            aux['avg_iteration_feedback'] = aux['iteration_feedback_sum'] / aux['iteration_feedback_count']
        if aux['confidence_count'] > 0:
            aux['avg_confidence'] = aux['confidence_sum'] / aux['confidence_count']
        
        return h, current_cache, aux



# Note: RecursiveMemoryModel was removed - use RecursiveRefinementModel from 
# recursive_refinement_model.py for the full ARC-compatible implementation.
# This file provides the building blocks: ComputeBlock and UnifiedMemoryLayer.
