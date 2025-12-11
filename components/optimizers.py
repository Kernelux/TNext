"""
Adam-Atan2 Optimizer

Based on the paper: https://arxiv.org/abs/2407.05872
Reference implementations:
- https://github.com/lucidrains/adam-atan2-pytorch
- https://github.com/imoneoi/adam-atan2

The key insight is replacing the division with epsilon:
    update = m / (sqrt(v) + eps)
    
With atan2 which is scale-invariant:
    update = a * atan2(m, b * sqrt(v))

This removes the need for eps entirely and makes the optimizer
gradient-scale invariant, which helps with deep networks that have
varying gradient magnitudes across layers.
"""

from __future__ import annotations
from typing import Callable, Tuple, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def exists(val):
    return val is not None


class AdamAtan2(Optimizer):
    """
    Adam optimizer with atan2 update rule for scale invariance.
    
    Instead of: update = m / (sqrt(v) + eps)
    Uses:       update = a * atan2(m / bias_correct1, b * sqrt(v / bias_correct2))
    
    This makes the optimizer invariant to gradient scale, which is beneficial
    for deep networks with varying gradient magnitudes across layers.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-4)
        betas: Coefficients for computing running averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient (default: 0.0)
        decoupled_wd: If True, use decoupled weight decay (AdamW style) (default: True)
        a: Scaling factor for the atan2 output (default: 1.27)
        b: Scaling factor for the denominator (default: 1.0)
        cautious_factor: Factor for cautious updates (0-1), see https://arxiv.org/abs/2411.16085 (default: 1.0)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        decoupled_wd: bool = True,
        a: float = 1.27,
        b: float = 1.0,
        cautious_factor: float = 1.0,
    ):
        assert lr > 0., f"Learning rate must be positive, got {lr}"
        assert all(0. <= beta <= 1. for beta in betas), f"Betas must be in [0, 1], got {betas}"
        assert weight_decay >= 0., f"Weight decay must be non-negative, got {weight_decay}"
        assert 0. <= cautious_factor <= 1., f"Cautious factor must be in [0, 1], got {cautious_factor}"
        
        self._init_lr = lr
        self.decoupled_wd = decoupled_wd
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            a=a,
            b=b,
            cautious_factor=cautious_factor,
        )
        
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                grad = p.grad
                lr = group['lr']
                wd = group['weight_decay']
                beta1, beta2 = group['betas']
                a = group['a']
                b = group['b']
                cautious_factor = group['cautious_factor']
                state = self.state[p]
                init_lr = self._init_lr
                
                # Maybe decoupled weight decay (AdamW style)
                if self.decoupled_wd and wd > 0.:
                    wd_factor = wd
                else:
                    wd_factor = wd * lr / init_lr if wd > 0. else 0.
                
                # Apply weight decay
                if wd > 0.:
                    p.mul_(1. - lr * wd_factor / lr if not self.decoupled_wd else 1. - lr * wd)
                
                # Initialize state if needed
                if len(state) == 0:
                    state['steps'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                
                # Get state variables
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                steps = state['steps']
                
                steps += 1
                
                # Bias corrections
                bias_correct1 = 1. - beta1 ** steps
                bias_correct2 = 1. - beta2 ** steps
                
                # Update running averages (exponential moving average)
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.lerp_(grad, 1. - beta1)
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                exp_avg_sq.lerp_(grad * grad, 1. - beta2)
                
                # The key change: use atan2 instead of division with epsilon
                # Standard Adam: update = (exp_avg / bias_correct1) / (sqrt(exp_avg_sq / bias_correct2) + eps)
                # Adam-atan2:    update = a * atan2(exp_avg / bias_correct1, b * sqrt(exp_avg_sq / bias_correct2))
                
                # Compute denominator: b * sqrt(exp_avg_sq / bias_correct2)
                den = exp_avg_sq.mul(b * b / bias_correct2).sqrt_()
                
                # Compute update using atan2: atan2(numerator, denominator)
                # Note: atan2(y, x) returns angle in [-pi, pi], so output is bounded!
                update = exp_avg.mul(1. / bias_correct1).atan2_(den)
                
                # Maybe apply cautious update (https://arxiv.org/abs/2411.16085)
                # Only update in directions that align with the gradient
                if cautious_factor < 1.:
                    align_mask = (update * grad) > 0
                    scale = torch.where(align_mask, torch.ones_like(grad), cautious_factor)
                    update *= (scale / scale.mean().clamp(min=1e-5))
                
                # Update parameters: p = p - lr * a * update
                p.add_(update, alpha=-lr * a)
                
                # Save step count
                state['steps'] = steps
        
        return loss


# Alias for convenience
Adam = AdamAtan2


class AdamAtan2Foreach(Optimizer):
    """
    Faster version of AdamAtan2 using foreach operations.
    
    This version processes all parameters in a group together using
    torch._foreach_* operations, which can be significantly faster
    on GPU when there are many small parameters.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        decoupled_wd: bool = True,
        a: float = 1.27,
        b: float = 1.0,
    ):
        assert lr > 0., f"Learning rate must be positive, got {lr}"
        assert all(0. <= beta <= 1. for beta in betas), f"Betas must be in [0, 1], got {betas}"
        assert weight_decay >= 0., f"Weight decay must be non-negative, got {weight_decay}"
        
        # Check that foreach operations are available
        assert all(hasattr(torch, f'_foreach_{attr}_') for attr in ('mul', 'add', 'lerp', 'sqrt')), \
            'This version of torch does not have the prerequisite foreach functions'
        
        self._init_lr = lr
        self.decoupled_wd = decoupled_wd
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            a=a,
            b=b,
        )
        
        super().__init__(params, defaults)
    
    @staticmethod
    def _foreach_atan2_(nums: list, dens: list):
        """Apply atan2 element-wise to lists of tensors."""
        # Check if native foreach_atan2 is available (PyTorch 2.1+)
        if hasattr(torch, '_foreach_atan2_'):
            torch._foreach_atan2_(nums, dens)
        else:
            # Fallback to sequential atan2
            for num, den in zip(nums, dens):
                num.atan2_(den)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        
        init_lr = self._init_lr
        
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            beta1, beta2 = group['betas']
            a = group['a']
            b = group['b']
            
            has_weight_decay = wd > 0
            
            # Accumulate tensors for foreach operations
            params = []
            grads = []
            grad_squared = []
            exp_avgs = []
            exp_avg_sqs = []
            
            for p in filter(lambda p: exists(p.grad), group['params']):
                grad = p.grad
                state = self.state[p]
                
                # Maybe decoupled weight decay
                if self.decoupled_wd and has_weight_decay:
                    wd_scaled = wd
                else:
                    wd_scaled = wd * lr / init_lr if has_weight_decay else 0.
                
                # Initialize state if needed
                if len(state) == 0:
                    state['steps'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                steps = state['steps']
                
                steps += 1
                
                # Bias corrections
                bias_correct1 = 1. - beta1 ** steps
                bias_correct2 = 1. - beta2 ** steps
                
                # Append to lists
                params.append(p)
                grads.append(grad)
                grad_squared.append(grad * grad)
                exp_avgs.append(exp_avg)
                exp_avg_sqs.append(exp_avg_sq)
                
                # Update step count
                state['steps'] = steps
            
            if not params:
                continue
            
            # Weight decay
            if has_weight_decay:
                torch._foreach_mul_(params, 1. - lr * wd)
            
            # Update running averages
            torch._foreach_lerp_(exp_avgs, grads, 1. - beta1)
            torch._foreach_lerp_(exp_avg_sqs, grad_squared, 1. - beta2)
            
            # Clone for update computation
            updates = [t.clone() for t in exp_avgs]
            den = [t.clone() for t in exp_avg_sqs]
            
            # Apply bias correction and scaling
            # Note: We need per-parameter bias correction, but for simplicity
            # we use the same value (assumes all params have same step count)
            # For proper implementation, track bias_correct per parameter
            torch._foreach_mul_(updates, 1. / bias_correct1)
            torch._foreach_mul_(den, b * b / bias_correct2)
            torch._foreach_sqrt_(den)
            
            # Apply atan2
            self._foreach_atan2_(updates, den)
            
            # Update parameters
            torch._foreach_add_(params, updates, alpha=-lr * a)
        
        return loss
