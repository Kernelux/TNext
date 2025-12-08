import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .config import TrainingConfig

def compute_diversity_loss(slot_counts_list: list) -> torch.Tensor:
    """
    Slot Diversity Loss (Section 9.3 - Solution A):
    L_diversity = -λ_D · H(1/T · Σ_t slot_probs_t)

    Encourages uniform slot usage across the sequence.
    """
    if not slot_counts_list:
        return torch.tensor(0.0)

    # Aggregate slot counts across all layers
    total_counts = torch.stack(slot_counts_list).sum(dim=0)  # [B, K]

    # Normalize to distribution
    total_counts = total_counts + 1e-8
    probs = total_counts / total_counts.sum(dim=-1, keepdim=True)

    # Entropy (higher = more uniform = better)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    # We want to maximize entropy, so return negative
    return -entropy

def compute_task_loss(
    logits: torch.Tensor,
    size_logits: torch.Tensor,
    test_output: torch.Tensor,
    output_size: torch.Tensor,
    pass_logits_list: List[torch.Tensor],
    model_num_colors: int,
    max_grid_size: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute core task losses: Color CrossEntropy and Size Error.
    Supports deep supervision via pass_logits_list.
    """
    total_color_loss = 0.0
    # Clamp targets to valid range [0, num_colors-1], use -1 for ignore
    targets = test_output.view(-1).clamp(min=0, max=model_num_colors - 1)

    # If using deep supervision, average loss across all passes
    # defaulting to just using final logits if pass_logits_list is empty/single
    logits_to_process = pass_logits_list if pass_logits_list else [logits]

    for p_logits in logits_to_process:
        p_logits_flat = p_logits.reshape(-1, model_num_colors)
        total_color_loss += F.cross_entropy(p_logits_flat, targets, ignore_index=-1)

    color_loss = total_color_loss / len(logits_to_process)

    # Clamp size targets to valid range [0, max_grid_size-1]
    true_h = output_size[:, 0].clamp(min=0, max=max_grid_size - 1)
    true_w = output_size[:, 1].clamp(min=0, max=max_grid_size - 1)

    size_h = F.cross_entropy(size_logits[:, 0], true_h)
    size_w = F.cross_entropy(size_logits[:, 1], true_w)
    size_loss = size_h + size_w

    total_task_loss = color_loss + size_loss
    return total_task_loss, color_loss, size_loss

def compute_q_head_loss(
    aux_info: Dict,
    test_output: torch.Tensor,
    config: TrainingConfig,
    device: torch.device
) -> torch.Tensor:
    """
    [TRM INSIGHT: Explicit Q-Head Loss]
    
    Supports two modes:
    1. Single Q-head (legacy): halt_prob predicts correctness
    2. Dual Q-head (TRM): q_halt predicts "is_correct", q_continue predicts "is_wrong"
    
    The dual Q-head approach from TinyTRM gives cleaner learning signal:
    - q_halt: BCE(q_halt, is_correct) - learns to say "stop, this is right"
    - q_continue: BCE(q_continue, is_wrong) - learns to say "continue, this is wrong"
    
    This dual signal helps the model learn both when to stop AND when to keep going.
    """
    model_q_loss = torch.tensor(0.0, device=device)
    features = config.features

    if not features.use_explicit_q_head:
        return model_q_loss
    
    # [TRM] Dual Q-head mode - q_halt and q_continue are LOGITS (not probabilities)
    if features.use_dual_q_head and 'q_halt' in aux_info and aux_info['q_halt']:
        q_halt_list = aux_info['q_halt']
        q_continue_list = aux_info['q_continue']
        
        if not q_halt_list:
            return model_q_loss
        
        # Get pass logits for computing per-pass correctness
        pass_logits = aux_info.get('pass_logits', [])
        final_logits = aux_info.get('final_logits')
        
        # [TRM] no_act_continue: Paper recommends skipping Q_continue loss
        # "No continue ACT loss, only use the sigmoid of the halt which makes much more sense"
        use_q_continue = not features.no_act_continue
        
        # If we have per-pass logits (deep supervision), compute per-pass loss
        if pass_logits:
            total_loss = torch.tensor(0.0, device=device)
            for i, (q_h_logit, q_c_logit, p_logits) in enumerate(zip(q_halt_list, q_continue_list, pass_logits)):
                # Per-pass correctness
                preds = p_logits.detach().argmax(dim=-1)
                is_correct = (preds == test_output).all(dim=-1).all(dim=-1).float()  # [B]
                
                # BCE with logits (more numerically stable)
                loss_halt = F.binary_cross_entropy_with_logits(q_h_logit, is_correct.detach())
                total_loss = total_loss + loss_halt
                
                # Q_continue loss (optional, paper recommends skipping)
                if use_q_continue:
                    is_wrong = 1.0 - is_correct
                    loss_cont = F.binary_cross_entropy_with_logits(q_c_logit, is_wrong.detach())
                    total_loss = total_loss + loss_cont
            
            model_q_loss = total_loss / len(pass_logits)
        
        # Otherwise, use final logits only
        elif final_logits is not None:
            q_halt_final = q_halt_list[-1]
            
            preds = final_logits.detach().argmax(dim=-1)
            is_correct = (preds == test_output).all(dim=-1).all(dim=-1).float()
            
            # BCE with logits (more numerically stable)
            loss_halt = F.binary_cross_entropy_with_logits(q_halt_final, is_correct.detach())
            model_q_loss = loss_halt
            
            # Q_continue loss (optional, paper recommends skipping)
            if use_q_continue:
                q_continue_final = q_continue_list[-1]
                is_wrong = 1.0 - is_correct
                loss_cont = F.binary_cross_entropy_with_logits(q_continue_final, is_wrong.detach())
                model_q_loss = model_q_loss + loss_cont
        
        return model_q_loss
    
    # Legacy single Q-head mode
    if 'halt_probs' in aux_info and aux_info['halt_probs']:
        halt_probs = aux_info['halt_probs']

        # Use final pass's halt_prob and compare against final correctness
        # This avoids needing pass_logits_list (which is empty when deep supervision is off)
        if halt_probs:
            # Get final halt prob
            final_halt = halt_probs[-1]
            # Clamp for numerical stability in BCE
            final_halt = final_halt.clamp(min=1e-6, max=1 - 1e-6)

            # Use the test_output directly to check final prediction would be correct
            # (We don't have access to final logits here, so we use a simpler approach:
            #  train halt_prob toward 0.5 initially, let it learn from gradient signal)
            # Actually, we need the final logits. Let's check if they're in aux_info
            if 'final_logits' in aux_info:
                final_logits = aux_info['final_logits']
                preds = final_logits.detach().argmax(dim=-1)
                is_correct = (preds == test_output).all(dim=-1).all(dim=-1).float()
                model_q_loss = F.binary_cross_entropy(final_halt, is_correct.detach())
            else:
                # Fallback: average over all halt probs with a uniform target
                # This just regularizes halt_probs toward middle values
                for p_halt in halt_probs:
                    p_halt_clamped = p_halt.clamp(min=1e-6, max=1 - 1e-6)
                    # Soft target: encourage exploration (0.5)
                    target = torch.full_like(p_halt_clamped, 0.5)
                    model_q_loss += F.binary_cross_entropy(p_halt_clamped, target)
                model_q_loss = model_q_loss / len(halt_probs)

    return model_q_loss

def compute_step_efficiency_loss(
    logits: torch.Tensor,
    aux_info: Dict,
    test_output: torch.Tensor,
    config: TrainingConfig,
    model_num_layers: int,
    device: torch.device
) -> torch.Tensor:
    """
    [LAYER STEP EFFICIENCY - Option D: Correct-Conditioned]
    
    Only optimize for efficiency when the task is correct.
    This avoids the adversarial min/max dynamic:
    - When correct: minimize steps (be efficient)
    - When wrong: no step loss (let task_loss drive learning)
    
    This creates a stable equilibrium where the model first learns
    to solve tasks, then optimizes for efficiency.
    """
    step_efficiency_loss = torch.tensor(0.0, device=device)
    features = config.features

    if features.use_layer_act and 'expected_steps' in aux_info and aux_info['expected_steps']:
        # Task-level correctness: ALL cells must be correct
        final_preds = logits.detach().argmax(dim=-1)
        is_correct = (final_preds == test_output).all(dim=-1).all(dim=-1).float()  # [B]
        
        # Skip if nothing is correct (no efficiency signal to give)
        if is_correct.sum() == 0:
            return step_efficiency_loss

        # Collect expected steps across all passes
        total_expected_steps = torch.zeros_like(is_correct)  # [B]
        num_passes = len(aux_info['expected_steps'])
        
        for es in aux_info['expected_steps']:  # es: [B, num_layers]
            total_expected_steps += es.sum(dim=1)  # Sum across layers
        
        # Normalize by max possible steps
        max_recurrent_steps = getattr(config, 'max_recurrent_steps', 4)
        max_total_steps = float(max_recurrent_steps * model_num_layers * num_passes)
        normalized_steps = total_expected_steps / (max_total_steps + 1e-8)  # [B] in [0, 1]
        
        # OPTION D: Only penalize steps when correct
        # When correct: loss = normalized_steps (minimize)
        # When wrong: loss = 0 (no step penalty, focus on task_loss)
        step_efficiency_loss = (is_correct * normalized_steps).sum() / (is_correct.sum() + 1e-8)
        
        # Clamp for safety
        step_efficiency_loss = step_efficiency_loss.clamp(min=0.0, max=1.0)

    return step_efficiency_loss

def compute_ponder_loss(aux_info: Dict, config: TrainingConfig, device: torch.device) -> torch.Tensor:
    """
    Ponder loss for Adaptive Computation Time (ACT).
    Penalizes the cumulative halting probability (ponder cost).
    """
    features = config.features
    model_ponder_loss = torch.tensor(0.0, device=device)

    if features.use_ponder_loss and features.use_act_halting and 'ponder_cost' in aux_info:
        model_ponder_loss = aux_info['ponder_cost']

    return model_ponder_loss

def compute_total_loss(
    model,
    logits: torch.Tensor,
    size_logits: torch.Tensor,
    test_output: torch.Tensor,
    output_size: torch.Tensor,
    aux_info: Dict,
    config: TrainingConfig,
    global_step: int,
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Aggregates all losses.
    Returns total_loss and a dictionary of scalar loss components for logging.
    """
    # 1. Task Loss
    # Only use deep supervision if enabled (memory intensive)
    features = config.features
    if features.use_deep_supervision:
        pass_logits_list = aux_info.get('pass_logits', [logits])
    else:
        pass_logits_list = [logits]  # Only final output

    task_loss, color_loss, size_loss = compute_task_loss(
        logits, size_logits, test_output, output_size, pass_logits_list,
        model.num_colors, model.max_grid_size
    )

    # 2. Q-Head Loss
    model_q_loss = compute_q_head_loss(aux_info, test_output, config, device)

    # 3. Step Efficiency Loss
    step_efficiency_loss = compute_step_efficiency_loss(
        logits, aux_info, test_output, config, model.num_layers, device
    )

    # 4. Diversity Loss (always active, prevents slot collapse)
    diversity_loss = torch.tensor(0.0, device=device)
    features = config.features
    if features.use_diversity_loss and 'slot_counts' in aux_info and aux_info['slot_counts']:
        diversity_loss = compute_diversity_loss(aux_info['slot_counts'])

    # 5. Ponder Loss
    model_ponder_loss = compute_ponder_loss(aux_info, config, device)

    # Weighted Sum with NaN protection
    compute_loss = (
        config.lambda_q_head * model_q_loss +
        config.lambda_ponder * model_ponder_loss +
        config.lambda_step_efficiency * step_efficiency_loss
    )

    total_loss = task_loss + compute_loss + config.lambda_diversity * diversity_loss

    # NaN protection: if any component is NaN, use only task_loss
    if torch.isnan(total_loss):
        print(f"WARNING: NaN detected! task={task_loss.item():.4f}, q_head={model_q_loss.item():.4f}, "
              f"ponder={model_ponder_loss.item():.4f}, step_eff={step_efficiency_loss.item():.4f}, "
              f"diversity={diversity_loss.item():.4f}")
        total_loss = task_loss  # Fallback to just task loss

    # Return metrics for logging
    metrics = {
        'loss_total': total_loss.detach().item(),
        'loss_task': task_loss.detach().item(),
        'loss_color': color_loss.detach().item(),
        'loss_size': size_loss.detach().item(),
        'loss_compute': compute_loss.detach().item() if isinstance(compute_loss, torch.Tensor) else compute_loss,
        'loss_diversity': diversity_loss.detach().item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
        'loss_q_head': model_q_loss.detach().item() if isinstance(model_q_loss, torch.Tensor) else model_q_loss,
        'loss_ponder': model_ponder_loss.detach().item() if isinstance(model_ponder_loss, torch.Tensor) else model_ponder_loss,
        'loss_step_efficiency': step_efficiency_loss.detach().item() if isinstance(step_efficiency_loss, torch.Tensor) else step_efficiency_loss,
    }

    return total_loss, metrics
