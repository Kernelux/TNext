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
    model_num_colors: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute core task losses: Color CrossEntropy and Size Error.
    Supports deep supervision via pass_logits_list.
    """
    total_color_loss = 0.0
    targets = test_output.view(-1)
    
    # If using deep supervision, average loss across all passes
    # defaulting to just using final logits if pass_logits_list is empty/single
    logits_to_process = pass_logits_list if pass_logits_list else [logits]
    
    for p_logits in logits_to_process:
        p_logits_flat = p_logits.reshape(-1, model_num_colors)
        total_color_loss += F.cross_entropy(p_logits_flat, targets, ignore_index=-1)
        
    color_loss = total_color_loss / len(logits_to_process)
    
    true_h = output_size[:, 0]
    true_w = output_size[:, 1]
    
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
    Trains the halt_net to predict correctness of the current state.
    """
    model_q_loss = torch.tensor(0.0, device=device)
    features = config.features
    
    if features.use_explicit_q_head and 'halt_probs' in aux_info and aux_info['halt_probs']:
        halt_probs = aux_info['halt_probs']
        pass_logits_list = aux_info.get('pass_logits', [])
        
        num_passes_with_probs = min(len(halt_probs), len(pass_logits_list))
        
        for i in range(num_passes_with_probs):
            p_logits = pass_logits_list[i]
            preds = p_logits.detach().argmax(dim=-1)
            # Correct if all cells match target
            is_correct = (preds == test_output).all(dim=-1).all(dim=-1).float()
            p_halt = halt_probs[i]
            # BCE: predicted halt prob should match is_correct (1.0 or 0.0)
            model_q_loss += F.binary_cross_entropy(p_halt, is_correct.detach())
        
        if num_passes_with_probs > 0:
            model_q_loss = model_q_loss / num_passes_with_probs
            
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
    [LAYER STEP EFFICIENCY]
    Minimize steps when correct, maximize steps when wrong (to encourage finding solution).
    """
    step_efficiency_loss = torch.tensor(0.0, device=device)
    features = config.features
    
    if features.use_layer_act and 'expected_steps' in aux_info and aux_info['expected_steps']:
        # Compute is_correct from final logits
        final_preds = logits.detach().argmax(dim=-1)
        is_correct = (final_preds == test_output).all(dim=-1).all(dim=-1).float()  # [B]
        
        # Stack all expected steps with numerical stability
        expected_list = []
        for es in aux_info['expected_steps']:
            # Clamp to valid range to prevent NaN
            es_clamped = es.sum(dim=-1).clamp(min=1e-6, max=1e6)
            expected_list.append(es_clamped)
        
        if not expected_list:
            return step_efficiency_loss
            
        all_expected = torch.cat(expected_list, dim=0)  # [num_passes * B]
        is_correct_expanded = is_correct.repeat(len(aux_info['expected_steps']))  # [num_passes * B]
        
        max_recurrent_steps = getattr(config, 'max_recurrent_steps', 4)
        max_total_steps = float(max_recurrent_steps * model_num_layers)
        
        # Loss: correct → minimize steps, wrong → maximize steps (search harder)
        # Clamp the loss components to prevent extreme values
        minimize_term = (is_correct_expanded * all_expected).clamp(max=max_total_steps)
        maximize_term = ((1 - is_correct_expanded) * (max_total_steps - all_expected)).clamp(min=0, max=max_total_steps)
        
        step_efficiency_loss = (minimize_term + maximize_term).mean()
        
        # Normalize by max possible (with epsilon for safety)
        step_efficiency_loss = step_efficiency_loss / (max_total_steps + 1e-8)
        
        # Final safety clamp
        step_efficiency_loss = step_efficiency_loss.clamp(min=0, max=10.0)
        
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
    pass_logits_list = aux_info.get('pass_logits', [logits])
    task_loss, color_loss, size_loss = compute_task_loss(
        logits, size_logits, test_output, output_size, pass_logits_list, model.num_colors
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
    
    # Weighted Sum
    compute_loss = (
        config.lambda_q_head * model_q_loss +
        config.lambda_ponder * model_ponder_loss +
        config.lambda_step_efficiency * step_efficiency_loss
    )
    
    total_loss = task_loss + compute_loss + config.lambda_diversity * diversity_loss
    
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
