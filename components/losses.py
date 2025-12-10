import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .config import TrainingConfig
from .dataset import VOCAB_SIZE


def compute_gate_polarization_loss(
    read_gates: List[torch.Tensor],
    write_gates: List[torch.Tensor],
) -> torch.Tensor:
    """
    Gate Polarization Loss - encourages gates to learn decisive values.
    
    Without this, gates tend to stay at ~0.5 (sigmoid(0)) because:
    1. Initial bias = 0 → sigmoid(0) = 0.5
    2. Main loss doesn't directly depend on gate values
    3. Model can learn around 0.5 scaling
    
    This loss encourages gates to be CLOSE TO 0 or 1 (not stuck at 0.5).
    
    L_polar = E[4 * g * (1 - g)]  
    - Maximum at g=0.5 (loss=1.0)
    - Minimum at g=0 or g=1 (loss=0.0)
    
    This creates gradient pressure to push gates away from 0.5.
    """
    if not read_gates and not write_gates:
        return torch.tensor(0.0)
    
    all_gates = []
    
    for g in read_gates:
        if g is not None and g.numel() > 0:
            all_gates.append(g.view(-1))
    
    for g in write_gates:
        if g is not None and g.numel() > 0:
            all_gates.append(g.view(-1))
    
    if not all_gates:
        return torch.tensor(0.0)
    
    gates = torch.cat(all_gates)  # Flatten all gates
    
    # Polarization loss: 4 * g * (1 - g)
    # This is maximized at g=0.5 (=1.0) and minimized at g=0 or g=1 (=0.0)
    polar_loss = 4.0 * gates * (1.0 - gates)
    
    return polar_loss.mean()


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


def compute_output_entropy_loss(logits: torch.Tensor, test_output: torch.Tensor) -> torch.Tensor:
    """
    Output Entropy Regularization - prevents mode collapse.
    
    When the model predicts the same class for all positions, the softmax
    becomes saturated and gradients vanish. This loss PENALIZES low entropy
    (overconfident wrong predictions) to keep gradients flowing.
    
    Returns a POSITIVE loss that should be minimized.
    """
    # Get probability distribution over classes
    log_probs = F.log_softmax(logits, dim=-1)  # [B, S, V]
    probs = torch.exp(log_probs)
    
    # Entropy per position: -sum(p * log(p))
    # Higher entropy = more uniform = less confident
    # Max entropy = log(vocab_size) ≈ 2.6 for 14 classes
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, S]
    
    # Mask for valid positions
    valid_mask = (test_output != IGNORE_LABEL).float()  # [B, S]
    num_valid = valid_mask.sum() + 1e-8
    
    # Average entropy across all valid positions
    avg_entropy = (entropy * valid_mask).sum() / num_valid
    
    # Target: we want entropy to be at least 1.0 (not too confident)
    # If entropy < 1.0, penalize; if entropy >= 1.0, no penalty
    # This prevents the model from becoming overconfident too quickly
    min_entropy_target = 1.0
    entropy_deficit = F.relu(min_entropy_target - avg_entropy)
    
    # Scale up to make it meaningful (entropy_deficit is in [0, 1])
    return entropy_deficit * 0.5


def compute_prediction_diversity_loss(logits: torch.Tensor, test_output: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Prediction Diversity Loss - prevents the model from collapsing to predict
    a single class for all positions.
    
    Encourages the marginal distribution over predictions to be diverse,
    matching roughly the target distribution.
    
    This provides strong gradient signal through the entire model.
    """
    # Get predicted probabilities
    probs = F.softmax(logits, dim=-1)  # [B, S, V]
    
    # Marginal distribution: average prediction across all positions
    valid_mask = (test_output != IGNORE_LABEL).unsqueeze(-1).float()  # [B, S, 1]
    masked_probs = probs * valid_mask
    num_valid = valid_mask.sum(dim=(0, 1)) + 1e-8  # [1]
    
    # Average prediction distribution (what the model predicts on average)
    marginal_pred = masked_probs.sum(dim=(0, 1)) / num_valid  # [V]
    
    # Target distribution: frequency of each class in targets
    valid_targets = test_output[test_output != IGNORE_LABEL]
    if valid_targets.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    
    target_counts = torch.bincount(valid_targets.clamp(min=0), minlength=vocab_size).float()
    target_dist = target_counts / (target_counts.sum() + 1e-8)  # [V]
    
    # KL divergence: target || pred  (encourage pred to match target distribution)
    # Using log(pred) for numerical stability
    log_marginal = torch.log(marginal_pred + 1e-8)
    kl_div = (target_dist * (torch.log(target_dist + 1e-8) - log_marginal)).sum()
    
    # Clamp to prevent explosion
    return kl_div.clamp(max=5.0)

# TinyRecursiveModels convention: -100 means ignore this position in loss
IGNORE_LABEL = -100

def compute_class_weights(targets: torch.Tensor, num_classes: int, ignore_label: int = -100) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle class imbalance.
    
    This prevents mode collapse where the model predicts only the most common class.
    Weight_c = 1 / sqrt(count_c + 1) normalized so mean weight = 1.
    Using sqrt for softer reweighting (full inverse would over-correct).
    """
    device = targets.device
    
    # Count valid targets only
    valid_mask = (targets != ignore_label)
    valid_targets = targets[valid_mask]
    
    if valid_targets.numel() == 0:
        return torch.ones(num_classes, device=device)
    
    # Count per class
    counts = torch.bincount(valid_targets.clamp(min=0), minlength=num_classes).float()
    
    # Inverse sqrt frequency (softer than full inverse)
    weights = 1.0 / torch.sqrt(counts + 1.0)
    
    # Normalize so mean weight = 1 (doesn't change loss scale)
    weights = weights * num_classes / weights.sum()
    
    return weights

def compute_task_loss(
    logits: torch.Tensor,
    test_output: torch.Tensor,
    pass_logits_list: List[torch.Tensor],
    vocab_size: int = VOCAB_SIZE,
) -> torch.Tensor:
    """
    Compute task loss: standard sequence cross-entropy.
    
    Simplified for sequence format:
    - logits: [B, S, vocab_size]
    - test_output: [B, S] with IGNORE_LABEL (-100) for padding
    
    Supports deep supervision via pass_logits_list.
    Class-balanced weights prevent mode collapse to dominant tokens.
    """
    device = logits.device
    
    # If using deep supervision, average loss across all passes
    logits_to_process = pass_logits_list if pass_logits_list else [logits]
    
    # Compute class weights for this batch (inverse frequency)
    class_weights = compute_class_weights(test_output, vocab_size, IGNORE_LABEL)
    
    total_loss = torch.tensor(0.0, device=device)
    
    for p_logits in logits_to_process:
        # Flatten for cross-entropy: [B*S, vocab_size] and [B*S]
        logits_flat = p_logits.reshape(-1, vocab_size)
        targets_flat = test_output.reshape(-1)
        
        # Use ignore_index=-100 to skip padded positions
        # Class weights prevent collapse to dominant token
        # LABEL SMOOTHING (0.1) prevents softmax saturation and keeps gradients flowing
        total_loss = total_loss + F.cross_entropy(
            logits_flat, targets_flat, 
            weight=class_weights,
            ignore_index=IGNORE_LABEL,
            label_smoothing=0.1  # Key fix for gradient flow!
        )

    color_loss = total_loss / len(logits_to_process)
    
    return color_loss

def compute_q_head_loss(
    aux_info: Dict,
    test_output: torch.Tensor,
    config: TrainingConfig,
    device: torch.device
) -> torch.Tensor:
    """
    Q-Head Loss (TRM-style Adaptive Computation Time).
    
    Trains the dual Q-head to predict correctness for halting:
    - q_halt: BCE(q_halt, is_correct) - learns to say "stop, this is right"
    - q_continue: BCE(q_continue, is_wrong) - learns to say "continue, this is wrong"
                  (optional, controlled by no_act_continue flag)
    
    This helps the model learn when to stop iterating vs when to continue.
    
    Now works with sequences: test_output is [B, S] with IGNORE_LABEL for padding.
    """
    model_q_loss = torch.tensor(0.0, device=device)
    features = config.features
    
    # Get Q-head outputs (now always dual mode: q_halt and q_continue are LOGITS)
    q_halt_list = aux_info.get('q_halt', [])
    q_continue_list = aux_info.get('q_continue', [])
    
    if not q_halt_list:
        return model_q_loss
    
    # Get pass logits for computing per-pass correctness
    pass_logits = aux_info.get('pass_logits', [])
    final_logits = aux_info.get('final_logits')
    
    # no_act_continue: Paper recommends skipping Q_continue loss
    # "No continue ACT loss, only use the sigmoid of the halt which makes much more sense"
    use_q_continue = not features.no_act_continue
    
    # If we have per-pass logits (deep supervision), compute per-pass loss
    if pass_logits:
        total_loss = torch.tensor(0.0, device=device)
        for i, (q_h_logit, q_c_logit, p_logits) in enumerate(zip(q_halt_list, q_continue_list, pass_logits)):
            # Per-pass correctness (only in valid positions, not padding)
            # p_logits: [B, S, vocab_size], test_output: [B, S]
            preds = p_logits.detach().argmax(dim=-1)  # [B, S]
            valid_mask = (test_output != IGNORE_LABEL)  # [B, S]
            # Compute per-sample correctness: all valid positions must match
            correct_at_valid = (preds == test_output) | ~valid_mask  # correct or don't care
            is_correct = correct_at_valid.all(dim=-1).float()  # [B]
            
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
        
        preds = final_logits.detach().argmax(dim=-1)  # [B, S]
        valid_mask = (test_output != IGNORE_LABEL)  # [B, S]
        correct_at_valid = (preds == test_output) | ~valid_mask
        is_correct = correct_at_valid.all(dim=-1).float()  # [B]
        
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

def compute_step_efficiency_loss(
    logits: torch.Tensor,
    aux_info: Dict,
    test_output: torch.Tensor,
    config: TrainingConfig,
    model_num_layers: int,
    device: torch.device
) -> torch.Tensor:
    """
    [LAYER STEP EFFICIENCY - Error * Steps formulation]
    
    Penalize steps proportional to error rate:
    - High error + high steps = worst (wasted computation, still wrong)
    - High error + low steps = OK (fail fast)
    - Low error + high steps = better (correct, but inefficient)
    - Low error + low steps = best (efficient and correct)
    
    loss = error_rate * normalized_steps
    
    This creates intuitive behavior:
    - If wrong anyway, at least fail fast
    - If correct, be as efficient as possible
    
    Now works with sequences: test_output and logits are [B, S].
    """
    step_efficiency_loss = torch.tensor(0.0, device=device)
    features = config.features

    if features.use_layer_act and 'expected_steps' in aux_info and aux_info['expected_steps']:
        # Compute token-level error rate using SOFT probabilities (not detached argmax!)
        # This allows gradients to flow back through the step_predictor
        # logits: [B, S, vocab_size], test_output: [B, S]
        valid_mask = (test_output != IGNORE_LABEL)  # [B, S]
        
        # Use soft cross-entropy per token as error signal (differentiable!)
        # Lower CE = more confident correct prediction
        vocab_size = logits.shape[-1]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, S, V]
        
        # Gather log-prob of target class at each position
        # Clamp test_output to valid range for gather (ignore positions get 0 anyway)
        target_clamped = test_output.clamp(min=0, max=vocab_size-1)  # [B, S]
        target_log_probs = log_probs.gather(dim=-1, index=target_clamped.unsqueeze(-1)).squeeze(-1)  # [B, S]
        
        # Mask invalid positions and compute mean negative log-prob (like cross-entropy)
        # Higher value = higher error (wrong predictions)
        masked_nll = -target_log_probs * valid_mask.float()  # [B, S]
        total_valid = valid_mask.float().sum(dim=-1).clamp(min=1)  # [B]
        error_signal = masked_nll.sum(dim=-1) / total_valid  # [B] - soft error rate
        
        # Normalize error to [0, 1] range (log_probs are negative, so NLL is positive)
        # Max NLL is ~log(vocab_size) for uniform distribution
        max_nll = float(torch.log(torch.tensor(vocab_size, dtype=torch.float32)))
        error_rate = (error_signal / max_nll).clamp(0, 1)  # [B] in [0, 1]

        # Collect expected steps across all passes
        total_expected_steps = torch.zeros_like(error_rate)  # [B]
        num_passes = len(aux_info['expected_steps'])
        
        for es in aux_info['expected_steps']:  # es: [B, num_layers]
            total_expected_steps += es.sum(dim=1)  # Sum across layers
        
        # Normalize by max possible steps
        max_recurrent_steps = getattr(config, 'max_recurrent_steps', 4)
        max_total_steps = float(max_recurrent_steps * model_num_layers * num_passes)
        normalized_steps = total_expected_steps / (max_total_steps + 1e-8)  # [B] in [0, 1]
        
        # Core formulation: error * steps
        # High error + high steps = high loss (wasted computation)
        # Low error + low steps = low loss (efficient success)
        # BOTH error_rate and normalized_steps now have gradients!
        per_sample_loss = error_rate.detach() * normalized_steps  # error is target, steps learn from it
        
        # Also add small direct penalty on steps to encourage efficiency even when correct
        # This gives step_predictor gradient even when error is 0
        step_penalty = 0.1 * normalized_steps.mean()
        
        # Average across batch
        step_efficiency_loss = per_sample_loss.mean() + step_penalty
        
        # Clamp for safety
        step_efficiency_loss = step_efficiency_loss.clamp(min=0.0, max=2.0)

    return step_efficiency_loss


def compute_total_loss(
    model,
    logits: torch.Tensor,
    test_output: torch.Tensor,
    aux_info: Dict,
    config: TrainingConfig,
    global_step: int,
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Aggregates all losses.
    Returns total_loss and a dictionary of scalar loss components for logging.
    
    Simplified for sequence format:
    - logits: [B, S, vocab_size]
    - test_output: [B, S] with IGNORE_LABEL for padding
    """
    # 1. Task Loss
    # Only use deep supervision if enabled (memory intensive)
    features = config.features
    if features.use_deep_supervision:
        pass_logits_list = aux_info.get('pass_logits', [logits])
    else:
        pass_logits_list = [logits]  # Only final output

    task_loss = compute_task_loss(
        logits, test_output, pass_logits_list,
        model.vocab_size
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

    # 5. Output Entropy Loss - prevents mode collapse and provides gradient to embeddings/output
    output_entropy_loss = compute_output_entropy_loss(logits, test_output)
    
    # 6. Prediction Diversity Loss - encourages diverse predictions matching target distribution
    pred_diversity_loss = compute_prediction_diversity_loss(logits, test_output, model.vocab_size)

    # 7. Gate Polarization Loss - prevents gates from staying stuck at 0.5
    gate_polar_loss = torch.tensor(0.0, device=device)
    read_gates = aux_info.get('read_gates', [])
    write_gates = aux_info.get('write_gates', [])
    if read_gates or write_gates:
        gate_polar_loss = compute_gate_polarization_loss(read_gates, write_gates)

    # Weighted Sum with NaN protection
    # NOTE: Ponder loss removed - dual Q-head supersedes it
    compute_loss = (
        config.lambda_q_head * model_q_loss +
        config.lambda_step_efficiency * step_efficiency_loss
    )
    
    # Gradient flow losses (help prevent vanishing gradients to embeddings/output)
    # Reduced weights to prevent gradient explosion in deep recurrent networks
    gradient_flow_loss = output_entropy_loss + 0.05 * pred_diversity_loss
    
    # Gate polarization: small weight to encourage decisive gates without dominating
    gate_loss = 0.01 * gate_polar_loss

    total_loss = task_loss + compute_loss + config.lambda_diversity * diversity_loss + gradient_flow_loss + gate_loss

    # NaN protection: if any component is NaN, use only task_loss
    if torch.isnan(total_loss):
        print(f"WARNING: NaN detected! task={task_loss.item():.4f}, q_head={model_q_loss.item():.4f}, "
              f"step_eff={step_efficiency_loss.item():.4f}, diversity={diversity_loss.item():.4f}, "
              f"entropy={output_entropy_loss.item():.4f}, pred_div={pred_diversity_loss.item():.4f}")
        total_loss = task_loss  # Fallback to just task loss

    # Return metrics for logging
    metrics = {
        'loss_total': total_loss.detach().item(),
        'loss_task': task_loss.detach().item(),
        'loss_compute': compute_loss.detach().item() if isinstance(compute_loss, torch.Tensor) else compute_loss,
        'loss_diversity': diversity_loss.detach().item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
        'loss_q_head': model_q_loss.detach().item() if isinstance(model_q_loss, torch.Tensor) else model_q_loss,
        'loss_step_efficiency': step_efficiency_loss.detach().item() if isinstance(step_efficiency_loss, torch.Tensor) else step_efficiency_loss,
        'loss_output_entropy': output_entropy_loss.detach().item() if isinstance(output_entropy_loss, torch.Tensor) else output_entropy_loss,
        'loss_pred_diversity': pred_diversity_loss.detach().item() if isinstance(pred_diversity_loss, torch.Tensor) else pred_diversity_loss,
        'loss_gate_polar': gate_polar_loss.detach().item() if isinstance(gate_polar_loss, torch.Tensor) else gate_polar_loss,
    }

    return total_loss, metrics
