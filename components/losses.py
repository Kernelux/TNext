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


def compute_feedback_gate_polarization_loss(
    aux_info: Dict,
) -> torch.Tensor:
    """
    Feedback Gate Polarization Loss - encourages answer/iteration gates to be decisive.
    
    These feedback gates control:
    - answer_feedback: How much to inject answer prototype back into processing
    - iteration_feedback: How much to inject layer thoughts between iterations
    
    If these stay at ~0.5, the model can't learn to selectively use feedback.
    This loss pushes them toward 0 or 1 for decisive behavior.
    
    L_polar = 4 * g * (1 - g)  (same formula as memory gates)
    """
    all_gates = []
    
    # Collect answer feedback gate tensors
    answer_gates = aux_info.get('answer_feedback_gates', [])
    for g in answer_gates:
        if g is not None and isinstance(g, torch.Tensor) and g.numel() > 0:
            all_gates.append(g.view(-1))
    
    # Collect iteration feedback gate tensors
    iter_gates = aux_info.get('iteration_feedback_gates', [])
    for g in iter_gates:
        if g is not None and isinstance(g, torch.Tensor) and g.numel() > 0:
            all_gates.append(g.view(-1))
    
    if not all_gates:
        return torch.tensor(0.0)
    
    gates = torch.cat(all_gates)
    
    # Same polarization formula: maximum at 0.5, minimum at 0 or 1
    polar_loss = 4.0 * gates * (1.0 - gates)
    
    return polar_loss.mean()


def compute_read_gate_sparsity_loss(
    read_gates: List[torch.Tensor],
    write_gates: List[torch.Tensor],
    read_target_ratio: float = 0.4,
    write_target_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Gate Sparsity Loss - prevents read/write gates from saturating to 1.0.
    
    Without this, the model finds shortcuts:
    - Read gates all → 1: always read from cache (bypasses selective memory)
    - Write gates all → 1: always overwrite cache (destroys learned patterns)
    
    The loss penalizes deviation from target activation ratios:
    L = (read_ratio - read_target)^2 + (write_ratio - write_target)^2
    
    Args:
        read_gates: List of read gate tensors from each layer/pass
        write_gates: List of write gate tensors from each layer/pass
        read_target_ratio: Target fraction of read gates open (default 0.4 = 40%)
        write_target_ratio: Target fraction of write gates open (default 0.3 = 30%)
                           Lower than read because writes should be more selective
    
    Returns:
        Scalar loss penalizing deviation from target activation ratios
    """
    total_loss = torch.tensor(0.0)
    
    # Read gate sparsity
    if read_gates:
        all_read_gates = []
        for g in read_gates:
            if g is not None and g.numel() > 0:
                all_read_gates.append(g.view(-1))
        
        if all_read_gates:
            gates = torch.cat(all_read_gates)
            actual_ratio = gates.mean()
            total_loss = total_loss + (actual_ratio - read_target_ratio) ** 2
    
    # Write gate sparsity (more selective - lower target)
    if write_gates:
        all_write_gates = []
        for g in write_gates:
            if g is not None and g.numel() > 0:
                all_write_gates.append(g.view(-1))
        
        if all_write_gates:
            gates = torch.cat(all_write_gates)
            actual_ratio = gates.mean()
            total_loss = total_loss + (actual_ratio - write_target_ratio) ** 2
    
    return total_loss


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


# ============================================================================
# Confidence & Ponder-Based Losses
# ============================================================================

def compute_confidence_calibration_loss(
    aux_info: Dict,
    test_output: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Confidence Calibration Loss - aligns confidence with actual correctness.
    
    The model's confidence should correlate with actual prediction accuracy.
    If the model is 80% confident, it should be ~80% correct.
    
    This is crucial for:
    1. Meaningful halt decisions (halt when actually correct, not just confident)
    2. Trust calibration (confidence means something)
    3. Better exploration (low confidence = need more passes)
    
    Loss: BCE(confidence, per_position_correctness)
    """
    device = test_output.device
    
    # Get final confidence
    final_confidence = aux_info.get('final_confidence')
    pass_logits = aux_info.get('pass_logits', [])
    
    if final_confidence is None or not pass_logits:
        return torch.tensor(0.0, device=device)
    
    # Use final logits
    final_logits = pass_logits[-1]  # [B, S, V]
    
    # NaN protection - if logits are NaN, skip this loss
    if torch.isnan(final_logits).any() or torch.isnan(final_confidence).any():
        return torch.tensor(0.0, device=device)
    
    # Compute per-position correctness (ground truth for confidence)
    valid_mask = (test_output != IGNORE_LABEL)  # [B, S]
    predictions = final_logits.argmax(dim=-1)  # [B, S]
    
    # Correctness: 1 where correct, 0 where wrong
    correct = ((predictions == test_output) & valid_mask).float()  # [B, S]
    
    # Confidence is [B, S] or [B] - need to handle both
    if final_confidence.dim() == 1:
        # Per-sample confidence: compare to sample-level correctness
        sample_correct = correct.sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1)  # [B]
        loss = F.binary_cross_entropy(
            final_confidence.clamp(0.01, 0.99),
            sample_correct.detach(),
            reduction='mean'
        )
    else:
        # Per-position confidence
        # Only compute loss on valid positions
        if valid_mask.any():
            conf_flat = final_confidence[valid_mask].clamp(0.01, 0.99)
            correct_flat = correct[valid_mask].detach()
            loss = F.binary_cross_entropy(conf_flat, correct_flat, reduction='mean')
        else:
            loss = torch.tensor(0.0, device=device)
    
    return loss


def compute_confidence_monotonicity_loss(
    aux_info: Dict,
) -> torch.Tensor:
    """
    Confidence Monotonicity Loss - later passes should have higher/equal confidence.
    
    Intuition: As the model refines its answer, confidence should increase.
    If confidence drops between passes, something is wrong (the model got worse).
    
    This regularizes the confidence estimator to produce sensible values and
    encourages the model to actually improve with more computation.
    
    Loss: max(0, conf[i] - conf[i+1])^2 for each consecutive pair
    """
    pass_confidences = aux_info.get('pass_confidences', [])
    
    if len(pass_confidences) < 2:
        return torch.tensor(0.0)
    
    device = pass_confidences[0].device
    total_loss = torch.tensor(0.0, device=device)
    
    for i in range(len(pass_confidences) - 1):
        conf_current = pass_confidences[i].mean()
        conf_next = pass_confidences[i + 1].mean()
        
        # Penalize when confidence decreases (conf_current > conf_next)
        violation = F.relu(conf_current - conf_next)
        total_loss = total_loss + violation ** 2
    
    return total_loss / (len(pass_confidences) - 1)


def compute_adaptive_ponder_cost(
    aux_info: Dict,
    test_output: torch.Tensor,
    max_passes: int,
) -> torch.Tensor:
    """
    Adaptive Ponder Cost - penalizes computation based on task difficulty.
    
    Unlike simple ponder_cost = passes/max_passes, this considers:
    1. Confidence: High confidence but many passes = wasteful
    2. Correctness: Wrong answer with many passes = very wasteful
    3. Early correct: Reward halting early when already correct
    
    Key insight: The cost of computation should scale with redundancy.
    - If confident after pass 1 but ran 3 passes: high cost
    - If needed all passes to get confident: low cost
    - If still wrong after all passes: some cost (tried hard)
    
    Loss formula:
      ponder_cost = (passes_used / max) * (1 - difficulty_adjusted_need)
    
    Where difficulty_adjusted_need considers how confidence evolved.
    """
    device = test_output.device
    
    passes_run = aux_info.get('passes_run', 1)
    pass_confidences = aux_info.get('pass_confidences', [])
    pass_logits = aux_info.get('pass_logits', [])
    
    if not pass_confidences or not pass_logits:
        # Fallback to simple ponder cost
        return torch.tensor(passes_run / max_passes, device=device)
    
    # Base utilization
    pass_utilization = passes_run / max_passes
    
    # Find first pass where confidence exceeded threshold
    # This tells us "when could we have stopped"
    threshold = 0.8  # Standard confidence threshold
    first_confident_pass = passes_run  # Default: never confident
    
    for i, conf in enumerate(pass_confidences):
        if conf.mean().item() > threshold:
            first_confident_pass = i + 1
            break
    
    # Redundant passes = passes after we were already confident
    redundant_ratio = max(0, passes_run - first_confident_pass) / max_passes
    
    # Check if final answer is correct
    if pass_logits:
        final_logits = pass_logits[-1]
        valid_mask = (test_output != IGNORE_LABEL)
        predictions = final_logits.argmax(dim=-1)
        correct = ((predictions == test_output) | ~valid_mask).all(dim=-1)  # [B]
        correctness = correct.float().mean()
    else:
        correctness = torch.tensor(0.5, device=device)  # Unknown
    
    # Adaptive ponder cost:
    # - High when: many redundant passes
    # - Low when: needed all passes (never confident early)
    # - Extra penalty when: wrong AND many passes (wasted compute)
    
    wrong_penalty = (1 - correctness) * 0.5  # Extra cost when wrong
    
    ponder_cost = pass_utilization * 0.3 + redundant_ratio * 0.5 + wrong_penalty * 0.2
    
    return ponder_cost.clamp(0, 1)

def compute_q_halt_loss(
    aux_info: Dict,
    test_output: torch.Tensor,
    ignore_label: int = -1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Q-Halt Loss (TRM-inspired) - trains the halt head to predict sequence correctness.
    
    Key insight from TRM: The halt decision should be trained with supervision!
    - During training, we KNOW if the answer is correct
    - Train q_halt to output: >0 if correct, <0 if wrong
    - Loss: BCE(sigmoid(q_halt), seq_is_correct)
    
    This is much smarter than entropy-based halting because:
    1. High confidence ≠ correct answer (model can be confidently wrong)
    2. Direct supervision: correctness is the actual signal we care about
    3. Learns to recognize "what does a wrong answer look like?"
    
    Args:
        aux_info: Dict containing 'q_halt_logits_list' and 'pass_logits'
        test_output: [B, S] ground truth tokens
        ignore_label: Token to ignore in correctness computation
        
    Returns:
        q_halt_loss: Scalar loss (BCE against correctness)
        metrics: Dict with debugging info (q_halt_accuracy, etc.)
    """
    device = test_output.device
    
    q_halt_logits_list = aux_info.get('q_halt_logits_list', [])
    pass_logits = aux_info.get('pass_logits', [])
    
    if not q_halt_logits_list or not pass_logits:
        return torch.tensor(0.0, device=device), {}
    
    # Ensure we have same number of q_halt logits as pass logits
    n_passes = min(len(q_halt_logits_list), len(pass_logits))
    
    total_loss = torch.tensor(0.0, device=device)
    total_q_halt_accuracy = 0.0
    valid_mask = (test_output != ignore_label)  # [B, S]
    
    for i in range(n_passes):
        q_halt = q_halt_logits_list[i]  # [B] - raw logits
        logits = pass_logits[i]  # [B, S, V]
        
        # Compute sequence-level correctness at this pass
        predictions = logits.argmax(dim=-1)  # [B, S]
        correct_positions = (predictions == test_output) | ~valid_mask  # [B, S]
        seq_correct = correct_positions.all(dim=-1).float()  # [B] - 1 if all correct, 0 otherwise
        
        # BCE loss: train q_halt to predict correctness
        # q_halt > 0 should mean "answer is correct" (halt)
        # q_halt < 0 should mean "answer is wrong" (continue)
        # Clamp q_halt to prevent extreme logits causing NaN
        q_halt_clamped = q_halt.clamp(min=-20.0, max=20.0)
        loss = F.binary_cross_entropy_with_logits(
            q_halt_clamped,
            seq_correct,
            reduction='mean'
        )
        
        # Check for NaN and skip if present
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf in Q-halt loss at pass {i}. Skipping this pass.")
            continue
            
        total_loss = total_loss + loss
        
        # Track accuracy of halt prediction
        with torch.no_grad():
            halt_decision = (q_halt > 0).float()
            q_halt_correct = (halt_decision == seq_correct).float().mean().item()
            total_q_halt_accuracy += q_halt_correct
    
    # Average over passes
    avg_loss = total_loss / n_passes
    avg_accuracy = total_q_halt_accuracy / n_passes
    
    metrics = {
        'q_halt_accuracy': avg_accuracy,
        'n_passes_with_q_halt': n_passes,
    }
    
    return avg_loss, metrics


def compute_model_info_gain_loss(
    aux_info: Dict,
    maximize: bool = True,
    decay: float = 0.7,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Model-Level Info Gain Loss - encourages productive passes.
    
    At model level, we have actual logits so info_gain = entropy reduction
    between consecutive passes. This is a direct measure of progress.
    
    Combined with Q-halt, this enables smart halting:
    - q_halt > 0 (thinks correct) → halt
    - q_halt < 0 + high info_gain → continue (making progress)
    - q_halt < 0 + low info_gain → abort (stuck, wasting compute)
    
    Args:
        aux_info: Contains 'pass_info_gains' list of scalars
        maximize: If True, maximize info gain (loss = -info_gain)
        decay: Weight decay for later passes (earlier passes matter more)
    
    Returns:
        loss: Scalar loss (negative if maximizing = reward for high info gain)
        metrics: Dict with info gain stats
    """
    pass_info_gains = aux_info.get('pass_info_gains', [])
    
    if not pass_info_gains:
        return torch.tensor(0.0), {'avg_model_info_gain': 0.0}
    
    # Get device from first info gain if it's a tensor
    device = 'cpu'
    for ig in pass_info_gains:
        if isinstance(ig, torch.Tensor):
            device = ig.device
            break
    
    total_loss = torch.tensor(0.0, device=device)
    total_info_gain = 0.0
    
    for i, info_gain in enumerate(pass_info_gains):
        # Convert to tensor if needed
        if not isinstance(info_gain, torch.Tensor):
            ig_tensor = torch.tensor(info_gain, device=device)
        else:
            ig_tensor = info_gain
        
        # Weight: earlier passes should contribute more
        weight = decay ** i  # 1.0, 0.7, 0.49, ...
        
        if maximize:
            # Maximize info gain: loss = -info_gain
            total_loss = total_loss - weight * ig_tensor.abs()
        else:
            # Just track, no loss
            pass
        
        # Detach for logging to avoid warning about .item() on gradient tensor
        total_info_gain += ig_tensor.detach().abs().item() if isinstance(ig_tensor, torch.Tensor) else abs(info_gain)
    
    avg_info_gain = total_info_gain / len(pass_info_gains)
    
    # Normalize by number of passes
    if len(pass_info_gains) > 0:
        total_loss = total_loss / len(pass_info_gains)
    
    metrics = {
        'avg_model_info_gain': avg_info_gain,
        'num_passes_with_info_gain': len(pass_info_gains),
    }
    
    return total_loss, metrics


def should_abort_refinement(
    q_halt_logits: torch.Tensor,  # [B] - Q-halt prediction
    info_gain: float,              # Current info gain
    info_gain_threshold: float = 0.01,  # Min info gain to continue
) -> torch.Tensor:
    """
    Combined halt decision: Q-halt + Info Gain.
    
    Logic:
    - q_halt > 0 (thinks correct) → halt (True)
    - q_halt < 0 + info_gain >= threshold → continue (False) 
    - q_halt < 0 + info_gain < threshold → abort (True) - stuck!
    
    This prevents wasting compute when:
    1. Answer seems wrong AND
    2. We're not making progress
    
    Args:
        q_halt_logits: [B] Q-halt logits (>0 = halt)
        info_gain: Scalar info gain for this pass
        info_gain_threshold: Minimum info gain to justify continuing
    
    Returns:
        should_halt: [B] boolean tensor
    """
    # Q-halt decision: >0 means halt
    q_halt_says_halt = q_halt_logits > 0  # [B]
    
    # Info gain decision: low info gain = abort
    low_info_gain = info_gain < info_gain_threshold
    
    # Combined: halt if q_halt says so, OR if wrong + stuck
    # q_halt < 0 means "thinks wrong", combined with low info gain = abort
    q_halt_says_wrong = q_halt_logits < 0  # [B]
    stuck_and_wrong = q_halt_says_wrong & low_info_gain  # [B] (broadcasts scalar)
    
    should_halt = q_halt_says_halt | stuck_and_wrong
    
    return should_halt


def compute_early_exit_bonus(
    aux_info: Dict,
    test_output: torch.Tensor,
    max_passes: int,
) -> torch.Tensor:
    """
    Early Exit Bonus - reward halting early when correct.
    
    This creates a direct incentive structure:
    - Halt early + correct = REWARD (negative loss)
    - Halt early + wrong = no reward
    - Use all passes + correct = small reward
    - Use all passes + wrong = no reward
    
    The bonus is: -bonus_scale * (1 - pass_utilization) * correctness
    
    This is NEGATIVE loss (reward) for efficient correct answers.
    """
    device = test_output.device
    
    passes_run = aux_info.get('passes_run', 1)
    pass_logits = aux_info.get('pass_logits', [])
    halted_early = aux_info.get('halted_early', False)
    
    if not pass_logits:
        return torch.tensor(0.0, device=device)
    
    # Check final correctness
    final_logits = pass_logits[-1]
    valid_mask = (test_output != IGNORE_LABEL)
    predictions = final_logits.argmax(dim=-1)
    correct_positions = (predictions == test_output) | ~valid_mask
    seq_correct = correct_positions.all(dim=-1).float()  # [B]
    
    # Savings from early halt
    passes_saved = max_passes - passes_run
    savings_ratio = passes_saved / max_passes
    
    # Bonus only if halted early AND correct
    if halted_early:
        bonus = -0.1 * savings_ratio * seq_correct.mean()  # Negative = reward
    else:
        bonus = torch.tensor(0.0, device=device)
    
    return bonus


def compute_threshold_supervision_loss(
    model,
    aux_info: Dict,
    test_output: torch.Tensor,
    config: 'TrainingConfig',
    device: torch.device,
) -> torch.Tensor:
    """
    Threshold Supervision Loss - provides gradients to learned threshold networks.
    
    Supervises both:
    1. Model halt thresholds: should be LOW when answer is correct, HIGH when wrong
    2. Layer halt thresholds: similar logic based on layer-level improvement
    
    This is a SOFT target loss:
    - When correct: threshold should approach confidence (allow halting)
    - When wrong: threshold should be high (prevent premature halting)
    """
    total_loss = torch.tensor(0.0, device=device)
    num_terms = 0
    
    # === Model-Level Threshold Supervision ===
    pass_logits = aux_info.get('pass_logits', [])
    pass_confidences = aux_info.get('pass_confidences', [])
    model_halt_thresholds = aux_info.get('model_halt_thresholds', [])
    
    if pass_logits and pass_confidences and model_halt_thresholds:
        valid_mask = (test_output != IGNORE_LABEL)
        
        for pass_idx, (logits, confidence, halt_thresh) in enumerate(
            zip(pass_logits, pass_confidences, model_halt_thresholds)
        ):
            # Compute correctness at this pass
            predictions = logits.argmax(dim=-1)  # [B, S]
            correct_positions = (predictions == test_output) | ~valid_mask  # [B, S]
            seq_correct = correct_positions.all(dim=-1).float()  # [B]
            
            # Get confidence mean
            conf_mean = confidence.mean(dim=-1) if confidence.dim() > 1 else confidence  # [B]
            
            # Target threshold based on correctness
            margin = 0.1
            target_thresh = torch.where(
                seq_correct > 0.5,
                conf_mean - margin,  # Correct: threshold below confidence → halt allowed
                conf_mean + margin,  # Wrong: threshold above confidence → no halt
            ).clamp(0.1, 0.9).detach()
            
            # MSE loss
            loss = F.mse_loss(halt_thresh.clamp(0.1, 0.9), target_thresh)
            total_loss = total_loss + loss
            num_terms += 1
    
    # === Layer-Level Threshold Supervision ===
    layer_halt_thresholds = aux_info.get('layer_halt_thresholds', [])
    layer_confidences = aux_info.get('layer_confidences', [])
    
    if layer_halt_thresholds and layer_confidences:
        # For layer thresholds, we use a simpler heuristic:
        # Threshold should be slightly below confidence (allow halting when confident)
        # We don't have per-layer correctness, so use weak supervision
        for halt_thresh, confidence in zip(layer_halt_thresholds, layer_confidences):
            if halt_thresh is None or confidence is None:
                continue
            
            conf_mean = confidence.mean(dim=-1) if confidence.dim() > 1 else confidence  # [B]
            
            # Target: threshold slightly below confidence (allow halting)
            # This encourages the model to halt when it becomes confident
            target_thresh = (conf_mean - 0.05).clamp(0.1, 0.9).detach()
            
            loss = F.mse_loss(halt_thresh.clamp(0.1, 0.9), target_thresh)
            total_loss = total_loss + 0.5 * loss  # Lower weight for layer-level
            num_terms += 1
    
    if num_terms > 0:
        total_loss = total_loss / num_terms
    
    return total_loss


def compute_layer_divergence_loss(
    aux_info: Dict,
    min_divergence: float = 0.1,
    divergence_decay: float = 0.5,
    maximize_info_gain: bool = True,
) -> torch.Tensor:
    """
    Layer Divergence Loss - encourages meaningful representation change between iterations.
    
    Problem: Transformer layers with residual connections naturally produce outputs
    very similar to inputs (high cosine similarity). This causes the layer halt
    estimator to trigger immediately (stability > threshold after 1 iteration).
    
    Solution: Two complementary signals:
    
    1. **Cosine Divergence**: Penalize when cos_dist = 1 - cos_sim is too low
       - Early iterations MUST change the representation geometrically
       - Penalty: ReLU(min_divergence - cos_dist)^2
    
    2. **Information Gain Maximization**: Directly maximize |entropy_change|
       - Each iteration should produce maximum information change
       - Loss: -|info_gain| (negative = maximize)
       - This encourages productive iterations without arbitrary thresholds
    
    3. **Iteration Decay**: Divergence requirement decreases with iteration count
       - Iteration 0→1: full requirement
       - Iteration 1→2: requirement * decay
       - Later: can settle
    
    Args:
        aux_info: Contains:
            - 'layer_stabilities': [B] cosine similarities (from LayerHaltEstimator)
            - 'layer_info_gains': [B] entropy changes (from layer)
        min_divergence: Minimum cos_dist for first iteration
        divergence_decay: Reduce requirement by this factor per iteration
        maximize_info_gain: If True, maximize info gain directly instead of threshold
    
    Returns:
        Scalar loss (can be negative when maximizing info gain)
    """
    device = None
    total_loss = torch.tensor(0.0)
    num_terms = 0
    
    # === Part 1: Cosine Divergence (stability-based) ===
    layer_stabilities = aux_info.get('layer_stabilities', [])
    if not layer_stabilities:
        layer_stabilities = aux_info.get('layer_confidences', [])
    
    for i, stability in enumerate(layer_stabilities):
        if stability is None:
            continue
        
        if device is None and isinstance(stability, torch.Tensor):
            device = stability.device
            total_loss = total_loss.to(device)
        
        # Divergence = 1 - stability (cosine distance)
        if isinstance(stability, torch.Tensor):
            divergence = 1.0 - stability
        else:
            divergence = torch.tensor(1.0 - stability, device=device)
        
        # Decaying requirement
        required_divergence = min_divergence * (divergence_decay ** i)
        deficit = F.relu(required_divergence - divergence)
        total_loss = total_loss + (deficit ** 2).mean()
        num_terms += 1
    
    # === Part 2: Information Gain ===
    layer_info_gains = aux_info.get('layer_info_gains', [])
    
    for i, info_gain in enumerate(layer_info_gains):
        if info_gain is None:
            continue
        
        if device is None and isinstance(info_gain, torch.Tensor):
            device = info_gain.device
            total_loss = total_loss.to(device)
        
        if isinstance(info_gain, torch.Tensor):
            abs_info_gain = info_gain.abs()
        else:
            abs_info_gain = torch.tensor(abs(info_gain), device=device)
        
        if maximize_info_gain:
            # Maximize info gain: loss = -|info_gain|
            # Weight earlier iterations more (they should do more work)
            weight = divergence_decay ** i  # 1.0, 0.5, 0.25, ...
            total_loss = total_loss - weight * abs_info_gain.mean()
        else:
            # Old behavior: penalize below threshold
            min_info_gain = 0.01
            required_info_gain = min_info_gain * (divergence_decay ** i)
            info_deficit = F.relu(required_info_gain - abs_info_gain)
            total_loss = total_loss + (info_deficit ** 2).mean()
        
        num_terms += 1
    
    if num_terms > 0:
        total_loss = total_loss / num_terms
    
    return total_loss


def compute_representation_entropy_change_loss(
    aux_info: Dict,
    min_entropy_change: float = 0.05,
) -> torch.Tensor:
    """
    Representation Entropy Change Loss - ensures iterations produce information change.
    
    Measures the entropy of the hidden state's activation distribution and
    penalizes when it doesn't change between iterations.
    
    Why entropy? If a layer iteration doesn't change the "shape" of activations,
    it's not doing meaningful computation. Entropy captures this distribution change.
    
    Entropy of hidden state h: H(h) = -sum(p * log(p)) where p = softmax(h, dim=-1)
    
    Args:
        aux_info: Contains 'all_layer_outputs' or similar
        min_entropy_change: Minimum expected entropy change between iterations
    
    Returns:
        Scalar loss penalizing stagnant entropy
    """
    # Get layer outputs across iterations
    all_outputs = aux_info.get('all_layer_outputs', [])
    
    if len(all_outputs) < 2:
        return torch.tensor(0.0)
    
    device = all_outputs[0].device
    total_loss = torch.tensor(0.0, device=device)
    num_pairs = 0
    
    for i in range(len(all_outputs) - 1):
        h_curr = all_outputs[i]      # [B, S, D]
        h_next = all_outputs[i + 1]  # [B, S, D]
        
        # Compute entropy of activation distribution
        # Treat hidden dim as "classes" and compute entropy per position
        p_curr = F.softmax(h_curr, dim=-1)  # [B, S, D]
        p_next = F.softmax(h_next, dim=-1)  # [B, S, D]
        
        entropy_curr = -(p_curr * torch.log(p_curr + 1e-8)).sum(dim=-1)  # [B, S]
        entropy_next = -(p_next * torch.log(p_next + 1e-8)).sum(dim=-1)  # [B, S]
        
        # Entropy change (absolute value - we care about change, not direction)
        entropy_change = (entropy_next - entropy_curr).abs().mean()  # Scalar
        
        # Penalize if change is too small
        deficit = F.relu(min_entropy_change - entropy_change)
        total_loss = total_loss + deficit ** 2
        num_pairs += 1
    
    if num_pairs > 0:
        total_loss = total_loss / num_pairs
    
    return total_loss


def compute_memory_threshold_regularization(
    model,
    aux_info: Dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Memory Threshold Regularization - provides gradients to read/write threshold networks.
    
    Unlike halt thresholds (which have clear targets), read/write thresholds
    control memory access patterns. We provide weak supervision:
    
    1. Thresholds should be around 0.5 initially (not too restrictive/permissive)
    2. As training progresses, let them move based on gate correlation
    
    The loss is:
    - Weak prior toward 0.5 (regularization)
    - Correlation with gate values (threshold should track gate difficulty)
    """
    read_thresholds = aux_info.get('read_thresholds', [])
    write_thresholds = aux_info.get('write_thresholds', [])
    
    total_loss = torch.tensor(0.0, device=device)
    
    # Weak prior: thresholds near 0.5 initially
    # This provides gradient signal while letting them adapt
    for thresh in read_thresholds:
        if thresh is not None and isinstance(thresh, torch.Tensor) and thresh.requires_grad:
            # Encourage threshold near 0.5 (weak regularization)
            total_loss = total_loss + 0.1 * ((thresh - 0.5) ** 2).mean()
    
    for thresh in write_thresholds:
        if thresh is not None and isinstance(thresh, torch.Tensor) and thresh.requires_grad:
            # Encourage threshold near 0.5 (weak regularization)  
            total_loss = total_loss + 0.1 * ((thresh - 0.5) ** 2).mean()
    
    return total_loss

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
    l1_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute task loss: cross-entropy + Smooth L1 distance.
    
    Two components:
    1. Per-cell cross-entropy (standard, for classification)
    2. Smooth L1 distance between softmax probs and one-hot targets
    
    Why Smooth L1 helps over L2:
    - L2 squares small errors → tiny gradients for small mistakes
    - L1/Smooth L1 gives linear gradient even for small errors
    - Smooth L1 is less sensitive to outliers than pure L1
    - Provides consistent gradient signal throughout training
    
    Smooth L1: 0.5*x^2 if |x|<1 else |x|-0.5
    
    Args:
        logits: [B, S, vocab_size]
        test_output: [B, S] with IGNORE_LABEL (-100) for padding
        pass_logits_list: List of logits from each pass (for deep supervision)
        vocab_size: Number of output classes
        l1_weight: Weight for Smooth L1 component (default 0.5)
    
    Returns:
        total_loss: Combined CE + Smooth L1
        metrics: Dict with ce_loss, l1_loss for logging
    """
    device = logits.device
    
    # NaN protection
    if torch.isnan(logits).any():
        return torch.tensor(10.0, device=device, requires_grad=True), {
            'task_ce': 10.0, 'task_l1': 1.0
        }
    
    logits_to_process = pass_logits_list if pass_logits_list else [logits]
    class_weights = compute_class_weights(test_output, vocab_size, IGNORE_LABEL)
    
    ce_loss = torch.tensor(0.0, device=device)
    l1_loss = torch.tensor(0.0, device=device)
    
    for p_logits in logits_to_process:
        # === 1. Standard Cross-Entropy ===
        logits_flat = p_logits.reshape(-1, vocab_size)
        targets_flat = test_output.reshape(-1)
        
        ce_loss = ce_loss + F.cross_entropy(
            logits_flat, targets_flat, 
            weight=class_weights,
            ignore_index=IGNORE_LABEL,
            label_smoothing=0.1
        )
        
        # === 2. Smooth L1: ||softmax(logits) - one_hot(target)|| ===
        valid_mask = (test_output != IGNORE_LABEL)  # [B, S]
        probs = F.softmax(p_logits, dim=-1)  # [B, S, V]
        
        # One-hot targets (clamp to handle -100)
        targets_clamped = test_output.clamp(min=0)  # [B, S]
        one_hot = F.one_hot(targets_clamped, num_classes=vocab_size).float()  # [B, S, V]
        
        # Smooth L1 per position (sum over vocab, then average over valid positions)
        # smooth_l1_loss expects same shape, so compute element-wise then reduce
        l1_per_elem = F.smooth_l1_loss(probs, one_hot, reduction='none')  # [B, S, V]
        l1_per_pos = l1_per_elem.sum(dim=-1)  # [B, S] - sum over vocab
        l1_per_pos = l1_per_pos * valid_mask.float()
        num_valid = valid_mask.sum().clamp(min=1)
        l1_loss = l1_loss + l1_per_pos.sum() / num_valid

    ce_loss = ce_loss / len(logits_to_process)
    l1_loss = l1_loss / len(logits_to_process)
    
    # Combine: CE for classification + Smooth L1 for consistent gradient
    total_loss = ce_loss + l1_weight * l1_loss
    
    metrics = {
        'task_ce': ce_loss.detach().item(),
        'task_l1': l1_loss.detach().item(),
    }
    
    return total_loss, metrics


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
    Aggregates all losses including confidence-based and ponder losses.
    
    Loss Components:
    1. Task Loss: Cross-entropy with deep supervision
    2. Confidence Calibration: Align confidence with actual correctness
    3. Halt Prediction: Train halt decision via Q-learning style BCE
    4. Confidence Monotonicity: Later passes should have higher confidence
    5. Adaptive Ponder Cost: Penalize redundant computation
    6. Early Exit Bonus: Reward efficient correct halting
    7. Gate Regularization: Polarization + sparsity
    8. Diversity Losses: Output entropy + prediction diversity
    
    Returns total_loss and metrics dict for logging.
    """
    features = config.features
    
    # 1. Task Loss with deep supervision + L2 distance
    if features.use_deep_supervision:
        pass_logits_list = aux_info.get('pass_logits', [logits])
    else:
        pass_logits_list = [logits]

    task_loss, task_metrics = compute_task_loss(
        logits, test_output, pass_logits_list,
        model.vocab_size,
        l1_weight=0.5,  # Smooth L1 provides consistent gradient signal
    )

    # 2. Q-Halt Loss (TRM-style) - train halt head to predict correctness
    # This is the main halting loss - trains q_halt to output >0 when correct
    q_halt_loss, q_halt_metrics = compute_q_halt_loss(aux_info, test_output)

    # 3. Early Exit Bonus - reward efficient correct halts (negative loss = reward)
    early_exit_bonus = compute_early_exit_bonus(
        aux_info, test_output, model.max_passes
    )

    # 4. Threshold Supervision Loss - provides gradients to halt threshold networks
    threshold_supervision_loss = compute_threshold_supervision_loss(
        model, aux_info, test_output, config, device
    )
    
    # 5. Memory Threshold Regularization - provides gradients to read/write thresholds
    memory_threshold_loss = compute_memory_threshold_regularization(
        model, aux_info, device
    )
    
    # 6. Layer Divergence Loss - encourages meaningful iteration (prevents immediate halt)
    # Now maximizes info gain directly instead of penalizing below threshold
    layer_divergence_loss = compute_layer_divergence_loss(
        aux_info,
        min_divergence=0.1,      # Require at least 10% divergence in first iteration
        divergence_decay=0.5,    # Halve requirement each subsequent iteration
        maximize_info_gain=True, # Maximize |entropy_change| between iterations
    )

    # 7. Model-Level Info Gain Loss - encourages productive passes
    # Maximizes information gain at model level (logits entropy change)
    model_info_gain_loss, model_ig_metrics = compute_model_info_gain_loss(
        aux_info,
        maximize=True,  # Maximize info gain (negative loss = reward)
        decay=0.7,      # Earlier passes matter more
    )

    # 8. Output Entropy Loss - prevents mode collapse
    output_entropy_loss = compute_output_entropy_loss(logits, test_output)
    
    # 11. Prediction Diversity Loss
    pred_diversity_loss = compute_prediction_diversity_loss(logits, test_output, model.vocab_size)

    # 12. Gate Regularization
    gate_polar_loss = torch.tensor(0.0, device=device)
    gate_sparsity_loss = torch.tensor(0.0, device=device)
    read_gates = aux_info.get('read_gates', [])
    write_gates = aux_info.get('write_gates', [])
    if read_gates or write_gates:
        gate_polar_loss = compute_gate_polarization_loss(read_gates, write_gates)
        gate_sparsity_loss = compute_read_gate_sparsity_loss(
            read_gates, write_gates,
            read_target_ratio=0.4,
            write_target_ratio=0.3
        )

    # 13. Feedback Gate Polarization (answer/iteration gates)
    feedback_polar_loss = compute_feedback_gate_polarization_loss(aux_info)

    # === LOSS WEIGHTING ===
    # Use config weights where available, sensible defaults otherwise
    
    # Halting losses
    lambda_q_halt = getattr(config, 'lambda_q_halt', 1.0)  # TRM-style q_halt loss (primary)
    lambda_threshold = getattr(config, 'lambda_threshold_supervision', 0.2)
    
    halt_loss = (
        lambda_q_halt * q_halt_loss +  # TRM-style: train q_halt to predict correctness
        lambda_threshold * threshold_supervision_loss +
        lambda_threshold * memory_threshold_loss +
        early_exit_bonus  # Already scaled internally, can be negative
    )
    
    # Gradient flow losses
    gradient_flow_loss = output_entropy_loss + 0.05 * pred_diversity_loss
    
    # Gate regularization
    lambda_gate_polar = getattr(config, 'lambda_gate_polar', 0.1)
    lambda_gate_sparsity = getattr(config, 'lambda_gate_sparsity', 0.1)
    lambda_feedback_polar = getattr(config, 'lambda_feedback_polar', 0.1)
    gate_loss = (
        lambda_gate_polar * gate_polar_loss + 
        lambda_gate_sparsity * gate_sparsity_loss +
        lambda_feedback_polar * feedback_polar_loss
    )
    
    # Layer divergence loss - encourage iterations to do meaningful work
    lambda_layer_divergence = getattr(config, 'lambda_layer_divergence', 0.5)
    divergence_loss = lambda_layer_divergence * layer_divergence_loss
    
    # Model-level info gain loss - encourage productive passes
    lambda_model_info_gain = getattr(config, 'lambda_model_info_gain', 0.3)
    model_ig_loss = lambda_model_info_gain * model_info_gain_loss

    # Total loss
    total_loss = (
        task_loss + 
        halt_loss +
        gradient_flow_loss + 
        gate_loss +
        divergence_loss +        # Encourage meaningful layer iterations
        model_ig_loss            # Encourage productive model passes
    )

    # NaN protection
    if torch.isnan(total_loss):
        print(f"WARNING: NaN detected in loss! Falling back to task_loss only.")
        print(f"  task={task_loss.item():.4f}, q_halt={q_halt_loss:.4f}")
        total_loss = task_loss

    # Metrics for logging
    metrics = {
        'loss_total': total_loss.detach().item(),
        'loss_task': task_loss.detach().item(),
        'loss_task_ce': task_metrics.get('task_ce', 0.0),
        'loss_task_l1': task_metrics.get('task_l1', 0.0),
        # Halting metrics
        'loss_q_halt': _to_scalar(q_halt_loss),
        'loss_threshold_supervision': _to_scalar(threshold_supervision_loss),
        'loss_memory_threshold': _to_scalar(memory_threshold_loss),
        'loss_early_exit_bonus': _to_scalar(early_exit_bonus),
        'q_halt_accuracy': q_halt_metrics.get('q_halt_accuracy', 0.0),
        # Regularization metrics
        'loss_output_entropy': _to_scalar(output_entropy_loss),
        'loss_pred_diversity': _to_scalar(pred_diversity_loss),
        'loss_gate_polar': _to_scalar(gate_polar_loss),
        'loss_gate_sparsity': _to_scalar(gate_sparsity_loss),
        'loss_feedback_polar': _to_scalar(feedback_polar_loss),
        'loss_layer_divergence': _to_scalar(layer_divergence_loss),
        # Model info gain
        'loss_model_info_gain': _to_scalar(model_info_gain_loss),
        'avg_model_info_gain': model_ig_metrics.get('avg_model_info_gain', 0.0),
    }

    return total_loss, metrics


def _to_scalar(x) -> float:
    """Convert tensor or scalar to float for logging."""
    if isinstance(x, torch.Tensor):
        return x.detach().item()
    return float(x)
