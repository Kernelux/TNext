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


def compute_halt_prediction_loss(
    aux_info: Dict,
    test_output: torch.Tensor,
) -> torch.Tensor:
    """
    Halt Prediction Loss (Q-learning style) - train halt decision to predict correctness.
    
    TRM uses a Q-head that predicts "should I halt?" with BCE against actual correctness.
    We use our confidence as the halt signal, so we train confidence to predict correctness.
    
    This is similar to confidence calibration but focuses on the DECISION:
    - If we halted (or would halt) at this pass, was it correct?
    - Train halt threshold to say "halt" when answer is correct
    
    The key difference from calibration:
    - Calibration: confidence ≈ correctness (soft matching)
    - Halt prediction: halt_decision ≈ full_sequence_correct (binary task)
    """
    device = test_output.device
    
    pass_confidences = aux_info.get('pass_confidences', [])
    pass_logits = aux_info.get('pass_logits', [])
    
    if not pass_confidences or not pass_logits:
        return torch.tensor(0.0, device=device)
    
    total_loss = torch.tensor(0.0, device=device)
    valid_mask = (test_output != IGNORE_LABEL)
    
    for i, (conf, logits) in enumerate(zip(pass_confidences, pass_logits)):
        # Compute sequence-level correctness at this pass
        predictions = logits.argmax(dim=-1)  # [B, S]
        correct_positions = (predictions == test_output) | ~valid_mask  # [B, S]
        seq_correct = correct_positions.all(dim=-1).float()  # [B] - 1 if all correct
        
        # Confidence should predict sequence correctness
        # Use mean confidence as "would halt" decision
        conf_mean = conf.mean(dim=-1) if conf.dim() > 1 else conf  # [B]
        
        # BCE loss: confidence should match correctness
        loss = F.binary_cross_entropy(
            conf_mean.clamp(0.01, 0.99),
            seq_correct.detach(),
            reduction='mean'
        )
        
        # Weight later passes more (more important to get halt right at the end)
        weight = (i + 1) / len(pass_confidences)
        total_loss = total_loss + weight * loss
    
    return total_loss / len(pass_confidences)


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
    
    Now handles TWO-PHASE architecture:
    - Reflection phase: Q-head learns when reflection is "complete" (cache is ready)
    - Answer phase: Q-head learns when answer is "correct" (prediction is good)
    
    For reflection phase, we use a heuristic: reflection is "complete" when
    cache state is sufficiently distinct from initial state (learned implicitly).
    We don't have ground truth for "when is reflection done", so we train it
    with a soft target based on pass number (later passes should halt).
    
    For answer phase, we use actual correctness against test_output.
    
    Now works with sequences: test_output is [B, S] with IGNORE_LABEL for padding.
    """
    model_q_loss = torch.tensor(0.0, device=device)
    features = config.features
    
    # no_act_continue: Paper recommends skipping Q_continue loss
    use_q_continue = not features.no_act_continue
    
    # === ANSWER PHASE Q-HEAD LOSS ===
    # Get answer Q-head outputs (these are LOGITS)
    q_halt_list = aux_info.get('q_halt', [])
    q_continue_list = aux_info.get('q_continue', [])
    
    if q_halt_list:
        # Get pass logits for computing per-pass correctness
        pass_logits = aux_info.get('pass_logits', [])
        final_logits = aux_info.get('final_logits')
        
        # If we have per-pass logits (deep supervision), compute per-pass loss
        if pass_logits:
            total_loss = torch.tensor(0.0, device=device)
            for i, (q_h_logit, q_c_logit, p_logits) in enumerate(zip(q_halt_list, q_continue_list, pass_logits)):
                # Per-pass correctness (only in valid positions, not padding)
                preds = p_logits.detach().argmax(dim=-1)  # [B, S]
                valid_mask = (test_output != IGNORE_LABEL)  # [B, S]
                correct_at_valid = (preds == test_output) | ~valid_mask
                is_correct = correct_at_valid.all(dim=-1).float()  # [B]
                
                # BCE with logits (more numerically stable)
                loss_halt = F.binary_cross_entropy_with_logits(q_h_logit, is_correct.detach())
                total_loss = total_loss + loss_halt
                
                if use_q_continue:
                    is_wrong = 1.0 - is_correct
                    loss_cont = F.binary_cross_entropy_with_logits(q_c_logit, is_wrong.detach())
                    total_loss = total_loss + loss_cont
            
            model_q_loss = model_q_loss + total_loss / len(pass_logits)
        
        # Otherwise, use final logits only
        elif final_logits is not None and q_halt_list:
            q_halt_final = q_halt_list[-1]
            
            preds = final_logits.detach().argmax(dim=-1)  # [B, S]
            valid_mask = (test_output != IGNORE_LABEL)  # [B, S]
            correct_at_valid = (preds == test_output) | ~valid_mask
            is_correct = correct_at_valid.all(dim=-1).float()  # [B]
            
            loss_halt = F.binary_cross_entropy_with_logits(q_halt_final, is_correct.detach())
            model_q_loss = model_q_loss + loss_halt
            
            if use_q_continue and q_continue_list:
                q_continue_final = q_continue_list[-1]
                is_wrong = 1.0 - is_correct
                loss_cont = F.binary_cross_entropy_with_logits(q_continue_final, is_wrong.detach())
                model_q_loss = model_q_loss + loss_cont
    
    # === REFLECTION PHASE Q-HEAD LOSS ===
    # For reflection, we use a soft target: later passes should be more likely to halt
    # This is a heuristic since we don't know the "true" completion point
    reflection_q_halt = aux_info.get('reflection_q_halt', [])
    reflection_q_continue = aux_info.get('reflection_q_continue', [])
    
    if reflection_q_halt:
        num_reflection_passes = len(reflection_q_halt)
        reflection_loss = torch.tensor(0.0, device=device)
        
        for i, (q_h_logit, q_c_logit) in enumerate(zip(reflection_q_halt, reflection_q_continue)):
            # Soft target: probability of halting increases with pass number
            # Pass 1: target 0.2, Pass 2: 0.4, Pass 3: 0.6, etc.
            # This encourages the model to learn when reflection is "enough"
            halt_target = min(0.8, (i + 1) / num_reflection_passes)
            batch_size = q_h_logit.shape[0]
            halt_targets = torch.full((batch_size,), halt_target, device=device)
            
            loss_halt = F.binary_cross_entropy_with_logits(q_h_logit, halt_targets)
            reflection_loss = reflection_loss + loss_halt
            
            if use_q_continue:
                continue_targets = 1.0 - halt_targets
                loss_cont = F.binary_cross_entropy_with_logits(q_c_logit, continue_targets)
                reflection_loss = reflection_loss + loss_cont
        
        # Weight reflection Q-loss less than answer (answer correctness is more important)
        model_q_loss = model_q_loss + 0.5 * reflection_loss / num_reflection_passes
    
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
    
    # 1. Task Loss with deep supervision
    if features.use_deep_supervision:
        pass_logits_list = aux_info.get('pass_logits', [logits])
    else:
        pass_logits_list = [logits]

    task_loss = compute_task_loss(
        logits, test_output, pass_logits_list,
        model.vocab_size
    )

    # 2. Confidence Calibration Loss - confidence should match correctness
    confidence_calibration_loss = compute_confidence_calibration_loss(
        aux_info, test_output, model.vocab_size
    )

    # 3. Halt Prediction Loss (Q-learning style) - train halt to predict correctness
    halt_prediction_loss = compute_halt_prediction_loss(aux_info, test_output)

    # 4. Confidence Monotonicity Loss - confidence should increase across passes
    confidence_monotonicity_loss = compute_confidence_monotonicity_loss(aux_info)

    # 5. Adaptive Ponder Cost - penalize wasteful computation
    adaptive_ponder_loss = compute_adaptive_ponder_cost(
        aux_info, test_output, model.max_passes
    )

    # 6. Early Exit Bonus - reward efficient correct halts (negative loss = reward)
    early_exit_bonus = compute_early_exit_bonus(
        aux_info, test_output, model.max_passes
    )

    # 7. Legacy Q-Head Loss (for backward compatibility if q_halt/q_continue present)
    model_q_loss = compute_q_head_loss(aux_info, test_output, config, device)

    # 8. Step Efficiency Loss (layer-level)
    step_efficiency_loss = compute_step_efficiency_loss(
        logits, aux_info, test_output, config, model.num_layers, device
    )

    # 9. Diversity Loss (slot usage)
    diversity_loss = torch.tensor(0.0, device=device)
    if features.use_diversity_loss and 'slot_counts' in aux_info and aux_info['slot_counts']:
        diversity_loss = compute_diversity_loss(aux_info['slot_counts'])

    # 10. Output Entropy Loss - prevents mode collapse
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

    # === LOSS WEIGHTING ===
    # Use config weights where available, sensible defaults otherwise
    
    # Confidence-based losses (new)
    lambda_conf_calib = getattr(config, 'lambda_confidence_calibration', 0.5)
    lambda_halt_pred = getattr(config, 'lambda_halt_prediction', 0.3)
    lambda_conf_mono = getattr(config, 'lambda_confidence_monotonicity', 0.1)
    lambda_ponder = getattr(features, 'lambda_ponder', 0.01)
    
    confidence_loss = (
        lambda_conf_calib * confidence_calibration_loss +
        lambda_halt_pred * halt_prediction_loss +
        lambda_conf_mono * confidence_monotonicity_loss +
        lambda_ponder * adaptive_ponder_loss +
        early_exit_bonus  # Already scaled internally, can be negative
    )
    
    # Compute efficiency losses
    compute_loss = (
        config.lambda_q_head * model_q_loss +
        config.lambda_step_efficiency * step_efficiency_loss
    )
    
    # Gradient flow losses
    gradient_flow_loss = output_entropy_loss + 0.05 * pred_diversity_loss
    
    # Gate regularization
    gate_loss = 0.01 * gate_polar_loss + 0.1 * gate_sparsity_loss

    # Total loss
    total_loss = (
        task_loss + 
        confidence_loss +
        compute_loss + 
        config.lambda_diversity * diversity_loss + 
        gradient_flow_loss + 
        gate_loss
    )

    # NaN protection
    if torch.isnan(total_loss):
        print(f"WARNING: NaN detected in loss! Falling back to task_loss only.")
        print(f"  task={task_loss.item():.4f}, conf_calib={confidence_calibration_loss:.4f}, "
              f"halt_pred={halt_prediction_loss:.4f}, ponder={adaptive_ponder_loss:.4f}")
        total_loss = task_loss

    # Metrics for logging
    metrics = {
        'loss_total': total_loss.detach().item(),
        'loss_task': task_loss.detach().item(),
        # Confidence-based metrics
        'loss_confidence_calibration': _to_scalar(confidence_calibration_loss),
        'loss_halt_prediction': _to_scalar(halt_prediction_loss),
        'loss_confidence_monotonicity': _to_scalar(confidence_monotonicity_loss),
        'loss_ponder_adaptive': _to_scalar(adaptive_ponder_loss),
        'loss_early_exit_bonus': _to_scalar(early_exit_bonus),
        # Compute efficiency metrics
        'loss_compute': _to_scalar(compute_loss),
        'loss_q_head': _to_scalar(model_q_loss),
        'loss_step_efficiency': _to_scalar(step_efficiency_loss),
        # Regularization metrics
        'loss_diversity': _to_scalar(diversity_loss),
        'loss_output_entropy': _to_scalar(output_entropy_loss),
        'loss_pred_diversity': _to_scalar(pred_diversity_loss),
        'loss_gate_polar': _to_scalar(gate_polar_loss),
        'loss_gate_sparsity': _to_scalar(gate_sparsity_loss),
    }

    return total_loss, metrics


def _to_scalar(x) -> float:
    """Convert tensor or scalar to float for logging."""
    if isinstance(x, torch.Tensor):
        return x.detach().item()
    return float(x)
