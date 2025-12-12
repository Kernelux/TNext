"""
Training Script for RecursiveRefinementModel
=============================================

This script trains the new unified recursive model that implements:
1. Two-level recursive refinement (model passes + layer iterations)
2. Layer-level thought injection (previous iteration feedback)
3. Model-level answer feedback (TRM-style hint injection)
4. Confidence-based halting at both levels

Usage:
    python train_recursive.py [preset]
    
Presets: fast, fast_full, runpod, full
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm
import os
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict

# Import modular components
from components import (
    FeatureFlags,
    TrainingConfig,
    FEATURE_PRESETS,
    ARCDataset,
    RecursiveRefinementModel,
    EMA,
    compute_total_loss,
    AdamAtan2,
)
from components.dataset import (
    VOCAB_SIZE, MAX_SEQ_LEN, PAD_TOKEN, EOS_TOKEN, COLOR_OFFSET,
    INPUT_MARKER, OUTPUT_MARKER, MAX_GRID_SIZE,
)


# ============================================================================
# Visualization Utilities (from dlsmn_arc.py)
# ============================================================================

ARC_PALETTE = np.array([
    [0, 0, 0],        # 0: Black
    [0, 116, 217],    # 1: Blue
    [255, 65, 54],    # 2: Red
    [46, 204, 64],    # 3: Green
    [255, 220, 0],    # 4: Yellow
    [170, 170, 170],  # 5: Gray
    [240, 18, 190],   # 6: Magenta
    [255, 133, 27],   # 7: Orange
    [127, 219, 255],  # 8: Cyan
    [135, 86, 53],    # 9: Brown
], dtype=np.uint8)

ARC_COLORS = {
    0: '\033[40m', 1: '\033[44m', 2: '\033[41m', 3: '\033[42m', 4: '\033[43m',
    5: '\033[100m', 6: '\033[45m', 7: '\033[48;5;208m', 8: '\033[46m', 9: '\033[48;5;94m',
}
RESET = '\033[0m'


def sequence_to_grid(seq: np.ndarray) -> np.ndarray:
    """Convert sequence to grid (30x30 aligned format)."""
    seq_2d = seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    content_mask = seq_2d >= COLOR_OFFSET
    
    if not content_mask.any():
        return np.zeros((1, 1), dtype=np.uint8)
    
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    seq_crop = seq_2d[rmin:rmax+1, cmin:cmax+1].copy()
    grid = np.zeros_like(seq_crop, dtype=np.uint8)
    for r in range(seq_crop.shape[0]):
        for c in range(seq_crop.shape[1]):
            token = seq_crop[r, c]
            grid[r, c] = max(0, token - COLOR_OFFSET) if token >= COLOR_OFFSET else 0
    
    return np.clip(grid, 0, 9)


def target_to_grid(target: np.ndarray) -> np.ndarray:
    """Convert target sequence to grid."""
    target_2d = target.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    valid_mask = (target_2d != -100) & (target_2d >= COLOR_OFFSET)
    
    if not valid_mask.any():
        return np.zeros((1, 1), dtype=np.uint8)
    
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    target_crop = target_2d[rmin:rmax+1, cmin:cmax+1].copy()
    grid = np.zeros_like(target_crop, dtype=np.uint8)
    for r in range(target_crop.shape[0]):
        for c in range(target_crop.shape[1]):
            token = target_crop[r, c]
            grid[r, c] = max(0, token - COLOR_OFFSET) if token >= COLOR_OFFSET else 0
    
    return np.clip(grid, 0, 9)


def prediction_to_grid(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Convert prediction to grid using target's valid region."""
    pred_2d = prediction.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    target_2d = target.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    
    valid_mask = (target_2d != -100) & (target_2d != PAD_TOKEN) & (target_2d != EOS_TOKEN)
    
    if not valid_mask.any():
        return np.zeros((1, 1), dtype=np.uint8)
    
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    pred_crop = pred_2d[rmin:rmax+1, cmin:cmax+1].copy()
    grid = np.zeros_like(pred_crop, dtype=np.uint8)
    for r in range(pred_crop.shape[0]):
        for c in range(pred_crop.shape[1]):
            token = pred_crop[r, c]
            grid[r, c] = max(0, token - COLOR_OFFSET) if token >= COLOR_OFFSET else 0
    
    return np.clip(grid, 0, 9)


def grid_to_ascii(grid, max_width: int = 15, max_height: int = 10) -> list:
    """Convert grid to colored ASCII art."""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    h, w = min(grid.shape[0], max_height), min(grid.shape[1], max_width)
    lines = []
    for row in range(h):
        line = ""
        for col in range(w):
            color = int(np.clip(grid[row, col], 0, 9))
            line += f"{ARC_COLORS[color]}  {RESET}"
        lines.append(line)
    lines.append(f"({grid.shape[0]}×{grid.shape[1]})")
    return lines


def grid_to_rgb(grid: np.ndarray, cell_size: int = 10) -> np.ndarray:
    """Convert grid to RGB image."""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    grid = np.clip(grid, 0, 9).astype(np.int32)
    rgb = ARC_PALETTE[grid]
    if cell_size > 1:
        rgb = np.repeat(np.repeat(rgb, cell_size, axis=0), cell_size, axis=1)
    return rgb


def visualize_sample(
    demo_inputs: torch.Tensor,
    demo_outputs: torch.Tensor,
    test_input: torch.Tensor,
    test_output: torch.Tensor,
    prediction: torch.Tensor,
    sample_idx: int = 0,
    step: int = 0,
    aux: Optional[Dict] = None,
) -> str:
    """Visualize a sample with metrics."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"[Step {step}] Sample #{sample_idx}")
    
    # Add recursive refinement stats if available
    if aux:
        passes = aux.get('passes_run', 1)
        max_passes = aux.get('max_passes', passes)
        layer_iters = aux.get('layer_iterations', [])
        avg_iters = sum(layer_iters) / len(layer_iters) if layer_iters else 1
        total_steps = aux.get('total_layer_steps', sum(layer_iters) if layer_iters else 0)
        
        lines.append(f"  Model Passes: {passes}/{max_passes} | Layer Iters: avg={avg_iters:.1f}, total={total_steps}")
        
        # Pass confidences
        pass_confs = aux.get('pass_confidences', [])
        if pass_confs:
            conf_str = ", ".join([f"{c.mean():.3f}" for c in pass_confs])
            lines.append(f"  Pass Confidences: [{conf_str}]")
        
        # Feedback gates
        # Feedback gates (now scalars)
        fb_parts = []
        if aux.get('answer_feedback_count', 0) > 0:
            avg_ans = aux['answer_feedback_sum'] / aux['answer_feedback_count']
            fb_parts.append(f"answer={avg_ans:.3f}")
        if aux.get('iteration_feedback_count', 0) > 0:
            avg_iter = aux['iteration_feedback_sum'] / aux['iteration_feedback_count']
            fb_parts.append(f"thought={avg_iter:.3f}")
        if fb_parts:
            lines.append(f"  Feedback Gates: {', '.join(fb_parts)}")
        
        # Memory gates (now scalars)
        gate_parts = []
        if aux.get('read_gate_count', 0) > 0:
            avg_read = aux['read_gate_sum'] / aux['read_gate_count']
            gate_parts.append(f"read={avg_read:.3f}")
        if aux.get('write_gate_count', 0) > 0:
            avg_write = aux['write_gate_sum'] / aux['write_gate_count']
            gate_parts.append(f"write={avg_write:.3f}")
        if gate_parts:
            lines.append(f"  Memory Gates: {', '.join(gate_parts)}")
    
    lines.append(f"{'='*70}")
    
    num_demos = demo_inputs.shape[0]
    
    # Demo pairs
    for d in range(min(num_demos, 2)):
        demo_in_grid = sequence_to_grid(demo_inputs[d].cpu().numpy())
        demo_out_grid = sequence_to_grid(demo_outputs[d].cpu().numpy())
        
        in_lines = grid_to_ascii(demo_in_grid, max_width=10, max_height=8)
        out_lines = grid_to_ascii(demo_out_grid, max_width=10, max_height=8)
        
        max_h = max(len(in_lines), len(out_lines))
        while len(in_lines) < max_h: in_lines.insert(-1, " " * 20)
        while len(out_lines) < max_h: out_lines.insert(-1, " " * 20)
        
        lines.append(f"\n  Demo {d+1}:")
        lines.append(f"  {'Input':^22} → {'Output':^22}")
        for i in range(max_h):
            lines.append(f"  {in_lines[i]:22}   {out_lines[i]:22}")
    
    # Test
    test_in_grid = sequence_to_grid(test_input.cpu().numpy())
    target_grid = target_to_grid(test_output.cpu().numpy())
    pred_grid = prediction_to_grid(prediction.cpu().numpy(), test_output.cpu().numpy())
    
    test_lines = grid_to_ascii(test_in_grid, max_width=10, max_height=8)
    tgt_lines = grid_to_ascii(target_grid, max_width=10, max_height=8)
    pred_lines = grid_to_ascii(pred_grid, max_width=10, max_height=8)
    
    max_h = max(len(test_lines), len(tgt_lines), len(pred_lines))
    while len(test_lines) < max_h: test_lines.insert(-1, " " * 20)
    while len(tgt_lines) < max_h: tgt_lines.insert(-1, " " * 20)
    while len(pred_lines) < max_h: pred_lines.insert(-1, " " * 20)
    
    lines.append(f"\n  Test:")
    lines.append(f"  {'Input':^22}   {'Target':^22}   {'Prediction':^22}")
    for i in range(max_h):
        lines.append(f"  {test_lines[i]:22}   {tgt_lines[i]:22}   {pred_lines[i]:22}")
    
    # Accuracy
    valid_mask = (test_output != -100)
    if valid_mask.any():
        correct = (prediction[valid_mask] == test_output[valid_mask]).float().mean().item()
        lines.append(f"\n  Accuracy: {correct*100:.1f}%")
    
    lines.append(f"{'='*70}\n")
    return '\n'.join(lines)


def create_sample_image(
    demo_inputs: torch.Tensor,
    demo_outputs: torch.Tensor,
    test_input: torch.Tensor,
    test_output: torch.Tensor,
    prediction: torch.Tensor,
    max_demos: int = 2,
    cell_size: int = 8,
) -> np.ndarray:
    """Create combined image for TensorBoard."""
    num_demos = min(demo_inputs.shape[0], max_demos)
    
    demo_in_grids = [sequence_to_grid(demo_inputs[d].cpu().numpy()) for d in range(num_demos)]
    demo_out_grids = [sequence_to_grid(demo_outputs[d].cpu().numpy()) for d in range(num_demos)]
    test_in_grid = sequence_to_grid(test_input.cpu().numpy())
    target_grid = target_to_grid(test_output.cpu().numpy())
    pred_grid = prediction_to_grid(prediction.cpu().numpy(), test_output.cpu().numpy())
    
    all_grids = demo_in_grids + demo_out_grids + [test_in_grid, target_grid, pred_grid]
    max_h = max(g.shape[0] for g in all_grids)
    max_w = max(g.shape[1] for g in all_grids)
    
    def pad_grid(g, h, w):
        padded = np.zeros((h, w), dtype=g.dtype)
        padded[:g.shape[0], :g.shape[1]] = g
        return padded
    
    demo_in_grids = [pad_grid(g, max_h, max_w) for g in demo_in_grids]
    demo_out_grids = [pad_grid(g, max_h, max_w) for g in demo_out_grids]
    test_in_grid = pad_grid(test_in_grid, max_h, max_w)
    target_grid = pad_grid(target_grid, max_h, max_w)
    pred_grid = pad_grid(pred_grid, max_h, max_w)
    
    demo_in_imgs = [grid_to_rgb(g, cell_size) for g in demo_in_grids]
    demo_out_imgs = [grid_to_rgb(g, cell_size) for g in demo_out_grids]
    test_in_img = grid_to_rgb(test_in_grid, cell_size)
    target_img = grid_to_rgb(target_grid, cell_size)
    pred_img = grid_to_rgb(pred_grid, cell_size)
    
    sep_h, sep_w = 2, max_w * cell_size
    h_sep = np.ones((sep_h, sep_w * 2 + sep_h, 3), dtype=np.uint8) * 255
    v_sep = np.ones((max_h * cell_size, sep_h, 3), dtype=np.uint8) * 255
    
    rows = []
    for i in range(num_demos):
        row = np.concatenate([demo_in_imgs[i], v_sep, demo_out_imgs[i]], axis=1)
        rows.append(row)
        rows.append(h_sep)
    
    v_sep_small = np.ones((max_h * cell_size, sep_h, 3), dtype=np.uint8) * 255
    test_row = np.concatenate([test_in_img, v_sep_small, target_img, v_sep_small, pred_img], axis=1)
    
    # Normalize widths
    max_row_width = max(r.shape[1] for r in rows + [test_row])
    final_rows = []
    for r in rows + [test_row]:
        if r.shape[1] < max_row_width:
            pad = np.zeros((r.shape[0], max_row_width - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        final_rows.append(r)
    
    return np.concatenate(final_rows, axis=0)


# ============================================================================
# Loss for RecursiveRefinementModel
# ============================================================================

def compute_recursive_loss(
    model: RecursiveRefinementModel,
    logits: torch.Tensor,
    targets: torch.Tensor,
    aux: Dict,
    config: TrainingConfig,
    step: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute loss for recursive refinement model.
    
    Now uses comprehensive loss from losses.py which includes:
    1. Task loss with deep supervision
    2. Confidence calibration (confidence ≈ correctness)
    3. Halt prediction (Q-learning style)
    4. Confidence monotonicity (later passes = higher confidence)
    5. Adaptive ponder cost (penalize redundant computation)
    6. Early exit bonus (reward efficient correct halts)
    7. Gate regularization
    8. Diversity losses
    """
    from components.losses import compute_total_loss
    
    # Use the comprehensive loss function from losses.py
    total_loss, metrics = compute_total_loss(
        model=model,
        logits=logits,
        test_output=targets,
        aux_info=aux,
        config=config,
        global_step=step,
        device=device,
    )
    
    # Add pass-level metrics for deep supervision tracking
    if config.features.use_deep_supervision:
        pass_logits = aux.get('pass_logits', [])
        IGNORE_LABEL = -100
        
        for i, pass_logit in enumerate(pass_logits):
            pass_loss = F.cross_entropy(
                pass_logit.reshape(-1, pass_logit.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_LABEL,
                label_smoothing=0.1,
            )
            metrics[f'loss_pass_{i}'] = pass_loss.detach().item()
    
    # Add halting info
    metrics['halted_early'] = aux.get('halted_early', False)
    metrics['passes_run'] = aux.get('passes_run', 1)
    
    return total_loss, metrics


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: RecursiveRefinementModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    global_step: int = 0,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 10,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, float, float, int]:
    """Training epoch for RecursiveRefinementModel."""
    model.train()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_tasks = 0
    total_tasks = 0
    
    IGNORE_LABEL = -100
    
    # Determine autocast dtype based on device
    if use_amp:
        if device.type == 'mps':
            amp_dtype = torch.float16  # MPS doesn't support bfloat16 well
        elif device.type == 'cuda' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    else:
        amp_dtype = None
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        
        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        amp_context = autocast(device.type, dtype=amp_dtype) if use_amp else nullcontext()
        with amp_context:
            logits, cache, aux = model(
                demo_inputs, demo_outputs, test_input,
                config=config,
                step=global_step,
                return_aux=True,
            )
            
            # Compute loss
            loss, metrics = compute_recursive_loss(
                model, logits, test_output,
                aux, config, global_step, device
            )
        
        # Backward with scaler if using AMP
        scaled_loss = loss / grad_accum_steps
        if use_amp and scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Optimizer step
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                # Check for NaN gradients
                has_nan = any(
                    torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    for p in model.parameters() if p.grad is not None
                )
                if has_nan:
                    print(f"[Step {global_step}] Skipping due to NaN/Inf gradients")
                    optimizer.zero_grad()
                else:
                    scaler.step(optimizer)
                scaler.update()
            else:
                # Check for NaN gradients
                has_nan = any(
                    torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    for p in model.parameters() if p.grad is not None
                )
                if has_nan:
                    print(f"[Step {global_step}] Skipping due to NaN/Inf gradients")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
        
        loss_val = loss.detach().item()
        total_loss += loss_val
        global_step += 1
        
        # Metrics
        preds = logits.detach().argmax(dim=-1)
        for i in range(preds.shape[0]):
            valid_mask = (test_output[i] != IGNORE_LABEL)
            if valid_mask.any():
                correct_cells += (preds[i][valid_mask] == test_output[i][valid_mask]).sum().item()
                total_cells += valid_mask.sum().item()
                if (preds[i][valid_mask] == test_output[i][valid_mask]).all():
                    correct_tasks += 1
            total_tasks += 1
        
        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        
        # Logging
        if writer is not None and batch_idx % log_interval == 0:
            writer.add_scalar('Loss/total', metrics['loss_total'], global_step)
            writer.add_scalar('Loss/task', metrics['loss_task'], global_step)
            if 'loss_efficiency' in metrics:
                writer.add_scalar('Loss/efficiency', metrics['loss_efficiency'], global_step)
            if 'loss_gate_polar' in metrics:
                writer.add_scalar('Loss/gate_polar', metrics['loss_gate_polar'], global_step)
            if 'loss_feedback_polar' in metrics:
                writer.add_scalar('Loss/feedback_polar', metrics['loss_feedback_polar'], global_step)
            if 'loss_gate_sparsity' in metrics:
                writer.add_scalar('Loss/gate_sparsity', metrics['loss_gate_sparsity'], global_step)
            
            # === Deep Supervision Metrics ===
            if 'loss_task_weighted' in metrics:
                writer.add_scalar('Loss/task_weighted', metrics['loss_task_weighted'], global_step)
            # Log per-pass losses
            for key, val in metrics.items():
                if key.startswith('loss_pass_'):
                    writer.add_scalar(f'DeepSupervision/{key}', val, global_step)
            
            # === Ponder Cost ===
            if 'ponder_cost' in metrics:
                writer.add_scalar('Compute/ponder_cost', metrics['ponder_cost'], global_step)
            if 'loss_ponder' in metrics:
                writer.add_scalar('Loss/ponder', metrics['loss_ponder'], global_step)
            
            # === Halting Stats ===
            if 'halted_early' in metrics:
                writer.add_scalar('Compute/halted_early', float(metrics['halted_early']), global_step)
            
            writer.add_scalar('Metrics/cell_accuracy', cell_acc, global_step)
            writer.add_scalar('Metrics/task_accuracy', task_acc, global_step)
            
            # === Two-Level Recursive Refinement Stats ===
            # Model-level (passes)
            passes = aux.get('passes_run', 1)
            max_passes = aux.get('max_passes', config.max_passes)
            writer.add_scalar('Compute/model_passes', passes, global_step)
            writer.add_scalar('Compute/pass_utilization', passes / max_passes, global_step)
            
            # Layer-level (iterations)
            layer_iters = aux.get('layer_iterations', [])
            avg_iters = sum(layer_iters) / len(layer_iters) if layer_iters else 1
            total_layer_steps = aux.get('total_layer_steps', sum(layer_iters))
            writer.add_scalar('Compute/avg_layer_iterations', avg_iters, global_step)
            writer.add_scalar('Compute/total_layer_steps', total_layer_steps, global_step)
            
            # Total compute utilization
            max_compute = max_passes * len(model.layers) * model.max_internal_iterations
            actual_compute = total_layer_steps
            writer.add_scalar('Compute/total_utilization', actual_compute / max(max_compute, 1), global_step)
            
            # === Feedback Gate Statistics ===
            # Model-level answer feedback (now scalar)
            if aux.get('answer_feedback_count', 0) > 0:
                avg_answer_fb = aux['answer_feedback_sum'] / aux['answer_feedback_count']
                writer.add_scalar('Feedback/answer_gate_mean', avg_answer_fb, global_step)
            
            # Layer-level thought injection (now scalar)
            if aux.get('iteration_feedback_count', 0) > 0:
                avg_iter_fb = aux['iteration_feedback_sum'] / aux['iteration_feedback_count']
                writer.add_scalar('Feedback/iteration_gate_mean', avg_iter_fb, global_step)
            
            # === Memory Gate Statistics (now scalar) ===
            if aux.get('read_gate_count', 0) > 0:
                avg_read = aux['read_gate_sum'] / aux['read_gate_count']
                writer.add_scalar('Gates/read_mean', avg_read, global_step)
            if aux.get('write_gate_count', 0) > 0:
                avg_write = aux['write_gate_sum'] / aux['write_gate_count']
                writer.add_scalar('Gates/write_mean', avg_write, global_step)
            
            # === Confidence Tracking ===
            pass_confs = aux.get('pass_confidences', [])
            if pass_confs:
                for i, conf in enumerate(pass_confs):
                    writer.add_scalar(f'Confidence/pass_{i}', conf.mean().item(), global_step)
                writer.add_scalar('Confidence/final', pass_confs[-1].mean().item(), global_step)
            
            # Temperature
            temp = aux.get('temperature', config.get_temperature(global_step))
            writer.add_scalar('Training/temperature', temp, global_step)
        
        # Visualization
        if writer is not None and global_step % 50 == 0:
            viz_str = visualize_sample(
                demo_inputs=demo_inputs[0],
                demo_outputs=demo_outputs[0],
                test_input=test_input[0],
                test_output=test_output[0],
                prediction=preds[0],
                sample_idx=0,
                step=global_step,
                aux=aux,
            )
            print(viz_str)
            
            try:
                sample_img = create_sample_image(
                    demo_inputs=demo_inputs[0],
                    demo_outputs=demo_outputs[0],
                    test_input=test_input[0],
                    test_output=test_output[0],
                    prediction=preds[0],
                )
                sample_img_chw = np.transpose(sample_img, (2, 0, 1)).astype(np.float32) / 255.0
                writer.add_image('Samples/prediction', sample_img_chw, global_step)
            except Exception as e:
                print(f"[WARN] Image logging failed: {e}")
        
        # Progress bar
        # P = Model passes (how many times we ran the full architecture)
        # I = Layer iterations (average internal iterations per layer per pass)
        passes = aux.get('passes_run', 1)
        layer_iters = aux.get('layer_iterations', [1])
        avg_iters = sum(layer_iters) / max(len(layer_iters), 1)
        pbar.set_postfix({
            'loss': f'{loss_val:.3f}',
            'cell': f'{cell_acc:.3f}',
            'task': f'{task_acc:.3f}',
            'Pass': f'{passes}',        # Model-level: full architecture passes
            'LayerIt': f'{avg_iters:.1f}',  # Layer-level: avg iterations within each layer
        })
        
        # Memory cleanup
        del loss, logits, cache
        for key in list(aux.keys()):
            if isinstance(aux[key], list):
                aux[key].clear()
        aux.clear()
        
        if batch_idx % 10 == 0:
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return total_loss / len(dataloader), cell_acc, task_acc, global_step


@torch.no_grad()
def evaluate(
    model: RecursiveRefinementModel,
    dataloader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    global_step: int = 0,
    writer: Optional[SummaryWriter] = None,
    visualize_samples: int = 2,
) -> Tuple[float, float]:
    """Evaluation loop."""
    model.eval()
    correct_tasks = 0
    total_tasks = 0
    correct_cells = 0
    total_cells = 0
    samples_visualized = 0
    
    IGNORE_LABEL = -100
    
    eval_config = TrainingConfig(
        tau_min=0.1,
        tau_start=0.1,
        max_passes=config.max_passes,
        max_recurrent_steps=config.max_recurrent_steps,
        features=config.features,
    )
    
    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for batch in pbar:
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        
        logits, _, aux = model(
            demo_inputs, demo_outputs, test_input,
            config=eval_config,
            step=100000,
            return_aux=True,
        )
        
        preds = logits.argmax(dim=-1)
        
        for i in range(preds.shape[0]):
            valid_mask = (test_output[i] != IGNORE_LABEL)
            if valid_mask.any():
                correct_cells += (preds[i][valid_mask] == test_output[i][valid_mask]).sum().item()
                total_cells += valid_mask.sum().item()
                
                task_correct = (preds[i][valid_mask] == test_output[i][valid_mask]).all()
                if task_correct:
                    correct_tasks += 1
                
                if samples_visualized < visualize_samples:
                    marker = "✓" if task_correct else "✗"
                    print(f"\n[Eval Sample {samples_visualized + 1}] {marker}")
                    viz_str = visualize_sample(
                        demo_inputs=demo_inputs[i],
                        demo_outputs=demo_outputs[i],
                        test_input=test_input[i],
                        test_output=test_output[i],
                        prediction=preds[i],
                        sample_idx=samples_visualized,
                        step=global_step,
                        aux=aux,
                    )
                    print(viz_str)
                    samples_visualized += 1
            
            total_tasks += 1
        
        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        pbar.set_postfix({'cell': f'{cell_acc:.3f}', 'task': f'{task_acc:.3f}'})
    
    return cell_acc, task_acc


def print_model_summary(model: torch.nn.Module):
    """Print model parameter summary."""
    print("\n" + "="*60)
    print(f"Model: {model.__class__.__name__}")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    breakdown = {"Embeddings": 0, "Layers": 0, "Output": 0, "Other": 0}
    
    for name, param in model.named_parameters():
        num = param.numel()
        if "embed" in name:
            breakdown["Embeddings"] += num
        elif "layer" in name:
            breakdown["Layers"] += num
        elif "output" in name or "proj" in name:
            breakdown["Output"] += num
        else:
            breakdown["Other"] += num
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("-" * 60)
    for cat, count in breakdown.items():
        pct = (count / total_params) * 100 if total_params else 0
        print(f"  {cat:<20}: {count:,} ({pct:.1f}%)")
    print("="*60 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    import sys
    
    # Check data
    data_dir = Path("./ARC-AGI-2/data")
    if not data_dir.exists():
        print("Downloading ARC-AGI-2 dataset...")
        os.system("git clone https://github.com/arcprize/ARC-AGI-2.git")
    
    # Device
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Preset
    preset_name = sys.argv[1] if len(sys.argv) > 1 else "fast"
    features = FEATURE_PRESETS.get(preset_name, FEATURE_PRESETS["fast"])
    print(f"\nUsing preset: '{preset_name}'")
    print(f"Features: {features.describe()}")
    
    # Model configuration per preset
    if preset_name == "fast_full":
        d_model, d_cache = 64, 48
        num_layers, num_slots, num_heads = 4, 32, 2  # Reduced layers and slots
        batch_size = 8  # Single sample - 30x30 grids are huge!
        max_passes, max_internal_iterations = 3, 3
        grad_accum_steps = 1  # Compensate for tiny batch
    elif preset_name == "runpod":
        d_model, d_cache = 128, 64
        num_layers, num_slots, num_heads = 4, 16, 4
        batch_size = 16
        max_passes, max_internal_iterations = 4, 3
    elif preset_name == "full":
        d_model, d_cache = 128, 64
        num_layers, num_slots, num_heads = 4, 16, 4
        batch_size = 2
        max_passes, max_internal_iterations = 6, 4
    else:  # fast
        d_model, d_cache = 64, 48
        num_layers, num_slots, num_heads = 2, 8, 2
        batch_size = 8
        max_passes = 2 if features.use_multi_pass else 1
        max_internal_iterations = 2 if features.use_layer_act else 1
    
    max_grid_size = 30 if preset_name in ["fast_full", "full"] else (10 if preset_name == "runpod" else 15)
    
    # Datasets
    train_dataset = ARCDataset(str(data_dir), split="training", max_grid_size=max_grid_size, augment=True)
    eval_dataset = ARCDataset(str(data_dir), split="evaluation", max_grid_size=max_grid_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Config
    config = TrainingConfig(
        tau_start=1.0,
        tau_min=0.1,
        max_passes=max_passes,
        max_recurrent_steps=max_internal_iterations,
        features=features,
        lambda_step_efficiency=0.02,
        lambda_diversity=0.01,
    )
    
    # Model
    model = RecursiveRefinementModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        d_cache=d_cache,
        num_layers=num_layers,
        num_slots=num_slots,
        num_heads=num_heads,
        max_seq_len=MAX_SEQ_LEN,
        max_internal_iterations=max_internal_iterations,
        max_passes=max_passes,
        dropout=0.0,
        confidence_threshold=0.8,
        use_fixed_gate_threshold=config.use_fixed_gate_threshold,
        fixed_gate_threshold=config.gate_threshold,
        use_hippo_init=features.use_hippo_init,
    ).to(device)
    
    print_model_summary(model)
    
    # Optimizer
    lr = 3e-4 if preset_name == "runpod" else 1e-4
    optimizer = AdamAtan2(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        #weight_decay=0.1,
    )
    print(f"Using AdamAtan2 optimizer (lr={lr})")
    
    # Gradient accumulation - use preset value if set, otherwise default
    # if preset_name == "fast_full":
    #     grad_accum_steps = 8  # Compensate for batch_size=1
    # elif preset_name in ["full", "runpod"]:
    #     grad_accum_steps = 2
    # else:
    grad_accum_steps = 1
    print(f"Gradient accumulation steps: {grad_accum_steps}, effective batch: {batch_size * grad_accum_steps}")
    
    # Mixed precision training (reduces memory by ~50%)
    # Note: MPS has issues with float16, so disable AMP for MPS
    use_amp = (device.type == 'cuda')  # Only enable for CUDA
    if use_amp:
        scaler = GradScaler()
        print("Using mixed precision (AMP) with GradScaler")
    else:
        scaler = None
        print(f"Using full precision (fp32) - AMP disabled for {device.type}")
    
    # Logging
    log_dir = Path("logs") / f"recursive_{preset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Training loop
    best_task_acc = 0
    global_step = 0
    
    for epoch in range(50):
        train_loss, train_cell_acc, train_task_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, config, global_step,
            writer=writer, log_interval=10, grad_accum_steps=grad_accum_steps,
            use_amp=use_amp, scaler=scaler,
        )
        
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/train_cell_acc', train_cell_acc, epoch)
        writer.add_scalar('Epoch/train_task_acc', train_task_acc, epoch)
        
        if (epoch + 1) % 10 == 0:
            eval_cell_acc, eval_task_acc = evaluate(
                model, eval_loader, device, config,
                global_step=global_step, writer=writer, visualize_samples=2,
            )
            
            if eval_task_acc > best_task_acc:
                best_task_acc = eval_task_acc
                # Save best model
                torch.save(model.state_dict(), log_dir / "best_model.pt")
            
            writer.add_scalar('Epoch/eval_cell_acc', eval_cell_acc, epoch)
            writer.add_scalar('Epoch/eval_task_acc', eval_task_acc, epoch)
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Eval Task: {eval_task_acc:.3f} | Best: {best_task_acc:.3f}")
        else:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train Task: {train_task_acc:.3f}")
    
    writer.close()
    print(f"\nTraining complete. Best task accuracy: {best_task_acc:.3f}")


if __name__ == "__main__":
    main()
