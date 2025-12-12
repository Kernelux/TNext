"""
Training Script for Mini-ARC
============================

Mini-ARC version of train_recursive.py for fast iteration.

Key differences from full ARC:
- 5x5 grids instead of 30x30 → 36x faster per sample
- Smaller model recommended (less data, simpler tasks)
- Faster epochs, quicker feedback loop

Usage:
    python train_miniarc.py [preset]
    
Presets:
    debug       - Minimal config for debugging
    fast        - Quick training for iteration
    fast_full   - Same as train_recursive.py fast_full
    full        - Full model for best performance
    trm         - TRM-style (H_cycles=4, L_cycles=6, pure TRM)
    trm_minimal - TRM-style with fewer cycles (faster)
    trm_deep    - TRM with more layers
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
    RecursiveRefinementModel,
    EMA,
    compute_total_loss,
    AdamAtan2,
)
from components.miniarc_dataset import (
    MiniARCDataset,
    VOCAB_SIZE, MAX_SEQ_LEN, PAD_TOKEN, COLOR_OFFSET, MAX_GRID_SIZE,
    sequence_to_grid, target_to_grid, download_miniarc,
)


# ============================================================================
# Visualization Utilities
# ============================================================================

ARC_COLORS = {
    0: '\033[40m', 1: '\033[44m', 2: '\033[41m', 3: '\033[42m', 4: '\033[43m',
    5: '\033[100m', 6: '\033[45m', 7: '\033[48;5;208m', 8: '\033[46m', 9: '\033[48;5;94m',
}
RESET = '\033[0m'

# RGB palette for TensorBoard images
ARC_PALETTE = np.array([
    [0, 0, 0],       # 0: Black
    [0, 116, 217],   # 1: Blue
    [255, 65, 54],   # 2: Red
    [46, 204, 64],   # 3: Green
    [255, 220, 0],   # 4: Yellow
    [170, 170, 170], # 5: Grey
    [240, 18, 190],  # 6: Fuschia
    [255, 133, 27],  # 7: Orange
    [127, 219, 255], # 8: Teal
    [135, 12, 37],   # 9: Maroon
], dtype=np.uint8)


def grid_to_ascii(grid, max_width: int = 10, max_height: int = 10) -> list:
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


def prediction_to_grid(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Convert prediction to grid using target's valid region."""
    pred_2d = prediction.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    target_2d = target.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    
    grid = np.zeros_like(pred_2d, dtype=np.uint8)
    valid_mask = (target_2d != -100) & (pred_2d >= COLOR_OFFSET)
    grid[valid_mask] = np.clip(pred_2d[valid_mask] - COLOR_OFFSET, 0, 9)
    
    return grid


def grid_to_rgb(grid: np.ndarray, cell_size: int = 10) -> np.ndarray:
    """Convert grid to RGB image for TensorBoard."""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    grid = np.clip(grid, 0, 9).astype(np.int32)
    rgb = ARC_PALETTE[grid]
    if cell_size > 1:
        rgb = np.repeat(np.repeat(rgb, cell_size, axis=0), cell_size, axis=1)
    return rgb


def create_sample_image(
    demo_inputs: torch.Tensor,
    demo_outputs: torch.Tensor,
    test_input: torch.Tensor,
    test_output: torch.Tensor,
    prediction: torch.Tensor,
    max_demos: int = 2,
    cell_size: int = 16,  # Larger cells for Mini-ARC's 5x5 grids
) -> np.ndarray:
    """Create combined image for TensorBoard visualization."""
    num_demos = min(demo_inputs.shape[0], max_demos)
    
    # Convert to grids
    demo_in_grids = [sequence_to_grid(demo_inputs[d].cpu().numpy()) for d in range(num_demos)]
    demo_out_grids = [sequence_to_grid(demo_outputs[d].cpu().numpy()) for d in range(num_demos)]
    test_in_grid = sequence_to_grid(test_input.cpu().numpy())
    target_grid = target_to_grid(test_output.cpu().numpy())
    pred_grid = prediction_to_grid(prediction.cpu().numpy(), test_output.cpu().numpy())
    
    # Pad all grids to same size
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
    
    # Convert to RGB
    demo_in_imgs = [grid_to_rgb(g, cell_size) for g in demo_in_grids]
    demo_out_imgs = [grid_to_rgb(g, cell_size) for g in demo_out_grids]
    test_in_img = grid_to_rgb(test_in_grid, cell_size)
    target_img = grid_to_rgb(target_grid, cell_size)
    pred_img = grid_to_rgb(pred_grid, cell_size)
    
    # Create separators
    sep_h, sep_w = 4, max_w * cell_size
    h_sep = np.ones((sep_h, sep_w * 2 + sep_h, 3), dtype=np.uint8) * 128
    v_sep = np.ones((max_h * cell_size, sep_h, 3), dtype=np.uint8) * 128
    
    # Build rows: Demo pairs
    rows = []
    for i in range(num_demos):
        row = np.concatenate([demo_in_imgs[i], v_sep, demo_out_imgs[i]], axis=1)
        rows.append(row)
        rows.append(h_sep[:, :row.shape[1], :])
    
    # Test row: Input | Target | Prediction
    v_sep_small = np.ones((max_h * cell_size, sep_h, 3), dtype=np.uint8) * 128
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
    lines.append(f"\n{'='*60}")
    lines.append(f"[Step {step}] Mini-ARC Sample #{sample_idx}")
    
    # Recursive refinement stats
    if aux:
        passes = aux.get('passes_run', 1)
        max_passes = aux.get('max_passes', passes)
        layer_iters = aux.get('layer_iterations', [])
        avg_iters = sum(layer_iters) / len(layer_iters) if layer_iters else 1
        lines.append(f"  Passes: {passes}/{max_passes} | Avg Layer Iters: {avg_iters:.1f}")
        
        pass_confs = aux.get('pass_confidences', [])
        if pass_confs:
            conf_str = ", ".join([f"{c.mean():.3f}" for c in pass_confs])
            lines.append(f"  Confidences: [{conf_str}]")
    
    lines.append(f"{'='*60}")
    
    # Demo pairs (show max 2)
    num_demos = demo_inputs.shape[0]
    for d in range(min(num_demos, 2)):
        demo_in_grid = sequence_to_grid(demo_inputs[d].cpu().numpy())
        demo_out_grid = sequence_to_grid(demo_outputs[d].cpu().numpy())
        
        in_lines = grid_to_ascii(demo_in_grid)
        out_lines = grid_to_ascii(demo_out_grid)
        
        max_h = max(len(in_lines), len(out_lines))
        while len(in_lines) < max_h: in_lines.insert(-1, " " * 12)
        while len(out_lines) < max_h: out_lines.insert(-1, " " * 12)
        
        lines.append(f"\n  Demo {d+1}:")
        lines.append(f"  {'Input':^14} → {'Output':^14}")
        for i in range(max_h):
            lines.append(f"  {in_lines[i]:14}   {out_lines[i]:14}")
    
    # Test
    test_in_grid = sequence_to_grid(test_input.cpu().numpy())
    target_grid = target_to_grid(test_output.cpu().numpy())
    pred_grid = prediction_to_grid(prediction.cpu().numpy(), test_output.cpu().numpy())
    
    test_lines = grid_to_ascii(test_in_grid)
    tgt_lines = grid_to_ascii(target_grid)
    pred_lines = grid_to_ascii(pred_grid)
    
    max_h = max(len(test_lines), len(tgt_lines), len(pred_lines))
    while len(test_lines) < max_h: test_lines.insert(-1, " " * 12)
    while len(tgt_lines) < max_h: tgt_lines.insert(-1, " " * 12)
    while len(pred_lines) < max_h: pred_lines.insert(-1, " " * 12)
    
    lines.append(f"\n  Test:")
    lines.append(f"  {'Input':^14}   {'Target':^14}   {'Prediction':^14}")
    for i in range(max_h):
        lines.append(f"  {test_lines[i]:14}   {tgt_lines[i]:14}   {pred_lines[i]:14}")
    
    # Accuracy
    valid_mask = (test_output != -100)
    if valid_mask.any():
        correct = (prediction[valid_mask] == test_output[valid_mask]).float().mean().item()
        marker = "✓" if correct == 1.0 else "✗"
        lines.append(f"\n  Accuracy: {correct*100:.1f}% {marker}")
    
    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)


# ============================================================================
# Loss Computation
# ============================================================================

def compute_miniarc_loss(
    model: RecursiveRefinementModel,
    logits: torch.Tensor,
    targets: torch.Tensor,
    aux: Dict,
    config: TrainingConfig,
    step: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """Compute loss for Mini-ARC (same as full ARC)."""
    total_loss, metrics = compute_total_loss(
        model=model,
        logits=logits,
        test_output=targets,
        aux_info=aux,
        config=config,
        global_step=step,
        device=device,
    )
    
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
) -> Tuple[float, float, float, int]:
    """Training epoch."""
    model.train()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_tasks = 0
    total_tasks = 0
    
    IGNORE_LABEL = -100
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, cache, aux = model(
            demo_inputs, demo_outputs, test_input,
            config=config,
            step=global_step,
            return_aux=True,
        )
        
        # Compute loss
        loss, metrics = compute_miniarc_loss(
            model, logits, test_output,
            aux, config, global_step, device
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Check for NaN gradients
        has_nan = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in model.parameters()
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
        
        # TensorBoard logging
        if writer is not None and batch_idx % log_interval == 0:
            # === Loss Metrics ===
            writer.add_scalar('Loss/total', metrics['loss_total'], global_step)
            writer.add_scalar('Loss/task', metrics['loss_task'], global_step)
            writer.add_scalar('Metrics/cell_accuracy', cell_acc, global_step)
            writer.add_scalar('Metrics/task_accuracy', task_acc, global_step)
            
            # === Compute Statistics ===
            passes = aux.get('passes_run', 1)
            max_passes = config.max_passes
            writer.add_scalar('Compute/model_passes', passes, global_step)
            writer.add_scalar('Compute/pass_utilization', passes / max_passes, global_step)
            
            layer_iters = aux.get('layer_iterations', [])
            if layer_iters:
                avg_iters = sum(layer_iters) / len(layer_iters)
                total_layer_steps = sum(layer_iters)
                writer.add_scalar('Compute/avg_layer_iterations', avg_iters, global_step)
                writer.add_scalar('Compute/total_layer_steps', total_layer_steps, global_step)
                
                # Total compute utilization
                max_compute = max_passes * len(model.layers) * model.max_internal_iterations
                writer.add_scalar('Compute/total_utilization', total_layer_steps / max(max_compute, 1), global_step)
            
            # === Feedback Gate Statistics ===
            if aux.get('answer_feedback_count', 0) > 0:
                avg_answer_fb = aux['answer_feedback_sum'] / aux['answer_feedback_count']
                writer.add_scalar('Feedback/answer_gate_mean', avg_answer_fb, global_step)
            
            if aux.get('iteration_feedback_count', 0) > 0:
                avg_iter_fb = aux['iteration_feedback_sum'] / aux['iteration_feedback_count']
                writer.add_scalar('Feedback/iteration_gate_mean', avg_iter_fb, global_step)
            
            # === Memory Gate Statistics ===
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
            
            # === Temperature ===
            temp = aux.get('temperature', config.get_temperature(global_step))
            writer.add_scalar('Training/temperature', temp, global_step)
        
        # Visualization and Image Logging
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
            
            # TensorBoard image logging
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
                print(f"[Warning] Image logging failed: {e}")
        
        # Progress bar
        passes = aux.get('passes_run', 1)
        pbar.set_postfix({
            'loss': f'{loss_val:.3f}',
            'cell': f'{cell_acc:.3f}',
            'task': f'{task_acc:.3f}',
            'P': f'{passes}',
        })
        
        # Cleanup
        del loss, logits, cache
        for key in list(aux.keys()):
            if isinstance(aux[key], list):
                aux[key].clear()
        aux.clear()
        
        if batch_idx % 20 == 0:
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
    visualize_samples: int = 3,
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
    print("\n" + "="*50)
    print(f"Model: {model.__class__.__name__}")
    print("="*50)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("="*50 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    import sys
    
    # Download Mini-ARC if needed
    data_path = download_miniarc("./Mini-ARC")
    
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
    # Mini-ARC needs smaller models due to simpler tasks and less data
    # 
    # TRM-style parameters:
    #   trm_outer_cycles: H_cycles - high-level refinement passes
    #   trm_inner_cycles: L_cycles - full stack repetitions per outer cycle
    #   max_internal_iterations: Per-layer ACT pondering (our addition, 1 = pure TRM)
    #
    # Default learning rate (can be overridden per-preset)
    lr = 5e-4
    
    if preset_name == "debug":
        # Minimal config for debugging
        d_model, d_cache = 32, 24
        num_layers, num_slots, num_heads = 2, 8, 2
        batch_size = 16
        trm_outer_cycles, trm_inner_cycles = 2, 2
        max_internal_iterations = 2
        num_epochs = 10
    elif preset_name == "fast":
        # Quick training for iteration
        d_model, d_cache = 64, 48
        num_layers, num_slots, num_heads = 3, 16, 2
        batch_size = 32
        trm_outer_cycles, trm_inner_cycles = 3, 2
        max_internal_iterations = 3
        num_epochs = 50
    elif preset_name == "fast_full":
        # Same as train_recursive.py fast_full - for comparison
        d_model, d_cache = 64, 48
        num_layers, num_slots, num_heads = 4, 32, 2
        batch_size = 8
        trm_outer_cycles, trm_inner_cycles = 3, 2  # TRM-style cycles
        max_internal_iterations = 3  # Per-layer pondering
        num_epochs = 100
    elif preset_name == "trm":
        # TRM-style: Matches TinyRecursiveModels architecture + our Memory Cache
        # Key TRM insights implemented:
        #   - z_H/z_L start from learned constants (H_init/L_init), NOT input embeddings
        #   - Input only enters via context injection: z_L = layers(z_L + (z_H + input))
        #   - z_H updates from z_L once per outer cycle: z_H = layers(z_H + z_L)
        #   - No-grad for H_cycles-1, only last outer cycle has gradients
        d_model, d_cache = 64, 48
        num_layers, num_slots, num_heads = 4, 32, 2
        batch_size = 16
        trm_outer_cycles = 4   # TRM's H_cycles (default in TRM is 4)
        trm_inner_cycles = 6   # TRM's L_cycles (default in TRM is 6)
        max_internal_iterations = 1  # No per-layer pondering (pure TRM style)
        num_epochs = 100
        lr = 3e-4  # Lower LR for TRM stability
    elif preset_name == "trm_minimal":
        # TRM-style but with minimal cycles for faster iteration
        # Good for debugging TRM architecture without long training times
        d_model, d_cache = 64, 48
        num_layers, num_slots, num_heads = 4, 32, 2
        batch_size = 16
        trm_outer_cycles = 2   # Fewer outer cycles (still needs at least 2 for z_H updates)
        trm_inner_cycles = 3   # Fewer inner cycles
        max_internal_iterations = 1  # No per-layer pondering
        num_epochs = 50
        lr = 3e-4  # Lower LR for TRM stability
    elif preset_name == "trm_deep":
        # TRM with more layers - tests if depth helps Mini-ARC
        d_model, d_cache = 64, 48
        num_layers, num_slots, num_heads = 6, 32, 2  # More layers
        batch_size = 16
        trm_outer_cycles = 3
        trm_inner_cycles = 4
        max_internal_iterations = 1
        num_epochs = 100
        lr = 3e-4  # Lower LR for TRM stability
    else:  # full
        # Full model for best performance
        d_model, d_cache = 128, 64
        num_layers, num_slots, num_heads = 4, 32, 4
        batch_size = 16
        trm_outer_cycles, trm_inner_cycles = 4, 3
        max_internal_iterations = 4
        num_epochs = 100
    
    # Dataset
    dataset = MiniARCDataset(str(data_path), augment=True)
    
    # Train/eval split (80/20)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nTrain samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    print(f"Sequence length: {MAX_SEQ_LEN} tokens (vs 900 for ARC-AGI-2)")
    print(f"Speedup: ~{900 // MAX_SEQ_LEN}x faster per sample")
    print(f"TRM cycles: outer={trm_outer_cycles}, inner={trm_inner_cycles}, layer_iter={max_internal_iterations}")
    
    # Config - use trm_outer_cycles as max_passes for backward compatibility
    config = TrainingConfig(
        tau_start=1.0,
        tau_min=0.1,
        max_passes=trm_outer_cycles,  # Maps to trm_outer_cycles
        max_recurrent_steps=max_internal_iterations,
        features=features,
        lambda_step_efficiency=0.02,
        lambda_diversity=0.01,
    )
    
    # Model - use Mini-ARC's smaller vocab and sequence length
    model = RecursiveRefinementModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        d_cache=d_cache,
        num_layers=num_layers,
        num_slots=num_slots,
        num_heads=num_heads,
        max_seq_len=MAX_SEQ_LEN,
        # TRM-style cycle parameters
        trm_outer_cycles=trm_outer_cycles,
        trm_inner_cycles=trm_inner_cycles,
        max_internal_iterations=max_internal_iterations,
        dropout=0.0,
        confidence_threshold=0.8,
        use_fixed_gate_threshold=config.use_fixed_gate_threshold,
        fixed_gate_threshold=config.gate_threshold,
        use_hippo_init=features.use_hippo_init,
    ).to(device)
    
    print_model_summary(model)
    
    # Optimizer (lr is set per-preset above)
    optimizer = AdamAtan2(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
    )
    print(f"Using AdamAtan2 optimizer (lr={lr})")
    
    # Logging
    log_dir = Path("logs") / f"miniarc_{preset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Training loop
    best_task_acc = 0
    global_step = 0
    
    for epoch in range(num_epochs):
        train_loss, train_cell_acc, train_task_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, config, global_step,
            writer=writer, log_interval=10,
        )
        
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/train_cell_acc', train_cell_acc, epoch)
        writer.add_scalar('Epoch/train_task_acc', train_task_acc, epoch)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            eval_cell_acc, eval_task_acc = evaluate(
                model, eval_loader, device, config,
                global_step=global_step, visualize_samples=2,
            )
            
            if eval_task_acc > best_task_acc:
                best_task_acc = eval_task_acc
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
