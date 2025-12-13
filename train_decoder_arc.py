"""
Training Script for Decoder-Only CNN + Cache Model on Full ARC-AGI-2
====================================================================

GPT-style decoder-only model with CNN compute blocks and selective cache memory.
Adapted from train_decoder_miniarc.py for full ARC-AGI-2 dataset (30x30 grids, 1000 tasks).

Key Differences from Mini-ARC:
- Larger grids: 30x30 max (vs 5x5)
- More tasks: 1000 training + 120 evaluation
- Longer sequences: 900 tokens per grid
- Variable demo counts: 2-10 demos per task

Usage:
    python train_decoder_arc.py [preset]
    
Presets:
    debug   - Minimal for testing
    fast    - Quick training (laptop-friendly)  
    medium  - Balanced performance
    full    - Best performance (needs GPU)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

# Import components
from components import (
    FeatureFlags,
    TrainingConfig,
    FEATURE_PRESETS,
    AdamAtan2,
)
from components.decoder_cache_model import DecoderCacheModel, create_decoder_cache_model
from components.logging import MetricsLogger
from components.dataset import (
    ARCDataset, ARCDatasetNoAug,
    VOCAB_SIZE, MAX_SEQ_LEN, PAD_TOKEN, COLOR_OFFSET, MAX_GRID_SIZE,
    sequence_to_grid, EOS_TOKEN, IGNORE_LABEL,
)


# ============================================================================
# Visualization (adapted for 30x30 grids)
# ============================================================================

ARC_COLORS = {
    0: '\033[40m', 1: '\033[44m', 2: '\033[41m', 3: '\033[42m', 4: '\033[43m',
    5: '\033[100m', 6: '\033[45m', 7: '\033[48;5;208m', 8: '\033[46m', 9: '\033[48;5;94m',
}
RESET = '\033[0m'

ARC_PALETTE = np.array([
    [0, 0, 0], [0, 116, 217], [255, 65, 54], [46, 204, 64], [255, 220, 0],
    [170, 170, 170], [240, 18, 190], [255, 133, 27], [127, 219, 255], [135, 12, 37],
], dtype=np.uint8)


def grid_to_rgb(grid: np.ndarray, cell_size: int = 10) -> np.ndarray:
    """Convert grid to RGB image."""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    # Handle float grids (from prediction)
    if grid.dtype == np.float32 or grid.dtype == np.float64:
        grid = grid.astype(np.int32)
        
    grid = np.clip(grid, 0, 9).astype(np.int32)
    rgb = ARC_PALETTE[grid]
    
    # Upscale
    if cell_size > 1:
        rgb = np.repeat(np.repeat(rgb, cell_size, axis=0), cell_size, axis=1)
        
    return rgb


def create_sample_image(
    demo_inputs: torch.Tensor,
    demo_outputs: torch.Tensor,
    test_input: torch.Tensor,
    test_output: torch.Tensor,
    prediction: torch.Tensor,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Create a composite image of the sample."""
    # Convert all to grids
    num_demos = min(demo_inputs.shape[0], 3) # Limit to 3 demos
    
    grids = []
    
    # Demos
    for d in range(num_demos):
        in_grid = aligned_seq_to_grid(demo_inputs[d].cpu().numpy())
        out_grid = aligned_seq_to_grid(demo_outputs[d].cpu().numpy())
        grids.append((in_grid, out_grid))
        
    # Test
    test_in_grid = aligned_seq_to_grid(test_input.cpu().numpy())
    
    # Target
    target_np = test_output.cpu().numpy()
    target_for_grid = np.where(target_np == IGNORE_LABEL, PAD_TOKEN, target_np)
    target_grid = aligned_seq_to_grid(target_for_grid)
    
    # Prediction
    out_size_tuple = None
    if output_size is not None:
        if isinstance(output_size, torch.Tensor):
            out_size_tuple = (output_size[0].item(), output_size[1].item())
        else:
            out_size_tuple = output_size
            
    pred_grid = prediction_to_grid(
        prediction.cpu().numpy(), 
        test_output.cpu().numpy(),
        output_size=out_size_tuple,
    )
    
    # Convert to RGB
    cell_size = 5 # Smaller cell size for 30x30 grids
    
    # Helper to pad grids to same size (30x30) for consistent layout
    def pad_to_30(g):
        h, w = g.shape
        padded = np.zeros((30, 30), dtype=g.dtype)
        padded[:h, :w] = g
        return padded
        
    # Create rows
    rows = []
    
    # Demos: Input -> Output
    for in_g, out_g in grids:
        in_rgb = grid_to_rgb(pad_to_30(in_g), cell_size)
        out_rgb = grid_to_rgb(pad_to_30(out_g), cell_size)
        # Add arrow/spacing
        sep = np.zeros((in_rgb.shape[0], 10, 3), dtype=np.uint8) + 255
        row = np.concatenate([in_rgb, sep, out_rgb], axis=1)
        rows.append(row)
        
    # Test: Input -> Target -> Prediction
    test_in_rgb = grid_to_rgb(pad_to_30(test_in_grid), cell_size)
    target_rgb = grid_to_rgb(pad_to_30(target_grid), cell_size)
    pred_rgb = grid_to_rgb(pad_to_30(pred_grid), cell_size)
    
    sep = np.zeros((test_in_rgb.shape[0], 10, 3), dtype=np.uint8) + 255
    test_row = np.concatenate([test_in_rgb, sep, target_rgb, sep, pred_rgb], axis=1)
    
    # Pad demo rows to match test row width
    max_w = test_row.shape[1]
    final_rows = []
    for row in rows:
        if row.shape[1] < max_w:
            pad_w = max_w - row.shape[1]
            pad = np.zeros((row.shape[0], pad_w, 3), dtype=np.uint8) + 255
            row = np.concatenate([row, pad], axis=1)
        final_rows.append(row)
        
    final_rows.append(test_row)
    
    # Stack vertically with spacing
    sep_h = np.zeros((10, max_w, 3), dtype=np.uint8) + 255
    full_img = final_rows[0]
    for i in range(1, len(final_rows)):
        full_img = np.concatenate([full_img, sep_h, final_rows[i]], axis=0)
        
    return full_img


def grid_to_ascii(grid, max_width: int = 15, max_height: int = 10) -> list:
    """Convert grid to ASCII art (truncated for display)."""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    h, w = min(grid.shape[0], max_height), min(grid.shape[1], max_width)
    lines = []
    for row in range(h):
        line = ""
        for col in range(w):
            color = int(np.clip(grid[row, col], 0, 9))
            line += f"{ARC_COLORS[color]}  {RESET}"
        if w < grid.shape[1]:
            line += "..."
        lines.append(line)
    if h < grid.shape[0]:
        lines.append("...")
    lines.append(f"({grid.shape[0]}×{grid.shape[1]})")
    return lines


def aligned_seq_to_grid(seq: np.ndarray, output_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Convert aligned sequence (30x30 flattened) back to grid.
    
    The aligned sequence is a flattened 30x30 grid where:
    - PAD_TOKEN (0) = padding
    - EOS_TOKEN (1) = boundary markers  
    - Colors 4-13 = actual content
    """
    # Reshape to 30x30
    grid_30x30 = seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    
    # Find actual content bounds (non-PAD, non-EOS)
    content_mask = (grid_30x30 >= COLOR_OFFSET) & (grid_30x30 < COLOR_OFFSET + 10)
    
    if not content_mask.any():
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Find bounding box
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    
    # Extract and convert colors
    cropped = grid_30x30[r_min:r_max+1, c_min:c_max+1]
    result = np.zeros_like(cropped, dtype=np.uint8)
    color_mask = (cropped >= COLOR_OFFSET) & (cropped < COLOR_OFFSET + 10)
    result[color_mask] = cropped[color_mask] - COLOR_OFFSET
    
    return result


def prediction_to_grid(
    prediction: np.ndarray, 
    target: np.ndarray,
    output_size: Tuple[int, int] = None,
) -> np.ndarray:
    """Convert prediction sequence to grid, using target to find valid positions."""
    # Both are flattened 30x30
    pred_2d = prediction.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    target_2d = target.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    
    # Find valid mask from target (non-IGNORE_LABEL AND non-EOS positions = content only)
    # EOS_TOKEN = 1, we exclude these from valid positions too
    valid_mask = (target_2d != IGNORE_LABEL) & (target_2d != EOS_TOKEN)
    
    if not valid_mask.any():
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Find bounding box of valid region (content only, no EOS)
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    
    # If output_size provided, use it directly for more accurate cropping
    if output_size is not None:
        h, w = output_size
        # Crop from (r_min, c_min) with exact size
        cropped_pred = pred_2d[r_min:r_min+h, c_min:c_min+w]
        cropped_valid = valid_mask[r_min:r_min+h, c_min:c_min+w]
    else:
        # Extract predictions in valid region
        cropped_pred = pred_2d[r_min:r_max+1, c_min:c_max+1]
        cropped_valid = valid_mask[r_min:r_max+1, c_min:c_max+1]
    
    # Convert to colors
    result = np.zeros_like(cropped_pred, dtype=np.uint8)
    color_mask = cropped_valid & (cropped_pred >= COLOR_OFFSET) & (cropped_pred < COLOR_OFFSET + 10)
    result[color_mask] = cropped_pred[color_mask] - COLOR_OFFSET
    
    return result


def visualize_sample(
    demo_inputs: torch.Tensor,
    demo_outputs: torch.Tensor,
    test_input: torch.Tensor,
    test_output: torch.Tensor,
    prediction: torch.Tensor,
    output_size: torch.Tensor,
    sample_idx: int = 0,
    step: int = 0,
    aux: Optional[Dict] = None,
    task_id: str = "",
) -> str:
    """Visualize an ARC sample with demos, test, and prediction."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"[Step {step}] ARC-AGI-2 Task: {task_id} (Sample #{sample_idx})")
    
    if aux:
        read_gates = aux.get('read_gates', [])
        write_gates = aux.get('write_gates', [])
        if read_gates:
            avg_read = sum(g.detach().mean().item() for g in read_gates) / len(read_gates)
            lines.append(f"  Avg Read Gate: {avg_read:.3f}")
        if write_gates:
            avg_write = sum(g.detach().mean().item() for g in write_gates) / len(write_gates)
            lines.append(f"  Avg Write Gate: {avg_write:.3f}")
    
    lines.append(f"{'='*70}")
    
    # Demo pairs (show first 2)
    num_demos = demo_inputs.shape[0]
    for d in range(min(num_demos, 2)):
        demo_in_grid = aligned_seq_to_grid(demo_inputs[d].cpu().numpy())
        demo_out_grid = aligned_seq_to_grid(demo_outputs[d].cpu().numpy())
        
        in_lines = grid_to_ascii(demo_in_grid)
        out_lines = grid_to_ascii(demo_out_grid)
        
        max_h = max(len(in_lines), len(out_lines))
        while len(in_lines) < max_h: in_lines.insert(-1, " " * 20)
        while len(out_lines) < max_h: out_lines.insert(-1, " " * 20)
        
        lines.append(f"\n  Demo {d+1}:")
        lines.append(f"  {'Input':^20} → {'Output':^20}")
        for i in range(max_h):
            lines.append(f"  {in_lines[i]:20}   {out_lines[i]:20}")
    
    if num_demos > 2:
        lines.append(f"  ... and {num_demos - 2} more demos")
    
    # Test
    test_in_grid = aligned_seq_to_grid(test_input.cpu().numpy())
    
    # For target, we need to handle IGNORE_LABEL
    target_np = test_output.cpu().numpy()
    target_for_grid = np.where(target_np == IGNORE_LABEL, PAD_TOKEN, target_np)
    target_grid = aligned_seq_to_grid(target_for_grid)
    
    # Convert output_size tensor to tuple if provided
    out_size_tuple = None
    if output_size is not None:
        if isinstance(output_size, torch.Tensor):
            out_size_tuple = (output_size[0].item(), output_size[1].item())
        else:
            out_size_tuple = tuple(output_size)
    
    pred_grid = prediction_to_grid(
        prediction.cpu().numpy(), 
        test_output.cpu().numpy(),
        output_size=out_size_tuple,
    )
    
    test_lines = grid_to_ascii(test_in_grid)
    tgt_lines = grid_to_ascii(target_grid)
    pred_lines = grid_to_ascii(pred_grid)
    
    max_h = max(len(test_lines), len(tgt_lines), len(pred_lines))
    while len(test_lines) < max_h: test_lines.insert(-1, " " * 20)
    while len(tgt_lines) < max_h: tgt_lines.insert(-1, " " * 20)
    while len(pred_lines) < max_h: pred_lines.insert(-1, " " * 20)
    
    lines.append(f"\n  Test:")
    lines.append(f"  {'Input':^20}   {'Target':^20}   {'Prediction':^20}")
    for i in range(max_h):
        lines.append(f"  {test_lines[i]:20}   {tgt_lines[i]:20}   {pred_lines[i]:20}")
    
    # Accuracy
    valid_mask = (test_output != IGNORE_LABEL) & (test_output != EOS_TOKEN)
    if valid_mask.any():
        correct = (prediction[valid_mask] == test_output[valid_mask]).float().mean().item()
        marker = "✓" if correct == 1.0 else "✗"
        lines.append(f"\n  Accuracy: {correct*100:.1f}% {marker}")
    
    lines.append(f"{'='*70}\n")
    return '\n'.join(lines)


# ============================================================================
# Loss Computation (same as Mini-ARC)
# ============================================================================

def compute_decoder_loss(
    logits: torch.Tensor,       # [B, S, V]
    targets: torch.Tensor,      # [B, S]
    aux: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """
    Loss for decoder model on ARC-AGI.
    
    Components:
    1. Task loss: Cross-entropy on valid positions only
    2. Gate usage loss: Encourage gates to stay near target values (prevent collapse)
    """
    # === 1. TASK LOSS ===
    B, S, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    
    task_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE_LABEL)
    
    # === 2. GATE TARGET LOSS ===
    # Encourage gates to stay near target values to prevent collapse
    # read_target ~0.4: model should read from cache moderately
    # write_target ~0.3: model should write selectively but not rarely
    gate_loss = torch.tensor(0.0, device=device)
    avg_read = torch.tensor(0.0, device=device)
    avg_write = torch.tensor(0.0, device=device)
    
    read_target = 0.4
    write_target = 0.3
    
    read_gates = aux.get('read_gates', [])
    write_gates = aux.get('write_gates', [])
    
    if read_gates:
        # Calculate averages
        avg_read = torch.stack([g.mean() for g in read_gates]).mean()
        avg_write = torch.stack([g.mean() for g in write_gates]).mean() if write_gates else torch.tensor(0.0, device=device)
        
        # Target gate loss: penalize deviation from target
        # Use squared error so small deviations are tolerated
        read_loss = (avg_read - read_target) ** 2
        write_loss = (avg_write - write_target) ** 2 if write_gates else torch.tensor(0.0, device=device)
        
        gate_loss = read_loss + write_loss
        
    # === 3. TOTAL LOSS ===
    # Weight gate loss to be meaningful but not dominant
    # 0.1 weight means ~10% of gradient comes from gate regularization
    total_loss = task_loss + 0.1 * gate_loss
    
    metrics = {
        'loss_total': total_loss.detach().item(),
        'loss_task': task_loss.detach().item(),
        'loss_gate': gate_loss.detach().item(),
        'avg_read_gate': avg_read.detach().item() if read_gates else 0.0,
        'avg_write_gate': avg_write.detach().item() if (read_gates and write_gates) else 0.0,
    }
    
    return total_loss, metrics


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: DecoderCacheModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_step: int = 0,
    logger: Optional[MetricsLogger] = None,
    log_interval: int = 10,
    temperature: float = 1.0,
    grad_accum_steps: int = 1,
    visualize_interval: int = 100,
) -> Tuple[float, float, float, int]:
    """Training epoch with teacher forcing and gradient accumulation."""
    model.train()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_tasks = 0
    total_tasks = 0
    accum_loss = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        
        # Forward with teacher forcing
        logits, cache, aux = model(
            demo_inputs, demo_outputs, test_input, test_output,
            temperature=temperature,
            hard=(temperature < 0.2),
            return_aux=True,
        )
        
        # Loss (scaled for accumulation)
        loss, metrics = compute_decoder_loss(logits, test_output, aux, device)
        loss = loss / grad_accum_steps
        
        # Backward
        loss.backward()
        accum_loss += loss.detach().item()
        
        # Optimizer step after accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Check for NaN
            has_nan = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters()
            )
            if has_nan:
                print(f"[Step {global_step}] Skipping due to NaN/Inf gradients")
                optimizer.zero_grad()
                accum_loss = 0
            else:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += accum_loss * grad_accum_steps
            accum_loss = 0
            global_step += 1
        
        # Metrics
        preds = logits.detach().argmax(dim=-1)
        for i in range(preds.shape[0]):
            valid_mask = (test_output[i] != IGNORE_LABEL) & (test_output[i] != EOS_TOKEN)
            if valid_mask.any():
                correct_cells += (preds[i][valid_mask] == test_output[i][valid_mask]).sum().item()
                total_cells += valid_mask.sum().item()
                if (preds[i][valid_mask] == test_output[i][valid_mask]).all():
                    correct_tasks += 1
            total_tasks += 1
        
        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        
        # Logging
        if logger is not None and batch_idx % log_interval == 0:
            read_gates = aux.get('read_gates', [])
            write_gates = aux.get('write_gates', [])
            
            log_aux = {'passes_run': 1}
            
            if read_gates:
                read_sum = sum(g.sum().detach().item() for g in read_gates)
                read_count = sum(g.numel() for g in read_gates)
                log_aux['read_gate_sum'] = read_sum
                log_aux['read_gate_count'] = read_count
            
            if write_gates:
                write_sum = sum(g.sum().detach().item() for g in write_gates)
                write_count = sum(g.numel() for g in write_gates)
                log_aux['write_gate_sum'] = write_sum
                log_aux['write_gate_count'] = write_count
            
            logger.log_step(
                step=global_step,
                metrics=metrics,
                aux=log_aux,
                config=None,
                cell_acc=cell_acc,
                task_acc=task_acc,
            )
        
        # Visualization
        if global_step % visualize_interval == 0 and global_step > 0:
            task_id = batch.get("task_id", ["unknown"])[0] if "task_id" in batch else "unknown"
            output_size = batch.get("output_size", torch.tensor([[10, 10]]))[0]
            viz_str = visualize_sample(
                demo_inputs=demo_inputs[0],
                demo_outputs=demo_outputs[0],
                test_input=test_input[0],
                test_output=test_output[0],
                prediction=preds[0],
                output_size=output_size,
                sample_idx=0,
                step=global_step,
                aux=aux,
                task_id=task_id,
            )
            print(viz_str)
            
            # Log image to TensorBoard
            if logger is not None:
                img = create_sample_image(
                    demo_inputs=demo_inputs[0],
                    demo_outputs=demo_outputs[0],
                    test_input=test_input[0],
                    test_output=test_output[0],
                    prediction=preds[0],
                    output_size=output_size,
                )
                # Convert to CHW format for TensorBoard (HWC -> CHW)
                img_chw = np.transpose(img, (2, 0, 1))
                logger.log_image("Visualization/sample", img_chw, global_step)
        
        # Progress bar
        pbar.set_postfix({
            'loss': f'{metrics["loss_total"]:.3f}',
            'cell': f'{cell_acc:.3f}',
            'task': f'{task_acc:.3f}',
        })
        
        # Cleanup
        del loss, logits, cache
        for key in list(aux.keys()):
            if isinstance(aux[key], list):
                aux[key].clear()
        aux.clear()
        
        if batch_idx % 50 == 0:
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Handle remaining gradients
    if (len(dataloader) % grad_accum_steps) != 0:
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += accum_loss * grad_accum_steps
        global_step += 1
    
    return total_loss / max(len(dataloader) // grad_accum_steps, 1), cell_acc, task_acc, global_step


@torch.no_grad()
def evaluate(
    model: DecoderCacheModel,
    dataloader: DataLoader,
    device: torch.device,
    global_step: int = 0,
    visualize_samples: int = 3,
    use_generation: bool = False,
    logger: Optional[MetricsLogger] = None,
) -> Tuple[float, float]:
    """
    Evaluation.
    
    Args:
        use_generation: If True, use autoregressive generation (slower but realistic).
                       If False, use teacher forcing (faster).
    """
    model.eval()
    correct_tasks = 0
    total_tasks = 0
    correct_cells = 0
    total_cells = 0
    samples_visualized = 0
    
    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for batch in pbar:
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        
        if use_generation:
            # Autoregressive generation
            preds = model.generate(
                demo_inputs, demo_outputs, test_input,
                max_len=test_output.shape[1],
                temperature=0.0,  # Greedy
            )
            aux = None
        else:
            # Teacher forcing (faster)
            logits, _, aux = model(
                demo_inputs, demo_outputs, test_input, test_output,
                temperature=0.5,
                hard=True,
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
                    task_id = batch.get("task_id", ["unknown"])[i] if "task_id" in batch else "unknown"
                    output_size = batch.get("output_size", torch.tensor([[10, 10]]))[i]
                    marker = "✓" if task_correct else "✗"
                    print(f"\n[Eval Sample {samples_visualized + 1}] {marker}")
                    viz_str = visualize_sample(
                        demo_inputs=demo_inputs[i],
                        demo_outputs=demo_outputs[i],
                        test_input=test_input[i],
                        test_output=test_output[i],
                        prediction=preds[i],
                        output_size=output_size,
                        sample_idx=samples_visualized,
                        step=global_step,
                        aux=aux,
                        task_id=task_id,
                    )
                    print(viz_str)
                    
                    # Log image to TensorBoard
                    if logger is not None:
                        img = create_sample_image(
                            demo_inputs=demo_inputs[i],
                            demo_outputs=demo_outputs[i],
                            test_input=test_input[i],
                            test_output=test_output[i],
                            prediction=preds[i],
                            output_size=output_size,
                        )
                        # Convert to CHW format for TensorBoard (HWC -> CHW)
                        img_chw = np.transpose(img, (2, 0, 1))
                        logger.log_image(f"Visualization/eval_sample_{samples_visualized}", img_chw, global_step)
                        
                    samples_visualized += 1
            
            total_tasks += 1
        
        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        pbar.set_postfix({'cell': f'{cell_acc:.3f}', 'task': f'{task_acc:.3f}'})
    
    return cell_acc, task_acc


def print_model_summary(model: torch.nn.Module):
    """Print model parameter summary."""
    print("\n" + "="*70)
    print(f"Model: {model.__class__.__name__}")
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    embed_params = sum(p.numel() for n, p in model.named_parameters() if 'embed' in n)
    conv_params = sum(p.numel() for n, p in model.named_parameters() if 'conv' in n)
    memory_params = sum(p.numel() for n, p in model.named_parameters() if 'memory' in n)
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"  - Embeddings:       {embed_params:,}")
    print(f"  - Convolutions:     {conv_params:,}")
    print(f"  - Memory/Cache:     {memory_params:,}")
    print("="*70 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    import sys
    
    # Data path
    data_dir = "./ARC-AGI-2/data"
    
    # Device
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Preset
    preset_name = sys.argv[1] if len(sys.argv) > 1 else "fast"
    print(f"\nUsing preset: '{preset_name}'")
    
    # Model configuration per preset
    # ARC-AGI-2 needs larger model due to 30x30 grids (900 tokens vs 25)
    lr = 3e-4
    grad_accum_steps = 1
    soft_eviction = False  # Default to hard eviction
    
    if preset_name == "debug":
        d_model, d_cache = 64, 32
        num_layers, num_slots = 3, 16
        kernel_size, num_conv_layers = 5, 2
        batch_size = 2
        num_epochs = 5
        eval_split_ratio = 0.9  # Use 90% train, 10% eval
        grad_accum_steps = 1
    elif preset_name == "fast":
        d_model, d_cache = 128, 64
        num_layers, num_slots = 4, 32
        kernel_size, num_conv_layers = 5, 2
        batch_size = 2
        num_epochs = 20
        lr = 2e-4
        eval_split_ratio = 0.9
        grad_accum_steps = 2
    elif preset_name == "medium":
        d_model, d_cache = 192, 96
        num_layers, num_slots = 6, 48
        kernel_size, num_conv_layers = 7, 3
        batch_size = 2
        num_epochs = 50
        lr = 1e-4
        eval_split_ratio = 0.9
        grad_accum_steps = 4
    elif preset_name == "fast_full":
        # Larger model for serious training
        # d_model, d_cache = 256, 128
        # num_layers, num_slots = 8, 64
        # kernel_size, num_conv_layers = 7, 3
        # batch_size = 1
        # num_epochs = 100
        # lr = 1e-4
        d_model, d_cache = 32, 16
        num_layers, num_slots = 5, 32  # Increased slots from 32 to 192
        kernel_size, num_conv_layers = 3, 5
        batch_size = 4
        num_epochs = 100
        lr = 1e-3
        eval_split_ratio = 0.9
        grad_accum_steps = 1
        soft_eviction = True
    else:  # full
        d_model, d_cache = 384, 192
        num_layers, num_slots = 10, 96
        kernel_size, num_conv_layers = 7, 4
        batch_size = 1
        num_epochs = 200
        lr = 5e-5
        eval_split_ratio = 0.9
        grad_accum_steps = 16
    
    # Dataset
    print(f"\nLoading ARC-AGI-2 dataset from {data_dir}...")
    train_dataset = ARCDataset(data_dir, split="training", augment=True)
    
    # Split into train/eval (ARC doesn't have separate train/eval with answers)
    total_samples = len(train_dataset)
    train_size = int(eval_split_ratio * total_samples)
    eval_size = total_samples - train_size
    
    train_subset, eval_subset = torch.utils.data.random_split(
        train_dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # For eval, we want no augmentation
    eval_dataset = ARCDatasetNoAug(data_dir, split="training")
    eval_indices = eval_subset.indices
    eval_subset_noaug = torch.utils.data.Subset(eval_dataset, eval_indices)
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )
    eval_loader = DataLoader(
        eval_subset_noaug, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
    )
    
    print(f"Train samples: {len(train_subset)}, Eval samples: {len(eval_subset)}")
    print(f"Sequence length: {MAX_SEQ_LEN} (30x30 grids)")
    print(f"\nConfig: d_model={d_model}, d_cache={d_cache}")
    print(f"        {num_layers} layers × {num_slots} slots")
    print(f"        kernel={kernel_size}, conv_layers={num_conv_layers}")
    print(f"        batch_size={batch_size}, grad_accum={grad_accum_steps}")
    print(f"        effective_batch={batch_size * grad_accum_steps}")
    
    # Model - calculate actual max sequence length
    # Format: [demo_in_1, demo_out_1, ..., demo_in_N, demo_out_N, test_in, test_out]
    # Each grid = MAX_SEQ_LEN (900 tokens for 30x30)
    # With 3 demos: 3*900 + 3*900 + 900 + 900 = 7200 tokens
    num_demos = 3
    actual_max_seq_len = (2 * num_demos + 2) * MAX_SEQ_LEN  # 8 * 900 = 7200
    
    model = DecoderCacheModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        d_cache=d_cache,
        num_layers=num_layers,
        num_slots=num_slots,
        kernel_size=kernel_size,
        num_conv_layers_per_block=num_conv_layers,
        max_seq_len=actual_max_seq_len,
        #dropout=0.1,  # Regularization for larger dataset
        soft_eviction=soft_eviction,
    ).to(device)
    
    print_model_summary(model)
    
    # Optimizer
    optimizer = AdamAtan2(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        #weight_decay=0.01,  # L2 regularization
    )
    print(f"Using AdamAtan2 optimizer (lr={lr}, wd=0.01)")
    
    # Learning rate scheduler (cosine annealing per epoch)
    # Note: step() is called once per epoch, so T_max = num_epochs
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_epochs, eta_min=lr * 0.01  # Decay to 1% of initial LR
    # )
    #print(f"Using cosine LR schedule (T_max={num_epochs} epochs, min_lr={lr*0.01:.2e})")
    
    # Logging
    log_dir = Path("logs") / f"decoder_arc_{preset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger = MetricsLogger(writer)
    print(f"TensorBoard logs: {log_dir}")
    
    # Temperature schedule (anneal from 1.0 to 0.1)
    def get_temperature(epoch: int, num_epochs: int) -> float:
        return max(0.1, 1.0 - 0.9 * epoch / num_epochs)
    
    # Training
    best_task_acc = 0
    best_cell_acc = 0
    global_step = 0
    
    for epoch in range(num_epochs):
        temperature = get_temperature(epoch, num_epochs)
        
        train_loss, train_cell_acc, train_task_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, global_step,
            logger=logger, 
            log_interval=10,
            temperature=temperature,
            grad_accum_steps=grad_accum_steps,
            visualize_interval=200,
        )
        
        # Step scheduler
        # scheduler.step()
        # current_lr = scheduler.get_last_lr()[0]
        
        logger.log_epoch(epoch, train_loss, train_cell_acc, train_task_acc)
        
        # Evaluate every 5 epochs (or every epoch for debug)
        eval_interval = 1 if preset_name == "debug" else 5
        if (epoch + 1) % eval_interval == 0:
            eval_cell_acc, eval_task_acc = evaluate(
                model, eval_loader, device,
                global_step=global_step, 
                visualize_samples=2,
                use_generation=False,
                logger=logger,
            )
            
            # Save best model (prefer task accuracy, then cell accuracy)
            if eval_task_acc > best_task_acc or (eval_task_acc == best_task_acc and eval_cell_acc > best_cell_acc):
                best_task_acc = eval_task_acc
                best_cell_acc = eval_cell_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_task_acc': best_task_acc,
                    'best_cell_acc': best_cell_acc,
                }, log_dir / "best_model.pt")
                print(f"  → Saved best model (task={eval_task_acc:.3f}, cell={eval_cell_acc:.3f})")
            
            logger.log_epoch(epoch, train_loss, train_cell_acc, train_task_acc,
                            eval_cell_acc=eval_cell_acc, eval_task_acc=eval_task_acc)
            
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {train_loss:.4f} | "
                  f"Eval Task: {eval_task_acc:.3f} | Eval Cell: {eval_cell_acc:.3f} | ")
                  #f"Best: {best_task_acc:.3f} | τ: {temperature:.2f} | lr: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {train_loss:.4f} | "
                  f"Train Task: {train_task_acc:.3f} | Train Cell: {train_cell_acc:.3f} | ")
                   #f"τ: {temperature:.2f} | lr: {current_lr:.2e}")
        
        # Checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'scheduler_state_dict': scheduler.state_dict(),
            }, log_dir / f"checkpoint_epoch{epoch+1}.pt")
    
    # Final evaluation with autoregressive generation
    print("\n" + "="*70)
    print("Final evaluation with autoregressive generation...")
    print("="*70)
    eval_cell_acc, eval_task_acc = evaluate(
        model, eval_loader, device,
        global_step=global_step, 
        visualize_samples=5,
        use_generation=True,
        logger=logger,
    )
    print(f"Autoregressive Eval: Cell={eval_cell_acc:.3f}, Task={eval_task_acc:.3f}")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'final_task_acc': eval_task_acc,
        'final_cell_acc': eval_cell_acc,
    }, log_dir / "final_model.pt")
    print(f"  → Saved final model")
    
    logger.close()
    print(f"\n{'='*70}")
    print(f"ARC-AGI-2 Decoder + Cache Training Complete!")
    print(f"Best task accuracy: {best_task_acc:.3f}")
    print(f"Best cell accuracy: {best_cell_acc:.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
