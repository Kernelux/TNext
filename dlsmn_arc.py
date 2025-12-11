"""
DLSMN on ARC-AGI-2
==================
Faithful implementation of DLSM_V0.1.md for the Abstraction and Reasoning Corpus.

This file now serves as the entry point, importing modular components.

Now uses sequence-based format (TRM-style) instead of 2D grids:
- Grids flattened to sequences with EOS markers
- Vocab: PAD=0, EOS=1, INPUT=2, OUTPUT=3, colors 0-9 → tokens 4-13
- Role markers (INPUT/OUTPUT) make input/output pairing explicit
- Standard cross-entropy loss on sequences
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
import psutil
from pathlib import Path
from typing import Tuple, Optional

# Import modular components
from components import (
    FeatureFlags,
    TrainingConfig,
    FEATURE_PRESETS,
    ARCDataset,
    DLSMN_ARC,
    EMA,
    compute_total_loss,
    AdamAtan2,
)
from components.dataset import (
    VOCAB_SIZE, MAX_SEQ_LEN, PAD_TOKEN, EOS_TOKEN, COLOR_OFFSET,
    INPUT_MARKER, OUTPUT_MARKER, MAX_GRID_SIZE,
    grid_to_sequence,  # Note: sequence_to_grid defined locally for TRM-style
)


# ============================================================================
# Gradient Monitoring
# ============================================================================

def monitor_gradients(model: DLSMN_ARC, writer: Optional[SummaryWriter], step: int, verbose: bool = False):
    """
    Monitor gradient flow through different parts of the model.
    Helps diagnose vanishing/exploding gradients.
    """
    grad_stats = {}
    
    # === 1. Embeddings ===
    if model.token_embed.weight.grad is not None:
        grad_stats['embed/token'] = model.token_embed.weight.grad.abs().mean().item()
    if model.slot_embeddings.grad is not None:
        grad_stats['embed/slots'] = model.slot_embeddings.grad.abs().mean().item()
    
    # === 2. Per-Layer Statistics ===
    for i, layer in enumerate(model.layers):
        layer_grads = []
        for name, param in layer.named_parameters():
            if param.grad is not None:
                layer_grads.append(param.grad.abs().mean().item())
        if layer_grads:
            grad_stats[f'layer_{i}/mean'] = sum(layer_grads) / len(layer_grads)
            grad_stats[f'layer_{i}/max'] = max(layer_grads)
        
        # Specific components within layer
        for name, param in layer.named_parameters():
            if param.grad is not None:
                if 'router' in name:
                    grad_stats[f'layer_{i}/router'] = param.grad.abs().mean().item()
                elif 'W_compress' in name:
                    grad_stats[f'layer_{i}/compress'] = param.grad.abs().mean().item()
                elif 'W_decompress' in name:
                    grad_stats[f'layer_{i}/decompress'] = param.grad.abs().mean().item()
                elif 'gate' in name.lower():
                    grad_stats[f'layer_{i}/gates'] = param.grad.abs().mean().item()
    
    # === 3. Cache Self-Attention ===
    for name, param in model.cache_self_attn.named_parameters():
        if param.grad is not None:
            grad_stats[f'cache_attn/{name}'] = param.grad.abs().mean().item()
    
    # === 4. Phase-specific Q-Heads (halting) ===
    for phase_name, q_head in [('reflection', model.reflection_q_head), ('answer', model.answer_q_head)]:
        for i, layer in enumerate(q_head):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                grad_stats[f'q_head_{phase_name}/{i}'] = layer.weight.grad.abs().mean().item()
    
    # === 5. Output Head ===
    if model.output_proj.weight.grad is not None:
        grad_stats['output/proj'] = model.output_proj.weight.grad.abs().mean().item()
    
    # === 6. Phase-specific Step Predictors ===
    for phase_name, step_pred in [('reflection', model.reflection_step_predictor), ('answer', model.answer_step_predictor)]:
        for i, layer in enumerate(step_pred):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                grad_stats[f'step_pred_{phase_name}/{i}'] = layer.weight.grad.abs().mean().item()
    
    # === Log to TensorBoard ===
    if writer is not None:
        for name, value in grad_stats.items():
            writer.add_scalar(f'Gradients/{name}', value, step)
    
    # === Console Output ===
    if verbose:
        print(f"\n[Step {step}] Gradient Flow:")
        print("-" * 50)
        
        # Group by category
        categories = {}
        for name, value in grad_stats.items():
            cat = name.split('/')[0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, value))
        
        for cat, items in sorted(categories.items()):
            values = [v for _, v in items]
            avg = sum(values) / len(values) if values else 0
            print(f"  {cat:20s}: avg={avg:.2e}, max={max(values) if values else 0:.2e}")
        
        # Check for vanishing/exploding gradients
        # NOTE: AdamAtan2 is scale-invariant, so large gradients are handled naturally.
        # These warnings are mainly informational for monitoring.
        all_grads = list(grad_stats.values())
        if all_grads:
            if max(all_grads) < 1e-7:
                print("  ⚠️ WARNING: All gradients very small (< 1e-7) - vanishing gradients!")
            if max(all_grads) > 1e6:
                print(f"  ℹ️ Note: Large gradients ({max(all_grads):.2e}) - AdamAtan2 handles this")
    
    return grad_stats


# ============================================================================
# Sample Visualization (Sequences → Grids)
# ============================================================================

import numpy as np

# ARC color palette (RGB values for TensorBoard images)
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

# ANSI color codes for terminal visualization
ARC_COLORS = {
    0: '\033[40m',   # Black
    1: '\033[44m',   # Blue
    2: '\033[41m',   # Red
    3: '\033[42m',   # Green
    4: '\033[43m',   # Yellow
    5: '\033[100m',  # Gray (bright black)
    6: '\033[45m',   # Magenta
    7: '\033[48;5;208m',  # Orange
    8: '\033[46m',   # Cyan
    9: '\033[48;5;94m',   # Brown
}
RESET = '\033[0m'


def grid_to_rgb(grid: np.ndarray, cell_size: int = 10) -> np.ndarray:
    """
    Convert a grid (H, W) with values 0-9 to an RGB image.
    
    Args:
        grid: numpy array [H, W] with values 0-9
        cell_size: pixels per cell (for scaling up small grids)
        
    Returns:
        RGB image [H*cell_size, W*cell_size, 3]
    """
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    h, w = grid.shape
    
    # Clamp to valid range
    grid = np.clip(grid, 0, 9).astype(np.int32)
    
    # Map to colors
    rgb = ARC_PALETTE[grid]  # [H, W, 3]
    
    # Scale up for visibility
    if cell_size > 1:
        rgb = np.repeat(np.repeat(rgb, cell_size, axis=0), cell_size, axis=1)
    
    return rgb


def prediction_to_grid(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Convert prediction sequence to grid using TRM-style aligned format.
    
    The prediction and target are both 30x30 flattened sequences (900 tokens).
    Target uses -100 (IGNORE_LABEL) for positions we don't evaluate (PAD).
    
    Args:
        prediction: [900] predicted tokens at each position
        target: [900] target tokens (-100 for ignored positions, 4-13 for colors)
        
    Returns:
        [H, W] grid with values 0-9 (cropped to actual content)
    """
    # Reshape to 30x30
    pred_2d = prediction.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    target_2d = target.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    
    # Find bounding box of valid content (where target != -100 and target != PAD)
    valid_mask = (target_2d != -100) & (target_2d != PAD_TOKEN) & (target_2d != EOS_TOKEN)
    
    if not valid_mask.any():
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Get bounding box
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Extract prediction grid for valid region
    pred_crop = pred_2d[rmin:rmax+1, cmin:cmax+1].copy()
    
    # Convert tokens to colors (0-9)
    # Prediction tokens: PAD=0, EOS=1, INPUT=2, OUTPUT=3, colors 4-13
    grid = np.zeros_like(pred_crop, dtype=np.uint8)
    for r in range(pred_crop.shape[0]):
        for c in range(pred_crop.shape[1]):
            token = pred_crop[r, c]
            if token >= COLOR_OFFSET:
                grid[r, c] = token - COLOR_OFFSET
            else:
                grid[r, c] = 0  # Treat special tokens as black
    
    return np.clip(grid, 0, 9)


def target_to_grid(target: np.ndarray) -> np.ndarray:
    """
    Convert target sequence to grid using TRM-style aligned format.
    
    The target is a 30x30 flattened sequence (900 tokens).
    -100 marks padding positions (IGNORE_LABEL).
    
    Args:
        target: [900] target tokens (-100 for ignored positions, 4-13 for colors)
        
    Returns:
        [H, W] grid with values 0-9 (cropped to actual content)
    """
    # Reshape to 30x30
    target_2d = target.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    
    # Find bounding box of valid content (where target != -100 and target is a color)
    valid_mask = (target_2d != -100) & (target_2d >= COLOR_OFFSET)
    
    if not valid_mask.any():
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Get bounding box
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Extract target grid for valid region
    target_crop = target_2d[rmin:rmax+1, cmin:cmax+1].copy()
    
    # Convert tokens to colors (0-9)
    grid = np.zeros_like(target_crop, dtype=np.uint8)
    for r in range(target_crop.shape[0]):
        for c in range(target_crop.shape[1]):
            token = target_crop[r, c]
            if token >= COLOR_OFFSET:
                grid[r, c] = token - COLOR_OFFSET
            elif token == -100:
                grid[r, c] = 0  # Shouldn't happen in cropped region
            else:
                grid[r, c] = 0  # EOS or special token
    
    return np.clip(grid, 0, 9)


def sequence_to_grid(seq: np.ndarray) -> np.ndarray:
    """
    Convert a sequence to a grid using TRM-style aligned format.
    
    The sequence is a 30x30 flattened grid (900 tokens).
    
    Args:
        seq: [900] sequence with PAD=0, EOS=1, colors 4-13
        
    Returns:
        [H, W] grid with values 0-9 (cropped to actual content)
    """
    # Reshape to 30x30
    seq_2d = seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    
    # Find bounding box of content (where seq is a color token, not PAD or EOS)
    content_mask = seq_2d >= COLOR_OFFSET
    
    if not content_mask.any():
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Get bounding box
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Extract grid for content region
    seq_crop = seq_2d[rmin:rmax+1, cmin:cmax+1].copy()
    
    # Convert tokens to colors (0-9)
    grid = np.zeros_like(seq_crop, dtype=np.uint8)
    for r in range(seq_crop.shape[0]):
        for c in range(seq_crop.shape[1]):
            token = seq_crop[r, c]
            if token >= COLOR_OFFSET:
                grid[r, c] = token - COLOR_OFFSET
            else:
                grid[r, c] = 0
    
    return np.clip(grid, 0, 9)


def create_sample_image(
    demo_inputs: torch.Tensor,   # [num_demos, S]
    demo_outputs: torch.Tensor,  # [num_demos, S]
    test_input: torch.Tensor,    # [S]
    test_output: torch.Tensor,   # [S] (target)
    prediction: torch.Tensor,    # [S] (model output)
    max_demos: int = 2,
    cell_size: int = 8,
) -> np.ndarray:
    """
    Create a combined image showing demos, test input, target, and prediction.
    
    Layout:
    +-------------------+-------------------+
    | Demo 1 In         | Demo 1 Out        |
    +-------------------+-------------------+
    | Demo 2 In         | Demo 2 Out        |
    +-------------------+-------------------+
    | Test Input        | Target | Pred     |
    +-------------------+-------------------+
    
    Returns:
        RGB image [H, W, 3] suitable for TensorBoard
    """
    # Convert sequences to grids
    num_demos = min(demo_inputs.shape[0], max_demos)
    
    demo_in_grids = [sequence_to_grid(demo_inputs[d].cpu().numpy()) for d in range(num_demos)]
    demo_out_grids = [sequence_to_grid(demo_outputs[d].cpu().numpy()) for d in range(num_demos)]
    test_in_grid = sequence_to_grid(test_input.cpu().numpy())
    # Target uses -100 for EOS/PAD, need special handling
    target_grid = target_to_grid(test_output.cpu().numpy())
    # Use target's structure (EOS positions) to reshape prediction
    pred_grid = prediction_to_grid(prediction.cpu().numpy(), test_output.cpu().numpy())
    
    # Find max dimensions for consistent sizing
    all_grids = demo_in_grids + demo_out_grids + [test_in_grid, target_grid, pred_grid]
    max_h = max(g.shape[0] for g in all_grids)
    max_w = max(g.shape[1] for g in all_grids)
    
    # Pad all grids to same size
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
    
    # Create separator (white line)
    sep_h = 2
    sep_w = max_w * cell_size
    h_sep = np.ones((sep_h, sep_w * 2 + sep_h, 3), dtype=np.uint8) * 255
    v_sep = np.ones((max_h * cell_size, sep_h, 3), dtype=np.uint8) * 255
    
    rows = []
    
    # Demo rows
    for i in range(num_demos):
        row = np.concatenate([demo_in_imgs[i], v_sep, demo_out_imgs[i]], axis=1)
        rows.append(row)
        rows.append(h_sep)
    
    # Test row (input | target | pred) - need 3 columns
    # Make separator for 3 columns
    v_sep_small = np.ones((max_h * cell_size, sep_h, 3), dtype=np.uint8) * 255
    test_row = np.concatenate([test_in_img, v_sep_small, target_img, v_sep_small, pred_img], axis=1)
    
    # Pad test row to match demo row width
    demo_row_width = max_w * cell_size * 2 + sep_h
    test_row_width = test_row.shape[1]
    if test_row_width < demo_row_width:
        pad = np.zeros((test_row.shape[0], demo_row_width - test_row_width, 3), dtype=np.uint8)
        test_row = np.concatenate([test_row, pad], axis=1)
    
    # Adjust h_sep width
    h_sep_test = np.ones((sep_h, test_row.shape[1], 3), dtype=np.uint8) * 255
    
    # For demos, also adjust to match test row width if needed
    for i in range(len(rows)):
        if rows[i].shape[1] < test_row.shape[1]:
            pad = np.zeros((rows[i].shape[0], test_row.shape[1] - rows[i].shape[1], 3), dtype=np.uint8)
            rows[i] = np.concatenate([rows[i], pad], axis=1)
        elif rows[i].shape[1] > test_row.shape[1]:
            test_row_padded = np.zeros((test_row.shape[0], rows[i].shape[1], 3), dtype=np.uint8)
            test_row_padded[:, :test_row.shape[1]] = test_row
            test_row = test_row_padded
    
    rows.append(test_row)
    
    # Stack all rows
    # Ensure all rows have same width
    max_row_width = max(r.shape[1] for r in rows)
    final_rows = []
    for r in rows:
        if r.shape[1] < max_row_width:
            pad = np.zeros((r.shape[0], max_row_width - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        final_rows.append(r)
    
    combined = np.concatenate(final_rows, axis=0)
    
    return combined


def grid_to_ascii(grid, max_width: int = 15, max_height: int = 10) -> list:
    """
    Convert a grid to colored ASCII art lines.
    
    Args:
        grid: numpy array [H, W] with values 0-9
        max_width: max columns to display
        max_height: max rows to display
        
    Returns:
        List of strings (one per row)
    """
    import numpy as np
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    h, w = grid.shape
    h = min(h, max_height)
    w = min(w, max_width)
    
    lines = []
    for row in range(h):
        line = ""
        for col in range(w):
            color = int(grid[row, col])
            color = max(0, min(color, 9))  # Clamp to valid range
            line += f"{ARC_COLORS[color]}  {RESET}"
        lines.append(line)
    
    # Add size indicator
    lines.append(f"({grid.shape[0]}×{grid.shape[1]})")
    
    return lines


def visualize_sample(
    demo_inputs: torch.Tensor,   # [num_demos, S]
    demo_outputs: torch.Tensor,  # [num_demos, S]
    test_input: torch.Tensor,    # [S]
    test_output: torch.Tensor,   # [S] (target)
    prediction: torch.Tensor,    # [S] (model output)
    sample_idx: int = 0,
    step: int = 0,
) -> str:
    """
    Visualize a sample with demos, test, target, and prediction.
    
    Returns a formatted string with side-by-side grid comparisons.
    """
    import numpy as np
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"[Step {step}] Sample #{sample_idx} Visualization")
    lines.append(f"{'='*70}")
    
    # Convert sequences to grids
    num_demos = demo_inputs.shape[0]
    
    # Demo pairs
    for d in range(min(num_demos, 2)):  # Show max 2 demos
        demo_in_grid = sequence_to_grid(demo_inputs[d].cpu().numpy())
        demo_out_grid = sequence_to_grid(demo_outputs[d].cpu().numpy())
        
        in_lines = grid_to_ascii(demo_in_grid, max_width=10, max_height=8)
        out_lines = grid_to_ascii(demo_out_grid, max_width=10, max_height=8)
        
        # Pad to same height
        max_h = max(len(in_lines), len(out_lines))
        while len(in_lines) < max_h:
            in_lines.insert(-1, " " * 20)
        while len(out_lines) < max_h:
            out_lines.insert(-1, " " * 20)
        
        lines.append(f"\n  Demo {d+1}:")
        lines.append(f"  {'Input':^22} → {'Output':^22}")
        for i in range(max_h):
            lines.append(f"  {in_lines[i]:22}   {out_lines[i]:22}")
    
    # Test input, target, prediction
    test_in_grid = sequence_to_grid(test_input.cpu().numpy())
    # Target uses -100 for EOS/PAD, need special handling
    target_grid = target_to_grid(test_output.cpu().numpy())
    # Use target's structure (EOS positions) to reshape prediction
    pred_grid = prediction_to_grid(prediction.cpu().numpy(), test_output.cpu().numpy())
    
    test_lines = grid_to_ascii(test_in_grid, max_width=10, max_height=8)
    tgt_lines = grid_to_ascii(target_grid, max_width=10, max_height=8)
    pred_lines = grid_to_ascii(pred_grid, max_width=10, max_height=8)
    
    # Pad to same height
    max_h = max(len(test_lines), len(tgt_lines), len(pred_lines))
    while len(test_lines) < max_h:
        test_lines.insert(-1, " " * 20)
    while len(tgt_lines) < max_h:
        tgt_lines.insert(-1, " " * 20)
    while len(pred_lines) < max_h:
        pred_lines.insert(-1, " " * 20)
    
    lines.append(f"\n  Test:")
    lines.append(f"  {'Input':^22}   {'Target':^22}   {'Prediction':^22}")
    for i in range(max_h):
        lines.append(f"  {test_lines[i]:22}   {tgt_lines[i]:22}   {pred_lines[i]:22}")
    
    # Accuracy for this sample
    valid_mask = (test_output != -100)
    if valid_mask.any():
        correct = (prediction[valid_mask] == test_output[valid_mask]).float().mean().item()
        lines.append(f"\n  Accuracy: {correct*100:.1f}%")
    
    lines.append(f"{'='*70}\n")
    
    return '\n'.join(lines)


def analyze_predictions(logits: torch.Tensor, test_output: torch.Tensor, step: int):
    """
    Analyze prediction distribution to diagnose bias toward certain colors.
    Only analyzes valid (non-padded) positions.
    """
    IGNORE_LABEL = -100
    preds = logits.argmax(dim=-1)  # [B, H, W]
    
    # Mask for valid positions (not padded)
    valid_mask = (test_output != IGNORE_LABEL)
    
    if not valid_mask.any():
        print(f"\n[Step {step}] No valid positions to analyze")
        return None, None
    
    # Only count valid positions
    valid_preds = preds[valid_mask]
    valid_targets = test_output[valid_mask]
    
    # Count predicted colors (only in valid region)
    pred_counts = torch.bincount(valid_preds.flatten(), minlength=10).float()
    pred_dist = pred_counts / pred_counts.sum()
    
    # Count target colors (only in valid region)
    tgt_counts = torch.bincount(valid_targets.flatten(), minlength=10).float()
    tgt_dist = tgt_counts / tgt_counts.sum()
    
    # Softmax probabilities (model confidence) - only valid positions
    probs = F.softmax(logits, dim=-1)  # [B, H, W, C]
    max_probs = probs.max(dim=-1).values  # [B, H, W]
    valid_max_probs = max_probs[valid_mask]
    
    print(f"\n[Step {step}] Prediction Analysis:")
    print("-" * 50)
    print(f"  Color distribution (pred vs target):")
    for c in range(10):
        marker = "⚠️" if abs(pred_dist[c] - tgt_dist[c]) > 0.1 else ""
        print(f"    Color {c}: pred={pred_dist[c]:.3f} vs tgt={tgt_dist[c]:.3f} {marker}")
    
    print(f"\n  Model confidence: mean={valid_max_probs.mean():.3f}, min={valid_max_probs.min():.3f}, max={valid_max_probs.max():.3f}")
    
    # Check if model is just predicting black
    black_pred_ratio = pred_dist[0].item()
    if black_pred_ratio > 0.9:
        print(f"  ⚠️ WARNING: {black_pred_ratio*100:.1f}% predictions are black/0!")
        print(f"     Logit stats for color 0: mean={logits[..., 0].mean():.3f}, max={logits[..., 0].max():.3f}")
        print(f"     Logit stats for color 1: mean={logits[..., 1].mean():.3f}, max={logits[..., 1].max():.3f}")
    
    return pred_dist, tgt_dist

# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: DLSMN_ARC,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    global_step: int = 0,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 10,
    grad_accum_steps: int = 1,
    use_adam_atan2: bool = True,
) -> Tuple[float, float, float, int]:
    """
    Training epoch with simplified loss.
    """
    model.train()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_tasks = 0
    total_tasks = 0
    cell_acc = 0.0
    task_acc = 0.0
    
    # IGNORE_LABEL for sequence format
    IGNORE_LABEL = -100

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        demo_inputs = batch["demo_inputs"].to(device)   # [B, 3, S]
        demo_outputs = batch["demo_outputs"].to(device) # [B, 3, S]
        test_input = batch["test_input"].to(device)     # [B, S]
        test_output = batch["test_output"].to(device)   # [B, S] with IGNORE_LABEL
        
        # DEBUG: Check batch shapes
        if global_step == 0:
            print(f"\n[DEBUG] Train Epoch Batch 0:")
            print(f"  test_output shape: {test_output.shape}")
            print(f"  test_input shape: {test_input.shape}")
            print(f"  demo_inputs shape: {demo_inputs.shape}")

        # Only zero grad at accumulation boundaries
        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        # Forward pass with training config
        # Returns: logits [B, S, vocab_size], cache, aux_info
        logits, cache, aux_info = model(
            demo_inputs, demo_outputs, test_input,
            config=config,
            step=global_step,
            return_aux=True,
        )

        # Compute Loss (Refactored for sequences)
        loss, metrics = compute_total_loss(
            model, logits, test_output,
            aux_info, config, global_step, device
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        # Only step optimizer at accumulation boundaries
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Check for NaN/Inf gradients and skip step if found
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        if global_step % 50 == 0:
                            print(f"WARNING: NaN/Inf gradient in {name}")
                        break
            
            if has_nan_grad:
                print(f"[Step {global_step}] Skipping optimizer step due to NaN/Inf gradients")
                optimizer.zero_grad()
            else:
                # === GRADIENT MONITORING (Before Clipping) ===
                # Log gradients every 50 steps, verbose every 200 steps
                if global_step % 50 == 0:
                    verbose = (global_step % 50 == 0)
                    monitor_gradients(model, writer, global_step, verbose=verbose)
                    
                    # Log compute statistics
                    num_passes = aux_info.get('num_passes', 1)
                    max_passes = aux_info.get('max_passes', config.max_passes)
                    avg_steps = aux_info.get('avg_layer_steps', 1.0)
                    max_steps = aux_info.get('max_layer_steps', config.max_recurrent_steps)
                    total_steps = aux_info.get('total_layer_steps', 0)
                    
                    print(f"\n[Step {global_step}] Compute Statistics:")
                    print("-" * 50)
                    print(f"  Model passes: {num_passes}/{max_passes} ({num_passes/max_passes:.0%} utilization)")
                    print(f"  Avg layer steps: {avg_steps:.2f}/{max_steps} ({avg_steps/max_steps:.0%} utilization)")
                    print(f"  Total layer steps: {total_steps}")
                    
                    # Also log gate values to diagnose gradient flow
                    if 'read_gates' in aux_info and aux_info['read_gates']:
                        print(f"\n[Step {global_step}] Memory Gate Values:")
                        print("-" * 50)
                        for layer_idx, rg in enumerate(aux_info['read_gates']):
                            rg_mean = rg.mean().item()
                            rg_min = rg.min().item()
                            rg_max = rg.max().item()
                            wg = aux_info['write_gates'][layer_idx] if 'write_gates' in aux_info else None
                            wg_mean = wg.mean().item() if wg is not None else 0
                            print(f"  Layer {layer_idx}: read={rg_mean:.3f} (min={rg_min:.3f}, max={rg_max:.3f}), write={wg_mean:.3f}")
                
                # Analyze predictions every 200 steps
                if global_step % 200 == 0:
                    analyze_predictions(logits.detach(), test_output, global_step)
                
                # === GRADIENT HANDLING ===
                # AdamAtan2 is scale-invariant - no gradient scaling/clipping needed.
                # For legacy AdamW support, we keep the conditional.
                if not use_adam_atan2:
                    # Legacy: Scale and clip for AdamW stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

        # === CRITICAL: Memory cleanup to prevent graph accumulation ===
        loss_val = loss.detach().item()
        total_loss += loss_val
        global_step += 1

        # Metrics: Only count non-padded positions (where target != IGNORE_LABEL)
        # logits: [B, S, vocab_size], preds: [B, S]
        preds = logits.detach().argmax(dim=-1)  # [B, S]
        
        for i in range(preds.shape[0]):
            # Mask: True where we should evaluate (not padded)
            valid_mask = (test_output[i] != IGNORE_LABEL)
            
            if valid_mask.any():
                correct_cells += (preds[i][valid_mask] == test_output[i][valid_mask]).sum().item()
                total_cells += valid_mask.sum().item()
                
                if (preds[i][valid_mask] == test_output[i][valid_mask]).all():
                    correct_tasks += 1
            total_tasks += 1

        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)

        pass_mode_val = aux_info.get('pass_mode', 'adaptive')
        temperature_val = aux_info.get('temperature', 1.0)

        if writer is not None and batch_idx % log_interval == 0:
            # === Core Losses ===
            writer.add_scalar('Loss/total', metrics['loss_total'], global_step)
            writer.add_scalar('Loss/task', metrics['loss_task'], global_step)
            if metrics.get('loss_compute', 0) > 0:
                writer.add_scalar('Loss/compute', metrics['loss_compute'], global_step)
            if metrics.get('loss_diversity', 0) > 0:
                writer.add_scalar('Loss/diversity', metrics['loss_diversity'], global_step)
            if metrics.get('loss_gate_polar', 0) > 0:
                writer.add_scalar('Loss/gate_polarization', metrics['loss_gate_polar'], global_step)
            if metrics.get('loss_gate_sparsity', 0) > 0:
                writer.add_scalar('Loss/gate_sparsity', metrics['loss_gate_sparsity'], global_step)

            # === Component Losses ===
            if metrics.get('loss_q_head', 0) > 0:
                writer.add_scalar('Loss/q_head', metrics['loss_q_head'], global_step)
            if metrics.get('loss_step_efficiency', 0) > 0:
                writer.add_scalar('Loss/step_efficiency', metrics['loss_step_efficiency'], global_step)

            # === Metrics ===
            writer.add_scalar('Metrics/cell_accuracy', cell_acc, global_step)
            writer.add_scalar('Metrics/task_accuracy', task_acc, global_step)

            # === Pass/Step Statistics (key for understanding model behavior) ===
            num_passes = aux_info.get('num_passes', 1)
            max_passes = aux_info.get('max_passes', config.max_passes)
            avg_layer_steps = aux_info.get('avg_layer_steps', 1.0)
            max_layer_steps = aux_info.get('max_layer_steps', config.max_recurrent_steps)
            total_layer_steps = aux_info.get('total_layer_steps', 0)
            
            # Actual passes used
            writer.add_scalar('Compute/model_passes', num_passes, global_step)
            writer.add_scalar('Compute/pass_utilization', num_passes / max_passes, global_step)
            
            # Layer recurrence statistics
            writer.add_scalar('Compute/avg_layer_steps', avg_layer_steps, global_step)
            writer.add_scalar('Compute/layer_step_utilization', avg_layer_steps / max_layer_steps, global_step)
            writer.add_scalar('Compute/total_layer_steps', total_layer_steps, global_step)
            
            # Total compute: passes × layers × avg_steps_per_layer
            num_layers = len(model.layers)
            total_compute = num_passes * num_layers * avg_layer_steps
            max_compute = max_passes * num_layers * max_layer_steps
            writer.add_scalar('Compute/total_ops', total_compute, global_step)
            writer.add_scalar('Compute/utilization', total_compute / max_compute, global_step)

            # === Training Dynamics ===
            writer.add_scalar('Training/temperature', temperature_val, global_step)
            writer.add_scalar('Training/lr', optimizer.param_groups[0]['lr'], global_step)
            
            # Log curriculum learning progress
            if config.use_curriculum:
                writer.add_scalar('Training/curriculum_passes', config.get_curriculum_passes(global_step), global_step)
                writer.add_scalar('Training/curriculum_recurrence', config.get_curriculum_recurrence(global_step), global_step)

            # === Memory Gate Statistics (aggregated) ===
            if 'read_gates' in aux_info and aux_info['read_gates']:
                all_read = torch.cat([rg.view(-1) for rg in aux_info['read_gates']])
                writer.add_scalar('Gates/read_mean', all_read.mean().item(), global_step)
                writer.add_scalar('Gates/read_std', all_read.std().item(), global_step)
            if 'write_gates' in aux_info and aux_info['write_gates']:
                all_write = torch.cat([wg.view(-1) for wg in aux_info['write_gates']])
                writer.add_scalar('Gates/write_mean', all_write.mean().item(), global_step)
                writer.add_scalar('Gates/write_std', all_write.std().item(), global_step)

        # === Sample Visualization (every 50 steps) - OUTSIDE log_interval check ===
        if writer is not None and global_step % 50 == 0:
            # Visualize first sample in batch
            viz_str = visualize_sample(
                demo_inputs=demo_inputs[0],      # [num_demos, S]
                demo_outputs=demo_outputs[0],    # [num_demos, S]
                test_input=test_input[0],        # [S]
                test_output=test_output[0],      # [S]
                prediction=preds[0],             # [S]
                sample_idx=0,
                step=global_step,
            )
            print(viz_str)
            
            # === TensorBoard Image Logging ===
            try:
                sample_img = create_sample_image(
                    demo_inputs=demo_inputs[0],      # [num_demos, S]
                    demo_outputs=demo_outputs[0],    # [num_demos, S]
                    test_input=test_input[0],        # [S]
                    test_output=test_output[0],      # [S]
                    prediction=preds[0],             # [S]
                    max_demos=2,
                    cell_size=8,
                )
                # TensorBoard expects [C, H, W] with uint8 [0-255] or float [0-1]
                # sample_img is [H, W, C] uint8 - transpose and normalize to float
                sample_img_chw = np.transpose(sample_img, (2, 0, 1)).astype(np.float32) / 255.0
                writer.add_image('Samples/prediction', sample_img_chw, global_step)
                writer.flush()  # Force write to disk
                print(f"[INFO] Logged image to TensorBoard at step {global_step}")
            except Exception as e:
                import traceback
                print(f"[WARN] Failed to create TensorBoard image: {e}")
                traceback.print_exc()

        if writer is not None and batch_idx % log_interval == 0:
            # Compute statistics for progress bar
            num_passes = aux_info.get('num_passes', 1)
            avg_steps = aux_info.get('avg_layer_steps', 1.0)
            compute_util = aux_info.get('total_layer_steps', 0) / max(1, config.max_passes * len(model.layers) * config.max_recurrent_steps)
            
            pbar.set_postfix({
                'loss': f'{loss_val:.3f}',
                'cell': f'{cell_acc:.3f}',
                'task': f'{task_acc:.3f}',
                'τ': f'{temperature_val:.2f}',
                'P': f'{num_passes}',  # Model passes
                'S': f'{avg_steps:.1f}',  # Avg layer steps
                'U': f'{compute_util:.0%}',  # Compute utilization
            })

        # === CRITICAL: Memory cleanup after each batch ===
        # Clear all tensor references
        del loss, logits, cache

        # Clear aux_info lists
        for key, val in aux_info.items():
            if isinstance(val, list):
                val.clear()
        aux_info.clear()

        # Force cache clear every 10 batches
        if batch_idx % 10 == 0:
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()

    return total_loss / len(dataloader), cell_acc, task_acc, global_step


@torch.no_grad()
def evaluate(
    model: DLSMN_ARC,
    dataloader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    visualize_samples: int = 2,  # Number of samples to visualize
    global_step: int = 0,
    writer: SummaryWriter = None,  # TensorBoard writer for image logging
) -> Tuple[float, float]:
    model.eval()
    correct_tasks = 0
    total_tasks = 0
    correct_cells = 0
    total_cells = 0
    cell_acc = 0.0
    task_acc = 0.0
    samples_visualized = 0

    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for batch_idx, batch in enumerate(pbar):
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)

        # Use hard routing during evaluation
        eval_config = TrainingConfig(
            tau_min=0.1,
            tau_start=0.1,  # Hard routing
            max_passes=config.max_passes,
            max_recurrent_steps=config.max_recurrent_steps,
        )

        logits, _, _ = model(
            demo_inputs, demo_outputs, test_input,
            config=eval_config,
            step=100000,  # High step for minimum temperature
            return_aux=False,
        )

        preds = logits.argmax(dim=-1)

        # Only count non-padded positions (where target != -100)
        IGNORE_LABEL = -100
        for i in range(preds.shape[0]):
            valid_mask = (test_output[i] != IGNORE_LABEL)
            
            if valid_mask.any():
                correct_cells += (preds[i][valid_mask] == test_output[i][valid_mask]).sum().item()
                total_cells += valid_mask.sum().item()
                
                task_correct = (preds[i][valid_mask] == test_output[i][valid_mask]).all()
                if task_correct:
                    correct_tasks += 1
                    
                # Visualize some samples (both correct and incorrect)
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
                    )
                    print(viz_str)
                    
                    # === TensorBoard Image Logging for Eval ===
                    if writer is not None:
                        try:
                            sample_img = create_sample_image(
                                demo_inputs=demo_inputs[i],
                                demo_outputs=demo_outputs[i],
                                test_input=test_input[i],
                                test_output=test_output[i],
                                prediction=preds[i],
                                max_demos=2,
                                cell_size=8,
                            )
                            # TensorBoard expects [C, H, W] with float [0-1]
                            sample_img_chw = np.transpose(sample_img, (2, 0, 1)).astype(np.float32) / 255.0
                            tag = f'Eval/sample_{samples_visualized}_{marker}'
                            writer.add_image(tag, sample_img_chw, global_step)
                            writer.flush()
                        except Exception as e:
                            import traceback
                            print(f"[WARN] Failed to create eval TensorBoard image: {e}")
                            traceback.print_exc()
                    
                    samples_visualized += 1
                    
            total_tasks += 1

        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        pbar.set_postfix({'cell': f'{cell_acc:.3f}', 'task': f'{task_acc:.3f}'})

    return cell_acc, task_acc


def print_model_summary(model: torch.nn.Module):
    print("\n" + "="*50)
    print(f"Model Summary: {model.__class__.__name__}")
    print("="*50)
    
    total_params = 0
    trainable_params = 0
    
    # Track breakdown
    breakdown = {
        "Embeddings": 0,
        "Layers": 0,
        "Heads/Projections": 0,
        "Other": 0
    }
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            
        if any(x in name for x in ["embed", "embedding"]):
            breakdown["Embeddings"] += num_params
        elif "layers" in name:
            breakdown["Layers"] += num_params
        elif any(x in name for x in ["head", "proj", "net", "predictor", "gate"]):
            breakdown["Heads/Projections"] += num_params
        else:
            breakdown["Other"] += num_params
            
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("-" * 50)
    print("Breakdown:")
    for category, count in breakdown.items():
        percent = (count / total_params) * 100 if total_params > 0 else 0
        print(f"  {category:<20}: {count:,} ({percent:.1f}%)")
    print("="*50 + "\n")

def main():
    # Check if data exists
    data_dir = Path("./ARC-AGI-2/data")
    if not data_dir.exists():
        print("Downloading ARC-AGI-2 dataset...")
        os.system("git clone https://github.com/arcprize/ARC-AGI-2.git")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    import sys
    preset_name = sys.argv[1] if len(sys.argv) > 1 else "fast"
    features = FEATURE_PRESETS.get(preset_name, FEATURE_PRESETS["fast"])
    print(f"\nUsing feature preset: '{preset_name}'")
    print(f"Features: {features.describe()}")

    # Configure model params
    if preset_name == "fast_full":
        max_grid_size = 30
        d_model = 64
        d_cache = 48
        num_layers = 4
        num_slots = 32
        num_heads = 2
        batch_size = 8
        max_passes = 4
        max_recurrent_steps = 4
    elif preset_name == "runpod":
        # Optimized for ~44GB GPU with LinearAttention (O(S) memory)
        # Increased capacity since we no longer have O(S²) attention memory
        max_grid_size = 10
        d_model = 128      # Restored - LinearAttention makes this feasible
        d_cache = 64       # Restored
        num_layers = 4
        num_slots = 16
        num_heads = 4
        batch_size = 16    # Can increase with linear attention
        max_passes = 4
        max_recurrent_steps = 3
    elif preset_name == "full":
        max_grid_size = 30
        d_model = 128
        d_cache = 64
        num_layers = 4
        num_slots = 16
        num_heads = 4
        batch_size = 2  # Reduced for GPU memory
        max_passes = 6
        max_recurrent_steps = 4
    else:
        max_grid_size = 15
        d_model = 64
        d_cache = 48
        num_layers = 2
        num_slots = 8
        num_heads = 2
        batch_size = 8
        max_passes = 2 if features.use_multi_pass else 1
        max_recurrent_steps = 1

    # Training dataset WITH augmentation (dihedral + color permutation)
    train_dataset = ARCDataset(str(data_dir), split="training", max_grid_size=max_grid_size, augment=True)
    # Evaluation dataset WITHOUT augmentation (we want consistent results)
    eval_dataset = ARCDataset(str(data_dir), split="evaluation", max_grid_size=max_grid_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize training configuration with appropriate settings for each preset
    if preset_name == "runpod":
        # For runpod preset, adjust loss weights to improve task completion
        config = TrainingConfig(
            tau_start=1.0,
            tau_min=0.2,  # Slightly higher minimum temperature for better exploration
            anneal_rate=0.0005,  # Adjusted annealing rate
            max_passes=max_passes,
            max_recurrent_steps=max_recurrent_steps,
            lambda_diversity=0.005,      # Reduced to focus more on primary task
            lambda_q_head=0.05,          # Reduced to focus on primary task
            lambda_step_efficiency=0.02, # Reduced to allow more exploration
            features=features,
        )
    else:
        config = TrainingConfig(
            tau_start=1.0,
            tau_min=0.1,
            max_passes=max_passes,
            max_recurrent_steps=max_recurrent_steps,
            features=features,
        )

    model = DLSMN_ARC(
        vocab_size=VOCAB_SIZE,  # PAD=0, EOS=1, INPUT=2, OUTPUT=3, colors 0-9 → 4-13
        d_model=d_model,
        d_cache=d_cache,
        num_layers=num_layers,
        num_slots=num_slots,
        num_heads=num_heads,
        max_seq_len=900,  # max 30x30 grid = 900 tokens
        max_recurrent_steps=max_recurrent_steps,
        max_passes=max_passes,
        dropout=0,
    ).to(device)

    print_model_summary(model)

    # Print curriculum schedule if enabled
    if config.use_curriculum:
        print(f"\n📚 Curriculum Learning Schedule:")
        print(f"  Warmup: {config.curriculum_warmup_steps} steps (1 pass)")
        print(f"  Increase every: {config.curriculum_increase_every} steps")
        print(f"  Target max passes: {config.max_passes}")
        for p in range(1, config.max_passes + 1):
            if p == 1:
                print(f"    Pass {p}: steps 0-{config.curriculum_warmup_steps}")
            else:
                start = config.curriculum_warmup_steps + (p - 2) * config.curriculum_increase_every
                end = config.curriculum_warmup_steps + (p - 1) * config.curriculum_increase_every
                if p == config.max_passes:
                    print(f"    Pass {p}: steps {start}+ (max)")
                else:
                    print(f"    Pass {p}: steps {start}-{end}")

    # Higher LR for runpod with cosine schedule
    lr = 3e-4 if preset_name == "runpod" else 1e-4
    
    # Use AdamAtan2 optimizer (scale-invariant, no need for gradient clipping)
    # Based on https://arxiv.org/abs/2407.05872
    use_adam_atan2 = True  # Set to False to use standard AdamW
    
    if use_adam_atan2:
        optimizer = AdamAtan2(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),  # TinyRM uses (0.9, 0.95)
            weight_decay=0.1,   # TinyRM uses 0.1
            a=1.27,             # Default scaling factor
            b=1.0,
        )
        print(f"  Using AdamAtan2 optimizer (scale-invariant)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        print(f"  Using AdamW optimizer")

    # Gradient accumulation - effective batch = batch_size * grad_accum_steps
    grad_accum_steps = 2 if preset_name in ["full", "runpod"] else 1

    log_dir = Path("logs") / f"dlsmn_{preset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # DEBUG: Check dataset shapes
    print(f"\n[DEBUG] Dataset Check:")
    print(f"  MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    print(f"  Batch Size: {batch_size}")
    sample0 = train_dataset[0]
    print(f"  Sample 0 test_output shape: {sample0['test_output'].shape}")
    print(f"  Sample 0 valid targets: {(sample0['test_output'] != -100).sum().item()}")

    best_task_acc = 0
    global_step = 0

    for epoch in range(50):
        train_loss, train_cell_acc, train_task_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, config, global_step,
            writer=writer, log_interval=10, grad_accum_steps=grad_accum_steps,
            use_adam_atan2=use_adam_atan2,
        )

        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/train_cell_acc', train_cell_acc, epoch)

        if (epoch + 1) % 10 == 0:
            eval_cell_acc, eval_task_acc = evaluate(
                model, eval_loader, device, config,
                visualize_samples=2,  # Show 2 samples during eval
                global_step=global_step,
                writer=writer,  # Pass writer for TensorBoard images
            )
            if eval_task_acc > best_task_acc:
                best_task_acc = eval_task_acc

            writer.add_scalar('Epoch/eval_cell_acc', eval_cell_acc, epoch)
            writer.add_scalar('Epoch/eval_task_acc', eval_task_acc, epoch)

            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Eval Task: {eval_task_acc:.3f}")
        else:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train Task: {train_task_acc:.3f}")

    writer.close()

if __name__ == "__main__":
    main()
