"""
Training Script for CNN + Cache Model on Mini-ARC
=================================================

This trains a 1D CNN model augmented with the DLSMN cache system.
The goal is to show that cache can give CNNs transformer-like
global reasoning capabilities.

Key differences from transformer-based train_miniarc.py:
- Uses 1D convolutions instead of self-attention for local processing
- Cache provides global context (instead of O(N²) attention)
- Potentially faster and more parameter-efficient

Usage:
    python train_cnn_miniarc.py [preset]
    
Presets:
    debug   - Minimal for testing
    fast    - Quick training iteration
    full    - Best performance
    compare - Same params as transformer for fair comparison
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
from typing import Tuple, Optional, Dict

# Import components
from components import (
    FeatureFlags,
    TrainingConfig,
    FEATURE_PRESETS,
    EMA,
    AdamAtan2,
)
from components.cnn_cache_model import CNNCacheModel, create_cnn_cache_model
from components.logging import MetricsLogger
from components.miniarc_dataset import (
    MiniARCDataset,
    VOCAB_SIZE, MAX_SEQ_LEN, PAD_TOKEN, COLOR_OFFSET, MAX_GRID_SIZE,
    sequence_to_grid, target_to_grid, download_miniarc,
)


# ============================================================================
# Visualization (same as train_miniarc.py)
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


def grid_to_ascii(grid, max_width: int = 10, max_height: int = 10) -> list:
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
    pred_2d = prediction.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    target_2d = target.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    grid = np.zeros_like(pred_2d, dtype=np.uint8)
    valid_mask = (target_2d != -100) & (pred_2d >= COLOR_OFFSET)
    grid[valid_mask] = np.clip(pred_2d[valid_mask] - COLOR_OFFSET, 0, 9)
    return grid


def grid_to_rgb(grid: np.ndarray, cell_size: int = 10) -> np.ndarray:
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
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"[Step {step}] CNN+Cache Mini-ARC Sample #{sample_idx}")
    
    if aux:
        M = aux.get('passes_run', 1)
        K = aux.get('layer_iters', 1)
        lines.append(f"  Passes (M): {M} | Layer Iters (K): {K}")
        
        read_gates = aux.get('read_gates', [])
        write_gates = aux.get('write_gates', [])
        if read_gates:
            # Gates are tensors, convert to scalar for display
            avg_read = sum(g.mean().item() for g in read_gates) / len(read_gates)
            lines.append(f"  Avg Read Gate: {avg_read:.3f}")
        if write_gates:
            avg_write = sum(g.mean().item() for g in write_gates) / len(write_gates)
            lines.append(f"  Avg Write Gate: {avg_write:.3f}")
    
    lines.append(f"{'='*60}")
    
    # Demo pairs
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
# Loss Computation (Simple version for CNN + Cache)
# ============================================================================

IGNORE_LABEL = -100

def compute_cnn_loss(
    model: CNNCacheModel,
    logits: torch.Tensor,
    targets: torch.Tensor,
    aux: Dict,
    config: TrainingConfig,
    step: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """
    Simple loss for CNN + Cache model.
    
    Components:
    1. Task loss: Cross-entropy with deep supervision (train on all passes)
    2. Gate usage loss: Encourage gates to be > 0 (prevent dead cache)
    
    This is intentionally simpler than the transformer loss to show
    the cache mechanism works without complex halting machinery.
    """
    # === 1. TASK LOSS with deep supervision ===
    pass_logits = aux.get('pass_logits', [])
    if not pass_logits:
        pass_logits = [logits.detach()]
    
    # Weight later passes more (they should be better)
    num_passes = len(pass_logits)
    pass_weights = [0.5 ** (num_passes - 1 - i) for i in range(num_passes)]
    total_weight = sum(pass_weights)
    pass_weights = [w / total_weight for w in pass_weights]
    
    task_loss = torch.tensor(0.0, device=device)
    
    for pass_idx, p_logits in enumerate(pass_logits):
        # Ensure we use the version with gradients for the final pass
        if pass_idx == len(pass_logits) - 1:
            p_logits = logits  # Use original logits with gradients
        
        # Flatten for cross entropy
        B, S, V = p_logits.shape
        flat_logits = p_logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)
        
        # Cross entropy (ignores -100)
        ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE_LABEL)
        task_loss = task_loss + pass_weights[pass_idx] * ce
    
    # === 2. GATE USAGE LOSS ===
    # Encourage gates to be used (prevent all-zero gates)
    # This helps the cache actually store and retrieve information
    gate_loss = torch.tensor(0.0, device=device)
    avg_read = torch.tensor(0.0, device=device)
    avg_write = torch.tensor(0.0, device=device)
    
    read_gates = aux.get('read_gates', [])
    write_gates = aux.get('write_gates', [])
    
    if read_gates:
        # Average gate activation across all passes
        avg_read = torch.stack([g.mean() for g in read_gates]).mean()
        avg_write = torch.stack([g.mean() for g in write_gates]).mean() if write_gates else torch.tensor(0.0, device=device)
        
        # Encourage gates to be around 0.3-0.5 (not too sparse, not too dense)
        # Loss = 0 when gate ≈ 0.4, increases as it moves away
        target_gate = 0.4
        read_gate_loss = (avg_read - target_gate).abs()
        write_gate_loss = (avg_write - target_gate).abs()
        
        gate_loss = 0.1 * (read_gate_loss + write_gate_loss)
    
    # === 3. TOTAL LOSS ===
    total_loss = task_loss + gate_loss
    
    # Metrics
    metrics = {
        'loss_total': total_loss.detach().item(),
        'loss_task': task_loss.detach().item(),
        'loss_gate': gate_loss.detach().item() if isinstance(gate_loss, torch.Tensor) else gate_loss,
        'avg_read_gate': avg_read.item() if read_gates else 0.0,
        'avg_write_gate': avg_write.item() if (read_gates and write_gates) else 0.0,
        'passes_run': aux.get('passes_run', 1),
        'layer_iters': aux.get('layer_iters', 1),
    }
    
    return total_loss, metrics


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: CNNCacheModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    global_step: int = 0,
    logger: Optional[MetricsLogger] = None,
    log_interval: int = 10,
    max_passes: Optional[int] = None,
    max_layer_iters: Optional[int] = None,
) -> Tuple[float, float, float, int]:
    """Training epoch."""
    model.train()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_tasks = 0
    total_tasks = 0
    cell_acc = 0.0
    task_acc = 0.0
    
    IGNORE_LABEL = -100
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        
        optimizer.zero_grad()
        
        # Forward (simple: fixed M passes, K layer iters)
        logits, cache, aux = model(
            demo_inputs, demo_outputs, test_input,
            config=config,
            step=global_step,
            return_aux=True,
            max_passes=max_passes,
            max_layer_iters=max_layer_iters,
        )
        
        # Loss
        loss, metrics = compute_cnn_loss(
            model, logits, test_output,
            aux, config, global_step, device
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Check for NaN
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
        
        # Logging
        if logger is not None and batch_idx % log_interval == 0:
            logger.log_step(
                step=global_step,
                metrics=metrics,
                aux=aux,
                config=config,
                cell_acc=cell_acc,
                task_acc=task_acc,
            )
        
        # Visualization
        if global_step % 50 == 0:
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
        
        # Progress bar
        M = aux.get('passes_run', 1)
        K = aux.get('layer_iters', 1)
        pbar.set_postfix({
            'loss': f'{loss_val:.3f}',
            'cell': f'{cell_acc:.3f}',
            'task': f'{task_acc:.3f}',
            'M': M, 'K': K,
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
    model: CNNCacheModel,
    dataloader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    global_step: int = 0,
    visualize_samples: int = 3,
) -> Tuple[float, float]:
    """Evaluation."""
    model.eval()
    correct_tasks = 0
    total_tasks = 0
    correct_cells = 0
    total_cells = 0
    samples_visualized = 0
    cell_acc = 0.0
    task_acc = 0.0
    
    IGNORE_LABEL = -100
    
    eval_config = TrainingConfig(
        tau_min=0.1,
        tau_start=0.1,
        max_passes=config.max_passes,
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
    
    # Count by component
    embed_params = sum(p.numel() for n, p in model.named_parameters() if 'embed' in n)
    conv_params = sum(p.numel() for n, p in model.named_parameters() if 'conv' in n)
    memory_params = sum(p.numel() for n, p in model.named_parameters() if 'memory' in n)
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"  - Embeddings:       {embed_params:,}")
    print(f"  - Convolutions:     {conv_params:,}")
    print(f"  - Memory/Cache:     {memory_params:,}")
    print("="*60 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    import sys
    
    # Download Mini-ARC
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
    # CNN + Cache version - Simple refinement with fixed M passes and K layer iters
    # 
    # Refinement parameters:
    #   max_passes (M): Model-level refinement passes
    #   max_layer_iters (K): Layer-level iterations per pass
    #
    # Default learning rate (can be overridden per-preset)
    lr = 5e-4
    
    if preset_name == "debug":
        # Minimal config for debugging
        d_model, d_cache = 32, 24
        num_layers, num_slots = 2, 8
        kernel_size, num_conv_layers = 3, 1
        batch_size = 16
        max_passes = 2      # M
        max_layer_iters = 1 # K
        num_epochs = 10
    elif preset_name == "fast":
        # Quick training for iteration
        d_model, d_cache = 64, 48
        num_layers, num_slots = 3, 16
        kernel_size, num_conv_layers = 5, 2
        batch_size = 32
        max_passes = 3      # M
        max_layer_iters = 1 # K
        num_epochs = 50
    elif preset_name == "fast_full":
        # Full features, moderate compute
        d_model, d_cache = 64, 32
        num_layers, num_slots = 8, 120  # (24 slots for each layer)
        kernel_size, num_conv_layers = 3, 3
        batch_size = 8
        max_passes = 1      # M
        max_layer_iters = 1 # K
        num_epochs = 100
    elif preset_name == "deep":
        # More layer iterations for deeper refinement
        d_model, d_cache = 64, 48
        num_layers, num_slots = 4, 32
        kernel_size, num_conv_layers = 5, 2
        batch_size = 16
        max_passes = 2      # M - fewer passes
        max_layer_iters = 4 # K - more layer iters
        num_epochs = 100
        lr = 3e-4
    elif preset_name == "wide":
        # More passes, fewer layer iterations
        d_model, d_cache = 64, 48
        num_layers, num_slots = 4, 32
        kernel_size, num_conv_layers = 5, 2
        batch_size = 16
        max_passes = 6      # M - more passes
        max_layer_iters = 1 # K - single layer iter
        num_epochs = 100
        lr = 3e-4
    elif preset_name == "compare":
        # Direct comparison with transformer - matched params
        d_model, d_cache = 64, 48
        num_layers, num_slots = 4, 32
        kernel_size, num_conv_layers = 5, 2
        batch_size = 8
        max_passes = 3      # M
        max_layer_iters = 1 # K
        num_epochs = 100
    else:  # full
        # Full model for best performance
        d_model, d_cache = 128, 64
        num_layers, num_slots = 4, 32
        kernel_size, num_conv_layers = 5, 3
        batch_size = 16
        max_passes = 4      # M
        max_layer_iters = 2 # K
        num_epochs = 100
    
    # Dataset
    dataset = MiniARCDataset(str(data_path), augment=True)
    
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nTrain samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    print(f"CNN config: kernel={kernel_size}, conv_layers={num_conv_layers}")
    print(f"Cache config: {num_layers} layers × {num_slots} slots")
    print(f"Refinement: M={max_passes} model passes, K={max_layer_iters} layer iters")
    
    # Config
    config = TrainingConfig(
        tau_start=1.0,
        tau_min=0.1,
        max_passes=max_passes,
        features=features,
    )
    
    # Model
    model = CNNCacheModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        d_cache=d_cache,
        num_layers=num_layers,
        num_slots=num_slots,
        kernel_size=kernel_size,
        num_conv_layers_per_block=num_conv_layers,
        max_seq_len=MAX_SEQ_LEN,
        max_passes=max_passes,
        max_layer_iters=max_layer_iters,
        dropout=0.0,
    ).to(device)
    
    print_model_summary(model)
    
    # Optimizer
    optimizer = AdamAtan2(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
    )
    print(f"Using AdamAtan2 optimizer (lr={lr})")
    
    # Logging
    log_dir = Path("logs") / f"cnn_cache_{preset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    logger = MetricsLogger(writer)
    print(f"TensorBoard logs: {log_dir}")
    
    # Training
    best_task_acc = 0
    global_step = 0
    
    for epoch in range(num_epochs):
        train_loss, train_cell_acc, train_task_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, config, global_step,
            logger=logger, log_interval=10,
            max_passes=max_passes,
            max_layer_iters=max_layer_iters,
        )
        
        logger.log_epoch(epoch, train_loss, train_cell_acc, train_task_acc)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            eval_cell_acc, eval_task_acc = evaluate(
                model, eval_loader, device, config,
                global_step=global_step, visualize_samples=2,
            )
            
            # Save if improved (use >= for first save when both are 0)
            if eval_task_acc > best_task_acc or (eval_task_acc == 0 and best_task_acc == 0 and epoch == 4):
                best_task_acc = max(eval_task_acc, best_task_acc)
                torch.save(model.state_dict(), log_dir / "best_model.pt")
                print(f"  → Saved best model (task_acc={eval_task_acc:.3f})")
            
            logger.log_epoch(epoch, train_loss, train_cell_acc, train_task_acc,
                             eval_cell_acc=eval_cell_acc, eval_task_acc=eval_task_acc)
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Eval Task: {eval_task_acc:.3f} | Best: {best_task_acc:.3f}")
        else:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train Task: {train_task_acc:.3f}")
    
    # Always save final model
    torch.save(model.state_dict(), log_dir / "final_model.pt")
    print(f"  → Saved final model")
    
    logger.close()
    print(f"\n{'='*60}")
    print(f"CNN + Cache Training Complete!")
    print(f"Best task accuracy: {best_task_acc:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
