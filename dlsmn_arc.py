"""
DLSMN on ARC-AGI-2
==================
Faithful implementation of DLSM_V0.1.md for the Abstraction and Reasoning Corpus.

This file now serves as the entry point, importing modular components.
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
    grid_to_rgb
)

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

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        output_mask = batch["output_mask"].to(device)
        output_size = batch["output_size"].to(device)

        # Only zero grad at accumulation boundaries
        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        # Forward pass with training config
        logits, size_logits, cache, aux_info = model(
            demo_inputs, demo_outputs, test_input,
            config=config,
            step=global_step,
            return_aux=True,
        )

        # Compute Loss (Refactored)
        loss, metrics = compute_total_loss(
            model, logits, size_logits, test_output, output_size,
            aux_info, config, global_step, device
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        # Only step optimizer at accumulation boundaries
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # === CRITICAL: Memory cleanup to prevent graph accumulation ===
        loss_val = loss.detach().item()
        total_loss += loss_val
        global_step += 1

        # Metrics
        preds = logits.detach().argmax(dim=-1)
        correct = ((preds == test_output) & ~output_mask).sum().item()
        total = (~output_mask).sum().item()
        correct_cells += correct
        total_cells += total

        for i in range(preds.shape[0]):
            mask = ~output_mask[i]
            if (preds[i][mask] == test_output[i][mask]).all():
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

            # === Component Losses ===
            if metrics.get('loss_q_head', 0) > 0:
                writer.add_scalar('Loss/model_q_head', metrics['loss_q_head'], global_step)
            if metrics.get('loss_ponder', 0) > 0:
                writer.add_scalar('Loss/model_ponder', metrics['loss_ponder'], global_step)
            if metrics.get('loss_step_efficiency', 0) > 0:
                writer.add_scalar('Loss/step_efficiency', metrics['loss_step_efficiency'], global_step)

            # === Metrics ===
            writer.add_scalar('Metrics/cell_accuracy', cell_acc, global_step)
            writer.add_scalar('Metrics/task_accuracy', task_acc, global_step)

            # === Training Dynamics ===
            writer.add_scalar('Training/temperature', temperature_val, global_step)
            writer.add_scalar('Training/num_passes', aux_info.get('num_passes', 1), global_step)
            writer.add_scalar('Training/lr', optimizer.param_groups[0]['lr'], global_step)

            # Log pass exploration mode
            mode_map = {'force_max': 0, 'adaptive': 1, 'adaptive_penalized': 2}
            writer.add_scalar('Training/pass_mode', mode_map.get(pass_mode_val, 1), global_step)

            # === Visualizations ===
            # Input (demo 0 input)
            inp_grid = demo_inputs[0, 0] # [H, W]
            writer.add_image('Vis/1_Input', grid_to_rgb(inp_grid), global_step)

            # Target
            tgt_grid = test_output[0] # [H, W]
            writer.add_image('Vis/2_Target', grid_to_rgb(tgt_grid), global_step)

            # Prediction (Final)
            pred_grid = logits[0].argmax(dim=-1) # [H, W]
            writer.add_image('Vis/3_Pred_Final', grid_to_rgb(pred_grid), global_step)

            # Prediction (Pass 1) if multi-pass
            if 'pass_logits' in aux_info and len(aux_info['pass_logits']) > 1:
                p1_grid = aux_info['pass_logits'][0][0].argmax(dim=-1)
                writer.add_image('Vis/4_Pred_Pass1', grid_to_rgb(p1_grid), global_step)

            pbar.set_postfix({
                'loss': f'{loss_val:.3f}',
                'cell': f'{cell_acc:.3f}',
                'task': f'{task_acc:.3f}',
                'τ': f'{temperature_val:.2f}',
                'mode': pass_mode_val[:3],
            })

        # === CRITICAL: Memory cleanup after each batch ===
        # Clear all tensor references
        del loss, logits, size_logits, cache

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
    config: TrainingConfig
) -> Tuple[float, float]:
    model.eval()
    correct_tasks = 0
    total_tasks = 0
    correct_cells = 0
    total_cells = 0
    cell_acc = 0.0
    task_acc = 0.0

    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for batch in pbar:
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        output_mask = batch["output_mask"].to(device)

        # Use hard routing during evaluation
        eval_config = TrainingConfig(
            tau_min=0.1,
            tau_start=0.1,  # Hard routing
            max_passes=config.max_passes,
            max_recurrent_steps=config.max_recurrent_steps,
        )

        logits, _, _, _ = model(
            demo_inputs, demo_outputs, test_input,
            config=eval_config,
            step=100000,  # High step for minimum temperature
            return_aux=False,
        )

        preds = logits.argmax(dim=-1)

        for i in range(preds.shape[0]):
            mask = ~output_mask[i]
            correct_cells += (preds[i][mask] == test_output[i][mask]).sum().item()
            total_cells += mask.sum().item()
            if (preds[i][mask] == test_output[i][mask]).all():
                correct_tasks += 1
            total_tasks += 1

        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        pbar.set_postfix({'cell': f'{cell_acc:.3f}', 'task': f'{task_acc:.3f}'})

    return cell_acc, task_acc


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

    # Configure model params
    if preset_name == "fast_full":
        max_grid_size = 15
        d_model = 64
        d_cache = 48
        num_layers = 2
        num_slots = 8
        num_patterns = 8
        num_heads = 2
        batch_size = 8
        max_passes = 4
        max_recurrent_steps = 2
    elif preset_name == "runpod":
        # Optimized for ~44GB GPU with LinearAttention (O(S) memory)
        # Increased capacity since we no longer have O(S²) attention memory
        max_grid_size = 30
        d_model = 128      # Restored - LinearAttention makes this feasible
        d_cache = 64       # Restored
        num_layers = 4
        num_slots = 16
        num_patterns = 16
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
        num_patterns = 16
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
        num_patterns = 8
        num_heads = 2
        batch_size = 8
        max_passes = 2 if features.use_multi_pass else 1
        max_recurrent_steps = 1

    train_dataset = ARCDataset(str(data_dir), split="training", max_grid_size=max_grid_size)
    eval_dataset = ARCDataset(str(data_dir), split="evaluation", max_grid_size=max_grid_size)

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
            lambda_ponder=0.02,          # Reduced to allow more passes when needed
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
        num_colors=10,
        d_model=d_model,
        d_cache=d_cache,
        num_layers=num_layers,
        num_slots=num_slots,
        num_patterns=num_patterns,
        num_heads=num_heads,
        max_grid_size=max_grid_size,
        max_recurrent_steps=max_recurrent_steps,
        max_passes=max_passes,
        dropout=0.1,
    ).to(device)

    # Higher LR for runpod with cosine schedule
    lr = 3e-4 if preset_name == "runpod" else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Gradient accumulation - effective batch = batch_size * grad_accum_steps
    grad_accum_steps = 2 if preset_name in ["full", "runpod"] else 1

    log_dir = Path("logs") / f"dlsmn_{preset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")

    best_task_acc = 0
    global_step = 0

    for epoch in range(50):
        train_loss, train_cell_acc, train_task_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, config, global_step,
            writer=writer, log_interval=10, grad_accum_steps=grad_accum_steps
        )

        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/train_cell_acc', train_cell_acc, epoch)

        if (epoch + 1) % 10 == 0:
            eval_cell_acc, eval_task_acc = evaluate(model, eval_loader, device, config)
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
