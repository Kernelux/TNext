"""
Training Script for Decoder-Only CNN + Cache Model on Mini-ARC
==============================================================

GPT-style decoder-only model with CNN compute blocks and selective cache memory.
Single forward pass (no refinement) — efficient like modern LLMs.

Key Features:
- Fully causal processing (autoregressive)
- Teacher forcing during training
- LTM (Long-Term Memory) + WM (Working Memory) dual memory system
- Simple loss: CE on generation tokens + gate regularization

Usage:
    python train_decoder_miniarc.py [preset]
    
Presets:
    debug   - Minimal for testing
    fast    - Quick training (laptop-friendly)
    medium  - Balanced performance
    fast_full - Same config as ARC-AGI-2 training
    full    - Best performance
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict

# Import components
from components import (
    FeatureFlags,
    TrainingConfig,
    FEATURE_PRESETS,
    AdamAtan2,
)
from components.decoder_cache_model import DecoderCacheModel, create_decoder_cache_model
from components.logging import MetricsLogger
from components.miniarc_dataset import (
    MiniARCDataset,
    VOCAB_SIZE, MAX_SEQ_LEN, PAD_TOKEN, COLOR_OFFSET, MAX_GRID_SIZE,
    sequence_to_grid, target_to_grid, download_miniarc,
)


# ============================================================================
# Visualization
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
    lines.append(f"[Step {step}] Decoder+Cache Mini-ARC Sample #{sample_idx}")
    
    if aux:
        ltm_read = aux.get('ltm_read_gates', [])
        ltm_write = aux.get('ltm_write_gates', [])
        wm_read = aux.get('wm_read_gates', [])
        wm_write = aux.get('wm_write_gates', [])
        
        if ltm_read:
            avg_read = sum(g.detach().mean().item() for g in ltm_read) / len(ltm_read)
            lines.append(f"  Avg LTM Read: {avg_read:.3f}")
        if ltm_write:
            avg_write = sum(g.detach().mean().item() for g in ltm_write) / len(ltm_write)
            lines.append(f"  Avg LTM Write: {avg_write:.3f}")
        if wm_read:
            avg_read = sum(g.detach().mean().item() for g in wm_read) / len(wm_read)
            lines.append(f"  Avg WM Read: {avg_read:.3f}")
        if wm_write:
            avg_write = sum(g.detach().mean().item() for g in wm_write) / len(wm_write)
            lines.append(f"  Avg WM Write: {avg_write:.3f}")
    
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
# Loss Computation (same as train_decoder_arc.py)
# ============================================================================

IGNORE_LABEL = -100


def compute_decoder_loss(
    logits: torch.Tensor,       # [B, S, V]
    targets: torch.Tensor,      # [B, S]
    aux: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """
    Loss for decoder model on ARC.
    
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
    gate_loss = torch.tensor(0.0, device=device)
    avg_read = torch.tensor(0.0, device=device)
    avg_write = torch.tensor(0.0, device=device)
    
    read_target = 0.4
    write_target = 0.3
    
    # Use LTM gates for regularization
    read_gates = aux.get('ltm_read_gates', [])
    write_gates = aux.get('ltm_write_gates', [])
    
    if read_gates:
        avg_read = torch.stack([g.mean() for g in read_gates]).mean()
        avg_write = torch.stack([g.mean() for g in write_gates]).mean() if write_gates else torch.tensor(0.0, device=device)
        
        # Target gate loss: squared error
        read_loss = (avg_read - read_target) ** 2
        write_loss = (avg_write - write_target) ** 2 if write_gates else torch.tensor(0.0, device=device)
        
        gate_loss = read_loss + write_loss
        
    # === 3. TOTAL LOSS ===
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
# Training Loop (with gradient accumulation like train_decoder_arc.py)
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
    visualize_interval: int = 50,
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
            ltm_read_gates = aux.get('ltm_read_gates', [])
            ltm_write_gates = aux.get('ltm_write_gates', [])
            wm_read_gates = aux.get('wm_read_gates', [])
            wm_write_gates = aux.get('wm_write_gates', [])
            wm_validity = aux.get('wm_validity', [])
            
            log_aux = {'passes_run': 1}
            
            def get_gate_rate(gates):
                if not gates: return 0.0
                total_mean = sum(g.detach().mean().item() for g in gates) / len(gates)
                return total_mean * 100
            
            if ltm_read_gates:
                log_aux['ltm_read_rate'] = get_gate_rate(ltm_read_gates)
            if ltm_write_gates:
                log_aux['ltm_write_rate'] = get_gate_rate(ltm_write_gates)
            if wm_read_gates:
                log_aux['wm_read_rate'] = get_gate_rate(wm_read_gates)
            if wm_write_gates:
                log_aux['wm_write_rate'] = get_gate_rate(wm_write_gates)
            
            if wm_validity:
                all_valid = torch.cat([v.flatten() for v in wm_validity])
                pct_occupied = all_valid.mean().item() * 100
                log_aux['wm_slots_occupied_pct'] = pct_occupied
            
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
    parser = argparse.ArgumentParser(description="Train Decoder+Cache on Mini-ARC")
    parser.add_argument("preset", nargs="?", default="fast", help="debug|fast|medium|fast_full|full")
    parser.add_argument("--ltm-wta", action="store_true", help="Use winner-take-all for LTM write collisions")
    parser.add_argument("--wm-blend", action="store_true", help="Use blended writes for WM collisions")
    args = parser.parse_args()
    
    # Download Mini-ARC
    data_path = download_miniarc("./Mini-ARC")
    
    # Device
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Preset
    preset_name = args.preset
    ltm_wta_write = 'WTA'
    wm_wta_write = 'WTA'
    
    print(f"\nUsing preset: '{preset_name}'")
    print(f"LTM write collision: {'WTA' if ltm_wta_write else 'blend'}")
    print(f"WM write collision: {'WTA' if wm_wta_write else 'blend'}")
    
    # Model configuration per preset (matching train_decoder_arc.py style)
    lr = 3e-4
    grad_accum_steps = 1
    soft_eviction = False
    num_working_memory_slots = 16
    
    if preset_name == "debug":
        d_model, d_cache = 32, 16
        num_layers, num_slots = 2, 8
        num_working_memory_slots = 8
        kernel_size, num_conv_layers = 3, 1
        batch_size = 8
        num_epochs = 10
        grad_accum_steps = 1
    elif preset_name == "fast":
        d_model, d_cache = 64, 32
        num_layers, num_slots = 4, 16
        num_working_memory_slots = 16
        kernel_size, num_conv_layers = 5, 2
        batch_size = 16
        num_epochs = 50
        lr = 2e-4
        grad_accum_steps = 1
    elif preset_name == "medium":
        d_model, d_cache = 96, 48
        num_layers, num_slots = 6, 24
        num_working_memory_slots = 24
        kernel_size, num_conv_layers = 5, 2
        batch_size = 8
        num_epochs = 100
        lr = 1e-4
        grad_accum_steps = 2
    elif preset_name == "fast_full":
        # Same config as train_decoder_arc.py fast_full
        d_model, d_cache = 32, 16
        num_layers, num_slots = 8, 32
        num_working_memory_slots = 32
        kernel_size, num_conv_layers = 3, 5
        batch_size = 8
        num_epochs = 100
        lr = 1e-3
        grad_accum_steps = 2
        soft_eviction = False
    else:  # full
        d_model, d_cache = 128, 64
        num_layers, num_slots = 8, 48
        num_working_memory_slots = 48
        kernel_size, num_conv_layers = 5, 3
        batch_size = 4
        num_epochs = 200
        lr = 5e-5
        grad_accum_steps = 4
    
    # Dataset
    dataset = MiniARCDataset(str(data_path), augment=True)
    
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
    )
    
    print(f"\nTrain samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    print(f"Sequence length: {MAX_SEQ_LEN} (5x5 grids)")
    print(f"\nConfig: d_model={d_model}, d_cache={d_cache}")
    print(f"        {num_layers} layers × {num_slots} LTM slots × {num_working_memory_slots} WM slots")
    print(f"        kernel={kernel_size}, conv_layers={num_conv_layers}")
    print(f"        batch_size={batch_size}, grad_accum={grad_accum_steps}")
    print(f"        effective_batch={batch_size * grad_accum_steps}")
    
    # Model - Mini-ARC uses smaller sequences
    # Format: [demo_in_1, demo_out_1, ..., demo_in_N, demo_out_N, test_in, test_out]
    num_demos = 3
    actual_max_seq_len = (2 * num_demos + 2) * MAX_SEQ_LEN  # 8 * 25 = 200
    
    model = DecoderCacheModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        d_cache=d_cache,
        num_layers=num_layers,
        num_slots=num_slots,  # LTM slots
        kernel_size=kernel_size,
        num_conv_layers_per_block=num_conv_layers,
        max_seq_len=actual_max_seq_len,
        soft_eviction=soft_eviction,
        ltm_wta_write=ltm_wta_write,
        wm_wta_write=wm_wta_write,
        num_wm_slots=num_working_memory_slots,
    ).to(device)
    
    print_model_summary(model)
    
    # Optimizer (same as train_decoder_arc.py)
    optimizer = AdamAtan2(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
    )
    print(f"Using AdamAtan2 optimizer (lr={lr})")
    
    # Logging
    log_dir = Path("logs") / f"decoder_miniarc_{preset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
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
            visualize_interval=50,
        )
        
        logger.log_epoch(epoch, train_loss, train_cell_acc, train_task_acc)
        
        # Evaluate every 5 epochs (or every epoch for debug)
        eval_interval = 1 if preset_name == "debug" else 5
        if (epoch + 1) % eval_interval == 0:
            eval_cell_acc, eval_task_acc = evaluate(
                model, eval_loader, device,
                global_step=global_step, 
                visualize_samples=2,
                use_generation=True,  # Use autoregressive generation
                logger=logger,
            )
            
            # Save best model
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
                  f"Eval Task: {eval_task_acc:.3f} | Eval Cell: {eval_cell_acc:.3f} | "
                  f"Best: {best_task_acc:.3f} | τ: {temperature:.2f}")
        else:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {train_loss:.4f} | "
                  f"Train Task: {train_task_acc:.3f} | Train Cell: {train_cell_acc:.3f} | "
                  f"τ: {temperature:.2f}")
        
        # Checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
    print(f"Mini-ARC Decoder + Cache Training Complete!")
    print(f"Best task accuracy: {best_task_acc:.3f}")
    print(f"Best cell accuracy: {best_cell_acc:.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
