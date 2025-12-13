"""
Training Script for Decoder-Only CNN + Cache Model on Mini-ARC
==============================================================

GPT-style decoder-only model with CNN compute blocks and selective cache memory.
Single forward pass (no refinement) — efficient like modern LLMs.

Key Features:
- Fully causal processing (autoregressive)
- Teacher forcing during training
- Cache provides global memory beyond CNN receptive field
- Simple loss: CE on generation tokens + gate regularization

Usage:
    python train_decoder_miniarc.py [preset]
    
Presets:
    debug   - Minimal for testing
    fast    - Quick training (laptop-friendly)
    medium  - Balanced performance
    full    - Best performance
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
        read_gates = aux.get('read_gates', [])
        write_gates = aux.get('write_gates', [])
        if read_gates:
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
# Loss Computation
# ============================================================================

IGNORE_LABEL = -100


def compute_decoder_loss(
    logits: torch.Tensor,       # [B, S, V]
    targets: torch.Tensor,      # [B, S]
    aux: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """
    Simple loss for decoder model.
    
    Components:
    1. Task loss: Cross-entropy on generation tokens only
    2. Gate usage loss: Encourage gates to be active (prevent dead cache)
    """
    # === 1. TASK LOSS ===
    B, S, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    
    task_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE_LABEL)
    
    # === 2. GATE USAGE LOSS ===
    gate_loss = torch.tensor(0.0, device=device)
    avg_read = torch.tensor(0.0, device=device)
    avg_write = torch.tensor(0.0, device=device)
    
    read_gates = aux.get('read_gates', [])
    write_gates = aux.get('write_gates', [])
    
    if read_gates:
        avg_read = torch.stack([g.mean() for g in read_gates]).mean()
        avg_write = torch.stack([g.mean() for g in write_gates]).mean() if write_gates else torch.tensor(0.0, device=device)
        
        # Encourage gates to be around 0.4
        target_gate = 0.4
        read_gate_loss = (avg_read - target_gate).abs()
        write_gate_loss = (avg_write - target_gate).abs()
        
        gate_loss = 0.1 * (read_gate_loss + write_gate_loss)
    
    # === 3. TOTAL LOSS ===
    total_loss = task_loss + gate_loss
    
    # Metrics (detach before .item() to avoid warning)
    metrics = {
        'loss_total': total_loss.detach().item(),
        'loss_task': task_loss.detach().item(),
        'loss_gate': gate_loss.detach().item() if isinstance(gate_loss, torch.Tensor) else gate_loss,
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
) -> Tuple[float, float, float, int]:
    """Training epoch with teacher forcing."""
    model.train()
    total_loss = 0
    correct_cells = 0
    total_cells = 0
    correct_tasks = 0
    total_tasks = 0
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        demo_inputs = batch["demo_inputs"].to(device)
        demo_outputs = batch["demo_outputs"].to(device)
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        
        optimizer.zero_grad()
        
        # Forward with teacher forcing
        logits, cache, aux = model(
            demo_inputs, demo_outputs, test_input, test_output,
            temperature=temperature,
            hard=(temperature < 0.2),
            return_aux=True,
        )
        
        # Loss
        loss, metrics = compute_decoder_loss(logits, test_output, aux, device)
        
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
        
        # Logging - convert gate tensors to format expected by logger
        if logger is not None and batch_idx % log_interval == 0:
            # Aggregate gate stats for logging
            read_gates = aux.get('read_gates', [])
            write_gates = aux.get('write_gates', [])
            
            log_aux = {
                'passes_run': 1,  # Single pass (decoder-only)
            }
            
            if read_gates:
                read_sum = sum(g.sum().item() for g in read_gates)
                read_count = sum(g.numel() for g in read_gates)
                log_aux['read_gate_sum'] = read_sum
                log_aux['read_gate_count'] = read_count
            
            if write_gates:
                write_sum = sum(g.sum().item() for g in write_gates)
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
        pbar.set_postfix({
            'loss': f'{loss_val:.3f}',
            'cell': f'{cell_acc:.3f}',
            'task': f'{task_acc:.3f}',
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
    model: DecoderCacheModel,
    dataloader: DataLoader,
    device: torch.device,
    global_step: int = 0,
    visualize_samples: int = 3,
    use_generation: bool = False,
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
                        aux=aux if not use_generation else None,
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
    print(f"\nUsing preset: '{preset_name}'")
    
    # Model configuration per preset (laptop-friendly defaults)
    lr = 5e-4
    
    if preset_name == "debug":
        d_model, d_cache = 32, 24
        num_layers, num_slots = 2, 8
        kernel_size, num_conv_layers = 3, 1
        batch_size = 8
        num_epochs = 10
    elif preset_name == "fast":
        d_model, d_cache = 64, 48
        num_layers, num_slots = 3, 16
        kernel_size, num_conv_layers = 5, 2
        batch_size = 16
        num_epochs = 50
    elif preset_name == "medium":
        d_model, d_cache = 96, 64
        num_layers, num_slots = 4, 24
        kernel_size, num_conv_layers = 5, 2
        batch_size = 8
        num_epochs = 100
        lr = 3e-4
    elif preset_name == "fast_full":
        # Larger model, quick epochs
        d_model, d_cache = 128, 64
        num_layers, num_slots = 6, 32
        kernel_size, num_conv_layers = 5, 3
        batch_size = 4
        num_epochs = 100
        lr = 2e-4
    else:  # full
        d_model, d_cache = 128, 64
        num_layers, num_slots = 6, 32
        kernel_size, num_conv_layers = 5, 3
        batch_size = 4
        num_epochs = 100
        lr = 2e-4
    
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
    print(f"Config: d_model={d_model}, d_cache={d_cache}")
    print(f"        {num_layers} layers × {num_slots} slots")
    print(f"        kernel={kernel_size}, conv_layers={num_conv_layers}")
    
    # Model
    model = DecoderCacheModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        d_cache=d_cache,
        num_layers=num_layers,
        num_slots=num_slots,
        kernel_size=kernel_size,
        num_conv_layers_per_block=num_conv_layers,
        max_seq_len=256,
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
    log_dir = Path("logs") / f"decoder_cache_{preset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger = MetricsLogger(writer)
    print(f"TensorBoard logs: {log_dir}")
    
    # Temperature schedule (anneal from 1.0 to 0.1)
    def get_temperature(epoch: int, num_epochs: int) -> float:
        return max(0.1, 1.0 - 0.9 * epoch / num_epochs)
    
    # Training
    best_task_acc = 0
    global_step = 0
    
    for epoch in range(num_epochs):
        temperature = get_temperature(epoch, num_epochs)
        
        train_loss, train_cell_acc, train_task_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, global_step,
            logger=logger, log_interval=10,
            temperature=temperature,
        )
        
        logger.log_epoch(epoch, train_loss, train_cell_acc, train_task_acc)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            eval_cell_acc, eval_task_acc = evaluate(
                model, eval_loader, device,
                global_step=global_step, visualize_samples=2,
                use_generation=False,  # Teacher forcing for speed
            )
            
            if eval_task_acc > best_task_acc or (eval_task_acc == 0 and best_task_acc == 0 and epoch == 4):
                best_task_acc = max(eval_task_acc, best_task_acc)
                torch.save(model.state_dict(), log_dir / "best_model.pt")
                print(f"  → Saved best model (task_acc={eval_task_acc:.3f})")
            
            logger.log_epoch(epoch, train_loss, train_cell_acc, train_task_acc,
                            eval_cell_acc=eval_cell_acc, eval_task_acc=eval_task_acc)
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Eval Task: {eval_task_acc:.3f} | Best: {best_task_acc:.3f} | τ: {temperature:.2f}")
        else:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train Task: {train_task_acc:.3f} | τ: {temperature:.2f}")
    
    # Final evaluation with autoregressive generation
    print("\n" + "="*60)
    print("Final evaluation with autoregressive generation...")
    print("="*60)
    eval_cell_acc, eval_task_acc = evaluate(
        model, eval_loader, device,
        global_step=global_step, visualize_samples=5,
        use_generation=True,  # True autoregressive
    )
    print(f"Autoregressive Eval: Cell={eval_cell_acc:.3f}, Task={eval_task_acc:.3f}")
    
    # Save final model
    torch.save(model.state_dict(), log_dir / "final_model.pt")
    print(f"  → Saved final model")
    
    logger.close()
    print(f"\n{'='*60}")
    print(f"Decoder + Cache Training Complete!")
    print(f"Best task accuracy: {best_task_acc:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
