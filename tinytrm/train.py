"""
Training loop for TinyTRM on ARC-AGI
====================================

Key training components from the paper:
- Deep supervision: train on all supervision steps
- ACT loss: Q-head predicts halt/continue with Q-learning
- Stable-max loss: improves stability
- EMA: prevents overfitting
- Heavy augmentation: 1000 augmentations per example
- Global slot memory: persists across supervision steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

# Handle both direct execution and module import
try:
    from .model import TinyTRMForARC, TRMConfig, EMA
    from .dataset import ARCDataset, collate_arc
except ImportError:
    from model import TinyTRMForARC, TRMConfig, EMA
    from dataset import ARCDataset, collate_arc


def stable_softmax_cross_entropy(
    logits: torch.Tensor, 
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Stable softmax cross-entropy loss.
    Uses log-sum-exp trick for numerical stability.
    """
    # logits: [B, H, W, C]
    # targets: [B, H, W]
    B, H, W, C = logits.shape
    
    # Use reshape instead of view for non-contiguous tensors
    logits_flat = logits.reshape(-1, C)
    targets_flat = targets.reshape(-1)
    
    if mask is not None:
        mask_flat = mask.reshape(-1)
        # Only compute loss for non-masked positions
        logits_flat = logits_flat[~mask_flat]
        targets_flat = targets_flat[~mask_flat]
    
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
    return loss


def compute_act_loss(
    q_halt: torch.Tensor,      # [B, n_sup]
    q_continue: torch.Tensor,  # [B, n_sup]
    all_logits: list,          # List of [B, H, W, C]
    targets: torch.Tensor,     # [B, H, W]
    mask: torch.Tensor,        # [B, H, W]
) -> torch.Tensor:
    """
    ACT loss with Q-learning style targets.
    
    Q_halt should be high when prediction is correct (should stop).
    Q_continue should be high when prediction is wrong (should continue).
    
    Uses binary cross-entropy with soft targets.
    """
    B = q_halt.shape[0]
    n_sup = q_halt.shape[1]
    
    act_loss = 0.0
    for step in range(n_sup):
        logits = all_logits[step]  # [B, H, W, C]
        preds = logits.argmax(dim=-1)  # [B, H, W]
        
        # Check if prediction is correct (per sample)
        correct = []
        for b in range(B):
            m = ~mask[b]
            is_correct = (preds[b][m] == targets[b][m]).all().float()
            correct.append(is_correct)
        correct = torch.stack(correct)  # [B]
        
        # Q_halt should predict correctness
        q_h = torch.sigmoid(q_halt[:, step])  # [B]
        halt_loss = F.binary_cross_entropy(q_h, correct)
        
        # Q_continue should predict incorrectness (inverted)
        q_c = torch.sigmoid(q_continue[:, step])  # [B]
        continue_loss = F.binary_cross_entropy(q_c, 1 - correct)
        
        act_loss += halt_loss + continue_loss
    
    return act_loss / (2 * n_sup)


def train_epoch(
    model: TinyTRMForARC,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: Optional[EMA] = None,
    lambda_act: float = 0.5,
    n_supervision: int = 4,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
    log_interval: int = 10,
) -> Tuple[float, float, float, int]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct_cells = 0
    total_cells = 0
    correct_tasks = 0
    total_tasks = 0
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        demo_inputs = batch['demo_inputs'].to(device)
        demo_outputs = batch['demo_outputs'].to(device)
        test_input = batch['test_input'].to(device)
        test_output = batch['test_output'].to(device)
        output_mask = batch['output_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        result = model(
            demo_inputs, demo_outputs, test_input,
            n_supervision=n_supervision,
            return_all_steps=True,
        )
        
        # Task loss (final prediction)
        logits = result['logits']
        task_loss = stable_softmax_cross_entropy(logits, test_output, output_mask)
        
        # Deep supervision loss (all steps)
        deep_loss: torch.Tensor = torch.tensor(0.0, device=device)
        if 'all_logits' in result and len(result['all_logits']) > 1:
            for step_logits in result['all_logits'][:-1]:  # Exclude last (already in task_loss)
                deep_loss = deep_loss + stable_softmax_cross_entropy(step_logits, test_output, output_mask)
            deep_loss = deep_loss / (len(result['all_logits']) - 1)
        
        # ACT loss
        act_loss = compute_act_loss(
            result['q_halt'], 
            result['q_continue'],
            result['all_logits'], 
            test_output, 
            output_mask
        )
        
        # Total loss
        loss = task_loss + deep_loss + lambda_act * act_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Metrics
        total_loss += loss.detach().item()
        global_step += 1
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            for i in range(preds.shape[0]):
                m = ~output_mask[i]
                correct_cells += (preds[i][m] == test_output[i][m]).sum().item()
                total_cells += m.sum().item()
                if (preds[i][m] == test_output[i][m]).all():
                    correct_tasks += 1
            total_tasks += 1
        
        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        
        if writer is not None and batch_idx % log_interval == 0:
            writer.add_scalar('Loss/total', loss.item(), global_step)
            writer.add_scalar('Loss/task', task_loss.item(), global_step)
            writer.add_scalar('Loss/deep', deep_loss.item(), global_step)
            writer.add_scalar('Loss/act', act_loss.item(), global_step)
            writer.add_scalar('Metrics/cell_acc', cell_acc, global_step)
            writer.add_scalar('Metrics/task_acc', task_acc, global_step)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'cell': f'{cell_acc:.3f}',
            'task': f'{task_acc:.3f}',
        })
        
        # Memory cleanup
        del loss, logits, result
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
    
    cell_acc = correct_cells / max(total_cells, 1) if total_cells > 0 else 0.0
    task_acc = correct_tasks / max(total_tasks, 1) if total_tasks > 0 else 0.0
    return total_loss / len(dataloader), cell_acc, task_acc, global_step


@torch.no_grad()
def evaluate(
    model: TinyTRMForARC,
    dataloader: DataLoader,
    device: torch.device,
    ema: Optional[EMA] = None,
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    
    # Use EMA weights for evaluation
    if ema is not None:
        ema.apply_shadow()
    
    correct_cells = 0
    total_cells = 0
    correct_tasks = 0
    total_tasks = 0
    
    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for batch in pbar:
        demo_inputs = batch['demo_inputs'].to(device)
        demo_outputs = batch['demo_outputs'].to(device)
        test_input = batch['test_input'].to(device)
        test_output = batch['test_output'].to(device)
        output_mask = batch['output_mask'].to(device)
        
        result = model(
            demo_inputs, demo_outputs, test_input,
            n_supervision=1,  # Single pass for evaluation
            return_all_steps=False,
        )
        
        logits = result['logits']
        preds = logits.argmax(dim=-1)
        
        for i in range(preds.shape[0]):
            m = ~output_mask[i]
            correct_cells += (preds[i][m] == test_output[i][m]).sum().item()
            total_cells += m.sum().item()
            if (preds[i][m] == test_output[i][m]).all():
                correct_tasks += 1
            total_tasks += 1
        
        cell_acc = correct_cells / max(total_cells, 1)
        task_acc = correct_tasks / max(total_tasks, 1)
        pbar.set_postfix({'cell': f'{cell_acc:.3f}', 'task': f'{task_acc:.3f}'})
    
    # Restore original weights
    if ema is not None:
        ema.restore()
    
    cell_acc = correct_cells / max(total_cells, 1) if total_cells > 0 else 0.0
    task_acc = correct_tasks / max(total_tasks, 1) if total_tasks > 0 else 0.0
    return cell_acc, task_acc


def train(
    config: Optional[TRMConfig] = None,
    data_dir: str = "./ARC-AGI-2/data",
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    device: Optional[torch.device] = None,
):
    """Main training function."""
    if config is None:
        config = TRMConfig()
    
    if device is None:
        device = torch.device(
            'mps' if torch.backends.mps.is_available() 
            else 'cuda' if torch.cuda.is_available() 
            else 'cpu'
        )
    
    print(f"Device: {device}")
    print(f"Config: {config}")
    
    # Load data
    train_dataset = ARCDataset(data_dir, split="training", max_grid_size=config.max_grid_size)
    eval_dataset = ARCDataset(data_dir, split="evaluation", max_grid_size=config.max_grid_size, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_arc,
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_arc,
    )
    
    # Create model
    model = TinyTRMForARC(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer (paper: AdamW with β1=0.9, β2=0.95)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    
    # EMA
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    
    # Logging
    log_dir = Path("logs") / f"tinytrm_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"Logs: {log_dir}")
    
    # Training loop
    global_step = 0
    best_task_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_cell, train_task, global_step = train_epoch(
            model, train_loader, optimizer, device,
            ema=ema,
            n_supervision=4,
            writer=writer,
            global_step=global_step,
        )
        
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/train_cell', train_cell, epoch)
        writer.add_scalar('Epoch/train_task', train_task, epoch)
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            eval_cell, eval_task = evaluate(model, eval_loader, device, ema)
            writer.add_scalar('Epoch/eval_cell', eval_cell, epoch)
            writer.add_scalar('Epoch/eval_task', eval_task, epoch)
            
            if eval_task > best_task_acc:
                best_task_acc = eval_task
                # Save best model
                torch.save(model.state_dict(), log_dir / "best_model.pt")
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Eval: {eval_task:.3f} (best: {best_task_acc:.3f})")
        else:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train: {train_task:.3f}")
    
    writer.close()
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TinyTRM on ARC-AGI")
    parser.add_argument("--data-dir", type=str, default="../ARC-AGI-2/data", help="Path to ARC data")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num-slots", type=int, default=8, help="Number of memory slots")
    parser.add_argument("--d-slot", type=int, default=64, help="Slot dimension")
    parser.add_argument("--max-grid-size", type=int, default=20, help="Max grid size (20 for MPS, 30 for GPU)")
    
    args = parser.parse_args()
    
    config = TRMConfig(
        hidden_size=args.hidden_size,
        num_slots=args.num_slots,
        d_slot=args.d_slot,
        max_grid_size=args.max_grid_size,
    )
    
    train(
        config=config,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )