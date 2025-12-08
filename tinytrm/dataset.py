"""
ARC-AGI Dataset for TinyTRM
===========================

Handles loading and augmentation of ARC-AGI tasks.
Heavy data augmentation is crucial per the paper:
- Color permutation
- Dihedral group transforms (rotations, flips)
- Translations
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


def load_arc_task(path: str) -> Dict:
    """Load a single ARC task from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def grid_to_tensor(grid: List[List[int]]) -> torch.Tensor:
    """Convert grid list to tensor."""
    return torch.tensor(grid, dtype=torch.long)


def pad_grid(grid: torch.Tensor, max_size: int, pad_value: int = 10) -> torch.Tensor:
    """Pad or crop grid to max_size x max_size."""
    H, W = grid.shape
    padded = torch.full((max_size, max_size), pad_value, dtype=torch.long)
    # Clip if grid is larger than max_size
    h_end = min(H, max_size)
    w_end = min(W, max_size)
    padded[:h_end, :w_end] = grid[:h_end, :w_end]
    return padded


class ARCAugmentor:
    """
    Data augmentation for ARC tasks.
    Following the paper: 1000 augmentations per example using:
    - Color permutation (keeping 0=background)
    - Dihedral group (8 transforms: rotations + flips)
    - Translations
    """
    
    def __init__(self, num_colors: int = 10, max_grid_size: int = 30):
        self.num_colors = num_colors
        self.max_grid_size = max_grid_size
    
    def random_color_permutation(self) -> Dict[int, int]:
        """Generate random color permutation (keep 0 as background)."""
        colors = list(range(1, self.num_colors))
        random.shuffle(colors)
        perm = {0: 0}  # Keep background
        for i, c in enumerate(colors):
            perm[i + 1] = c
        return perm
    
    def apply_color_permutation(
        self, 
        grid: torch.Tensor, 
        perm: Dict[int, int]
    ) -> torch.Tensor:
        """Apply color permutation to grid."""
        result = grid.clone()
        for old_c, new_c in perm.items():
            result[grid == old_c] = new_c
        return result
    
    def random_dihedral(self) -> int:
        """Get random dihedral transform index (0-7)."""
        return random.randint(0, 7)
    
    def apply_dihedral(self, grid: torch.Tensor, transform_idx: int) -> torch.Tensor:
        """
        Apply dihedral group transform.
        0: identity
        1-3: rotations (90, 180, 270)
        4: horizontal flip
        5: vertical flip
        6-7: diagonal flips
        """
        if transform_idx == 0:
            return grid
        elif transform_idx == 1:
            return torch.rot90(grid, 1, [0, 1])
        elif transform_idx == 2:
            return torch.rot90(grid, 2, [0, 1])
        elif transform_idx == 3:
            return torch.rot90(grid, 3, [0, 1])
        elif transform_idx == 4:
            return torch.flip(grid, [1])  # horizontal
        elif transform_idx == 5:
            return torch.flip(grid, [0])  # vertical
        elif transform_idx == 6:
            return torch.transpose(grid, 0, 1)  # diagonal
        elif transform_idx == 7:
            return torch.flip(torch.transpose(grid, 0, 1), [0, 1])
        return grid
    
    def random_translation(self, grid: torch.Tensor, max_shift: int = 3) -> torch.Tensor:
        """Random translation (with wrap-around or padding)."""
        H, W = grid.shape
        shift_h = random.randint(-max_shift, max_shift)
        shift_w = random.randint(-max_shift, max_shift)
        return torch.roll(grid, shifts=(shift_h, shift_w), dims=(0, 1))
    
    def augment(
        self,
        demo_inputs: List[torch.Tensor],
        demo_outputs: List[torch.Tensor],
        test_input: torch.Tensor,
        test_output: torch.Tensor,
        use_color_perm: bool = True,
        use_dihedral: bool = True,
        use_translation: bool = False,  # Can break some tasks
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Apply consistent augmentation to entire task."""
        
        # Color permutation
        if use_color_perm:
            perm = self.random_color_permutation()
            demo_inputs = [self.apply_color_permutation(g, perm) for g in demo_inputs]
            demo_outputs = [self.apply_color_permutation(g, perm) for g in demo_outputs]
            test_input = self.apply_color_permutation(test_input, perm)
            test_output = self.apply_color_permutation(test_output, perm)
        
        # Dihedral transform
        if use_dihedral:
            transform = self.random_dihedral()
            demo_inputs = [self.apply_dihedral(g, transform) for g in demo_inputs]
            demo_outputs = [self.apply_dihedral(g, transform) for g in demo_outputs]
            test_input = self.apply_dihedral(test_input, transform)
            test_output = self.apply_dihedral(test_output, transform)
        
        return demo_inputs, demo_outputs, test_input, test_output


class ARCDataset(Dataset):
    """
    ARC-AGI Dataset.
    
    Loads tasks from JSON files and applies augmentation.
    Each task has:
    - 2-3 demo input/output pairs
    - 1-2 test input/output pairs
    
    We train on one test pair at a time.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "training",  # "training" or "evaluation"
        max_grid_size: int = 30,
        max_demos: int = 3,
        augment: bool = True,
        num_augmentations: int = 1,  # Per-sample augmentations during iteration
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_grid_size = max_grid_size
        self.max_demos = max_demos
        self.augment = augment
        self.num_augmentations = num_augmentations
        
        self.augmentor = ARCAugmentor(num_colors=10, max_grid_size=max_grid_size)
        
        # Load task paths
        split_dir = self.data_dir / split
        self.task_paths = list(split_dir.glob("*.json"))
        
        # Expand to include test pairs
        self.samples = []
        for path in self.task_paths:
            task = load_arc_task(str(path))
            for test_idx in range(len(task['test'])):
                self.samples.append((path, test_idx))
        
        print(f"Loaded {len(self.samples)} samples from {len(self.task_paths)} tasks ({split})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, test_idx = self.samples[idx]
        task = load_arc_task(str(path))
        
        # Get demos
        demos = task['train'][:self.max_demos]
        demo_inputs = [grid_to_tensor(d['input']) for d in demos]
        demo_outputs = [grid_to_tensor(d['output']) for d in demos]
        
        # Get test
        test = task['test'][test_idx]
        test_input = grid_to_tensor(test['input'])
        test_output = grid_to_tensor(test['output'])
        
        # Augment
        if self.augment:
            demo_inputs, demo_outputs, test_input, test_output = self.augmentor.augment(
                demo_inputs, demo_outputs, test_input, test_output
            )
        
        # Pad all grids
        demo_inputs = [pad_grid(g, self.max_grid_size) for g in demo_inputs]
        demo_outputs = [pad_grid(g, self.max_grid_size) for g in demo_outputs]
        test_input_padded = pad_grid(test_input, self.max_grid_size)
        test_output_padded = pad_grid(test_output, self.max_grid_size)
        
        # Pad demos to max_demos
        while len(demo_inputs) < self.max_demos:
            demo_inputs.append(torch.full((self.max_grid_size, self.max_grid_size), 10, dtype=torch.long))
            demo_outputs.append(torch.full((self.max_grid_size, self.max_grid_size), 10, dtype=torch.long))
        
        # Stack demos
        demo_inputs = torch.stack(demo_inputs)  # [N_demo, H, W]
        demo_outputs = torch.stack(demo_outputs)  # [N_demo, H, W]
        
        # Output mask (True = padding, shouldn't be predicted)
        output_mask = (test_output_padded == 10)
        
        # Original output size
        H, W = test_output.shape
        output_size = torch.tensor([H, W], dtype=torch.long)
        
        return {
            'demo_inputs': demo_inputs,
            'demo_outputs': demo_outputs,
            'test_input': test_input_padded,
            'test_output': test_output_padded,
            'output_mask': output_mask,
            'output_size': output_size,
            'task_id': str(path.stem),
        }


def collate_arc(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for ARC batches."""
    return {
        'demo_inputs': torch.stack([b['demo_inputs'] for b in batch]),
        'demo_outputs': torch.stack([b['demo_outputs'] for b in batch]),
        'test_input': torch.stack([b['test_input'] for b in batch]),
        'test_output': torch.stack([b['test_output'] for b in batch]),
        'output_mask': torch.stack([b['output_mask'] for b in batch]),
        'output_size': torch.stack([b['output_size'] for b in batch]),
        'task_id': [b['task_id'] for b in batch],
    }
