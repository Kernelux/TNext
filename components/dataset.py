import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import glob
import numpy as np
from typing import Tuple, Callable, Optional


# ============================================================================
# Data Augmentation (from TinyTRM)
# ============================================================================

def dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Apply one of 8 dihedral group transformations.
    
    transform_id:
        0: identity
        1: rotate 90° CCW
        2: rotate 180°
        3: rotate 270° CCW (90° CW)
        4: flip horizontal
        5: flip horizontal + rotate 90° CCW
        6: flip horizontal + rotate 180°
        7: flip horizontal + rotate 270° CCW
    """
    if transform_id == 0:
        return grid
    elif transform_id == 1:
        return np.rot90(grid, k=1)
    elif transform_id == 2:
        return np.rot90(grid, k=2)
    elif transform_id == 3:
        return np.rot90(grid, k=3)
    elif transform_id == 4:
        return np.fliplr(grid)
    elif transform_id == 5:
        return np.rot90(np.fliplr(grid), k=1)
    elif transform_id == 6:
        return np.rot90(np.fliplr(grid), k=2)
    elif transform_id == 7:
        return np.rot90(np.fliplr(grid), k=3)
    else:
        raise ValueError(f"Invalid transform_id: {transform_id}")


def inverse_dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """Inverse of dihedral_transform."""
    if transform_id == 0:
        return grid
    elif transform_id == 1:
        return np.rot90(grid, k=-1)
    elif transform_id == 2:
        return np.rot90(grid, k=-2)
    elif transform_id == 3:
        return np.rot90(grid, k=-3)
    elif transform_id == 4:
        return np.fliplr(grid)
    elif transform_id == 5:
        return np.fliplr(np.rot90(grid, k=-1))
    elif transform_id == 6:
        return np.fliplr(np.rot90(grid, k=-2))
    elif transform_id == 7:
        return np.fliplr(np.rot90(grid, k=-3))
    else:
        raise ValueError(f"Invalid transform_id: {transform_id}")


def create_color_permutation(rng: np.random.Generator) -> np.ndarray:
    """
    Create a color permutation mapping.
    
    Color 0 (black/background) is kept fixed.
    Colors 1-9 are randomly permuted.
    
    Returns: mapping array where mapping[old_color] = new_color
    """
    mapping = np.arange(10, dtype=np.uint8)
    mapping[1:10] = rng.permutation(np.arange(1, 10, dtype=np.uint8))
    return mapping


def apply_augmentation(
    grid: np.ndarray,
    transform_id: int,
    color_mapping: np.ndarray,
) -> np.ndarray:
    """Apply dihedral transform + color permutation to a grid."""
    # Apply color permutation first
    grid = color_mapping[grid]
    # Apply dihedral transform
    grid = dihedral_transform(grid, transform_id)
    return grid


def create_augmentation_fn(rng: np.random.Generator) -> Tuple[Callable, int, np.ndarray]:
    """
    Create an augmentation function with random transform.
    
    Returns:
        - aug_fn: function that applies the augmentation to a grid
        - transform_id: the dihedral transform ID (for inverse)
        - color_mapping: the color permutation (for inverse)
    """
    transform_id = rng.integers(0, 8)
    color_mapping = create_color_permutation(rng)
    
    def aug_fn(grid: np.ndarray) -> np.ndarray:
        return apply_augmentation(grid, transform_id, color_mapping)
    
    return aug_fn, transform_id, color_mapping


class ARCDataset(Dataset):
    """
    ARC-AGI dataset loader with data augmentation.
    
    Each task has:
    - train: list of {input: grid, output: grid} demo pairs
    - test: list of {input: grid, output: grid} test pairs
    
    Augmentation (during training):
    - Dihedral transforms: 8 possible (rotations + reflections)
    - Color permutation: random permutation of colors 1-9 (0 stays fixed)
    
    Files are loaded on-the-fly to avoid slow startup.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = "training", 
        max_grid_size: int = 30,
        augment: bool = True,
        seed: int = 42,
    ):
        self.max_grid_size = max_grid_size
        self.task_dir = Path(data_dir) / split
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        
        # Just get file list (fast)
        self.task_files = sorted(self.task_dir.glob("*.json"))
        print(f"Found {len(self.task_files)} tasks in {self.task_dir}")
        
        # Build index: (file_idx, test_idx) for each sample
        # We need to know how many test cases per file - do a quick scan
        self.index = []
        for file_idx, task_file in enumerate(self.task_files):
            with open(task_file) as f:
                task = json.load(f)
            num_tests = len(task["test"])
            for test_idx in range(num_tests):
                self.index.append((file_idx, test_idx))
        
        print(f"Created {len(self.index)} samples")
        
        # Cache for recently loaded files
        self._cache = {}
        self._cache_size = 100
    
    def __len__(self):
        return len(self.index)
    
    def _load_task(self, file_idx):
        """Load task file with simple caching."""
        if file_idx not in self._cache:
            # Evict old entries if cache is full
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            
            with open(self.task_files[file_idx]) as f:
                self._cache[file_idx] = json.load(f)
        
        return self._cache[file_idx]
    
    def pad_grid(self, grid: np.ndarray, size: int):
        """Pad grid to fixed size, return grid and mask."""
        h, w = grid.shape
        padded = torch.zeros(size, size, dtype=torch.long)
        mask = torch.ones(size, size, dtype=torch.bool)  # True = padding
        
        for i in range(min(h, size)):
            for j in range(min(w, size)):
                padded[i, j] = int(grid[i, j])
                mask[i, j] = False
        
        return padded, mask, h, w
    
    def _grid_to_np(self, grid_list) -> np.ndarray:
        """Convert grid list to numpy array."""
        return np.array(grid_list, dtype=np.uint8)
    
    def __getitem__(self, idx):
        # Get file and test indices from our index
        file_idx, test_idx = self.index[idx]
        task = self._load_task(file_idx)
        
        size = self.max_grid_size
        
        # Create augmentation for this sample (if training)
        if self.augment:
            aug_fn, transform_id, color_mapping = create_augmentation_fn(self.rng)
        else:
            aug_fn = lambda x: x
            transform_id = 0
            color_mapping = np.arange(10, dtype=np.uint8)
        
        # Encode demo pairs (from "train" in the JSON)
        demo_inputs = []
        demo_outputs = []
        demo_masks = []
        
        for pair in task["train"]:
            inp_np = aug_fn(self._grid_to_np(pair["input"]))
            out_np = aug_fn(self._grid_to_np(pair["output"]))
            
            inp, inp_mask, _, _ = self.pad_grid(inp_np, size)
            out, out_mask, _, _ = self.pad_grid(out_np, size)
            demo_inputs.append(inp)
            demo_outputs.append(out)
            demo_masks.append(inp_mask)
        
        # Pad to fixed number of demos (3)
        while len(demo_inputs) < 3:
            demo_inputs.append(torch.zeros(size, size, dtype=torch.long))
            demo_outputs.append(torch.zeros(size, size, dtype=torch.long))
            demo_masks.append(torch.ones(size, size, dtype=torch.bool))
        
        demo_inputs = torch.stack(demo_inputs[:3])  # [3, H, W]
        demo_outputs = torch.stack(demo_outputs[:3])  # [3, H, W]
        demo_masks = torch.stack(demo_masks[:3])  # [3, H, W]
        
        # Encode test (use specific test_idx)
        test_pair = task["test"][test_idx]
        test_inp_np = aug_fn(self._grid_to_np(test_pair["input"]))
        test_out_np = aug_fn(self._grid_to_np(test_pair["output"]))
        
        test_input, test_mask, _, _ = self.pad_grid(test_inp_np, size)
        test_output, output_mask, out_h, out_w = self.pad_grid(test_out_np, size)
        
        # Task ID from filename
        task_id = self.task_files[file_idx].stem
        
        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "demo_masks": demo_masks,
            "test_input": test_input,
            "test_mask": test_mask,
            "test_output": test_output,
            "output_mask": output_mask,
            "output_size": torch.tensor([out_h, out_w]),
            "task_id": task_id,
            # Store augmentation info for inverse transform during evaluation
            "transform_id": transform_id,
            "color_mapping": torch.from_numpy(color_mapping),
        }


class ARCDatasetNoAug(ARCDataset):
    """ARCDataset with augmentation disabled (for evaluation)."""
    
    def __init__(self, data_dir: str, split: str = "training", max_grid_size: int = 30):
        super().__init__(data_dir, split, max_grid_size, augment=False)
