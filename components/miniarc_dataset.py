"""
Mini-ARC Dataset Loader
=======================

Mini-ARC is a simplified version of ARC with smaller 5x5 grids instead of 30x30.
This allows for much faster iteration during development.

Reference: "Playgrounds for Abstraction and Reasoning" (NeurIPS 2022 Workshop)
https://github.com/ksb21ST/Mini-ARC

Grid format:
- Maximum size: 5x5 (vs 30x30 for ARC-AGI-2)
- Colors: 0-9 (same as ARC)
- Structure: Same JSON format as ARC

Sequence format (same as ARC but much shorter):
    [demo_in_1, demo_out_1, ..., demo_in_N, demo_out_N, test_in] → test_out
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np
from typing import List, Tuple, Optional, Dict
import random


# ============================================================================
# Constants for Mini-ARC
# ============================================================================

MAX_GRID_SIZE = 5  # Mini-ARC uses 5x5 max grids
MAX_SEQ_LEN = MAX_GRID_SIZE * MAX_GRID_SIZE  # 25 tokens per segment

# Vocabulary (same as ARC)
PAD_TOKEN = 0
EOS_TOKEN = 1
INPUT_MARKER = 2
OUTPUT_MARKER = 3
COLOR_OFFSET = 4  # Colors 0-9 map to tokens 4-13
VOCAB_SIZE = 14   # PAD, EOS, INPUT, OUTPUT, + 10 colors


# ============================================================================
# Dataset Class
# ============================================================================

class MiniARCDataset(Dataset):
    """
    Mini-ARC dataset with 5x5 grids for fast iteration.
    
    Benefits over ARC-AGI-2:
    - 36x smaller sequences (5x5=25 vs 30x30=900 tokens per segment)
    - Much faster training iterations
    - Same task structure (demos + test)
    - Ideal for architecture prototyping
    """
    
    def __init__(
        self,
        data_dir: str,
        max_demos: int = 3,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Path to Mini-ARC data folder (contains JSON files)
            max_demos: Maximum number of demo pairs to use
            augment: Whether to apply color permutation augmentation
            max_samples: Limit number of samples (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.max_demos = max_demos
        self.augment = augment
        self.max_grid_size = MAX_GRID_SIZE
        
        # Load all tasks
        self.samples = []
        self._load_tasks(max_samples)
        
        print(f"Loaded {len(self.samples)} Mini-ARC samples")
    
    def _load_tasks(self, max_samples: Optional[int] = None):
        """Load all Mini-ARC tasks from JSON files."""
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            # Check if data is in subdirectory
            json_files = list(self.data_dir.glob("**/*.json"))
        
        print(f"Found {len(json_files)} Mini-ARC task files")
        
        for task_file in json_files:
            try:
                with open(task_file, 'r') as f:
                    task = json.load(f)
                
                train_pairs = task.get('train', [])
                test_pairs = task.get('test', [])
                
                # Skip if no demos or tests
                if not train_pairs or not test_pairs:
                    continue
                
                # Each test pair becomes a sample
                for test_idx, test_pair in enumerate(test_pairs):
                    self.samples.append({
                        'task_id': task_file.stem,
                        'train': train_pairs,
                        'test_input': test_pair['input'],
                        'test_output': test_pair['output'],
                    })
                    
                    if max_samples and len(self.samples) >= max_samples:
                        return
                        
            except Exception as e:
                print(f"Error loading {task_file}: {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _grid_to_sequence(self, grid: List[List[int]]) -> np.ndarray:
        """Convert grid to padded sequence."""
        seq = np.full(MAX_SEQ_LEN, PAD_TOKEN, dtype=np.int64)
        
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                if r < MAX_GRID_SIZE and c < MAX_GRID_SIZE:
                    idx = r * MAX_GRID_SIZE + c
                    seq[idx] = color + COLOR_OFFSET
        
        return seq
    
    def _augment_colors(
        self,
        demo_inputs: np.ndarray,
        demo_outputs: np.ndarray,
        test_input: np.ndarray,
        test_output: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply random color permutation augmentation."""
        # Create random permutation of colors 0-9
        perm = np.random.permutation(10)
        
        def apply_perm(arr):
            result = arr.copy()
            mask = (arr >= COLOR_OFFSET) & (arr < COLOR_OFFSET + 10)
            result[mask] = perm[arr[mask] - COLOR_OFFSET] + COLOR_OFFSET
            return result
        
        return (
            apply_perm(demo_inputs),
            apply_perm(demo_outputs),
            apply_perm(test_input),
            apply_perm(test_output),
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Convert demo pairs
        train_pairs = sample['train'][:self.max_demos]
        num_demos = len(train_pairs)
        
        # Initialize arrays
        demo_inputs = np.zeros((self.max_demos, MAX_SEQ_LEN), dtype=np.int64)
        demo_outputs = np.zeros((self.max_demos, MAX_SEQ_LEN), dtype=np.int64)
        
        for i, pair in enumerate(train_pairs):
            demo_inputs[i] = self._grid_to_sequence(pair['input'])
            demo_outputs[i] = self._grid_to_sequence(pair['output'])
        
        # Pad remaining demos if needed
        for i in range(num_demos, self.max_demos):
            demo_inputs[i] = demo_inputs[num_demos - 1]
            demo_outputs[i] = demo_outputs[num_demos - 1]
        
        # Convert test
        test_input = self._grid_to_sequence(sample['test_input'])
        test_output = self._grid_to_sequence(sample['test_output'])
        
        # Create target with proper masking
        target = np.full(MAX_SEQ_LEN, -100, dtype=np.int64)  # -100 = ignore
        
        # Find valid region in test output
        test_h = len(sample['test_output'])
        test_w = len(sample['test_output'][0]) if test_h > 0 else 0
        
        for r in range(test_h):
            for c in range(test_w):
                idx = r * MAX_GRID_SIZE + c
                target[idx] = test_output[idx]
        
        # Apply augmentation
        if self.augment and random.random() > 0.5:
            demo_inputs, demo_outputs, test_input, target_aug = self._augment_colors(
                demo_inputs, demo_outputs, test_input, test_output
            )
            # Update target with augmented values
            mask = target != -100
            target[mask] = target_aug[mask]
        
        return {
            'demo_inputs': torch.from_numpy(demo_inputs),
            'demo_outputs': torch.from_numpy(demo_outputs),
            'test_input': torch.from_numpy(test_input),
            'test_output': torch.from_numpy(target),
            'task_id': sample['task_id'],
        }


# ============================================================================
# Utility Functions
# ============================================================================

def sequence_to_grid(seq: np.ndarray) -> np.ndarray:
    """Convert sequence back to grid for visualization."""
    seq_2d = seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    grid = np.zeros_like(seq_2d, dtype=np.uint8)
    
    mask = seq_2d >= COLOR_OFFSET
    grid[mask] = seq_2d[mask] - COLOR_OFFSET
    
    return np.clip(grid, 0, 9)


def target_to_grid(target: np.ndarray) -> np.ndarray:
    """Convert target sequence to grid (handling -100 ignore)."""
    target_2d = target.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    grid = np.zeros_like(target_2d, dtype=np.uint8)
    
    valid_mask = (target_2d != -100) & (target_2d >= COLOR_OFFSET)
    grid[valid_mask] = target_2d[valid_mask] - COLOR_OFFSET
    
    return np.clip(grid, 0, 9)


def download_miniarc(target_dir: str = "./Mini-ARC") -> Path:
    """Download Mini-ARC dataset if not present."""
    import subprocess
    
    target_path = Path(target_dir)
    data_path = target_path / "data" / "MiniARC"
    
    if data_path.exists() and list(data_path.glob("*.json")):
        print(f"Mini-ARC already exists at {data_path}")
        return data_path
    
    print("Downloading Mini-ARC dataset...")
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/ksb21ST/Mini-ARC.git",
        str(target_path)
    ], check=True)
    
    print(f"Mini-ARC downloaded to {data_path}")
    return data_path


if __name__ == "__main__":
    # Test the dataset
    data_dir = download_miniarc()
    dataset = MiniARCDataset(str(data_dir), augment=False)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Sequence length: {MAX_SEQ_LEN} tokens")
    print(f"Vocab size: {VOCAB_SIZE}")
    
    # Show a sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  demo_inputs:  {sample['demo_inputs'].shape}")
    print(f"  demo_outputs: {sample['demo_outputs'].shape}")
    print(f"  test_input:   {sample['test_input'].shape}")
    print(f"  test_output:  {sample['test_output'].shape}")
    
    # Visualize
    print(f"\nTask: {sample['task_id']}")
    print(f"Test input grid:\n{sequence_to_grid(sample['test_input'].numpy())}")
    print(f"Test target grid:\n{target_to_grid(sample['test_output'].numpy())}")
