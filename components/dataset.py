import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import glob

class ARCDataset(Dataset):
    """
    ARC-AGI dataset loader (lazy loading).
    
    Each task has:
    - train: list of {input: grid, output: grid} demo pairs
    - test: list of {input: grid, output: grid} test pairs
    
    Files are loaded on-the-fly to avoid slow startup.
    """
    
    def __init__(self, data_dir: str, split: str = "training", max_grid_size: int = 30):
        self.max_grid_size = max_grid_size
        self.task_dir = Path(data_dir) / split
        
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
    
    def pad_grid(self, grid, size):
        """Pad grid to fixed size, return grid and mask."""
        h, w = len(grid), len(grid[0]) if grid else 0
        padded = torch.zeros(size, size, dtype=torch.long)
        mask = torch.ones(size, size, dtype=torch.bool)  # True = padding
        
        for i in range(min(h, size)):
            for j in range(min(w, size)):
                padded[i, j] = grid[i][j]
                mask[i, j] = False
        
        return padded, mask, h, w
    
    def __getitem__(self, idx):
        # Get file and test indices from our index
        file_idx, test_idx = self.index[idx]
        task = self._load_task(file_idx)
        
        size = self.max_grid_size
        
        # Encode demo pairs (from "train" in the JSON)
        demo_inputs = []
        demo_outputs = []
        demo_masks = []
        
        for pair in task["train"]:
            inp, inp_mask, _, _ = self.pad_grid(pair["input"], size)
            out, out_mask, _, _ = self.pad_grid(pair["output"], size)
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
        test_input, test_mask, _, _ = self.pad_grid(test_pair["input"], size)
        test_output, output_mask, out_h, out_w = self.pad_grid(test_pair["output"], size)
        
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
        }
