import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import glob
import numpy as np
from typing import Tuple, Callable, Optional

# Constants matching TRM (extended with role markers)
IGNORE_LABEL = -100
PAD_TOKEN = 0
EOS_TOKEN = 1
INPUT_MARKER = 2   # Marks start of input grid - makes input/output pairing explicit
OUTPUT_MARKER = 3  # Marks start of output grid
COLOR_OFFSET = 4   # Colors 0-9 become tokens 4-13
VOCAB_SIZE = 14    # PAD, EOS, INPUT, OUTPUT, colors 0-9
MAX_GRID_SIZE = 30
MAX_SEQ_LEN = MAX_GRID_SIZE * MAX_GRID_SIZE  # 900 tokens per grid


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


def grid_to_sequence(grid: np.ndarray, add_eos: bool = True) -> np.ndarray:
    """
    Convert a 2D grid to a 1D sequence with EOS markers (TRM-style).
    
    Format: Flatten grid, add EOS markers at row/column boundaries.
    - PAD_TOKEN (0): padding
    - EOS_TOKEN (1): end-of-row/grid marker
    - INPUT_MARKER (2): marks start of input grid (added by caller)
    - OUTPUT_MARKER (3): marks start of output grid (added by caller)
    - Colors 0-9 → tokens 4-13
    
    The EOS token marks where the actual content ends:
    - After each row (at column = grid_width)
    - After the last row (at row = grid_height)
    
    Args:
        grid: [H, W] numpy array with values 0-9
        add_eos: Whether to add EOS markers
        
    Returns:
        [MAX_SEQ_LEN] numpy array with token values
    """
    h, w = grid.shape
    
    # Create output sequence padded with PAD_TOKEN
    seq = np.full(MAX_SEQ_LEN, PAD_TOKEN, dtype=np.int64)
    
    # Fill in the grid content with color offset
    # We store row by row, marking the end of each row with EOS
    idx = 0
    for row in range(h):
        for col in range(w):
            if idx < MAX_SEQ_LEN:
                seq[idx] = grid[row, col] + COLOR_OFFSET
                idx += 1
        # Add EOS at end of row (marks column boundary)
        if add_eos and idx < MAX_SEQ_LEN:
            seq[idx] = EOS_TOKEN
            idx += 1
    
    return seq


def grids_to_aligned_sequences(
    inp_grid: np.ndarray, 
    out_grid: np.ndarray,
    do_translation: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert input and output grids to aligned sequences (TRM-style).
    
    CRITICAL: Both grids are padded to 30x30 using the SAME top-left offset.
    This ensures position i in input corresponds to the same spatial location
    as position i in output - essential for position-wise prediction!
    
    Format per grid (flattened 30x30 = 900 positions):
    - PAD_TOKEN (0): padding (areas outside both grids)
    - EOS_TOKEN (1): marks actual grid boundaries
    - Colors 0-9 → tokens 4-13
    
    Args:
        inp_grid: [H_in, W_in] input grid with values 0-9
        out_grid: [H_out, W_out] output grid with values 0-9
        do_translation: If True, apply random translation (training augmentation)
        rng: Random generator for translation offset
        
    Returns:
        inp_seq: [MAX_SEQ_LEN] input sequence (900 tokens)
        out_seq: [MAX_SEQ_LEN] output sequence (900 tokens) - use as labels
    """
    h_in, w_in = inp_grid.shape
    h_out, w_out = out_grid.shape
    
    # Use the maximum dimensions to ensure both fit
    max_h = max(h_in, h_out)
    max_w = max(w_in, w_out)
    
    # Compute padding offset (same for both!)
    if do_translation and rng is not None:
        pad_r = rng.integers(0, MAX_GRID_SIZE - max_h + 1)
        pad_c = rng.integers(0, MAX_GRID_SIZE - max_w + 1)
    else:
        pad_r = pad_c = 0
    
    def pad_and_flatten(grid: np.ndarray) -> np.ndarray:
        """Pad a single grid to 30x30 with EOS markers, then flatten."""
        h, w = grid.shape
        
        # Create 30x30 padded grid
        padded = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.int64)
        
        # Place grid content with color offset
        padded[pad_r:pad_r+h, pad_c:pad_c+w] = grid + COLOR_OFFSET
        
        # Add EOS markers at grid boundaries (TRM-style)
        # EOS after each row's last column
        eos_row = pad_r + h
        eos_col = pad_c + w
        
        if eos_row < MAX_GRID_SIZE:
            padded[eos_row, pad_c:eos_col] = EOS_TOKEN
        if eos_col < MAX_GRID_SIZE:
            padded[pad_r:eos_row, eos_col] = EOS_TOKEN
        
        return padded.flatten()
    
    inp_seq = pad_and_flatten(inp_grid)
    out_seq = pad_and_flatten(out_grid)
    
    return inp_seq, out_seq


def grid_to_input_sequence(grid: np.ndarray) -> np.ndarray:
    """
    Convert a grid to an INPUT sequence (with INPUT_MARKER prefix).
    
    Format: [INPUT_MARKER] [grid content with EOS markers]
    
    Args:
        grid: [H, W] numpy array with values 0-9
        
    Returns:
        [MAX_SEQ_LEN] numpy array with token values
    """
    seq = np.full(MAX_SEQ_LEN, PAD_TOKEN, dtype=np.int64)
    seq[0] = INPUT_MARKER
    
    h, w = grid.shape
    idx = 1  # Start after INPUT_MARKER
    
    for row in range(h):
        for col in range(w):
            if idx < MAX_SEQ_LEN:
                seq[idx] = grid[row, col] + COLOR_OFFSET
                idx += 1
        # Add EOS at end of row
        if idx < MAX_SEQ_LEN:
            seq[idx] = EOS_TOKEN
            idx += 1
    
    return seq


def grid_to_output_sequence(grid: np.ndarray) -> np.ndarray:
    """
    Convert a grid to an OUTPUT sequence (with OUTPUT_MARKER prefix).
    
    Format: [OUTPUT_MARKER] [grid content with EOS markers]
    
    Args:
        grid: [H, W] numpy array with values 0-9
        
    Returns:
        [MAX_SEQ_LEN] numpy array with token values
    """
    seq = np.full(MAX_SEQ_LEN, PAD_TOKEN, dtype=np.int64)
    seq[0] = OUTPUT_MARKER
    
    h, w = grid.shape
    idx = 1  # Start after OUTPUT_MARKER
    
    for row in range(h):
        for col in range(w):
            if idx < MAX_SEQ_LEN:
                seq[idx] = grid[row, col] + COLOR_OFFSET
                idx += 1
        # Add EOS at end of row
        if idx < MAX_SEQ_LEN:
            seq[idx] = EOS_TOKEN
            idx += 1
    
    return seq


def sequence_to_grid(seq: np.ndarray) -> np.ndarray:
    """
    Convert a sequence back to a 2D grid.
    
    Parses EOS tokens to determine row boundaries.
    Skips INPUT_MARKER and OUTPUT_MARKER tokens if present.
    
    Args:
        seq: [S] numpy array with token values (PAD=0, EOS=1, INPUT=2, OUTPUT=3, colors=4-13)
        
    Returns:
        [H, W] numpy array with values 0-9
    """
    rows = []
    current_row = []
    
    for token in seq:
        if token == PAD_TOKEN:
            # End of content
            break
        elif token == EOS_TOKEN:
            # End of row
            if current_row:
                rows.append(current_row)
                current_row = []
        elif token == INPUT_MARKER or token == OUTPUT_MARKER:
            # Skip role markers
            continue
        else:
            # Color token (4-13 → 0-9)
            current_row.append(token - COLOR_OFFSET)
    
    # Handle case where last row doesn't end with EOS
    if current_row:
        rows.append(current_row)
    
    if not rows:
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Pad rows to same width
    max_width = max(len(row) for row in rows)
    grid = np.zeros((len(rows), max_width), dtype=np.uint8)
    for i, row in enumerate(rows):
        grid[i, :len(row)] = row
    
    return grid


class ARCDataset(Dataset):
    """
    ARC-AGI dataset loader with data augmentation (TRM-style sequence format).
    
    Data format:
    - Grids are flattened to sequences with EOS markers
    - Vocab: PAD=0, EOS=1, INPUT=2, OUTPUT=3, colors 0-9 → tokens 4-13
    - Each grid becomes a sequence of MAX_SEQ_LEN (900) tokens
    - Role markers (INPUT/OUTPUT) make input/output pairing explicit
    
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
    
    def _grid_to_np(self, grid_list) -> np.ndarray:
        """Convert grid list to numpy array."""
        return np.array(grid_list, dtype=np.uint8)
    
    def _create_label_sequence(self, grid: np.ndarray) -> np.ndarray:
        """
        Create label sequence for a grid output.
        
        Uses OUTPUT_MARKER prefix and IGNORE_LABEL for padding positions
        so the loss only applies to actual content.
        
        Format: [OUTPUT_MARKER] [grid content with EOS] [IGNORE_LABEL padding...]
        """
        seq = grid_to_output_sequence(grid)  # [OUTPUT_MARKER] [grid...]
        
        # Replace PAD tokens with IGNORE_LABEL
        # Also mark OUTPUT_MARKER as IGNORE_LABEL (we don't predict it)
        labels = seq.copy()
        labels[0] = IGNORE_LABEL  # Don't predict the OUTPUT_MARKER
        labels = np.where(labels == PAD_TOKEN, IGNORE_LABEL, labels)
        
        return labels
    
    def __getitem__(self, idx):
        # Get file and test indices from our index
        file_idx, test_idx = self.index[idx]
        task = self._load_task(file_idx)
        
        # Create augmentation for this sample (if training)
        if self.augment:
            aug_fn, transform_id, color_mapping = create_augmentation_fn(self.rng)
            do_translation = True  # Random translation for training
        else:
            aug_fn = lambda x: x
            transform_id = 0
            color_mapping = np.arange(10, dtype=np.uint8)
            do_translation = False
        
        # === ALIGNED SEQUENCE ENCODING (TRM-style) ===
        # CRITICAL: Input and output must be padded to same 30x30 space
        # using the SAME offset so position i maps to same spatial location!
        
        # Encode demo pairs with aligned sequences
        demo_inputs = []
        demo_outputs = []
        
        for pair in task["train"]:
            inp_np = aug_fn(self._grid_to_np(pair["input"]))
            out_np = aug_fn(self._grid_to_np(pair["output"]))
            
            # Create ALIGNED sequences (same padding for both)
            inp_seq, out_seq = grids_to_aligned_sequences(
                inp_np, out_np,
                do_translation=do_translation,
                rng=self.rng if do_translation else None,
            )
            
            demo_inputs.append(torch.from_numpy(inp_seq))
            demo_outputs.append(torch.from_numpy(out_seq))
        
        # Pad to fixed number of demos (3)
        if demo_inputs:
            template_inp = demo_inputs[0]
            template_out = demo_outputs[0]
        else:
            template_inp = torch.zeros(MAX_SEQ_LEN, dtype=torch.long)
            template_out = torch.zeros(MAX_SEQ_LEN, dtype=torch.long)
            
        while len(demo_inputs) < 3:
            demo_inputs.append(template_inp.clone())
            demo_outputs.append(template_out.clone())
        
        demo_inputs = torch.stack(demo_inputs[:3])   # [3, S]
        demo_outputs = torch.stack(demo_outputs[:3]) # [3, S]
        
        # Encode test pair with aligned sequences
        test_pair = task["test"][test_idx]
        test_inp_np = aug_fn(self._grid_to_np(test_pair["input"]))
        test_out_np = aug_fn(self._grid_to_np(test_pair["output"]))
        
        # Create ALIGNED sequences for test (same padding for input and output!)
        test_input_seq, test_output_seq = grids_to_aligned_sequences(
            test_inp_np, test_out_np,
            do_translation=do_translation,
            rng=self.rng if do_translation else None,
        )
        
        test_input = torch.from_numpy(test_input_seq)
        
        # Create labels: mark PAD positions as IGNORE_LABEL
        # Note: We predict at ALL positions where output has content (including EOS)
        test_output = test_output_seq.copy()
        test_output = np.where(test_output == PAD_TOKEN, IGNORE_LABEL, test_output)
        test_output = torch.from_numpy(test_output)
        
        # Store original grid dimensions for evaluation/visualization
        out_h, out_w = test_out_np.shape
        
        # Task ID from filename
        task_id = self.task_files[file_idx].stem
        
        return {
            "demo_inputs": demo_inputs,    # [3, S] aligned sequences
            "demo_outputs": demo_outputs,  # [3, S] aligned sequences  
            "test_input": test_input,      # [S] aligned sequence
            "test_output": test_output,    # [S] labels (IGNORE_LABEL for pad)
            "output_size": torch.tensor([out_h, out_w]),  # For evaluation
            "task_id": task_id,
            # Store augmentation info for inverse transform during evaluation
            "transform_id": transform_id,
            "color_mapping": torch.from_numpy(color_mapping),
        }


class ARCDatasetNoAug(ARCDataset):
    """ARCDataset with augmentation disabled (for evaluation)."""
    
    def __init__(self, data_dir: str, split: str = "training", max_grid_size: int = 30):
        super().__init__(data_dir, split, max_grid_size, augment=False)
