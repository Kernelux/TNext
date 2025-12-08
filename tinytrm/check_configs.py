
import torch
from model import TinyTRM, TRMConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_config(name, **kwargs):
    config = TRMConfig(**kwargs)
    model = TinyTRM(config)
    count = count_parameters(model)
    print(f"Config: {name:<30} | {kwargs} => {count:,} params")

if __name__ == "__main__":
    # Baseline (No Slot Memory)
    check_config("Baseline (No Slot)", use_slot_memory=False)
    
    # Hypothesis 1: Smaller Expansion
    check_config("Expansion=3", use_slot_memory=False, expansion=3.0)
    
    # Hypothesis 2: No Puzzle Embeddings
    check_config("No Puzzle Emb", use_slot_memory=False, puzzle_emb_ndim=0)
    
    # Hypothesis 3: Both
    check_config("Exp=3 + No Puzzle", use_slot_memory=False, expansion=3.0, puzzle_emb_ndim=0)
    
    # Hypothesis 4: 4/3 Expansion (GLU approximation of 4x GeLU)
    # 4 * 2/3 = 2.66
    check_config("Expansion=2.66", use_slot_memory=False, expansion=2.6666)
