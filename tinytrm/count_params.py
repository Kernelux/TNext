
import torch
from model import TinyTRM, TRMConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_stats():
    print("--- Configuration 1: Default (with Slot Memory) ---")
    config = TRMConfig()
    model = TinyTRM(config)
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    
    # improved breakdown
    slot_mem_params = count_parameters(model.inner.slot_memory) if hasattr(model.inner, 'slot_memory') else 0
    trm_params = total_params - slot_mem_params
    print(f"Slot Memory Params: {slot_mem_params:,}")
    print(f"Base TRM Params: {trm_params:,}")

    print("\n--- Configuration 2: Without Slot Memory ---")
    config_no_slot = TRMConfig(use_slot_memory=False)
    model_no_slot = TinyTRM(config_no_slot)
    total_no_slot = count_parameters(model_no_slot)
    print(f"Total Parameters: {total_no_slot:,}")

if __name__ == "__main__":
    print_model_stats()
