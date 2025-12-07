"""
Configuration adjustment for runpod preset to improve task accuracy.

This addresses the issue where the model achieves good cell accuracy (76.8%) 
but poor task accuracy (0.001), which suggests the losses may be competing 
with each other or the model is getting trapped in local optima.
"""

from components.config import TrainingConfig, FeatureFlags

def get_runpod_config():
    """
    Get a more balanced training configuration for the runpod preset.
    
    The key insight is that when cell accuracy is high (76.8%) but task accuracy 
    is low (0.001), the model is learning to predict individual cells well but 
    struggling to get the entire task correct. This suggests that:
    
    1. The auxiliary losses might be competing with the task loss
    2. The step efficiency loss might be driving the model to use fewer steps too early
    3. The ponder loss might be making the model stop too early
    4. The model needs to focus more on task-level accuracy vs cell-level accuracy
    """
    # Use the runpod feature flags but with adjusted loss weights
    features = FeatureFlags(
        use_diversity_loss=True,     # Keep diversity to prevent slot collapse
        use_ponder_loss=True,        # Keep ponder for efficiency but with lower weight
        use_explicit_q_head=True,    # Keep Q-head for halting prediction
        use_answer_feedback=True,    # Keep answer feedback mechanism
    )
    
    config = TrainingConfig(
        # Temperature for Gumbel-Softmax routing
        tau_start=1.0,       # Initial temperature (soft routing)
        tau_min=0.2,         # Lower temperature for sharper routing later in training
        anneal_rate=0.0005,  # Faster annealing to encourage harder decisions
        
        # Adjusted loss weights for better task completion
        lambda_diversity=0.005,      # Reduced diversity loss to focus more on task
        lambda_ponder=0.02,          # Reduced ponder loss to allow more passes when needed
        lambda_q_head=0.05,          # Reduced Q-head loss to focus on primary task
        lambda_step_efficiency=0.02, # Reduced step efficiency to allow more exploration
        
        # Importance threshold
        write_threshold=0.5,
        
        # Hybrid routing
        alpha_learned=1.0,  # 1.0 = pure learned routing
        
        # Compute budget (model starts at max, learns to reduce)
        max_passes=6,              # Reduce max passes slightly
        max_recurrent_steps=3,     # Reduce max recurrent steps slightly
        
        # EMA for training stability
        use_ema=True,
        ema_decay=0.999,
        
        # Use the runpod-specific feature flags
        features=features
    )
    
    return config

if __name__ == "__main__":
    # Example usage
    config = get_runpod_config()
    print("Runpod configuration loaded:")
    print(f"lambda_diversity: {config.lambda_diversity}")
    print(f"lambda_ponder: {config.lambda_ponder}")
    print(f"lambda_q_head: {config.lambda_q_head}")
    print(f"lambda_step_efficiency: {config.lambda_step_efficiency}")
    print(f"tau_min: {config.tau_min}")
    print(f"tau_start: {config.tau_start}")
    print(f"features: {config.features.describe()}")