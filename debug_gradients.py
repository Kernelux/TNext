"""
Gradient Debugging for RecursiveRefinementModel
================================================

Tests gradient flow across different scenarios:
1. Single pass vs multiple passes
2. Early halting vs full passes
3. Deep supervision enabled/disabled
4. Answer feedback gate usage
5. Layer-level iterations
6. Confidence estimator gradients

Usage:
    python debug_gradients.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from components import RecursiveRefinementModel
from components.config import TrainingConfig, FeatureFlags, FEATURE_PRESETS
from components.dataset import ARCDataset, VOCAB_SIZE
from components.losses import compute_total_loss
from components.optimizers import AdamAtan2


def get_gradient_stats(model):
    """Collect gradient statistics grouped by component."""
    stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is None:
            stats[name] = {'has_grad': False, 'abs_max': 0, 'abs_mean': 0, 'shape': param.shape}
            continue
        
        g = param.grad
        stats[name] = {
            'has_grad': True,
            'abs_max': g.abs().max().item(),
            'abs_mean': g.abs().mean().item(),
            'has_nan': torch.isnan(g).any().item(),
            'has_inf': torch.isinf(g).any().item(),
            'shape': tuple(param.shape),
        }
    
    return stats


def group_gradients(stats):
    """Group gradient stats by component type."""
    groups = {
        'token_embed': [],
        'segment_embed': [],
        'prev_answer_embed': [],
        'answer_feedback_gate': [],
        'output_proj': [],
        'rotary_emb': [],
        'confidence_estimator': [],
        'layers.memory': [],
        'layers.compute': [],
        'layers.iteration_feedback': [],
        'layers.cache_self_attn': [],
        'slot_embeddings': [],
        'layer_id_embeddings': [],
        'other': [],
    }
    
    for name, stat in stats.items():
        categorized = False
        for group_key in groups.keys():
            if group_key != 'other' and group_key in name:
                groups[group_key].append((name, stat))
                categorized = True
                break
        if not categorized:
            groups['other'].append((name, stat))
    
    return groups


def print_gradient_summary(groups, title="Gradient Summary", verbose=False):
    """Print gradient summary by group."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    for group_name, params in groups.items():
        if not params:
            continue
        
        has_grad = [s for _, s in params if s['has_grad']]
        no_grad = [(n, s) for n, s in params if not s['has_grad']]
        
        if has_grad:
            max_grad = max(s['abs_max'] for s in has_grad)
            mean_grad = sum(s['abs_mean'] for s in has_grad) / len(has_grad)
            has_nan = any(s.get('has_nan', False) for s in has_grad)
            has_inf = any(s.get('has_inf', False) for s in has_grad)
            
            flag = "🔥" if max_grad > 100 else ("⚠️" if max_grad > 10 else "✓")
            nan_flag = " [NaN!]" if has_nan else ""
            inf_flag = " [Inf!]" if has_inf else ""
            
            print(f"  {flag} {group_name}: {len(has_grad)} params with grad, "
                  f"max={max_grad:.2e}, mean={mean_grad:.2e}{nan_flag}{inf_flag}")
        
        if no_grad:
            print(f"  ❌ {group_name}: {len(no_grad)} params WITHOUT grad")
            if verbose:
                for name, stat in no_grad[:5]:  # Show first 5
                    print(f"      - {name} {stat['shape']}")
                if len(no_grad) > 5:
                    print(f"      ... and {len(no_grad) - 5} more")


def test_scenario(
    model,
    demo_inputs,
    demo_outputs,
    test_input,
    test_output,
    config,
    device,
    scenario_name,
    step=0,
):
    """Run a single test scenario and report gradients."""
    print(f"\n{'#'*70}")
    print(f"# SCENARIO: {scenario_name}")
    print(f"{'#'*70}")
    
    model.train()
    model.zero_grad()
    
    # Forward pass
    logits, cache, aux = model(
        demo_inputs, demo_outputs, test_input,
        config=config,
        step=step,
        return_aux=True,
    )
    
    # Print forward stats
    print(f"\n[Forward Pass Stats]")
    print(f"  Passes run: {aux.get('passes_run', 'N/A')}")
    print(f"  Halted early: {aux.get('halted_early', False)}")
    print(f"  Pass logits collected: {len(aux.get('pass_logits', []))}")
    print(f"  Pass confidences: {len(aux.get('pass_confidences', []))}")
    
    if aux.get('pass_confidences'):
        conf_str = ", ".join([f"{c.mean().item():.3f}" for c in aux['pass_confidences']])
        print(f"  Confidence values: [{conf_str}]")
    
    print(f"  Layer iterations: {aux.get('layer_iterations', [])}")
    print(f"  Answer feedback count: {aux.get('answer_feedback_count', 0)}")
    
    if aux.get('answer_feedback_count', 0) > 0:
        avg_fb = aux['answer_feedback_sum'] / aux['answer_feedback_count']
        print(f"  Answer feedback gate mean: {avg_fb:.4f}")
    
    # Compute loss
    loss, metrics = compute_total_loss(
        model, logits, test_output, aux, config, step, device
    )
    
    print(f"\n[Loss]")
    print(f"  Total: {loss.detach().item():.4f}")
    print(f"  Task: {metrics.get('loss_task', 0):.4f}")
    print(f"  Confidence calibration: {metrics.get('loss_confidence_calibration', 0):.4f}")
    print(f"  Halt prediction: {metrics.get('loss_halt_prediction', 0):.4f}")
    print(f"  Ponder adaptive: {metrics.get('loss_ponder_adaptive', 0):.4f}")
    
    # Backward pass
    loss.backward()
    
    # Collect and print gradient stats
    stats = get_gradient_stats(model)
    groups = group_gradients(stats)
    print_gradient_summary(groups, title="Gradient Flow", verbose=True)
    
    # Specific checks
    print(f"\n[Critical Checks]")
    
    # Check answer feedback gate
    fb_gate_grads = [s for n, s in stats.items() if 'answer_feedback_gate' in n and s['has_grad']]
    if fb_gate_grads:
        max_fb = max(s['abs_max'] for s in fb_gate_grads)
        print(f"  ✓ Answer feedback gate has gradients (max={max_fb:.2e})")
    else:
        if aux.get('answer_feedback_count', 0) == 0:
            print(f"  ⚠️ Answer feedback gate: NO gradients (expected - only {aux.get('passes_run', 1)} pass ran)")
        else:
            print(f"  ❌ Answer feedback gate: NO gradients despite being used!")
    
    # Check confidence estimator
    conf_grads = [s for n, s in stats.items() if 'confidence' in n and s['has_grad']]
    if conf_grads:
        max_conf = max(s['abs_max'] for s in conf_grads)
        print(f"  ✓ Confidence estimator has gradients (max={max_conf:.2e})")
    else:
        print(f"  ❌ Confidence estimator: NO gradients!")
    
    # Check memory components
    mem_grads = [s for n, s in stats.items() if 'memory' in n and s['has_grad']]
    if mem_grads:
        max_mem = max(s['abs_max'] for s in mem_grads)
        print(f"  ✓ Memory controller has gradients (max={max_mem:.2e})")
    else:
        print(f"  ❌ Memory controller: NO gradients!")
    
    # Check cache self-attention (memory consolidation)
    cache_attn_grads = [s for n, s in stats.items() if 'cache_self_attn' in n and s['has_grad']]
    if cache_attn_grads:
        max_cache_attn = max(s['abs_max'] for s in cache_attn_grads)
        print(f"  ✓ Cache self-attention has gradients (max={max_cache_attn:.2e})")
    else:
        print(f"  ❌ Cache self-attention: NO gradients!")
    
    # Return metrics for comparison
    return {
        'passes_run': aux.get('passes_run', 0),
        'halted_early': aux.get('halted_early', False),
        'answer_feedback_used': aux.get('answer_feedback_count', 0) > 0,
        'loss': loss.item(),
        'num_params_with_grad': sum(1 for s in stats.values() if s['has_grad']),
        'max_gradient': max((s['abs_max'] for s in stats.values() if s['has_grad']), default=0),
    }


def run_all_scenarios():
    """Run all gradient debugging scenarios."""
    print("="*70)
    print("RECURSIVE REFINEMENT MODEL - GRADIENT DEBUGGING")
    print("="*70)
    
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Load real data
    data_dir = Path("ARC-AGI-2/data")
    if not data_dir.exists():
        print("ERROR: ARC-AGI-2 data not found!")
        return
    
    dataset = ARCDataset(str(data_dir), split="training", max_grid_size=30, augment=False)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    
    demo_inputs = batch["demo_inputs"].to(device)
    demo_outputs = batch["demo_outputs"].to(device)
    test_input = batch["test_input"].to(device)
    test_output = batch["test_output"].to(device)
    
    print(f"\nData shapes:")
    print(f"  demo_inputs: {demo_inputs.shape}")
    print(f"  test_input: {test_input.shape}")
    print(f"  test_output: {test_output.shape}")
    
    # Create model
    model = RecursiveRefinementModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        d_cache=48,
        num_layers=4,
        num_slots=16,
        num_heads=2,
        max_seq_len=900,
        max_internal_iterations=5,
        max_passes=5,
        dropout=0.0,
        confidence_threshold=0.8,
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    results = {}
    
    # ========================================
    # SCENARIO 1: Force single pass (no halting opportunity)
    # ========================================
    features_1 = FeatureFlags(
        use_deep_supervision=False,
        use_answer_feedback=True,
        halt_exploration_prob=0.0,  # Never explore
        use_ponder_cost=False,
    )
    config_1 = TrainingConfig(features=features_1, max_passes=1, max_recurrent_steps=1)
    
    results['single_pass'] = test_scenario(
        model, demo_inputs, demo_outputs, test_input, test_output,
        config_1, device, "Single Pass (max_passes=1)"
    )
    
    # ========================================
    # SCENARIO 2: Multiple passes, high exploration (force full passes)
    # ========================================
    features_2 = FeatureFlags(
        use_deep_supervision=True,
        use_answer_feedback=True,
        halt_exploration_prob=1.0,  # Always explore (never halt)
        use_ponder_cost=True,
    )
    config_2 = TrainingConfig(features=features_2, max_passes=3, max_recurrent_steps=3)
    
    results['full_passes'] = test_scenario(
        model, demo_inputs, demo_outputs, test_input, test_output,
        config_2, device, "Full Passes (exploration=1.0, max_passes=3)"
    )
    
    # ========================================
    # SCENARIO 3: Normal training config (ε-greedy halting)
    # ========================================
    features_3 = FeatureFlags(
        use_deep_supervision=True,
        use_answer_feedback=True,
        halt_exploration_prob=0.3,  # 30% exploration
        use_ponder_cost=True,
    )
    config_3 = TrainingConfig(features=features_3, max_passes=5, max_recurrent_steps=5)
    
    results['normal'] = test_scenario(
        model, demo_inputs, demo_outputs, test_input, test_output,
        config_3, device, "Normal Training (exploration=0.3, max_passes=5)"
    )
    
    # ========================================
    # SCENARIO 4: No halting during warmup
    # ========================================
    # Simulate warmup by using high exploration
    features_4 = FeatureFlags(
        use_deep_supervision=True,
        use_answer_feedback=True,
        halt_exploration_prob=1.0,  # Force all passes during warmup
        use_ponder_cost=False,  # Don't penalize ponder during warmup
    )
    config_4 = TrainingConfig(features=features_4, max_passes=5, max_recurrent_steps=5)
    
    results['warmup'] = test_scenario(
        model, demo_inputs, demo_outputs, test_input, test_output,
        config_4, device, "Warmup Mode (exploration=1.0, no ponder cost)"
    )
    
    # ========================================
    # SCENARIO 5: Deep supervision disabled
    # ========================================
    features_5 = FeatureFlags(
        use_deep_supervision=False,
        use_answer_feedback=True,
        halt_exploration_prob=1.0,
        use_ponder_cost=True,
    )
    config_5 = TrainingConfig(features=features_5, max_passes=3, max_recurrent_steps=3)
    
    results['no_deep_supervision'] = test_scenario(
        model, demo_inputs, demo_outputs, test_input, test_output,
        config_5, device, "No Deep Supervision (final pass only)"
    )
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    print(f"\n{'Scenario':<30} {'Passes':>8} {'Halted':>8} {'FB Used':>8} {'Loss':>10} {'Grad Params':>12} {'Max Grad':>12}")
    print("-"*90)
    
    for name, r in results.items():
        print(f"{name:<30} {r['passes_run']:>8} {str(r['halted_early']):>8} "
              f"{str(r['answer_feedback_used']):>8} {r['loss']:>10.4f} "
              f"{r['num_params_with_grad']:>12} {r['max_gradient']:>12.2e}")
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Check if answer feedback gets gradients when multiple passes run
    if results['full_passes']['answer_feedback_used']:
        print("✓ Answer feedback gate IS used when multiple passes run")
    else:
        print("⚠️ Answer feedback gate NOT used even with multiple passes - check implementation!")
    
    if results['single_pass']['passes_run'] == 1 and not results['single_pass']['answer_feedback_used']:
        print("✓ Answer feedback correctly skipped on single pass")
    
    if results['full_passes']['num_params_with_grad'] > results['single_pass']['num_params_with_grad']:
        diff = results['full_passes']['num_params_with_grad'] - results['single_pass']['num_params_with_grad']
        print(f"✓ More parameters get gradients with multiple passes (+{diff})")
    
    # Check gradient magnitudes
    max_grads = [r['max_gradient'] for r in results.values()]
    if max(max_grads) > 100:
        print("⚠️ Some scenarios have large gradients (>100) - consider gradient clipping")
    elif max(max_grads) < 1e-6:
        print("⚠️ Gradients are very small - possible vanishing gradient issue")
    else:
        print("✓ Gradient magnitudes look healthy")


def debug_answer_feedback_specifically():
    """Specifically debug why answer_feedback_gate stays at 0.5."""
    print("\n" + "="*70)
    print("ANSWER FEEDBACK GATE - DETAILED DEBUG")
    print("="*70)
    
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create small model for debugging
    model = RecursiveRefinementModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        d_cache=48,
        num_layers=2,
        num_slots=8,
        num_heads=2,
        max_seq_len=100,
        max_internal_iterations=2,
        max_passes=5,
        dropout=0.0,
        confidence_threshold=0.99,  # Very high - almost never confident
    ).to(device)
    
    # Simple test data
    B, S = 2, 50
    demo_inputs = torch.randint(0, VOCAB_SIZE, (B, 1, S), device=device)
    demo_outputs = torch.randint(0, VOCAB_SIZE, (B, 1, S), device=device)
    test_input = torch.randint(0, VOCAB_SIZE, (B, S), device=device)
    test_output = torch.randint(0, VOCAB_SIZE, (B, S), device=device)
    
    # Force ALL passes with exploration=1.0
    features = FeatureFlags(
        use_deep_supervision=True,
        use_answer_feedback=True,
        halt_exploration_prob=1.0,  # ALWAYS continue
        use_ponder_cost=False,
    )
    config = TrainingConfig(features=features, max_passes=5, max_recurrent_steps=2)
    
    model.train()
    model.zero_grad()
    
    # Forward pass
    logits, cache, aux = model(
        demo_inputs, demo_outputs, test_input,
        config=config,
        step=0,
        return_aux=True,
    )
    
    print(f"\n[Forward Results]")
    print(f"  Passes run: {aux.get('passes_run', 'N/A')}")
    print(f"  Answer feedback count: {aux.get('answer_feedback_count', 0)}")
    
    if aux.get('answer_feedback_count', 0) > 0:
        avg_fb = aux['answer_feedback_sum'] / aux['answer_feedback_count']
        print(f"  Answer feedback gate mean: {avg_fb:.4f}")
        
        # Check initial weights
        print(f"\n[Answer Feedback Gate Weights]")
        gate = model.answer_feedback_gate
        print(f"  Weight shape: {gate.weight.shape}")
        print(f"  Weight mean: {gate.weight.mean().item():.4f}")
        print(f"  Weight std: {gate.weight.std().item():.4f}")
        print(f"  Bias: {gate.bias.mean().item():.4f}")
        
        # The issue: if bias=0 and input is symmetric, sigmoid(0) = 0.5
        # Let's check if the gate output is actually learning
    else:
        print(f"\n  ❌ Answer feedback was NEVER used!")
        print(f"     This means pass_idx > 0 never happened OR use_answer_feedback=False")
        print(f"     Config: use_answer_feedback={features.use_answer_feedback}")
        print(f"     Expected passes: {config.max_passes}")
    
    # Compute loss and backward
    loss, _ = compute_total_loss(model, logits, test_output, aux, config, 0, device)
    loss.backward()
    
    # Check gate gradients
    print(f"\n[Answer Feedback Gate Gradients]")
    if model.answer_feedback_gate.weight.grad is not None:
        print(f"  Weight grad max: {model.answer_feedback_gate.weight.grad.abs().max().item():.2e}")
        print(f"  Weight grad mean: {model.answer_feedback_gate.weight.grad.abs().mean().item():.2e}")
        print(f"  Bias grad: {model.answer_feedback_gate.bias.grad.abs().max().item():.2e}")
    else:
        print(f"  ❌ NO gradients on answer feedback gate!")
    
    # Check prev_answer_embed gradients
    print(f"\n[Previous Answer Embedding Gradients]")
    if model.prev_answer_embed.weight.grad is not None:
        print(f"  Weight grad max: {model.prev_answer_embed.weight.grad.abs().max().item():.2e}")
        print(f"  Weight grad mean: {model.prev_answer_embed.weight.grad.abs().mean().item():.2e}")
    else:
        print(f"  ❌ NO gradients on prev_answer_embed!")


def test_training_steps():
    """Test multiple training steps like train_recursive.py does."""
    print("\n" + "="*70)
    print("FULL TRAINING STEP TEST")
    print("="*70)
    
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Match fast_full config
    model = RecursiveRefinementModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        d_cache=48,
        num_layers=4,
        num_slots=16,
        num_heads=2,
        max_seq_len=900,
        max_internal_iterations=5,
        max_passes=5,
        dropout=0.0,
        confidence_threshold=0.8,
    ).to(device)
    
    # Load real data
    data_dir = Path("ARC-AGI-2/data")
    dataset = ARCDataset(str(data_dir), split="training", max_grid_size=30, augment=False)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    
    demo_inputs = batch["demo_inputs"].to(device)
    demo_outputs = batch["demo_outputs"].to(device)
    test_input = batch["test_input"].to(device)
    test_output = batch["test_output"].to(device)
    
    # Optimizer
    optimizer = AdamAtan2(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    
    # Config matching fast_full
    features = FeatureFlags(
        use_deep_supervision=True,
        use_answer_feedback=True,
        halt_exploration_prob=0.3,
        use_ponder_cost=True,
    )
    config = TrainingConfig(
        features=features,
        max_passes=5,
        max_recurrent_steps=5,
    )
    
    print(f"\nRunning 5 training steps...")
    print("-"*70)
    
    for step in range(5):
        optimizer.zero_grad()
        
        logits, cache, aux = model(
            demo_inputs, demo_outputs, test_input,
            config=config,
            step=step,
            return_aux=True,
        )
        
        loss, metrics = compute_total_loss(
            model, logits, test_output, aux, config, step, device
        )
        
        loss.backward()
        optimizer.step()
        
        passes = aux.get('passes_run', 1)
        halted = aux.get('halted_early', False)
        fb_count = aux.get('answer_feedback_count', 0)
        fb_mean = aux['answer_feedback_sum'] / fb_count if fb_count > 0 else 0
        layer_iters = aux.get('layer_iterations', [])
        avg_layer_iter = sum(layer_iters) / len(layer_iters) if layer_iters else 0
        
        print(f"Step {step+1}: loss={loss.item():.4f}, passes={passes}, "
              f"halted={halted}, fb_count={fb_count}, fb_mean={fb_mean:.3f}, "
              f"avg_layer_iter={avg_layer_iter:.1f}")
        
        # Clean up
        del logits, cache, loss
        for key in list(aux.keys()):
            if isinstance(aux[key], list):
                aux[key].clear()
        aux.clear()
    
    print("\n✓ Training steps completed successfully!")


def analyze_layer_iterations():
    """Analyze how layer iterations affect gradient flow."""
    print("\n" + "="*70)
    print("LAYER ITERATION GRADIENT ANALYSIS")
    print("="*70)
    
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = RecursiveRefinementModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        d_cache=48,
        num_layers=2,
        num_slots=8,
        num_heads=2,
        max_seq_len=100,
        max_internal_iterations=5,
        max_passes=3,
        dropout=0.0,
        confidence_threshold=0.99,  # High threshold to ensure full iterations
    ).to(device)
    
    B, S = 2, 50
    demo_inputs = torch.randint(0, VOCAB_SIZE, (B, 1, S), device=device)
    demo_outputs = torch.randint(0, VOCAB_SIZE, (B, 1, S), device=device)
    test_input = torch.randint(0, VOCAB_SIZE, (B, S), device=device)
    test_output = torch.randint(0, VOCAB_SIZE, (B, S), device=device)
    
    # Test different max_iterations settings
    for max_iter in [1, 3, 5]:
        print(f"\n--- max_internal_iterations = {max_iter} ---")
        
        features = FeatureFlags(
            use_deep_supervision=True,
            use_answer_feedback=True,
            halt_exploration_prob=1.0,  # Force all passes
            use_layer_act=True,
        )
        config = TrainingConfig(
            features=features, 
            max_passes=3, 
            max_recurrent_steps=max_iter
        )
        
        model.zero_grad()
        
        logits, cache, aux = model(
            demo_inputs, demo_outputs, test_input,
            config=config,
            step=0,
            return_aux=True,
        )
        
        loss, _ = compute_total_loss(model, logits, test_output, aux, config, 0, device)
        loss.backward()
        
        # Check iteration feedback gate gradients
        iter_fb_grads = []
        for name, param in model.named_parameters():
            if 'iteration_feedback' in name and param.grad is not None:
                iter_fb_grads.append(param.grad.abs().max().item())
        
        layer_iters = aux.get('layer_iterations', [])
        avg_iters = sum(layer_iters) / len(layer_iters) if layer_iters else 0
        
        print(f"  Passes: {aux.get('passes_run', 0)}, Avg layer iters: {avg_iters:.1f}")
        print(f"  Layer iterations per layer: {layer_iters}")
        print(f"  Loss: {loss.item():.4f}")
        if iter_fb_grads:
            print(f"  Iteration feedback gate grad max: {max(iter_fb_grads):.2e}")
        else:
            print(f"  ⚠️ No iteration feedback gradients")
        
        # Check read/write gate stats
        read_sum = aux.get('read_gate_sum', 0)
        read_count = aux.get('read_gate_count', 1)
        write_sum = aux.get('write_gate_sum', 0)
        write_count = aux.get('write_gate_count', 1)
        
        print(f"  Avg read gate: {read_sum/read_count:.3f}, Avg write gate: {write_sum/write_count:.3f}")


def compare_halting_strategies():
    """Compare different halting strategies and their effect on gradients."""
    print("\n" + "="*70)
    print("HALTING STRATEGY COMPARISON")
    print("="*70)
    
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = RecursiveRefinementModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        d_cache=48,
        num_layers=2,
        num_slots=8,
        num_heads=2,
        max_seq_len=100,
        max_internal_iterations=3,
        max_passes=5,
        dropout=0.0,
        confidence_threshold=0.5,  # Medium threshold
    ).to(device)
    
    B, S = 2, 50
    demo_inputs = torch.randint(0, VOCAB_SIZE, (B, 1, S), device=device)
    demo_outputs = torch.randint(0, VOCAB_SIZE, (B, 1, S), device=device)
    test_input = torch.randint(0, VOCAB_SIZE, (B, S), device=device)
    test_output = torch.randint(0, VOCAB_SIZE, (B, S), device=device)
    
    strategies = [
        ("Always halt (exploration=0)", 0.0),
        ("Rare exploration (exploration=0.1)", 0.1),
        ("Normal (exploration=0.3)", 0.3),
        ("Frequent exploration (exploration=0.5)", 0.5),
        ("Never halt (exploration=1.0)", 1.0),
    ]
    
    results = []
    
    for name, exploration_prob in strategies:
        print(f"\n--- {name} ---")
        
        features = FeatureFlags(
            use_deep_supervision=True,
            use_answer_feedback=True,
            halt_exploration_prob=exploration_prob,
            use_ponder_cost=True,
        )
        config = TrainingConfig(features=features, max_passes=5, max_recurrent_steps=3)
        
        # Run multiple times to see variance
        passes_list = []
        halted_list = []
        
        for trial in range(5):
            model.zero_grad()
            
            logits, cache, aux = model(
                demo_inputs, demo_outputs, test_input,
                config=config,
                step=trial,
                return_aux=True,
            )
            
            passes_list.append(aux.get('passes_run', 0))
            halted_list.append(aux.get('halted_early', False))
            
            # Clean up aux
            for key in list(aux.keys()):
                if isinstance(aux[key], list):
                    aux[key].clear()
        
        avg_passes = sum(passes_list) / len(passes_list)
        halt_rate = sum(halted_list) / len(halted_list)
        
        print(f"  Avg passes: {avg_passes:.1f}, Halt rate: {halt_rate*100:.0f}%")
        print(f"  Individual runs: {passes_list}")
        
        results.append({
            'name': name,
            'exploration': exploration_prob,
            'avg_passes': avg_passes,
            'halt_rate': halt_rate,
        })
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Strategy':<40} {'Exploration':>12} {'Avg Passes':>12} {'Halt Rate':>12}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<40} {r['exploration']:>12.1f} {r['avg_passes']:>12.1f} {r['halt_rate']*100:>11.0f}%")


if __name__ == "__main__":
    # Run all gradient scenarios
    run_all_scenarios()
    
    print("\n\n")
    
    # Debug answer feedback specifically
    debug_answer_feedback_specifically()
    
    print("\n\n")
    
    # Analyze layer iterations
    analyze_layer_iterations()
    
    print("\n\n")
    
    # Compare halting strategies
    compare_halting_strategies()
    
    print("\n\n")
    
    # Test full training step
    test_training_steps()
