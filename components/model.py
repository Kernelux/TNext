import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint

from .config import TrainingConfig, FeatureFlags
from .layer import DLSMNLayer
from .modules import CacheSelfAttention

class DLSMN_ARC(nn.Module):
    """
    DLSMN model for ARC-AGI tasks.
    
    Faithful implementation of DLSM_V0.1.md:
    - Global cache: C ∈ ℝ^{(L×K) × D_cache} (Section 2.1)
    - Learned slot embeddings as concept anchors (Section 7.7)
    - Layer-ID embeddings for representational separation (Section 7.6)
    - Per-layer W_compress and W_decompress projections (Section 2.1)
    - Pass-aware masking (Section 2.3)
    - Cache-to-cache attention (Section 10.2)
    - ACT halting (Section 10.1)
    """
    
    def __init__(
        self,
        num_colors: int = 10,
        d_model: int = 128,
        d_cache: int = 64,
        num_layers: int = 3,
        num_slots: int = 16,
        num_patterns: int = 16,
        num_heads: int = 4,
        max_grid_size: int = 30,
        max_recurrent_steps: int = 4,
        max_passes: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_colors = num_colors
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.num_patterns = num_patterns
        self.total_slots = num_layers * num_slots
        self.max_grid_size = max_grid_size
        self.max_recurrent_steps = max_recurrent_steps
        self.max_passes = max_passes
        self.d_layer = d_cache // 4  # Layer-ID embedding dimension
        
        # Embeddings
        self.color_embed = nn.Embedding(num_colors + 1, d_model)
        self.pos_embed_h = nn.Embedding(max_grid_size, d_model // 2)
        self.pos_embed_w = nn.Embedding(max_grid_size, d_model // 2)
        self.type_embed = nn.Embedding(4, d_model)
        
        # Learned slot embeddings (Section 7.7) - concept anchors
        # S ∈ ℝ^{(L×K) × D_cache}
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_layers, num_slots, d_cache) * 0.02
        )
        
        # Layer-ID embeddings (Section 7.6) - for representational separation
        self.layer_id_embeddings = nn.Parameter(
            torch.randn(num_layers, self.d_layer) * 0.02
        )
        
        # DLSMN layers with proper signatures
        self.layers = nn.ModuleList([
            DLSMNLayer(
                layer_idx=i,
                d_model=d_model,
                d_cache=d_cache,
                num_slots=num_slots,
                num_layers=num_layers,
                num_patterns=num_patterns,
                num_heads=num_heads,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        
        # Cache-to-cache attention (Section 10.2)
        self.cache_self_attn = CacheSelfAttention(d_cache, num_heads, dropout)
        
        # === HIERARCHICAL ACT ===
        # Model-level halt network (on cache state) - Q-head for pass halting
        self.halt_net = nn.Sequential(
            nn.Linear(d_cache, d_cache // 2),
            nn.ReLU(),
            nn.Linear(d_cache // 2, 1),
            nn.Sigmoid(),
        )
        
        # Layer-level: Step Predictor (predicts distribution over recurrent steps)
        # Context-aware: uses layer_idx and pass_num embeddings
        # Embeddings for layer position and pass number
        self.layer_step_embed = nn.Embedding(num_layers, d_model // 4)
        self.pass_step_embed = nn.Embedding(self.max_passes, d_model // 4)
        
        # Predictor: h_pooled + layer_ctx + pass_ctx -> step distribution
        self.step_predictor = nn.Sequential(
            nn.Linear(d_model + d_model // 4 + d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, self.max_recurrent_steps),
            # No softmax - applied with temperature in forward
        )
        
        # Predictive head for auxiliary loss (Section 9.1)
        self.predictor = nn.Sequential(
            nn.Linear(d_cache, d_cache),
            nn.ReLU(),
            nn.Linear(d_cache, d_model),
        )
        
        # Output head
        self.output_proj = nn.Linear(d_model, num_colors)
        
        # [TRM INSIGHT: Answer Feedback]
        # Project previous answer back to embedding space for next pass
        self.answer_embed = nn.Embedding(num_colors, d_model)
        # Gate to control how much previous answer influences next pass
        self.answer_gate = nn.Linear(d_model * 2, d_model)
        
        # Size prediction head
        self.size_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, max_grid_size * 2),
        )
    
    def get_initial_cache(
        self, 
        batch_size: int, 
        device: torch.device,
        features: Optional[FeatureFlags] = None,
    ) -> torch.Tensor:
        """
        Initialize cache with learned slot embeddings (Section 7.7).
        C^{(0)} = S
        """
        if features is None:
            features = FeatureFlags()
        
        # [ABLATION: use_slot_embeddings]
        if features.use_slot_embeddings:
            # Reshape slot embeddings from [L, K, D_cache] to [L*K, D_cache]
            cache = self.slot_embeddings.view(self.total_slots, self.d_cache)
            return cache.unsqueeze(0).expand(batch_size, -1, -1).clone()
        else:
            # Zero initialization
            return torch.zeros(batch_size, self.total_slots, self.d_cache, device=device)
    
    def get_layer_ids(self, features: Optional[FeatureFlags] = None) -> torch.Tensor:
        """
        Get layer-ID embeddings for all slots (Section 7.6).
        Returns [L*K, D_layer] tensor where each slot has its layer's embedding.
        """
        if features is None:
            features = FeatureFlags()
        
        # [ABLATION: use_layer_id]
        if features.use_layer_id:
            # Repeat each layer's embedding K times
            layer_ids = self.layer_id_embeddings.unsqueeze(1).expand(-1, self.num_slots, -1)
            return layer_ids.reshape(self.total_slots, self.d_layer)
        else:
            # Zero layer IDs (no layer separation)
            return torch.zeros(self.total_slots, self.d_layer, device=self.layer_id_embeddings.device)
    
    def embed_grid(self, grid: torch.Tensor, grid_type: int) -> torch.Tensor:
        """Embed a grid with positional and type information."""
        B, H, W = grid.shape
        device = grid.device
        
        color_emb = self.color_embed(grid)
        
        h_pos = torch.arange(H, device=device)
        w_pos = torch.arange(W, device=device)
        h_emb = self.pos_embed_h(h_pos)
        w_emb = self.pos_embed_w(w_pos)
        
        pos_emb = torch.cat([
            h_emb.unsqueeze(1).expand(-1, W, -1),
            w_emb.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1)
        
        type_emb = self.type_embed(torch.tensor(grid_type, device=device))
        emb = color_emb + pos_emb.unsqueeze(0) + type_emb
        
        return emb.view(B, H * W, -1)
    
    def get_cache_mask(
        self, 
        batch_size: int, 
        layer_idx: int, 
        pass_num: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Pass-aware read mask (Section 2.3):
        - Pass 1: Layer j reads C[0:j*K] (earlier layers only)
        - Pass 2+: Layer j reads C[:] (full cache)
        
        Returns mask where True = BLOCKED.
        
        Note: For layer 0 pass 1, no earlier layers exist, so we allow 
        reading its own slots (otherwise everything is masked -> NaN).
        """
        mask = torch.zeros(batch_size, self.total_slots, dtype=torch.bool, device=device)
        if pass_num == 1:
            first_blocked_slot = layer_idx * self.num_slots
            # Special case: layer 0 can read its own slots (initialized from slot_embeddings)
            if first_blocked_slot > 0:
                mask[:, first_blocked_slot:] = True
            # For layer 0, first_blocked_slot=0, so nothing is masked - this is intentional
            # The initialized slot_embeddings serve as the initial "memory" to read from
        return mask
    
    def apply_cache_updates(
        self, 
        cache: torch.Tensor, 
        updates: Dict,
        layer_idx: int,
        threshold: float,
        features: Optional[FeatureFlags] = None,
    ) -> torch.Tensor:
        """
        Apply weighted cache writes (Section 3.1 Step 3, 7.3).
        
        Weighted mean aggregation with Soft WTA.
        """
        if features is None:
            features = FeatureFlags()
            
        B = cache.shape[0]
        y_cache = updates['y_cache']         # [B, num_patterns, D_cache]
        scores = updates['scores']           # [B, num_patterns]
        slot_probs = updates['slot_probs']   # [B, num_patterns, K]
        K = self.num_slots
        
        # Write mask based on importance threshold
        write_mask = (scores > threshold).float().unsqueeze(-1)  # [B, num_patterns, 1]
        
        # [REFINEMENT: use_soft_wta_update]
        if features.use_soft_wta_update:
            # Clamp scores to prevent exp overflow (exp(88) ≈ 1e38, close to float32 max)
            clamped_scores = scores.clamp(min=-50, max=50)
            weights = torch.exp(clamped_scores).unsqueeze(-1) * write_mask
        else:
            weights = scores.unsqueeze(-1) * write_mask
            
        weighted_y = weights * y_cache  # [B, num_patterns, D_cache]
        
        # Aggregate to slots
        slot_writes = torch.einsum('bsk,bsd->bkd', slot_probs, weighted_y)
        slot_weights = torch.einsum('bsk,bs->bk', slot_probs, weights.squeeze(-1))
        
        # Normalize
        slot_weights = slot_weights.unsqueeze(-1).clamp(min=1e-8)
        slot_writes = slot_writes / slot_weights
        
        # Update cache at this layer's slots
        new_cache = cache.clone()
        start_idx = layer_idx * K
        end_idx = start_idx + K
        
        # Only update slots that received writes
        has_writes = (slot_weights.squeeze(-1) > 1e-6).unsqueeze(-1).float()
        
        new_cache[:, start_idx:end_idx] = (
            has_writes * slot_writes + (1 - has_writes) * cache[:, start_idx:end_idx]
        )
        
        return new_cache
    
    def compute_halt_prob(self, cache: torch.Tensor) -> torch.Tensor:
        """
        ACT halting probability (Section 10.1).
        """
        pooled = cache.mean(dim=1)  # [B, D_cache]
        return self.halt_net(pooled).squeeze(-1)  # [B]
    
    def forward(
        self,
        demo_inputs: torch.Tensor,    # [B, num_demos, H, W]
        demo_outputs: torch.Tensor,   # [B, num_demos, H, W]
        test_input: torch.Tensor,     # [B, H, W]
        config: Optional[TrainingConfig] = None,
        step: int = 0,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Unified forward pass following DLSM_V0.1.md (Section 11).
        """
        if config is None:
            config = TrainingConfig()
        
        B = test_input.shape[0]
        device = test_input.device
        num_demos = demo_inputs.shape[1]
        H, W = test_input.shape[1], test_input.shape[2]
        
        temperature = config.get_temperature(step)
        hard = (temperature < 0.2)  # Use hard routing when temperature is low
        threshold = config.write_threshold
        features = config.features
        
        # Always use adaptive mode - efficiency losses drive reduction
        pass_mode = 'adaptive'
        force_max_passes = False
        use_ponder = True  # Always apply ponder loss
        
        # Determine number of passes
        num_passes = config.max_passes if features.use_multi_pass else 1
        
        # Build unified sequence
        seq_parts = []
        for demo_idx in range(num_demos):
            demo_in = self.embed_grid(demo_inputs[:, demo_idx], grid_type=0)
            demo_out = self.embed_grid(demo_outputs[:, demo_idx], grid_type=1)
            seq_parts.extend([demo_in, demo_out])
        
        test_emb = self.embed_grid(test_input, grid_type=2)
        seq_parts.append(test_emb)
        
        full_seq = torch.cat(seq_parts, dim=1)
        test_start_idx = 2 * num_demos * H * W
        
        # Initialize cache
        cache = self.get_initial_cache(B, device, features)
        layer_ids = self.get_layer_ids(features).to(device)
        
        # Track auxiliary losses
        aux_data = {
            'entropy': [],
            'slot_counts': [],
            'halt_probs': [],
        }
        
        cumulative_halt = torch.zeros(B, device=device)
        total_ponder_cost = 0.0
        
        write_counts = torch.zeros(B, self.total_slots, device=device)
        slot_ages = torch.zeros(B, self.total_slots, device=device)
        
        h = full_seq
        pass_num = 1
        pass_logits = None
        prev_answer = None
        
        no_grad_passes = getattr(config, 'no_grad_passes', 0)
        
        for pass_num in range(1, num_passes + 1):
            is_no_grad_pass = (pass_num <= no_grad_passes) and self.training
            
            # Answer Feedback
            if pass_num > 1 and features.use_answer_feedback and prev_answer is not None:
                prev_answer_emb = self.answer_embed(prev_answer.detach())
                prev_answer_emb = prev_answer_emb.view(B, H * W, -1)
                
                test_emb_orig = full_seq[:, test_start_idx:]
                combined = torch.cat([test_emb_orig, prev_answer_emb], dim=-1)
                gate = torch.sigmoid(self.answer_gate(combined))
                test_emb_refined = gate * test_emb_orig + (1 - gate) * prev_answer_emb
                
                h = torch.cat([full_seq[:, :test_start_idx], test_emb_refined], dim=1)
            else:
                h = full_seq
            
            context = torch.no_grad() if is_no_grad_pass else nullcontext()
            
            # Use the model's max_recurrent_steps (matches step_predictor output size)
            max_recurrent_steps = self.max_recurrent_steps
            
            # === EFFICIENT STEP SELECTION PER LAYER ===
            # During training: run fixed max steps but weight by step_probs for gradient
            # During inference: can use argmax for actual compute savings
            pass_expected_steps = []  # Collect for efficiency loss
            
            with context:
                for layer_idx, layer in enumerate(self.layers):
                    # Predict step distribution for this layer
                    h_pooled = h.mean(dim=1)  # [B, D]
                    layer_ctx = self.layer_step_embed(
                        torch.tensor(layer_idx, device=device)
                    ).unsqueeze(0).expand(B, -1)  # [B, D//4]
                    pass_ctx = self.pass_step_embed(
                        torch.tensor(min(pass_num - 1, self.max_passes - 1), device=device)
                    ).unsqueeze(0).expand(B, -1)  # [B, D//4]
                    
                    predictor_input = torch.cat([h_pooled, layer_ctx, pass_ctx], dim=-1)
                    step_logits = self.step_predictor(predictor_input)  # [B, max_steps]
                    step_temp = max(temperature, 0.5)
                    step_probs = F.softmax(step_logits / step_temp, dim=-1)  # [B, max_steps]
                    step_probs = step_probs.clamp(min=1e-6, max=1.0 - 1e-6)
                    
                    # Expected steps (differentiable) - for efficiency loss
                    steps_range = torch.arange(1, max_recurrent_steps + 1, device=device, dtype=torch.float)
                    expected_steps = (step_probs * steps_range).sum(dim=-1)  # [B]
                    pass_expected_steps.append(expected_steps)
                    
                    # SIMPLIFIED: Just run a fixed number of steps (1 during training for speed)
                    # The step_predictor learns via efficiency loss, not by actually varying steps
                    # This decouples "learning to predict steps" from "running variable steps"
                    num_steps_to_run = 1 if self.training else max(1, int(expected_steps.mean().item() + 0.5))
                    
                    for recur_step in range(num_steps_to_run):
                        effective_pass = (pass_num - 1) * max_recurrent_steps + recur_step + 1
                        cache_mask = self.get_cache_mask(B, layer_idx, effective_pass, device)
                        slot_emb = self.slot_embeddings[layer_idx]
                        
                        h, updates = layer(
                            h, cache, slot_emb, layer_ids, cache_mask,
                            write_counts=write_counts,
                            slot_ages=slot_ages,
                            temperature=temperature, hard=hard, features=features,
                            step=step,
                        )
                        
                        # Update cache after each step
                        if features.use_cache:
                            cache = self.apply_cache_updates(cache, updates, layer_idx, threshold, features=features)
                            
                            start_idx = layer_idx * self.num_slots
                            end_idx = start_idx + self.num_slots
                            layer_slot_counts = updates['slot_probs'].sum(dim=1)
                            write_counts[:, start_idx:end_idx] += layer_slot_counts
                            
                            slot_ages += 1
                            written_mask = (layer_slot_counts > 1e-3).float()
                            slot_ages[:, start_idx:end_idx] = slot_ages[:, start_idx:end_idx] * (1 - written_mask)
                    
                    # Log only final step stats
                    if return_aux and features.use_cache:
                        aux_data['entropy'].append(updates['entropy'].detach())
                        aux_data['slot_counts'].append(updates['slot_counts'].detach())
                        
                        if 'read_gates' not in aux_data: aux_data['read_gates'] = []
                        if 'write_gates' not in aux_data: aux_data['write_gates'] = []
                        if 'read_slots' not in aux_data: aux_data['read_slots'] = []
                        
                        aux_data['read_gates'].append(updates['read_gate'].detach())
                        aux_data['read_slots'].append(updates['read_slot_probs'].detach())
                        aux_data['write_gates'].append(updates['write_gate'].detach())
            
            # Collect expected steps for this pass
            if 'expected_steps' not in aux_data: aux_data['expected_steps'] = []
            aux_data['expected_steps'].append(torch.stack(pass_expected_steps, dim=1))  # [B, num_layers]
            
            if pass_num < num_passes and features.use_cache_self_attn:
                cache = self.cache_self_attn(cache)
            
            halt_prob = self.compute_halt_prob(cache)
            aux_data['halt_probs'].append(halt_prob)
            
            test_h = h[:, test_start_idx:]
            pass_logits = self.output_proj(test_h).view(B, H, W, -1)
            
            # Only store pass_logits if deep supervision enabled (memory optimization)
            if features.use_deep_supervision:
                if 'pass_logits' not in aux_data: aux_data['pass_logits'] = []
                aux_data['pass_logits'].append(pass_logits)
            
            prev_answer = pass_logits.detach().argmax(dim=-1)
            
            # === ACT HALTING (respects pass exploration mode) ===
            # force_max: Run all passes, no early stopping
            # adaptive: Use halting but no ponder penalty  
            # adaptive_penalized: Use halting with ponder penalty
            if features.use_act_halting and not features.use_explicit_q_head and not force_max_passes:
                remaining = 1 - cumulative_halt
                if use_ponder:
                    total_ponder_cost += remaining.mean()
                cumulative_halt = cumulative_halt + (1 - cumulative_halt) * halt_prob
                
                if not self.training and halt_prob.mean() > 0.5:
                    break
        
        if pass_logits is None:
            test_h = h[:, test_start_idx:]
            pass_logits = self.output_proj(test_h).view(B, H, W, -1)
        logits = pass_logits
        
        test_h = h[:, test_start_idx:]
        size_logits_flat = self.size_proj(test_h.mean(dim=1))
        size_logits = size_logits_flat.view(-1, 2, self.max_grid_size)
        
        aux_info = {
            'temperature': temperature, 
            'pass_mode': pass_mode,
            'ponder_cost': total_ponder_cost, 
            'num_passes': pass_num,
            'pass_logits': aux_data.get('pass_logits', []),
            'halt_probs': aux_data['halt_probs'],
            'final_logits': logits,  # For Q-head loss without deep supervision
            'expected_steps': aux_data.get('expected_steps', []),  # For step efficiency loss
        }
        
        if 'layer_halt_probs' in aux_data: aux_info['layer_halt_probs'] = aux_data['layer_halt_probs']
        if 'layer_stability' in aux_data: aux_info['layer_stability'] = aux_data['layer_stability']
        
        if return_aux:
            if aux_data['entropy']:
                aux_info['avg_entropy'] = torch.stack(aux_data['entropy']).mean()
                aux_info['slot_counts'] = aux_data['slot_counts']
            
            # [LOGGING] Add collected stats to output
            if 'read_gates' in aux_data: aux_info['read_gates'] = aux_data['read_gates']
            if 'write_gates' in aux_data: aux_info['write_gates'] = aux_data['write_gates']
            if 'read_slots' in aux_data: aux_info['read_slots'] = aux_data['read_slots']
            
        return logits, size_logits, cache, aux_info
