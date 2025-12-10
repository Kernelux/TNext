import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint

from .config import TrainingConfig, FeatureFlags
from .layer import DLSMNLayer
from .modules import CacheSelfAttention, RotaryEmbedding
from .dataset import (
    VOCAB_SIZE, MAX_SEQ_LEN, PAD_TOKEN, EOS_TOKEN, COLOR_OFFSET,
    INPUT_MARKER, OUTPUT_MARKER,
)

class DLSMN_ARC(nn.Module):
    """
    DLSMN model for ARC-AGI tasks (Sequence-based, TRM-style).
    
    Key change: Uses flat sequences instead of 2D grids.
    - Vocab: PAD=0, EOS=1, INPUT_MARKER=2, OUTPUT_MARKER=3, colors 0-9 → 4-13
    - Role markers make input/output pairing explicit in the sequence itself
    - No size prediction needed - EOS markers indicate row boundaries
    - Standard sequence-to-sequence transformer approach
    
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
        vocab_size: int = VOCAB_SIZE,  # PAD, EOS, 10 colors
        d_model: int = 128,
        d_cache: int = 64,
        num_layers: int = 3,
        num_slots: int = 16,
        num_heads: int = 4,
        max_seq_len: int = MAX_SEQ_LEN,  # 900 tokens per grid
        max_recurrent_steps: int = 4,
        max_passes: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.total_slots = num_layers * num_slots
        self.max_seq_len = max_seq_len
        self.max_recurrent_steps = max_recurrent_steps
        self.max_passes = max_passes
        self.d_layer = d_cache // 4  # Layer-ID embedding dimension
        
        # Token embedding (unified for all tokens)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding: RoPE (zero params) or learned (heavy)
        # RoPE saves ~1.6M params compared to learned embeddings!
        # With RoPE, position info is applied in attention via rotation,
        # not added to embeddings. See modules.py for RotaryEmbedding.
        max_positions = 7 * max_seq_len + 100  # 7 segments * 900 + buffer
        self.pos_encoding_type = "rope"  # Default, can be overridden by flags
        
        # RoPE operates on head_dim, not d_model
        head_dim = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_positions,
            base=10000.0,
        )
        
        # Keep learned pos_embed as fallback (disabled by default)
        # To use: set flags.pos_encoding = "learned"
        # self.pos_embed = nn.Embedding(max_positions, d_model)  # ~1.6M params!
        
        # Segment embedding: context=0 (demos + test_input), answer=1 (test_output)
        # This tells the model WHAT to predict (answer segment) vs WHAT to use as context
        # Note: INPUT_MARKER/OUTPUT_MARKER tokens already encode role within the sequence
        self.segment_embed = nn.Embedding(2, d_model)
        
        # Learned slot embeddings (Section 7.7) - concept anchors
        # S ∈ ℝ^{(L×K) × D_cache}
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_layers, num_slots, d_cache) * 0.02
        )
        
        # Layer-ID embeddings (Section 7.6) - for representational separation
        self.layer_id_embeddings = nn.Parameter(
            torch.randn(num_layers, self.d_layer) * 0.02
        )
        
        # Temporal context embeddings for cache entries
        # These encode WHEN information was written (pass + recurrent step)
        # Dimension: D_cache // 8 each, so total temporal = D_cache // 4
        self.d_temporal = d_cache // 8
        self.pass_embeddings = nn.Embedding(max_passes + 1, self.d_temporal)  # +1 for pass 0/init
        self.step_embeddings = nn.Embedding(max_recurrent_steps + 1, self.d_temporal)  # +1 for step 0
        
        # Projection to add temporal context to cache entries (additive, not concat)
        # This keeps cache dimension fixed while encoding temporal info
        self.temporal_proj = nn.Linear(self.d_temporal * 2, d_cache)
        
        # DLSMN layers with proper signatures
        self.layers = nn.ModuleList([
            DLSMNLayer(
                layer_idx=i,
                d_model=d_model,
                d_cache=d_cache,
                num_slots=num_slots,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_recurrent_steps=max_recurrent_steps,
                max_passes=max_passes,
            )
            for i in range(num_layers)
        ])
        
        # Cache-to-cache attention (Section 10.2)
        # Using LinearAttention by default for O(S) memory efficiency
        self.cache_self_attn = CacheSelfAttention(d_cache, num_heads, dropout, use_linear=True)
        
        # === ACT HALTING (TRM-style Dual Q-head) ===
        # Q_halt: learns "stop, this is correct"
        # Q_continue: learns "continue, this is wrong" (optional, paper recommends skipping)
        # Uses logits (not probabilities) for stable BCE training
        self.q_head = nn.Sequential(
            nn.Linear(d_cache, d_cache // 2),
            nn.SiLU(),
            nn.Linear(d_cache // 2, 2),  # Output: [q_halt_logit, q_continue_logit]
        )
        
        # [TRM] Refinement read: project cache to model space for answer refinement
        self.refinement_query = nn.Linear(d_model, d_cache)
        self.refinement_value = nn.Linear(d_cache, d_model)
        self.refinement_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
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
            nn.SiLU(),
            nn.Linear(d_model // 2, self.max_recurrent_steps),
            # No softmax - applied with temperature in forward
        )
        
        # Predictive head for auxiliary loss (Section 9.1)
        self.predictor = nn.Sequential(
            nn.Linear(d_cache, d_cache),
            nn.SiLU(),
            nn.Linear(d_cache, d_model),
        )
        
        # Output head: project to vocab size
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # [TRM INSIGHT: Answer Feedback]
        # Project previous answer back to embedding space for next pass
        self.answer_embed = nn.Embedding(vocab_size, d_model)
        # Gate to control how much previous answer influences next pass
        self.answer_gate = nn.Linear(d_model * 2, d_model)
        
        # Initialize all weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights with appropriate strategies for each module type.
        
        Strategy:
        - Embeddings: Normal(0, 0.02)
        - Linear in FFN/attention: Kaiming for hidden, Normal(0, 0.02) for output
        - Gates (sigmoid output): Xavier + bias toward desired initial state
        - LayerNorm: weight=1, bias=0
        - Q-head: Small init for stable bootstrapping (like TRM)
        """
        from .utils import (
            init_embedding, init_linear_kaiming, init_linear_xavier,
            init_linear_normal, init_layer_norm, init_sequential, init_gate_bias
        )
        
        # === Embeddings ===
        init_embedding(self.token_embed)
        # NOTE: pos_embed removed - using RoPE which has no learnable params
        init_embedding(self.segment_embed)
        init_embedding(self.pass_embeddings)
        init_embedding(self.step_embeddings)
        init_embedding(self.answer_embed)
        
        # Slot and layer embeddings (nn.Parameter, not nn.Embedding)
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.layer_id_embeddings, mean=0.0, std=0.02)
        
        # === Projections ===
        init_linear_normal(self.temporal_proj, std=0.02)
        init_linear_normal(self.output_proj, std=0.02)
        init_linear_xavier(self.answer_gate)
        
        # Step predictor
        init_sequential(self.step_predictor)
        
        # === Q-head: Initialize with small values for stable bootstrapping ===
        # This matches TRM's approach: start Q near zero, let it learn
        init_sequential(self.q_head)
        # Set Q-head output bias to slightly negative (initially predict "don't halt")
        final_layer = list(self.q_head.children())[-1]
        if isinstance(final_layer, nn.Linear):
            init_gate_bias(final_layer, initial_value=-1.0)
        
        # Refinement modules
        init_linear_normal(self.refinement_query, std=0.02)
        init_linear_normal(self.refinement_value, std=0.02)
        init_sequential(self.refinement_gate, final_is_gate=True, gate_bias=0.0)
        
        # Predictor head
        init_sequential(self.predictor)
        
        # Layer step embeddings
        init_embedding(self.layer_step_embed)
        init_embedding(self.pass_step_embed)
    
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
    
    def embed_sequence(self, seq: torch.Tensor, is_answer: bool = False) -> torch.Tensor:
        """
        Embed a sequence with segment information.
        
        The role (input vs output) is now encoded by INPUT_MARKER/OUTPUT_MARKER
        tokens that are already in the sequence. We only need to distinguish:
        - context (is_answer=False): demos + test_input - model should read/understand
        - answer (is_answer=True): test_output - model should predict
        
        Args:
            seq: [B, S] token sequence (already contains role markers)
            is_answer: True if this is the answer segment (what model predicts)
            
        Returns:
            [B, S, D_model] embeddings
        """
        device = seq.device
        
        # Token embedding (includes INPUT_MARKER, OUTPUT_MARKER as regular tokens)
        token_emb = self.token_embed(seq)  # [B, S, D_model]
        
        # Segment embedding: context=0, answer=1
        # This tells the model which part to predict vs which to use as context
        segment_id = 1 if is_answer else 0
        segment_emb = self.segment_embed(torch.tensor(segment_id, device=device))  # [D_model]
        
        # Combine: token + segment (RoPE handles position in attention)
        emb = token_emb + segment_emb
        
        return emb
    
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
    
    def get_cache_self_attn_mask(
        self,
        layer_idx: int,
        pass_num: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Pass-aware attention mask for cache-to-cache self-attention (v0.1.1 §9.1).
        
        Applied AFTER each layer writes to enable inter-slot reasoning:
        - Pass 1: Causal - slot at layer N can only attend to layers 0..N-1
                  (not including its own layer's slots - prevents self-reinforcement)
        - Pass 2+: No mask - all slots can attend to everything (full global context)
        
        Returns:
            None if no masking needed (pass 2+)
            [L*K, L*K] bool tensor where True = BLOCKED
        """
        if pass_num > 1:
            # Full attention - no masking needed
            return None
        
        # Pass 1: Build causal mask based on layer structure
        # Each slot can only attend to slots from earlier layers
        L, K = self.num_layers, self.num_slots
        total = L * K
        
        mask = torch.ones(total, total, dtype=torch.bool, device=device)
        
        for query_layer in range(L):
            query_start = query_layer * K
            query_end = query_start + K
            
            # Query slots at query_layer can attend to all slots from layers 0..query_layer-1
            # (strictly earlier layers, not including itself)
            for key_layer in range(query_layer):  # 0 to query_layer-1
                key_start = key_layer * K
                key_end = key_start + K
                mask[query_start:query_end, key_start:key_end] = False  # Allow attention
        
        # Special case: Layer 0 has no earlier layers to attend to
        # Allow it to attend to itself (initialized slot_embeddings)
        mask[0:K, 0:K] = False
        
        return mask
    
    def refinement_read(
        self,
        h: torch.Tensor,     # [B, S, D_model] - answer representation
        cache: torch.Tensor, # [B, L*K, D_cache]
    ) -> torch.Tensor:
        """
        [TRM] Refinement Read: Answer reads from cache before final output.
        
        This allows the answer to benefit from ALL patterns stored during
        reasoning, not just what was explicitly passed through layers.
        
        Similar to TinyTRM's refinement step where z_H reads from slots
        before being projected to output.
        """
        B, S, D = h.shape
        
        # Query the cache
        q = self.refinement_query(h)  # [B, S, D_cache]
        
        # Attention over cache slots
        scores = torch.matmul(q, cache.transpose(-2, -1)) / (self.d_cache ** 0.5)
        attn = F.softmax(scores, dim=-1)  # [B, S, L*K]
        
        # Read values
        context = torch.matmul(attn, self.refinement_value(cache))  # [B, S, D_model]
        
        # Gated fusion
        gate = self.refinement_gate(torch.cat([h, context], dim=-1))
        h_refined = gate * context + (1 - gate) * h
        
        return h_refined
    
    def compute_q_halt(self, cache: torch.Tensor) -> tuple:
        """
        Compute Q_halt and Q_continue logits from cache state (TRM-style ACT).
        
        Returns LOGITS (not probabilities). Apply sigmoid for probabilities.
        
        Q_halt: should be high (positive logit) when prediction is correct → stop
        Q_continue: should be high when prediction is wrong → continue
                   (optional, paper recommends skipping Q_continue loss)
        
        Inference halting: stop if sigmoid(q_halt) > 0.5 (i.e., q_halt > 0)
        """
        cache_pooled = cache.mean(dim=1)  # [B, D_cache]
        q_logits = self.q_head(cache_pooled)  # [B, 2]
        q_halt = q_logits[:, 0]  # [B] - logit
        q_continue = q_logits[:, 1]  # [B] - logit
        return q_halt, q_continue
    
    def forward(
        self,
        demo_inputs: torch.Tensor,    # [B, num_demos, S] - sequences
        demo_outputs: torch.Tensor,   # [B, num_demos, S] - sequences
        test_input: torch.Tensor,     # [B, S] - sequence
        config: Optional[TrainingConfig] = None,
        step: int = 0,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Unified forward pass following DLSM_V0.1.md (Section 11).
        
        Now uses flat sequences instead of 2D grids.
        Returns: (logits, cache, aux_info)
        - logits: [B, S, vocab_size] - predictions for test output sequence
        - cache: [B, L*K, D_cache] - final cache state
        - aux_info: dict with auxiliary information
        """
        if config is None:
            config = TrainingConfig()
        
        B = test_input.shape[0]
        device = test_input.device
        num_demos = demo_inputs.shape[1]
        S = test_input.shape[1]  # Sequence length (MAX_SEQ_LEN)
        
        temperature = config.get_temperature(step)
        hard = (temperature < 0.2)  # Use hard routing when temperature is low
        threshold = config.write_threshold
        features = config.features
        
        # Always use adaptive mode - efficiency losses drive reduction
        pass_mode = 'adaptive'
        force_max_passes = False
        use_ponder = True  # Always apply ponder loss
        
        # Determine number of passes (with curriculum learning)
        if features.use_multi_pass:
            num_passes = config.get_curriculum_passes(step)
        else:
            num_passes = 1
        
        # Build unified sequence from demos + test input
        # All context (demos + test_input) gets segment=0 (context)
        # Using TRM-style aligned sequences: 30x30 grid → 900 tokens
        # Position i corresponds to same spatial location for input and output
        seq_parts = []
        
        for demo_idx in range(num_demos):
            # Demo input (context, 30x30 padded grid flattened)
            demo_in = self.embed_sequence(demo_inputs[:, demo_idx], is_answer=False)
            seq_parts.append(demo_in)
            
            # Demo output (context, aligned with demo input)
            demo_out = self.embed_sequence(demo_outputs[:, demo_idx], is_answer=False)
            seq_parts.append(demo_out)
        
        # Test input (context, 30x30 padded grid flattened)
        test_emb = self.embed_sequence(test_input, is_answer=False)
        test_start_idx = num_demos * 2 * S  # Position where test starts in concatenated sequence
        seq_parts.append(test_emb)
        
        full_seq = torch.cat(seq_parts, dim=1)  # [B, total_seq_len, D_model]
        total_seq_len = full_seq.shape[1]
        
        # === RoPE: Get cos/sin for this sequence length ===
        # RoPE encodes position via rotation in attention, not embeddings
        cos_sin = self.rotary_emb(seq_len=total_seq_len)
        
        # Initialize cache
        cache = self.get_initial_cache(B, device, features)
        layer_ids = self.get_layer_ids(features).to(device)
        
        # Track auxiliary losses
        aux_data = {
            'entropy': [],
            'slot_counts': [],
            'q_halt': [],      # Q-head halting logits
            'q_continue': [],  # Q-head continue logits (optional)
            'layer_steps': [],  # Track actual recurrent steps per layer
        }
        
        cumulative_halt = torch.zeros(B, device=device)
        total_ponder_cost = 0.0
        total_layer_steps = 0  # Total recurrent steps across all passes/layers
        
        write_counts = torch.zeros(B, self.total_slots, device=device)
        slot_ages = torch.zeros(B, self.total_slots, device=device)
        
        # === INPUT INJECTION MASK (DEMOS ONLY) ===
        # Create a mask that zeros out the test portion of embeddings
        # This prevents identity shortcuts where model copies test_input → test_output
        # Demo portions (indices 0 to test_start_idx) get injection, test doesn't
        demo_only_seq = full_seq.clone()
        demo_only_seq[:, test_start_idx:, :] = 0.0  # Zero out test portion
        
        h = full_seq
        pass_num = 1
        pass_logits = None
        prev_answer = None
        
        # Only last pass gets gradients (memory optimization)
        # With AdamAtan2, we could train all passes without explosion, but this saves memory.
        for pass_num in range(1, num_passes + 1):
            is_no_grad_pass = (pass_num < num_passes) and self.training and features.use_gradient_free_passes
            
            # Answer Feedback: incorporate previous prediction into test embedding
            if pass_num > 1 and features.use_answer_feedback and prev_answer is not None:
                # prev_answer is [B, S] - sequence of predicted tokens
                prev_answer_emb = self.answer_embed(prev_answer.detach())  # [B, S, D_model]
                
                test_emb_orig = full_seq[:, test_start_idx:test_start_idx + S]  # [B, S, D_model]
                combined = torch.cat([test_emb_orig, prev_answer_emb], dim=-1)  # [B, S, 2*D_model]
                gate = torch.sigmoid(self.answer_gate(combined))  # [B, S, D_model]
                test_emb_refined = gate * test_emb_orig + (1 - gate) * prev_answer_emb
                
                h = torch.cat([full_seq[:, :test_start_idx], test_emb_refined], dim=1)
            else:
                h = full_seq
            
            context = torch.no_grad() if is_no_grad_pass else nullcontext()
            
            # Use the model's max_recurrent_steps (matches step_predictor output size)
            # Apply curriculum learning cap if enabled
            max_recurrent_steps = self.max_recurrent_steps
            curriculum_max_recurrence = config.get_curriculum_recurrence(step)
            
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
                    
                    # Curriculum learning: mask out steps beyond current curriculum cap
                    # This ensures the model only learns to predict within available range
                    if curriculum_max_recurrence < max_recurrent_steps:
                        # Mask logits for unavailable steps with -inf
                        mask = torch.zeros_like(step_logits)
                        mask[:, curriculum_max_recurrence:] = float('-inf')
                        step_logits = step_logits + mask
                    
                    step_temp = max(temperature, 0.5)
                    step_probs = F.softmax(step_logits / step_temp, dim=-1)  # [B, max_steps]
                    step_probs = step_probs.clamp(min=1e-6, max=1.0 - 1e-6)
                    
                    # Expected steps (differentiable) - for efficiency loss
                    steps_range = torch.arange(1, max_recurrent_steps + 1, device=device, dtype=torch.float)
                    expected_steps = (step_probs * steps_range).sum(dim=-1)  # [B]
                    pass_expected_steps.append(expected_steps)
                    
                    # === VARIABLE STEPS: Actually use predicted steps ===
                    # Training: Sample from distribution (with Gumbel noise for exploration)
                    # Inference: Use expected value (deterministic)
                    if features.use_layer_act:
                        if self.training:
                            # Sample number of steps per batch item
                            # Use categorical sampling with straight-through for gradients
                            with torch.no_grad():
                                sampled_steps = torch.multinomial(step_probs, num_samples=1).squeeze(-1)  # [B]
                            # Use max across batch to ensure all items get processed
                            # (simpler than per-item variable steps, still learns from distribution)
                            num_steps_to_run = max(1, int(sampled_steps.float().mean().item() + 0.5))
                        else:
                            # Inference: use expected steps (deterministic)
                            num_steps_to_run = max(1, int(expected_steps.mean().item() + 0.5))
                        
                        # Apply curriculum cap during training
                        num_steps_to_run = min(num_steps_to_run, curriculum_max_recurrence)
                    else:
                        # Layer ACT disabled: always 1 step
                        num_steps_to_run = 1
                    
                    # Track actual steps for logging
                    aux_data['layer_steps'].append(num_steps_to_run)
                    total_layer_steps += num_steps_to_run
                    
                    # Use demo-only injection (test portion zeroed out)
                    input_injection = demo_only_seq
                    
                    # Only last recurrent step gets gradients (memory optimization)
                    for recur_step in range(num_steps_to_run):
                        is_last_recur_step = (recur_step == num_steps_to_run - 1)
                        step_context = nullcontext() if (is_last_recur_step and not is_no_grad_pass) else torch.no_grad()
                        
                        with step_context:
                            h, cache, updates = layer(
                                h, cache, 
                                cos_sin=cos_sin,
                                input_injection=input_injection,
                                temperature=temperature, 
                                hard=hard, 
                                features=features,
                            )
                        
                        # Track slot usage for diversity regularization
                        if features.use_cache:
                            start_idx = layer_idx * self.num_slots
                            end_idx = start_idx + self.num_slots
                            layer_slot_counts = updates['slot_probs'].sum(dim=1)
                            write_counts[:, start_idx:end_idx] += layer_slot_counts
                            
                            slot_ages += 1
                            written_mask = (layer_slot_counts > 1e-3).float()
                            slot_ages[:, start_idx:end_idx] = slot_ages[:, start_idx:end_idx] * (1 - written_mask)
                    
                    # === CACHE SELF-ATTENTION AFTER WRITE (v0.1.1 §9.1) ===
                    # Apply cache-to-cache attention AFTER each layer writes
                    # This enables inter-slot reasoning with pass-aware masking:
                    # - Pass 1: Causal (layer N attends to 0..N-1 only)
                    # - Pass 2+: Full attention (all slots see all slots)
                    if features.use_cache_self_attn and features.use_cache:
                        cache_attn_mask = self.get_cache_self_attn_mask(
                            layer_idx, pass_num, device
                        )
                        cache = self.cache_self_attn(cache, attn_mask=cache_attn_mask)
                    
                    # Log only final step stats
                    if return_aux and features.use_cache:
                        aux_data['entropy'].append(updates['entropy'].detach())
                        aux_data['slot_counts'].append(updates['slot_counts'].detach())
                        
                        if 'read_gates' not in aux_data: aux_data['read_gates'] = []
                        if 'write_gates' not in aux_data: aux_data['write_gates'] = []
                        
                        aux_data['read_gates'].append(updates['read_gate'].detach())
                        aux_data['write_gates'].append(updates['write_gate'].detach())
            
            # Collect expected steps for this pass
            # MEMORY OPTIMIZATION: Detach expected_steps from earlier passes
            # The efficiency loss only needs gradients from the current pass
            # We keep the values for computing the loss, but break the graph
            if 'expected_steps' not in aux_data: aux_data['expected_steps'] = []
            current_expected_steps = torch.stack(pass_expected_steps, dim=1)  # [B, num_layers]
            
            # Detach all previous expected_steps to free memory
            # Only the last pass (or current accumulation) needs gradients
            if pass_num < num_passes:
                aux_data['expected_steps'].append(current_expected_steps.detach())
            else:
                # Last pass - keep gradients for backprop
                aux_data['expected_steps'].append(current_expected_steps)
            
            # NOTE: Cache self-attention now happens AFTER each layer's write (above)
            # with pass-aware masking. The between-pass consolidation is kept as optional
            # "sleep-like" global consolidation for backward compatibility.
            if pass_num < num_passes and features.use_cache_self_attn and features.use_between_pass_consolidation:
                # Full unmasked attention between passes (global consolidation)
                cache = self.cache_self_attn(cache, attn_mask=None)
                # MEMORY OPTIMIZATION: Detach cache after self-attention between passes
                # This breaks the gradient chain but dramatically reduces memory
                if features.detach_cache_between_passes:
                    cache = cache.detach()
            
            # Compute Q-head halting signals
            q_halt, q_continue = self.compute_q_halt(cache)
            # MEMORY OPTIMIZATION: Only keep gradients for the last pass
            if pass_num < num_passes:
                aux_data['q_halt'].append(q_halt.detach())
                aux_data['q_continue'].append(q_continue.detach())
            else:
                aux_data['q_halt'].append(q_halt)
                aux_data['q_continue'].append(q_continue)
            
            # [TRM] Refinement read: answer reads from cache before output
            # test_h is [B, S, D_model] - the test portion of the sequence
            test_h = h[:, test_start_idx:test_start_idx + S]
            if features.use_refinement_read and pass_num == num_passes:
                test_h = self.refinement_read(test_h, cache)
            
            # NOTE: Gradient highway removed - it was creating a shortcut where the model
            # could just pass through input colors instead of learning transformations.
            # Gradient flow to embeddings is maintained via:
            # 1. Refinement read (above) - cache contains embedded info
            # 2. Normal backprop through attention and memory operations
            # 3. The pred_diversity_loss which provides direct gradient signal
            
            # Output: [B, S, vocab_size] - logits for each position in output sequence
            pass_logits = self.output_proj(test_h)  # [B, S, vocab_size]
            
            # Only store pass_logits if deep supervision enabled (memory optimization)
            if features.use_deep_supervision:
                if 'pass_logits' not in aux_data: aux_data['pass_logits'] = []
                aux_data['pass_logits'].append(pass_logits)
            
            # prev_answer is [B, S] - predicted token for each position
            prev_answer = pass_logits.detach().argmax(dim=-1)
            
            # === ACT HALTING ===
            # During inference: stop if Q_halt predicts "correct" (positive logit → prob > 0.5)
            # During training: always run all passes (halting learned via Q-head loss)
            if features.use_act_halting and not force_max_passes:
                # Ponder cost: penalize remaining computation
                halt_prob = torch.sigmoid(q_halt)
                remaining = 1 - cumulative_halt
                if use_ponder:
                    total_ponder_cost += remaining.mean()
                cumulative_halt = cumulative_halt + (1 - cumulative_halt) * halt_prob
                
                # Early stopping during inference only
                if not self.training and halt_prob.mean() > 0.5:
                    break
        
        if pass_logits is None:
            test_h = h[:, test_start_idx:test_start_idx + S]
            pass_logits = self.output_proj(test_h)  # [B, S, vocab_size]
        logits = pass_logits
        
        aux_info = {
            'temperature': temperature, 
            'pass_mode': pass_mode,
            'ponder_cost': total_ponder_cost, 
            'num_passes': pass_num,
            'pass_logits': aux_data.get('pass_logits', []),
            'final_logits': logits,  # For Q-head loss without deep supervision
            'expected_steps': aux_data.get('expected_steps', []),  # For step efficiency loss
            'q_halt': aux_data['q_halt'],       # Q-head halt logits per pass
            'q_continue': aux_data['q_continue'],  # Q-head continue logits per pass
            # Pass/step statistics for monitoring
            'total_layer_steps': total_layer_steps,  # Total recurrent steps across all passes/layers
            'layer_steps': aux_data.get('layer_steps', []),  # Steps per layer (flattened across passes)
            'avg_layer_steps': total_layer_steps / max(1, len(self.layers) * pass_num),  # Avg steps per layer
            'max_layer_steps': self.max_recurrent_steps,  # For computing utilization
            'max_passes': num_passes,  # Max allowed passes (for utilization)
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
            
        return logits, cache, aux_info
