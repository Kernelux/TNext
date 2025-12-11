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
        
        # NOTE: Phase-specific Q-heads defined below with step predictors
        # NOTE: refinement_read removed - refinement happens through multi-pass at layer/module level
        
        # Layer-level: Step Predictor (predicts distribution over recurrent steps)
        # Context-aware: uses layer_idx and pass_num embeddings
        # Embeddings for layer position and pass number
        self.layer_step_embed = nn.Embedding(num_layers, d_model // 4)
        self.pass_step_embed = nn.Embedding(self.max_passes, d_model // 4)
        
        # === TWO-PHASE ARCHITECTURE ===
        # Each phase (Reflection, Answer) has its own ACT system:
        # - Q-head for pass-level halting
        # - Step predictor for layer-level recurrence
        # This allows each phase to independently learn how much compute it needs
        
        # Phase embedding: 0=reflection, 1=answer
        self.phase_embed = nn.Embedding(2, d_model // 4)
        
        # Phase-specific step predictors
        # Input: h_pooled + layer_ctx + pass_ctx + phase_ctx -> step distribution
        step_input_dim = d_model + d_model // 4 + d_model // 4 + d_model // 4
        
        self.reflection_step_predictor = nn.Sequential(
            nn.Linear(step_input_dim, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, self.max_recurrent_steps),
        )
        
        self.answer_step_predictor = nn.Sequential(
            nn.Linear(step_input_dim, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, self.max_recurrent_steps),
        )
        
        # Phase-specific Q-heads for pass-level halting
        # Q_halt: learns "stop, this phase is complete" based on OUTPUT quality
        # NOT cache - cache is working memory, not a completion signal
        self.reflection_q_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2),  # [q_halt, q_continue]
        )
        
        self.answer_q_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2),  # [q_halt, q_continue]
        )
        
        # Legacy (kept for compatibility, can remove later)
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
        
        # Step predictor (legacy)
        init_sequential(self.step_predictor)
        
        # Phase-specific step predictors
        init_sequential(self.reflection_step_predictor)
        init_sequential(self.answer_step_predictor)
        
        # Phase embedding
        init_embedding(self.phase_embed)
        
        # === Phase-specific Q-heads: Initialize with small values for stable bootstrapping ===
        init_sequential(self.reflection_q_head)
        init_sequential(self.answer_q_head)
        
        # Set Q-head output bias to slightly negative (initially predict "don't halt")
        for q_head in [self.reflection_q_head, self.answer_q_head]:
            final_layer = list(q_head.children())[-1]
            if isinstance(final_layer, nn.Linear):
                init_gate_bias(final_layer, initial_value=-1.0)
        
        # NOTE: refinement modules removed - refinement through multi-pass
        
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
    
    def compute_q_halt(self, h: torch.Tensor, phase: str = 'answer') -> tuple:
        """
        Compute Q_halt and Q_continue logits from hidden state (TRM-style ACT).
        
        IMPORTANT: Halting is based on OUTPUT quality, not cache state.
        The cache is working memory for computation - not a signal of completion.
        The model should halt when its output is good enough, assessed from h.
        
        Args:
            h: [B, S, D_model] - current hidden state (output of the phase)
            phase: 'reflection' or 'answer' - selects which Q-head to use
        
        Returns LOGITS (not probabilities). Apply sigmoid for probabilities.
        
        Q_halt: should be high (positive logit) when output is good → stop
        Q_continue: should be high when output needs refinement → continue
        
        Inference halting: stop if sigmoid(q_halt) > 0.5 (i.e., q_halt > 0)
        """
        h_pooled = h.mean(dim=1)  # [B, D_model]
        
        if phase == 'reflection':
            q_logits = self.reflection_q_head(h_pooled)  # [B, 2]
        else:  # 'answer'
            q_logits = self.answer_q_head(h_pooled)  # [B, 2]
        
        q_halt = q_logits[:, 0]  # [B] - logit
        q_continue = q_logits[:, 1]  # [B] - logit
        return q_halt, q_continue
    
    def predict_steps(
        self, 
        h: torch.Tensor, 
        layer_idx: int, 
        pass_num: int, 
        phase: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Predict distribution over recurrent steps for a layer.
        
        Args:
            h: [B, S, D_model] - current hidden state
            layer_idx: which layer (0 to num_layers-1)
            pass_num: which pass within the phase (1-indexed)
            phase: 'reflection' or 'answer'
            device: torch device
            
        Returns:
            step_logits: [B, max_recurrent_steps] - logits for each step count
        """
        B = h.shape[0]
        
        h_pooled = h.mean(dim=1)  # [B, D_model]
        
        layer_ctx = self.layer_step_embed(
            torch.tensor(layer_idx, device=device)
        ).unsqueeze(0).expand(B, -1)  # [B, D//4]
        
        pass_ctx = self.pass_step_embed(
            torch.tensor(min(pass_num - 1, self.max_passes - 1), device=device)
        ).unsqueeze(0).expand(B, -1)  # [B, D//4]
        
        phase_id = 0 if phase == 'reflection' else 1
        phase_ctx = self.phase_embed(
            torch.tensor(phase_id, device=device)
        ).unsqueeze(0).expand(B, -1)  # [B, D//4]
        
        predictor_input = torch.cat([h_pooled, layer_ctx, pass_ctx, phase_ctx], dim=-1)
        
        if phase == 'reflection':
            step_logits = self.reflection_step_predictor(predictor_input)
        else:
            step_logits = self.answer_step_predictor(predictor_input)
        
        return step_logits
    
    def init_answer_state(self, test_h: torch.Tensor) -> torch.Tensor:
        """
        Initialize answer state from test representation.
        
        The answer state starts from test_h (the model's understanding of the test input
        after reflection phase) and will be refined in the answer phase.
        
        Args:
            test_h: [B, S, D_model] - test portion representation from reflection phase
            
        Returns:
            [B, S, D_model] - initial answer state
        """
        # Direct pass-through: gradients flow back to reflection phase
        # No clone() - we WANT gradients to propagate through both phases
        return test_h
    
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
        Two-Phase Forward Pass:
        
        === PHASE 1: REFLECTION ===
        Process [demos, test_input] to fill cache with patterns.
        - Demos teach the transformation pattern
        - Test input context is encoded
        - Cache accumulates relevant patterns
        
        === PHASE 2: ANSWER ===
        Generate/refine answer by reading from cache.
        - answer_state initialized from test_h (reflection output)
        - Same layers used, read/write to cache enabled
        - Multiple passes refine the answer
        
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
        S = test_input.shape[1]  # Sequence length (MAX_SEQ_LEN = 900)
        
        temperature = config.get_temperature(step)
        hard = (temperature < 0.2)  # Use hard routing when temperature is low
        features = config.features
        
        # Determine number of answer passes (with curriculum learning)
        if features.use_multi_pass:
            num_answer_passes = config.get_curriculum_passes(step)
        else:
            num_answer_passes = 1
        
        # === BUILD REFLECTION SEQUENCE ===
        # All context (demos + test_input) processed together
        # Using TRM-style aligned sequences: 30x30 grid → 900 tokens
        reflection_parts = []
        
        for demo_idx in range(num_demos):
            # Demo input
            demo_in = self.embed_sequence(demo_inputs[:, demo_idx], is_answer=False)
            reflection_parts.append(demo_in)
            
            # Demo output (aligned with demo input)
            demo_out = self.embed_sequence(demo_outputs[:, demo_idx], is_answer=False)
            reflection_parts.append(demo_out)
        
        # Test input
        test_emb = self.embed_sequence(test_input, is_answer=False)
        test_start_idx = num_demos * 2 * S  # Position where test starts in concatenated sequence
        reflection_parts.append(test_emb)
        
        reflection_seq = torch.cat(reflection_parts, dim=1)  # [B, 6300, D_model] typically
        reflection_len = reflection_seq.shape[1]
        
        # === RoPE: Get cos/sin for sequences ===
        cos_sin_reflection = self.rotary_emb(seq_len=reflection_len)
        cos_sin_answer = self.rotary_emb(seq_len=S)  # Answer is just S tokens
        
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
            'read_gates': [],
            'write_gates': [],
        }
        
        cumulative_halt = torch.zeros(B, device=device)
        total_ponder_cost = 0.0
        total_layer_steps = 0  # Total recurrent steps across all phases
        
        # Use the model's max_recurrent_steps (matches step_predictor output size)
        max_recurrent_steps = self.max_recurrent_steps
        curriculum_max_recurrence = config.get_curriculum_recurrence(step)
        
        # Number of passes for each phase (both use curriculum learning)
        if features.use_multi_pass:
            num_reflection_passes = config.get_curriculum_passes(step)
            num_answer_passes = config.get_curriculum_passes(step)
        else:
            num_reflection_passes = 1
            num_answer_passes = 1
        
        # ===================================================================
        # PHASE 1: REFLECTION
        # Process demos + test_input to fill cache with transformation patterns
        # Multiple passes with adaptive recurrence per layer
        # ===================================================================
        h = reflection_seq
        reflection_cumulative_halt = torch.zeros(B, device=device)
        reflection_pass = 1  # Will be updated in loop
        
        for reflection_pass in range(1, num_reflection_passes + 1):
            is_no_grad_pass = (reflection_pass < num_reflection_passes) and self.training and features.use_gradient_free_passes
            context = torch.no_grad() if is_no_grad_pass else nullcontext()
            
            pass_expected_steps = []
            updates = None
            
            with context:
                for layer_idx, layer in enumerate(self.layers):
                    # Predict step distribution for this layer (reflection phase)
                    step_logits = self.predict_steps(h, layer_idx, reflection_pass, 'reflection', device)
                    
                    # Curriculum learning: mask unavailable steps
                    if curriculum_max_recurrence < max_recurrent_steps:
                        mask = torch.zeros_like(step_logits)
                        mask[:, curriculum_max_recurrence:] = float('-inf')
                        step_logits = step_logits + mask
                    
                    step_temp = max(temperature, 0.5)
                    step_probs = F.softmax(step_logits / step_temp, dim=-1)
                    step_probs = step_probs.clamp(min=1e-6, max=1.0 - 1e-6)
                    
                    # Expected steps (differentiable) - for efficiency loss
                    steps_range = torch.arange(1, max_recurrent_steps + 1, device=device, dtype=torch.float)
                    expected_steps = (step_probs * steps_range).sum(dim=-1)
                    pass_expected_steps.append(expected_steps)
                    
                    # Determine actual steps to run
                    if features.use_layer_act:
                        if self.training:
                            with torch.no_grad():
                                sampled_steps = torch.multinomial(step_probs, num_samples=1).squeeze(-1)
                            num_steps_to_run = max(1, int(sampled_steps.float().mean().item() + 0.5))
                        else:
                            num_steps_to_run = max(1, int(expected_steps.mean().item() + 0.5))
                        num_steps_to_run = min(num_steps_to_run, curriculum_max_recurrence)
                    else:
                        num_steps_to_run = 1
                    
                    aux_data['layer_steps'].append(num_steps_to_run)
                    total_layer_steps += num_steps_to_run
                    
                    # Run recurrent steps
                    for recur_step in range(num_steps_to_run):
                        is_last_recur_step = (recur_step == num_steps_to_run - 1)
                        step_context = nullcontext() if (is_last_recur_step and not is_no_grad_pass) else torch.no_grad()
                        
                        with step_context:
                            h, cache, updates = layer(
                                h, cache, 
                                cos_sin=cos_sin_reflection,
                                input_injection=None,
                                temperature=temperature, 
                                hard=hard, 
                                features=features,
                            )
                    
                    # Cache self-attention after layer (pass-aware)
                    if features.use_cache_self_attn and features.use_cache:
                        cache_attn_mask = self.get_cache_self_attn_mask(layer_idx, reflection_pass, device)
                        cache = self.cache_self_attn(cache, attn_mask=cache_attn_mask)
                    
                    # Log stats
                    if updates is not None and features.use_cache:
                        aux_data['entropy'].append(updates['entropy'].detach())
                        aux_data['slot_counts'].append(updates['slot_counts'].detach())
                        aux_data['read_gates'].append(updates['read_gate'].detach())
                        aux_data['write_gates'].append(updates['write_gate'].detach())
            
            # Collect expected steps for this reflection pass
            if 'reflection_expected_steps' not in aux_data: aux_data['reflection_expected_steps'] = []
            current_expected_steps = torch.stack(pass_expected_steps, dim=1)
            if reflection_pass < num_reflection_passes:
                aux_data['reflection_expected_steps'].append(current_expected_steps.detach())
            else:
                aux_data['reflection_expected_steps'].append(current_expected_steps)
            
            # Optional cache consolidation between passes
            if reflection_pass < num_reflection_passes and features.use_cache_self_attn and features.use_between_pass_consolidation:
                cache = self.cache_self_attn(cache, attn_mask=None)
                if features.detach_cache_between_passes:
                    cache = cache.detach()
            
            # Compute Q-head halting for reflection phase (based on output h, not cache)
            q_halt, q_continue = self.compute_q_halt(h, phase='reflection')
            if 'reflection_q_halt' not in aux_data: aux_data['reflection_q_halt'] = []
            if 'reflection_q_continue' not in aux_data: aux_data['reflection_q_continue'] = []
            
            if reflection_pass < num_reflection_passes:
                aux_data['reflection_q_halt'].append(q_halt.detach())
                aux_data['reflection_q_continue'].append(q_continue.detach())
            else:
                aux_data['reflection_q_halt'].append(q_halt)
                aux_data['reflection_q_continue'].append(q_continue)
            
            # ACT halting for reflection
            if features.use_act_halting:
                halt_prob = torch.sigmoid(q_halt)
                remaining = 1 - reflection_cumulative_halt
                total_ponder_cost += remaining.mean()
                reflection_cumulative_halt = reflection_cumulative_halt + (1 - reflection_cumulative_halt) * halt_prob
                
                if not self.training and halt_prob.mean() > 0.5:
                    break
        
        # Extract test_h from reflection output
        test_h = h[:, test_start_idx:test_start_idx + S]  # [B, S, D_model]
        
        # ===================================================================
        # PHASE 2: ANSWER
        # Generate/refine answer using cached knowledge
        # Multiple passes with adaptive recurrence per layer
        # ===================================================================
        
        # Initialize answer state from test representation
        answer_state = self.init_answer_state(test_h)  # [B, S, D_model]
        
        pass_logits = None
        prev_answer = None
        answer_cumulative_halt = torch.zeros(B, device=device)
        answer_pass = 1  # Will be updated in loop
        
        for answer_pass in range(1, num_answer_passes + 1):
            is_no_grad_pass = (answer_pass < num_answer_passes) and self.training and features.use_gradient_free_passes
            
            # Answer Feedback: incorporate previous prediction
            if answer_pass > 1 and features.use_answer_feedback and prev_answer is not None:
                prev_answer_emb = self.answer_embed(prev_answer.detach())  # [B, S, D_model]
                combined = torch.cat([answer_state, prev_answer_emb], dim=-1)  # [B, S, 2*D_model]
                gate = torch.sigmoid(self.answer_gate(combined))  # [B, S, D_model]
                answer_state = gate * answer_state + (1 - gate) * prev_answer_emb
            
            context = torch.no_grad() if is_no_grad_pass else nullcontext()
            
            pass_expected_steps = []
            
            with context:
                h_answer = answer_state
                updates = None
                
                for layer_idx, layer in enumerate(self.layers):
                    # Predict step distribution for this layer (answer phase)
                    step_logits = self.predict_steps(h_answer, layer_idx, answer_pass, 'answer', device)
                    
                    # Curriculum learning: mask unavailable steps
                    if curriculum_max_recurrence < max_recurrent_steps:
                        mask = torch.zeros_like(step_logits)
                        mask[:, curriculum_max_recurrence:] = float('-inf')
                        step_logits = step_logits + mask
                    
                    step_temp = max(temperature, 0.5)
                    step_probs = F.softmax(step_logits / step_temp, dim=-1)
                    step_probs = step_probs.clamp(min=1e-6, max=1.0 - 1e-6)
                    
                    # Expected steps (differentiable) - for efficiency loss
                    steps_range = torch.arange(1, max_recurrent_steps + 1, device=device, dtype=torch.float)
                    expected_steps = (step_probs * steps_range).sum(dim=-1)
                    pass_expected_steps.append(expected_steps)
                    
                    # Determine actual steps to run
                    if features.use_layer_act:
                        if self.training:
                            with torch.no_grad():
                                sampled_steps = torch.multinomial(step_probs, num_samples=1).squeeze(-1)
                            num_steps_to_run = max(1, int(sampled_steps.float().mean().item() + 0.5))
                        else:
                            num_steps_to_run = max(1, int(expected_steps.mean().item() + 0.5))
                        num_steps_to_run = min(num_steps_to_run, curriculum_max_recurrence)
                    else:
                        num_steps_to_run = 1
                    
                    aux_data['layer_steps'].append(num_steps_to_run)
                    total_layer_steps += num_steps_to_run
                    
                    # Run layer on answer state
                    for recur_step in range(num_steps_to_run):
                        is_last_recur_step = (recur_step == num_steps_to_run - 1)
                        step_context = nullcontext() if (is_last_recur_step and not is_no_grad_pass) else torch.no_grad()
                        
                        with step_context:
                            h_answer, cache, updates = layer(
                                h_answer, cache, 
                                cos_sin=cos_sin_answer,
                                input_injection=None,
                                temperature=temperature, 
                                hard=hard, 
                                features=features,
                            )
                    
                    # Cache self-attention after layer
                    # Use pass_num = reflection_passes + answer_pass for proper masking
                    effective_pass = num_reflection_passes + answer_pass
                    if features.use_cache_self_attn and features.use_cache:
                        cache_attn_mask = self.get_cache_self_attn_mask(layer_idx, effective_pass, device)
                        cache = self.cache_self_attn(cache, attn_mask=cache_attn_mask)
                    
                    # Log stats
                    if updates is not None and features.use_cache:
                        aux_data['entropy'].append(updates['entropy'].detach())
                        aux_data['slot_counts'].append(updates['slot_counts'].detach())
                        aux_data['read_gates'].append(updates['read_gate'].detach())
                        aux_data['write_gates'].append(updates['write_gate'].detach())
                
                answer_state = h_answer
            
            # Collect expected steps for this answer pass
            if 'answer_expected_steps' not in aux_data: aux_data['answer_expected_steps'] = []
            current_expected_steps = torch.stack(pass_expected_steps, dim=1)
            
            if answer_pass < num_answer_passes:
                aux_data['answer_expected_steps'].append(current_expected_steps.detach())
            else:
                aux_data['answer_expected_steps'].append(current_expected_steps)
            
            # Optional cache consolidation between passes
            if answer_pass < num_answer_passes and features.use_cache_self_attn and features.use_between_pass_consolidation:
                cache = self.cache_self_attn(cache, attn_mask=None)
                if features.detach_cache_between_passes:
                    cache = cache.detach()
            
            # Compute Q-head halting for answer phase (based on answer_state, not cache)
            q_halt, q_continue = self.compute_q_halt(answer_state, phase='answer')
            aux_data['q_halt'].append(q_halt if answer_pass == num_answer_passes else q_halt.detach())
            aux_data['q_continue'].append(q_continue if answer_pass == num_answer_passes else q_continue.detach())
            
            # Output logits directly from answer_state
            # Refinement happens through multi-pass mechanism, not a separate read
            pass_logits = self.output_proj(answer_state)  # [B, S, vocab_size]
            
            if features.use_deep_supervision:
                if 'pass_logits' not in aux_data: aux_data['pass_logits'] = []
                aux_data['pass_logits'].append(pass_logits)
            
            prev_answer = pass_logits.detach().argmax(dim=-1)
            
            # ACT halting for answer phase
            if features.use_act_halting:
                halt_prob = torch.sigmoid(q_halt)
                remaining = 1 - answer_cumulative_halt
                total_ponder_cost += remaining.mean()
                answer_cumulative_halt = answer_cumulative_halt + (1 - answer_cumulative_halt) * halt_prob
                
                if not self.training and halt_prob.mean() > 0.5:
                    break
        
        logits = pass_logits
        
        # Combine expected steps from both phases for efficiency loss
        all_expected_steps = aux_data.get('reflection_expected_steps', []) + aux_data.get('answer_expected_steps', [])
        
        aux_info = {
            'temperature': temperature, 
            'pass_mode': 'two_phase',
            'ponder_cost': total_ponder_cost, 
            'num_reflection_passes': reflection_pass,
            'num_answer_passes': answer_pass,
            'num_passes': reflection_pass + answer_pass,  # Total passes
            'pass_logits': aux_data.get('pass_logits', []),
            'final_logits': logits,
            'expected_steps': all_expected_steps,
            'reflection_expected_steps': aux_data.get('reflection_expected_steps', []),
            'answer_expected_steps': aux_data.get('answer_expected_steps', []),
            'q_halt': aux_data['q_halt'],  # Answer phase Q-halts
            'q_continue': aux_data['q_continue'],
            'reflection_q_halt': aux_data.get('reflection_q_halt', []),
            'reflection_q_continue': aux_data.get('reflection_q_continue', []),
            # Statistics
            'total_layer_steps': total_layer_steps,
            'layer_steps': aux_data.get('layer_steps', []),
            'avg_layer_steps': total_layer_steps / max(1, len(self.layers) * (reflection_pass + answer_pass)),
            'max_layer_steps': self.max_recurrent_steps,
            'max_passes': num_reflection_passes + num_answer_passes,
        }
        
        if aux_data['entropy']:
            aux_info['avg_entropy'] = torch.stack(aux_data['entropy']).mean()
            aux_info['slot_counts'] = aux_data['slot_counts']
        
        if 'read_gates' in aux_data: aux_info['read_gates'] = aux_data['read_gates']
        if 'write_gates' in aux_data: aux_info['write_gates'] = aux_data['write_gates']
            
        return logits, cache, aux_info
