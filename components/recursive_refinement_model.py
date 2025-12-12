"""
Unified Recursive Refinement Model
===================================

This model implements the key insights from TRM (Tiny Recursive Model) applied to DLSMN:
1. Unified recursive refinement instead of separate reflection/answer phases
2. True iterative refinement with confidence-based halting
3. Recursive reasoning within layers AND across model passes
4. Clean separation of memory operations from computation

The architecture follows the design principles from proposal.md:
- Memory operations handled by unified MemoryController
- Computation handled by any compute block
- Halting decisions based on output confidence, not cache state
- Recursive refinement until confidence threshold met

Components imported from unified_layer.py:
- ComputeBlock: Standard transformer-style computation
- UnifiedMemoryLayer: Layer with memory read/compute/write cycle

This file provides:
- RecursiveRefinementModel: Full ARC-compatible model with TRM-style refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

from .config import TrainingConfig, FeatureFlags, SLOT_DIMS
from .memory_controller import ModelHaltEstimator
from .unified_layer import UnifiedMemoryLayer, ComputeBlock  # Import from unified_layer
from .modules import RotaryEmbedding
from .dataset import (
    VOCAB_SIZE, MAX_SEQ_LEN, PAD_TOKEN, EOS_TOKEN, COLOR_OFFSET,
    INPUT_MARKER, OUTPUT_MARKER,
)


# Note: ComputeBlock and UnifiedMemoryLayer are imported from unified_layer.py
# No need to duplicate them here - use the canonical implementations


class RecursiveRefinementModel(nn.Module):
    """
    Unified Recursive Model with multi-pass architecture and confidence-based refinement.

    Key changes from original DLSMN:
    1. Single forward pass instead of reflection/answer phases
    2. Each layer can recursively refine its output until confident (internal iterations)
    3. Whole model can recursively refine its output across multiple passes
    4. Halting decisions based on output confidence, not cache state
    5. TRM-inspired recursive reasoning with adaptive computation at multiple levels

    Architecture:
    Input → [Embedding] → [Pass 0: Layer 0 → Layer 1 → ... → Output] →
    [Pass 1: Layer 0 → Layer 1 → ... → Output] → ... → Final Output
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        d_cache: int = 64,
        num_layers: int = 3,
        num_slots: int = 16,
        num_heads: int = 4,
        max_seq_len: int = MAX_SEQ_LEN,
        max_internal_iterations: int = 4,  # For layers
        max_passes: int = 6,               # For model-level
        dropout: float = 0.1,
        confidence_threshold: float = 0.8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.total_slots = num_layers * num_slots
        self.max_internal_iterations = max_internal_iterations
        self.max_passes = max_passes
        self.confidence_threshold = confidence_threshold

        # Token embedding (unified for all tokens)
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Positional encoding: RoPE (zero params) or learned (heavy)
        max_positions = 7 * max_seq_len + 100  # 7 segments * 900 + buffer
        self.pos_encoding_type = "rope"  # Default

        head_dim = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_positions,
            base=10000.0,
        )

        # Segment embedding: context=0 (demos + test_input), answer=1 (test_output)
        self.segment_embed = nn.Embedding(2, d_model)

        # === Slot Metadata Dimensions (from centralized config) ===
        self.d_meta = SLOT_DIMS.d_meta
        self.d_slot = SLOT_DIMS.d_slot(d_cache)

        # Learned slot embeddings (concept anchors) - content only
        # Metadata (confidence + temporal) is initialized to zero
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_layers, num_slots, d_cache) * 0.02
        )

        # Layer-ID embeddings for representational separation
        # Dimension from centralized SLOT_DIMS
        self.layer_id_embeddings = nn.Parameter(
            torch.randn(num_layers, SLOT_DIMS.d_layer_embed) * 0.02
        )

        # === Model-Level Halt Estimator (entropy-based) ===
        # Separate from per-layer estimators because:
        # 1. Model-level has access to logits (can use entropy)
        # 2. Layer-level only has hidden states (uses stability)
        # 3. Different halting criteria: "is answer confident?" vs "has layer converged?"
        self.model_halt_estimator = ModelHaltEstimator(
            d_model=d_model,
            vocab_size=vocab_size,
            max_passes=max_passes,
        )

        # Unified Memory Layers (imported from unified_layer.py)
        # Each layer creates its own LayerHaltEstimator internally
        self.layers = nn.ModuleList([
            UnifiedMemoryLayer(
                d_model=d_model,
                d_cache=d_cache,
                num_slots=num_slots,
                num_layers=num_layers,
                layer_idx=i,
                num_heads=num_heads,
                dropout=dropout,

                use_linear_attention=True,
                use_checkpoint=(max_seq_len > 500),  # Enable for large sequences
            )
            for i in range(num_layers)
        ])

        # DEPRECATED: No longer share estimator - each layer has its own
        # for layer in self.layers:
        #     layer.confidence_estimator = self.confidence_estimator

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Answer feedback mechanism (TRM insight)
        self.prev_answer_embed = nn.Embedding(vocab_size, d_model)
        self.answer_feedback_gate = nn.Linear(d_model * 2, d_model)
        
        # === Pass Embeddings (Model-level iteration awareness) ===
        # Let the model know which pass it's on (how many times it's seen this input)
        # This is crucial for the model to know when to refine vs when to commit
        self.pass_embeddings = nn.Embedding(max_passes, d_model)
        
        # === Centralized Temporal Embeddings for Cache Metadata ===
        # These are shared across all MemoryControllers to ensure consistency
        # Smaller dimensions than d_model since they're stored in cache slots
        self.cache_layer_embed = nn.Embedding(num_layers, SLOT_DIMS.d_layer_embed)
        self.cache_iter_embed = nn.Embedding(max_internal_iterations, SLOT_DIMS.d_iter_embed)
        self.cache_pass_embed = nn.Embedding(max_passes, SLOT_DIMS.d_pass_embed)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all components."""
        # Token embedding
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

        # Segment embedding
        nn.init.normal_(self.segment_embed.weight, mean=0.0, std=0.02)

        # Slot embeddings
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)

        # Layer-ID embeddings
        nn.init.normal_(self.layer_id_embeddings, mean=0.0, std=0.02)

        # Output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Answer feedback gate
        nn.init.xavier_uniform_(self.answer_feedback_gate.weight)
        
        # Pass embeddings (small init so pass 0 doesn't dominate)
        nn.init.normal_(self.pass_embeddings.weight, mean=0.0, std=0.02)
        
        # Cache temporal embeddings (shared across all MemoryControllers)
        nn.init.normal_(self.cache_layer_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cache_iter_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cache_pass_embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.answer_feedback_gate.bias)

    def get_initial_cache(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize cache with learned slot embeddings + layer identity metadata.
        
        Cache structure: [content (d_cache) | confidence (1) | temporal (16)]
        - Content: learned slot embeddings
        - Confidence: 0 (empty slots have no confidence)
        - Temporal: [layer_id | iter | pass] from SLOT_DIMS
        """
        # Content from learned embeddings
        content = self.slot_embeddings.view(self.total_slots, self.d_cache)  # [L*K, D_cache]
        content = content.unsqueeze(0).expand(batch_size, -1, -1).clone()   # [B, L*K, D_cache]
        
        # Confidence: 0 for empty slots
        confidence = torch.zeros(batch_size, self.total_slots, 1, device=device)
        
        # Layer identity embedding: each layer's slots get that layer's embedding
        layer_ids = self.layer_id_embeddings.unsqueeze(1).expand(
            -1, self.num_slots, -1
        ).reshape(self.total_slots, SLOT_DIMS.d_layer_embed)
        layer_ids = layer_ids.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Iteration and pass embeddings: 0 (not yet written)
        iter_pass_dim = SLOT_DIMS.d_iter_embed + SLOT_DIMS.d_pass_embed  # 4 + 4 = 8
        iter_pass = torch.zeros(batch_size, self.total_slots, iter_pass_dim, device=device)
        
        # Combine temporal metadata: [layer_id | iter | pass]
        temporal = torch.cat([layer_ids, iter_pass], dim=-1)  # [B, L*K, d_temporal]
        
        # Full metadata: [confidence (1) | temporal]
        metadata = torch.cat([confidence, temporal], dim=-1)  # [B, L*K, d_meta]
        
        # Combine into full cache
        cache = torch.cat([content, metadata], dim=-1)  # [B, L*K, D_slot]
        return cache

    def embed_sequence(self, seq: torch.Tensor, is_answer: bool = False) -> torch.Tensor:
        """Embed sequence with segment information."""
        device = seq.device

        # Token embedding
        token_emb = self.token_embed(seq)  # [B, S, D_model]

        # Segment embedding: context=0, answer=1
        segment_id = 1 if is_answer else 0
        segment_emb = self.segment_embed(torch.tensor(segment_id, device=device))  # [D_model]

        # Combine: token + segment
        emb = token_emb + segment_emb

        return emb

    def forward(
        self,
        demo_inputs: torch.Tensor,    # [B, num_demos, S] - sequences
        demo_outputs: torch.Tensor,   # [B, num_demos, S] - sequences
        test_input: torch.Tensor,     # [B, S] - sequence
        config: Optional[TrainingConfig] = None,
        step: int = 0,
        return_aux: bool = True,
        force_max_passes: bool = False,  # Warmup mode: disable early halting
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Multi-pass recursive refinement forward pass.

        Implements TRM-style recursive reasoning where the full model
        iteratively refines its answer until confident or max passes reached.

        Args:
            demo_inputs: [B, num_demos, S] - demonstration inputs
            demo_outputs: [B, num_demos, S] - demonstration outputs
            test_input: [B, S] - test input to transform
            config: Training configuration
            step: Training step (for curriculum)
            return_aux: Return auxiliary information

        Returns:
            logits: [B, S, vocab_size] - predictions for test output sequence
            final_cache: [B, L*K, D_cache] - final cache state
            aux: Auxiliary information dict
        """
        if config is None:
            config = TrainingConfig()

        B = test_input.shape[0]
        device = test_input.device
        num_demos = demo_inputs.shape[1]
        S = test_input.shape[1]  # Sequence length (MAX_SEQ_LEN = 900)

        temperature = config.get_temperature(step)
        hard = (temperature < 0.2)
        features = config.features

        # Build single sequence from all input data
        all_parts = []

        for demo_idx in range(num_demos):
            # Demo input
            demo_in = self.embed_sequence(demo_inputs[:, demo_idx], is_answer=False)
            all_parts.append(demo_in)

            # Demo output (aligned with demo input)
            demo_out = self.embed_sequence(demo_outputs[:, demo_idx], is_answer=False)
            all_parts.append(demo_out)

        # Test input
        test_emb = self.embed_sequence(test_input, is_answer=False)
        test_start_idx = num_demos * 2 * S  # Position where test starts in concatenated sequence
        all_parts.append(test_emb)

        full_seq = torch.cat(all_parts, dim=1)  # [B, total_seq_len, D_model]
        total_len = full_seq.shape[1]

        # Get RoPE embeddings
        cos_sin_full = self.rotary_emb(seq_len=total_len)

        # Initialize cache
        cache = self.get_initial_cache(B, device)

        # Track auxiliary information - use scalars to save memory
        aux = {
            # Model-level stats
            'passes_run': 0,
            'max_passes': self.max_passes,
            'pass_confidences': [],  # Keep as list (small: just max_passes items)
            'pass_logits': [],       # For deep supervision: logits at each pass
            'answer_feedback_sum': 0.0,
            'answer_feedback_count': 0,
            
            # Layer-level stats (aggregated as scalars)
            'layer_iterations': [],  # Keep as list (small: passes * layers items)
            'total_layer_steps': 0,
            'confidence_sum': 0.0,
            'confidence_count': 0,
            'read_gate_sum': 0.0,
            'read_gate_count': 0,
            'write_gate_sum': 0.0,
            'write_gate_count': 0,
            'iteration_feedback_sum': 0.0,
            'iteration_feedback_count': 0,
            
            # Compute tracking
            'temperature': temperature,
            'halted_early': False,   # Track if we halted before max_passes
        } if return_aux else {}

        # Current sequence for input to the model
        current_seq = full_seq
        prev_logits = None
        final_logits = None
        
        # === INPUT INJECTION (TRM-style) ===
        # Save original embedded sequence for re-injection at each pass
        # This keeps the model grounded to the original input
        input_embedding = full_seq  # [B, total_seq_len, D_model]

        # === MODEL-LEVEL RECURSIVE REFINEMENT LOOP ===
        # Use config.max_passes if provided (for debug scenarios), else model default
        effective_max_passes = min(config.max_passes, self.max_passes) if config.max_passes else self.max_passes
        for pass_idx in range(effective_max_passes):
            if return_aux:
                aux['passes_run'] = pass_idx + 1
            
            # === PASS EMBEDDING (iteration awareness) ===
            # Add pass embedding so model knows which iteration it's on
            # This helps the model know when to refine vs commit to an answer
            pass_emb = self.pass_embeddings(torch.tensor(pass_idx, device=device))  # [D_model]
            current_seq = current_seq + pass_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, D_model] broadcast
            
            # === INPUT INJECTION at pass level (TRM-style) ===
            # Re-inject original input embedding at each pass (after first)
            # TRM does: hidden_states = hidden_states + input_injection
            if pass_idx > 0:
                current_seq = current_seq + input_embedding

            # === ANSWER FEEDBACK (TRM-style) ===
            # Inject previous answer as a "hint" for refinement
            # Only after first pass, and only if enabled
            if pass_idx > 0 and prev_logits is not None and features.use_answer_feedback:
                # Get previous predictions (detached - no gradients through this path)
                with torch.no_grad():
                    prev_predictions = prev_logits.argmax(dim=-1)  # [B, S]
                
                # Embed previous answer
                prev_answer_emb = self.prev_answer_embed(prev_predictions)  # [B, S, D_model]
                
                # Get current test region
                test_region = current_seq[:, test_start_idx:test_start_idx + S]  # [B, S, D_model]
                
                # Gated fusion: learn how much to incorporate previous answer
                gate_input = torch.cat([test_region, prev_answer_emb], dim=-1)  # [B, S, 2*D_model]
                gate = torch.sigmoid(self.answer_feedback_gate(gate_input))  # [B, S, D_model]
                
                # Update test region with gated feedback (additive)
                updated_test_region = test_region + gate * prev_answer_emb
                
                # Track answer feedback gate values
                if return_aux:
                    aux['answer_feedback_sum'] += gate.mean().detach().item()
                    aux['answer_feedback_count'] += 1
                    # Store gate tensor for polarization loss
                    if 'answer_feedback_gates' not in aux:
                        aux['answer_feedback_gates'] = []
                    aux['answer_feedback_gates'].append(gate)
                
                # Rebuild current_seq with updated test region (no in-place ops)
                # This avoids the autograd issue with slice assignment
                current_seq = torch.cat([
                    current_seq[:, :test_start_idx],
                    updated_test_region,
                    current_seq[:, test_start_idx + S:]
                ], dim=1) if test_start_idx + S < current_seq.shape[1] else torch.cat([
                    current_seq[:, :test_start_idx],
                    updated_test_region
                ], dim=1)

            # Process through recursive layers WITHIN the current pass
            h = current_seq
            prev_layer_output = None  # For cross-layer skip connection
            
            for layer_idx, layer in enumerate(self.layers):
                # Use config.max_recurrent_steps if provided, else model default
                if features.use_layer_act:
                    max_iterations = min(config.max_recurrent_steps, self.max_internal_iterations) if config.max_recurrent_steps else self.max_internal_iterations
                else:
                    max_iterations = 1
                h, cache, layer_aux = layer(
                    h, cache,
                    cos_sin=cos_sin_full,
                    temperature=temperature,
                    hard=hard,
                    max_iterations=max_iterations,
                    confidence_threshold=self.confidence_threshold,
                    pass_idx=pass_idx,  # Pass temporal index for slot metadata
                    # Centralized temporal embeddings for cache
                    cache_layer_embed=self.cache_layer_embed,
                    cache_iter_embed=self.cache_iter_embed,
                    cache_pass_embed=self.cache_pass_embed,
                )
                
                # === CROSS-LAYER SKIP CONNECTION ===
                # Add previous layer's output to current layer's output
                # Creates dense connectivity: layer_0 output contributes to all later layers
                if prev_layer_output is not None:
                    h = h + prev_layer_output
                prev_layer_output = h.detach()  # Detach to avoid double gradient through skip

                if return_aux and layer_aux:
                    iters_run = layer_aux.get('iterations_run', 1)
                    aux['layer_iterations'].append(iters_run)
                    aux['total_layer_steps'] += iters_run
                    
                    # Aggregate scalar stats from layer
                    aux['confidence_sum'] += layer_aux.get('confidence_sum', 0.0)
                    aux['confidence_count'] += layer_aux.get('confidence_count', 0)
                    aux['read_gate_sum'] += layer_aux.get('read_gate_sum', 0.0)
                    aux['read_gate_count'] += layer_aux.get('read_gate_count', 0)
                    aux['write_gate_sum'] += layer_aux.get('write_gate_sum', 0.0)
                    aux['write_gate_count'] += layer_aux.get('write_gate_count', 0)
                    aux['iteration_feedback_sum'] += layer_aux.get('iteration_feedback_sum', 0.0)
                    aux['iteration_feedback_count'] += layer_aux.get('iteration_feedback_count', 0)
                    
                    # Aggregate threshold tensors (for gradient flow)
                    if 'read_thresholds' not in aux:
                        aux['read_thresholds'] = []
                    if 'write_thresholds' not in aux:
                        aux['write_thresholds'] = []
                    if 'layer_halt_thresholds' not in aux:
                        aux['layer_halt_thresholds'] = []
                    if 'layer_confidences' not in aux:
                        aux['layer_confidences'] = []
                    if 'read_gates' not in aux:
                        aux['read_gates'] = []
                    if 'write_gates' not in aux:
                        aux['write_gates'] = []
                    if 'layer_stabilities' not in aux:
                        aux['layer_stabilities'] = []
                    if 'layer_info_gains' not in aux:
                        aux['layer_info_gains'] = []
                    if 'layer_entropies' not in aux:
                        aux['layer_entropies'] = []
                    aux['read_thresholds'].extend(layer_aux.get('read_thresholds', []))
                    aux['write_thresholds'].extend(layer_aux.get('write_thresholds', []))
                    aux['layer_halt_thresholds'].extend(layer_aux.get('layer_halt_thresholds', []))
                    aux['layer_confidences'].extend(layer_aux.get('layer_confidences', []))
                    # Layer divergence tracking (for divergence loss)
                    aux['layer_stabilities'].extend(layer_aux.get('layer_stabilities', []))
                    aux['layer_info_gains'].extend(layer_aux.get('layer_info_gains', []))
                    aux['layer_entropies'].extend(layer_aux.get('layer_entropies', []))
                    # Gate tensors for polarization loss
                    aux['read_gates'].extend(layer_aux.get('read_gates', []))
                    aux['write_gates'].extend(layer_aux.get('write_gates', []))

            # Extract logits for test output positions
            test_logits = self.output_proj(h[:, test_start_idx:test_start_idx + S])  # [B, S, vocab_size]
            final_logits = test_logits
            
            # Store logits for deep supervision (if enabled)
            if return_aux and features.use_deep_supervision:
                aux['pass_logits'].append(test_logits)

            # Confidence assessment for model-level halting (Q-halt + entropy)
            if return_aux:
                h_test_region = h[:, test_start_idx:test_start_idx + S]  # [B, S, D_model]
                
                # Use model-level halt estimator (Q-halt + entropy)
                confidence, q_halt_logits, halt_aux = self.model_halt_estimator(
                    h_test_region,
                    logits=test_logits,
                    prev_logits=prev_logits,
                    current_pass=pass_idx,
                    max_pass=self.max_passes,
                )
                
                aux['pass_confidences'].append(confidence.detach())
                
                # Store Q-halt logits for loss computation (needs gradients!)
                if 'q_halt_logits_list' not in aux:
                    aux['q_halt_logits_list'] = []
                aux['q_halt_logits_list'].append(q_halt_logits)  # Keep gradients!
                
                # Track entropy and info_gain from halt_aux
                if 'pass_entropies' not in aux:
                    aux['pass_entropies'] = []
                aux['pass_entropies'].append(halt_aux['entropy'].mean().item())
                
                # Store info_gain as TENSOR for gradient flow (maximization loss)
                if 'pass_info_gains' not in aux:
                    aux['pass_info_gains'] = []
                # Keep tensor for loss computation (detach for the halt decision below)
                aux['pass_info_gains'].append(halt_aux['info_gain'].mean())  # Tensor for loss
                
                # Track q_halt values for debugging
                if 'pass_q_halt' not in aux:
                    aux['pass_q_halt'] = []
                aux['pass_q_halt'].append(halt_aux['q_halt_logits'].mean().item())

                # Store legacy threshold for backward compat
                threshold_val = halt_aux['threshold'].mean().item()
                aux['model_halt_threshold'] = threshold_val
                
                # Store threshold TENSOR for loss computation (legacy)
                if 'model_halt_thresholds' not in aux:
                    aux['model_halt_thresholds'] = []
                aux['model_halt_thresholds'].append(halt_aux['threshold'])

                # === HALTING DECISION (Q-halt + Info Gain) ===
                # Combined logic:
                # - Pass 0: Never halt based on info_gain (no prev pass to compare)
                # - q_halt > 0 (thinks correct) → HALT
                # - q_halt < 0 + high info_gain → CONTINUE (making progress)
                # - q_halt < 0 + low info_gain → ABORT (stuck, wasting compute)
                
                # Use detached value for halt decision (no gradient needed here)
                info_gain = halt_aux['info_gain'].mean().detach().item()
                info_gain_threshold = 0.01  # Minimum info gain to justify continuing
                
                q_halt_says_halt = q_halt_logits.mean() > 0  # Model thinks answer is correct
                q_halt_says_wrong = q_halt_logits.mean() < 0  # Model thinks answer is wrong
                low_info_gain = info_gain < info_gain_threshold
                
                # Pass 0 can only halt if q_halt explicitly says correct
                # Later passes can halt if stuck (wrong + no progress)
                if pass_idx == 0:
                    should_halt = q_halt_says_halt
                else:
                    should_halt = q_halt_says_halt or (q_halt_says_wrong and low_info_gain)
                
                # Track reason for halting
                if should_halt and q_halt_says_halt:
                    aux['halt_reason'] = 'q_halt_correct'
                elif should_halt and low_info_gain:
                    aux['halt_reason'] = 'stuck_no_progress'
                
                # === WARMUP MODE: Skip early halting ===
                # During warmup, force max passes to let Q-halt learn from diverse pass counts
                if force_max_passes:
                    should_halt = False
                
                if should_halt:
                    if self.training:
                        # ε-greedy exploration during training (30% chance to continue)
                        explore = torch.rand(1, device=device).item() < 0.3
                        if not explore:
                            aux['halted_early'] = True
                            break
                    else:
                        aux['halted_early'] = True
                        break

            # Store for next pass (before any break)
            prev_logits = test_logits.detach()

        # Final output and confidence
        if return_aux and final_logits is not None:
            h_test_region = h[:, test_start_idx:test_start_idx + S]
            final_confidence, _, _ = self.model_halt_estimator(
                h_test_region,
                logits=final_logits,
                prev_logits=prev_logits,
                current_pass=aux.get('passes_run', 1) - 1,
            )
            aux['final_confidence'] = final_confidence
            
            # Compute ponder cost: penalize using more passes
            # Normalized by max_passes so cost is in [0, 1]
            if features.use_ponder_cost:
                passes_used = aux['passes_run']
                aux['ponder_cost'] = passes_used / self.max_passes

        return final_logits, cache, aux