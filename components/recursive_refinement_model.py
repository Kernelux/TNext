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

from .config import TrainingConfig, FeatureFlags
from .memory_controller import ConfidenceEstimator
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

        # === Slot Metadata Dimensions (must match MemoryController) ===
        # Metadata: confidence (1) + layer_embed (8) + iter_embed (4) + pass_embed (4) = 17
        self.d_meta = 17
        self.d_slot = d_cache + self.d_meta  # Total slot dimension

        # Learned slot embeddings (concept anchors) - content only
        # Metadata (confidence + temporal) is initialized to zero
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_layers, num_slots, d_cache) * 0.02
        )

        # Layer-ID embeddings for representational separation
        # Used in initial cache to give each layer's slots a distinct identity
        # Dimension matches d_layer_embed in MemoryController (8)
        self.layer_id_embeddings = nn.Parameter(
            torch.randn(num_layers, 8) * 0.02  # 8 = d_layer_embed
        )

        # Shared confidence estimator (created BEFORE layers so it can be shared)
        # Now includes learned halt thresholds (budget-aware)
        self.confidence_estimator = ConfidenceEstimator(
            d_model=d_model,
            vocab_size=vocab_size,
            max_iterations=max_internal_iterations,
            max_passes=max_passes,
        )

        # Unified Memory Layers (imported from unified_layer.py)
        self.layers = nn.ModuleList([
            UnifiedMemoryLayer(
                d_model=d_model,
                d_cache=d_cache,
                num_slots=num_slots,
                num_layers=num_layers,
                layer_idx=i,
                num_heads=num_heads,
                dropout=dropout,
                max_write_tokens=64,
                use_linear_attention=True,
                use_checkpoint=(max_seq_len > 500),  # Enable for large sequences
            )
            for i in range(num_layers)
        ])

        # Share confidence estimator with layers for recursive refinement
        for layer in self.layers:
            layer.confidence_estimator = self.confidence_estimator

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Answer feedback mechanism (TRM insight)
        self.prev_answer_embed = nn.Embedding(vocab_size, d_model)
        self.answer_feedback_gate = nn.Linear(d_model * 2, d_model)

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
        nn.init.zeros_(self.answer_feedback_gate.bias)

    def get_initial_cache(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize cache with learned slot embeddings + layer identity metadata.
        
        Cache structure: [content (d_cache) | confidence (1) | temporal (16)]
        - Content: learned slot embeddings
        - Confidence: 0 (empty slots have no confidence)
        - Temporal: [layer_id (8) | iter (4) | pass (4)]
          - layer_id: Learned layer identity embedding (gives gradient flow!)
          - iter: 0 (not yet written by any iteration)
          - pass: 0 (not yet written by any pass)
        """
        # Content from learned embeddings
        content = self.slot_embeddings.view(self.total_slots, self.d_cache)  # [L*K, D_cache]
        content = content.unsqueeze(0).expand(batch_size, -1, -1).clone()   # [B, L*K, D_cache]
        
        # Confidence: 0 for empty slots
        confidence = torch.zeros(batch_size, self.total_slots, 1, device=device)
        
        # Layer identity embedding: each layer's slots get that layer's embedding
        # This provides gradient flow to layer_id_embeddings!
        # Shape: [num_layers, 8] -> expand to [num_layers, num_slots, 8] -> [L*K, 8]
        layer_ids = self.layer_id_embeddings.unsqueeze(1).expand(
            -1, self.num_slots, -1
        ).reshape(self.total_slots, 8)  # [L*K, 8]
        layer_ids = layer_ids.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L*K, 8]
        
        # Iteration and pass embeddings: 0 (not yet written)
        iter_pass = torch.zeros(batch_size, self.total_slots, 8, device=device)  # 4 + 4 = 8
        
        # Combine temporal metadata: [layer_id (8) | iter (4) | pass (4)]
        temporal = torch.cat([layer_ids, iter_pass], dim=-1)  # [B, L*K, 16]
        
        # Full metadata: [confidence (1) | temporal (16)]
        metadata = torch.cat([confidence, temporal], dim=-1)  # [B, L*K, 17]
        
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

        # === MODEL-LEVEL RECURSIVE REFINEMENT LOOP ===
        # Use config.max_passes if provided (for debug scenarios), else model default
        effective_max_passes = min(config.max_passes, self.max_passes) if config.max_passes else self.max_passes
        for pass_idx in range(effective_max_passes):
            if return_aux:
                aux['passes_run'] = pass_idx + 1

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
                )

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
                    aux['read_thresholds'].extend(layer_aux.get('read_thresholds', []))
                    aux['write_thresholds'].extend(layer_aux.get('write_thresholds', []))
                    aux['layer_halt_thresholds'].extend(layer_aux.get('layer_halt_thresholds', []))
                    aux['layer_confidences'].extend(layer_aux.get('layer_confidences', []))

            # Extract logits for test output positions
            test_logits = self.output_proj(h[:, test_start_idx:test_start_idx + S])  # [B, S, vocab_size]
            final_logits = test_logits
            
            # Store logits for deep supervision (if enabled)
            if return_aux and features.use_deep_supervision:
                aux['pass_logits'].append(test_logits)

            # Confidence assessment for model-level halting
            if return_aux:
                # Calculate confidence of current output
                # ConfidenceEstimator expects [B, S, D] for h and [B, S, V] for logits
                h_test_region = h[:, test_start_idx:test_start_idx + S]  # [B, S, D_model]
                current_confidence = self.confidence_estimator(
                    h_test_region,
                    logits=test_logits,
                    prev_logits=prev_logits,
                )
                aux['pass_confidences'].append(current_confidence.detach())

                # Get learned halt threshold (budget + confidence aware)
                # Threshold sees: h_pooled, confidence, pass_ratio, budget_remaining
                # Low confidence + high budget → higher threshold (keep going)
                # High confidence OR low budget → lower threshold (halt)
                halt_threshold = self.confidence_estimator.get_model_halt_threshold(
                    h_test_region,
                    confidence=current_confidence,  # No detach - allow gradient flow for learning
                    current_pass=pass_idx,
                    max_pass=self.max_passes,
                )
                threshold_val = halt_threshold.mean().detach().item()
                aux['model_halt_threshold'] = threshold_val  # Scalar for debugging
                
                # Store threshold TENSOR (with gradients) for loss computation
                # This allows threshold network to receive supervision
                if 'model_halt_thresholds' not in aux:
                    aux['model_halt_thresholds'] = []
                aux['model_halt_thresholds'].append(halt_threshold)  # [B] with gradients

                # === HALTING DECISION ===
                # Check if confident enough to halt
                should_halt = current_confidence.mean() > threshold_val
                
                if should_halt:
                    if self.training:
                        # During training: ε-greedy exploration
                        # With prob ε, ignore confidence and continue anyway
                        explore = torch.rand(1, device=device).item() < features.halt_exploration_prob
                        if not explore:
                            aux['halted_early'] = True
                            break  # Halt: confident and not exploring
                        # else: continue despite confidence (exploration)
                    else:
                        # During inference: always respect confidence
                        aux['halted_early'] = True
                        break

            # Store for next pass (before any break)
            prev_logits = test_logits.detach()

        # Final output and confidence
        if return_aux and final_logits is not None:
            h_test_region = h[:, test_start_idx:test_start_idx + S]
            final_confidence = self.confidence_estimator(
                h_test_region,
                logits=final_logits,
            )
            aux['final_confidence'] = final_confidence
            
            # Compute ponder cost: penalize using more passes
            # Normalized by max_passes so cost is in [0, 1]
            if features.use_ponder_cost:
                passes_used = aux['passes_run']
                aux['ponder_cost'] = passes_used / self.max_passes

        return final_logits, cache, aux