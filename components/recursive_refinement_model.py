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

        # Learned slot embeddings (concept anchors)
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_layers, num_slots, d_cache) * 0.02
        )

        # Layer-ID embeddings for representational separation
        self.layer_id_embeddings = nn.Parameter(
            torch.randn(num_layers, d_cache // 4) * 0.02
        )

        # Shared confidence estimator (created BEFORE layers so it can be shared)
        self.confidence_estimator = ConfidenceEstimator(
            d_model=d_model,
            vocab_size=vocab_size,
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
        """Initialize cache with learned slot embeddings."""
        cache = self.slot_embeddings.view(self.total_slots, self.d_cache)
        return cache.unsqueeze(0).expand(batch_size, -1, -1).clone()

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

        # Track auxiliary information
        aux = {
            # Model-level stats
            'passes_run': 0,
            'max_passes': self.max_passes,
            'pass_confidences': [],
            'answer_feedback_gates': [],  # Model-level feedback gating
            
            # Layer-level stats (aggregated across all passes)
            'layer_iterations': [],
            'iteration_feedback_gates': [],  # Layer-level thought injection
            'confidences': [],
            'read_gates': [],
            'write_gates': [],
            
            # Compute tracking
            'total_layer_steps': 0,
            'temperature': temperature,
        } if return_aux else {}

        # Current sequence for input to the model
        current_seq = full_seq
        prev_logits = None
        final_logits = None

        # === MODEL-LEVEL RECURSIVE REFINEMENT LOOP ===
        for pass_idx in range(self.max_passes):
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
                    aux['answer_feedback_gates'].append(gate.mean().detach())
                
                # Clone and update current_seq
                current_seq = current_seq.clone()
                current_seq[:, test_start_idx:test_start_idx + S] = updated_test_region

            # Process through recursive layers WITHIN the current pass
            h = current_seq
            for layer_idx, layer in enumerate(self.layers):
                max_iterations = self.max_internal_iterations if features.use_layer_act else 1
                h, cache, layer_aux = layer(
                    h, cache,
                    cos_sin=cos_sin_full,
                    temperature=temperature,
                    hard=hard,
                    max_iterations=max_iterations,
                    confidence_threshold=self.confidence_threshold,
                )

                if return_aux and layer_aux:
                    iters_run = layer_aux.get('iterations_run', 1)
                    aux['layer_iterations'].append(iters_run)
                    aux['total_layer_steps'] += iters_run
                    aux['confidences'].extend(layer_aux.get('confidences', []))
                    aux['read_gates'].extend(layer_aux.get('read_gates', []))
                    aux['write_gates'].extend(layer_aux.get('write_gates', []))
                    # Track layer-level thought injection gates
                    if 'iteration_feedback_gates' in layer_aux:
                        aux['iteration_feedback_gates'].extend(layer_aux['iteration_feedback_gates'])

            # Extract logits for test output positions
            test_logits = self.output_proj(h[:, test_start_idx:test_start_idx + S])  # [B, S, vocab_size]
            final_logits = test_logits

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

                # Check if we should stop based on confidence
                if not self.training and current_confidence.mean() > self.confidence_threshold:
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

        return final_logits, cache, aux