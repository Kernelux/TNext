"""
Volatile Memory Controller (Working Memory)
============================================

A "clipboard" style memory that complements the main cache (LTM).

Key Differences from MemoryController:
1. READ clears/decays the slot (use-once semantics)
2. WRITE uses harder slot assignment (winner-take-all)
3. Simpler slot structure (content + freshness only)
4. Fewer slots (working memory is limited capacity)

Together with LTM cache:
- LTM: "What rules/patterns have I learned?" (persistent)
- WM:  "What am I currently holding?" (transient, clipboard)

Use Cases in ARC:
- Copy/paste: Store source pixel, use at destination
- Intermediate values: Hold partial computation results
- Pattern buffering: Store sequence to repeat elsewhere

Slot Structure: [content (D_slot) | freshness (1)]
- content: The actual stored representation
- freshness: How recently written (decays on read, refreshes on write)

Based on cognitive science working memory models:
- Limited capacity (typically 4-7 items in humans)
- Rapid decay without rehearsal
- Content-addressable but volatile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


class VolatileMemoryController(nn.Module):
    """
    Working Memory with use-once semantics.
    
    Read Operation:
        1. Query slots by content similarity
        2. Retrieve weighted combination
        3. DECAY slots that were read (key difference from LTM)
        
    Write Operation:
        1. Score token importance (xLSTM-style exp gating)
        2. Select target slot (hard winner-take-all)
        3. OVERWRITE slot completely (no blending)
        4. Set freshness to 1.0
        
    The "freshness" field enables:
        - Prioritizing recently written slots in reads
        - Natural decay of unused information
        - Detecting "empty" vs "stale" slots
    """
    
    def __init__(
        self,
        d_model: int,
        d_slot: int,
        num_slots: int = 8,  # Working memory is small (cognitive limit ~7±2)
        dropout: float = 0.1,
        read_decay: float = 0.3,  # How much freshness decays per read (0=instant clear, 1=no decay)
        freshness_threshold: float = 0.1,  # Below this, slot is considered "empty"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_slot = d_slot
        self.num_slots = num_slots
        self.read_decay = read_decay
        self.freshness_threshold = freshness_threshold
        
        # Full slot: content + freshness
        self.d_full_slot = d_slot + 1
        
        # === Read Components ===
        # Query projection: "What am I looking for in WM?"
        self.read_query = nn.Linear(d_model, d_slot)
        
        # Output projection: slot content -> model space
        self.from_wm = nn.Linear(d_slot, d_model)
        
        # Read gate: "Do I need working memory here?" (Griffin-style)
        self.read_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # === Write Components ===
        # Input projection: model space -> slot space
        self.to_wm = nn.Linear(d_model, d_slot)
        
        # Write query: "Which slot should I write to?"
        self.write_query = nn.Linear(d_model, d_slot)
        
        # Importance scorer (xLSTM-style: raw logit, exp in forward)
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Write decision: "Should I write at all?"
        self.write_decision = nn.Sequential(
            nn.Linear(d_model + d_slot, d_model // 2),  # input + WM context
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Learnable temperature for write gating
        self.write_temperature = nn.Parameter(torch.tensor(1.0))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.read_query, self.from_wm, self.to_wm, self.write_query, self.fusion]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        for seq in [self.read_gate, self.importance_scorer, self.write_decision]:
            for m in seq:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Bias write decision slightly positive (encourage writes early)
        if self.write_decision[-1].bias is not None:
            nn.init.constant_(self.write_decision[-1].bias, 0.5)
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize empty working memory.
        
        Returns:
            wm_state: [B, num_slots, d_full_slot] with zeros (empty slots)
        """
        # All zeros: empty content, zero freshness
        return torch.zeros(batch_size, self.num_slots, self.d_full_slot, device=device)
    
    def _split_slot(self, wm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split slot into content and freshness."""
        content = wm[..., :self.d_slot]      # [B, K, D_slot]
        freshness = wm[..., -1:]              # [B, K, 1]
        return content, freshness
    
    def _merge_slot(self, content: torch.Tensor, freshness: torch.Tensor) -> torch.Tensor:
        """Merge content and freshness into full slot."""
        return torch.cat([content, freshness], dim=-1)
    
    def read(
        self,
        x: torch.Tensor,           # [B, S, D_model]
        wm: torch.Tensor,          # [B, K, D_full_slot]
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Read from working memory with decay-on-read semantics.
        
        Process:
        1. Query slots by content similarity (weighted by freshness)
        2. Gate whether to use WM context
        3. Decay freshness of read slots
        
        Args:
            x: Input tensor [B, S, D_model]
            wm: Working memory state [B, K, D_full_slot]
            hard: Use hard gating (inference)
            
        Returns:
            Dict with:
                - 'x_enhanced': Input fused with WM context [B, S, D_model]
                - 'wm_updated': WM with decayed freshness [B, K, D_full_slot]
                - 'read_gate': Gate values [B, S, 1]
                - 'attn_weights': Attention over slots [B, S, K]
        """
        B, S, _ = x.shape
        K = self.num_slots
        
        # Split WM
        content, freshness = self._split_slot(wm)  # [B, K, D_slot], [B, K, 1]
        
        # === Step 1: Compute read gate ===
        gate = self.read_gate(x)  # [B, S, 1]
        if hard:
            gate = (gate > 0.5).float()
        
        # === Step 2: Query working memory ===
        query = self.read_query(x)  # [B, S, D_slot]
        
        # Attention scores (content-based)
        scores = torch.matmul(query, content.transpose(-2, -1))  # [B, S, K]
        scores = scores / math.sqrt(self.d_slot)
        
        # Weight by freshness: fresh slots are more relevant
        # freshness: [B, K, 1] -> [B, 1, K]
        freshness_weight = freshness.transpose(-2, -1)  # [B, 1, K]
        
        # Mask stale slots (freshness below threshold)
        stale_mask = (freshness_weight < self.freshness_threshold)
        scores = scores.masked_fill(stale_mask, -1e9)
        
        # Softmax attention
        attn_weights = F.softmax(scores, dim=-1)  # [B, S, K]
        attn_weights = self.dropout(attn_weights)
        
        # Retrieve context
        context = torch.matmul(attn_weights, content)  # [B, S, D_slot]
        context = self.from_wm(context)  # [B, S, D_model]
        
        # === Step 3: Gated fusion ===
        combined = torch.cat([x, context], dim=-1)
        fused = self.fusion(combined)
        x_enhanced = (1 - gate) * x + gate * fused
        
        # === Step 4: Decay freshness based on read attention ===
        # Slots that were attended to more get decayed more
        # read_pressure[k] = sum over positions of attn_weights[:, :, k]
        read_pressure = attn_weights.sum(dim=1, keepdim=True)  # [B, 1, K]
        read_pressure = read_pressure.transpose(-2, -1)  # [B, K, 1]
        
        # Normalize read pressure to [0, 1]
        max_pressure = read_pressure.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        read_pressure_norm = read_pressure / max_pressure
        
        # Decay: freshness *= (1 - read_pressure * (1 - read_decay))
        # If read_pressure=1 and read_decay=0.3, decay_factor=0.3
        # If read_pressure=0, decay_factor=1.0 (no decay)
        decay_factor = 1.0 - read_pressure_norm * (1.0 - self.read_decay)
        new_freshness = freshness * decay_factor
        
        # Merge back
        wm_updated = self._merge_slot(content, new_freshness)
        
        return {
            'x_enhanced': x_enhanced,
            'wm_updated': wm_updated,
            'read_gate': gate,
            'attn_weights': attn_weights,
            'context': context,
        }
    
    def write(
        self,
        x: torch.Tensor,           # [B, S, D_model]
        wm: torch.Tensor,          # [B, K, D_full_slot]
        hard: bool = False,
        mask: Optional[torch.Tensor] = None,  # [B, S] or [B, S, 1]
    ) -> Dict[str, torch.Tensor]:
        """
        Write to working memory with hard slot assignment.
        
        Process:
        1. Score token importance (which tokens want to write)
        2. Decide whether to write (based on input vs current WM)
        3. Select target slot (winner-take-all)
        4. Overwrite slot completely, set freshness=1.0
        
        Args:
            x: Input tensor [B, S, D_model]
            wm: Working memory state [B, K, D_full_slot]
            hard: Use hard decisions
            mask: Optional mask for valid tokens
            
        Returns:
            Dict with:
                - 'wm_updated': Updated WM [B, K, D_full_slot]
                - 'write_gate': Which tokens wrote [B, S, 1]
                - 'importance': Token importance [B, S, 1]
                - 'slot_selection': Which slot each token targeted [B, S, K]
        """
        B, S, _ = x.shape
        K = self.num_slots
        device = x.device
        
        # Split WM
        old_content, old_freshness = self._split_slot(wm)
        
        # Project to slot space
        x_slot = self.to_wm(x)  # [B, S, D_slot]
        
        # === Step 1: Importance scoring (xLSTM-style) ===
        importance_logit = self.importance_scorer(x)  # [B, S, 1]
        importance = torch.exp(importance_logit / self.write_temperature.clamp(min=0.1))
        # Normalize across sequence
        importance = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)
        importance = importance * S  # Re-scale
        
        # === Step 2: Write decision ===
        # Compare input to what's in WM (via attention)
        query_for_decision = self.write_query(x)  # [B, S, D_slot]
        wm_context = torch.matmul(
            F.softmax(torch.matmul(query_for_decision, old_content.transpose(-2, -1)) / math.sqrt(self.d_slot), dim=-1),
            old_content
        )  # [B, S, D_slot]
        
        decision_input = torch.cat([x, wm_context], dim=-1)
        decision_logit = self.write_decision(decision_input)  # [B, S, 1]
        
        # xLSTM-style exp gating
        decision_exp = torch.exp(decision_logit / self.write_temperature.clamp(min=0.1))
        write_gate = decision_exp / (1.0 + decision_exp)  # Soft sigmoid
        
        if hard:
            write_gate = (write_gate > 0.5).float()
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            write_gate = write_gate * mask
        
        # === Step 3: Slot selection (winner-take-all) ===
        # Query which slot to write to
        write_query = self.write_query(x)  # [B, S, D_slot]
        
        # Prefer stale slots (low freshness) to avoid overwriting fresh data
        # slot_scores = content_similarity - freshness_penalty
        content_scores = torch.matmul(write_query, old_content.transpose(-2, -1))  # [B, S, K]
        content_scores = content_scores / math.sqrt(self.d_slot)
        
        # Freshness penalty: prefer to overwrite stale slots
        freshness_penalty = old_freshness.transpose(-2, -1) * 2.0  # [B, 1, K] -> scale
        slot_scores = content_scores - freshness_penalty
        
        # Hard slot selection (winner-take-all for WM)
        if hard or not self.training:
            slot_selection = F.one_hot(slot_scores.argmax(dim=-1), K).float()  # [B, S, K]
        else:
            # Gumbel-softmax for differentiable hard selection
            slot_selection = F.gumbel_softmax(slot_scores, tau=0.5, hard=True)
        
        # === Step 4: Update slots ===
        # Only tokens with write_gate > 0 actually write
        # For each slot, find which token (if any) writes to it
        
        # Weighted slot assignment: [B, S, K] * [B, S, 1] -> [B, S, K]
        write_weights = slot_selection * write_gate * importance
        
        # Aggregate: which tokens write to each slot? [B, K, S]
        write_weights_t = write_weights.transpose(1, 2)  # [B, K, S]
        
        # Total write weight per slot
        slot_write_total = write_weights_t.sum(dim=-1, keepdim=True)  # [B, K, 1]
        
        # New content: weighted average of writing tokens
        # [B, K, S] @ [B, S, D_slot] -> [B, K, D_slot]
        new_content_raw = torch.matmul(write_weights_t, x_slot)
        
        # Normalize by total weight (avoid div by zero)
        slot_write_safe = slot_write_total.clamp(min=1e-8)
        new_content = new_content_raw / slot_write_safe
        
        # Determine which slots received writes
        has_writes = (slot_write_total > 0.1).float()  # [B, K, 1]
        
        # HARD overwrite: slots with writes get completely replaced
        final_content = has_writes * new_content + (1 - has_writes) * old_content
        
        # Freshness: 1.0 for written slots, unchanged for others
        final_freshness = has_writes * 1.0 + (1 - has_writes) * old_freshness
        
        # Merge back
        wm_updated = self._merge_slot(final_content, final_freshness)
        
        return {
            'wm_updated': wm_updated,
            'write_gate': write_gate,
            'importance': importance,
            'slot_selection': slot_selection,
            'num_writes': write_gate.sum(dim=1).squeeze(-1),  # [B]
        }
    
    def forward(
        self,
        x: torch.Tensor,
        wm: torch.Tensor,
        hard: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full read-then-write cycle.
        
        Returns combined results from read and write operations.
        """
        # Read first (get context, decay freshness)
        read_result = self.read(x, wm, hard=hard)
        
        # Write to updated WM (with decayed freshness)
        write_result = self.write(x, read_result['wm_updated'], hard=hard, mask=mask)
        
        return {
            'x_enhanced': read_result['x_enhanced'],
            'wm_updated': write_result['wm_updated'],
            'read_gate': read_result['read_gate'],
            'write_gate': write_result['write_gate'],
            'read_attn': read_result['attn_weights'],
            'write_slot_selection': write_result['slot_selection'],
        }
