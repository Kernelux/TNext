import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

from .config import FeatureFlags
from .modules import MemoryRouter, SelectionHead

class DLSMNLayer(nn.Module):
    """
    DLSMN Layer following DLSM_V0.1.md specification.
    
    Implements the Compute-Select-Cache cycle (Section 3.1):
    1. READ: Query cache for relevant context
    2. COMPUTE: Head A processes input (± context)
    3. SELECT: Head B decides what to cache and where
    """
    
    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        d_cache: int,
        num_slots: int,
        num_layers: int,
        num_patterns: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.d_cache = d_cache
        self.num_slots = num_slots
        self.num_patterns = num_patterns
        
        # Layer-ID embedding (Section 7.6)
        self.layer_embed = nn.Parameter(torch.randn(1, 1, d_cache // 4) * 0.02)
        self.d_layer = d_cache // 4
        
        # === DUAL-INPUT PROCESSING ===
        # The layer takes two inputs:
        #   1. x: input tokens [B, S, D_model]
        #   2. memory_context: read from cache [B, S, D_model] or None
        
        # Cross-attention: x attends to memory (if available)
        self.memory_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        
        # Self-attention on (x + memory info)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Memory fusion gate: learns how much to use memory per-token
        # g = σ(W · [x, memory]) → how much memory to incorporate
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        # "No memory" embedding - learnable sentinel for when no read happens
        self.no_memory_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Pattern pooling: learnable queries that extract pattern summaries
        self.pattern_queries = nn.Parameter(torch.randn(num_patterns, d_model) * 0.02)
        self.pattern_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Head B: Selection (Section 2.2)
        self.selection_head = SelectionHead(d_model, d_cache, num_slots)
        
        # MoE-style Memory Router (decides read/write per token)
        self.memory_router = MemoryRouter(d_model, d_cache, num_slots)
        
        # Cache projections (Section 2.1)
        # W_compress: D_model → D_cache (per-layer)
        self.W_compress = nn.Linear(d_model, d_cache)
        # W_decompress: D_cache → D_model (per-layer)
        self.W_decompress = nn.Linear(d_cache + self.d_layer, d_model)  # +layer_id
        
        # Cache read attention (Section 3.1 Step 1)
        self.cache_query = nn.Linear(d_model, d_cache)
        # +layer_id + age (1 dim)
        self.cache_key = nn.Linear(d_cache + self.d_layer + 1, d_cache)  
        self.cache_value = nn.Linear(d_cache + self.d_layer + 1, d_cache)  # +layer_id + age
        
    def read_cache(
        self, 
        x: torch.Tensor,           # [B, S, D_model]
        cache: torch.Tensor,       # [B, L*K, D_cache]
        layer_ids: torch.Tensor,   # [L*K, D_layer] - layer embeddings for each slot
        cache_mask: Optional[torch.Tensor] = None,  # [B, L*K] True=blocked
        write_counts: Optional[torch.Tensor] = None, # [B, L*K] number of writes per slot
        slot_ages: Optional[torch.Tensor] = None,    # [B, L*K] age since last write
        features: Optional[FeatureFlags] = None,
        read_gate: Optional[torch.Tensor] = None,    # [B, S, 1] MoE read gate
        read_slot_probs: Optional[torch.Tensor] = None,  # [B, S, K] MoE slot selection
    ) -> torch.Tensor:
        """
        Step 1: READ - Retrieve context from cache (Section 3.1)
        
        With MoE Memory Router:
        - read_gate: per-token decision of whether to read at all
        - read_slot_probs: which slots each token should attend to
        
        q = W_Q · x
        α = Softmax(q · (W_K · [C; layer_id])^T / √d + M)
        raw_context = α · (W_V · [C; layer_id])
        context = W_decompress · raw_context
        """
        if features is None:
            features = FeatureFlags()
            
        B, S, _ = x.shape
        _, total_slots, _ = cache.shape
        
        # Concatenate layer-ID embeddings to cache entries (Section 7.6)
        cache_with_id = torch.cat([cache, layer_ids.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        
        # [REFINEMENT: use_temporal_decay]
        # Section 8.4: Append age to cache key projection
        if features.use_temporal_decay and slot_ages is not None:
            # Normalize age (simple scaling)
            normalized_age = slot_ages.unsqueeze(-1) / 100.0  # Scale down
            cache_with_id = torch.cat([cache_with_id, normalized_age], dim=-1)
        else:
            # Append zero age if feature disabled or ages not provided
            zeros = torch.zeros(B, total_slots, 1, device=x.device)
            cache_with_id = torch.cat([cache_with_id, zeros], dim=-1)
        
        q = self.cache_query(x)  # [B, S, D_cache]
        k = self.cache_key(cache_with_id)  # [B, L*K, D_cache]
        v = self.cache_value(cache_with_id)  # [B, L*K, D_cache]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_cache)
        
        # Apply pass-aware mask (Section 2.3)
        if cache_mask is not None:
            mask = cache_mask.unsqueeze(1).expand(-1, S, -1)
            scores = scores.masked_fill(mask, float('-inf'))
            
        # [REFINEMENT: use_write_count_masking]
        # Section 7 & 8.1: Mask unwritten slots to mitigate Cold Start
        if features.use_write_count_masking and write_counts is not None:
            # Mask where write_count == 0
            unwritten = (write_counts == 0).unsqueeze(1).expand(-1, S, -1)
            scores = scores.masked_fill(unwritten, float('-inf'))
        
        # [MoE MEMORY: Apply learned slot selection]
        # read_slot_probs from MemoryRouter tells us which slots each token prefers
        if features.use_moe_memory and read_slot_probs is not None:
            # read_slot_probs: [B, S, K] for THIS layer's slots
            # We need to expand to cover all L*K slots, but only boost this layer's slots
            # For now, apply to this layer's K slots within total_slots
            # This layer's slots are at indices [layer_idx*K : (layer_idx+1)*K]
            start_idx = self.layer_idx * self.num_slots
            end_idx = start_idx + self.num_slots
            
            # Create a bias that boosts attention to preferred slots
            # High read_slot_weight = more attention to that slot
            slot_bias = torch.zeros(B, S, total_slots, device=x.device)
            slot_bias[:, :, start_idx:end_idx] = read_slot_probs * 5.0  # Scale for softmax
            scores = scores + slot_bias
        
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        
        raw_context = torch.matmul(attn, v)  # [B, S, D_cache]
        
        # Decompress with layer info
        raw_context_with_id = torch.cat([
            raw_context, 
            self.layer_embed.expand(B, S, -1)
        ], dim=-1)
        context = self.W_decompress(raw_context_with_id)  # [B, S, D_model]
        
        # [MoE MEMORY: Apply read gate]
        # If read_gate is low, token doesn't want cache context → return zeros
        if features.use_moe_memory and read_gate is not None:
            context = context * read_gate  # [B, S, D] * [B, S, 1]
        
        return context
    
    def process_with_memory(
        self, 
        x: torch.Tensor,                              # [B, S, D_model] - input tokens
        memory_context: Optional[torch.Tensor],       # [B, S, D_model] - from cache, or None
        features: Optional[FeatureFlags] = None,
    ) -> torch.Tensor:
        """
        Process input with optional memory context.
        
        Two inputs:
          - x: the actual input tokens
          - memory_context: retrieved from cache (or None if no read)
        
        The layer decides how to combine them:
          1. If memory_context is None → use no_memory_embed as placeholder
          2. Cross-attention: x queries the memory
          3. Gated fusion: learn how much memory to incorporate
          4. Self-attention + FFN on the fused representation
        """
        if features is None:
            features = FeatureFlags()
            
        B, S, D = x.shape
        
        # Handle "no memory" case with learnable sentinel
        if memory_context is None:
            # Expand no_memory_embed to match input shape
            memory_context = self.no_memory_embed.expand(B, S, -1)
            has_memory = False
        else:
            has_memory = True
        
        # Option 1: Cross-attention (x attends to memory)
        # This lets each token selectively pull from its memory context
        if has_memory and features.use_gated_fusion:
            # x queries memory: "what in my memory is relevant to me?"
            cross_out, _ = self.memory_cross_attn(x, memory_context, memory_context)
            x_with_mem = self.norm_cross(x + cross_out)
            
            # Gated fusion: how much to trust memory vs original input
            gate = self.memory_gate(torch.cat([x, x_with_mem], dim=-1))
            x_fused = gate * x_with_mem + (1 - gate) * x
        else:
            # Simple additive (or no memory)
            x_fused = x + memory_context if has_memory else x
        
        # Self-attention on fused representation
        attn_out, _ = self.self_attn(x_fused, x_fused, x_fused)
        x_fused = self.norm1(x_fused + attn_out)
        
        # FFN
        ffn_out = self.ffn(x_fused)
        y = self.norm2(x_fused + ffn_out)
        
        return y
    
    def forward(
        self, 
        x: torch.Tensor,                           # [B, S, D_model]
        cache: torch.Tensor,                       # [B, L*K, D_cache]
        slot_embeddings: torch.Tensor,             # [K, D_cache] - this layer's slot anchors
        layer_ids: torch.Tensor,                   # [L*K, D_layer] - all layer embeddings
        cache_mask: Optional[torch.Tensor] = None, # [B, L*K]
        write_counts: Optional[torch.Tensor] = None,
        slot_ages: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = False,
        features: Optional[FeatureFlags] = None,
        step: int = 0,
        exploration_steps: int = 0,
        exploration_start: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        DUAL-INPUT Compute-Select-Cache cycle.
        
        Flow:
          1. READ: Get memory context (or None if no read)
          2. PROCESS: Layer takes (x, memory_context) and decides fusion
          3. SELECT: Decide what/where to write
        """
        if features is None:
            features = FeatureFlags()
            
        B, S, _ = x.shape
        
        # === STEP 1: READ from cache ===
        memory_context = None
        if features.use_cache:
            # Get routing decisions (should I read? what slots?)
            mem_routing = None
            if features.use_moe_memory:
                mem_routing = self.memory_router(x, temperature=temperature, hard=hard, mode='read')
            
            read_gate = mem_routing['read_gate'] if mem_routing else None
            read_slot_probs = mem_routing['read_slot_probs'] if mem_routing else None
            
            # Read from cache → memory_context [B, S, D_model]
            memory_context = self.read_cache(
                x, cache, layer_ids, cache_mask, 
                write_counts=write_counts, slot_ages=slot_ages, features=features,
                read_gate=read_gate, read_slot_probs=read_slot_probs
            )
        
        # === STEP 2: PROCESS with dual inputs ===
        # Layer receives (x, memory_context) and decides how to combine
        y = self.process_with_memory(x, memory_context, features)
        
        # === Pattern extraction for writing ===
        if features.use_pattern_pooling:
            queries = self.pattern_queries.unsqueeze(0).expand(B, -1, -1)
            patterns, _ = self.pattern_attn(queries, y, y)
        else:
            if S <= self.num_patterns:
                patterns = F.pad(y, (0, 0, 0, self.num_patterns - S))
            else:
                patterns = y[:, :self.num_patterns, :]
        
        # === STEP 3: SELECT - decide what to write and where ===
        if features.use_moe_memory:
            # Route on patterns (post-computation)
            pattern_routing = self.memory_router(patterns, temperature=temperature, hard=hard, mode='write')
            
            selection = {
                'scores': pattern_routing['write_scores'],
                'slot_probs': pattern_routing['write_slot_probs'],
                'soft_probs': F.softmax(
                    self.memory_router.write_slot_router(patterns) * self.memory_router.gamma, 
                    dim=-1
                ),
                'alpha': torch.ones(B, patterns.shape[1], device=x.device),
            }
        elif features.use_selection_head:
            selection = self.selection_head(
                patterns, slot_embeddings, temperature, hard, features,
                step=step,
                exploration_steps=exploration_steps,
                exploration_start=exploration_start
            )
        else:
            num_p = patterns.shape[1]
            K = self.num_slots
            selection = {
                'scores': torch.ones(B, num_p, device=x.device),
                'slot_probs': torch.ones(B, num_p, K, device=x.device) / K,
                'soft_probs': torch.ones(B, num_p, K, device=x.device) / K,
                'alpha': torch.ones(B, num_p, device=x.device),
            }
        
        # Compress patterns to cache space
        patterns_cache = self.W_compress(patterns)
        
        # Compute entropy for diversity loss
        soft_probs = selection['soft_probs']
        entropy = -(soft_probs * torch.log(soft_probs + 1e-8)).sum(dim=-1).mean()
        
        # Compute slot usage distribution
        slot_counts = selection['slot_probs'].sum(dim=1)
        
        cache_updates = {
            'y_cache': patterns_cache,
            'scores': selection['scores'],
            'slot_probs': selection['slot_probs'],
            'soft_probs': soft_probs,
            'slot_counts': slot_counts,
            'entropy': entropy,
            'alpha': selection['alpha'],
            'patterns': patterns,
            # [LOGGING] Detailed stats
            'read_gate': read_gate if read_gate is not None else torch.zeros(B, S, 1, device=x.device),
            'read_slot_probs': read_slot_probs if read_slot_probs is not None else torch.zeros(B, S, self.num_slots, device=x.device),
            'write_gate': pattern_routing['write_gate'] if features.use_moe_memory else torch.zeros(B, patterns.shape[1], 1, device=x.device),
        }
        
        return y, cache_updates
