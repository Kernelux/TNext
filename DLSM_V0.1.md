# Dual-Head Layered Selective Memory Network (DLSMN)

> A memory-augmented neural architecture with hierarchical, selective, and structured memory using a unified global cache.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Architecture](#2-core-architecture)
3. [Memory Operations](#3-memory-operations)
4. [Design Goals & Novelty](#4-design-goals--novelty)
5. [Related Work](#5-related-work)
6. [Comparison with Existing Approaches](#6-comparison-with-existing-approaches)
7. [Challenges & Mitigations](#7-challenges--mitigations)
8. [Architectural Refinements](#8-architectural-refinements)
9. [Training Stability & Curriculum](#9-training-stability--curriculum)
10. [Auxiliary Losses](#10-auxiliary-losses)
11. [Advanced Extensions](#11-advanced-extensions)
12. [RNN/Recursive Mode (DLSMN-RNN)](#12-rnnrecursive-mode-dlsmn-rnn)
13. [Summary](#13-summary)

---

## 1. Executive Summary

The **Dual-Head Layered Selective Memory Network (DLSMN)** augments each layer of a neural network with selective caching capabilities. Unlike standard Transformers that cache all Key-Value pairs, or RNNs that compress history into a single vector, this architecture makes memory:

- **Hierarchical** — each layer writes to its own partition of a global cache
- **Selective** — only salient features are cached (gated by importance)
- **Structured** — one contiguous cache tensor enables simple cross-layer access

This transforms each layer from a static function $f(x) = y$ into a **stateful system** $f(x, cache) = (y, cache')$.

---

## 2. Core Architecture

### 2.1 Global Cache Structure

A single contiguous cache tensor stores all layer memories:

$$\mathbf{C} \in \mathbb{R}^{(L \times K) \times D_{cache}}$$

Where:
- $L$ = number of layers
- $K$ = slots per layer
- $D_{cache}$ = cache dimension (may differ from model hidden dimension $D_{model}$)

**Dimension Alignment:** Each layer has its own projections to/from a shared $D_{cache}$ space:

- **Write projection (per-layer):** $y_{cache} = W_{compress}^{(j)} \cdot y \quad \text{where } W_{compress}^{(j)} \in \mathbb{R}^{D_{cache} \times D_{model}}$
- **Read projection (per-layer):** $context = W_{decompress}^{(j)} \cdot raw\_context \quad \text{where } W_{decompress}^{(j)} \in \mathbb{R}^{D_{model} \times D_{cache}}$

The shared $D_{cache}$ space acts as a **lingua franca** — each layer encodes/decodes through its own "lens" while maintaining cross-layer compatibility.

```
┌───────────────────────────────────────────────────────────────┐
│                   GLOBAL CACHE  [L×K, D_cache]                │
├───────────────────────────────────────────────────────────────┤
│  Layer 1:  [slot_0, slot_1, ..., slot_{K-1}]  ← idx 0:K       │
│  Layer 2:  [slot_0, slot_1, ..., slot_{K-1}]  ← idx K:2K      │
│  Layer 3:  [slot_0, slot_1, ..., slot_{K-1}]  ← idx 2K:3K     │
│  ...                                                          │
│  Layer L:  [slot_0, slot_1, ..., slot_{K-1}]  ← idx (L-1)K:LK │
└───────────────────────────────────────────────────────────────┘
```

**Access patterns:**
- **Local (own layer):** `C[j*K : (j+1)*K]` — Layer $j$'s dedicated slots
- **Global (all layers):** `C[:]` — Full cache for cross-layer reads

### 2.2 Per-Layer Dual Heads

Each layer $j$ processes input $x$ through two sequential components:

#### Head A: Computation Head ("The Worker")

- Performs standard transformations (Self-Attention, MLP, Convolution, etc.)
- Produces output feature map $y$

#### Head B: Selection Head ("The Gatekeeper")

- Analyzes output $y$ (after Head A)
- Outputs two decisions per token:
  1. **Importance score:** Should this token be cached?
  2. **Slot index:** Which slot to write to?

**Concrete specification:**
```python
# For each token's output y at layer j:
score = σ(W_gate · y)              # importance ∈ [0, 1]
slot_logits = W_slot · y           # K logits for slot selection
slot_logits = γ[j] * slot_logits   # attention sharpening (γ learned per-layer)
slot_idx = argmax(slot_logits)     # hard slot selection

if score > τ:
    y_cache = W_compress[j] · y    # per-layer projection to cache space
    C[j*K + slot_idx] ← aggregate(C[j*K + slot_idx], y_cache)
```

**Attention sharpening (γ):** A learnable scalar per layer that controls slot selection precision. High γ → sharper selection (more confident routing). Low γ → softer selection (more exploration). Initialize γ = 1.0.

**Gradient flow:** Uses Straight-Through Estimator (STE) for hard argmax — forward pass uses discrete index, backward pass flows gradients through softmax(slot_logits).

```
Input x
    │
    ▼
┌─────────┐
│ Head A  │
│ Compute │
└────┬────┘
     │
     ▼
  Output y ─────────────┐
     │                  ▼
     │            ┌───────────┐
     │            │  Head B   │
     │            │  Select   │
     │            └─────┬─────┘
     │                  │
     │                  ▼
     │            Cache y?
     │                  │
     │            If yes: write to C[j*K : (j+1)*K]
     ▼
  Next Layer
```

### 2.3 Cross-Layer Cache Access (Pass-Aware Masking)

Layers can read from the global cache with **pass-aware masking**:

| Pass | Layer $j$ Can Read From | Rationale |
|------|-------------------------|-----------|
| **Pass 1** | `C[0 : j*K]` (layers 1 to j-1) | Lower layers already computed this pass |
| **Pass 2+** | `C[:]` (all layers) | Full cache populated from previous pass |

```
Pass 1 (Bottom-Up):                    Pass 2+ (Full Access):
┌─────────────────────────┐            ┌─────────────────────────┐
│ Layer 3: read C[0:2K]   │            │ Layer 3: read C[:]      │
│ Layer 2: read C[0:K]    │            │ Layer 2: read C[:]      │
│ Layer 1: read ∅         │            │ Layer 1: read C[:]      │
└─────────────────────────┘            └─────────────────────────┘
```

**Implementation:** Standard attention mask (like causal masks in Transformers).

### 2.4 Intuitive Visualization

> **Corporate Building Analogy:**
> - **Standard Network:** Information flows from mailroom to CEO. No one writes anything down.
> - **DLSMN:** Each floor has a worker (Head A) and a secretary (Head B). After processing, the secretary decides: *"Important—file it"* or *"Noise—discard."* All floors share one filing system, so Floor 5 can check what Floor 2 filed earlier.

---

## 3. Memory Operations

### 3.1 The Compute-Select-Cache Cycle

For each layer $j$, per input:

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. READ (optional)  →  2. COMPUTE (Head A)  →  3. SELECT (Head B)  │
│                                                                     │
│  Query cache for         Process input           Decide: cache?     │
│  relevant context        (± context)             If yes → write     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Step 1: READ — Retrieve Context from Cache (Optional)

**Query generation:**
$$q = W_Q \cdot x$$

**Attention over accessible cache:**
$$\alpha = \text{Softmax}\left( \frac{q \cdot (W_K \cdot \mathbf{C}_{accessible})^T}{\sqrt{d}} + \mathbf{M} \right)$$

where $\mathbf{M}$ is the pass-aware mask.

**Fetch raw context (in cache space):**
$$raw\_context = \alpha \cdot (W_V \cdot \mathbf{C}_{accessible})$$

**Project to layer's model space:**
$$context = W_{decompress}^{(j)} \cdot raw\_context$$

#### Step 2: COMPUTE — Process Input

```python
# Option A: No fusion (simple)
y = HeadA(x)

# Option B: With cache context
x_fused = Fuse(x, context)
y = HeadA(x_fused)
```

#### Step 3: SELECT & WRITE — Cache Worthy Outputs

**Per-token selection:** Each token in the sequence independently decides:

$$score_t = \sigma(W_{gate} \cdot y_t) \in [0, 1]$$
$$slot_t = \arg\max(W_{slot} \cdot y_t)$$

**Write decision:** If $score_t > \tau$, token $t$ writes to its selected slot.

**Projection to cache space (per-layer):**
$$y_{cache,t} = W_{compress}^{(j)} \cdot y_t$$

**Collision handling:** Multiple tokens may select the same slot in one forward pass. Resolution:
$$\mathbf{C}[slot] \leftarrow \text{Aggregate}(\{y_{cache,t} : slot_t = slot\})$$

Aggregation options:
- **Mean:** Average all tokens writing to the same slot
- **Weighted mean:** Weight by importance score $score_t$
- **Soft Winner-Take-All (Recommended):** Exponentially weight by importance to let the "most important" token dominate
- **Learned:** Small network combines colliding writes

**Gradient flow:** Straight-Through Estimator (STE) for $\arg\max$ — forward uses hard index, backward flows through $\text{softmax}(W_{slot} \cdot y_t)$.

### 3.2 Fusion Strategies (When Reading Cache)

| Strategy | Implementation | Notes |
|----------|---------------|-------|
| **None** | `y = HeadA(x)` | Simplest; cache only for future layers |
| **Concatenation** | `y = HeadA(Linear([x, context]))` | Simple; increases dim |
| **Gated** | `g = σ(W[x,context]); y = HeadA(g·x + (1-g)·context)` | Recommended |
| **Cross-Attention** | `y = HeadA(x) + CrossAttn(x, C)` | Most expressive |
| **Layer-Selective** | `g_j = σ(W_inject[j]·x); y = HeadA(x + g_j·context)` | Per-layer cache influence |

**Layer-Selective Cache Injection:** Each layer learns how much cache context to incorporate:

$$g_j = \sigma(W_{inject}^{(j)} \cdot x) \in [0, 1]$$
$$x_{fused} = x + g_j \cdot context$$

This allows early layers to ignore cache (focus on local features) while later layers use cache heavily (semantic reasoning). Initialize $W_{inject}$ bias to allow gradual cache usage.

### 3.3 Example: Reading a Book

```
Sentence 1: "She baked an apple pie."
  Layer 3 processes "pie" → Head B: important → cache["dessert context"]

Sentence 2: "She took a bite of the apple."
  Layer 3 processes "apple" → reads cache → finds "pie/dessert"
  → understands: fruit (food), not Apple Inc.
```

---

## 4. Design Goals & Novelty

### 4.1 Primary Goals

| Goal | How DLSMN Achieves It |
|------|----------------------|
| **Hierarchical Memory** | Layer 1 caches low-level features; Layer 12 caches semantic concepts |
| **Selective Retention** | Head B gates what enters cache—no bloat |
| **Cross-Layer Reasoning** | Global cache tensor enables any layer to read other layers' cached info (full access in multi-pass mode) |
| **Efficiency** | Fixed-size cache $O(L \times K)$ instead of $O(N)$ linear growth with sequence length |

### 4.2 Novelty vs. Prior Work

| Architecture | Memory Model | DLSMN Difference |
|--------------|--------------|------------------|
| **Transformers (KV Cache)** | Store every token | Store only *selected* features |
| **Neural Turing Machines** | Global tape, single controller | Distributed: each layer controls its cache partition |
| **RNNs** | Single hidden state | Multiple slots per layer + cross-layer access |
| **Compressive Transformer** | Compress old tokens to summary | Per-layer selective caching with cross-layer reads |

**The "Dual-Head" Innovation:** Separation of compute (Head A) and selection (Head B) decouples *what to output* from *what to remember*.

---

## 5. Related Work

DLSMN builds upon a rich history of memory-augmented neural networks. This section positions our work within the existing literature and clarifies our contributions.

### 5.1 Foundational Memory Architectures

#### Neural Turing Machines (NTM) — Graves et al., 2014
The seminal work on differentiable external memory. NTM couples a neural network controller to a memory matrix via content-based and location-based addressing.

| NTM | DLSMN |
|-----|-------|
| Single controller manages all memory | Each layer controls its own partition |
| Content + location addressing | Learned importance gating + slot routing |
| Global memory tape | Layer-partitioned unified cache |
| Read/write heads are part of controller | Explicit dual-head (compute vs. select) |

**DLSMN Difference:** Distributed control — each layer is its own "controller" for its memory partition, enabling hierarchical abstraction.

#### Hierarchical Attentive Memory (HAM) — Andrychowicz & Kurach, 2016
Binary tree structure for O(log n) memory access. Memory cells are leaves, inner nodes aggregate via JOIN operations.

| HAM | DLSMN |
|-----|-------|
| Tree structure for efficiency | Flat partitioned structure |
| O(log n) access via tree traversal | O(L×K) access via attention |
| Focus on algorithmic tasks | General-purpose architecture |

**DLSMN Difference:** Per-layer semantic partitioning rather than tree-based efficiency optimization. Layers encode different abstraction levels, not just memory addresses.

#### Hierarchical Memory Networks (HMN) — Chandar et al., 2016
Uses Maximum Inner Product Search (MIPS) for coarse-to-fine memory retrieval, enabling scalability to large memories.

| HMN | DLSMN |
|-----|-------|
| MIPS-based retrieval | Attention-based retrieval |
| Hierarchical for efficiency | Hierarchical for semantics |
| Single memory bank | Per-layer partitions |

**DLSMN Difference:** Hierarchy represents layer-wise abstraction (low-level → high-level features), not retrieval efficiency.

### 5.2 Per-Layer External Memory

#### ELMUR (External Layer Memory with Update/Rewrite) — Cherepanov et al., 2025
**Most closely related work.** ELMUR augments each transformer layer with external memory embeddings, using bidirectional cross-attention and LRU-based updates.

| ELMUR | DLSMN |
|-------|-------|
| Per-layer memory | Per-layer memory |
| LRU-based replacement | Learned importance gating |
| Layer-local access only | Cross-layer cache reads |
| No compute/select separation | Dual-head architecture |
| No cache-to-cache attention | Memory self-refinement between passes |
| RL/Robotics focus | General-purpose |

**DLSMN Differences:**
1. **Learned selective writing** — Head B learns importance gating, not rule-based LRU
2. **Cross-layer access** — Layer j can read from layer k's partition (with pass-aware masking)
3. **Dual-head separation** — Explicit decoupling of compute and selection
4. **Cache-to-cache attention** — Memory refines itself between passes
5. **Pass-aware masking** — Multi-pass reasoning with progressive access

ELMUR validates the per-layer memory direction; DLSMN extends it with learned selectivity and cross-layer reasoning.

### 5.3 Cache-Based Communication

#### Cache-to-Cache (C2C) — Fu et al., 2025
Direct semantic communication between LLMs by projecting and fusing KV-caches across models.

| C2C | DLSMN |
|-----|-------|
| Inter-model cache transfer | Intra-model layer communication |
| KV-cache projection | Selective cache with learned routing |
| Between separate LLMs | Between layers of same model |

**DLSMN Difference:** Intra-model hierarchical communication, not inter-model transfer.

### 5.4 Recursive Reasoning Architectures

#### Tiny Recursive Model (TRM) / Hierarchical Reasoning Model (HRM) — Jolicoeur-Martineau, 2025
Small recursive networks achieving strong ARC-AGI performance through iterative passes.

| TRM/HRM | DLSMN |
|---------|-------|
| Recursive weight sharing | Multi-pass with structured memory |
| No explicit memory | Explicit selective cache |
| 7M params, 45% ARC-AGI-1 | Scalable architecture |

**DLSMN Difference:** Combines recursive passes with structured addressable memory. TRM shows small networks can reason recursively; DLSMN provides the memory substrate for that reasoning.

### 5.5 Other Relevant Work

| Architecture | Key Idea | DLSMN Difference |
|--------------|----------|------------------|
| **Memory Networks** (Weston et al., 2015) | End-to-end memory with attention | Per-layer partitioning, selective write |
| **Compressive Transformer** (Rae et al., 2020) | Compress old tokens | Per-layer selective caching |
| **Memorizing Transformers** (Wu et al., 2022) | kNN retrieval over past | Fixed cache, learned selection |
| **Perceiver IO** (Jaegle et al., 2022) | Latent cross-attention | Similar local+cross pattern |
| **Slot Attention** (Locatello et al., 2020) | Competitive slot binding | Similar slot routing concept |

### 5.6 Novelty Summary

DLSMN's core contributions relative to prior work:

| Contribution | Novel? | Notes |
|--------------|--------|-------|
| Per-layer partitioned memory | **Partial** | ELMUR has this; we add cross-layer access |
| Learned selective writing (Head B) | **Yes** | vs. LRU (ELMUR) or store-all (Transformers) |
| Cross-layer cache reads | **Yes** | Other per-layer memories are layer-local |
| Dual-head compute/select | **Yes** | Explicit architectural separation |
| Pass-aware masking | **Yes** | Progressive access control for multi-pass |
| Cache-to-cache self-attention | **Yes** | Memory refines without full forward pass |
| Local + gated cross attention | **Partial** | Pattern exists; novel application to layer cache |

**Positioning:** DLSMN is best understood as extending ELMUR-style per-layer memory with (1) learned selectivity, (2) cross-layer communication, and (3) memory self-refinement.

---

## 6. Comparison with Existing Approaches

### 6.1 DLSMN vs. LLM Scratchpads (CoT)

| Feature | LLM Scratchpad | DLSMN |
|---------|----------------|-------|
| **Storage** | Tokens (text) | Vectors (embeddings) |
| **Location** | Context window | Dedicated cache tensor |
| **Write Cost** | Expensive (autoregressive) | Cheap (single forward pass) |
| **Differentiable** | No | Yes |
| **Information Density** | Low | High |

### 6.2 Complexity Comparison

| Approach | Cache Storage | Cache Growth | Per-Token Access |
|----------|---------------|--------------|------------------|
| **Full Attention** | N/A (recompute) | N/A | $O(N)$ |
| **KV Cache** | $O(N \cdot D)$ | Linear with $N$ | $O(N)$ |
| **DLSMN** | $O(L \times K \times D_{cache})$ | **Fixed** | $O(L \times K)$ |

Where $N$ = sequence length, $L$ = layers, $K$ = slots per layer, $D$ = dimension.

**Key advantage:** DLSMN cache size is independent of sequence length.

### 6.3 DLSMN vs. Linear RNNs (Mamba/S4/RWKV)

| Feature | Mamba/S4/RWKV | DLSMN |
|---------|---------------|-------|
| **State Structure** | Fixed compressed state | Structured addressable slots |
| **Access Pattern** | Sequential scan (fixed compression) | Random access (attention over slots) |
| **Memory Retrieval** | Implicit in state dynamics | Explicit content-based query |
| **Complexity** | $O(N)$ linear | $O(N \cdot L \cdot K)$ (still efficient for reasonable $K$) |
| **Selective Retrieval** | Limited | Strong (attention mechanism) |

**DLSMN Advantage:** Random access to specific historical features without decompression. If the model cached "apple = fruit" in slot 5, it can retrieve exactly that without reconstructing from a compressed state.

**Mamba/S4 Advantage:** More efficient for pure sequential processing; no attention overhead.

### 6.4 DLSMN vs. Transformer-XL

| Feature | Transformer-XL | DLSMN |
|---------|----------------|-------|
| **Cached Content** | Raw token segments | Semantically selected features |
| **Selection** | All tokens in segment | Gated by importance |
| **Cross-Layer** | Per-layer segment memory | Global cache with cross-layer reads |
| **Memory Efficiency** | $O(\text{segment\_length})$ | $O(K)$ fixed |

**DLSMN Advantage:** Semantic compression — only salient features are cached, not raw token representations.

---

## 7. Challenges & Mitigations

| Challenge | Risk | Mitigation |
|-----------|------|------------|
| **Cold Start** | Empty cache → meaningless reads | Learned null embeddings; write-count masking |
| **Over-Caching** | Everything gets cached → no selectivity | Gumbel-softmax annealing achieves selective hard routing |
| **Under-Caching** | Nothing gets cached → useless | Task loss penalizes poor cache utilization; importance threshold tuning |
| **Staleness** | Old cache entries mislead | Age vectors; learned decay; model learns to overwrite via task loss |
| **Gradient Flow** | Hard gating blocks gradients | Straight-Through Estimator (STE) for slot selection; sigmoid for importance score |
| **Training Instability** | Hard routing (argmax, threshold) causes unstable gradients | Gumbel-softmax annealing; curriculum from soft→hard (Section 9) |
| **Representational Clash** | Early-layer textures interfere with late-layer semantics in shared $D_{cache}$ | Layer-ID embeddings; rotary positional encoding per partition (Section 8.6) |
| **Slot Collisions** | Many tokens writing to same slot → oversmoothing | Diversity loss during warmup (Section 10.2) |
| **Multi-Pass Cost** | Inference cost grows linearly with passes | Adaptive Computation Time (ACT); 2-pass default; ponder loss (Section 11.1) |

---

## 8. Architectural Refinements

### 8.1 Cache Initialization

**Problem:** At start, cache is zeros → attention is meaningless.

**Solutions:**
- Initialize with learned embeddings per slot
- Add a "null slot" as fallback
- Mask unwritten slots: if $writes[i] = 0$, set attention score to $-\infty$

### 8.2 Write Addressing (Where to Cache)

**Learned slot selection (Default):** The model learns to organize its cache via direct prediction:
$$slot = \arg\max(W_{slot} \cdot y)$$

No handcrafted eviction policy — the model learns:
- Which concepts deserve caching (importance score)
- Where to store them (slot selection)
- When to overwrite (implicitly, by selecting an occupied slot)
- **What is "stale"** — through task loss, the model learns that overwriting low-utility content improves performance

**Key insight:** The network does not need explicit "staleness" tracking. By training end-to-end, $W_{slot}$ learns to route new content to slots containing outdated or less useful information. The task loss naturally rewards slots that contain useful, retrievable content — if the cache helps performance, it will be used; if not, the model learns to update it.

**Alternative (content-based):** Attention over existing slots for similarity-based organization:
$$slot = \arg\max_i \left( (W_q \cdot y) \cdot (W_k \cdot \mathbf{C}[j*K + i])^T \right)$$

**Not recommended as primary mechanism.** Content-based routing clusters similar content together, but this may not be optimal — sometimes you want to overwrite similar (redundant) content, not add to it. Use only as optional auxiliary signal (see Section 8.8).

### 8.3 Slot Update Rule

**Within a forward pass:** When multiple tokens select the same slot, aggregate their contributions:
$$\mathbf{C}[i] \leftarrow \text{Mean}(\{y_{cache,t} : slot_t = i\})$$

Or weighted by importance:
$$\mathbf{C}[i] \leftarrow \frac{\sum_t score_t \cdot y_{cache,t} \cdot \mathbb{1}[slot_t = i]}{\sum_t score_t \cdot \mathbb{1}[slot_t = i]}$$

**Soft Winner-Take-All (Recommended):** Exponential weighting ensures the most important token dominates the slot rather than creating a "muddy" average:
$$\mathbf{C}[i] \leftarrow (1-\alpha)\mathbf{C}[i] + \alpha \frac{\sum_t y_{cache,t} \cdot e^{score_t} \cdot \mathbb{1}[slot_t = i]}{\sum_t e^{score_t} \cdot \mathbb{1}[slot_t = i]}$$

where $\alpha$ is a high update rate (e.g., 0.9). This prevents oversmoothing when many tokens collide on a single slot.

**Across time/passes:** Hard overwrite (new content replaces old). The model learns when to reuse vs. overwrite slots.

**Optional interpolation:** For smoother updates:
$$\mathbf{C}[i]_{new} = (1 - \gamma) \cdot \mathbf{C}[i]_{old} + \gamma \cdot v_{write}$$

where $\gamma$ is learned (like LSTM forget gate). This can help preserve important older information.

### 8.4 Temporal Decay

Optionally store age $\tau_i$ per slot:
$$k_{read}^i = W_K \cdot [\mathbf{C}[i]; \tau_i]$$

Model learns to discount old memories.

### 8.5 Shared vs. Per-Layer Attention Projections

For cache **read attention** ($W_Q$, $W_K$, $W_V$ used in Step 1):

| Choice | Trade-off |
|--------|-----------|
| **Shared** | Fewer params; consistent query/key space across layers |
| **Per-layer** | More expressive; each layer attends to cache differently |
| **Hybrid** | Shared base + per-layer offset |

**Note:** This is separate from $W_{compress}^{(j)}$ and $W_{decompress}^{(j)}$, which are always per-layer (Section 2.1).

Start with shared attention projections; add per-layer if expressiveness is insufficient.

### 8.6 Layer Partitioning & Representational Separation

**Problem:** Even with per-layer projections, all entries share $D_{cache}$ space. Early-layer features (edges, textures) may interfere with late-layer semantics (concepts, relations). This "semantic clash" is a real risk that must be addressed.

**Solutions:**

#### Option A: Layer-ID Embeddings (Recommended Baseline)

Add a learned layer identifier to each cache entry:

$$\mathbf{C}[j \cdot K + i] \leftarrow [y_{cache}; \mathbf{e}_{layer}^{(j)}]$$

where $\mathbf{e}_{layer}^{(j)} \in \mathbb{R}^{D_{layer}}$ is a learned embedding for layer $j$.

**This is the recommended default.** It is cheaper than Block-Diagonal matrices and allows the network to learn soft-partitions dynamically. The layer embedding acts as a "namespace" that the attention mechanism can use to distinguish early-layer features from late-layer semantics.

During read, the key computation includes layer info:

$$k_i = W_K \cdot [\mathbf{C}[i]; \mathbf{e}_{layer}^{(\lfloor i/K \rfloor)}]$$

#### Option B: Rotary Positional Encoding per Partition

Apply rotary embeddings (RoPE) with layer-specific frequencies:

$$q_{rotated} = \text{RoPE}(q, \theta_j) \quad k_{rotated} = \text{RoPE}(k, \theta_{source\_layer})$$

This creates layer-specific "subspaces" without explicit dimension separation.

#### Option C: Block-Diagonal Projections

Partition $D_{cache}$ into $L$ blocks, each layer primarily uses its block:

$$W_{compress}^{(j)} = \text{BlockDiag}(0, ..., W_j, ..., 0) + W_{shared}$$

Each layer has a "home" subspace but can also use shared dimensions.

**Recommendation:** Start with Layer-ID embeddings (simplest); add RoPE if cross-layer attention remains noisy.

### 8.7 Learned Slot Embeddings (Concept Anchors)

**Motivation:** Empty slots are hard to route to. Learned slot embeddings act as **concept attractors** — similar to codebook entries in VQ-VAE or latent anchors in Perceiver IO.

**Implementation:**

$$\mathbf{S} \in \mathbb{R}^{(L \times K) \times D_{cache}} \quad \text{(learned slot embeddings)}$$

Initialize cache with these embeddings:

$$\mathbf{C}^{(0)} = \mathbf{S}$$

**Slot routing now has an anchor:**

$$slot\_sim_i = (W_q \cdot y)^T \cdot \mathbf{S}[j \cdot K + i]$$

$$slot\_scores = \alpha \cdot (W_{slot} \cdot y) + (1 - \alpha) \cdot slot\_sim$$

where $\alpha$ is learned or scheduled.

**Benefits:**

- Slots develop semantic meaning during training
- Routing becomes content-based (similarity to anchor) + learned
- Dramatically improves organization of memory

### 8.8 Hybrid Slot Routing (Optional)

**Default: Pure learned routing ($\alpha = 1$).** The model learns slot selection entirely through $W_{slot}$.

**Optional hybrid:** Combine learned logits with content-based similarity:

$$slot\_scores = \alpha \cdot \underbrace{(W_{slot} \cdot y)}_{\text{learned routing}} + (1 - \alpha) \cdot \underbrace{\text{Softmax}\left(\frac{(W_q \cdot y) \cdot (W_k \cdot \mathbf{C})^T}{\sqrt{d}}\right)}_{\text{content-based similarity}}$$

**Recommended: Start with $\alpha = 1$ (pure learned).** Only add content-based component if the model struggles to organize memory.

**Dynamic $\alpha$:** If using hybrid, can be learned per-layer:

$$\alpha = \sigma(W_{\alpha} \cdot y)$$

**When content-based helps:**
- Tasks with explicit clustering (e.g., entity tracking)
- When slot embeddings (Section 8.7) are not used

**When pure learned is better:**
- The model should learn to overwrite stale/redundant content
- Avoid clustering similar content that should replace each other

---

## 9. Training Stability & Curriculum

### 9.1 The Hard Routing Problem

**Risk:** Using STE for `argmax(slot_logits)` and hard threshold `score > τ` works but may cause:

- Gradient variance
- Mode collapse (all tokens → few slots)
- Unstable early training

### 9.2 Gumbel-Softmax Annealing

**Solution:** Start with soft routing, gradually harden.

**Slot selection with Gumbel-Softmax:**

$$slot\_probs = \text{Softmax}\left(\frac{W_{slot} \cdot y + g}{\tau_{temp}}\right)$$

where $g \sim \text{Gumbel}(0, 1)$ and $\tau_{temp}$ is temperature.

### 9.2.1 Random Noise Injection (Cold Start Exploration)

**Problem:** Hard `argmax` routing prevents gradients from flowing to unselected slots early in training, causing some slots to never receive signal.

**Solution:** Add random noise to router logits during the first 10-15% of training steps:

```python
# Exploration noise (decays over training)
if step < exploration_steps:
    noise_scale = exploration_start * (1 - step / exploration_steps)
    slot_logits = slot_logits + noise_scale * torch.randn_like(slot_logits)
```

This acts as "exploration" for memory slots, ensuring all slots receive gradient signal before the network becomes confident in its routing decisions.

**Annealing schedule:**

```python
# Temperature schedule over training
τ_temp = max(τ_min, τ_start * exp(-anneal_rate * step))

# Example: τ_start=1.0, τ_min=0.1, anneal over 10k steps
```

**Write operation during soft phase:**

$$\mathbf{C}[j \cdot K : (j+1) \cdot K] \leftarrow \mathbf{C} + slot\_probs^T \cdot y_{cache}$$

All slots receive weighted updates; as $\tau_{temp} \to 0$, converges to hard routing.

### 9.3 Importance Threshold Annealing

Similarly for the write gate:

**Soft phase:**

$$write\_weight = \sigma\left(\frac{W_{gate} \cdot y - \tau}{\tau_{gate\_temp}}\right)$$

**Hard phase (inference):**

$$write\_weight = \mathbb{1}[\sigma(W_{gate} \cdot y) > \tau]$$

### 9.4 Training Curriculum

| Phase | Steps | Temperature | Routing | Notes |
|-------|-------|-------------|---------|-------|
| **Warm-up** | 0 - 5k | High (1.0) | Soft | All slots active; learn representations |
| **Transition** | 5k - 20k | Annealing | Mixed | Gradually specialize slots |
| **Hard** | 20k+ | Low (0.1) | Near-hard | STE for remaining gradient flow |

### 9.5 Auxiliary Stabilization Losses

During soft phase, add:

**Entropy regularization (prevent collapse):**

$$\mathcal{L}_{entropy} = -\lambda_H \cdot H(slot\_probs)$$

**Load balancing (uniform slot usage):**

$$\mathcal{L}_{balance} = \lambda_B \cdot \text{CV}(slot\_counts)^2$$

where CV = coefficient of variation of slot selection counts.

---

## 10. Auxiliary Losses

### 10.1 Design Philosophy: Minimal Auxiliary Losses

Many memory-augmented architectures propose complex auxiliary loss landscapes. After careful analysis, we adopt a **minimal approach** — auxiliary losses should only address problems that the primary task loss cannot solve through gradient flow alone.

**Principles:**
1. **Avoid redundancy** — If a mechanism (e.g., Gumbel-softmax annealing) already achieves a goal, don't add a loss for it
2. **Prefer implicit learning** — The model should learn staleness, utility, and slot organization through task loss
3. **Auxiliary losses for pathologies only** — Only add losses that prevent degenerate solutions the task loss cannot fix

### 10.2 Slot Diversity Loss (Anti-Collision) — Warmup Only

**Problem:** Many tokens selecting the same slot causes oversmoothing. This is especially problematic early in training before the model learns meaningful slot organization.

**Solution:** Entropy maximization over slot distribution:

$$\mathcal{L}_{diversity} = -\lambda_D \cdot H\left(\frac{1}{T}\sum_t slot\_probs_t\right)$$

**Schedule:** Active only during warmup phase (first ~5-10% of training), then decays to zero. Once the model learns diverse slot usage, this loss is unnecessary and may even interfere with learned organization.

**Why needed:** Without this, the model may collapse to using only a few slots early in training, and gradient flow through Gumbel-softmax is insufficient to escape this local minimum.

### 10.3 Ponder Loss (ACT Cost) — When ACT Enabled

When using Adaptive Computation Time (ACT) for variable-pass inference:

$$\mathcal{L}_{ponder} = \lambda_P \cdot \frac{1}{B} \sum_b N_b$$

where $N_b$ is the number of passes for batch element $b$.

**Purpose:** Encourages the model to minimize computation by halting early when confident. Without this, ACT may always use maximum passes.

**Default:** $\lambda_P = 0.01$

### 10.4 Q-Head Loss (TRM-Inspired) — Optional

For tasks with discrete outputs, a "query head" that predicts answer correctness can provide deep supervision:

$$\mathcal{L}_{q\_head} = \lambda_Q \cdot \text{BCE}(q_{pred}, \mathbb{1}[\hat{y} = y])$$

**Purpose:** Gives the model a meta-objective: learn to know when it's likely correct. This improves calibration and can guide early stopping during inference.

**Default:** $\lambda_Q = 0.1$ when enabled.

### 10.5 Removed Losses (Design Decisions)

The following losses were considered but **not included** in the final design:

| Loss | Reason for Removal |
|------|-------------------|
| **Predictive Loss** | Cache utility emerges from task loss — if cached content isn't useful, task performance suffers |
| **Consistency Loss** | Cross-layer alignment is learned implicitly; no evidence explicit loss improves over task-driven learning |
| **Sparsity Loss** | Gumbel-softmax annealing (τ: 1.0→0.1) already achieves selective hard routing |
| **Balance Loss** | Mathematically similar to diversity loss; coefficient-of-variation adds complexity without benefit |
| **Distinctness Loss** | Slot content differentiation emerges from task requirements; forcing distinctness may harm useful redundancy |

### 10.6 Combined Training Objective

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_D \mathcal{L}_{diversity} + \lambda_P \mathcal{L}_{ponder} + \lambda_Q \mathcal{L}_{q\_head}$$

**Recommended values:**

| Loss | $\lambda$ | Schedule | Condition |
|------|-----------|----------|-----------|
| Diversity | 0.01 | Decay to 0 after warmup | Always (warmup only) |
| Ponder | 0.01 | Constant | ACT enabled |
| Q-Head | 0.1 | Constant | Discrete output tasks |

**Note:** This is intentionally minimal. The task loss does most of the work; auxiliary losses only prevent early-training pathologies.

---

## 11. Advanced Extensions

### 11.1 Adaptive Computation Time (ACT)

**Problem:** Multi-pass inference is expensive; not all inputs need multiple passes.

**Solution:** Learn when to stop iterating.

**Halting mechanism:**

$$h^{(p)} = \sigma(W_h \cdot \text{pool}(\mathbf{C}^{(p)}))$$

where $p$ = pass number.

**Pondering cost:**

$$\mathcal{L}_{ponder} = \lambda_P \cdot \sum_p (1 - \prod_{p'<p}(1 - h^{(p')}))$$

Encourages early halting.

**Inference:**

```python
for p in range(max_passes):
    cache, halt_prob = forward_pass(x, cache, p)
    if halt_prob > 0.5 or p == max_passes - 1:
        break
```

**Benefits:**

- Easy inputs: 1 pass
- Hard inputs: 2-3 passes
- Amortized cost much lower than fixed multi-pass

### 11.2 Cache-to-Cache Attention

**Motivation:** Allow "memory-only computation" between passes — reasoning over cached content without full forward passes. This transforms the cache from passive storage into an **active reasoning substrate**.

**Core insight:** Once the cache contains structured representations, it can be useful to let the memory itself perform computation:
- Running message passing among memories
- Propagating relational structure (if slot A stores "X→Y" and slot B stores "Y→Z", derive "X→Z")
- Denoising and stabilizing representations
- Building a latent concept graph

**When it happens:** Cache-to-cache attention occurs *after* all layers have written in a given pass, not during the forward pass:

```
Pass 1: Layer 1 → Layer 2 → ... → Layer L (all write to cache)
        ↓
        Cache-to-Cache Attention (memory refines itself)
        ↓
Pass 2: Layer 1 → Layer 2 → ... → Layer L (read improved cache)
```

#### 10.2.1 Locality Modes

Cache-to-cache attention can operate at different scopes. Since the cache is partitioned (layer $j$ owns `C[j*K : (j+1)*K]`), we distinguish between **local** (within-layer) and **cross** (other layers):

| Mode | Local (Self) | Cross (Others) | Use Case |
|------|--------------|----------------|----------|
| **Local-only** | ✓ Self-attention within layer $j$'s K slots | ✗ None | Cheap cleanup, collision resolution |
| **Cross-only** | ✗ None | ✓ Attend to all other layers | Pure cross-layer reasoning |
| **Local + Gated Cross** | ✓ Always | ✓ Gated by $g$ | **Recommended** — adaptive mixing |

**Important:** Cross-layer attention excludes the current layer's slots to avoid double-counting. Layer $j$ attends to `C[:j*K]` and `C[(j+1)*K:]`, not to `C[j*K:(j+1)*K]`.

#### 10.2.2 Learned Locality via Gating (Recommended)

**Problem:** Hard-coding locality (global vs local) is brittle. Different tasks and different layers may need different mixing patterns.

**Key Implementation Detail:** The global cache is partitioned by layer. Each layer $j$ owns slice `C[j*K : (j+1)*K]`. To avoid double-counting when mixing local and cross-layer information:

- **Local:** Layer $j$'s slots attend to themselves (self-refinement)
- **Cross-layer:** Layer $j$'s slots attend to *other* layers only (excluding own slice)

```
Global Cache Structure:
┌─────────┬─────────┬─────────┬─────────┐
│ Layer 0 │ Layer 1 │ Layer 2 │ Layer 3 │  ...
└─────────┴─────────┴─────────┴─────────┘
     ↑
For Layer 1's update:
  - Local:  attend within Layer 1's slots only
  - Cross:  attend to Layer 0, 2, 3, ... (exclude Layer 1)
```

**Solution:** Let the network learn the optimal mixing via soft gating:

```python
class AdaptiveCacheAttention(nn.Module):
    def __init__(self, d_cache, L, K):
        self.d_cache = d_cache
        self.L = L
        self.K = K
        
        # Local: self-attention within each layer's K slots
        self.local_attn = MultiHeadAttention(d_cache)
        
        # Cross: attention from layer j to all OTHER layers
        self.cross_attn = MultiHeadAttention(d_cache)
        
        # Per-slot gate controlling cross-layer mixing
        self.gate = nn.Linear(d_cache, 1)
        
        # Initialize gate bias toward LOCAL (conservative start)
        nn.init.constant_(self.gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12
        
    def forward(self, cache):  # [L*K, D_cache]
        L, K = self.L, self.K
        output = torch.zeros_like(cache)
        
        for j in range(L):
            # Layer j's slice
            C_j = cache[j*K : (j+1)*K]  # [K, D]
            
            # Other layers (exclude layer j to avoid double-counting)
            C_others = torch.cat([
                cache[:j*K],           # layers before j
                cache[(j+1)*K:]        # layers after j
            ], dim=0)  # [(L-1)*K, D]
            
            # Local: self-refinement within layer j
            C_j_local = self.local_attn(
                query=C_j, key=C_j, value=C_j
            )  # [K, D]
            
            # Cross: attend to other layers only
            C_j_cross = self.cross_attn(
                query=C_j, key=C_others, value=C_others
            )  # [K, D]
            
            # Per-slot gate: how much cross-layer info to incorporate?
            g = torch.sigmoid(self.gate(C_j))  # [K, 1]
            
            # Combine: local refinement + gated cross-layer context
            output[j*K : (j+1)*K] = C_j_local + g * C_j_cross
        
        return cache + output  # residual connection
```

**Why this matters:**
- If we naively did `g * global_attn(C) + (1-g) * local_attn(C_j)`, layer j's information would appear in both terms
- By separating into **local** (self) and **cross** (others), the gate cleanly controls: *"How much should I incorporate from OTHER layers?"*
- `g = 0` → Pure local self-refinement
- `g = 1` → Maximum cross-layer influence (but still includes local via residual)

**Key design choices:**

1. **Initialize gate bias toward local (g ≈ 0):** Start conservative; let the network "earn" cross-layer access through training. If cross-layer mixing improves loss, gradients push g → 1.

2. **Per-slot vs per-layer gating:** Per-slot is more expressive but per-layer (L gates instead of L×K) may be sufficient and more interpretable.

3. **Interaction with Layer-ID embeddings:** With layer-ID embeddings in the cache, the cross-attention naturally learns which layers are relevant to query.

4. **Write restriction:** Each layer can only *write* to its own slice. The cross-attention is read-only from other layers.

**What the network learns:**
- Early layers often prefer local (g ≈ 0) — feature-level cleanup
- Middle layers may be mixed — partial cross-layer coordination  
- Top layers often prefer high cross-layer (g → 1) — semantic reasoning across hierarchy

#### 10.2.3 Relation to Existing Patterns

The update rule:

$$C_j^{new} = \text{Local}_j(C_j) + g \cdot \text{Cross}_j(C_j \leftarrow C_{others})$$

is a **universal pattern for hierarchical memory** that appears across many successful architectures:

| Architecture | Local Component | Cross Component |
|--------------|-----------------|-----------------|
| **Perceiver IO** | Latent self-attention | Cross-attention to inputs |
| **Slot Attention** | Slot competition/refinement | Cross-attention to image features |
| **Graph Neural Networks** | Node self-transform | Message passing from neighbors |
| **Encoder-Decoder Transformer** | Decoder self-attention | Cross-attention to encoder |
| **DLSMN Cache-to-Cache** | Within-layer slot attention | Gated cross-attention to other layers |

This confirms the design is not ad-hoc but follows established principles for mixing local refinement with contextual aggregation. The key insight: **refine yourself first, then selectively incorporate information from others**.

#### 10.2.4 Single-Pass vs Multi-Pass Usage

| Mode | Role of Cache-to-Cache Attention |
|------|----------------------------------|
| **Multi-pass** | Iterative reasoning between passes. Pass 1 gathers information, cache-to-cache propagates associations, Pass 2 refines decisions. |
| **Single-pass** | Final "memory consolidation" step. Resolves conflicts, builds relationships, compresses to more expressive latent structure. Still beneficial but acts as refinement rather than iteration. |

**Implementation (basic global version):**

```python
class CacheSelfAttention(nn.Module):
    def forward(self, cache):  # [L*K, D_cache]
        Q = self.W_Q(cache)
        K = self.W_K(cache)
        V = self.W_V(cache)
        attn = softmax(Q @ K.T / sqrt(d))
        return cache + attn @ V
```

**Benefits:**

- Symbolic reasoning over stored concepts
- Graph-like propagation between cache entries (GNN inside memory)
- Iterative refinement without expensive layer computation
- Enables compositional reasoning: "If A is cached and B is cached, derive C"
- Denoising: correlated slot entries reinforce each other

**Cost:** Only $O((L \cdot K)^2)$ — much cheaper than full forward pass since $L \cdot K \ll N$ (sequence length).

### 11.3 Hierarchical Cache Compression

**For very long sequences:** Periodically compress old cache entries.

$$\mathbf{C}_{compressed} = \text{Compress}(\mathbf{C}_{old})$$

Options:

- Learned compression network
- Attention-weighted pooling
- Keep top-k by access frequency

### 11.4 Cross-Sequence Cache Persistence

**For multi-turn or episodic tasks:** Persist cache across sequences.

```python
# End of sequence 1
persistent_cache = decay * cache  # soft forget

# Start of sequence 2  
cache = concat(persistent_cache, fresh_slots)
```

Enables:

- Multi-turn dialogue memory
- Episodic learning
- Few-shot in-context learning with cached examples

---

## 12. RNN/Recursive Mode (DLSMN-RNN)

DLSMN supports two distinct modes of operation. This section clarifies the differences.

### 12.0 Mode Comparison: Multi-Pass vs. RNN

| Aspect | Multi-Pass (Section 2.3) | RNN Mode |
|--------|--------------------------|----------|
| **Input** | Same input processed multiple times | Different inputs $x_t, x_{t+1}, ...$ |
| **Purpose** | Iterative refinement on one input | Sequential processing over time |
| **Cache lifetime** | Rebuilt per input, persists across passes | Persists indefinitely across timesteps |
| **Use case** | Complex reasoning (ARC-AGI) | Sequence modeling, dialogue |
| **Gradient flow** | Full backprop through all passes | Truncated BPTT or detached |

**Multi-Pass Refinement (Section 2.3):** Given input $x$, run Pass 1 (bottom-up), then Pass 2+ with full cache access. The cache is populated in Pass 1 and refined/used in Pass 2+. **Same input, multiple passes, improving output.**

**RNN Mode (this section):** Process a stream of different inputs $x_1, x_2, ..., x_T$. The cache carries information across time. **Different inputs, cache persists across time.**

### 12.1 Cache as Hidden State

In RNN mode, the Global Cache $\mathbf{C}$ becomes the persistent state carried forward in time:

$$\mathbf{C}_t = \text{DLSMN\_Block}(x_t, \mathbf{C}_{t-1})$$

**Key difference from stacked mode:**
- **Stacked (Deep):** Cache is flushed/rebuilt each forward pass
- **Recurrent:** Cache persists indefinitely across time steps

### 12.2 Explicit Forgetting Mechanism (Critical)

**Problem:** In stacked mode, old cache entries are naturally replaced. In RNN mode, the cache persists indefinitely. Without explicit decay, the cache will **saturate with noise** after 500+ steps.

**Solution:** Add a **Forget Gate** to Head B (analogous to LSTM):

$$f_t = \sigma(W_{forget} \cdot y_t + b_{forget})$$

$$\mathbf{C}[slot]_{new} = f_t \odot \mathbf{C}[slot]_{old} + (1 - f_t) \odot v_{write}$$

where $f_t$ controls how much of the old content to retain vs. overwrite.

**Per-slot forget gates:** For finer control, compute a forget gate per slot:

```python
# Forget gate per slot (K gates)
f_t = sigmoid(W_forget @ y_t)  # [K]

# Apply per-slot
for slot_idx in range(K):
    if writing_to_slot[slot_idx]:
        C[slot_idx] = f_t[slot_idx] * C[slot_idx] + (1 - f_t[slot_idx]) * new_value
```

### 12.3 Temporal Positioning in Cache

**Problem:** In an RNN, when reading from Slot 5, the model needs to know *when* the content was written to reason about recency.

**Solution:** Append a **normalized time-stamp** to each cache vector:

$$\mathbf{C}[i] = [y_{cache}; \tau_i / \tau_{max}]$$

where $\tau_i$ is the step count when slot $i$ was last written.

**Alternative (relative time):** Store $\Delta t = t_{current} - t_{written}$ and update each step:

```python
# Each time step, increment age of all slots
slot_ages += 1

# Reset age when writing
slot_ages[written_slots] = 0

# Concatenate age to cache for attention
cache_with_time = torch.cat([cache, slot_ages.unsqueeze(-1) / max_age], dim=-1)
```

The model can then learn to discount "stale" memories during attention.

### 12.4 RNN Mode Architecture Summary

```
At each time step t:

Input: x_t, C_{t-1} (cache from previous step)

1. Read:     context = Attend(x_t, C_{t-1})
2. Compute:  y_t = HeadA(Fuse(x_t, context))
3. Select:   score_t = σ(W_gate · y_t)
             slot_t = route(y_t)
             f_t = σ(W_forget · y_t)           # FORGET GATE
4. Update:   C_t[slot_t] = f_t · C_{t-1}[slot_t] + (1-f_t) · W_compress · y_t
5. Age:      increment slot_ages; reset written slots

Output: y_t, C_t (carry forward)
```

### 12.5 Comparison: DLSMN-RNN vs. Linear RNNs

| Architecture | Memory Structure | Access Pattern | Key Advantage |
|--------------|------------------|----------------|---------------|
| **LSTM/GRU** | Single vector $h_t$ | Compressed | Simple, fast |
| **Mamba/S4/RWKV** | Fixed state, linear | Sequential scan | Linear complexity |
| **DLSMN-RNN** | Structured slots | Random access (attention) | Retrieve specific features without decompression |

**DLSMN-RNN's "Killer Feature":** Random access to structured memory. Instead of compressing all history into a single vector (LSTM) or fixed state (Mamba), DLSMN maintains addressable slots that can be queried by content.

---

## 13. Summary

### Architecture at a Glance

```
Global Cache: C ∈ ℝ^{(L×K) × D_cache}
Slot Embeddings: S ∈ ℝ^{(L×K) × D_cache}  (learned anchors)

For each layer j, for each token t in sequence:
    1. (Optional) Read:  raw_ctx = Attend(x_t, C[accessible])
                         context_t = W_decompress[j] · raw_ctx    # project to D_model
    2. Compute:          y_t = HeadA(x_t or Fuse(x_t, context_t))
    3. Select:           score_t = σ(W_gate · y_t)                # importance
                         slot_t = HybridRoute(y_t, C, S)          # learned + content-based
    4. Collect writes:   if score_t > τ (or soft-weighted during training):
                             y_cache_t = W_compress[j] · y_t      # project to D_cache
                             pending_writes[slot_t].append((y_cache_t, score_t))

After all tokens processed:
    5. Aggregate:        for each slot with writes: C[j*K + slot] ← Aggregate(pending_writes[slot])
    6. Output:           pass y to layer j+1

(Optional) Between passes:
    7. Cache-to-Cache:   C ← AdaptiveCacheAttn(C)   # learned local/global mixing
    8. Halt decision:    if ACT_halt_prob > 0.5: stop
```

### Key Properties

| Property | Implementation |
|----------|----------------|
| **Unified Cache** | Single tensor `[L×K, D_cache]`, partitioned by layer |
| **Per-Layer Projections** | Each layer has own $W_{compress}^{(j)}$ and $W_{decompress}^{(j)}$ |
| **Layer-ID Embeddings** | Recommended baseline for representational separation |
| **Selective Write** | Head B gates what enters cache |
| **Soft-WTA Aggregation** | Exponential weighting prevents oversmoothing on collision |
| **Cross-Layer Read** | Attention over full cache (with pass-aware masking) |
| **Fixed Memory** | $O(L \times K)$ regardless of sequence length |
| **End-to-End Training** | Fully differentiable |
| **Learned Slot Anchors** | Slot embeddings act as concept attractors |
| **Hybrid Routing** | Combines learned logits + content-based similarity |
| **Training Stability** | Gumbel-softmax annealing + noise injection for exploration |
| **Adaptive Passes** | ACT-like halting for efficient multi-pass |
| **RNN Mode Support** | Forget gates + temporal positioning for recurrent use |
| **Learned Cache Locality** | Gated mixture: local self-attention + cross-layer attention (excluding own slice) |

### Design Principles

1. **Compute first, cache second** — Head A processes, Head B decides retention
2. **One cache, partitioned access** — Simple indexing for local/global reads
3. **Pass-aware masking** — Bottom-up on pass 1, full access on pass 2+
4. **Soft-to-hard curriculum** — Train with soft routing, deploy with hard
5. **Rich auxiliary supervision** — Predictive, consistency, diversity, distinctness losses
6. **Adaptive computation** — Easy inputs need fewer passes
7. **Memory-only reasoning** — Cache-to-cache attention for cheap refinement
8. **RNN-ready** — Explicit forgetting and temporal encoding for sequential processing

### Recommended Training Recipe

```python
# Phase 1: Warm-up (soft routing)
for step in range(5000):
    τ_temp = 1.0
    loss = task_loss + 0.1*predict + 0.01*diversity + 0.01*balance

# Phase 2: Annealing
for step in range(5000, 20000):
    τ_temp = max(0.1, 1.0 * exp(-0.0003 * step))
    loss = task_loss + 0.1*predict + 0.05*consistency + 0.01*sparse

# Phase 3: Hard routing
for step in range(20000, ...):
    τ_temp = 0.1  # near-hard with STE
    loss = task_loss + 0.1*predict + 0.05*consistency + 0.02*sparse
```

---

*Ready for prototyping.*
