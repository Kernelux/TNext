# Dual-Head Layered Selective Memory Network (DLSMN)

> A memory-augmented neural architecture with hierarchical, selective, and structured memory using a unified global cache.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Architecture](#2-core-architecture)
3. [Memory Operations](#3-memory-operations)
4. [Design Goals & Novelty](#4-design-goals--novelty)
5. [Comparison with Existing Approaches](#5-comparison-with-existing-approaches)
6. [Challenges & Mitigations](#6-challenges--mitigations)
7. [Architectural Refinements](#7-architectural-refinements)
8. [Summary](#8-summary)

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
slot_idx = argmax(slot_logits)     # hard slot selection

if score > τ:
    y_cache = W_compress[j] · y    # per-layer projection to cache space
    C[j*K + slot_idx] ← aggregate(C[j*K + slot_idx], y_cache)
```

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
- **Learned:** Small network combines colliding writes

**Gradient flow:** Straight-Through Estimator (STE) for $\arg\max$ — forward uses hard index, backward flows through $\text{softmax}(W_{slot} \cdot y_t)$.

### 3.2 Fusion Strategies (When Reading Cache)

| Strategy | Implementation | Notes |
|----------|---------------|-------|
| **None** | `y = HeadA(x)` | Simplest; cache only for future layers |
| **Concatenation** | `y = HeadA(Linear([x, context]))` | Simple; increases dim |
| **Gated** | `g = σ(W[x,context]); y = HeadA(g·x + (1-g)·context)` | Recommended |
| **Cross-Attention** | `y = HeadA(x) + CrossAttn(x, C)` | Most expressive |

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

## 5. Comparison with Existing Approaches

### 5.1 DLSMN vs. LLM Scratchpads (CoT)

| Feature | LLM Scratchpad | DLSMN |
|---------|----------------|-------|
| **Storage** | Tokens (text) | Vectors (embeddings) |
| **Location** | Context window | Dedicated cache tensor |
| **Write Cost** | Expensive (autoregressive) | Cheap (single forward pass) |
| **Differentiable** | No | Yes |
| **Information Density** | Low | High |

### 5.2 Complexity Comparison

| Approach | Cache Storage | Cache Growth | Per-Token Access |
|----------|---------------|--------------|------------------|
| **Full Attention** | N/A (recompute) | N/A | $O(N)$ |
| **KV Cache** | $O(N \cdot D)$ | Linear with $N$ | $O(N)$ |
| **DLSMN** | $O(L \times K \times D_{cache})$ | **Fixed** | $O(L \times K)$ |

Where $N$ = sequence length, $L$ = layers, $K$ = slots per layer, $D$ = dimension.

**Key advantage:** DLSMN cache size is independent of sequence length.

---

## 6. Challenges & Mitigations

| Challenge | Risk | Mitigation |
|-----------|------|------------|
| **Cold Start** | Empty cache → meaningless reads | Learned null embeddings; write-count masking |
| **Over-Caching** | Everything gets cached → no selectivity | Sparsity loss on gate activations |
| **Under-Caching** | Nothing gets cached → useless | Auxiliary loss (cache predicts future tokens) |
| **Staleness** | Old cache entries mislead | Age vectors; learned decay |
| **Gradient Flow** | Hard gating blocks gradients | Straight-Through Estimator (STE) for slot selection; sigmoid for importance score |

---

## 7. Architectural Refinements

### 7.1 Cache Initialization

**Problem:** At start, cache is zeros → attention is meaningless.

**Solutions:**
- Initialize with learned embeddings per slot
- Add a "null slot" as fallback
- Mask unwritten slots: if $writes[i] = 0$, set attention score to $-\infty$

### 7.2 Write Addressing (Where to Cache)

**Learned slot selection:** The model learns to organize its cache via direct prediction:
$$slot = \arg\max(W_{slot} \cdot y)$$

No handcrafted eviction policy — the model learns:
- Which concepts deserve caching (importance score)
- Where to store them (slot selection)
- When to overwrite (implicitly, by selecting an occupied slot)

**Alternative (content-based):** Attention over existing slots for similarity-based organization:
$$slot = \arg\max_i \left( (W_q \cdot y) \cdot (W_k \cdot \mathbf{C}[j*K + i])^T \right)$$

Content-based may help cluster related concepts; learned selection is simpler and recommended as default.

### 7.3 Slot Update Rule

**Within a forward pass:** When multiple tokens select the same slot, aggregate their contributions:
$$\mathbf{C}[i] \leftarrow \text{Mean}(\{y_{cache,t} : slot_t = i\})$$

Or weighted by importance:
$$\mathbf{C}[i] \leftarrow \frac{\sum_t score_t \cdot y_{cache,t} \cdot \mathbb{1}[slot_t = i]}{\sum_t score_t \cdot \mathbb{1}[slot_t = i]}$$

**Across time/passes:** Hard overwrite (new content replaces old). The model learns when to reuse vs. overwrite slots.

**Optional interpolation:** For smoother updates:
$$\mathbf{C}[i]_{new} = (1 - \gamma) \cdot \mathbf{C}[i]_{old} + \gamma \cdot v_{write}$$

where $\gamma$ is learned (like LSTM forget gate). This can help preserve important older information.

### 7.4 Temporal Decay

Optionally store age $\tau_i$ per slot:
$$k_{read}^i = W_K \cdot [\mathbf{C}[i]; \tau_i]$$

Model learns to discount old memories.

### 7.5 Shared vs. Per-Layer Attention Projections

For cache **read attention** ($W_Q$, $W_K$, $W_V$ used in Step 1):

| Choice | Trade-off |
|--------|-----------|
| **Shared** | Fewer params; consistent query/key space across layers |
| **Per-layer** | More expressive; each layer attends to cache differently |
| **Hybrid** | Shared base + per-layer offset |

**Note:** This is separate from $W_{compress}^{(j)}$ and $W_{decompress}^{(j)}$, which are always per-layer (Section 2.1).

Start with shared attention projections; add per-layer if expressiveness is insufficient.

---

## 8. Summary

### Architecture at a Glance

```
Global Cache: C ∈ ℝ^{(L×K) × D_cache}

For each layer j, for each token t in sequence:
    1. (Optional) Read:  raw_ctx = Attend(x_t, C[accessible])
                         context_t = W_decompress[j] · raw_ctx    # project to D_model
    2. Compute:          y_t = HeadA(x_t or Fuse(x_t, context_t))
    3. Select:           score_t, slot_t = HeadB(y_t)
    4. Collect writes:   if score_t > τ:
                             y_cache_t = W_compress[j] · y_t      # project to D_cache
                             pending_writes[slot_t].append(y_cache_t)

After all tokens processed:
    5. Aggregate:        for each slot with writes: C[j*K + slot] ← Aggregate(pending_writes[slot])
    6. Output:           pass y to layer j+1
```

### Key Properties

| Property | Implementation |
|----------|----------------|
| **Unified Cache** | Single tensor `[L×K, D_cache]`, partitioned by layer |
| **Per-Layer Projections** | Each layer has own $W_{compress}^{(j)}$ and $W_{decompress}^{(j)}$ |
| **Selective Write** | Head B gates what enters cache |
| **Cross-Layer Read** | Attention over full cache (with pass-aware masking) |
| **Fixed Memory** | $O(L \times K)$ regardless of sequence length |
| **End-to-End Training** | Fully differentiable |

### Design Principles

1. **Compute first, cache second** — Head A processes, Head B decides retention
2. **One cache, partitioned access** — Simple indexing for local/global reads
3. **Pass-aware masking** — Bottom-up on pass 1, full access on pass 2+
4. **This is a building block** — Pass count (1, N, or until convergence) is external

---

*Ready for prototyping.*
