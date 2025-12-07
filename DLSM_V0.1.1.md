# Dual-Head Layered Selective Memory Network (DLSMN) v0.1.1

> A memory-augmented neural architecture with hierarchical, selective, and structured memory, featuring active bi-directional routing and recursive refinement.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Architecture](#2-core-architecture)
3. [Memory Operations (The Dual-Gated Cycle)](#3-memory-operations-the-dual-gated-cycle)
4. [Hierarchical Adaptive Computation Time (H-ACT)](#4-hierarchical-adaptive-computation-time-h-act)
5. [Recursive Dynamics (TRM Integration)](#5-recursive-dynamics-trm-integration)
6. [Design Goals & Novelty](#6-design-goals--novelty)
7. [Training Stability & Curriculum](#7-training-stability--curriculum)
8. [Auxiliary Losses](#8-auxiliary-losses)
9. [Advanced Extensions](#9-advanced-extensions)
10. [Summary](#10-summary)

---

## 1. Executive Summary

The **Dual-Head Layered Selective Memory Network (DLSMN) v0.1.1** advances the concept of per-layer selective memory by introducing **active control** over both information retrieval and retention. While v0.1 focused on "selective writing," v0.1.1 implements a fully bi-directional **Mixture-of-Experts (MoE) Router** that governs both reading and writing.

This architecture acts as a **Recursive Reasoning Machine**:
1.  **Hierarchical:** Each layer owns a partition of a global cache.
2.  **Active:** A learned router decides *when* to read from memory and *what* to write, rather than passive attention.
3.  **Recursive:** It employs **Hierarchical Adaptive Computation Time (H-ACT)** to dynamically allocate compute power (passes) where needed, inspired by the Tiny Recursive Model (TRM).

This transforms the layer from `f(x) -> y` into a stateful, recursive agent `f(x, cache, feedback) -> (y, cache', confidence)`.

---

## 2. Core Architecture

### 2.1 Global Cache Structure

A single contiguous cache tensor stores all layer memories, enabling seamless cross-layer access:

$$\mathbf{C} \in \mathbb{R}^{(L \times K) \times D_{cache}}$$

*   $L$ = Number of layers
*   $K$ = Slots per layer
*   $D_{cache}$ = Cache dimension

**Layer-ID Embeddings:** To prevent semantic clash between early and late layers in the shared space, each slot is concatenated with a learned Layer-ID embedding:
$$k_{slot} = [k_{content}; E_{layer\_id}]$$
This allows the attention mechanism to naturally distinguish between "visual features" (Layer 1) and "abstract concepts" (Layer L).

### 2.2 pattern Pooling (The "bottleneck")

In v0.1, we cached raw token outputs $y$. In v0.1.1, we enforce higher-order abstraction via **Pattern Pooling**.

Each layer maintains a set of learnable **Pattern Queries** ($Q_{pats} \in \mathbb{R}^{N_{pats} \times D_{model}}$). Before interacting with the write router, the layer output $y$ is compressed:

$$Patterns = \text{MultiHeadAttention}(Q_{pats}, y, y)$$

**Benefits:**
*   **Decoupling:** Cache cost depends on $N_{pats}$, not sequence length $S$.
*   **Abstraction:** Forces the model to synthesize "concepts" rather than storing raw pixel/token data.

### 2.3 The MoE Memory Router

The core controller of DLSMN v0.1.1 is the **MoE Router**, which replaces the simple "Head B". It is a lightweight network that outputs decisions for both **Reading** (Input-driven) and **Writing** (Output-driven).

#### Read Controls (Input $x$)
*   **Read Gate ($g_{read}$):** $\sigma(W_r \cdot x) \in [0, 1]$. "Does this token need context?"
*   **Read Slot Dist ($p_{read}$):** $\text{Softmax}(W_{route\_r} \cdot x)$. "Which slots contain relevant info?"

#### Write Controls (Pattern $P$)
*   **Write Gate ($g_{write}$):** $\sigma(W_w \cdot P) \in [0, 1]$. "Is this pattern worth saving?"
*   **Importance ($s_{imp}$):** $\sigma(W_{imp} \cdot P) \in [0, 1]$. "Priority score."
*   **Write Slot Dist ($p_{write}$):** $\text{GumbelSoftmax}(W_{route\_w} \cdot P)$. "Where to store this?"

---

## 3. Memory Operations (The Dual-Gated Cycle)

For each layer $j$ and input $x$:

### Step 1: Active Read
Unlike standard attention which *always* attends, active reading is gated.

1.  **Query:** $q = W_Q \cdot x$
2.  **Attend:** Compute attention over cache $\mathbf{C}$.
    $$\alpha = \text{Softmax}\left( \frac{q \cdot K^T}{\sqrt{d}} \right)$$
3.  **Extract:** $ctx_{raw} = \alpha \cdot V$
4.  **MoE Gating:** The retrieved context is masked by the Read Gate.
    $$ctx_{final} = g_{read} \cdot (W_{decomp} \cdot ctx_{raw})$$

If $g_{read} \approx 0$, the operational cost of the Read block can be skipped (in sparse implementations).

### Step 2: Gated Fusion
The model learns to blend immediate input with retrieved memory:

$$g_{fuse} = \sigma(W_{fuse} \cdot [x; ctx_{final}])$$
$$x_{fused} = g_{fuse} \cdot ctx_{final} + (1 - g_{fuse}) \cdot x$$

### Step 3: Computation & Pooling
Standard layer processing (Self-Attention + MLP) followed by Pattern Pooling:

$$y = \text{Layer}(x_{fused})$$
$$P = \text{Pool}(y)$$

### Step 4: Active Write
The router evaluates the patterns $P$:

1.  **Score:** $\text{Score} = g_{write} \cdot s_{imp}$
2.  **Filter:** Only patterns with $\text{Score} > \tau$ are candidates for writing.
3.  **Route:** Select slot $k$ via $p_{write}$.
4.  **Update:**
    $$\mathbf{C}[k] \leftarrow (1-\alpha)\mathbf{C}[k] + \alpha \frac{\sum P \cdot e^{Score}}{\sum e^{Score}}$$
    (Soft Winner-Take-All update)

---

## 4. Hierarchical Adaptive Computation Time (H-ACT)

DLSMN v0.1.1 implements efficient reasoning through a two-tiered adaptive depth mechanism.

### 4.1 Model-Level ACT (The "Refiner")
*   **Scope:** Iterating the *entire model* (Pass 1, Pass 2, ...).
*   **Mechanism:** After each pass, a global **Q-Head** evaluates the cache state.
*   **Target Signal:** `is_correct` - whether the current answer matches ground truth.
*   **Logic:**
    ```python
    halt_prob = sigmoid(Q_Head(cache_state))
    # Training target: BCE(halt_prob, is_correct)
    # "If answer is correct, learn to halt. If wrong, learn to continue."
    if halt_prob > 0.5:
        RETURN output
    else:
        CONTINUE to next pass
    ```
*   **Purpose:** Easy samples exit early (1 pass); hard samples get deep thought (4+ passes).

### 4.2 Layer-Level ACT (The "Thinker")
*   **Scope:** Recurrence *within* a single layer (multiple recurrent steps per pass).
*   **Mechanism:** A layer-level Q-head predicts halting probability from hidden state.
*   **Target Signal:** Same `is_correct` from final answer (hindsight signal).
*   **Logic:**
    ```python
    for t in range(MAX_RECUR):
        h = Layer(h, cache, ...)
        halt_prob = layer_halt_net(h.mean(dim=1))  # Pool over sequence
        
        # Cumulative halting (ACT-style)
        cumulative_halt += (1 - cumulative_halt) * halt_prob
        
        if cumulative_halt > 0.99:
            BREAK
    
    # Training: BCE(halt_prob, is_correct) for all collected halt_probs
    # "If the final answer was correct, halting was OK. If wrong, should have kept refining."
    ```
*   **Purpose:** Layers learn to recognize when they've built sufficient representations for correct answers.

### 4.3 Why `is_correct` for Both Levels?

Previous approaches used **cosine similarity** (representation stability) as the layer-level target. This was problematic:
1. **Weak signal:** Stability ≠ correctness. A layer could stabilize on a wrong representation.
2. **Heuristic:** The 0.99 threshold was arbitrary.
3. **Buggy:** Shape mismatches and tracking overhead.

The **TRM insight**: The Q-head should learn "is my answer good enough?" not "has my representation stopped changing?" By using `is_correct` as the target for both levels:
- **Unified signal:** Both levels learn the same objective.
- **Meaningful gradient:** Layers learn what "ready to produce correct answers" looks like.
- **Hindsight learning:** The target is computed after the pass completes, providing a clear supervision signal.

---

## 5. Recursive Dynamics (TRM Integration)

Inspired by the Tiny Recursive Model (Reference 2510.04871v1), v0.1.1 adopts explicit recursive training dynamics.

### 5.1 Answer Feedback
The distinct answer from Pass $P$ is fed back effectively as a "hint" for Pass $P+1$.
$$x_{in}^{(p+1)} = \text{Embed}(Input) + \text{Gate}(\text{Embed}(\text{Answer}^{(p)}_{detached}))$$

**Crucial:** The feedback is **gradient-detached**. The model sees its past self as a fixed external oracle. This stabilizes training by preventing "gradient explosion through time."

### 5.2 No-Grad Warmup
To simulate deep recurrence without linear memory scaling:
*   **Pass 1 to T-1:** Run with `torch.no_grad()`. Populates cache, refines answer.
*   **Pass T:** Run with gradients. Backpropagate.

This allows training 8-16 pass reasoning chains with the memory footprint of a single pass.

### 5.3 Explicit Q-Head Training
The halting mechanism isn't magic; it's trained via supervision.
$$\mathcal{L}_{Q} = \text{BCE}(P_{halt}, \mathbb{1}_{AnswerIsCorrect})$$
The model is explicitly taught: *"If your answer is correct, output Halt=1. If wrong, output Halt=0."*

---

## 6. Design Goals & Novelty

### 6.1 Why Bi-Directional MoE?
In v0.1, we only filtered writes. However, **reading** is also noisy. By learning to *ignore* the cache ($g_{read} \approx 0$), the model constructs cleaner local features before attempting global reasoning. The MoE Router provides this explicit attention control.

### 6.2 Why Pattern Pooling?
Storing raw tokens ($S \times D$) in limited slots ($K \times D$) leads to "competition crowding." Pooling allows the layer to summarize "$S$ pixels of a blue line" into a single "Blue Line Pattern" before storage. It acts as a **semantic compression** step.

---

## 7. Training Stability & Curriculum

### 7.1 Cold Start: Noise Injection
In early training, the Router is dumb and may collapse to using only Slot 0.
**Mitigation:** Add decaying Gaussian noise to router logits for the first 10% of steps.
$$Logits = Logits + \mathcal{N}(0, \sigma_{decay})$$
This forces exploration of all memory slots.

### 7.2 Gumbel-Softmax Annealing
We transition from "soft" concepts to "hard" facts.
*   **Phase 1 (Warmup):** $\tau=1.0$. Soft routing. Everything writes everywhere weakly.
*   **Phase 2 (Hardening):** $\tau \to 0.1$. Hard routing. Discrete "fact storage."

---

## 8. Auxiliary Losses

We adopt a minimalist loss philosophy in v0.1.1.

### 8.1 Diversity Loss (Temporary)
Active **only** during warmup to seed the slots.
$$\mathcal{L}_{div} = -\lambda \cdot H(\bar{p}_{slots})$$
Ensures all slots get non-zero gradients initially. Disabled once routing stabilizes.

### 8.2 Q-Head Loss
$$\mathcal{L}_{Q} = \text{BCE}(P_{predicted\_halt}, P_{ground\_truth\_correct})$$
Critical for H-ACT. Without this, the model doesn't know *why* it should stop.

### 8.3 Ponder Loss
$$\mathcal{L}_{ponder} = \sum_{p} (1 - P_{halt}^{(p)})$$
A small penalty for thinking too long, rewarding efficient solution paths.

---

## 9. Advanced Extensions

### 9.1 Cache-to-Cache Attention
To enable "sleep-like" consolidation, we can run passes of **Inter-Cache Attention** between computation passes.
$$\mathbf{C}' = \text{SelfAttn}(\mathbf{C})$$
This allows the memory to deduplicate entries and propagate graph-like relationships (A->B, B->C implies A->C) without standard layer compute.

### 9.2 Cross-Sequence Persistence (RNN Mode)
Because the cache is structured ($L \times K$), it can be persisted across time for video/dialogue tasks. We append a normalized **Age Embedding** to each slot so the model can distinguish "historical facts" from "current observations."

---

## 10. Summary

DLSMN v0.1.1 represents a shift from **Passive Memory** (v0.1) to **Active Recursive Reasoning**.

*   **MoE Router** gives agency over memory traffic.
*   **Pattern Pooling** enforces semantic compression.
*   **H-ACT & TRM** provide the control loop for iterative thought.

This architecture is designed not just to "remember" but to **refine**—using memory as a workspace for solving reasoning-heavy tasks like the ARC Benchmark.
