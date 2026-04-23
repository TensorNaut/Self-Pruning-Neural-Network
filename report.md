# Self-Pruning Neural Network — Report
**Assignment:** Tredence Analytics – AI Engineer Case Study  
**Dataset:** CIFAR-10  
**Author:** Tushar Jagannath Jagatap

---

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

### The Gating Mechanism

Each weight `w` in a `PrunableLinear` layer is modulated by a learnable gate before use:

```
gate  =  sigmoid(gate_score)    ∈ (0, 1)
eff_weight  =  w × gate
```

The sparsity regularisation term penalises the network for keeping too many gates open:

```
Total Loss = CrossEntropy(logits, labels) + λ × Σ sigmoid(gate_score)
```

Since sigmoid outputs are strictly positive, the sum equals the **L1 norm of all gate values**.

### Why L1 — Not L2 — Produces Exact Zeros

The gradient of the L1 norm with respect to a gate value is **constant** (= ±1) regardless of how small the gate currently is. This means the optimiser applies an undiminishing pull toward zero throughout training. In contrast, an L2 penalty has a gradient that shrinks proportionally with the gate magnitude, so small-but-non-zero gates are barely pushed at all and rarely reach exactly zero.

| Penalty | Gradient near gate ≈ 0 | Practical outcome |
|---------|------------------------|-------------------|
| **L2** (quadratic) | → 0 (diminishes) | Gates shrink but almost never reach exactly 0 |
| **L1** (absolute value) | **Constant = λ** | Applies constant pressure → drives gates all the way to 0 |

### The Role of Sigmoid

The sigmoid keeps gates in `(0, 1)` and is differentiable everywhere, so gradients flow cleanly through `gate → gate_score` during backpropagation with no need for straight-through estimators or hard thresholds. Once a gate is driven near zero, `gate_score` has been pushed toward `−∞`, reinforcing the pruned state.

The net effect: **the classification loss defends gates that carry useful signal; the L1 penalty closes every gate it cannot justify**.

---

## 2. Experimental Setup

| Component | Detail |
|-----------|--------|
| Architecture | Flatten → 3 × PrunableLinear blocks → output |
| Hidden dims | 3072 → 512 → 256 → 10 |
| Regularisation | BatchNorm + Dropout |
| Optimiser | Adam |
| Epochs | 30 |
| Prune threshold | gate < 0.01 |
| λ values tested | 1e-5, 1e-4, 5e-4, 1e-3, 2e-3 |

---

## 3. Results: λ Trade-off Summary

| λ (Lambda) | Test Accuracy (%) | Sparsity Level (%) | Observation |
|:----------:|:-----------------:|:------------------:|-------------|
| 1e-5 | 55.37 | 10.97 | Minimal pruning; sparsity still growing at epoch 30 |
| **1e-4** | **56.09** | **59.33** | ✅ Best trade-off — highest accuracy, moderate compression |
| 5e-4 | 55.75 | 93.03 | Aggressive pruning; small accuracy cost |
| 1e-3 | 54.47 | 97.90 | Near-total sparsity; noticeable accuracy drop |
| 2e-3 | 53.08 | 99.27 | Extreme sparsity; network capacity heavily impaired |

### Key Observations

**1. λ = 1e-4 is the optimal operating point.**  
It achieves the highest test accuracy (56.09%) while pruning ~59% of all connections — roughly a 2.4× parameter reduction at minimal accuracy cost.

**2. Moderate pruning acts as implicit regularisation.**  
The accuracy at λ = 1e-4 *exceeds* the minimally-pruned λ = 1e-5 run (56.09% vs 55.37%). Removing redundant connections reduces overfitting, consistent with the known regularisation effect of sparsity.

**3. Over-pruning causes capacity collapse.**  
Beyond λ = 5e-4, sparsity exceeds 97% and accuracy falls below 55%. At this point the L1 penalty dominates the loss and suppresses connections that are genuinely informative, leading to underfitting.

**4. Accuracy plateau is stable.**  
For all λ values, accuracy stabilises within ~10 epochs and varies only marginally thereafter. The classification objective converges independently of how aggressively gates are being closed.

---

## 4. Training Dynamics

### Accuracy vs Epoch

All λ runs follow the same pattern: rapid improvement in epochs 1–10, followed by a stable plateau.

- **λ = 1e-5:** Peaks near ~56% around epoch 10, then drifts slightly downward — the network is underpruned and shows mild overfitting in later epochs.
- **λ = 1e-4:** Converges cleanly to ~56% and holds stable throughout. Ideal training behaviour.
- **λ ≥ 5e-4:** Plateaus below 56% with a slight downward drift as excessive pruning progressively removes useful connections.

### Sparsity vs Epoch — Sigmoidal Growth

All runs display a consistent **sigmoidal growth pattern** in the sparsity curve:

| Phase | Epochs | What happens |
|-------|--------|--------------|
| **Early** | 1–8 | Near-zero pruning; the network first learns meaningful representations |
| **Mid** | 8–22 | Rapid sigmoidal rise; redundant connections are identified and closed |
| **Late** | 22–30 | Saturation; remaining gates are informative and resist further pruning |

This confirms that **the model first learns, then compresses itself** — a desirable and well-structured training dynamic. The rate of the sigmoidal transition scales directly with λ: at λ = 2e-3 the transition completes before epoch 25, reaching ~100% sparsity. At λ = 1e-5 the curve has not yet saturated by epoch 30, indicating more epochs would yield further pruning.

---

## 5. Gate Value Distributions

### What a Successful Distribution Looks Like

A well-pruned model should show a **bimodal distribution**: a large spike near 0 (pruned connections) and a distinct cluster of retained connections at higher gate values. This separation indicates the network has learned to clearly distinguish important from unimportant weights.

### Observed Distributions by λ

**λ = 1e-5 (Plot 5):** Distribution is unimodal, centred around mid-to-high gate values with minimal mass near zero — consistent with 10.97% sparsity. Most gates remain open. The sparsity curve has not yet saturated, so further training would shift more mass toward zero.

**λ = 1e-4 (Plot 1 — Best Model):** A bimodal pattern is visible. A growing spike at near-zero values and a retained cluster of non-trivial gates reflects successful sparse gate learning. This is the target distribution: the network has clearly separated "keep" from "prune".

**λ = 5e-4 (Plot 4):** The zero-spike is dominant but a secondary cluster survives, corresponding to the ~7% of connections that are retained and responsible for the maintained accuracy.

**λ = 1e-3 (Plot 2) and λ = 2e-3 (Plot 3):** The distribution is almost entirely concentrated at zero. The network has closed virtually every connection including informative ones, explaining the accuracy penalty observed in the results table.

---

## 6. Model Compression Analysis

| λ | Sparsity (%) | Approx. Compression | Accuracy Cost vs Best |
|---|:---:|:---:|:---:|
| 1e-5 | 10.97 | ~1.1× | −0.72% |
| **1e-4** | **59.33** | **~2.4×** | **baseline** |
| 5e-4 | 93.03 | ~14.3× | −0.34% |
| 1e-3 | 97.90 | ~47.6× | −1.62% |
| 2e-3 | 99.27 | ~137× | −3.01% |

The λ = 5e-4 result is particularly noteworthy: a 14× compression (93% of weights removed) at only a 0.34% accuracy cost relative to the best model. For deployment scenarios where memory and compute are heavily constrained, this is a strong result.

---

## 7. Conclusion

The self-pruning mechanism works as designed. The L1 penalty on sigmoid gates drives the network to discover its own sparse substructure during training, without any post-training step. Key findings:

- **λ = 1e-4** is the recommended setting — 56.09% test accuracy with 59.33% sparsity (~2.4× compression).
- **Moderate sparsity improves accuracy** over the near-dense baseline, acting as implicit regularisation.
- **The sigmoidal sparsity growth curve** is a consistent signature of the gating mechanism working correctly across all λ values.
- **Over-pruning is the primary failure mode** and is easily controlled by reducing λ.
- At λ = 5e-4, a 93% sparse network retains competitive accuracy — a compelling result for edge or resource-constrained deployment.