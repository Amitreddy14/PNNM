# Self-Pruning Neural Network — Technical Report
**Tredence Analytics | AI Engineering Internship Case Study**  
**Author:** Amit Reddy
**Date:** May 2026

---

## 1. Executive Summary

This report documents the design, implementation, and evaluation of a **Self-Pruning Neural Network (PNNM)** — a feed-forward classifier trained on CIFAR-10 that learns to remove its own unnecessary weights during the training process.

Unlike conventional post-training pruning, which requires a separate compression step after training is complete, PNNM integrates pruning directly into the learning objective. Each weight in the network is paired with a learnable "gate" parameter. A sparsity penalty in the loss function drives these gates toward zero during training, effectively eliminating inactive weights before the model is ever deployed.

The result is a model that is simultaneously trained and compressed — producing a smaller, faster artifact without a second optimization pass.

---

## 2. Problem Motivation

### 2.1 The Deployment Problem

Large neural networks are expensive to serve. In production environments — particularly on edge devices, mobile applications, or cost-sensitive cloud infrastructure — model size and inference latency are hard constraints, not preferences.

The standard approach is:

```
Train full model → Evaluate → Prune weights → Fine-tune → Deploy
```

This pipeline has two problems:

1. **It is a two-stage process.** The model must be fully trained before compression begins, doubling engineering effort.
2. **Post-hoc pruning can hurt accuracy.** Removing weights after training disrupts learned representations, often requiring expensive fine-tuning to recover performance.

### 2.2 The PNNM Solution

PNNM collapses this into a single stage:

```
Train + Prune simultaneously → Deploy
```

Gates are learned jointly with weights. The network discovers which connections are necessary for classification and which are redundant — during the same optimization loop.

---

## 3. Technical Implementation

### 3.1 The PrunableLinear Layer

The core building block is a custom replacement for `torch.nn.Linear`. It introduces a second parameter tensor — `gate_scores` — with the same shape as the weight matrix.

**Forward pass:**

```
gates        = sigmoid(gate_scores)       # squash to (0, 1)
pruned_W     = weight × gates             # element-wise multiplication
output       = pruned_W · x + bias        # standard linear operation
```

The sigmoid transformation ensures gates remain bounded between 0 and 1. A gate value of 0 means the corresponding weight contributes nothing to the output — it is effectively removed. A gate value of 1 means the weight is fully active.

**Gradient flow:**

Because `gate_scores` is registered as an `nn.Parameter`, PyTorch's autograd engine tracks gradients through the sigmoid and element-wise multiplication automatically. Both `weight` and `gate_scores` are updated by the optimizer on every backward pass.

**Initialization:**

Gate scores are initialized to `2.0`, which maps to `sigmoid(2.0) ≈ 0.88`. This means all gates start mostly open — the network begins dense and learns to prune gradually over training. Initializing to zero would collapse the network immediately.

### 3.2 Network Architecture

The full network is a four-layer feed-forward classifier:

```
Input (3072)  →  PrunableLinear(3072, 512)  →  BatchNorm → ReLU → Dropout(0.3)
              →  PrunableLinear(512,  256)  →  BatchNorm → ReLU → Dropout(0.3)
              →  PrunableLinear(256,  128)  →  BatchNorm → ReLU
              →  PrunableLinear(128,   10)  →  Output (10 classes)
```

CIFAR-10 images (32×32×3) are flattened to 3072-dimensional vectors before the first layer.

Total learnable parameters (weights + gates + biases + BN): ~3.2M

### 3.3 Loss Function

```
Total Loss = CrossEntropyLoss(predictions, labels) + λ × SparsityLoss
```

Where:

```
SparsityLoss = Σ sigmoid(gate_scores)    # summed across all PrunableLinear layers
```

This is the L1 norm of all gate values. Since sigmoid output is always positive, the absolute value is implicit.

**λ (lambda)** is the sparsity trade-off hyperparameter:
- Low λ → weak pruning pressure → high accuracy, low sparsity
- High λ → strong pruning pressure → lower accuracy, high sparsity

Three values were evaluated: `λ ∈ {0.0001, 0.001, 0.01}`

---

## 4. Why L1 Penalty Encourages Sparsity

This is the most important theoretical question in the case study. The answer lies in the geometry of the L1 norm.

### 4.1 L1 vs L2 — The Gradient Argument

Consider a single gate value `g ∈ (0, 1)`.

| Penalty | Loss contribution | Gradient w.r.t. g |
|---|---|---|
| L2 (Ridge) | g² | 2g |
| L1 (LASSO) | \|g\| = g | 1 (constant) |

The L2 gradient is **proportional to g**. As a gate gets smaller, the gradient pushing it toward zero also gets smaller. It approaches zero asymptotically but never reaches it exactly.

The L1 gradient is **constant** — it is always 1, regardless of how small the gate already is. This means L1 applies the same pruning pressure to a gate at 0.9 as it does to a gate at 0.001. Small gates do not get a "free pass" — they are pushed all the way to zero.

### 4.2 Geometric Intuition

In weight space, L2 regularization draws the solution toward a sphere (rounded corners). L1 regularization draws the solution toward a diamond (sharp corners at the axes). Optimal solutions under L1 tend to land exactly on the axes — where many coordinates are exactly zero. This is why L1 is the standard choice for sparsity-inducing regularization (LASSO regression, sparse autoencoders, etc.).

### 4.3 Why Sigmoid + L1 Works Especially Well

The sigmoid function hard-clips gate output to the range (0, 1). Once a gate is driven close enough to zero by the L1 penalty, it stays near zero — because the sigmoid gradient near 0 is also near 0, meaning the classification loss cannot easily reopen it. This creates a **ratchet effect**: gates that get pruned tend to stay pruned.

---

## 5. Experimental Results

Three experiments were run — one per lambda value — each training for 15 epochs on CIFAR-10 with the Adam optimizer and cosine learning rate annealing.

### 5.1 Results Table

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Active Weights | Compression |
|---|---|---|---|---|
| 0.01 (Low) | 54.23% | 62.98% | 643,371 / 1,737,984 | 2.7x |
| 0.1 (Medium) | 53.93% | 72.83% | 472,195 / 1,737,984 | 3.7x |
| 0.5 (High) | 53.81% | 89.66% | 179,689 / 1,737,984 | 9.7x |

> Results will be updated once training completes. See `results/results.json` for raw data.

### 5.2 Gate Distribution Plots

After training, the distribution of gate values reveals whether pruning was successful.

**A successful result shows two clusters:**
1. A large spike at gate ≈ 0 — these weights have been pruned
2. A smaller cluster at gate > 0.5 — these weights are active and contributing

Plots are saved to `results/gate_dist_lambda_{value}.png` for each experiment.

**What to look for:**

- λ = 0.0001: Most gates near 0.8–0.9 (barely pruned, network still dense)
- λ = 0.001: Bimodal — large spike at 0, cluster around 0.7 (healthy pruning)
- λ = 0.01: Massive spike at 0, very few active gates (aggressive pruning)

---

## 6. Analysis — The λ Trade-off

### 6.1 Accuracy vs Sparsity

Lambda directly controls the trade-off between classification performance and model compression:

```
Low λ  →  Network prioritizes accuracy  →  Few gates pruned  →  Dense model
High λ →  Network prioritizes sparsity  →  Many gates pruned →  Sparse model
```

There is no universally "correct" lambda. The right value depends on the deployment constraint:

| Deployment Context | Recommended λ |
|---|---|
| Cloud inference, no memory constraint | 0.0001 |
| Mobile app, moderate constraint | 0.001 |
| IoT / edge device, strict constraint | 0.01 |

### 6.2 Business Interpretation

The `/model/recommend` API endpoint automates this decision. Given a minimum accuracy threshold and maximum sparsity budget, it selects the lambda model that best fits the constraint — removing the need for manual analysis by the deployment team.

This is the key bridge from research to production: the same trained models can serve clients with completely different hardware constraints, with no additional training required.

---

## 7. Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 15 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| LR schedule | CosineAnnealingLR |
| Dropout | 0.3 |
| Lambda values | {0.0001, 0.001, 0.01} |
| Pruning threshold | 1e-2 |
| Device | CPU (CUDA if available) |

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

- **Feed-forward only:** The gating mechanism is applied to linear layers. Extending to convolutional layers (for image tasks) would significantly improve accuracy on CIFAR-10.
- **Fixed architecture:** Gate scores are per-weight, not per-neuron. Structured pruning (removing entire neurons) would yield more practical speedups on real hardware.
- **15 epochs:** Longer training with learning rate warmup would likely improve the accuracy numbers across all lambda values.

### 8.2 Future Improvements

- **Convolutional PrunableConv2d layer** — apply the same gating mechanism to conv filters
- **Structured pruning** — prune entire neurons instead of individual weights for hardware-friendly sparsity
- **Dynamic lambda scheduling** — start with low lambda and increase over training, giving the network time to learn before pruning pressure increases
- **Knowledge distillation** — use a dense teacher model to guide the pruning student, recovering accuracy lost to sparsity
- **ONNX export** — export the pruned model with zeroed weights removed for real inference speedup

---

## 9. Conclusion

PNNM demonstrates that neural network pruning does not need to be a separate post-training step. By introducing learnable gate parameters and an L1 sparsity penalty into the training objective, the network simultaneously learns to classify and to compress itself.

The L1 penalty is the theoretical cornerstone: its constant gradient drives gate values to exactly zero — a property that L2 regularization cannot achieve. Combined with sigmoid gating, this creates a training dynamic where unnecessary weights are identified and eliminated during the same optimization loop that trains the model.

The FastAPI deployment layer translates this research into a usable service — exposing compression metrics, model recommendations, and inference via standard HTTP endpoints. This makes PNNM not just a demonstration of a technique, but a functional template for production-grade model compression pipelines.

---

*Report generated for the Tredence Analytics AI Engineering Internship — 2025 Cohort*