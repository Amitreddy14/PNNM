# PNNM — Self-Pruning Neural Network with FastAPI Deployment

> A production-ready implementation of a neural network that **learns to prune itself during training**, deployed as a FastAPI service with a business-oriented model selection API.

Built as part of the **Tredence Analytics AI Engineering Internship** case study.

---

##  What This Project Does

Most ML models are trained first, then pruned in a separate post-processing step. This project eliminates that two-step process.

**PNNM (Prunable Neural Network Model)** introduces learnable "gate" parameters directly into each linear layer. During training, a sparsity penalty drives these gates toward zero — effectively removing unnecessary weights *while the model is still learning*. The result is a smaller, faster model that is ready for edge deployment without any additional compression step.

### Business Value

| Problem | What PNNM Solves |
|---|---|
| Large models are expensive to serve | Pruning reduces active weights by up to 89.66% |
| Post-training pruning breaks accuracy | Gates are learned jointly — accuracy is preserved |
| Deployment constraints vary by client | Lambda API lets you pick the right sparsity level |
| No visibility into model compression | FastAPI exposes sparsity metrics in real time |

---

##  Architecture

```
PNNM/
├── prunable_layer.py     # Core: PrunableLinear with gate mechanism
├── network.py            # SelfPruningNetwork using PrunableLinear layers
├── train.py              # Training loop with sparsity loss + evaluation
├── report.md             # Analysis report with results and plots
├── requirements.txt      # Python dependencies
└── api/
    └── main.py           # FastAPI deployment with 4 endpoints
```

### How the Gate Mechanism Works

```
Standard Linear Layer:
  output = W · x + b

PrunableLinear Layer:
  gates        = sigmoid(gate_scores)        ← learned, range (0, 1)
  pruned_W     = W * gates                   ← element-wise kill switch
  output       = pruned_W · x + b

Total Loss = CrossEntropyLoss + λ × mean(all gate values)
                                  ↑
                         L1 penalty drives gates → 0
```

---

##  Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/Amitreddy14/PNNM.git
cd PNNM
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (runs 3 lambda experiments automatically)

```bash
python train.py
```

Training takes ~15–20 minutes on CPU. Results are saved to `results/`.

### 4. Start the FastAPI server

```bash
uvicorn api.main:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

---

##  API Endpoints

### `GET /model/results`
Returns the full comparison table for all three lambda values.

```json
[
  { "lambda": 0.01, "test_accuracy": 54.23, "sparsity": 62.98 },
  { "lambda": 0.1,  "test_accuracy": 53.93, "sparsity": 72.83 },
  { "lambda": 0.5,  "test_accuracy": 53.81, "sparsity": 89.66 }
]
```

---

### `GET /model/sparsity/{lambda_val}`
Returns detailed sparsity stats for a specific trained model.

```json
{
  "lambda": 0.1,
  "overall_sparsity_percent": 72.83,
  "active_weights": 472195,
  "total_weights": 1737984,
  "compression_ratio": "3.7x"
}
```

---

### `POST /model/predict`
Accepts a CIFAR-10 image (32×32 RGB), returns predicted class.

**Request:** `multipart/form-data` with image file + lambda value

**Response:**
```json
{
  "predicted_class": "automobile",
  "confidence": 0.87,
  "model_sparsity": 72.83
}
```

---

### `POST /model/recommend`  Unique Business Endpoint
Given deployment constraints (minimum accuracy + maximum sparsity), recommends the best lambda model to deploy.

**Request:**
```json
{
  "min_accuracy": 53.5,
  "max_sparsity": 75.0
}
```

**Response:**
```json
{
  "recommended_lambda": 0.1,
  "test_accuracy": 53.93,
  "sparsity": 72.83,
  "compression_ratio": "3.7x",
  "reason": "Lambda=0.1 gives the highest sparsity (72.83%) while maintaining 53.93% accuracy — above your minimum threshold of 53.5%."
}
```

This endpoint is designed for real business use — a DevOps or MLOps team can query it to get the optimal model for their edge device without manually analyzing results.

---

## 🖥️ Live API Preview

![Swagger UI](docs/swagger_ui.png)

![Swagger UI](docs/swagger_ui2.png)

##  Results Summary

| Lambda (λ) | Test Accuracy | Sparsity Level | Active Weights | Compression | Use Case |
|---|---|---|---|---|---|
| 0.01 (Low) | 54.23% | 62.98% | 643,371 / 1,737,984 | 2.7x | Accuracy-critical systems |
| 0.1 (Medium) | 53.93% | 72.83% | 472,195 / 1,737,984 | 3.7x | Balanced edge deployment |
| 0.5 (High) | 53.81% | 89.66% | 179,689 / 1,737,984 | 9.7x | IoT / heavily constrained devices |

**Key result:** λ=0.5 achieved **9.7x model compression** with less than **0.5% accuracy loss** compared to λ=0.01. Gate distribution plots are saved to `results/` after training.

---

##  Why L1 Penalty Drives Sparsity

The L1 norm has a geometric property that the L2 norm does not: its gradient is **constant** regardless of the value's magnitude. This means:

- A gate at 0.9 and a gate at 0.001 receive the **same push toward zero**
- Unlike L2, which only gently nudges small values, L1 actively drives them all the way to zero
- Combined with sigmoid gating, this creates a ratchet effect — gates that get pruned stay pruned

This is why LASSO regression (L1) produces sparse solutions, and why it is the standard choice for sparsity-inducing regularization in neural networks.

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Dataset | CIFAR-10 (via torchvision) |
| API Framework | FastAPI |
| Server | Uvicorn |
| Visualization | Matplotlib |
| Data Handling | NumPy |

---

##  Output Files (auto-generated after training)

```
results/
├── results.json                  # Full metrics for all lambda runs
├── model_lambda_0.01.pth         # Best model checkpoint (low lambda)
├── model_lambda_0.1.pth          # Best model checkpoint (medium lambda)
├── model_lambda_0.5.pth          # Best model checkpoint (high lambda)
├── gate_dist_lambda_0.01.png     # Gate distribution plot
├── gate_dist_lambda_0.1.png
└── gate_dist_lambda_0.5.png
```

> Model `.pth` files are excluded from the repo via `.gitignore` due to size. Run `train.py` to regenerate them locally.

---

##  Key Design Decisions

**Why initialize gate scores with `normal(mean=0, std=0.01)`?**
Gates start randomly mixed around `sigmoid(0) = 0.5` — some slightly above, some slightly below. This gives the optimizer room to push gates in either direction: toward 1 (active weight) or toward 0 (pruned weight). A uniform initialization would collapse all gates together with no differentiation.

**Why `mean()` instead of `sum()` for sparsity loss?**
Using `sum()` makes sparsity loss scale with the number of weights (1.7M+), completely overwhelming classification loss. Using `mean()` keeps sparsity loss bounded between 0 and 1 per layer — the same scale as cross entropy — so lambda values are meaningful and tunable.

**Why `CosineAnnealingLR`?**
It gradually reduces learning rate following a cosine curve, which prevents oscillation near convergence and improves final accuracy — standard practice in production training pipelines.

**Why save results to JSON?**
Decouples training from inference. The FastAPI server reads pre-computed results without needing to retrain, making it suitable for production deployment where training is a one-time offline job.

---

##  Author

**Amit** — AI Engineering Intern Applicant
Tredence Analytics — 2025 Cohort
GitHub: [Amitreddy14](https://github.com/Amitreddy14)

---

##  License

MIT License — free to use and modify.
