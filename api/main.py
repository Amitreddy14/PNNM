import os
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import torchvision.transforms as transforms
import sys

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network import SelfPruningNetwork


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="PNNM — Self-Pruning Neural Network API",
    description="""
    A production API for querying and serving self-pruning neural network models
    trained on CIFAR-10. Models are trained with different sparsity levels (lambda),
    allowing deployment teams to select the best model for their hardware constraints.
    
    ## Endpoints
    - **GET  /model/results** — Full comparison table for all lambda experiments
    - **GET  /model/sparsity/{lambda_val}** — Detailed sparsity stats for one model
    - **POST /model/predict** — Classify a CIFAR-10 image using a selected model
    - **POST /model/recommend** — Get the best model for your deployment constraints
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "results.json"
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

VALID_LAMBDAS = [0.01, 0.1, 0.5]

# Image transform — same normalization used during training
TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])


# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────

class RecommendRequest(BaseModel):
    min_accuracy: float
    max_sparsity: float

    class Config:
        json_schema_extra = {
            "example": {
                "min_accuracy": 48.0,
                "max_sparsity": 70.0
            }
        }


class RecommendResponse(BaseModel):
    recommended_lambda: Optional[float]
    test_accuracy: Optional[float]
    sparsity: Optional[float]
    active_weights: Optional[int]
    total_weights: Optional[int]
    compression_ratio: Optional[str]
    reason: str


class SparsityResponse(BaseModel):
    lambda_val: float
    overall_sparsity_percent: float
    active_weights: int
    total_weights: int
    compression_ratio: str
    pruned_weights: int


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_results() -> list:
    """Load training results from JSON file."""
    if not os.path.exists(RESULTS_PATH):
        raise HTTPException(
            status_code=503,
            detail="Results not found. Please run train.py first."
        )
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def load_model(lambda_val: float) -> SelfPruningNetwork:
    """Load a trained model checkpoint for a given lambda."""
    model_path = os.path.join(MODELS_DIR, f"model_lambda_{lambda_val}.pth")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model for lambda={lambda_val} not found. Run train.py first."
        )
    model = SelfPruningNetwork()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def compute_compression_ratio(active: int, total: int) -> str:
    if active == 0:
        return "∞x"
    ratio = total / active
    return f"{ratio:.1f}x"


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "PNNM — Self-Pruning Neural Network API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health():
    results_exist = os.path.exists(RESULTS_PATH)
    models_exist = all(
        os.path.exists(os.path.join(MODELS_DIR, f"model_lambda_{lam}.pth"))
        for lam in VALID_LAMBDAS
    )
    return {
        "status": "healthy",
        "results_available": results_exist,
        "models_available": models_exist,
        "ready_for_inference": results_exist and models_exist
    }


# ── Endpoint 1 ──────────────────────────────

@app.get("/model/results", tags=["Model Info"])
def get_all_results():
    """
    Returns the full comparison table for all three lambda experiments.
    Includes test accuracy, sparsity level, and weight counts.
    """
    results = load_results()
    enriched = []
    for r in results:
        active = r.get("active_weights", 0)
        total = r.get("total_weights", 1)
        enriched.append({
            "lambda": r["lambda"],
            "test_accuracy": r["test_accuracy"],
            "sparsity_percent": r["sparsity"],
            "active_weights": active,
            "total_weights": total,
            "compression_ratio": compute_compression_ratio(active, total),
            "pruned_weights": total - active
        })
    return JSONResponse(content={"experiments": enriched, "count": len(enriched)})


# ── Endpoint 2 ──────────────────────────────

@app.get("/model/sparsity/{lambda_val}", tags=["Model Info"], response_model=SparsityResponse)
def get_sparsity(lambda_val: float):
    """
    Returns detailed sparsity statistics for a specific trained model.

    Valid lambda values: 0.0001, 0.001, 0.01
    """
    if lambda_val not in VALID_LAMBDAS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid lambda. Choose from: {VALID_LAMBDAS}"
        )

    model = load_model(lambda_val)
    sparsity = model.overall_sparsity()
    active, total = model.count_active_weights()

    return SparsityResponse(
        lambda_val=lambda_val,
        overall_sparsity_percent=round(sparsity, 2),
        active_weights=active,
        total_weights=total,
        compression_ratio=compute_compression_ratio(active, total),
        pruned_weights=total - active
    )


# ── Endpoint 3 ──────────────────────────────

@app.post("/model/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="A 32x32 RGB image (any format)"),
    lambda_val: float = Query(0.1, description="Which trained model to use: 0.01, 0.1, or 0.5")
):
    """
    Classifies an uploaded image using a trained self-pruning model.

    - Upload any image (will be resized to 32x32)
    - Select which lambda model to use for inference
    - Returns predicted CIFAR-10 class with confidence score
    """
    if lambda_val not in VALID_LAMBDAS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid lambda. Choose from: {VALID_LAMBDAS}"
        )

    # Read and preprocess image
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    tensor = TRANSFORM(image).unsqueeze(0)  # shape: (1, 3, 32, 32)

    # Load model and run inference
    model = load_model(lambda_val)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    predicted_class = CIFAR10_CLASSES[predicted_idx.item()]
    sparsity = model.overall_sparsity()

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4),
        "model_lambda": lambda_val,
        "model_sparsity_percent": round(sparsity, 2),
        "all_class_probabilities": {
            cls: round(probabilities[0][i].item(), 4)
            for i, cls in enumerate(CIFAR10_CLASSES)
        }
    }


# ── Endpoint 4 — The Unique Business Endpoint ──

@app.post("/model/recommend", tags=["Business Logic"], response_model=RecommendResponse)
def recommend_model(request: RecommendRequest):
    """
    **The Smart Deployment Advisor.**

    Given your deployment constraints (minimum accuracy + maximum sparsity),
    this endpoint automatically recommends the best trained model to deploy.

    This is designed for MLOps / DevOps teams who need to select the right
    model for a specific hardware target without manually analyzing results.

    Example use cases:
    - "I need at least 48% accuracy and my device can't handle more than 70% weight removal"
    - "Give me the most compressed model that still hits 45% accuracy"
    """
    results = load_results()

    # Filter models that meet both constraints
    eligible = [
        r for r in results
        if r["test_accuracy"] >= request.min_accuracy
        and r["sparsity"] <= request.max_sparsity
    ]

    if not eligible:
        # No model meets both constraints — give helpful guidance
        best_accuracy = max(r["test_accuracy"] for r in results)
        lowest_sparsity = min(r["sparsity"] for r in results)
        return RecommendResponse(
            recommended_lambda=None,
            test_accuracy=None,
            sparsity=None,
            active_weights=None,
            total_weights=None,
            compression_ratio=None,
            reason=(
                f"No model meets your constraints. "
                f"Best available accuracy is {best_accuracy}% "
                f"and minimum sparsity is {lowest_sparsity:.1f}%. "
                f"Try relaxing your thresholds."
            )
        )

    # From eligible models, pick the one with highest sparsity
    # (most compressed model that still meets accuracy requirement)
    best = max(eligible, key=lambda r: r["sparsity"])
    active = best.get("active_weights", 0)
    total = best.get("total_weights", 1)

    return RecommendResponse(
        recommended_lambda=best["lambda"],
        test_accuracy=best["test_accuracy"],
        sparsity=best["sparsity"],
        active_weights=active,
        total_weights=total,
        compression_ratio=compute_compression_ratio(active, total),
        reason=(
            f"Lambda={best['lambda']} gives the highest sparsity ({best['sparsity']}%) "
            f"while maintaining {best['test_accuracy']}% accuracy — "
            f"above your minimum threshold of {request.min_accuracy}%."
        )
    )