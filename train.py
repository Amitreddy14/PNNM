import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from network import SelfPruningNetwork


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LAMBDAS = [0.01, 0.1, 0.5]  # low, medium, high
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
RESULTS_DIR = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Training on: {DEVICE}")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True,
        download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False,
        download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2
    )

    return train_loader, test_loader


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, lambda_sparse):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Total loss = classification loss + sparsity penalty
        cls_loss = criterion(outputs, labels)
        sparse_loss = model.sparsity_loss()
        loss = cls_loss + lambda_sparse * sparse_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, lambda_sparse):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)

            cls_loss = criterion(outputs, labels)
            sparse_loss = model.sparsity_loss()
            loss = cls_loss + lambda_sparse * sparse_loss

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# GATE DISTRIBUTION PLOT
# ─────────────────────────────────────────────

def plot_gate_distribution(model, lambda_val):
    all_gates = []
    for layer in model.get_prunable_layers():
        gates = layer.get_gates().cpu().numpy().flatten()
        all_gates.extend(gates)

    all_gates = np.array(all_gates)

    plt.figure(figsize=(8, 4))
    plt.hist(all_gates, bins=100, color='steelblue', edgecolor='black', alpha=0.8)
    plt.title(f"Gate Value Distribution — λ = {lambda_val}", fontsize=13)
    plt.xlabel("Gate Value (0 = pruned, 1 = active)")
    plt.ylabel("Count")
    plt.axvline(x=0.01, color='red', linestyle='--', label='Prune threshold (0.01)')
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"gate_dist_lambda_{lambda_val}.png")
    plt.savefig(path)
    plt.close()
    print(f"  Plot saved: {path}")


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP — runs for all 3 lambdas
# ─────────────────────────────────────────────

def run_experiment(lambda_sparse, train_loader, test_loader):
    print(f"\n{'='*50}")
    print(f"  Training with λ = {lambda_sparse}")
    print(f"{'='*50}")

    model = SelfPruningNetwork().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_test_acc = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, lambda_sparse
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, lambda_sparse
        )
        scheduler.step()

        sparsity = model.overall_sparsity()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                model.state_dict(),
                os.path.join(RESULTS_DIR, f"model_lambda_{lambda_sparse}.pth")
            )

        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Sparsity: {sparsity:.1f}%"
        )

    # Load best model for final reporting
    model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, f"model_lambda_{lambda_sparse}.pth"))
    )

    final_sparsity = model.overall_sparsity()
    active, total = model.count_active_weights()
    plot_gate_distribution(model, lambda_sparse)

    print(f"\n  ✓ Best Test Accuracy : {best_test_acc:.2f}%")
    print(f"  ✓ Final Sparsity     : {final_sparsity:.2f}%")
    print(f"  ✓ Active Weights     : {active:,} / {total:,}")

    return {
        "lambda": lambda_sparse,
        "test_accuracy": round(best_test_acc, 2),
        "sparsity": round(final_sparsity, 2),
        "active_weights": active,
        "total_weights": total
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    all_results = []
    for lam in LAMBDAS:
        result = run_experiment(lam, train_loader, test_loader)
        all_results.append(result)

    # Save results to JSON (API will read this later)
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*50)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*50)
    print(f"\n{'Lambda':<12} {'Test Acc':>10} {'Sparsity':>12}")
    print("-" * 36)
    for r in all_results:
        print(f"{r['lambda']:<12} {r['test_accuracy']:>9}% {r['sparsity']:>11}%")

    print(f"\nResults saved to: {results_path}")