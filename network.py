import torch
import torch.nn as nn
from prunable_layer import PrunableLinear


class SelfPruningNetwork(nn.Module):
    """
    A feed-forward network for CIFAR-10 image classification
    built entirely with PrunableLinear layers.

    CIFAR-10 images are 32x32 RGB = 3072 input features.
    10 output classes.

    Business value: same accuracy as a standard network but
    learns to shed unnecessary weights during training,
    producing a smaller, faster model ready for edge deployment.
    """

    def __init__(self, input_size=3072, num_classes=10):
        super(SelfPruningNetwork, self).__init__()

        self.network = nn.Sequential(
            # Layer 1: 3072 -> 512
            PrunableLinear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 512 -> 256
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 256 -> 128
            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Output layer: 128 -> 10
            PrunableLinear(128, 10)
        )

    def forward(self, x):
        # Flatten image: (batch, 3, 32, 32) -> (batch, 3072)
        x = x.view(x.size(0), -1)
        return self.network(x)

    def get_prunable_layers(self):
        # Returns only the PrunableLinear layers
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self):
        total_gate_loss = 0
        for layer in self.get_prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total_gate_loss += gates.mean()  # mean not sum — stays bounded 0 to 1
        return total_gate_loss

    def overall_sparsity(self, threshold=0.5):
        total_pruned = 0
        total_weights = 0
        for layer in self.get_prunable_layers():
            gates = layer.get_gates()
            total_pruned += (gates < threshold).sum().item()
            total_weights += gates.numel()
        return (total_pruned / total_weights) * 100

    def count_active_weights(self, threshold=0.5):
        total_active = 0
        total_weights = 0
        for layer in self.get_prunable_layers():
            gates = layer.get_gates()
            total_active += (gates >= threshold).sum().item()
            total_weights += gates.numel()
        return total_active, total_weights