import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gates.
    Uses a hard-sigmoid approach: gate_scores are learned,
    and a hard threshold determines pruning during evaluation.
    During training, soft gates allow gradients to flow.
    """

    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Gate scores — initialized near 0 so sigmoid(score) starts near 0.5
        # Small random init means gates naturally diverge during training
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        # Key fix: random init with small values so gates start mixed
        # Some will go to 0, some to 1 — creates differentiation
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

    def forward(self, x):
        # Soft gate during training — gradients flow through sigmoid
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold=0.5):
        # threshold=0.5 now — gate below 0.5 means score is negative = pruned
        gates = self.get_gates()
        pruned = (gates < threshold).sum().item()
        total = gates.numel()
        return pruned / total * 100