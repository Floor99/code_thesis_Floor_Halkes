import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, hidden_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(*layers)

    def forward(self, node_embeddings):
        """
        node_embeddings: Tensor of shape [num_nodes, embed_dim]
        """
        pooled = node_embeddings.mean(dim=0)  # [embed_dim]
        return self.critic(pooled).squeeze(-1)
