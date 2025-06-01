from typing import NamedTuple
import torch
import torch.nn as nn
from torch_geometric.nn import GAT


class StaticGATEncoder(nn.Module):
    """
    High-level GAT encoder for static features using PyG's GAT model.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_size: int = None,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        edge_attr_dim: int = None,
    ):
        super().__init__()
        out_size = out_size or hidden_size
        self.gat = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_size,
            out_channels=out_size,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_attr_dim,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_attr: [E, edge_attr_dim] or None
        returns: [N, out_size]
        """
        return self.gat(x, edge_index, edge_attr=edge_attr)


class DynamicGATEncoder(nn.Module):
    """
    High-level GAT encoder for dynamic features using PyG's GAT model.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_size: int = None,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        out_size = out_size or hidden_size
        self.gat = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_size,
            out_channels=out_size,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_attr: [E, edge_attr_dim] or None
        returns: [N, out_size]
        """
        return self.gat(x, edge_index, edge_attr=edge_attr)


class CacheStaticEmbedding(NamedTuple):
    """
    Cached static embeddings for the current episode:
    static_embedding: [N, H]
    """

    static_embedding: torch.Tensor

    def __getitem__(self, idx):
        return CacheStaticEmbedding(
            static_embedding=self.static_embedding[idx],
        )
