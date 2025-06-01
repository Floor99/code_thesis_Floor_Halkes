import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GAT, GATConv
from typing import NamedTuple, List, Tuple


class GATModelEncoderStatic(nn.Module):
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
        x: [N_total, in_channels]
        edge_index: [2, E_total]
        edge_attr: [E_total, edge_attr_dim] or None
        returns: [N_total, out_size]
        """
        return self.gat(x, edge_index, edge_attr=edge_attr)


class DynamicGATEncoder(nn.Module):
    """
    Two-layer GAT encoder for dynamic features (no edge attributes).
    """

    def __init__(
        self, in_channels: int, hidden_size: int, heads: int = 1, dropout: float = 0.2
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden_size, heads=heads, concat=False, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_size, hidden_size, heads=1, concat=False, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dyn: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x_dyn: [N_total, in_channels]
        edge_index: [2, E_total]
        returns: [N_total, hidden_size]
        """
        h = self.dropout(x_dyn)
        h = F.elu(self.conv1(h, edge_index))
        h = self.dropout(h)
        h = F.elu(self.conv2(h, edge_index))
        return h


class FixedContext(NamedTuple):
    """
    Cached static context for attention decoding:
      - h_static: [N_total, H]
      - K_s, V_s, L_s: [N_total, H]
    """

    h_static: torch.Tensor
    K_s: torch.Tensor
    V_s: torch.Tensor
    L_s: torch.Tensor

    def __getitem__(self, idx):
        # support slicing by global node indices
        return FixedContext(
            h_static=self.h_static[idx],
            K_s=self.K_s[idx],
            V_s=self.V_s[idx],
            L_s=self.L_s[idx],
        )


class AttentionDecoderBatch(nn.Module):
    """
    Batched version of AttentionDecoder:
      - Precomputes static K/V/L per episode
      - Runs dynamic projections per step
      - Computes per-graph attention, masks invalid actions
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.H = hidden_size
        self.project_static_kvl = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.project_dyn_kvl = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.project_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.fixed_ctx: FixedContext = None

    def clear_cache(self):
        self.fixed_ctx = None

    def precompute(self, h_static: torch.Tensor):
        # Static projections once per episode
        kvl = self.project_static_kvl(h_static)  # [N_total, 3H]
        K_s, V_s, L_s = kvl.chunk(3, dim=-1)  # each [N_total, H]
        self.fixed_ctx = FixedContext(h_static, K_s, V_s, L_s)

    def forward_batch(
        self,
        h_dynamic: torch.Tensor,  # [N_total, H]
        batch: torch.Tensor,  # [N_total] node->graph mapping
        current_nodes: torch.LongTensor,  # [B] global node idx per graph
        valid_actions: List[List[int]],  # length B lists of global node idx
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        if self.fixed_ctx is None:
            raise RuntimeError(
                "Static context not precomputed. Call precompute() first."
            )
        fixed = self.fixed_ctx
        N, H = h_dynamic.size()
        B = current_nodes.size(0)
        device = h_dynamic.device

        # Dynamic projections
        kvl_d = self.project_dyn_kvl(h_dynamic)  # [N, 3H]
        K_d, V_d, _ = kvl_d.chunk(3, dim=-1)
        K = fixed.K_s + K_d  # [N, H]
        V = fixed.V_s + V_d  # [N, H]

        # Build queries for each graph's current node
        # q: [B, H]
        q = self.project_q(fixed.h_static[current_nodes] + h_dynamic[current_nodes])

        # Preallocate output tensors
        actions = torch.empty(B, dtype=torch.long, device=device)
        log_probs = torch.empty(B, device=device)

        # Loop over graphs and only score *their* neighbors
        for i in range(B):
            neigh = valid_actions[i]  # list of global node‐indices
            if len(neigh) == 0:
                # no valid moves: stay in place
                actions[i] = current_nodes[i]
                log_probs[i] = 0.0
                continue

            # Query vector for this graph
            q_i = q[i]  # [H]

            # Gather K & V only at neighbor positions
            K_i = K[neigh]  # [n_i, H]
            V_i = V[neigh]  # [n_i, H]

            # Compute attention‐compatibility as logits
            compat = (K_i @ q_i) / math.sqrt(self.H)  # [n_i]

            # Sample next node among neighbors
            dist = Categorical(logits=compat)
            idx = dist.sample()  # integer in [0 .. n_i)
            actions[i] = neigh[idx]  # global node‐index
            log_probs[i] = dist.log_prob(idx)

        return actions, log_probs


class PolicyNetworkGATDynamicAttention(nn.Module):
    """
    Full policy with batch support: static GAT, dynamic GATConv, and batched attention decoder.
    """

    def __init__(
        self,
        in_static: int,
        in_dyn: int,
        hidden_size: int,
        static_layers: int = 2,
        static_heads: int = 4,
        dyn_heads: int = 1,
        dropout: float = 0.2,
        edge_attr_dim: int = None,
    ):
        super().__init__()
        self.in_static = in_static
        self.static_enc = GATModelEncoderStatic(
            in_channels=in_static,
            hidden_size=hidden_size,
            out_size=hidden_size,
            num_layers=static_layers,
            heads=static_heads,
            dropout=dropout,
            edge_attr_dim=edge_attr_dim,
        )
        self.dyn_enc = DynamicGATEncoder(
            in_channels=in_dyn,
            hidden_size=hidden_size,
            heads=dyn_heads,
            dropout=dropout,
        )
        self.decoder = AttentionDecoderBatch(hidden_size)

    def clear_static_cache(self):
        self.decoder.clear_cache()

    def forward_batch(
        self, data, current_nodes: torch.LongTensor, valid_actions: List[List[int]]
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        x = data.x
        x_s = x[:, : self.in_static]
        x_d = x[:, self.in_static :]
        ei = data.edge_index
        ea = data.edge_attr

        # Precompute static projections once per episode
        if self.decoder.fixed_ctx is None:
            h_s = self.static_enc(x_s, ei, ea)
            self.decoder.precompute(h_s)

        # Dynamic each step
        h_d = self.dyn_enc(x_d, ei)

        # Batched decode
        return self.decoder.forward_batch(
            h_dynamic=h_d,
            batch=data.batch,
            current_nodes=current_nodes,
            valid_actions=valid_actions,
        )
