import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import random


# ----- Encoders -----
class GATModelEncoderStatic(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        out_size=None,
        num_layers=2,
        heads=4,
        dropout=0.2,
        edge_attr_dim=None,
    ):
        super().__init__()
        out_size = out_size or hidden_size
        self.conv1 = GATConv(
            in_channels,
            hidden_size,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_attr_dim,
        )
        self.conv2 = GATConv(
            hidden_size * heads,
            out_size,
            heads=1,
            dropout=dropout,
            edge_dim=edge_attr_dim,
        )

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x


class DynamicGATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, heads=1, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden_size, heads=heads, concat=False, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_size, hidden_size, heads=1, concat=False, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dyn, edge_index):
        h = self.dropout(x_dyn)
        h = F.elu(self.conv1(h, edge_index))
        h = self.dropout(h)
        h = F.elu(self.conv2(h, edge_index))
        return h


# ----- Decoder -----
class AttentionDecoderChat(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward_step(self, node_embeddings, current_node_idx, invalid_action_mask=None):
        query = (
            self.query_proj(node_embeddings[current_node_idx]).unsqueeze(0).unsqueeze(0)
        )
        keys = self.key_proj(node_embeddings).unsqueeze(1)
        values = self.value_proj(node_embeddings).unsqueeze(1)

        if invalid_action_mask is not None:
            invalid_action_mask = invalid_action_mask.unsqueeze(0)

        _, attn_weights = self.attn(
            query, keys, values, key_padding_mask=invalid_action_mask
        )
        logits = attn_weights.squeeze(0).squeeze(0)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, logits


# ----- Embedding Utility -----
def embed_graph(data, encoder, graph_type="static"):
    x_static, x_dynamic, edge_index, edge_attr = data
    if graph_type == "static":
        return encoder(x_static, edge_index, edge_attr)
    elif graph_type == "dynamic":
        return encoder(x_dynamic, edge_index)
    else:
        raise ValueError("Invalid graph_type")


# ----- Simulated State -----
class DataWrapper:
    def __init__(self, x_static, x_dynamic, edge_index, edge_attr):
        self.x_static = x_static
        self.x_dynamic = x_dynamic
        self.edge_index = edge_index
        self.edge_attr = edge_attr


class State:
    def __init__(self, static_data, dynamic_data, current_node, visited):
        self.static_data = static_data
        self.dynamic_data = dynamic_data
        self.current_node = current_node
        self.visited = visited


# ----- Select Action -----
def select_action(state, static_encoder, dynamic_encoder, decoder):
    static_emb = embed_graph(
        (
            state.static_data.x_static,
            state.static_data.x_dynamic,
            state.static_data.edge_index,
            state.static_data.edge_attr,
        ),
        static_encoder,
        "static",
    )
    dynamic_emb = embed_graph(
        (
            state.dynamic_data.x_static,
            state.dynamic_data.x_dynamic,
            state.dynamic_data.edge_index,
            state.dynamic_data.edge_attr,
        ),
        dynamic_encoder,
        "dynamic",
    )
    final_emb = torch.cat((static_emb, dynamic_emb), dim=1)

    action, log_prob, logits = decoder.forward_step(
        final_emb, state.current_node, state.visited
    )
    return action, log_prob


# ----- Build Sample Graph -----
def create_test_graph(num_nodes=6):
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0, 2, 3], [1, 2, 3, 4, 5, 2, 4, 5]], dtype=torch.long
    )
    x_static = torch.randn(num_nodes, 1)  # 1 static node feature
    x_dynamic = torch.randn(num_nodes, 2)  # 2 dynamic node features
    edge_attr = torch.randn(edge_index.size(1), 2)  # 2 static edge features
    return x_static, x_dynamic, edge_index, edge_attr


# ----- Main Test -----
def main():
    num_nodes = 6
    hidden_size = 64
    static_encoder = GATModelEncoderStatic(
        in_channels=1, hidden_size=hidden_size, edge_attr_dim=2
    )
    dynamic_encoder = DynamicGATEncoder(in_channels=2, hidden_size=hidden_size)
    decoder = AttentionDecoderChat(embed_dim=hidden_size * 2, num_heads=4)

    x_static, x_dynamic, edge_index, edge_attr = create_test_graph(num_nodes)
    static_data = DataWrapper(x_static, x_dynamic, edge_index, edge_attr)
    dynamic_data = DataWrapper(x_static, x_dynamic, edge_index, edge_attr)
    visited = torch.tensor([False, True, False, False, False, True])  # Example mask
    current_node = 0

    state = State(static_data, dynamic_data, current_node, visited)
    action, log_prob = select_action(state, static_encoder, dynamic_encoder, decoder)

    print(f"Action selected: {action}")
    print(f"Log probability: {log_prob.item():.4f}")


if __name__ == "__main__":
    main()
