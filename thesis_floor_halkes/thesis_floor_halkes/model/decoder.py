import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class FixedContext(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_context = nn.Linear(3 * embed_dim, embed_dim)

    def forward(self, final_node_embeddings, current_idx, end_idx):
        graph_embedding = final_node_embeddings.mean(dim=0)  # (batch_size, embed_dim)

        current_node_embedding = final_node_embeddings[
            current_idx, :
        ]  # (batch_size, embed_dim)
        end_node_embedding = final_node_embeddings[
            end_idx, :
        ]  # (batch_size, embed_dim)

        context = torch.cat(
            [graph_embedding, current_node_embedding, end_node_embedding], dim=-1
        )  # (batch_size, 3*embed_dim)

        return context


class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_context = nn.Linear(3 * embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.project_keys = nn.Linear(embed_dim, embed_dim)
        self.project_values = nn.Linear(embed_dim, embed_dim)
        self.project_attn_output = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        context_vector,
        node_embeddings,
        invalid_action_mask: torch.Tensor | None = None,
        greedy: bool = False,
    ):
        # Prepare query from context vector
        query = (
            self.project_context(context_vector).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, embed_dim]

        # Select only valid node embeddings
        valid_indices = (~invalid_action_mask).nonzero(as_tuple=True)[0]  # [num_valid]
        
        
        valid_keys = (
            (node_embeddings[valid_indices]
            #  .clone()
             .unsqueeze(0))
        )  # [1, num_valid, embed_dim]
        
        valid_values = (
            (node_embeddings[valid_indices]
            #  .clone()
             .unsqueeze(0))
        )

        # Compute attention output (ignore weights for performance)
        attn_output, _ = self.attn(
            query, valid_keys, valid_values, need_weights=False
        )  # [1, 1, embed_dim]
        attn_output = attn_output.squeeze(0).squeeze(0)  # [embed_dim]

        # Compute scores for each valid node
        scores = torch.matmul(valid_keys.squeeze(0), attn_output)  # [num_valid]
        scores = scores / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32, device=scores.device))
        probs = F.softmax(scores, dim=-1)

        assert probs.shape == valid_indices.shape

        if greedy:
            sampled_idx = torch.argmax(probs)
            entropy = torch.tensor(0.0, device=probs.device)
            log_prob = torch.log(probs[sampled_idx])
        else: 
            # Sample action
            dist = Categorical(probs)
            sampled_idx = dist.sample()  # scalar tensor
            entropy = dist.entropy()
            log_prob = dist.log_prob(sampled_idx)
        
        action = valid_indices[sampled_idx.item()].item()
        return action, log_prob, entropy
