import torch
import torch.nn as nn

from thesis_floor_halkes.model.encoders import CacheStaticEmbedding
from thesis_floor_halkes.state import State

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # For testing purposes, use CPU


class DynamicAgent:
    """
    A dynamic agent that adapts its behavior based on the environment.
    """

    def __init__(
        self,
        static_encoder: nn.Module,
        dynamic_encoder: nn.Module,
        decoder: nn.Module,
        fixed_context: nn.Module,
        baseline: nn.Module = None,
    ):
        """
        Initialize the dynamic agent with static and dynamic encoders and a decoder.

        Args:
            static_encoder: The static encoder for processing static features.
            dynamic_encoder: The dynamic encoder for processing dynamic features.
            decoder: The decoder for generating actions.
        """
        self.static_encoder = static_encoder
        self.dynamic_encoder = dynamic_encoder
        self.cached_static = None
        self.decoder = decoder
        self.fixed_context = fixed_context
        self.baseline = baseline

    def _embed_graph(self, data, graph_type="static"):
        if graph_type == "static":
            x_static = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            return self.static_encoder(x_static, edge_index, edge_attr)
        elif graph_type == "dynamic":
            x_dynamic = data.x.to(device)
            edge_index = data.edge_index.to(device)
            return self.dynamic_encoder(x_dynamic, edge_index)
        else:
            raise ValueError("Invalid graph type. Use 'static' or 'dynamic'.")

    def select_action(self, state: State, greedy: bool = False):
        if self.cached_static is None:
            static_embedding = self._embed_graph(state.static_data, graph_type="static")
            self.cached_static = CacheStaticEmbedding(static_embedding)
        else:
            static_embedding = self.cached_static.static_embedding

        dynamic_embedding = self._embed_graph(state.dynamic_data, graph_type="dynamic")
        final_embedding = torch.cat((static_embedding, dynamic_embedding), dim=1)

        invalid_action_mask = self._get_action_mask(
            state.valid_actions, state.static_data.num_nodes
        )

        context_vector = self.fixed_context(
            final_node_embeddings=final_embedding,
            current_idx=state.current_node,
            end_idx=state.end_node,
        )

        action, action_log_prob, entropy = self.decoder(
            context_vector=context_vector,
            node_embeddings=final_embedding,
            invalid_action_mask=invalid_action_mask,
            greedy=greedy,
        )

        self.final_embedding = final_embedding

        return action, action_log_prob, entropy

    def _get_action_mask(
        self, valid_actions: list[int], num_nodes: int
    ) -> torch.Tensor:
        """
        Create a mask for valid actions.
        """
        action_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        action_mask[valid_actions] = 0
        return action_mask

    def reset(self):
        """
        Reset the agent's state.
        """
        self.cached_static = None
