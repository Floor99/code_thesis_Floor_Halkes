from thesis_floor_halkes.agent.base import Agent
import torch.nn as nn
import torch
import torch.nn.functional as F
from thesis_floor_halkes.model.encoders import CacheStaticEmbedding
from thesis_floor_halkes.state import State

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # For testing purposes, use CPU

# class DynamicAgent(Agent):
#     """
#     A dynamic agent that adapts its behavior based on the environment.
#     """

#     def __init__(
#         self,
#         static_encoder: nn.Module,
#         dynamic_encoder: nn.Module,
#         decoder: nn.Module,
#         fixed_context: nn.Module,
#         baseline: nn.Module = None,
#         gamma: float = 0.8,
#         entropy_coeff: float = 0.01,
#     ):
#         """
#         Initialize the dynamic agent with static and dynamic encoders and a decoder.

#         Args:
#             static_encoder: The static encoder for processing static features.
#             dynamic_encoder: The dynamic encoder for processing dynamic features.
#             decoder: The decoder for generating actions.
#         """
#         self.static_encoder = static_encoder
#         self.dynamic_encoder = dynamic_encoder
#         self.cached_static = None
#         self.decoder = decoder
#         self.fixed_context = fixed_context
#         self.baseline = baseline
#         self.gamma = gamma
#         self.entropy_coeff = entropy_coeff

#         params = (
#             list(static_encoder.parameters())
#             + list(dynamic_encoder.parameters())
#             + list(decoder.parameters())
#         )

#         if baseline is not None:
#             params += list(baseline.parameters())  # get_learnable_parameters() ?!

#         self.states = []  # list van states en dan telkens in select de laatste state pakken?
#         self.actions = []
#         self.action_log_probs = []
#         self.rewards = []
#         self.penalties = []
#         self.baseline_values = []
#         self.embeddings = []
#         self.current_route = []
#         self.entropies = []

#     def _embed_graph(self, data, graph_type="static"):
#         if graph_type == "static":
#             x_static = data.x.to(device)
#             edge_index = data.edge_index.to(device)
#             edge_attr = data.edge_attr.to(device)
#             return self.static_encoder(x_static, edge_index, edge_attr)
#         elif graph_type == "dynamic":
#             x_dynamic = data.x.to(device)
#             edge_index = data.edge_index.to(device)
#             return self.dynamic_encoder(x_dynamic, edge_index)
#         else:
#             raise ValueError("Invalid graph type. Use 'static' or 'dynamic'.")

#     def select_action(self, state: State, greedy: bool = False):
#         if self.cached_static is None:
#             static_embedding = self._embed_graph(state.static_data, graph_type="static")
#             self.cached_static = CacheStaticEmbedding(static_embedding)
#         else:
#             static_embedding = self.cached_static.static_embedding

#         dynamic_embedding = self._embed_graph(state.dynamic_data, graph_type="dynamic")

#         final_embedding = torch.cat((static_embedding, dynamic_embedding), dim=1)

#         graph_embedding = final_embedding.mean(dim=0)

#         self.embeddings.append(
#             {
#                 "static": static_embedding,
#                 "dynamic": dynamic_embedding,
#                 "final": final_embedding,
#                 "graph": graph_embedding,
#             }
#         )

#         invalid_action_mask = self._get_action_mask(
#             state.valid_actions, state.static_data.num_nodes
#         )

#         context_vector = self.fixed_context(
#             final_node_embeddings=final_embedding,
#             current_idx=state.current_node,
#             end_idx=state.end_node,
#         )

#         action, action_log_prob, entropy = self.decoder(
#             context_vector=context_vector,
#             node_embeddings=final_embedding,
#             invalid_action_mask=invalid_action_mask,
#         )

#         if not self.current_route:
#             self.current_route.append(state.start_node)

#         self.current_route.append(action)

#         return action, action_log_prob , entropy

#     def _get_action_mask(
#         self, valid_actions: list[int], num_nodes: int
#     ) -> torch.Tensor:
#         """
#         Create a mask for valid actions.
#         """
#         action_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
#         action_mask[valid_actions] = 0
#         return action_mask

#     def store_baseline_value(self, baseline_value):
#         self.baseline_values.append(baseline_value)

#     def store_state(self, state: State):
#         self.states.append(state)

#     def store_penalty(self, penalty: float):
#         self.penalties.append(penalty)

#     def store_reward(self, reward: float):
#         self.rewards.append(reward)

#     def store_action(self, action):
#         self.actions.append(action)

#     def store_action_log_prob(self, log_prob):
#         self.action_log_probs.append(log_prob)

#     def store_entropy(self, entropy):
#         self.entropies.append(entropy)

#     def decay_entropy_coeff(self, decay_rate: float=0.995, min_entropy_coeff: float=1e-4):
#         self.entropy_coeff = max(
#             self.entropy_coeff * decay_rate, min_entropy_coeff
#         )
    

#     def finish_episode(self):
#         R = 0
#         returns = []
#         for r in reversed(self.rewards):
#             R = r + self.gamma * R
#             returns.insert(0, R)
#         returns = torch.tensor(returns, device=device)
#         if returns.numel() > 1:
#             returns = (returns - returns.mean()) / (returns.std() + 1e-6)

#         if self.baseline is not None:
#             baseline_values = torch.stack(self.baseline_values).to(device)
#             advantages = (returns - baseline_values).detach()
#             # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
#             baseline_loss = F.mse_loss(baseline_values, returns)
#         else:
#             print("No baseline model provided. Using returns as advantages.")
#             advantages = returns
#             baseline_loss = 0

#         log_probs_tensor = torch.stack(self.action_log_probs).to(device)
#         policy_loss = -(log_probs_tensor * advantages)
#         policy_loss = -(log_probs_tensor * advantages).mean()
#         entropy_loss = torch.stack(self.entropies).mean()
#         policy_loss = policy_loss - self.entropy_coeff * entropy_loss

#         return policy_loss, baseline_loss, entropy_loss

#     def reset(self):
#         self.action_log_probs.clear()
#         self.rewards.clear()
#         self.states.clear()
#         self.actions.clear()
#         self.penalties.clear()
#         self.baseline_values.clear()
#         self.embeddings.clear()
#         self.cached_static = None
#         self.current_route.clear()
#         self.entropies.clear()

#     def backprop_model(self, optimizer, loss):
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


class DynamicAgent():
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
