import torch
from torch.distributions import Categorical
from typing import List, Tuple

# from torch_geometric.data import Data
from reinforce_baselines_batch import NoBaseline
import torch_geometric


class AmbulanceAgent:
    def __init__(
        self,
        policy,  # your PolicyNetworkGATDynamicAttention
        baseline=None,  # NoBaseline / ExponentialBaseline / CriticBaseline
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.policy = policy
        self.baseline = baseline or NoBaseline()
        # collect policy + any learnable baseline params
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + self.baseline.get_learnable_parameters(), lr=lr
        )
        self.gamma = gamma

        # Buffers now hold T lists of [B]-tensors:
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.collected_states: List[torch_geometric.data.Data] = []

    def select_action(
        self,
        obs_batch: torch_geometric.data.Data,
        current_nodes: torch.LongTensor,  # [B]
        valid_actions: List[List[int]],  # length B
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Runs one forward pass over the entire batch.
        Returns:
          - actions:   LongTensor[B]
          - log_probs: Tensor[B]
        """
        # Save state for baseline
        self.collected_states.append(obs_batch)
        # Policy returns (actions, log_probs)
        actions, log_probs = self.policy.forward_batch(
            data=obs_batch, current_nodes=current_nodes, valid_actions=valid_actions
        )
        # Store for training
        self.log_probs.append(log_probs)
        return actions, log_probs

    def store_reward(self, rewards: torch.Tensor):
        """
        rewards: FloatTensor[B], the vector of rewards for this step across the batch.
        """
        self.rewards.append(rewards)

    def finish_episode(self):
        """
        Compute per-sample returns & advantages, then do one policy+baseline update.
        """
        # 1) Stack into [T, B]
        rewards = torch.stack(self.rewards)  # T × B
        B = rewards.size(1)

        # 2) Compute discounted returns per sample (vectorized)
        #    returns[t, b] = sum_{k>=t} gamma^(k-t) * rewards[k,b]
        # We can do this efficiently by flipping & cumsum:
        discounts = self.gamma ** torch.arange(
            rewards.size(0), device=rewards.device, dtype=rewards.dtype
        )
        returns = (rewards * discounts[:, None]).flip(0).cumsum(0).flip(0) / discounts[
            :, None
        ]
        # Now returns is [T, B]

        # 3) Flatten to [T*B] for baseline & loss
        returns_flat = returns.reshape(-1)
        # 4) Evaluate baselines
        baselines_flat, loss_b = self.baseline.eval(
            self.collected_states,  # list of DataBatch, length T
            returns_flat,
        )
        # 5) Advantage
        adv = returns_flat - baselines_flat
        # 6) Actor loss
        logp = torch.cat(self.log_probs)  # [T*B]
        actor_loss = -(logp * adv.detach()).mean()

        # 7) Total loss & backward
        loss = actor_loss + loss_b
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 8) Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        self.collected_states.clear()

        return actor_loss.item(), loss_b
