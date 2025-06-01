from abc import ABC, abstractmethod


class Baseline(ABC):
    """
    Abstract base class for baselines.
    """

    @abstractmethod
    def eval(self, embeddings, returns):
        """
        Evaluate the baseline.

        Args:
            embeddings: list of state objects (with .graph_embedding)
            returns: torch.Tensor of shape [T]

        Returns:
            baselines: torch.Tensor of shape [T]
            loss_b: scalar MSE loss
        """
        pass
