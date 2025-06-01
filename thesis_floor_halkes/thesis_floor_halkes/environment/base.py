from abc import ABC, abstractmethod

from thesis_floor_halkes.action.base import Action
from thesis_floor_halkes.state import State


class Environment(ABC):
    """
    Abstract base class for environments.
    """

    @abstractmethod
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        pass

    @abstractmethod
    def _get_state(self) -> State:
        """
        Get the current state of the environment.

        Returns:
            The current state.
        """
        pass

    @abstractmethod
    def step(self, action: Action):
        """
        Take a step in the environment using the given action.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the next state, reward, done flag, and additional info.
        """
        pass

    @abstractmethod
    def get_valid_actions(
        self, adj_matrix: dict[int, list[tuple[int, int]]]
    ) -> list[int]:
        """
        Get the valid actions based on the adjacency matrix.
        """
        pass
