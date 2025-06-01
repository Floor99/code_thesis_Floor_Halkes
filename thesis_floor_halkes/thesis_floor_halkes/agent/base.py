from abc import ABC, abstractmethod

from thesis_floor_halkes.state import State


class Agent(ABC):
    """
    Abstract base class for agents.
    """

    @abstractmethod
    def select_action(self, state: State):
        """
        Select an action based on the current state.

        Args:
            state (State): The current state of the environment.

        Returns:
            Action: The selected action.
        """
        pass

    @abstractmethod
    def store_reward(self, reward: float):
        """
        Store the reward received from the environment.

        Args:
            reward (float): The reward received.
        """
        pass

    @abstractmethod
    def store_penalty(self, penalty: float):
        """
        Store the penalty received from the environment.

        Args:
            penalty (float): The penalty received.
        """
        pass

    @abstractmethod
    def store_state(self, state: State):
        """
        Store the current state.

        Args:
            state (State): The current state of the environment.
        """
        pass

    @abstractmethod
    def store_action(self, action):
        """
        Store the action taken.

        Args:
            action (Action): The action taken.
        """
        pass

    @abstractmethod
    def store_action_log_prob(self, log_prob):
        """
        Store the log probability of the action taken.

        Args:
            log_prob (float): The log probability of the action taken.
        """
        pass

    @abstractmethod
    def finish_episode(self):
        """
        Finish the episode and update the agent's policy.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the agent's internal state.
        """
        pass

    @abstractmethod
    def backprop_model(self):
        """
        Update the agent's policy based on the stored rewards and actions.
        """
        pass
