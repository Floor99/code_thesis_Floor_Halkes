from abc import ABC, abstractmethod
from typing import List

from thesis_floor_halkes.action.base import Action
from thesis_floor_halkes.environment.base import Environment
# from thesis_floor_halkes.environment.dynamic_ambulance import AmbulanceEnvDynamic


class RewardModifier(ABC):
    """
    Abstract base class for reward modifiers.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, **kwargs) -> float:
        pass


class Penalty(RewardModifier):
    """
    Abstract base class for penalties.
    """

    def __init__(self, name: str, penalty: float):
        self.name = name
        self.penalty = penalty

    @abstractmethod
    def __call__(self, **kwargs) -> float:
        """
        Call the penalty function.
        Compute the penalty using a flexible set of keyword arguments.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        pass


class Bonus(RewardModifier):
    """
    Abstract base class for bonuses.
    """

    def __init__(self, name: str, bonus: float):
        self.name = name
        self.bonus = bonus

    @abstractmethod
    def __call__(self, **kwargs) -> float:
        """
        Call the bonus function.
        Compute the bonus using a flexible set of keyword arguments.

        Args:
            environment: The environment to calculate the bonus for.
            action: The action taken in the environment.

        Returns:
            The calculated bonus.
        """
        pass
