from thesis_floor_halkes.environment.base import Environment
from thesis_floor_halkes.penalties.base import Penalty, RewardModifier


class PenaltyCalculator:
    def __init__(self, penalty_types: list[Penalty], penalty_weights: list[float]):
        """
        Initialize the penalty calculator with a list of penalties and their corresponding weights.

        Args:
            penalty_types (list[Penalty]): List of penalty types to be applied.
            penalty_weights (list[float]): List of weights for each penalty type.
        """
        self.penalty_types = penalty_types
        self.penalty_weights = penalty_weights
        assert len(penalty_types) == len(penalty_weights), (
            "Penalty types and weights must have the same length."
        )

    def calculate_penalty(self, **kwargs) -> float:
        """
        Calculate the total penalty for the given environment.

        Args:
            environment (Environment): The environment for which to calculate the penalty.

        Returns:
            float: The total penalty.
        """
        total_penalty = 0.0
        for penalty_type, weight in zip(self.penalty_types, self.penalty_weights):
            total_penalty += penalty_type(**kwargs) * weight
        return total_penalty


class RewardModifierCalculator:
    def __init__(self, modifiers: list[RewardModifier], weights: list[float]):
        """
        Initialize the penalty calculator with a list of penalties and their corresponding weights.

        Args:
            penalty_types (list[Penalty]): List of penalty types to be applied.
            penalty_weights (list[float]): List of weights for each penalty type.
        """
        self.modifiers = modifiers
        self.weights = weights
        assert len(modifiers) == len(weights), (
            "Penalty types and weights must have the same length."
        )

    def calculate_total(self, **kwargs) -> float:
        """
        Calculate the total reward modification for the given environment.

        Args:
            environment (Environment): The environment for which to calculate the penalty.

        Returns:
            float: The total penalty.
        """
        total_modification = 0.0
        for modifier, weight in zip(self.modifiers, self.weights):
            total_modification += modifier(**kwargs) * weight

        return total_modification

    def store_modifier_per_step(self, **kwargs) -> list[dict]:
        self.modifier_contributions = {}

        for modifier, weight in zip(self.modifiers, self.weights):
            contribution = modifier(**kwargs) * weight
            self.modifier_contributions.update({modifier.name: contribution})
        return self.modifier_contributions

    def reset(self):
        """
        Reset the stored modifiers.
        """
        self.modifier_contributions.clear()
