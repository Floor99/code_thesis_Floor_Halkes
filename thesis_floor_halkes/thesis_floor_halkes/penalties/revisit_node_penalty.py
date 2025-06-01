from typing import List
import torch

from thesis_floor_halkes.environment.base import Environment
from thesis_floor_halkes.penalties.base import Penalty, Bonus
from thesis_floor_halkes.utils.haversine import haversine


class RevisitNodePenalty(Penalty):
    """
    Penalty for revisiting a node in the environment.
    """

    def __init__(self, name: str, penalty: float):
        super().__init__(name, penalty)

    def __call__(self, **kwargs) -> float:
        visited_nodes = kwargs.get("visited_nodes", List[int])
        action = kwargs.get("action", int)

        if action in visited_nodes:
            return self.penalty
        return 0.0


class PenaltyPerStep(Penalty):  # augment it with a travel time / distance based cost
    """
    Penalty for each step taken in the environment.
    """

    def __init__(self, name: str, penalty: float):
        super().__init__(name, penalty)

    def __call__(self, **kwargs) -> float:
        return self.penalty


class AggregatedStepPenalty(Penalty):
    """
    Penalty for each step taken in the environment, aggregated over all steps.
    """

    def __init__(self, name: str, penalty: float):
        super().__init__(name, penalty)

    def __call__(self, **kwargs) -> float:
        environment = kwargs.get("environment", Environment)

        step_count = environment.steps_taken

        return step_count * self.penalty


class DeadEndPenalty(Penalty):
    """
    Penalty for reaching a dead end in the environment.
    """

    def __init__(self, name: str, penalty: float):
        super().__init__(name, penalty)

    def __call__(self, **kwargs) -> float:
        valid_actions = kwargs.get("valid_actions", List[int])
        environment = kwargs.get("environment", Environment)
        current_node = kwargs.get("current_node", int)
        end_node = kwargs.get("end_node", int)

        if valid_actions == [] and current_node != end_node:
            return self.penalty
        
        if environment.truncated and current_node != end_node:
            return self.penalty
        
        return 0.0


class WaitTimePenalty(Penalty):
    """
    Penalty for waiting at a traffic light in the environment.
    """

    def __init__(self, name: str, penalty: float = None):
        super().__init__(name, penalty)

    def __call__(self, **kwargs,) -> float:
        action = kwargs.get("action", int)
        environment = kwargs.get("environment", Environment)
        has_light_idx = kwargs.get("has_light_idx")
        status_idx = kwargs.get("status_idx")
        wait_time_idx = kwargs.get("wait_time_idx")
        
        has_light = bool(environment.states[-1].static_data.x[action, has_light_idx])
        status = bool(environment.states[-1].dynamic_data.x[action, status_idx])
        wait_time = float(environment.states[-1].dynamic_data.x[action, wait_time_idx])
        
        if has_light and not status:
            return -wait_time
        return 0.0
    

class NoSignalIntersectionPenalty(Penalty):
    def __init__(self, name: str, penalty: float = None):
        super().__init__(name, penalty)
        
    def __call__(self, **kwargs) -> float:
        has_light_idx = kwargs.get("has_light_idx")
        wait_time_idx = kwargs.get("wait_time_idx")
        environment = kwargs.get("environment", Environment)
        current_node = kwargs.get("current_node", int)
        
        has_light = bool(environment.states[-1].static_data.x[current_node, has_light_idx])
        wait_time = float(environment.states[-1].dynamic_data.x[current_node, wait_time_idx])
        
        if not has_light:
            return -wait_time
        return 0.0


        
        
class GoalBonus(Bonus):
    """
    Bonus for reaching the goal in the environment.
    """

    def __init__(self, name: str, bonus: float):
        super().__init__(name, bonus)

    def __call__(self, **kwargs) -> float:
        current_node = kwargs.get("current_node", int)
        end_node = kwargs.get("end_node", int)

        if current_node == end_node:
            return self.bonus
        return 0.0


class CloserToGoalBonus(Bonus):
    """
    Bonus for every step getting closer to the goal in the environment, based on Euclidean distance.

    Penalty when moving away from the goal.
    """

    def __init__(self, name: str, bonus: float, discount_factor: float = 0.99):
        super().__init__(
            name,
            bonus,
        )
        self.discount_factor = discount_factor

    def __call__(self, **kwargs) -> float:
        environment = kwargs.get("environment", Environment)
        dist_to_goal_idx = kwargs.get("dist_to_goal_idx", int)

        if len(environment.states) >= 2:
            previous_node = environment.states[-2].current_node
        else:
            previous_node = environment.states[-1].current_node
        current_node = environment.states[-1].current_node

        dist_to_goal = environment.states[-1].static_data.x[:, dist_to_goal_idx]
        distance_prev = dist_to_goal[previous_node].item()
        distance_curr = dist_to_goal[current_node].item()

        shaping = distance_prev - self.discount_factor * (distance_curr)
        if shaping < 0:
            shaping = 0.0
        return shaping * self.bonus


class HigherSpeedBonus(Bonus):
    """
    Bonus for roads with a higher speed limit in the environment.
    """

    def __init__(self, name, bonus):
        super().__init__(name, bonus)

    def __call__(self, **kwargs) -> float:
        environment = kwargs.get("environment", Environment)
        action = kwargs.get("action", int)
        speed_idx = kwargs.get("speed_idx", int)

        if environment.states[-1].static_data.edge_attr[:, speed_idx][action].item() > 50.0:
            return self.bonus
        return 0.0
