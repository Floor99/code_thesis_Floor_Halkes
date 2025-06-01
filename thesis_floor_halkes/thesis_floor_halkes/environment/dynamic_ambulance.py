import pandas as pd
import torch
from networkx import MultiDiGraph
from torch_geometric.data import Data, Dataset

from thesis_floor_halkes.environment.base import Environment
from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetterDataFrame
from thesis_floor_halkes.penalties.calculator import (
    RewardModifierCalculator,
)
from thesis_floor_halkes.state import State
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.utils.travel_time import calculate_edge_travel_time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # For testing purposes, use CPU


class DynamicEnvironment(Environment):
    def __init__(
        self,
        static_dataset: list[Data] | Dataset,
        dynamic_feature_getter: DynamicFeatureGetterDataFrame,
        reward_modifier_calculator: RewardModifierCalculator,
        max_steps: int = 30,
        start_timestamp: str | pd.Timestamp = None,
        dynamic_node_idx: dict = None,
        static_node_idx: dict = None,
        static_edge_idx: dict = None,
        action_mask_funcs: list[callable] = None,
    ):
        self.static_dataset = static_dataset
        self.dynamic_feature_getter = dynamic_feature_getter
        self.reward_modifier_calculator = reward_modifier_calculator
        self.max_steps = max_steps
        self.start_timestamp = start_timestamp
        self.dynamic_node_idx = dynamic_node_idx
        self.static_node_idx = static_node_idx
        self.static_edge_idx = static_edge_idx
        self.action_mask_funcs = action_mask_funcs or []

    def reset(self):
        self.steps_taken = 0
        self.terminated = False
        self.truncated = False
        self.time_stamps = sorted(self.static_data.timeseries["timestamp"].unique())
        self.adjecency_matrix = build_adjecency_matrix(
            self.static_data.num_nodes, self.static_data
        )
        self.current_time_idx = 0
        self.start_timestamp = self.time_stamps[0]

        self.states = []
        init_state = self._get_state()
        self.states.append(init_state)

        self.step_travel_time_route = []
        self.step_modifier_contributions = []

        return init_state

    def _get_state(self, action=None):
        """
        Get the current state of the environment.
        """

        if action is not None:
            current_node = action
            previous_visited_nodes = self.states[-1].visited_nodes
            visited_nodes = self.update_visited_nodes(
                previous_visited_nodes, current_node
            )

        else:
            current_node = self.static_data.start_node
            visited_nodes = [self.static_data.start_node]

        sub_node_df = self.static_data.timeseries
        # resample dynamic features
        dynamic_features = self.dynamic_feature_getter.get_dynamic_features(
            environment=self,
            traffic_light_idx=self.static_node_idx["has_light"],
            current_node=current_node,
            visited_nodes=visited_nodes,
            time_step=self.current_time_idx,
            sub_node_df=sub_node_df,
        )

        # get static features
        static_features = self.static_data

        # get valid actions
        valid_actions = self.get_valid_actions(
            self.adjecency_matrix, current_node, visited_nodes, self.static_data.G_sub
        )

        state = State(
            static_data=static_features,
            dynamic_data=dynamic_features,
            start_node=self.static_data.start_node,
            end_node=self.static_data.end_node,
            num_nodes=self.static_data.num_nodes,
            current_node=current_node,
            visited_nodes=visited_nodes,
            valid_actions=valid_actions,
        )

        return state

    def step(self, action):
        """
        Take a step in the environment using the given action.
        """
        old_state = self.states[-1]
        if self.steps_taken >= self.max_steps:
            self.truncated = True
            return old_state, reward, self.terminated, self.truncated, {}

        self.steps_taken += 1
        self.current_time_idx += 1

        new_state = self._get_state(action)
        self.states.append(new_state)

        # Check if action is valid
        if action not in old_state.valid_actions:
            raise ValueError(
                f"Invalid action {action} from node {new_state.current_node}."
            )
        # Compute the travel time
        edge_idx = next(
            idx
            for (v, idx) in self.adjecency_matrix[old_state.current_node]
            if v == action
        )
        travel_time_edge = calculate_edge_travel_time(
            self.static_data,
            edge_index=edge_idx,
            length_feature_idx=0,
            speed_feature_idx=1,
        )

        self.step_travel_time_route.append(travel_time_edge)

        # Compute the reward
        penalty = self.reward_modifier_calculator.calculate_total(
            visited_nodes=old_state.visited_nodes,
            action=action,
            current_node=new_state.current_node,
            end_node=new_state.end_node,
            valid_actions=new_state.valid_actions,
            environment=self,
            status_idx=self.dynamic_node_idx["status"],
            wait_time_idx=self.dynamic_node_idx["wait_time"],
            has_light_idx=self.static_node_idx["has_light"],
            dist_to_goal_idx=self.static_node_idx["dist_to_goal"],
            speed_idx=self.static_edge_idx["speed"],
        )

        self.modifier_contributions = (
            self.reward_modifier_calculator.store_modifier_per_step(
                visited_nodes=old_state.visited_nodes,
                action=action,
                current_node=new_state.current_node,
                end_node=new_state.end_node,
                valid_actions=new_state.valid_actions,
                environment=self,
                status_idx=self.dynamic_node_idx["status"],
                wait_time_idx=self.dynamic_node_idx["wait_time"],
                has_light_idx=self.static_node_idx["has_light"],
                dist_to_goal_idx=self.static_node_idx["dist_to_goal"],
                speed_idx=self.static_edge_idx["speed"],
            )
        )

        self.modifier_contributions.update({"step": self.steps_taken})
        self.step_modifier_contributions.append(self.modifier_contributions)

        reward = -travel_time_edge + penalty

        if new_state.valid_actions == []:
            self.truncated = True

        if new_state.current_node == new_state.end_node:
            self.terminated = True

        if self.steps_taken >= self.max_steps:
            self.truncated = True

        return new_state, reward, self.terminated, self.truncated, {}

    def get_valid_actions(
        self,
        adj_matrix: dict[int, list[tuple[int, int]]],
        current_node: int,
        visited_nodes: set[int],
        graph: MultiDiGraph,
    ) -> list[int]:
        """
        Return valid actions (neighbors) based on the adjacency matrix,
        masking out dead-ends (nodes whose only neighbor is `current_node`)
        except when that node is the goal.
        """
        goal = self.static_data.end_node

        current_node_neighbors = set(v for v, _ in adj_matrix[current_node])

        # remove neighbors that are already visited
        visited_removed = current_node_neighbors - set(visited_nodes)

        nodes_to_remove = set()
        params = {
            "start_node": current_node,
            "target_node": goal,
            "G": graph,
        }

        for func in self.action_mask_funcs or []:
            if callable(func):
                _nodes_to_remove = func(**params)
                nodes_to_remove.update(_nodes_to_remove)

        # remove nodes that are not valid according to the action mask functions
        intermediate_nodes_to_be_removed = visited_removed - nodes_to_remove

        if goal in current_node_neighbors:
            # if the goal is in the neighbors, we keep it
            goal_added = intermediate_nodes_to_be_removed | {goal}
        else:
            # otherwise, we remove the goal from the neighbors
            goal_added = intermediate_nodes_to_be_removed

        valid = list(goal_added)
        return valid

    def update_visited_nodes(self, prev_visited_nodes: list[int], action):
        return prev_visited_nodes + [action]
