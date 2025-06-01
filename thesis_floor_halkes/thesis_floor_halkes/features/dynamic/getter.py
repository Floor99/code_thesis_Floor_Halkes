from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import torch
from torch_geometric.data import Data

from thesis_floor_halkes.environment.base import Environment


class DynamicFeatureGetter(ABC):
    """
    Abstract base class for dynamic feature getters.
    """

    @abstractmethod
    def get_dynamic_features(
        self, environment: Environment, traffic_light_idx: int, max_wait: float = 10.0
    ) -> Data:
        """
        Abstract method to get dynamic features.
        """
        pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # For testing purposes, use CPU

class DynamicFeatureGetterDataFrame(DynamicFeatureGetter):
    def __init__(
        self,
    ):
        pass

    def get_dynamic_features(
        self,
        environment: Environment,
        traffic_light_idx: int,
        current_node: int,
        visited_nodes: List[int],
        time_step: int,
        sub_node_df: pd.DataFrame,
    ) -> Data:
        self.timestamps = sorted(sub_node_df["timestamp"].unique())
        t = self.timestamps[time_step]
        df_t = sub_node_df[sub_node_df["timestamp"] == t].sort_values("node_id")

        wait_times = torch.tensor(df_t["wait_time"].values, dtype=torch.float, device=device)
        num_nodes = environment.static_data.num_nodes

        has_light = environment.static_data.x[:, traffic_light_idx].bool()
        rand_bits = torch.randint(0, 2, (num_nodes,), dtype=torch.bool, device=device)  # get random bits
        light_status = (rand_bits & has_light).to(torch.float)
        
        # set light status to red
        # light_status = torch.zeros(num_nodes, device=device)

        is_current_node = torch.zeros(num_nodes, device=device)
        is_current_node[current_node] = 1.0

        is_visited = torch.zeros(num_nodes, device=device)
        is_visited[visited_nodes] = 1.0

        x = torch.stack(
            [
                light_status,
                wait_times,
                is_current_node,
                is_visited,
            ],
            dim=1,
        )

        return Data(
            x=x,
            edge_index=environment.static_data.edge_index,
        )
