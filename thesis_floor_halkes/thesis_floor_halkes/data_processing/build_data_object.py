from typing import List
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data

from thesis_floor_halkes.environment.base import Environment
from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetter


node_df = pd.read_parquet("data/processed/node_features_expanded.parquet")
edge_df = pd.read_parquet("data/processed/edge_features_helmond.parquet")
print(node_df.dtypes)


def build_static_data_helmond(node_df, edge_df):
    static_node_df = node_df.drop_duplicates(
        subset="node_id", keep="first"
    ).sort_values("node_id")

    static_vals = static_node_df[["lat", "lon", "has_light"]].to_numpy(
        dtype=np.float32
    )  # already a single ndarray
    static_x = torch.from_numpy(static_vals)

    edge_index = torch.tensor(
        [edge_df["u"].values, edge_df["v"].values], dtype=torch.long
    )

    edge_attr = torch.tensor(
        edge_df[["length", "maxspeed"]].values, dtype=torch.float32
    )

    static_data = Data(x=static_x, edge_index=edge_index, edge_attr=edge_attr)

    return static_data


class DynamicFeatureGetterDataFrame(DynamicFeatureGetter):
    def __init__(
        self,
        node_df,
    ):
        self.df = node_df.copy()
        self.timestamps = sorted(self.df["timestamp"].unique())

    def get_dynamic_features(
        self,
        environment: Environment,
        traffic_light_idx: int,
        current_node: int,
        visited_nodes: List[int],
        time_step: int,
    ) -> Data:
        t = self.timestamps[time_step]
        df_t = self.df[self.df["timestamp"] == t].sort_values("node_id")

        wait_times = torch.tensor(df_t["wait_time"].values, dtype=torch.float)
        num_nodes = environment.static_data.num_nodes

        has_light = environment.static_data.x[:, traffic_light_idx].bool()
        rand_bits = torch.randint(0, 2, (num_nodes,), dtype=torch.bool)
        light_status = (rand_bits & has_light).to(torch.float)

        is_current_node = torch.zeros(num_nodes)
        is_current_node[current_node] = 1.0

        is_visited = torch.zeros(num_nodes)
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
