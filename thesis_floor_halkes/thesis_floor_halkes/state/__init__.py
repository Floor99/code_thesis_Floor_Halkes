from dataclasses import dataclass
from torch_geometric.data import Data


@dataclass
class State:
    static_data: Data
    dynamic_data: Data
    start_node: int
    end_node: int
    num_nodes: int
    current_node: int
    visited_nodes: list[int] = None
    valid_actions: list[int] = None
