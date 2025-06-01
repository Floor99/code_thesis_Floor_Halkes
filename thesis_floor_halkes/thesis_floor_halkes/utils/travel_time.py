from torch_geometric.data import Data


def calculate_edge_travel_time(
    data: Data, edge_index: int, length_feature_idx: int, speed_feature_idx: int
):
    length = data.edge_attr[edge_index, length_feature_idx]
    speeds_kmh = data.edge_attr[edge_index, speed_feature_idx]
    speed_ms = speeds_kmh / 3.6
    edge_travel_time = length / speed_ms
    return edge_travel_time
