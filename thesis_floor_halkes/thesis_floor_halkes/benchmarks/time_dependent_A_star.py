import heapq
import time
import pandas as pd

from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetterDataFrame
from thesis_floor_halkes.features.static.new_getter import get_static_data_object_subgraph
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix


def time_dependent_a_star(
    static_data,
    dynamic_feature_getter,
    dynamic_node_idx,
    static_node_idx,
    static_edge_idx,
):
    df = static_data.timeseries.copy()
    start_node = static_data.start_node
    end_node = static_data.end_node
    # print(f"Start node: {start_node}")
    # print(f"End node: {end_node}")

    df_start = df[df['node_id'] == start_node]
    time_stamps = sorted(df_start["timestamp"].unique())
    ts0 = pd.to_datetime(time_stamps[0])  # ensures it's a pandas.Timestamp
    t0 = time_stamps.index(ts0) # index of the first timestamp
    
    T = len(time_stamps)

    adj = build_adjecency_matrix(static_data.num_nodes, static_data)

    # Heuristic: straight-line distance to goal / max speed (in seconds)
    max_speed = static_data.edge_attr[:, static_edge_idx['speed']].max().item()
    dist_to_goal = static_data.x[:, static_node_idx['dist_to_goal']]  # meters

    def heuristic(node):
        # max_speed in km/h, convert to m/s
        return dist_to_goal[node].item() / (max_speed / 3.6 + 1e-6)

    class _TmpEnv:
        pass
    tmp = _TmpEnv()
    tmp.static_data = static_data
    tmp.static_node_idx = static_node_idx

    heap = [(heuristic(start_node), 0.0, start_node, t0, [start_node])]
    best = {(start_node, t0): 0.0}

    while heap:
        est_total, cost, node, t_idx, path = heapq.heappop(heap)
        if cost > best.get((node, t_idx), float("inf")):
            continue
        if node == end_node:
            # print(f"[FOUND] cost={cost:.3f}, path={path}")
            return cost, path
        next_t = t_idx + 1
        if next_t >= T:
            continue

        dyn = dynamic_feature_getter.get_dynamic_features(
            environment=tmp,
            traffic_light_idx=static_node_idx["has_light"],
            current_node=node,
            visited_nodes=path,
            time_step=next_t,
            sub_node_df=df,
        )
        wait_times = dyn.x[:, dynamic_node_idx["wait_time"]]

        for nbr, eidx in adj[node]:
            length = static_data.edge_attr[eidx, static_edge_idx["length"]]
            speed = static_data.edge_attr[eidx, static_edge_idx["speed"]]
            travel_time = length / (speed / 3.6)
            
            light_status = dyn.x[nbr, dynamic_node_idx["status"]].item()
            if light_status == 1:  # green light
                wait = 0.0
            else:  # red light
                wait = wait_times[nbr].item()
            
            new_cost = cost + travel_time + wait
            key = (nbr, next_t)
            
            if new_cost < best.get(key, float("inf")):
                best[key] = new_cost
                est = new_cost + heuristic(nbr)
                heapq.heappush(heap, (est, new_cost, nbr, next_t, path + [nbr]))

    return None, float("inf")  # No path found



if __name__ == "__main__":
    static_data = get_static_data_object_subgraph(
        timeseries_subgraph_path="data/training_data/subgraph_0/timeseries.parquet",
        edge_features_path="data/training_data/subgraph_0/edge_features.parquet",
        G_cons_path="data/training_data/subgraph_0/G_cons.graphml",
        G_pt_cons_path="data/training_data/subgraph_0/G_pt_cons.graphml",
    )
    
    dynamic_node_idx = {
        "status": 0,
        "wait_time": 1,
        "current_node": 2,
        "visited_nodes": 3,
    }

    static_node_idx = {
        "lat": 0,
        "lon": 1,
        "has_light": 2,
        "dist_to_goal": 3,
    }

    static_edge_idx = {
        "length": 0,
        "speed": 1,
    }
    
    dynamic_feature_getter = DynamicFeatureGetterDataFrame()
    
    start = time.time()
    cost, route = time_dependent_a_star(static_data,
                          dynamic_feature_getter,
                          dynamic_node_idx, 
                          static_node_idx, 
                          static_edge_idx,
                          "2024-01-31 08:30:00")
    
    end = time.time()
    print(f"TD A* cost: {cost}, route: {route}")
    print(f"Time taken: {end - start:.2f} seconds")