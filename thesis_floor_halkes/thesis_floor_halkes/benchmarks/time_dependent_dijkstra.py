import time
from torch_geometric.data import Data
import pandas as pd
import heapq
import torch

from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetterDataFrame
from thesis_floor_halkes.features.graph.graph_generator import create_osmnx_sub_graph_only_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph, plot_sub_graph_in_and_out_nodes_helmond
from thesis_floor_halkes.features.static.new_getter import get_static_data_object_subgraph
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.utils.haversine import haversine



def time_dependent_dijkstra(
    static_data, 
    dynamic_feature_getter, 
    dynamic_node_idx,
    static_node_idx,
    static_edge_idx,
):
    df = static_data.timeseries.copy()
    start_node = static_data.start_node
    
    df_start = df[df['node_id'] == start_node]
    time_stamps = sorted(df_start["timestamp"].unique())
    ts0 = pd.to_datetime(time_stamps[0])  # ensures it's a pandas.Timestamp
    t0 = time_stamps.index(ts0) # index of the first timestamp
    
    T = len(time_stamps)
    
    adj = build_adjecency_matrix(static_data.num_nodes, static_data)
    start = static_data.start_node
    end = static_data.end_node
    
    class _TmpEnv:
        pass 
    tmp = _TmpEnv()
    tmp.static_data = static_data
    tmp.static_node_idx = static_node_idx
    
    heap = [(0.0, start, t0, [start])]
    best = {(start, t0): 0.0}
    
    while heap: 
        cost, node, t_idx, path = heapq.heappop(heap)
        # print(f"[POP] cost={cost:.3f}, node={node}, t_idx={t_idx}, path={path}")
        if cost > best.get((node, t_idx), float("inf")):
            # print("       → skipping stale entry")
            continue
        if node == end:
            # print(f"[FOUND] cost={cost:.3f}, path={path}")
            return cost, path
        next_t = t_idx + 1
        if next_t >= T:
            # print(f"       → next_t={next_t} ≥ T={T}, no further expansion")
            continue
        # print(f"       → expanding at time index {next_t} (ts={time_stamps[next_t]})")
    
        dyn = dynamic_feature_getter.get_dynamic_features(
            environment = tmp, 
            traffic_light_idx = static_node_idx["has_light"],
            current_node = node, 
            visited_nodes = path,
            time_step = next_t, 
            sub_node_df = df,
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
            
            ts = time_stamps[next_t]
            orig_wait_series = df[
                (df["node_id"] == nbr) & (df["timestamp"] == ts)
            ]
            # print(f"  Original wait series:\n {orig_wait_series}\n")
            # print(f"node = {nbr} @ ts={ts}"
            #       f"> dyn.get() = {wait:.2f}s ")
            
            
            new_cost = cost + travel_time + wait
            
            # print(
            #         f"         edge {node}→{nbr}: travel={travel_time:.2f}, "
            #         f"wait={wait:.2f} → new_cost={new_cost:.2f}"
            #     )
            
            key = (nbr, next_t)
            if new_cost < best.get(key, float("inf")):
                best[key] = new_cost
                heapq.heappush(heap, (new_cost, nbr, next_t, path + [nbr]))
                # print(f"           • pushed (cost={new_cost:.2f}, node={nbr}, t={next_t})")
        
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
    cost, route = time_dependent_dijkstra(static_data, 
                                          dynamic_feature_getter, 
                                          dynamic_node_idx, 
                                          static_node_idx, 
                                          static_edge_idx,
                                          "2024-01-31 08:30:00")
    end = time.time()
    print(f"TD Dijkstra cost: {cost}, route: {route}")
    print(f"Time taken: {end - start:.2f} seconds")
    
    class _TmpEnv:
        pass 
    tmp = _TmpEnv()
    tmp.static_data = static_data
    tmp.static_node_idx = static_node_idx
    
    
        # Print travel and wait times per edge in the route
    if route is not None and len(route) > 1:
        df = static_data.timeseries.copy()
        time_stamps = sorted(df[df['node_id'] == route[0]]["timestamp"].unique())
        ts0 = pd.to_datetime("2024-01-31 08:30:00")
        t_idx = time_stamps.index(ts0)
        print("\nRoute breakdown:")
        for i in range(len(route) - 1):
            node = route[i]
            nbr = route[i + 1]
            next_t = t_idx + 1
            dyn = dynamic_feature_getter.get_dynamic_features(
                environment=tmp,  # If needed, pass the correct environment
                traffic_light_idx=static_node_idx["has_light"],
                current_node=node,
                visited_nodes=route[:i+1],
                time_step=next_t,
                sub_node_df=df,
            )
            wait_times = dyn.x[:, dynamic_node_idx["wait_time"]]
            # Find edge index
            eidx = None
            for n, e in build_adjecency_matrix(static_data.num_nodes, static_data)[node]:
                if n == nbr:
                    eidx = e
                    break
            if eidx is None:
                print(f"Edge {node}->{nbr} not found!")
                continue
            length = static_data.edge_attr[eidx, static_edge_idx["length"]]
            speed = static_data.edge_attr[eidx, static_edge_idx["speed"]]
            travel_time = length / (speed / 3.6)
            wait = wait_times[nbr].item()
            print(f"{node} → {nbr}: length={length:.2f}m, speed={speed:.2f}km/h, travel_time={travel_time:.2f}s, wait_time={wait:.2f}s")
            t_idx = next_t
    
    
