import heapq
import time
import pandas as pd
import torch
from torch_geometric.data import Data
import os
from glob import glob
import numpy as np

from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetterDataFrame
from thesis_floor_halkes.features.graph.graph_generator import create_osmnx_sub_graph_only_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph, plot_sub_graph_in_and_out_nodes_helmond
from thesis_floor_halkes.features.static.new_getter import get_static_data_object_subgraph
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.utils.haversine import haversine

from thesis_floor_halkes.benchmarks.time_dependent_A_star import time_dependent_a_star
from thesis_floor_halkes.benchmarks.time_dependent_dijkstra import time_dependent_dijkstra


def run_on_dataset(split_name, base_dir, routing_fn):
    """
    Run a given routing function (e.g. time_dependent_dijkstra or a_star) 
    on all .pt graph files in the specified split directory.
    
    Parameters:
        split_name (str): "train", "val", or "test"
        base_dir (str): base directory for the split (e.g. "training_data")
        routing_fn (function): the routing function to apply to each graph
    """
    print(f"\nRunning {routing_fn.__name__} on {split_name.upper()} set")
    results = []

    network_dirs = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("network")
    ])

    for network_dir in network_dirs:
        pt_files = glob(os.path.join(network_dir, "*.pt"))
        print(f"Found {len(pt_files)} graphs in {network_dir}")

        for pt_file in pt_files:
            try:
                static_data = torch.load(pt_file, weights_only=False)
                dynamic_feature_getter = DynamicFeatureGetterDataFrame()

                start = time.time()
                cost, route = routing_fn(
                    static_data,
                    dynamic_feature_getter,
                    dynamic_node_idx,
                    static_node_idx,
                    static_edge_idx,
                )
                end = time.time()

                results.append({
                    "file": pt_file,
                    "network": os.path.basename(network_dir),
                    "success": cost is not None,
                    "cost": cost.item() if cost is not None else float("inf"),
                    "route": route,
                    "route_len": len(route),
                    "runtime_s": end - start,
                    "split": split_name,
                    "algorithm": routing_fn.__name__
                })
            except Exception as e:
                print(f"Error in file {pt_file}: {e}")
                results.append({
                    "file": pt_file,
                    "network": os.path.basename(network_dir),
                    "success": False,
                    "cost": float("inf"),
                    "route_len": 0,
                    "runtime_s": 0.0,
                    "split": split_name,
                    "algorithm": routing_fn.__name__
                })

    return pd.DataFrame(results)


def run_all_evaluations(base_dirs, output_root, algorithms):
    """
    Runs each algorithm on each dataset split and saves the results.
    
    Parameters:
        base_dirs (dict): {split_name: path_to_split}
                          e.g., {"train": "training_data", "val": "val_data", "test": "test_data"}
        output_root (str): Base output directory (results will go into subfolders here)
        algorithms (dict): {name: function} — e.g., {"dijkstra": time_dependent_dijkstra, "astar": time_dependent_astar}
    """
    os.makedirs(output_root, exist_ok=True)

    for algo_name, algo_fn in algorithms.items():
        algo_output_dir = os.path.join(output_root, f"results_{algo_name}")
        os.makedirs(algo_output_dir, exist_ok=True)

        print(f"\n=== Running algorithm: {algo_name.upper()} ===")
        for split_name, split_dir in base_dirs.items():
            print(f"\n--- Split: {split_name.upper()} ---")
            df = run_on_dataset(split_name, split_dir, routing_fn=algo_fn)

            output_path = os.path.join(algo_output_dir, f"{split_name}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {split_name} results to {output_path}")
            

def calculate_success_travel_time_score(df, alpha, beta, min_time=0, max_time=1000):
    """
    Calculate a normalized score from individual route success and travel times.
    
    - `alpha`: weight for success rate
    - `beta`: weight for normalized penalized travel time
    - `min_time`, `max_time`: used to normalize travel times
    """
    # Extract per-sample success (1.0/0.0) and travel time
    success_rates = df['success'].astype(float).tolist()
    travel_times = df['cost'].tolist()

    # Min-max normalize travel times
    penalized_travel_times = np.array(travel_times)
    penalized_travel_times = (penalized_travel_times - min_time) / (max_time - min_time)
    penalized_travel_times = np.clip(penalized_travel_times, 0, 1)

    # Compute final score
    score = (
        alpha * np.mean(success_rates)
        - beta * np.mean(penalized_travel_times)
    )
    
    return score.item()


def evaluate_all_scores(result_base_dir, alpha=1.0, beta=0.5, min_time=0, max_time=1000):
    """
    Evaluate scores for Dijkstra and A* on train/val/test splits.
    
    Parameters:
        result_base_dir (str): Directory containing results_dijkstra and results_astar
        alpha, beta (float): scoring coefficients
        min_time, max_time (float): travel time normalization bounds
    """
    algorithms = ["dijkstra", "astar"]
    splits = ["train", "val", "test"]
    
    summary = []

    for algo in algorithms:
        algo_path = os.path.join(result_base_dir, f"results_{algo}")
        for split in splits:
            csv_path = os.path.join(algo_path, f"{split}.csv")
            if not os.path.exists(csv_path):
                print(f"Missing file: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            score = calculate_success_travel_time_score(df, alpha, beta, min_time, max_time)

            summary.append({
                "algorithm": algo,
                "split": split,
                "score": score,
                "success_rate": df["success"].mean(),
                "avg_travel_time": df[df["success"]]["cost"].mean() if df["success"].any() else float("inf")
            })
    
        score_df = pd.DataFrame(summary)

        # Save score summary
        output_path = os.path.join(result_base_dir, "score_summary.csv")
        score_df.to_csv(output_path, index=False)
        print(f"Score summary saved to {output_path}")
    return score_df


if __name__ == "__main__":
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
    
    base_dirs = {
        "train": "data/training_data_2/not_smart",
        "val": "data/validation_data_2/not_smart",
        "test": "data/test_data_2/not_smart"
    }

    algorithms = {
        "dijkstra": time_dependent_dijkstra,
        "astar": time_dependent_a_star
    }

    run_all_evaluations(base_dirs, output_root="data/evaluation_results_benchmarks", algorithms=algorithms)
    
    score_df = evaluate_all_scores("data/evaluation_results_benchmarks", alpha=0.8, beta=0.2, min_time=0, max_time=600)
    print(score_df)
