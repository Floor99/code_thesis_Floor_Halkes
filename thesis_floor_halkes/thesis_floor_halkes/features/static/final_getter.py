import json
import os
from pathlib import Path
from typing import List
import concurrent.futures
import hydra
from omegaconf import DictConfig
import pandas as pd
import osmnx as ox
import torch
from torch_geometric.data import Data
import random
from thesis_floor_halkes.data_processing.create_subgraph import consilidate_subgraph, consolidate_edges_to_single, create_subgraph_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph
from thesis_floor_halkes.data_processing.expand_to_min_subgraph import expand_wait_times
from thesis_floor_halkes.data_processing.merge_sub_timeseries import merge_timeseries_pipeline
from thesis_floor_halkes.utils.haversine import haversine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_random_locations_with_traffic_lights(
        nodes_traffic_lights_path: str, 
        seed=42, 
        num_samples=16
    ):
    traffic_lights = pd.read_parquet(nodes_traffic_lights_path)
    traffic_lights_coords = traffic_lights[['lat', 'lon']].drop_duplicates()

    random_locations = traffic_lights_coords.sample(num_samples, random_state=seed)

    lijst_locations = [(row['lat'].item(), row['lon'].item()) for _, row in random_locations.iterrows()]

    return lijst_locations

def get_random_timestamp(timeseries_path: str, 
                         seed=42):
    df = pd.read_parquet(timeseries_path)
    random_timestamp = df.sample(1, random_state=seed).index[0]
    return random_timestamp


def create_sub_network(nodes_helmond_path,
                  lat,
                  lon,
                  node_features_path,
                  edge_features_path,
                  G_cons_path,
                  G_pt_cons_path,
                  dist,
                  ):
    
    G_sub, G_pt = create_subgraph_inside_helmond(nodes_helmond_path, lat, lon, dist=dist)
    G_cons = consilidate_subgraph(G_sub)
    G_pt_cons = consilidate_subgraph(G_pt)
    
    G_cons = consolidate_edges_to_single(G_cons, weight_attr="length", agg="min")
    
    node_features = get_node_features_subgraph(G_cons)
    edge_features = get_edge_features_subgraph(G_cons)
    
    ox.io.save_graphml(G_cons, G_cons_path)
    ox.io.save_graphml(G_pt_cons, G_pt_cons_path)
    
    node_features.to_parquet(node_features_path)
    edge_features.to_parquet(edge_features_path)
    
    return G_cons, G_pt_cons

def get_timeseries_sub_network( node_features_path, 
                                meta_path, 
                                measurement_path,
                                output_path_timeseries,
                                threshold,
                                ):
    
    timeseries_subgraph = merge_timeseries_pipeline(meta_path, 
                                   node_features_path,
                                   measurement_path,
                                   threshold)
    
    expanded_timeseries = expand_wait_times(timeseries_subgraph, num_peaks=2,
                                            amp_frac=0.1, sigma=1.0)
    
    expanded_timeseries.to_parquet(output_path_timeseries, index=False)
    
    return expanded_timeseries


def get_start_end_timestamps(
    expanded_timeseries,
    max_steps: int,
    start_timestamp: pd.Timestamp = None,
    seed: int = None
    
):
    time_stamps = sorted(expanded_timeseries['timestamp'].unique())
    max_start = max(0, len(time_stamps) - max_steps)
    if start_timestamp is None:
        # randomly select a start timestamp
        if seed is not None:
            random.seed(seed)
        idx = random.randrange(0, max_start)
        start_timestamp = time_stamps[idx]
    else:
        if not isinstance(start_timestamp, pd.Timestamp):
            start_timestamp = pd.to_datetime(start_timestamp)
        if start_timestamp not in time_stamps:
            raise ValueError(f"start_timestamp {start_timestamp} not in dynamic data")
    end_timestamp = time_stamps[min(idx + max_steps, len(time_stamps) - 1)]
    
    return start_timestamp, end_timestamp


def get_timeseries_from_start_timestamp(
    expanded_timeseries,
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp
):
    
    """
    Get timeseries data from a given start timestamp to an end timestamp.
    """
    # Ensure the timestamps are in datetime format
    if not isinstance(start_timestamp, pd.Timestamp):
        start_timestamp = pd.to_datetime(start_timestamp)
    if not isinstance(end_timestamp, pd.Timestamp):
        end_timestamp = pd.to_datetime(end_timestamp)

    # Filter the timeseries data
    filtered_timeseries = expanded_timeseries[
        (expanded_timeseries['timestamp'] >= start_timestamp) &
        (expanded_timeseries['timestamp'] <= end_timestamp)
    ]

    return filtered_timeseries


def process_coord(coord, successfully_created, base, nodes_helmond_path, dist, meta_path, measurement_path, threshold):
    subdir = base / f"network_{successfully_created}"
    subdir.mkdir(parents=True, exist_ok=True)

    node_features_path = subdir / "node_metadata.parquet"
    edge_features_path = subdir / "edge_features.parquet"
    output_path_timeseries = subdir / "timeseries.parquet"
    G_cons_path = subdir / "G_cons.graphml"
    G_pt_cons_path = subdir / "G_pt_cons.graphml"

    # Build subgraph
    G_cons, _ = create_sub_network(
        nodes_helmond_path=nodes_helmond_path,
        node_features_path=str(node_features_path),
        edge_features_path=str(edge_features_path),
        lat=coord[0],
        lon=coord[1],
        G_cons_path=G_cons_path,
        G_pt_cons_path=G_pt_cons_path,
        dist=dist,
    )

    # Generate timeseries
    ts = get_timeseries_sub_network(
        node_features_path=str(node_features_path),
        meta_path=meta_path,
        measurement_path=measurement_path,
        output_path_timeseries=str(output_path_timeseries),
        threshold=threshold,
    )

    print(f"‣ Finished subgraph {successfully_created}:")
    print(f"   • nodes: {G_cons.number_of_nodes()}  edges: {G_cons.number_of_edges()}")
    print(f"   • timeseries shape: {ts.shape}")
    return (coord, successfully_created)

def generate_train_data(
    nodes_helmond_path: str,
    meta_path: str,
    measurement_path: str,
    nodes_traffic_lights_path: str,
    base_output_dir: str = "data/training_data",
    num_samples: int = 16,
    threshold: float = 25,
    dist: int = 600,
    seed: int = 42
):
    base = Path(base_output_dir)
    seed = seed
    random_nodes_with_traffic_lights = get_random_locations_with_traffic_lights(
        nodes_traffic_lights_path=nodes_traffic_lights_path,
        seed=seed,
        num_samples=num_samples,
    )

    coord_dict = {}

    # Parallel execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_coord,
                coord,
                idx,
                base,
                nodes_helmond_path,
                dist,
                meta_path,
                measurement_path,
                threshold,
            )
            for idx, coord in enumerate(random_nodes_with_traffic_lights)
        ]
        for future in concurrent.futures.as_completed(futures):
            coord, idx = future.result()
            coord_dict.update({coord: idx})

    # tuple keys to strings for JSON serialization
    coord_dict = {f"{k[0]}, {k[1]}": v for k, v in coord_dict.items()}
    with open(f"{base_output_dir}/network_coords.json", 'w') as fp:
        json.dump(coord_dict, fp, indent=4)



import numpy as np

def get_static_data_object_subgraph(
    timeseries_subgraph_path: str,
    edge_features_path: str, 
    G_cons_path: str,
    G_pt_cons_path: str,
    max_steps: int,
    start_node: int = None,
    end_node: int = None,
    start_timestamp: pd.Timestamp = None,
        ):
    
    timeseries = pd.read_parquet(timeseries_subgraph_path)
    start_time, end_time = get_start_end_timestamps(
        timeseries,
        max_steps=max_steps,  # Assuming a maximum of 30 steps
        start_timestamp=start_timestamp,
        seed=None  # For reproducibility
    )
    timeseries = get_timeseries_from_start_timestamp(
        timeseries,
        start_timestamp=start_time,
        end_timestamp=end_time
    )
    # add start and end node to static data object
    if start_node is None:
        start_node = np.random.choice(timeseries.node_id.values).item()
    if end_node is None:
        end_node = np.random.choice(timeseries.node_id.values).item()
        while start_node == end_node:
            end_node = np.random.choice(timeseries.node_id.values).item()
    
    timeseries["start_node"] = start_node
    timeseries["end_node"] = end_node
    
    # compute distance-to-goal for each node
    goal_coords = timeseries.loc[
        timeseries["node_id"] == end_node, ["lat", "lon"]
    ].iloc[0]
    timeseries["distance_to_goal_meters"] = timeseries.apply(
        lambda row: haversine((row.lat, row.lon), (goal_coords.lat, goal_coords.lon)),
        axis=1,
    )
    
    edge_features = pd.read_parquet(edge_features_path)
    edge_features = edge_features.reset_index()
    edge_index = (
        torch.tensor(edge_features[["u", "v"]].values, dtype=torch.long, device=device)
        .t()
        .contiguous()
    )
    
    edge_attr = torch.tensor(edge_features[["length", "maxspeed"]].values, dtype=torch.float, device=device)
    
    node_features = timeseries.drop_duplicates(subset=["node_id"]).copy()
    node_features = torch.tensor(
        node_features[["lat", "lon", "has_light", "distance_to_goal_meters"]].values,
        dtype=torch.float,
        device=device,
    )
    
    static_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    
    static_data.start_node = start_node
    static_data.end_node = end_node
    
    G_cons = ox.io.load_graphml(G_cons_path)
    G_pt_cons = ox.io.load_graphml(G_pt_cons_path)
    
    static_data.G_sub = G_cons
    static_data.G_pt = G_pt_cons
    
    static_data.timeseries = timeseries
    
    return static_data


def create_and_save_static_data_objects(
    root_dir: str,
    max_steps: int,
    objects_per_network: int = 5,
    device: torch.device = torch.device("cpu")
) -> List[str]:
    
    saved_paths = []

    # Loop through all network subdirectories
    for network_id in os.listdir(root_dir):
        network_path = os.path.join(root_dir, network_id)
        if not os.path.isdir(network_path):
            continue

        # Define expected file paths
        timeseries_path = os.path.join(network_path, "timeseries.parquet")
        edge_features_path = os.path.join(network_path, "edge_features.parquet")
        G_cons_path = os.path.join(network_path, "G_cons.graphml")
        G_pt_cons_path = os.path.join(network_path, "G_pt_cons.graphml")

        for _ in range(objects_per_network):
            try:
                static_data = get_static_data_object_subgraph(
                    timeseries_path,
                    edge_features_path,
                    G_cons_path,
                    G_pt_cons_path,
                    max_steps
                )
                static_data = static_data.to(device)

                # Construct filename
                filename = f"{network_id}_{static_data.start_node}_{static_data.end_node}.pt"
                file_path = os.path.join(network_path, filename)

                # Save the object
                torch.save(static_data, file_path)
                saved_paths.append(file_path)
                print(f"Saved: {file_path}")

            except Exception as e:
                print(f"Error in network '{network_id}': {e}")

    return saved_paths

from torch_geometric.data import Dataset

class StaticDataObjectSet(Dataset):
    def __init__(self, base_dir: str, transform=None):
        super(StaticDataObjectSet, self).__init__(transform=transform)
        self.base_dir = base_dir
        self.network_dirs = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]
        self.file_paths = [ 
            os.path.join(base_dir, network_dir, file)
            for network_dir in self.network_dirs
            for file in os.listdir(os.path.join(base_dir, network_dir))
            if file.endswith(".pt")
        ] 
        
    def len(self):
        return len(self.file_paths)
    
    def get(self, idx):
        file_path = self.file_paths[idx]
        static_data = torch.load(file_path, weights_only=False)
        # static_data.start_node = static_data.start_node.item() if isinstance(static_data.start_node, torch.Tensor) else static_data.start_node
        # static_data.end_node = static_data.end_node.item() if isinstance(static_data.end_node, torch.Tensor) else static_data.end_node
        graph_id = os.path.basename(file_path)[:-3]  # Remove '.pt' from the end
        try:
            static_data.graph_id = graph_id
        except AttributeError as e:
            print(f"Error setting graph_id for {file_path}: {e}")
            static_data.graph_id = graph_id
            raise e
        # print(graph_id)
        return static_data


import os
import torch
from torch_geometric.data import InMemoryDataset, Data

class StaticDataObjectSet(InMemoryDataset):
    def __init__(self, root:str, transform=None, pre_filter=None, pre_transform=None, processed_file_names=["data.pt"]):
        self._procesed_file_names = processed_file_names
        super(StaticDataObjectSet, self).__init__(root, transform, pre_filter, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return self._procesed_file_names

    def process(self):
        base_dir = self.root
        file_paths = []
        for d in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, d)
            if os.path.isdir(dir_path):
                for f in os.listdir(dir_path):
                    if f.endswith(".pt"):
                        file_paths.append(os.path.join(dir_path, f))

        data_list = []
        for file_path in file_paths:
            print(f"Loading {file_path}")
            data = torch.load(file_path, weights_only=False)
            data.graph_id = os.path.basename(file_path)[:-3]
            data_list.append(data)
            print(f"Loaded {data.graph_id} with start node {data.start_node} and end node {data.end_node}")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

def main(
    nodes_traffic_lights_path: str,
    nodes_helmond_path: str,
    meta_path: str,
    measurement_path: str,
    num_samples: int,
    base_output_dir: str,
    threshold: float,
    dist: int,
    seed: int,
    objects_per_network: int,
):
    max_steps= 100
    
    generate_train_data(
        nodes_traffic_lights_path=nodes_traffic_lights_path,
        nodes_helmond_path=nodes_helmond_path,
        meta_path=meta_path,
        measurement_path=measurement_path,
        num_samples=num_samples,
        base_output_dir=base_output_dir,
        threshold=threshold,
        dist=dist,
        seed=seed,
    )
    create_and_save_static_data_objects(
        root_dir=base_output_dir,
        max_steps=max_steps,
        objects_per_network=objects_per_network,
    )
        
    
    


if __name__ == "__main__":
    # generate_train_data(
    #     nodes_traffic_lights_path="data/processed_new/intersection_lights.parquet",
    #     nodes_helmond_path="data/processed_new/helmond_nodes.parquet",
    #     meta_path="data/processed/intersection_metadata.csv",
    #     measurement_path="data/processed/intersection_measurements_31_01_24.csv",
    #     num_samples=3,
    #     base_output_dir="data/training_data_2",
    #     threshold=25,
    #     dist = 1000,
    #     seed= 1
    #     )
    
    # create_and_save_static_data_objects(
    #     root_dir="data/training_data_2",
    #     objects_per_network=1,
    # )
    
    # generate_train_data(
    #     nodes_traffic_lights_path="data/processed_new/intersection_lights.parquet",
    #     nodes_helmond_path="data/processed_new/helmond_nodes.parquet",
    #     meta_path="data/processed/intersection_metadata.csv",
    #     measurement_path="data/processed/intersection_measurements_31_01_24.csv",
    #     num_samples=3,
    #     base_output_dir="data/validation_data_2",
    #     threshold=25,
    #     dist=1000,
    #     seed = 12
    # )
    # create_and_save_static_data_objects(
    #     root_dir="data/validation_data_2",
    #     objects_per_network=1,
    # )
    
    # generate_train_data(
    #     nodes_traffic_lights_path="data/processed_new/intersection_lights.parquet",
    #     nodes_helmond_path="data/processed_new/helmond_nodes.parquet",
    #     meta_path="data/processed/intersection_metadata.csv",
    #     measurement_path="data/processed/intersection_measurements_31_01_24.csv",
    #     num_samples=3,
    #     base_output_dir="data/test_data_2",
    #     threshold=25,
    #     dist=1000,
    #     seed = 123
    # )
    # create_and_save_static_data_objects(
    #     root_dir="data/test_data_2",
    #     objects_per_network=1,
    # )
    
    from time import time
    start_time = time()
    main(
        nodes_traffic_lights_path="data/processed_new/intersection_lights.parquet",
        nodes_helmond_path="data/processed_new/helmond_nodes.parquet",
        meta_path="data/processed/intersection_metadata.csv",
        measurement_path="data/processed/intersection_measurements_31_01_24.csv",
        num_samples=16,
        base_output_dir="data/training_data_2",
        threshold=25,
        dist=1000,
        seed=1,
        objects_per_network=4,
    )
    
    main(
        nodes_traffic_lights_path="data/processed_new/intersection_lights.parquet",
        nodes_helmond_path="data/processed_new/helmond_nodes.parquet",
        meta_path="data/processed/intersection_metadata.csv",
        measurement_path="data/processed/intersection_measurements_31_01_24.csv",
        num_samples=8,
        base_output_dir="data/validation_data_2",
        threshold=25,
        dist=1000,
        seed=12,
        objects_per_network=4,
    )
    
    main(
        nodes_traffic_lights_path="data/processed_new/intersection_lights.parquet",
        nodes_helmond_path="data/processed_new/helmond_nodes.parquet",
        meta_path="data/processed/intersection_metadata.csv",
        measurement_path="data/processed/intersection_measurements_31_01_24.csv",
        num_samples=8,
        base_output_dir="data/test_data_2",
        threshold=25,
        dist=1000,
        seed=123,
        objects_per_network=4,
    )
    end_time = time()
    print(f"Total time taken: {end_time - start_time} seconds")