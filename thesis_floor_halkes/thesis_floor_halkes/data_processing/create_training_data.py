from pathlib import Path
import pandas as pd
import osmnx as ox

from thesis_floor_halkes.data_processing.create_subgraph import consilidate_subgraph, consolidate_edges_to_single, create_subgraph_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph
from thesis_floor_halkes.data_processing.expand_to_min_subgraph import expand_wait_times
from thesis_floor_halkes.data_processing.merge_sub_timeseries import merge_timeseries_pipeline

def get_sub_graph(nodes_helmond_path, 
                  node_features_path,
                  edge_features_path,
                  G_cons_path,
                  G_pt_cons_path,
                  seed,
                  dist,
                  ):
    nodes_helmond = pd.read_parquet(nodes_helmond_path)
    random_location = nodes_helmond.sample(1, random_state=seed)
    lat = random_location.iloc[0]["lat"]
    lon = random_location.iloc[0]["lon"]
    
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



def get_timeseries_subgraph( node_features_path, 
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


def generate_train_data(
    nodes_helmond_path: str,
    meta_path: str,
    measurement_path: str,
    base_output_dir: str = "data/training_data",
    n_subgraphs: int = 10,
    threshold: float = 25,
    dist: int = 100,
    seed: int = 0
):
    base = Path(base_output_dir)
    successfully_created = 0
    seed = seed
    
    while successfully_created < n_subgraphs:
        try:
            # Create subgraph directory
            subdir = base / f"subgraph_{successfully_created}"
            subdir.mkdir(parents=True, exist_ok=True)

            node_features_path = subdir / "node_metadata.parquet"
            edge_features_path = subdir / "edge_features.parquet"
            output_path_timeseries = subdir / "timeseries.parquet"
            G_cons_path = subdir / "G_cons.graphml"
            G_pt_cons_path = subdir / "G_pt_cons.graphml"

            # Build subgraph
            G_cons, _ = get_sub_graph(
                nodes_helmond_path=nodes_helmond_path,
                node_features_path=str(node_features_path),
                edge_features_path=str(edge_features_path),
                G_cons_path= G_cons_path,
                G_pt_cons_path= G_pt_cons_path,
                seed=seed,
                dist=dist,
            )
            
            # Generate timeseries
            ts = get_timeseries_subgraph(
                node_features_path=str(node_features_path),
                meta_path=meta_path,
                measurement_path=measurement_path,
                output_path_timeseries=str(output_path_timeseries),
                threshold=threshold,
            )

            print(f"‣ Finished subgraph {successfully_created}:")
            print(f"   • nodes: {G_cons.number_of_nodes()}  edges: {G_cons.number_of_edges()}")
            print(f"   • timeseries shape: {ts.shape}")

            successfully_created += 1
            seed += 1

        except OverflowError as e:
            print(f"Error creating subgraph {successfully_created}: {e}")
            seed += 1
            continue
    


if __name__ == "__main__":
    generate_train_data(
        nodes_helmond_path="data/processed_new/helmond_nodes.parquet",
        meta_path="data/processed/intersection_metadata.csv",
        measurement_path="data/processed/intersection_measurements_31_01_24.csv",
        n_subgraphs= 16,
        dist = 1000
    )
