import osmnx as ox 

def create_helmond_nodes_df():
    G = ox.graph_from_place("Helmond, Netherlands", network_type="drive", simplify=True, retain_all=False)
    nodes, _ = ox.graph_to_gdfs(G)
    nodes = nodes.reset_index()
    nodes = nodes.rename(columns={"y": "lat", "x": "lon"})
    
    return nodes

if __name__ == "__main__":
    nodes = create_helmond_nodes_df()
    nodes.to_parquet("data/processed_new/helmond_nodes.parquet")
