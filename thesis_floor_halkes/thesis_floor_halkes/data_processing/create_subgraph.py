from networkx import MultiDiGraph
import pandas as pd
import osmnx as ox
import networkx as nx

from thesis_floor_halkes.features.graph.graph_generator import parse_length, parse_speed

ox.settings.bidirectional_network_types = ["drive", "walk", "bike"]


def create_subgraph_inside_helmond(nodes_helmond_path, lat, lon, dist):
    helmond_nodes = pd.read_parquet(nodes_helmond_path)
    
    G_pt = ox.graph_from_point(
        (lat, lon),
        dist=dist,
        network_type="drive",
        simplify=True,
    )
    common_nodes = set(G_pt.nodes()).intersection(helmond_nodes.osmid.values)
    G_sub = G_pt.subgraph(common_nodes).copy()
    G_sub = ox.truncate.largest_component(G_sub, strongly=False)
    
    return G_sub, G_pt


def consilidate_subgraph(G_sub):
    G_proj = ox.project_graph(G_sub)
    
    G_cons = ox.simplification.consolidate_intersections(
        G_proj, tolerance=10, rebuild_graph=True, dead_ends=False, reconnect_edges=True
    )
    G = ox.project_graph(G_cons, to_crs='EPSG:4326')
    loops = list(nx.selfloop_edges(G, keys=True))
    G.remove_edges_from(loops)
    return G 

def consolidate_edges_to_single(
    G_multi: MultiDiGraph, weight_attr: str = "length", agg: str = "min"
) -> MultiDiGraph:
    """
    Convert a MultiDiGraph to a MultiDiGraph by collapsing multiple edges (u,v,k)
    into a single (u,v) edge using aggregation, while preserving geometry and metadata.
    """
    G_single = nx.MultiDiGraph()
    G_single.graph.update(G_multi.graph)

    # Collect edge data
    edge_groups = {}
    for u, v, k, data in G_multi.edges(data=True, keys=True):
        edge_groups.setdefault((u, v), []).append(data)

    # Aggregate each edge group
    for (u, v), edges in edge_groups.items():
        if agg == "min":
            best = min(edges, key=lambda e: e.get(weight_attr, float("inf")))
        elif agg == "max":
            best = max(edges, key=lambda e: e.get(weight_attr, float("-inf")))
        elif agg == "mean":
            # Pick first for geometry, average numerics
            base = edges[0].copy()
            base[weight_attr] = sum(e.get(weight_attr, 0) for e in edges) / len(edges)
            best = base
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

        G_single.add_edge(u, v, **best)

    # Copy node attributes
    for node, data in G_multi.nodes(data=True):
        G_single.add_node(node, **data)

    return G_single

def get_node_features_subgraph(G_cons):
    nodes, _ = ox.graph_to_gdfs(G_cons)
    nodes = nodes[["y", "x", "osmid_original"]].reset_index(names="node_id")
    nodes = nodes.rename(columns={"y": "lat", "x": "lon"})
    
    nodes['osmid_original'] = nodes['osmid_original'].apply(
        lambda x: x if isinstance(x, list) else [x]
    )

    return nodes

def consolidate_edges_to_single(
    G_multi: MultiDiGraph, weight_attr: str = "length", agg: str = "min"
) -> MultiDiGraph:
    """
    Convert a MultiDiGraph to a MultiDiGraph by collapsing multiple edges (u,v,k)
    into a single (u,v) edge using aggregation, while preserving geometry and metadata.
    """
    G_single = nx.MultiDiGraph()
    G_single.graph.update(G_multi.graph)

    # Collect edge data
    edge_groups = {}
    for u, v, k, data in G_multi.edges(data=True, keys=True):
        edge_groups.setdefault((u, v), []).append(data)

    # Aggregate each edge group
    for (u, v), edges in edge_groups.items():
        if agg == "min":
            best = min(edges, key=lambda e: e.get(weight_attr, float("inf")))
        elif agg == "max":
            best = max(edges, key=lambda e: e.get(weight_attr, float("-inf")))
        elif agg == "mean":
            # Pick first for geometry, average numerics
            base = edges[0].copy()
            base[weight_attr] = sum(e.get(weight_attr, 0) for e in edges) / len(edges)
            best = base
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

        G_single.add_edge(u, v, **best)

    # Copy node attributes
    for node, data in G_multi.nodes(data=True):
        G_single.add_node(node, **data)

    return G_single

def get_edge_features_subgraph(G_cons):
    """
    From subgraph G_sub, return an edges GeoDataFrame with:
      - maxspeed: float (km/h)
      - length:   float (m)
      - geometry: the edge geometries
    """
    # extract edges gdf
    _, edges = ox.graph_to_gdfs(G_cons)

    # apply parsers
    edges["maxspeed"] = edges.apply(
        lambda row: parse_speed(row.get("maxspeed"), row.get("highway")),
        axis=1,
    )
    edges["length"] = edges["length"].apply(parse_length)

    # ensure floats
    edges["maxspeed"] = edges["maxspeed"].astype(float)
    edges["length"]   = edges["length"].astype(float)
    
    edges['osmid'] = edges['osmid'].apply(
        lambda x: x if isinstance(x, list) else [x]
    )
    
    edges = edges[["maxspeed", "length", "osmid", "u_original", "v_original"]]
    edges = edges.reset_index()

    return edges

if __name__ == "__main__":
    nodes_helmond_path = "data/processed_new/helmond_nodes.parquet"
    lat = 51.474744
    lon = 5.679176
    dist = 100
    
    G_sub, G_pt = create_subgraph_inside_helmond(nodes_helmond_path, lat, lon, dist)
    nodes, edges = ox.graph_to_gdfs(G_sub)
    G_cons = consilidate_subgraph(G_sub)
    nodes, edges = ox.graph_to_gdfs(G_cons)
    nodes = get_node_features_subgraph(G_cons)
    print(f"nodes:\n {nodes.head()}")
    edges = get_edge_features_subgraph(G_cons)
    print(nodes.dtypes)
    nodes.to_parquet("data/processed_new/subgraph_nodes.parquet")