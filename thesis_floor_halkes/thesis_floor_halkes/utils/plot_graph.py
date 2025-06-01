from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


def plot_graph(
    data,
    save_path: str = "graph.png",
    with_labels: bool = True,
    node_size: int = 300,
    traffic_colors: dict = None,
):
    """
    Plot a PyTorch Geometric Data graph and save it as a PNG.

    Args:
        data (torch_geometric.data.Data): expects
            data.x[:,0] == is_traffic_light (0/1)
            data.x[:,1] == light_status    (0=red, 1=green)
        save_path (str): where to dump the PNG
        with_labels (bool): whether to draw node indices
        node_size (int): how big the circles should be
        traffic_colors (dict): override default colors:
            { 'green':..., 'red':..., 'none':... }
    """
    # Default color mapping
    traffic_colors = traffic_colors or {
        "green": "palegreen",
        "red": "tomato",
        "none": "grey",
    }

    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G)

    # Build color list
    node_colors = []
    if data.x is not None and data.x.size(1) >= 2:
        is_lights = data.x[:, 0].tolist()
        statuses = data.x[:, 1].tolist()
        for has_light, status in zip(is_lights, statuses):
            if int(has_light) == 1:
                # there is a light: choose green or red
                node_colors.append(
                    traffic_colors["green"]
                    if int(status) == 1
                    else traffic_colors["red"]
                )
            else:
                # no traffic light → grey
                node_colors.append(traffic_colors["none"])
    else:
        # fallback: all grey
        node_colors = [traffic_colors["none"]] * data.num_nodes

    # Draw
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=with_labels,
        node_color=node_colors,
        node_size=node_size,
        edge_color="gray",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
