"""
Generate a synthetic network from a Stochastic Block Model (SBM).

Outputs:
  - <out_prefix>_communities.csv  : node, community
  - <out_prefix>_edgelist.csv     : source, target
  - <out_prefix>_plot.png         : network visualization colored by community

Usage:
  python generate_starting_network.py --communities 5
  python generate_starting_network.py --communities 5 --nodes-per-community 50 --out-prefix my_net
"""

import argparse
import csv
import random
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# SBM edge probabilities
P_INTRA = 0.2   # within-community
P_INTER = 0.05  # between-community


def generate_sbm(n_communities: int, nodes_per_community: int, seed: Optional[int] = None):
    rng = random.Random(seed)
    n_nodes = n_communities * nodes_per_community

    # community assignment: node index -> community id
    community = {i: i // nodes_per_community for i in range(n_nodes)}

    edges = []
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            p = P_INTRA if community[u] == community[v] else P_INTER
            if rng.random() < p:
                edges.append((u, v))

    return community, edges


def write_communities(community: dict, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "community"])
        for node, comm in sorted(community.items()):
            writer.writerow([node, comm])


def write_edgelist(edges: list, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for u, v in edges:
            writer.writerow([u, v])


def write_plot(community: dict, edges: list, path: str, seed: Optional[int] = None):
    G = nx.Graph()
    G.add_nodes_from(community.keys())
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=seed)
    colors = [community[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=colors, cmap="tab10",
        node_size=40, width=0.3, alpha=0.85,
        with_labels=False, edge_color="gray",
    )
    ax.set_title(f"SBM network — {max(community.values()) + 1} communities "
                 f"(p_intra={P_INTRA}, p_inter={P_INTER})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate an SBM network as CSVs.")
    parser.add_argument("--communities", type=int, required=True,
                        help="Number of communities")
    parser.add_argument("--nodes-per-community", type=int, default=100,
                        help="Nodes per community (default: 100)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--out-prefix", type=str, default="network",
                        help="Output file prefix (default: network)")
    args = parser.parse_args()

    community, edges = generate_sbm(args.communities, args.nodes_per_community, args.seed)

    comm_path = f"{args.out_prefix}_communities.csv"
    edge_path = f"{args.out_prefix}_edgelist.csv"

    plot_path = f"{args.out_prefix}_plot.png"

    write_communities(community, comm_path)
    write_edgelist(edges, edge_path)
    write_plot(community, edges, plot_path, seed=args.seed)

    n_nodes = args.communities * args.nodes_per_community
    print(f"Generated {n_nodes} nodes across {args.communities} communities")
    print(f"  p_intra={P_INTRA}, p_inter={P_INTER}")
    print(f"  {len(edges)} edges")
    print(f"  -> {comm_path}")
    print(f"  -> {edge_path}")
    print(f"  -> {plot_path}")


if __name__ == "__main__":
    main()
