#!/usr/bin/env python3
import networkx as nx
import random
import dill as pickle  # or use 'import pickle' if you saved with the standard pickle module
import matplotlib.pyplot as plt

def plot_graph(G):
    """
    Plots the graph G using a spring layout.
    Nodes are colored based on their type: routers, users, and producers.
    """
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    color_map = {'router': 'skyblue', 'user': 'red', 'producer': 'green'}
    node_colors = [color_map.get(G.nodes[node].get('type', ''), 'grey') for node in G.nodes()]
    node_sizes = [50 if G.nodes[node].get('type', '') == 'router' else 20 for node in G.nodes()]
    
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    
    # Optionally label a random subset of nodes for clarity.
    small_sample = random.sample(list(G.nodes()), min(len(G.nodes()), 300))
    labels = {node: node for node in small_sample}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Network Topology")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Specify the name of the pickle file containing the graph.
    filename = "network_state.pkl"  # Change as needed
    try:
        with open(filename, "rb") as f:
            G = pickle.load(f)
        print(f"Graph loaded successfully from {filename}")
        plot_graph(G)
    except Exception as e:
        print(f"Error loading graph from {filename}: {e}")
