#!/usr/bin/env python3
import random
import matplotlib.pyplot as plt
import threading
import numpy as np
import time
from packet import Interest
from utils import ZipfDistribution, PrioritizedItem
from run import simulate_ndn, load_graph, plot_graph, save_graph

# Global flag (if used in your simulation functions)
alive_check_continue = True

def generate_interests(G, num_contents, num_rounds=10):
    """
    For a randomly chosen user node, generate a series of interest packets using a Zipf distribution.
    The connected user's make_interest method is assumed to return the hop count.
    """
    # Get all user nodes from the graph.
    user_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'user']
    if not user_nodes:
        return 0
    selected_user = random.choice(user_nodes)
    
    # Ensure at least one consumer is connected.
    if not G.nodes[selected_user]['router'].connected_users:
        return 0
    # Set the first connected user in test mode.
    G.nodes[selected_user]['router'].connected_users[0].mode = "test"
    
    # Create a Zipf distribution for content IDs.
    distribution = ZipfDistribution(num_contents, a=2, seed=69)
    total_hops = 0
    for _ in range(num_rounds):
        content_id = distribution.generate_content_id()
        interest = Interest(content_id, selected_user)
        # Call make_interest on the connected user.
        hop_count = G.nodes[selected_user]['router'].connected_users[0].make_interest(G, content_id, interest)
        total_hops += hop_count
    return total_hops / num_rounds

def run_test_simulation(graph, num_contents=50, num_rounds=10, num_requests=5):
    """
    Runs the main simulation function and then generates interests from a user.
    Returns the average hop count.
    """
    # In our minimal simulation, simulate_ndn processes interest and data messages.
    simulate_ndn(graph, [], [], num_contents, num_rounds, num_requests)
    avg_hops = generate_interests(graph, num_contents, num_rounds)
    return avg_hops

def get_router_hit_ratio(G):
    """
    Computes the average cache hit ratio across all routers.
    """
    router_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'router']
    ratios = []
    for node in router_nodes:
        router = G.nodes[node]['router']
        processed = router.router_stats.get("Interests Processed", 0)
        hits = router.router_stats.get("cache_hits", 0)
        ratios.append(hits / (processed + 1))
    return sum(ratios) / len(ratios) if ratios else 0

if __name__ == "__main__":
    # Simulation parameters
    num_contents = 100
    num_nodes = 100
    num_producers = 10
    num_users = 50
    num_rounds = 50
    num_requests = 5

    # Prepare a Zipf distribution (if needed for initial content creation)
    distribution = ZipfDistribution(num_contents, a=2)
    # Create a global 'contents' dictionary.
    global contents
    contents = {}
    for i in range(1, num_contents + 1):
        contents[f'content_{i:03}'] = 10  # Every content is of size 10

    # Load an existing graph from file (or use create_enhanced_graph to build one)
    G, users, producers = load_graph("my_graph16.pkl")
    
    # Optionally, plot the network topology.
    plot_graph(G)
    plt.show()

    # Run the simulation and generate interests.
    avg_hops = run_test_simulation(G, num_contents, num_rounds, num_requests)
    hit_ratio = get_router_hit_ratio(G)
    print(f"Average Hops: {avg_hops}")
    print(f"Average Router Cache Hit Ratio: {hit_ratio}")

    # Save the updated graph for later analysis.
    save_graph(G, 'my_graph_updated.pkl')
