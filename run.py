#!/usr/bin/env python3
import threading
import networkx as nx
import random
import time
import sys
import numpy as np
import pandas as pd
import logging
import warnings
import dill as pickle


from utils import PrioritizedItem, ZipfDistribution
from router import Router, stats
from endpoints import User, Producer

sys.setrecursionlimit(1000)
warnings.filterwarnings("ignore")

# Set up logging
Logger5 = logging.getLogger('logger4')
Logger5.setLevel(logging.DEBUG)
Logger5.debug("Init logging")

# Global thread variables for alive-check/disconnection management.
alive_check_thread = None
alive_check_thread2 = None
alive_check_continue = True

def create_enhanced_graph(num_nodes=30, num_producers=20, num_contents=300, num_users=500):
    """
    Creates a graph with:
      - num_nodes routers,
      - num_producers producers,
      - num_users users.
    """
    G = nx.Graph()
    global contents, distribution

    # Create routers
    for i in range(num_nodes):
        router = Router(router_id=i, capacity=100, type='router')
        router.set_mode("Update")
        # Set the content store to use the RL-based caching mode.
        router.content_store.mode = "dqn_cache"
        G.add_node(i, router=router, type='router')

    # Create random edges between routers.
    for _ in range(3 * num_nodes):
        node_a, node_b = random.sample(list(G.nodes()), 2)
        if (not G.has_edge(node_a, node_b) and 
            G.nodes[node_a]['type'] == 'router' and 
            G.nodes[node_b]['type'] == 'router'):
            G.add_edge(node_a, node_b)

    # Create producers and attach them to routers.
    producers = []
    for i in range(num_producers):
        producer_node = num_nodes + i
        G.add_node(producer_node,
                   router=Router(router_id=producer_node, capacity=random.randint(10, 15), type='producer'),
                   type='producer')
        num_contents_for_producer = random.randint(2, num_contents // 2)
        prod_contents_list = random.sample(list(contents.items()), num_contents_for_producer)
        prod_contents = dict(prod_contents_list)
        producer_router = G.nodes[producer_node]['router']
        producer_router.set_mode("Update")
        producer = Producer([producer_router], prod_contents)
        # Distribute content to the producer's router.
        for content_id, size in prod_contents.items():
            producer_router.data_store[content_id] = size
        # Connect the producer to a random set of routers.
        num_connections = random.randint(1, 5)
        connected_routers = random.sample(
            [n for n, d in G.nodes(data=True) if d['type'] == 'router'], num_connections)
        for router in connected_routers:
            G.add_edge(producer_node, router)
            for content_id in prod_contents:
                G.nodes[router]['router'].add_to_FIB(content_id, producer_router.router_id)
        producers.append(producer)

    # Create users and assign them to routers.
    users = []
    for i in range(num_users):
        user_node = num_nodes + num_producers + i
        router_id = random.choice([n for n, d in G.nodes(data=True) if d['type'] == 'router'])
        # Create a user-type router.
        router = Router(router_id=user_node, capacity=0, type='user')
        router.content_store.status = "user"
        router.set_mode("update")
        G.add_node(user_node, router=router, type='user')
        G.add_edge(user_node, router_id)
        user = User(user_node, i, G.nodes[user_node]['router'], distribution)
        # Append the user to the router's list of connected users.
        G.nodes[user_node]['router'].connected_users.append(user)
        router.set_mode("Update")
        users.append(user)

    return G, users, producers

def create_enhanced_graph_from_gml(file_path, num_nodes=30, num_producers=20, num_contents=30, num_users=500):
    """
    Creates a graph from a GML file.
    """
    global distribution
    G = nx.read_gml(file_path)
    for i in G.nodes():
        G.nodes[i]['router'] = Router(router_id=i, capacity=40, type='router')
        G.nodes[i]['type'] = 'router'
        print(i, G.nodes[i]['type'])
    producers = []
    num_nodes = len(list(G.nodes))
    pc_dict = {}
    for i in range(1, num_contents + 1):
        producer_id = (i % num_producers)
        if producer_id not in pc_dict:
            pc_dict[producer_id] = {}
        content_id = f'content_{i:03}' #This needs to be changed. 
        pc_dict[producer_id][content_id] = 10
    for i in range(num_producers):
        producer_node = num_nodes + i
        G.add_node(producer_node,
                   router=Router(router_id=producer_node, capacity=10, type='producer'),
                   type='producer')
        num_contents_for_producer = random.randint(3, num_contents // 2)
        producer_router = G.nodes[producer_node]['router']
        producer = Producer([producer_router], pc_dict[i])
        for content_id, size in contents.items():
            producer_router.data_store[content_id] = size
        num_connections = random.randint(1, 5)
        connected_routers = random.sample(
            [n for n, d in G.nodes(data=True) if d['type'] == 'router'], num_connections)
        for router in connected_routers:
            G.add_edge(producer_node, router)
            for content_id in contents:
                G.nodes[router]['router'].add_to_FIB(content_id, producer_router.router_id)
        producers.append(producer)
    users = []
    for i in range(num_users):
        user_node = num_nodes + num_producers + i
        router_id = random.choice([n for n, d in G.nodes(data=True) if d['type'] == 'router'])
        router = Router(router_id=user_node, capacity=0, type='user')
        router.content_store.status = "user"
        G.add_node(user_node, router=router, type='user')
        G.add_edge(user_node, router_id)
        user = User(user_node, i, router, distribution)
        router.connected_users.append(user)
        users.append(user)
    return G, users, producers

def save_graph(graph, filename):
    # Clean up non-picklable attributes before saving.
    for node in graph.nodes():
        if 'router' in graph.nodes[node]:
            router = graph.nodes[node]['router']
            # Remove thread objects and RL agents from router and its content store.
            router.worker_thread = None
            if hasattr(router, 'content_store'):
                router.content_store.dqn_agent = None
    with open(filename, 'wb') as file:
        pickle.dump(graph, file)

def load_graph(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def simulate_ndn(G, users, producers, num_contents=30, num_rounds=10, num_requests=5):
    for round in range(num_rounds):
        threads = []
        # Start threads for user requests.
        for user in users:
            thread = threading.Thread(target=user.run, args=(G, 1, 1, num_contents))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        # Wait until all router message queues are empty.
        while True:
            all_empty = True
            for node in G.nodes():
                router = G.nodes[node].get('router')
                if G.nodes[node]["type"] == 'router' and router and not router.message_queue.empty():
                    all_empty = False
                    break
            if all_empty:
                break
        Logger5.debug(f"\nStatistics for Round {round + 1}:")
        for node in G.nodes():
            router = G.nodes[node].get('router')
            if router and G.nodes[node]['type'] == 'router':
                Logger5.debug(f"Router {router.router_id} stats: {router.router_stats}")

def simulate_ndn_unstable_edges(G, users, producers, num_contents=30, num_rounds=10, num_requests=5, percent_of_failure=5):
    router_edges = [edge for edge in G.edges() if G.nodes[edge[0]]["type"] == "router" and G.nodes[edge[1]]["type"] == "router"]
    num_edges_to_disconnect = int((percent_of_failure / 100) * len(router_edges))
    edges_to_disconnect = random.sample(router_edges, num_edges_to_disconnect)
    print(f"Initially disconnecting {len(edges_to_disconnect)} edges")
    for edge in edges_to_disconnect:
        G.remove_edge(*edge)
        G.nodes[edge[0]]["router"].remove_edge_from_FIB(edge[1])
        G.nodes[edge[1]]["router"].remove_edge_from_FIB(edge[0])
    print("Waiting 10 seconds before starting the simulation rounds...")
    time.sleep(10)
    for round_ in range(num_rounds):
        print(f"\nStarting round {round_ + 1} of {num_rounds}")
        threads = []
        for user in users:
            thread = threading.Thread(target=user.run, args=(G, num_rounds, num_requests, num_contents))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        all_empty = False
        while not all_empty:
            all_empty = True
            for node in G.nodes():
                router = G.nodes[node].get('router')
                if G.nodes[node]["type"] == 'router' and router and not router.message_queue.empty():
                    all_empty = False
                    break
            if not all_empty:
                print("Waiting for all router queues to empty...")
                time.sleep(1)
    print("\nReconnecting the previously disconnected edges...")
    for edge in edges_to_disconnect:
        G.add_edge(*edge)
        G.nodes[edge[0]]["router"].add_to_FIB(edge[1])
        G.nodes[edge[1]]["router"].add_to_FIB(edge[0])
        print(f"Reconnected edge {edge}")
    print("\nSimulation completed.")

def simulate_ndn_unstable_nodes(G, users, producers, num_contents=30, num_rounds=10, num_requests=5, percent_of_failure=5, restore_time=5, cs_alloc_mode="dqn_cache"):
    global alive_check_continue
    router_nodes = [node for node in G.nodes() if G.nodes[node]["type"] == "router"]
    # Set each router's content store to RL mode.
    for node in router_nodes:
        G.nodes[node]["router"].content_store.mode = cs_alloc_mode
        print(f"Router {node} set to mode {cs_alloc_mode}")
    alive_check_continue = True
    alive_check_thread = threading.Thread(target=check_routers_alive, args=(G, 5))
    alive_check_thread.start()
    alive_check_thread2 = threading.Thread(target=disconnect_routers, args=(G, 60))
    alive_check_thread2.start()
    for round_ in range(num_rounds):
        print(f"\nStarting round {round_ + 1} of {num_rounds}")
        threads = []
        for user in users:
            thread = threading.Thread(target=user.run, args=(G, 1, num_requests, num_contents))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
    print("\nSimulation completed.")

def shut_down_routers(G):
    print("Shutting down routers")
    for _, data in G.nodes(data=True):
        if 'router' in data and hasattr(data['router'], 'shutdown'):
            data['router'].shutdown()
    print("Joining all router threads")
    for _, data in G.nodes(data=True):
        data['router'].join_thread()
    global alive_check_continue
    alive_check_continue = False
    print("Joining alive-check threads")
    if alive_check_thread is not None:
        alive_check_thread.join()
    if alive_check_thread2 is not None:
        alive_check_thread2.join()

def disconnect_routers(G, check_interval=60):
    global alive_check_continue
    router_nodes = [node for node in G.nodes() if G.nodes[node]["type"] == "router"]
    weights = [G.degree(node) for node in router_nodes]
    while alive_check_continue:
        disconnected_node = np.random.choice(router_nodes, p=np.array(weights) / np.sum(weights))
        print("Disconnected", disconnected_node)
        G.nodes[disconnected_node]["router"].status = "down"
        connected_edges = list(G.edges(disconnected_node))
        for edge in connected_edges:
            if edge[0] in router_nodes and edge[1] in router_nodes:
                G.remove_edge(*edge)
                G.nodes[edge[0]]["router"].disconnect_router(edge[1])
                G.nodes[edge[1]]["router"].disconnect_router(edge[0])
        time.sleep(check_interval // 3)
        G.nodes[disconnected_node]["router"].status = "up"
        print("Reconnected", disconnected_node)
        for edge in connected_edges:
            if edge[0] in router_nodes and edge[1] in router_nodes:
                G.add_edge(*edge)
                G.nodes[edge[0]]["router"].connect_router(edge[1])
                G.nodes[edge[1]]["router"].connect_router(edge[0])
        time.sleep(check_interval)

def check_routers_alive(G, check_interval=5):
    global alive_check_continue
    while alive_check_continue:
        router_nodes = [node for node in G.nodes() if G.nodes[node]["type"] == "router"]
        for node in router_nodes:
            try:
                message = ('alive', (G))
                G.nodes[node]['router'].message_queue.put(PrioritizedItem(1, message))
            except Exception as e:
                print(f"Error checking router {node}: {e}")
        alive_routers = [node for node in router_nodes if G.nodes[node]['router'].status == "up"]
        print(f"Alive routers: {alive_routers}")
        time.sleep(check_interval)

def plot_graph(G):
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    color_map = {'router': 'skyblue', 'user': 'red', 'producer': 'green'}
    node_colors = [color_map.get(G.nodes[node]['type'], 'grey') for node in G.nodes()]
    node_sizes = [50 if G.nodes[node]['type'] == 'router' else 20 for node in G.nodes()]
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)  # type: ignore
    small_sample = random.sample(list(G.nodes()), min(len(G.nodes()), 300))
    labels = {node: node for node in small_sample}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

if __name__ == "__main__":
    num_contents = 500
    num_producers = 10
    num_nodes = 100
    num_users = 50
    num_rounds = 3000
    num_requests = 5
    distribution = ZipfDistribution(num_contents, a=2)
    global contents
    contents = {}
    for i in range(1, num_contents + 1):
        contents[f'content_{i:03}'] = 10

    # Load the graph from a GML file.
    G, users, producers = create_enhanced_graph_from_gml(
        file_path="AttMpls.gml",
        num_nodes=num_nodes,
        num_producers=num_producers,
        num_contents=num_contents,
        num_users=num_users
    )
    for user in users:
        print(user)
    for producer in producers:
        print(producer)
    print([(node, G.nodes[node]['type']) for node in G.nodes()])
    try:
        simulate_ndn_unstable_nodes(G, users, producers,
                                    num_contents=num_contents,
                                    num_rounds=num_rounds,
                                    num_requests=num_requests,
                                    cs_alloc_mode="dqn_cache")
    except Exception as exp:
        print(exp)
    time.sleep(20)
    shut_down_routers(G)
    print("\nSimulation Statistics:")
    print(f"Nodes Traversed: {stats.nodes_traversed}")
    print(f"Cache Hits: {stats.cache_hits}")
    print(f"Data Packets Transferred: {stats.data_packets_transferred}")
    print(f"Total Data Size Transferred: {stats.total_data_size_transferred}")
    print(f"Total Time (arbitrary units): {stats.total_time}")
    save_graph(G, 'my_graph17.pkl')
    plot_graph(G)
