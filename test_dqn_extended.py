"""
Extended DQN Test with More Training Rounds
Tests DQN with significantly more rounds to allow proper learning
"""

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import create_network, run_simulation, warmup_cache, align_user_distributions_with_producers, initialize_bloom_filter_propagation
from analyze_dqn_learning import analyze_dqn_learning_curves

def run_extended_dqn_test():
    """Run extended DQN test with more training rounds"""
    
    print("="*80)
    print("EXTENDED DQN TEST - Extended Training")
    print("="*80)
    
    # Configuration for extended DQN training
    config = {
        'NDN_SIM_NODES': '50',
        'NDN_SIM_PRODUCERS': '10',
        'NDN_SIM_CONTENTS': '200',
        'NDN_SIM_USERS': '100',
        'NDN_SIM_ROUNDS': '200',           # Extended training rounds
        'NDN_SIM_REQUESTS': '20',
        'NDN_SIM_WARMUP_ROUNDS': '30',     # Extended warm-up
        'NDN_SIM_CACHE_CAPACITY': '1000',
        'NDN_SIM_ZIPF_PARAM': '1.2',
        'NDN_SIM_QUIET': '1',
        'NDN_SIM_SKIP_DELAYS': '1',
        'NDN_SIM_CACHE_POLICY': 'combined',
        'NDN_SIM_USE_DQN': '1'
    }
    
    # Set environment variables
    for key, value in config.items():
        os.environ[key] = str(value)
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set seeds
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n🔥 Creating network...")
    start_time = time.time()
    
    G, users, producers, runtime = create_network(
        num_nodes=int(config['NDN_SIM_NODES']),
        num_producers=int(config['NDN_SIM_PRODUCERS']),
        num_contents=int(config['NDN_SIM_CONTENTS']),
        num_users=int(config['NDN_SIM_USERS']),
        cache_policy=config['NDN_SIM_CACHE_POLICY']
    )
    
    # Enable DQN mode on all routers (CRITICAL: This was missing!)
    from main import setup_all_routers_to_dqn_mode
    print(f"\n🔧 Enabling DQN mode on all routers...")
    setup_all_routers_to_dqn_mode(G, logger=None)
    
    # Verify DQN agents are initialized
    dqn_count = 0
    dqn_failed = 0
    for node, data in G.nodes(data=True):
        if 'router' in data:
            router = data['router']
            if hasattr(router, 'content_store'):
                cs = router.content_store
                if hasattr(cs, 'mode') and cs.mode == "dqn_cache":
                    if hasattr(cs, 'dqn_agent') and cs.dqn_agent is not None:
                        dqn_count += 1
                    else:
                        dqn_failed += 1
    
    print(f"✅ Network created: {dqn_count} routers with DQN agents")
    if dqn_failed > 0:
        print(f"⚠️  Warning: {dqn_failed} routers failed to initialize DQN agents")
    
    # Align distributions
    print(f"\n📊 Aligning user distributions...")
    align_user_distributions_with_producers(users, producers, logger=None)
    
    # Initialize Bloom filters
    print(f"🔍 Initializing Bloom filter propagation...")
    initialize_bloom_filter_propagation(G, logger=None)
    
    # Extended warm-up
    warmup_rounds = int(config['NDN_SIM_WARMUP_ROUNDS'])
    print(f"\n🔥 Extended warm-up phase: {warmup_rounds} rounds...")
    warmup_cache(G, users, producers, num_warmup_rounds=warmup_rounds, logger=None)
    
    # Reset stats
    from router import stats as global_stats
    try:
        with global_stats.lock:
            global_stats.nodes_traversed = 0
            global_stats.cache_hits = 0
            global_stats.data_packets_transferred = 0
            global_stats.total_data_size_transferred = 0
    except Exception:
        pass
    
    # Extended training simulation
    num_rounds = int(config['NDN_SIM_ROUNDS'])
    num_requests = int(config['NDN_SIM_REQUESTS'])
    
    print(f"\n🚀 Starting extended DQN training: {num_rounds} rounds...")
    print(f"   (This will take longer but allows DQN to learn properly)")
    
    training_start = time.time()
    stats = run_simulation(
        G, users, producers,
        num_rounds=num_rounds,
        num_requests=num_requests
    )
    training_time = time.time() - training_start
    
    # Final statistics
    hit_rate = stats.get('hit_rate', 0)
    cache_hits = stats.get('cache_hits', 0)
    nodes_traversed = stats.get('nodes_traversed', 0)
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Hit Rate: {hit_rate*100:.2f}%")
    print(f"Cache Hits: {cache_hits:,}")
    print(f"Nodes Traversed: {nodes_traversed:,}")
    print(f"Training Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    
    # Analyze learning curves
    print(f"\n{'='*80}")
    print("ANALYZING DQN LEARNING CURVES")
    print(f"{'='*80}")
    
    output_file = "dqn_learning_curves_extended.json"
    learning_data = analyze_dqn_learning_curves(G, output_file=output_file)
    
    # Save final results
    results = {
        'hit_rate': hit_rate,
        'cache_hits': cache_hits,
        'nodes_traversed': nodes_traversed,
        'training_rounds': num_rounds,
        'warmup_rounds': warmup_rounds,
        'training_time_seconds': training_time,
        'dqn_agents': dqn_count
    }
    
    results_file = "dqn_extended_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"💾 Learning curves saved to: {output_file}")
    
    if runtime:
        runtime.shutdown()
    
    total_time = time.time() - start_time
    print(f"\n✅ Extended DQN test completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    return results, learning_data


if __name__ == '__main__':
    try:
        results, learning_data = run_extended_dqn_test()
        print("\n✅ Test completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

