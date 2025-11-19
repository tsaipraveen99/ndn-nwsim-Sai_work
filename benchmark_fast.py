"""
Fast benchmark configuration - Target: 5 minutes total runtime
Optimized for speed while maintaining reasonable hit rates
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import run_benchmark, save_results, load_results, CHECKPOINT_FILE, RESULTS_FILE

if __name__ == '__main__':
    print("="*80)
    print("FAST BENCHMARK - Target: 5 minutes")
    print("="*80)
    print("\nOptimizations:")
    print("  - 2 runs per algorithm (was 10)")
    print("  - 15 rounds (was 50)")
    print("  - 10 requests per round (was 20)")
    print("  - 30 routers (was 50)")
    print("  - Smaller network for faster simulation")
    print("\nExpected: 5-10 minutes total")
    print("="*80)
    
    results = load_results()
    
    # FAST config - optimized for speed
    base_config = {
        'NDN_SIM_NODES': '30',           # Reduced from 50
        'NDN_SIM_PRODUCERS': '6',        # Reduced from 10
        'NDN_SIM_CONTENTS': '150',       # Reduced from 200
        'NDN_SIM_USERS': '50',           # Reduced from 100
        'NDN_SIM_ROUNDS': '15',          # Reduced from 50
        'NDN_SIM_REQUESTS': '10',        # Reduced from 20
        'NDN_SIM_CACHE_CAPACITY': '1000', # Keep large for good hit rates
        'NDN_SIM_ZIPF_PARAM': '1.2',      # Keep strong popularity
        'NDN_SIM_USE_DQN': '0'
    }
    
    configs = {
        'FIFO': {**base_config, 'NDN_SIM_CACHE_POLICY': 'fifo'},
        'LRU': {**base_config, 'NDN_SIM_CACHE_POLICY': 'lru'},
        'LFU': {**base_config, 'NDN_SIM_CACHE_POLICY': 'lfu'},
        'Combined': {**base_config, 'NDN_SIM_CACHE_POLICY': 'combined'},
        'DQN': {**base_config, 'NDN_SIM_CACHE_POLICY': 'combined', 'NDN_SIM_USE_DQN': '1'}
    }
    
    fixed_seed = 42
    completed_algorithms = set(results.keys())
    
    if completed_algorithms:
        print(f"\n📊 Found {len(completed_algorithms)} completed algorithms: {', '.join(completed_algorithms)}")
    
    for name, config in configs.items():
        if name in completed_algorithms:
            print(f"\n⏭️  Skipping {name} (already completed)")
            continue
        
        print(f"\nTesting {name}...")
        result = run_benchmark(config, num_runs=2, seed=fixed_seed, checkpoint_key=f"Fast_{name}")  # Only 2 runs!
        
        if result:
            results[name] = result
            save_results({name: result})
            print(f"  ✅ {name} completed: {result['hit_rate']:.2f}% hit rate")
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS: Fast Benchmark")
    print("="*80)
    print(f"{'Policy':<15} {'Hit Rate':<15} {'Cache Hits':<15} {'Cached Items':<15}")
    print("-"*80)
    
    for name, result in results.items():
        if result:
            print(f"{name:<15} {result['hit_rate']:.2f}%      {result['cache_hits']:.0f}          {result['cached_items']:.0f}")
    
    print("\n" + "="*80)
    print("Fast benchmark completed!")
    print("="*80)
    print(f"\n📊 Total algorithms completed: {len(results)}")
    print(f"💾 Results saved to: {RESULTS_FILE}")

