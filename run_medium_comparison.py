#!/usr/bin/env python3
"""
Medium-sized network comparison for better results
Runs all caching algorithms on a medium network with more rounds
"""

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import run_benchmark
from visualize_comparison import plot_comparison_bar_chart

def run_medium_network_comparison():
    """Run comparison on medium network with more rounds"""
    print("\n" + "="*80)
    print("MEDIUM NETWORK COMPARISON")
    print("="*80)
    print("Network Configuration:")
    print("  Nodes: 50")
    print("  Producers: 10")
    print("  Contents: 500")
    print("  Users: 100")
    print("  Rounds: 20")
    print("  Requests per round: 5")
    print("  Cache Capacity: 200 items per router")
    print("="*80)
    print("This will take longer but should show better cache hit rates...")
    print("="*80)
    
    # Medium network configuration with more rounds
    base_config = {
        'NDN_SIM_NODES': '50',
        'NDN_SIM_PRODUCERS': '10',
        'NDN_SIM_CONTENTS': '500',
        'NDN_SIM_USERS': '100',
        'NDN_SIM_ROUNDS': '20',
        'NDN_SIM_REQUESTS': '5',
        'NDN_SIM_CACHE_CAPACITY': '200',
        'NDN_SIM_USE_DQN': '0',
        'NDN_SIM_WARMUP_ROUNDS': '5'
    }
    
    # Test all algorithms
    configs = {
        'FIFO': {**base_config, 'NDN_SIM_CACHE_POLICY': 'fifo'},
        'LRU': {**base_config, 'NDN_SIM_CACHE_POLICY': 'lru'},
        'LFU': {**base_config, 'NDN_SIM_CACHE_POLICY': 'lfu'},
        'Combined': {**base_config, 'NDN_SIM_CACHE_POLICY': 'combined'},
        'DQN': {**base_config, 'NDN_SIM_CACHE_POLICY': 'combined', 'NDN_SIM_USE_DQN': '1'}
    }
    
    results = {}
    fixed_seed = 42  # Same seed for all algorithms (fair comparison)
    num_runs = 3  # 3 runs for statistical significance
    
    total_start = time.time()
    
    for name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"Testing {name}...")
        print(f"{'='*80}")
        alg_start = time.time()
        
        try:
            result = run_benchmark(config, num_runs=num_runs, seed=fixed_seed)
            results[name] = result
            
            alg_time = time.time() - alg_start
            if result:
                print(f"  ✅ {name} completed in {alg_time:.1f}s ({alg_time/60:.1f} min)")
                print(f"     Hit Rate: {result.get('hit_rate', 0):.4f}%")
                print(f"     Cache Hits: {result.get('cache_hits', 0):.0f}")
                print(f"     Cached Items: {result.get('cached_items', 0):.0f}")
            else:
                print(f"  ⚠️  {name} completed but no results")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Policy':<15} {'Hit Rate %':<15} {'Cache Hits':<15} {'Cached Items':<15} {'Routers w/ Cache':<20}")
    print("-"*80)
    
    for name, result in results.items():
        if result:
            print(f"{name:<15} {result.get('hit_rate', 0):<15.4f} {result.get('cache_hits', 0):<15.0f} {result.get('cached_items', 0):<15.0f} {result.get('routers_with_cache', 0):<20.0f}")
        else:
            print(f"{name:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<20}")
    
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Calculate improvements
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    if 'FIFO' in results and results['FIFO']:
        fifo_rate = results['FIFO'].get('hit_rate', 0)
        print(f"\nBaseline (FIFO): {fifo_rate:.4f}%")
        
        for name in ['LRU', 'LFU', 'Combined', 'DQN']:
            if name in results and results[name]:
                rate = results[name].get('hit_rate', 0)
                if fifo_rate > 0:
                    improvement = rate / fifo_rate
                    improvement_pct = (rate - fifo_rate) / fifo_rate * 100
                    print(f"{name}: {rate:.4f}% ({improvement:.2f}x, {improvement_pct:+.1f}% improvement)")
                else:
                    print(f"{name}: {rate:.4f}%")
    
    # Find best algorithm
    valid_results = {k: v for k, v in results.items() if v and v.get('hit_rate', 0) > 0}
    if valid_results:
        best_alg = max(valid_results.items(), key=lambda x: x[1].get('hit_rate', 0))
        print(f"\n🏆 Best Algorithm: {best_alg[0]} with {best_alg[1].get('hit_rate', 0):.4f}% hit rate")
    
    # Save results to JSON
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "medium_network_comparison.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")
    
    # Prepare data for visualization
    viz_data = {}
    for name, result in results.items():
        if result:
            hit_rate = result.get('hit_rate', 0)
            # For visualization, we'll use simple stats (mean only for now)
            viz_data[name] = {
                'hit_rate': {
                    'mean': hit_rate,
                    'std': 0.0,  # Would need per-run data for std
                    'ci_95_lower': hit_rate,
                    'ci_95_upper': hit_rate
                }
            }
    
    # Generate visualization if matplotlib is available
    try:
        plot_file = output_dir / "medium_network_hit_rate_comparison.png"
        plot_comparison_bar_chart(
            viz_data,
            'hit_rate',
            str(plot_file),
            ylabel='Hit Rate (%)',
            title='Cache Hit Rate Comparison (Medium Network)'
        )
        print(f"✅ Visualization saved to {plot_file}")
    except Exception as e:
        print(f"⚠️  Could not generate visualization: {e}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_medium_network_comparison()
        print("\n✅ Comparison completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

