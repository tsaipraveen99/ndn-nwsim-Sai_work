"""
Performance Evaluation Script
Runs comprehensive benchmarks and full simulation with improved settings
"""

import os
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import run_benchmark, test_hit_rate_comparison, test_scalability
from main import main as run_main_simulation


def estimate_runtime(config: dict) -> float:
    """Estimate simulation runtime in seconds"""
    num_nodes = int(config.get('NDN_SIM_NODES', 300))
    num_rounds = int(config.get('NDN_SIM_ROUNDS', 20))
    num_requests = int(config.get('NDN_SIM_REQUESTS', 5))
    num_users = int(config.get('NDN_SIM_USERS', 2000))
    
    # Rough estimate: ~0.1 seconds per round per 100 nodes
    base_time = (num_nodes / 100) * num_rounds * num_requests * 0.1
    user_overhead = (num_users / 100) * 0.05
    
    return base_time + user_overhead


def run_quick_benchmark():
    """Run quick benchmark with smaller network"""
    print("\n" + "="*80)
    print("QUICK BENCHMARK: Comparing Caching Policies")
    print("="*80)
    print("Using smaller network for faster results...")
    print("")
    
    base_config = {
        'NDN_SIM_NODES': '30',
        'NDN_SIM_PRODUCERS': '6',
        'NDN_SIM_CONTENTS': '300',
        'NDN_SIM_USERS': '50',
        'NDN_SIM_ROUNDS': '10',
        'NDN_SIM_REQUESTS': '3',
        'NDN_SIM_CACHE_CAPACITY': '500',
        'NDN_SIM_USE_DQN': '0'
    }
    
    configs = {
        'FIFO': {**base_config, 'NDN_SIM_CACHE_POLICY': 'fifo'},
        'LRU': {**base_config, 'NDN_SIM_CACHE_POLICY': 'lru'},
        'Combined': {**base_config, 'NDN_SIM_CACHE_POLICY': 'combined'},
    }
    
    results = {}
    start_time = time.time()
    
    for name, config in configs.items():
        print(f"Testing {name}...")
        config_start = time.time()
        results[name] = run_benchmark(config, num_runs=2)
        config_time = time.time() - config_start
        print(f"  Completed in {config_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*80)
    print("QUICK BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Policy':<15} {'Hit Rate':<15} {'Cache Hits':<15} {'Cached Items':<15}")
    print("-"*80)
    
    for name, result in results.items():
        if result:
            print(f"{name:<15} {result['hit_rate']:.4f}%      {result['cache_hits']:.0f}          {result['cached_items']:.0f}")
    
    print(f"\nTotal benchmark time: {total_time:.1f}s")
    
    return results


def run_full_simulation():
    """Run full simulation with improved settings"""
    print("\n" + "="*80)
    print("FULL SIMULATION: Improved Configuration")
    print("="*80)
    
    # Set improved configuration
    os.environ["NDN_SIM_NODES"] = "300"
    os.environ["NDN_SIM_PRODUCERS"] = "60"
    os.environ["NDN_SIM_CONTENTS"] = "6000"
    os.environ["NDN_SIM_USERS"] = "2000"
    os.environ["NDN_SIM_ROUNDS"] = "20"
    os.environ["NDN_SIM_REQUESTS"] = "5"
    os.environ["NDN_SIM_CACHE_CAPACITY"] = "500"
    os.environ["NDN_SIM_CACHE_POLICY"] = "combined"
    os.environ["NDN_SIM_USE_DQN"] = "0"  # Start without DQN for baseline
    os.environ["NDN_SIM_WARMUP_ROUNDS"] = "5"
    
    config = {k.replace("NDN_SIM_", ""): v for k, v in os.environ.items() if k.startswith("NDN_SIM_")}
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    estimated_time = estimate_runtime(config)
    print(f"\nEstimated runtime: {estimated_time:.1f} seconds (~{estimated_time/60:.1f} minutes)")
    print("Starting simulation...\n")
    
    start_time = time.time()
    
    try:
        # Run main simulation
        run_main_simulation()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Simulation completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Simulation failed after {elapsed:.1f}s: {e}")
        return False


def run_dqn_simulation():
    """Run simulation with DQN enabled"""
    print("\n" + "="*80)
    print("FULL SIMULATION: DQN Caching Enabled")
    print("="*80)
    
    # Set improved configuration with DQN
    os.environ["NDN_SIM_NODES"] = "300"
    os.environ["NDN_SIM_PRODUCERS"] = "60"
    os.environ["NDN_SIM_CONTENTS"] = "6000"
    os.environ["NDN_SIM_USERS"] = "2000"
    os.environ["NDN_SIM_ROUNDS"] = "20"
    os.environ["NDN_SIM_REQUESTS"] = "5"
    os.environ["NDN_SIM_CACHE_CAPACITY"] = "500"
    os.environ["NDN_SIM_CACHE_POLICY"] = "combined"
    os.environ["NDN_SIM_USE_DQN"] = "1"  # Enable DQN
    os.environ["NDN_SIM_WARMUP_ROUNDS"] = "5"
    
    print("Configuration: Same as above but with DQN enabled")
    print("Note: DQN training may use GPU if available")
    
    config = {k.replace("NDN_SIM_", ""): v for k, v in os.environ.items() if k.startswith("NDN_SIM_")}
    estimated_time = estimate_runtime(config) * 1.5  # DQN adds overhead
    print(f"\nEstimated runtime: {estimated_time:.1f} seconds (~{estimated_time/60:.1f} minutes)")
    print("Starting DQN simulation...\n")
    
    start_time = time.time()
    
    try:
        run_main_simulation()
        elapsed = time.time() - start_time
        print(f"\n✅ DQN simulation completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ DQN simulation failed after {elapsed:.1f}s: {e}")
        return False


def main():
    """Main performance evaluation"""
    print("="*80)
    print("PERFORMANCE EVALUATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA GPU available - DQN training will use GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS (Metal) GPU available - DQN training will use GPU")
        else:
            print("⚠️  No GPU available - DQN training will use CPU (slower)")
    except:
        print("⚠️  PyTorch not available - DQN will not work")
    
    print("")
    
    # Time estimates
    print("TIME ESTIMATES:")
    print("-" * 80)
    
    quick_config = {'NDN_SIM_NODES': '30', 'NDN_SIM_ROUNDS': '10', 'NDN_SIM_REQUESTS': '3', 'NDN_SIM_USERS': '50'}
    medium_config = {'NDN_SIM_NODES': '300', 'NDN_SIM_ROUNDS': '20', 'NDN_SIM_REQUESTS': '5', 'NDN_SIM_USERS': '2000'}
    
    quick_time = estimate_runtime(quick_config)
    medium_time = estimate_runtime(medium_config)
    
    print(f"Quick benchmark (30 nodes, 10 rounds): ~{quick_time*3:.0f}s (~{quick_time*3/60:.1f} min)")
    print(f"Full simulation (300 nodes, 20 rounds): ~{medium_time:.0f}s (~{medium_time/60:.1f} min)")
    print(f"Full simulation with DQN: ~{medium_time*1.5:.0f}s (~{medium_time*1.5/60:.1f} min)")
    print("")
    
    # Ask user what to run
    print("What would you like to run?")
    print("1. Quick benchmark (fast, ~2-3 minutes)")
    print("2. Full simulation - Combined eviction (medium, ~5-10 minutes)")
    print("3. Full simulation - DQN enabled (slower, ~10-15 minutes)")
    print("4. All of the above (comprehensive, ~20-30 minutes)")
    print("")
    
    # For now, run quick benchmark
    print("Running quick benchmark first...")
    quick_results = run_quick_benchmark()
    
    print("\n" + "="*80)
    print("Would you like to continue with full simulation?")
    print("You can run it manually with: python main.py")
    print("="*80)


if __name__ == '__main__':
    main()

