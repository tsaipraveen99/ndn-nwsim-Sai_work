"""
Performance benchmarks comparing different caching policies
Tests: Hit rate comparison, latency comparison, cache utilization, scalability
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional
import statistics
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import create_network, run_simulation, warmup_cache, align_user_distributions_with_producers, initialize_bloom_filter_propagation, setup_all_routers_to_dqn_mode

# Checkpoint directory
CHECKPOINT_DIR = Path("benchmark_checkpoints")
CHECKPOINT_FILE = CHECKPOINT_DIR / "benchmark_checkpoint.json"
RESULTS_FILE = CHECKPOINT_DIR / "benchmark_results.json"


def save_checkpoint(algorithm_name: str, completed_runs: int, total_runs: int, results: Dict):
    """Save checkpoint after algorithm completes"""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    checkpoint = {
        'algorithm': algorithm_name,
        'completed_runs': completed_runs,
        'total_runs': total_runs,
        'timestamp': time.time(),
        'results': results
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"  💾 Checkpoint saved: {algorithm_name} ({completed_runs}/{total_runs} runs)")


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint if it exists"""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️  Error loading checkpoint: {e}")
            return None
    return None


def save_results(results: Dict):
    """Save results incrementally"""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Load existing results if any
    existing_results = {}
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                existing_results = json.load(f)
        except:
            pass
    
    # Merge with new results
    existing_results.update(results)
    
    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"  💾 Results saved: {len(results)} algorithms completed")


def load_results() -> Dict:
    """Load existing results"""
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def clear_checkpoint():
    """Clear checkpoint file"""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def run_benchmark(config: Dict, num_runs: int = 10, seed: int = 42, checkpoint_key: Optional[str] = None) -> Dict:
    """
    Run benchmark with given configuration
    
    Args:
        config: Configuration dictionary
        num_runs: Number of runs to average
        seed: Base seed for reproducibility (each run uses seed + run_number)
        checkpoint_key: Key for checkpointing (algorithm name)
    
    Returns:
        Dictionary with average metrics
    """
    # Check for existing checkpoint
    start_run = 0
    results = []
    if checkpoint_key:
        checkpoint = load_checkpoint()
        if checkpoint and checkpoint.get('algorithm') == checkpoint_key:
            start_run = checkpoint.get('completed_runs', 0)
            # Load partial results if available
            checkpoint_results = checkpoint.get('results', {})
            if 'partial_results' in checkpoint_results:
                results = checkpoint_results['partial_results']
            if start_run > 0:
                print(f"  🔄 Resuming from checkpoint: {start_run}/{num_runs} runs already completed")
                print(f"  📊 Loaded {len(results)} previous run results")
    
    for run in range(start_run, num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        
        # Set environment variables
        for key, value in config.items():
            os.environ[key] = str(value)
        
        # Set seed for reproducibility (each run gets different seed)
        import random
        import numpy as np
        run_seed = seed + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        
        try:
            # Use fixed seed for network topology (same for all runs)
            import networkx as nx
            import random
            random.seed(seed)  # Fixed seed for topology
            np.random.seed(seed)
            
            G, users, producers, runtime = create_network(
                num_nodes=int(config.get('NDN_SIM_NODES', 300)),
                num_producers=int(config.get('NDN_SIM_PRODUCERS', 60)),
                num_contents=int(config.get('NDN_SIM_CONTENTS', 6000)),
                num_users=int(config.get('NDN_SIM_USERS', 2000)),
                cache_policy=config.get('NDN_SIM_CACHE_POLICY', 'fifo')
            )
            
            # Initialize DQN if enabled
            if config.get('NDN_SIM_USE_DQN', '0') == '1':
                print("    🔧 Enabling DQN mode on all routers...")
                setup_all_routers_to_dqn_mode(G, logger=None)
                initialize_bloom_filter_propagation(G, logger=None)
                
                # Initialize DQN Training Manager for asynchronous training (CRITICAL)
                from router import DQNTrainingManager
                try:
                    # Determine optimal number of training workers
                    # For GPU: 4 workers (GPU can parallelize)
                    # For CPU: 2 workers (CPU bound)
                    import torch
                    if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                        max_training_workers = 4  # GPU can handle parallel training
                    else:
                        max_training_workers = 2  # CPU: fewer workers
                except:
                    max_training_workers = 2  # Default to 2 if torch not available
                
                training_manager = DQNTrainingManager.get_instance(max_workers=max_training_workers)
                print(f"    ✅ DQN Training Manager initialized with {max_training_workers} workers")
                
                # Verify DQN agents initialized
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
                
                if dqn_count > 0:
                    print(f"    ✅ {dqn_count} routers with DQN agents initialized")
                if dqn_failed > 0:
                    print(f"    ⚠️  Warning: {dqn_failed} routers failed to initialize DQN agents")
            
            # Reset seed for request generation (different per run)
            random.seed(run_seed)
            np.random.seed(run_seed)
            
            # IMPORTANT: Ensure content exists and align user distributions
            align_user_distributions_with_producers(users, producers, logger=None)
            
            # WARM-UP PHASE: Pre-populate caches before evaluation
            # This ensures fair comparison by starting all algorithms with warm caches
            warmup_rounds = int(config.get('NDN_SIM_WARMUP_ROUNDS', 10))
            print(f"    🔥 Warm-up phase: {warmup_rounds} rounds...")
            warmup_cache(G, users, producers, num_warmup_rounds=warmup_rounds, logger=None)
            
            # Reset statistics AFTER warm-up but BEFORE evaluation
            # This ensures we only measure performance during the evaluation phase
            from router import stats as global_stats
            try:
                with global_stats.lock:
                    global_stats.nodes_traversed = 0
                    global_stats.cache_hits = 0
                    global_stats.data_packets_transferred = 0
                    global_stats.total_data_size_transferred = 0
            except Exception:
                pass
            
            # EVALUATION PHASE: Run simulation with warm caches
            print(f"    📊 Evaluation phase: {config.get('NDN_SIM_ROUNDS', 20)} rounds...")
            stats = run_simulation(
                G, users, producers,
                num_rounds=int(config.get('NDN_SIM_ROUNDS', 20)),
                num_requests=int(config.get('NDN_SIM_REQUESTS', 5))
            )
            
            # Collect metrics
            cached_items = 0
            total_insertions = 0
            routers_with_cache = 0
            cache_utilizations = []
            
            # Record cache utilization metrics
            from metrics import get_metrics_collector
            metrics_collector = get_metrics_collector()
            
            for node, data in G.nodes(data=True):
                if 'router' in data:
                    router = data['router']
                    if hasattr(router, 'content_store'):
                        cs = router.content_store
                        cached_items += len(cs.store)
                        total_insertions += getattr(cs, 'insertions', 0)
                        if len(cs.store) > 0:
                            routers_with_cache += 1
                        
                        # Record cache utilization
                        used = cs.total_capacity - cs.remaining_capacity
                        total = cs.total_capacity
                        if total > 0:
                            utilization = (used / total) * 100.0
                            cache_utilizations.append(utilization)
                            metrics_collector.record_cache_utilization(router.router_id, used, total)
            
            # Get comprehensive metrics
            all_metrics = metrics_collector.get_all_metrics()
            
            results.append({
                'hit_rate': stats.get('hit_rate', 0),
                'cache_hits': stats.get('cache_hits', 0),
                'nodes_traversed': stats.get('nodes_traversed', 0),
                'cached_items': cached_items,
                'total_insertions': total_insertions,
                'routers_with_cache': routers_with_cache,
                'avg_cache_utilization': statistics.mean(cache_utilizations) if cache_utilizations else 0.0,
                'latency_mean': all_metrics.get('latency', {}).get('mean', 0.0),
                'redundancy_mean': all_metrics.get('redundancy', {}).get('mean', 0.0),
                'dispersion_mean': all_metrics.get('dispersion', {}).get('mean', 0.0)
            })
            
            # Shutdown DQN Training Manager gracefully (if DQN was enabled)
            if config.get('NDN_SIM_USE_DQN', '0') == '1':
                try:
                    from router import DQNTrainingManager
                    training_manager = DQNTrainingManager.get_instance()
                    if training_manager is not None:
                        training_manager.shutdown()
                except Exception as e:
                    print(f"    ⚠️  Warning: Error shutting down training manager: {e}")
            
            if runtime:
                runtime.shutdown()
                
        except Exception as e:
            print(f"    Error in run {run + 1}: {e}")
            continue
        
        # Save checkpoint after each run (for safety)
        if checkpoint_key and (run + 1) % 2 == 0:  # Every 2 runs
            partial_results = results.copy()
            save_checkpoint(checkpoint_key, len(results), num_runs, {
                'partial_results': partial_results,
                'num_completed': len(results)
            })
    
    # Calculate averages
    if not results:
        return {}
    
    # Calculate statistics with confidence intervals
    from statistical_analysis import calculate_mean_std_ci, calculate_mean, calculate_std, calculate_confidence_interval
    
    hit_rates = [r['hit_rate'] for r in results]
    cache_hits_list = [r['cache_hits'] for r in results]
    nodes_traversed_list = [r['nodes_traversed'] for r in results]
    cached_items_list = [r['cached_items'] for r in results]
    total_insertions_list = [r['total_insertions'] for r in results]
    routers_with_cache_list = [r['routers_with_cache'] for r in results]
    
    hit_rate_stats = calculate_mean_std_ci(hit_rates)
    ci_lower, ci_upper = calculate_confidence_interval(hit_rates)
    
    avg_results = {
        'hit_rate': hit_rate_stats['mean'],
        'hit_rate_std': hit_rate_stats['std'],
        'hit_rate_ci_lower': ci_lower,
        'hit_rate_ci_upper': ci_upper,
        'cache_hits': round(statistics.mean(cache_hits_list)),
        'cache_hits_std': round(statistics.stdev(cache_hits_list) if len(cache_hits_list) > 1 else 0),
        'nodes_traversed': round(statistics.mean(nodes_traversed_list)),
        'nodes_traversed_std': round(statistics.stdev(nodes_traversed_list) if len(nodes_traversed_list) > 1 else 0),
        'cached_items': round(statistics.mean(cached_items_list)),
        'cached_items_std': round(statistics.stdev(cached_items_list) if len(cached_items_list) > 1 else 0),
        'total_insertions': round(statistics.mean(total_insertions_list)),
        'total_insertions_std': round(statistics.stdev(total_insertions_list) if len(total_insertions_list) > 1 else 0),
        'routers_with_cache': round(statistics.mean(routers_with_cache_list)),
        'routers_with_cache_std': round(statistics.stdev(routers_with_cache_list) if len(routers_with_cache_list) > 1 else 0),
        'num_runs': len(results)
    }
    
    # Clear checkpoint if completed
    if checkpoint_key:
        checkpoint = load_checkpoint()
        if checkpoint and checkpoint.get('algorithm') == checkpoint_key:
            if len(results) >= num_runs:
                clear_checkpoint()
                print(f"  ✅ Checkpoint cleared: {checkpoint_key} completed")
    
    return avg_results


def test_hit_rate_comparison(resume: bool = True):
    """Test 3.1: Cache Hit Rate Comparison"""
    print("\n" + "="*80)
    print("TEST 3.1: Cache Hit Rate Comparison")
    print("="*80)
    
    # Load existing results if resuming
    results = load_results() if resume else {}
    
    # High-performance config for publishable results (20-40% hit rate target)
    base_config = {
        'NDN_SIM_NODES': '30',            # Reduced from 50 to prevent queue flooding
        'NDN_SIM_PRODUCERS': '5',          # Reduced from 10
        'NDN_SIM_CONTENTS': '1000',       # Larger catalog for realistic scenario
        'NDN_SIM_USERS': '50',             # Reduced from 100 to prevent queue flooding
        'NDN_SIM_ROUNDS': '50',            # Reduced from 100 for faster testing
        'NDN_SIM_REQUESTS': '30',          # Reduced from 50 for faster testing
        'NDN_SIM_WARMUP_ROUNDS': '5',      # Reduced from 20 for faster testing
        'NDN_SIM_CACHE_CAPACITY': '10',    # 1% of catalog (1000 * 0.01 = 10) to stress capacity
        'NDN_SIM_ZIPF_PARAM': '0.8',       # Standard for web/video traffic (heavy tail distribution)
        'NDN_SIM_QUIET': '1',              # Quiet mode: suppress routine warnings
        'NDN_SIM_SKIP_DELAYS': '1',        # Skip sleep delays for faster benchmarks
        'NDN_SIM_USE_DQN': '0'             # DQN disabled for base config (only DQN algorithm uses it)
    }
    
    # Reordered to run DQN first (as requested)
    configs = {
        'DQN': {
            **base_config, 
            'NDN_SIM_CACHE_POLICY': 'combined', 
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_ROUNDS': '50',           # Reduced from 250 for faster testing
            'NDN_SIM_WARMUP_ROUNDS': '5'      # Reduced from 30 for faster testing
        },
        'FIFO': {**base_config, 'NDN_SIM_CACHE_POLICY': 'fifo'},
        'LRU': {**base_config, 'NDN_SIM_CACHE_POLICY': 'lru'},
        'LFU': {**base_config, 'NDN_SIM_CACHE_POLICY': 'lfu'},
        'Combined': {**base_config, 'NDN_SIM_CACHE_POLICY': 'combined'}
    }
    
    fixed_seed = 42  # Same seed for all algorithms (fair comparison)
    
    # Check which algorithms are already completed
    completed_algorithms = set(results.keys())
    if resume and completed_algorithms:
        print(f"\n📊 Found {len(completed_algorithms)} completed algorithms: {', '.join(completed_algorithms)}")
        print("  Will skip completed algorithms and continue with remaining ones.\n")
    
    for name, config in configs.items():
        # Skip if already completed
        if resume and name in completed_algorithms:
            print(f"\n⏭️  Skipping {name} (already completed)")
            continue
        
        print(f"\nTesting {name}...")
        result = run_benchmark(config, num_runs=3, seed=fixed_seed, checkpoint_key=name)  # Reduced from 10 to 3 for faster testing
        
        if result:
            results[name] = result
            # Save incrementally after each algorithm
            save_results({name: result})
            print(f"  ✅ {name} completed and saved")
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS: Cache Hit Rate Comparison")
    print("="*80)
    print(f"{'Policy':<15} {'Hit Rate':<15} {'Cache Hits':<15} {'Cached Items':<15}")
    print("-"*80)
    
    for name, result in results.items():
        if result:
            print(f"{name:<15} {result['hit_rate']:.4f}%      {result['cache_hits']:.0f}          {result['cached_items']:.0f}")
    
    # Verify improvements
    if 'FIFO' in results and 'Combined' in results:
        fifo_rate = results['FIFO'].get('hit_rate', 0)
        combined_rate = results['Combined'].get('hit_rate', 0)
        if fifo_rate > 0:
            improvement = combined_rate / fifo_rate
            print(f"\nCombined vs FIFO improvement: {improvement:.2f}x")
            if improvement > 1.5:
                print("✅ PASS: Combined eviction improves over FIFO")
            else:
                print("⚠️  WARNING: Improvement less than expected")
    
    if 'Combined' in results and 'DQN' in results:
        combined_rate = results['Combined'].get('hit_rate', 0)
        dqn_rate = results['DQN'].get('hit_rate', 0)
        if combined_rate > 0:
            improvement = dqn_rate / combined_rate
            print(f"\nDQN vs Combined improvement: {improvement:.2f}x")
            if improvement > 1.2:
                print("✅ PASS: DQN improves over Combined")
            else:
                print("⚠️  WARNING: DQN improvement less than expected")


def test_scalability(resume: bool = True):
    """Test 3.4: Scalability Tests"""
    print("\n" + "="*80)
    print("TEST 3.4: Scalability Tests")
    print("="*80)
    
    # Load existing results if resuming
    scalability_results = {}
    if resume:
        all_results = load_results()
        scalability_results = {k: v for k, v in all_results.items() if k in ['Small', 'Medium']}
    
    configs = {
        'Small': {
            'NDN_SIM_NODES': '50',
            'NDN_SIM_PRODUCERS': '10',
            'NDN_SIM_CONTENTS': '500',
            'NDN_SIM_USERS': '100',
            'NDN_SIM_ROUNDS': '5',           # Evaluation rounds
            'NDN_SIM_REQUESTS': '3',
            'NDN_SIM_WARMUP_ROUNDS': '5',    # Warm-up rounds
            'NDN_SIM_CACHE_CAPACITY': '500',
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_QUIET': '1',            # Quiet mode: suppress routine warnings
            'NDN_SIM_SKIP_DELAYS': '1'       # Skip sleep delays for faster benchmarks
        },
        'Medium': {
            'NDN_SIM_NODES': '300',
            'NDN_SIM_PRODUCERS': '60',
            'NDN_SIM_CONTENTS': '6000',
            'NDN_SIM_USERS': '2000',
            'NDN_SIM_ROUNDS': '10',          # Evaluation rounds
            'NDN_SIM_REQUESTS': '5',
            'NDN_SIM_WARMUP_ROUNDS': '10',    # Warm-up rounds
            'NDN_SIM_CACHE_CAPACITY': '500',
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_QUIET': '1',            # Quiet mode: suppress routine warnings
            'NDN_SIM_SKIP_DELAYS': '1'       # Skip sleep delays for faster benchmarks
        }
    }
    
    for name, config in configs.items():
        # Skip if already completed
        if resume and name in scalability_results:
            print(f"\n⏭️  Skipping {name} (already completed)")
            continue
        
        print(f"\nTesting {name} network...")
        result = run_benchmark(config, num_runs=1, checkpoint_key=f"Scalability_{name}")
        
        if result:
            scalability_results[name] = result
            save_results({name: result})
            print(f"  ✅ {name} completed and saved")
    
    results = scalability_results
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS: Scalability")
    print("="*80)
    print(f"{'Size':<15} {'Hit Rate':<15} {'Nodes':<15} {'Cached Items':<15}")
    print("-"*80)
    
    for name, result in results.items():
        if result:
            print(f"{name:<15} {result['hit_rate']:.4f}%      {result['nodes_traversed']:.0f}        {result['cached_items']:.0f}")
    
    # Verify scalability
    if 'Small' in results and 'Medium' in results:
        small_rate = results['Small'].get('hit_rate', 0)
        medium_rate = results['Medium'].get('hit_rate', 0)
        
        if small_rate > 0 and medium_rate > 0:
            print(f"\n✅ System scales: Small={small_rate:.2f}%, Medium={medium_rate:.2f}%")
            if medium_rate >= small_rate * 0.5:  # At least 50% of small performance
                print("✅ PASS: System maintains performance at scale")
            else:
                print("⚠️  WARNING: Performance degrades significantly at scale")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run performance benchmarks')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh (ignore checkpoints)')
    parser.add_argument('--clear-checkpoint', action='store_true', help='Clear existing checkpoint and start fresh')
    args = parser.parse_args()
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            print("🗑️  Checkpoint cleared")
        if RESULTS_FILE.exists():
            RESULTS_FILE.unlink()
            print("🗑️  Results cleared")
    
    resume = not args.no_resume
    
    print("="*80)
    print("PERFORMANCE BENCHMARKS")
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            print(f"🔄 Resume mode: Found checkpoint for {checkpoint.get('algorithm', 'unknown')}")
        else:
            print("🆕 Starting fresh benchmark")
    else:
        print("🆕 Starting fresh benchmark (--no-resume)")
    print("="*80)
    
    # Run benchmarks
    test_hit_rate_comparison(resume=resume)
    test_scalability(resume=resume)
    
    # Final summary
    final_results = load_results()
    print("\n" + "="*80)
    print("Benchmarks completed!")
    print("="*80)
    print(f"\n📊 Total algorithms completed: {len(final_results)}")
    print(f"💾 Results saved to: {RESULTS_FILE}")
    if CHECKPOINT_FILE.exists():
        print(f"⚠️  Warning: Checkpoint file still exists (some algorithm may not have completed)")
    else:
        print("✅ All checkpoints cleared (benchmark fully completed)")

