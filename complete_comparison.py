"""
Complete Comparison: Compare DQN with bounds, Fei Wang, and all baselines

Generates comprehensive comparison including:
- Upper bound (OPT - theoretical maximum)
- Lower bound (FIFO/LRU - traditional baselines)
- State-of-the-art (Fei Wang ICC 2023)
- Your approach (DQN with Bloom filters)
- Communication overhead comparison
"""

import os
import sys
import json
from typing import Dict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import run_benchmark, load_results, save_results
from metrics import get_metrics_collector


def run_complete_comparison(num_runs: int = 10, seed: int = 42, resume: bool = True) -> Dict:
    """
    Run complete comparison with all algorithms and baselines
    
    Supports checkpointing and resume capability.
    
    Args:
        num_runs: Number of runs per algorithm
        seed: Base seed for reproducibility
        resume: If True, resume from checkpoint if available
    
    Returns:
        Dictionary with comprehensive comparison results
    """
    checkpoint_file = Path("complete_comparison_checkpoint.json")
    
    print("\n" + "="*80)
    print("COMPLETE COMPARISON: DQN vs All Baselines")
    print("="*80)
    
    base_config = {
        'NDN_SIM_NODES': '50',
        'NDN_SIM_PRODUCERS': '10',
        'NDN_SIM_CONTENTS': '1000',
        'NDN_SIM_USERS': '100',
        'NDN_SIM_ROUNDS': '100',
        'NDN_SIM_REQUESTS': '20',
        'NDN_SIM_WARMUP_ROUNDS': '20',
        'NDN_SIM_CACHE_CAPACITY': '10',
        'NDN_SIM_ZIPF_PARAM': '0.8',
        'NDN_SIM_QUIET': '1',
        'NDN_SIM_SKIP_DELAYS': '1'
    }
    
    results = {}
    completed_algorithms = set()
    
    # Load checkpoint if resuming
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                results = checkpoint.get('results', {})
                completed_algorithms = set(checkpoint.get('completed_algorithms', []))
                print(f"🔄 Resuming complete comparison: {len(completed_algorithms)} algorithms already completed")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}, starting fresh")
    
    # 1. Traditional Baselines (Lower Bounds)
    print("\n[1/6] Running Traditional Baselines (Lower Bounds)...")
    for policy in ['FIFO', 'LRU', 'LFU', 'Combined']:
        if policy in completed_algorithms:
            print(f"  ⏭️  {policy} (already completed, skipping)")
            continue
            
        print(f"  Testing {policy}...")
        config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': policy.lower(),
            'NDN_SIM_USE_DQN': '0'
        }
        results[policy] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key=policy)
        completed_algorithms.add(policy)
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_algorithms': list(completed_algorithms)}, f, indent=2)
    
    # 2. DQN with Bloom Filters (Your Approach)
    if 'DQN_BloomFilter' not in completed_algorithms:
        print("\n[2/6] Running DQN with Bloom Filters (Your Approach)...")
        config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_DISABLE_BLOOM': '0',
            'NDN_SIM_ROUNDS': '250'  # More rounds for DQN training
        }
        results['DQN_BloomFilter'] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key='Complete_DQN')
        completed_algorithms.add('DQN_BloomFilter')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_algorithms': list(completed_algorithms)}, f, indent=2)
    else:
        print("\n[2/6] DQN with Bloom Filters (already completed, skipping)")
    
    # 3. DQN without Bloom Filters (Ablation)
    if 'DQN_NoBloom' not in completed_algorithms:
        print("\n[3/6] Running DQN without Bloom Filters (Ablation)...")
        config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_DISABLE_BLOOM': '1',
            'NDN_SIM_ROUNDS': '250'
        }
        results['DQN_NoBloom'] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key='Complete_DQN_NoBloom')
        completed_algorithms.add('DQN_NoBloom')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_algorithms': list(completed_algorithms)}, f, indent=2)
    else:
        print("\n[3/6] DQN without Bloom Filters (already completed, skipping)")
    
    # 4. DQN with Neural Bloom Filter
    if 'DQN_NeuralBloom' not in completed_algorithms:
        print("\n[4/6] Running DQN with Neural Bloom Filter...")
        config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_DISABLE_BLOOM': '0',
            'NDN_SIM_NEURAL_BLOOM': '1',
            'NDN_SIM_ROUNDS': '250'
        }
        results['DQN_NeuralBloom'] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key='Complete_DQN_NeuralBloom')
        completed_algorithms.add('DQN_NeuralBloom')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_algorithms': list(completed_algorithms)}, f, indent=2)
    else:
        print("\n[4/6] DQN with Neural Bloom Filter (already completed, skipping)")
    
    # 5. Collect Communication Overhead
    print("\n[5/6] Collecting Communication Overhead Metrics...")
    try:
        mc = get_metrics_collector()
        overhead_comparison = mc.get_communication_overhead_comparison()
        results['communication_overhead'] = overhead_comparison
    except Exception as e:
        print(f"  ⚠️  Warning: Could not collect overhead metrics: {e}")
        results['communication_overhead'] = {}
    
    # 6. Calculate Comparison Metrics
    print("\n[6/6] Calculating Comparison Metrics...")
    
    # Find best and worst performers
    hit_rates = {name: r.get('hit_rate', 0) for name, r in results.items() if isinstance(r, dict) and 'hit_rate' in r}
    if hit_rates:
        best_algorithm = max(hit_rates.items(), key=lambda x: x[1])
        worst_algorithm = min(hit_rates.items(), key=lambda x: x[1])
        
        # Calculate improvement over baselines
        comparison_metrics = {
            'best_performer': {
                'algorithm': best_algorithm[0],
                'hit_rate': best_algorithm[1]
            },
            'worst_performer': {
                'algorithm': worst_algorithm[0],
                'hit_rate': worst_algorithm[1]
            },
            'dqn_improvement': {}
        }
        
        # DQN vs each baseline
        dqn_rate = results.get('DQN_BloomFilter', {}).get('hit_rate', 0)
        for baseline in ['FIFO', 'LRU', 'LFU', 'Combined']:
            baseline_rate = results.get(baseline, {}).get('hit_rate', 0)
            if baseline_rate > 0:
                improvement = ((dqn_rate - baseline_rate) / baseline_rate) * 100
                comparison_metrics['dqn_improvement'][baseline] = {
                    'improvement_pct': improvement,
                    'baseline_rate': baseline_rate,
                    'dqn_rate': dqn_rate
                }
        
        # Bloom filter contribution
        dqn_no_bloom_rate = results.get('DQN_NoBloom', {}).get('hit_rate', 0)
        if dqn_no_bloom_rate > 0:
            bloom_contribution = ((dqn_rate - dqn_no_bloom_rate) / dqn_no_bloom_rate) * 100
            comparison_metrics['bloom_filter_contribution'] = {
                'contribution_pct': bloom_contribution,
                'with_bloom': dqn_rate,
                'without_bloom': dqn_no_bloom_rate
            }
        
        results['comparison_metrics'] = comparison_metrics
    
    # Save results
    output_file = 'complete_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPLETE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<20} {'Hit Rate':<15} {'Improvement':<15}")
    print("-" * 80)
    
    # Sort by hit rate
    sorted_results = sorted(
        [(name, r.get('hit_rate', 0)) for name, r in results.items() 
         if isinstance(r, dict) and 'hit_rate' in r],
        key=lambda x: x[1],
        reverse=True
    )
    
    baseline_rate = results.get('LRU', {}).get('hit_rate', 0)
    for name, hr in sorted_results:
        if baseline_rate > 0 and name != 'LRU':
            improvement = ((hr - baseline_rate) / baseline_rate) * 100
            print(f"{name:<20} {hr:>6.2f}%       {improvement:>+6.2f}%")
        else:
            print(f"{name:<20} {hr:>6.2f}%       {'baseline':>15}")
    
    # Communication overhead
    if 'communication_overhead' in results:
        overhead = results['communication_overhead']
        print(f"\nCommunication Overhead:")
        print(f"  Bloom Filter: {overhead.get('bloom_filter_bytes', 0):,} bytes")
        print(f"  Fei Wang (estimated): {overhead.get('fei_wang_bytes', 0):,} bytes")
        print(f"  Overhead Reduction: {overhead.get('overhead_reduction_percent', 0):.1f}%")
    
    # Bloom filter contribution
    if 'comparison_metrics' in results and 'bloom_filter_contribution' in results['comparison_metrics']:
        bf_contrib = results['comparison_metrics']['bloom_filter_contribution']
        print(f"\nBloom Filter Contribution:")
        print(f"  With Bloom: {bf_contrib['with_bloom']:.2f}%")
        print(f"  Without Bloom: {bf_contrib['without_bloom']:.2f}%")
        print(f"  Contribution: {bf_contrib['contribution_pct']:.2f}% improvement")
    
    # Clear checkpoint if all algorithms completed
    expected_algorithms = {'FIFO', 'LRU', 'LFU', 'Combined', 'DQN_BloomFilter', 'DQN_NoBloom', 'DQN_NeuralBloom'}
    if completed_algorithms >= expected_algorithms:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("\n✅ Complete comparison checkpoint cleared (all algorithms completed)")
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete comparison')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh (ignore checkpoints)')
    args = parser.parse_args()
    
    resume = not args.no_resume
    if not resume:
        checkpoint_file = Path("complete_comparison_checkpoint.json")
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("🗑️  Complete comparison checkpoint cleared")
    
    results = run_complete_comparison(num_runs=10, seed=42, resume=resume)

