"""
Sensitivity Analysis for Multi-Agent DQN with Neighbor-Aware State Representation

Tests robustness to:
- Network size (50, 100, 200, 500 routers)
- Cache capacity (100, 200, 500, 1000 items)
- DQN hyperparameters (learning rate, epsilon decay, batch size)
- Bloom filter parameters (size, hash count)
"""

import os
import sys
import json
from typing import Dict, List
from pathlib import Path
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import run_benchmark
from statistical_analysis import calculate_mean_std_ci


def test_network_size_sensitivity(base_config: Dict, num_runs: int = 5, seed: int = 42, resume: bool = True) -> Dict:
    """
    Test sensitivity to network size
    
    Supports checkpointing and resume capability.
    
    Args:
        base_config: Base configuration
        num_runs: Number of runs per network size
        seed: Base seed
        resume: If True, resume from checkpoint if available
    
    Returns:
        Dictionary mapping network sizes to results
    """
    checkpoint_file = Path("sensitivity_network_size_checkpoint.json")
    results = {}
    completed_sizes = set()
    
    # Load checkpoint if resuming
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                results = checkpoint.get('results', {})
                completed_sizes = set(checkpoint.get('completed_sizes', []))
                print(f"🔄 Resuming network size sensitivity: {len(completed_sizes)}/4 sizes completed")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}, starting fresh")
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Network Size")
    print("="*80)
    
    network_sizes = [50, 100, 200, 500]
    
    for size in network_sizes:
        size_key = f'Network_{size}'
        if size_key in completed_sizes:
            print(f"\n⏭️  Network size {size} (already completed, skipping)")
            continue
            
        print(f"\nTesting network size: {size} routers")
        config = {
            **base_config,
            'NDN_SIM_NODES': str(size),
            'NDN_SIM_PRODUCERS': str(max(5, size // 5)),  # Scale producers with network
            'NDN_SIM_USERS': str(max(50, size * 2)),  # Scale users with network
            'NDN_SIM_CONTENTS': str(max(200, size * 10)),  # Scale contents with network
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1'
        }
        results[size_key] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key=f'Sensitivity_Network_{size}')
        completed_sizes.add(size_key)
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_sizes': list(completed_sizes)}, f, indent=2)
    
    # Clear checkpoint if all sizes completed
    if len(completed_sizes) >= len(network_sizes):
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("\n✅ Network size sensitivity checkpoint cleared (all sizes completed)")
    
    return results


def test_cache_capacity_sensitivity(base_config: Dict, num_runs: int = 5, seed: int = 42, resume: bool = True) -> Dict:
    """
    Test sensitivity to cache capacity
    
    Supports checkpointing and resume capability.
    
    Args:
        base_config: Base configuration
        num_runs: Number of runs per capacity
        seed: Base seed
        resume: If True, resume from checkpoint if available
    
    Returns:
        Dictionary mapping cache capacities to results
    """
    checkpoint_file = Path("sensitivity_cache_capacity_checkpoint.json")
    results = {}
    completed_capacities = set()
    
    # Load checkpoint if resuming
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                results = checkpoint.get('results', {})
                completed_capacities = set(checkpoint.get('completed_capacities', []))
                print(f"🔄 Resuming cache capacity sensitivity: {len(completed_capacities)}/4 capacities completed")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}, starting fresh")
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Cache Capacity")
    print("="*80)
    
    capacities = [100, 200, 500, 1000]
    
    for capacity in capacities:
        capacity_key = f'Capacity_{capacity}'
        if capacity_key in completed_capacities:
            print(f"\n⏭️  Cache capacity {capacity} (already completed, skipping)")
            continue
            
        print(f"\nTesting cache capacity: {capacity} items")
        config = {
            **base_config,
            'NDN_SIM_CACHE_CAPACITY': str(capacity),
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1'
        }
        results[capacity_key] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key=f'Sensitivity_Capacity_{capacity}')
        completed_capacities.add(capacity_key)
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_capacities': list(completed_capacities)}, f, indent=2)
    
    # Clear checkpoint if all capacities completed
    if len(completed_capacities) >= len(capacities):
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("\n✅ Cache capacity sensitivity checkpoint cleared (all capacities completed)")
    
    return results


def test_bloom_filter_sensitivity(base_config: Dict, num_runs: int = 5, seed: int = 42, resume: bool = True) -> Dict:
    """
    Test sensitivity to Bloom filter parameters
    
    Phase 4: Neural Bloom Filter evaluation
    Phase 3.2: Adaptive Bloom filter sizing with different FPR values
    
    Supports checkpointing and resume capability.
    
    Args:
        base_config: Base configuration
        num_runs: Number of runs per configuration
        seed: Base seed
        resume: If True, resume from checkpoint if available
    
    Returns:
        Dictionary mapping Bloom filter configs to results
    """
    checkpoint_file = Path("sensitivity_bloom_filter_checkpoint.json")
    results = {}
    completed_tests = set()
    
    # Load checkpoint if resuming
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                results = checkpoint.get('results', {})
                completed_tests = set(checkpoint.get('completed_tests', []))
                print(f"🔄 Resuming Bloom filter sensitivity: {len(completed_tests)}/4 tests completed")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}, starting fresh")
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Bloom Filter Parameters")
    print("="*80)
    
    # Test 1: Default Bloom filter (1% FPR, basic)
    if 'Bloom_Default_1pct' not in completed_tests:
        print("\n[1/4] Testing with default Bloom filter (FPR=0.01, basic)")
        config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_NEURAL_BLOOM': '0',
            'NDN_SIM_BLOOM_FPR': '0.01'  # 1% FPR
        }
        results['Bloom_Default_1pct'] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key='Sensitivity_Bloom_Default')
        completed_tests.add('Bloom_Default_1pct')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_tests': list(completed_tests)}, f, indent=2)
    else:
        print("\n[1/4] Default Bloom filter (already completed, skipping)")
    
    # Test 2: Lower FPR (0.5%)
    if 'Bloom_LowFPR_0.5pct' not in completed_tests:
        print("\n[2/4] Testing with lower FPR Bloom filter (FPR=0.005)")
        config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_NEURAL_BLOOM': '0',
            'NDN_SIM_BLOOM_FPR': '0.005'
        }
        results['Bloom_LowFPR_0.5pct'] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key='Sensitivity_Bloom_LowFPR')
        completed_tests.add('Bloom_LowFPR_0.5pct')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_tests': list(completed_tests)}, f, indent=2)
    else:
        print("\n[2/4] Lower FPR Bloom filter (already completed, skipping)")
    
    # Test 3: Higher FPR (2%)
    if 'Bloom_HighFPR_2pct' not in completed_tests:
        print("\n[3/4] Testing with higher FPR Bloom filter (FPR=0.02)")
        config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_NEURAL_BLOOM': '0',
            'NDN_SIM_BLOOM_FPR': '0.02'
        }
        results['Bloom_HighFPR_2pct'] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key='Sensitivity_Bloom_HighFPR')
        completed_tests.add('Bloom_HighFPR_2pct')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_tests': list(completed_tests)}, f, indent=2)
    else:
        print("\n[3/4] Higher FPR Bloom filter (already completed, skipping)")
    
    # Test 4: Neural Bloom filter (Phase 4)
    if 'Bloom_Neural' not in completed_tests:
        print("\n[4/4] Testing with Neural Bloom filter (Phase 4)")
        config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_NEURAL_BLOOM': '1',
            'NDN_SIM_BLOOM_FPR': '0.01'  # Reset to default
        }
        results['Bloom_Neural'] = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key='Sensitivity_Bloom_Neural')
        completed_tests.add('Bloom_Neural')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_tests': list(completed_tests)}, f, indent=2)
    else:
        print("\n[4/4] Neural Bloom filter (already completed, skipping)")
    
    # Clear checkpoint if all tests completed
    if len(completed_tests) >= 4:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("\n✅ Bloom filter sensitivity checkpoint cleared (all tests completed)")
    
    return results


def generate_sensitivity_report(all_results: Dict[str, Dict], output_file: str = "sensitivity_analysis_results.json"):
    """
    Generate sensitivity analysis report
    
    Args:
        all_results: Dictionary of all sensitivity test results
        output_file: Path to save JSON report
    """
    report = {
        'network_size_sensitivity': {},
        'cache_capacity_sensitivity': {},
        'bloom_filter_sensitivity': {},
        'summary': {}
    }
    
    # Organize results by test type
    for key, result in all_results.items():
        if key.startswith('Network_'):
            report['network_size_sensitivity'][key] = result
        elif key.startswith('Capacity_'):
            report['cache_capacity_sensitivity'][key] = result
        elif key.startswith('Bloom_'):
            report['bloom_filter_sensitivity'][key] = result
    
    # Calculate trends
    if report['network_size_sensitivity']:
        network_sizes = sorted([int(k.split('_')[1]) for k in report['network_size_sensitivity'].keys()])
        hit_rates = [report['network_size_sensitivity'][f'Network_{s}'].get('hit_rate', 0) for s in network_sizes]
        report['summary']['network_size_trend'] = {
            'sizes': network_sizes,
            'hit_rates': hit_rates,
            'scales_well': hit_rates[-1] >= hit_rates[0] * 0.8 if len(hit_rates) > 1 else True
        }
    
    if report['cache_capacity_sensitivity']:
        capacities = sorted([int(k.split('_')[1]) for k in report['cache_capacity_sensitivity'].keys()])
        hit_rates = [report['cache_capacity_sensitivity'][f'Capacity_{c}'].get('hit_rate', 0) for c in capacities]
        report['summary']['cache_capacity_trend'] = {
            'capacities': capacities,
            'hit_rates': hit_rates,
            'improves_with_capacity': hit_rates[-1] > hit_rates[0] if len(hit_rates) > 1 else True
        }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)
    
    if report['summary'].get('network_size_trend'):
        trend = report['summary']['network_size_trend']
        print(f"\nNetwork Size Trend: {'Scales well' if trend['scales_well'] else 'Degrades with size'}")
        for size, rate in zip(trend['sizes'], trend['hit_rates']):
            print(f"  {size} routers: {rate:.4f}% hit rate")
    
    if report['summary'].get('cache_capacity_trend'):
        trend = report['summary']['cache_capacity_trend']
        print(f"\nCache Capacity Trend: {'Improves' if trend['improves_with_capacity'] else 'No improvement'}")
        for cap, rate in zip(trend['capacities'], trend['hit_rates']):
            print(f"  {cap} items: {rate:.4f}% hit rate")
    
    print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sensitivity analysis')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh (ignore checkpoints)')
    args = parser.parse_args()
    
    base_config = {
        'NDN_SIM_NODES': '50',
        'NDN_SIM_PRODUCERS': '10',
        'NDN_SIM_CONTENTS': '1000',
        'NDN_SIM_USERS': '100',
        'NDN_SIM_ROUNDS': '100',
        'NDN_SIM_REQUESTS': '20',
        'NDN_SIM_CACHE_CAPACITY': '10',
        'NDN_SIM_ZIPF_PARAM': '0.8'
    }
    
    resume = not args.no_resume
    if not resume:
        # Clear checkpoints if starting fresh
        for checkpoint_file in [
            Path("sensitivity_network_size_checkpoint.json"),
            Path("sensitivity_cache_capacity_checkpoint.json"),
            Path("sensitivity_bloom_filter_checkpoint.json")
        ]:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
        print("🗑️  Sensitivity analysis checkpoints cleared")
    
    all_results = {}
    
    # Run sensitivity tests
    all_results.update(test_network_size_sensitivity(base_config, num_runs=5, seed=42, resume=resume))
    all_results.update(test_cache_capacity_sensitivity(base_config, num_runs=5, seed=42, resume=resume))
    all_results.update(test_bloom_filter_sensitivity(base_config, num_runs=5, seed=42, resume=resume))
    
    generate_sensitivity_report(all_results)

