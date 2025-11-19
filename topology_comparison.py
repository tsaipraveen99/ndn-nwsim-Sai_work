"""
Topology Comparison: Test DQN performance across different network topologies

Tests robustness of DQN approach across:
- Watts-Strogatz (small-world)
- Barabási-Albert (scale-free)
- Tree (hierarchical)
- Grid (regular)
"""

import os
import sys
import json
from typing import Dict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import run_benchmark


def run_topology_comparison(num_runs: int = 10, seed: int = 42, resume: bool = True) -> Dict:
    """
    Compare DQN performance across different network topologies
    
    Supports checkpointing and resume capability.
    
    Args:
        num_runs: Number of runs per topology
        seed: Base seed for reproducibility
        resume: If True, resume from checkpoint if available
    
    Returns:
        Dictionary with results for each topology
    """
    checkpoint_file = Path("topology_comparison_checkpoint.json")
    topologies = [
        ('watts_strogatz', {'NDN_SIM_TOPOLOGY_K': '4', 'NDN_SIM_TOPOLOGY_P': '0.2'}),
        ('barabasi_albert', {'NDN_SIM_TOPOLOGY_M': '2'}),
        ('tree', {}),
        ('grid', {})
    ]
    
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
        'NDN_SIM_USE_DQN': '1',
        'NDN_SIM_CACHE_POLICY': 'combined',
        'NDN_SIM_QUIET': '1',
        'NDN_SIM_SKIP_DELAYS': '1'
    }
    
    results = {}
    completed_topologies = set()
    
    # Load checkpoint if resuming
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                results = checkpoint.get('results', {})
                completed_topologies = set(checkpoint.get('completed_topologies', []))
                print(f"🔄 Resuming topology comparison: {len(completed_topologies)}/4 topologies completed")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}, starting fresh")
    
    print("\n" + "="*80)
    print("TOPOLOGY COMPARISON: Testing DQN Across Different Network Topologies")
    print("="*80)
    
    for topology_name, topology_params in topologies:
        if topology_name in completed_topologies:
            print(f"\n⏭️  Topology {topology_name} (already completed, skipping)")
            continue
            
        print(f"\n{'='*80}")
        print(f"Testing Topology: {topology_name.upper()}")
        print(f"{'='*80}")
        
        config = {
            **base_config,
            'NDN_SIM_TOPOLOGY': topology_name,
            **topology_params
        }
        
        results[topology_name] = run_benchmark(
            config, 
            num_runs=num_runs, 
            seed=seed,
            checkpoint_key=f'topology_{topology_name}'
        )
        completed_topologies.add(topology_name)
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_topologies': list(completed_topologies)}, f, indent=2)
    
    # Save results
    output_file = 'topology_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("TOPOLOGY COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Topology':<20} {'Hit Rate':<15} {'Std Dev':<15} {'Cache Hits':<15}")
    print("-" * 80)
    
    for topology, result in results.items():
        hr = result.get('hit_rate', 0)
        std = result.get('hit_rate_std', 0)
        hits = result.get('cache_hits', 0)
        print(f"{topology:<20} {hr:>6.2f}%       {std:>6.2f}%       {hits:>10}")
    
    # Clear checkpoint if all topologies completed
    if len(completed_topologies) >= len(topologies):
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("\n✅ Topology comparison checkpoint cleared (all topologies completed)")
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run topology comparison')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh (ignore checkpoints)')
    args = parser.parse_args()
    
    resume = not args.no_resume
    if not resume:
        checkpoint_file = Path("topology_comparison_checkpoint.json")
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("🗑️  Topology comparison checkpoint cleared")
    
    results = run_topology_comparison(num_runs=10, seed=42, resume=resume)

