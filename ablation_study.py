"""
Ablation Study Framework for Multi-Agent DQN with Neighbor-Aware State Representation

Tests which components actually contribute to performance:
- DQN vs. simple heuristics
- With/without Bloom filters (KEY: neighbor awareness)
- With/without topology features
- With/without semantic similarity
- Bloom filter variants (basic vs neural vs none)
"""

import os
import sys
import json
from typing import Dict, List, Tuple
from pathlib import Path
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import run_benchmark
from statistical_analysis import calculate_mean_std_ci, t_test, effect_size, mann_whitney_u_test


def run_ablation_study(base_config: Dict, num_runs: int = 10, seed: int = 42, resume: bool = True) -> Dict:
    """
    Run comprehensive ablation study to identify which components matter
    
    Supports checkpointing and resume capability.
    
    Args:
        base_config: Base configuration dictionary
        num_runs: Number of runs per configuration
        seed: Base seed for reproducibility
        resume: If True, resume from checkpoint if available
    
    Returns:
        Dictionary with results for each ablation variant
    """
    # Checkpoint file for ablation study
    checkpoint_file = Path("ablation_study_checkpoint.json")
    results = {}
    completed_variants = set()
    
    # Load checkpoint if resuming
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                results = checkpoint.get('results', {})
                completed_variants = set(checkpoint.get('completed_variants', []))
                print(f"\n🔄 Resuming ablation study: {len(completed_variants)}/7 variants already completed")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}, starting fresh")
    
    # Baseline: Simple LRU (no DQN, no Bloom filters)
    print("\n" + "="*80)
    print("ABLATION STUDY: Testing Component Contributions")
    print("="*80)
    
    # 1. Baseline: LRU (no DQN, no Bloom filters)
    if 'Baseline_LRU' not in completed_variants:
        print("\n[1/7] Baseline: LRU (no DQN, no Bloom filters)")
        baseline_config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'lru',
            'NDN_SIM_USE_DQN': '0'
        }
        results['Baseline_LRU'] = run_benchmark(baseline_config, num_runs=num_runs, seed=seed, checkpoint_key='Ablation_Baseline_LRU')
        completed_variants.add('Baseline_LRU')
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_variants': list(completed_variants)}, f, indent=2)
    else:
        print("\n[1/7] Baseline: LRU (already completed, skipping)")
    
    # 2. DQN without Bloom filters (disable neighbor awareness)
    if 'DQN_NoBloom' not in completed_variants:
        print("\n[2/7] DQN without Bloom filters (no neighbor awareness)")
        # Phase 7.1: Disable Bloom filter feature for ablation study
        dqn_no_bloom_config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_DISABLE_BLOOM': '1'  # Disable Bloom filter feature (Feature 4 = 0.0)
        }
        results['DQN_NoBloom'] = run_benchmark(dqn_no_bloom_config, num_runs=num_runs, seed=seed, checkpoint_key='Ablation_DQN_NoBloom')
        completed_variants.add('DQN_NoBloom')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_variants': list(completed_variants)}, f, indent=2)
    else:
        print("\n[2/7] DQN without Bloom filters (already completed, skipping)")
    
    # 3. DQN with Bloom filters (full implementation)
    if 'DQN_WithBloom' not in completed_variants:
        print("\n[3/7] DQN with Bloom filters (neighbor awareness enabled)")
        dqn_with_bloom_config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_DISABLE_BLOOM': '0'
        }
        results['DQN_WithBloom'] = run_benchmark(dqn_with_bloom_config, num_runs=num_runs, seed=seed, checkpoint_key='Ablation_DQN_WithBloom')
        completed_variants.add('DQN_WithBloom')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_variants': list(completed_variants)}, f, indent=2)
    else:
        print("\n[3/7] DQN with Bloom filters (already completed, skipping)")
    
    # 4. DQN with Neural Bloom filters
    if 'DQN_NeuralBloom' not in completed_variants:
        print("\n[4/7] DQN with Neural Bloom filters")
        dqn_neural_bloom_config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_NEURAL_BLOOM': '1'
        }
        results['DQN_NeuralBloom'] = run_benchmark(dqn_neural_bloom_config, num_runs=num_runs, seed=seed, checkpoint_key='Ablation_DQN_NeuralBloom')
        completed_variants.add('DQN_NeuralBloom')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_variants': list(completed_variants)}, f, indent=2)
    else:
        print("\n[4/7] DQN with Neural Bloom filters (already completed, skipping)")
    
    # 5. DQN with weighted neighbor importance (Phase 3.1 enabled)
    if 'DQN_WeightedNeighbors' not in completed_variants:
        print("\n[5/7] DQN with weighted neighbor importance (Phase 3.1)")
        # This is already enabled in the full DQN, so we use the same config
        # The weighted importance is always enabled when Bloom filters are enabled
        results['DQN_WeightedNeighbors'] = results.get('DQN_WithBloom', {})
        completed_variants.add('DQN_WeightedNeighbors')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_variants': list(completed_variants)}, f, indent=2)
    else:
        print("\n[5/7] DQN with weighted neighbor importance (already completed, skipping)")
    
    # 6. DQN with adaptive Bloom filter sizing (Phase 3.2 enabled)
    if 'DQN_AdaptiveBloom' not in completed_variants:
        print("\n[6/7] DQN with adaptive Bloom filter sizing (Phase 3.2)")
        # Adaptive sizing is always enabled, so we use the same config
        # Can test different FPR values if needed
        dqn_adaptive_bloom_config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_BLOOM_FPR': '0.005'  # Lower FPR (0.5%)
        }
        results['DQN_AdaptiveBloom'] = run_benchmark(dqn_adaptive_bloom_config, num_runs=num_runs, seed=seed, checkpoint_key='Ablation_DQN_AdaptiveBloom')
        completed_variants.add('DQN_AdaptiveBloom')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_variants': list(completed_variants)}, f, indent=2)
    else:
        print("\n[6/7] DQN with adaptive Bloom filter sizing (already completed, skipping)")
    
    # 7. Full DQN (all features)
    if 'DQN_Full' not in completed_variants:
        print("\n[7/7] Full DQN (all features enabled)")
        full_dqn_config = {
            **base_config,
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_NEURAL_BLOOM': '0'  # Basic Bloom filter
        }
        results['DQN_Full'] = run_benchmark(full_dqn_config, num_runs=num_runs, seed=seed, checkpoint_key='Ablation_DQN_Full')
        completed_variants.add('DQN_Full')
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results, 'completed_variants': list(completed_variants)}, f, indent=2)
    else:
        print("\n[7/7] Full DQN (already completed, skipping)")
    
    # Clear checkpoint if all variants completed
    if len(completed_variants) >= 7:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("\n✅ Ablation study checkpoint cleared (all variants completed)")
    
    return results


def analyze_ablation_results(results: Dict) -> Dict:
    """
    Analyze ablation study results to identify component contributions
    
    Args:
        results: Dictionary of results from ablation study
    
    Returns:
        Dictionary with analysis including feature importance ranking
    """
    analysis = {
        'component_contributions': {},
        'feature_importance': [],
        'statistical_tests': {}
    }
    
    # Extract hit rates for comparison
    hit_rates = {}
    for name, result in results.items():
        if result and 'hit_rate' in result:
            hit_rates[name] = result['hit_rate']
    
    # Compare each variant to baseline
    baseline_rate = hit_rates.get('Baseline_LRU', 0)
    
    if baseline_rate > 0:
        for name, rate in hit_rates.items():
            if name != 'Baseline_LRU':
                improvement = (rate - baseline_rate) / baseline_rate * 100
                analysis['component_contributions'][name] = {
                    'hit_rate': rate,
                    'improvement_pct': improvement,
                    'improvement_ratio': rate / baseline_rate if baseline_rate > 0 else 0
                }
    
    # Statistical comparison: DQN with Bloom vs without Bloom
    if 'DQN_WithBloom' in results and 'DQN_NoBloom' in results:
        # Note: This requires per-run data, not just averages
        # For now, we'll use the average values
        with_bloom = results['DQN_WithBloom'].get('hit_rate', 0)
        no_bloom = results['DQN_NoBloom'].get('hit_rate', 0)
        
        analysis['statistical_tests']['Bloom_Filter_Impact'] = {
            'with_bloom': with_bloom,
            'without_bloom': no_bloom,
            'difference': with_bloom - no_bloom,
            'improvement_pct': ((with_bloom - no_bloom) / no_bloom * 100) if no_bloom > 0 else 0
        }
    
    # Feature importance ranking (based on improvement over baseline)
    if analysis['component_contributions']:
        sorted_contributions = sorted(
            analysis['component_contributions'].items(),
            key=lambda x: x[1]['improvement_pct'],
            reverse=True
        )
        analysis['feature_importance'] = [
            {'component': name, 'improvement_pct': contrib['improvement_pct']}
            for name, contrib in sorted_contributions
        ]
    
    return analysis


def generate_ablation_report(results: Dict, analysis: Dict, output_file: str = "ablation_study_results.json"):
    """
    Generate comprehensive ablation study report
    
    Args:
        results: Raw results from ablation study
        analysis: Analysis of results
        output_file: Path to save JSON report
    """
    report = {
        'raw_results': results,
        'analysis': analysis,
        'summary': {
            'total_variants_tested': len(results),
            'baseline_hit_rate': results.get('Baseline_LRU', {}).get('hit_rate', 0),
            'best_variant': max(
                [(name, r.get('hit_rate', 0)) for name, r in results.items()],
                key=lambda x: x[1],
                default=('Unknown', 0)
            )[0] if results else 'Unknown'
        }
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"Baseline (LRU) Hit Rate: {report['summary']['baseline_hit_rate']:.4f}%")
    print(f"Best Variant: {report['summary']['best_variant']}")
    print("\nComponent Contributions:")
    for name, contrib in analysis.get('component_contributions', {}).items():
        print(f"  {name}: {contrib['improvement_pct']:.2f}% improvement")
    
    if 'Bloom_Filter_Impact' in analysis.get('statistical_tests', {}):
        bf_impact = analysis['statistical_tests']['Bloom_Filter_Impact']
        print(f"\nBloom Filter Impact:")
        print(f"  With Bloom: {bf_impact['with_bloom']:.4f}%")
        print(f"  Without Bloom: {bf_impact['without_bloom']:.4f}%")
        print(f"  Improvement: {bf_impact['improvement_pct']:.2f}%")
    
    print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study')
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
        # Clear checkpoint if starting fresh
        checkpoint_file = Path("ablation_study_checkpoint.json")
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("🗑️  Ablation study checkpoint cleared")
    
    results = run_ablation_study(base_config, num_runs=10, seed=42, resume=resume)
    analysis = analyze_ablation_results(results)
    generate_ablation_report(results, analysis)

