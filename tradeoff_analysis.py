"""
Trade-off Analysis for Multi-Agent DQN with Neighbor-Aware State Representation

Analyzes:
- Bloom filter false positive rate vs communication overhead
- State space size vs learning speed
- Accuracy vs overhead trade-offs
"""

import numpy as np
from typing import Dict, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


def calculate_bloom_filter_fpr(size: int, hash_count: int, num_items: int) -> float:
    """
    Calculate theoretical false positive rate for Bloom filter
    
    Args:
        size: Bloom filter size in bits
        hash_count: Number of hash functions
        num_items: Number of items in filter
    
    Returns:
        Theoretical false positive rate
    """
    if num_items == 0:
        return 0.0
    
    # Theoretical FPR: (1 - e^(-kn/m))^k
    # where k=hash_count, n=num_items, m=size
    exponent = -(hash_count * num_items) / size
    fpr = (1 - np.exp(exponent)) ** hash_count
    return float(fpr)


def analyze_bloom_filter_tradeoffs() -> Dict:
    """
    Analyze trade-offs for different Bloom filter configurations
    
    Returns:
        Dictionary with trade-off analysis
    """
    # Typical cache sizes: 100, 200, 500, 1000 items
    cache_sizes = [100, 200, 500, 1000]
    bloom_sizes = [1000, 2000, 4000, 8000]
    hash_counts = [3, 4, 5, 6]
    
    results = {
        'configurations': [],
        'optimal_configs': {}
    }
    
    for cache_size in cache_sizes:
        best_config = None
        best_score = float('inf')
        
        for bloom_size in bloom_sizes:
            for hash_count in hash_counts:
                fpr = calculate_bloom_filter_fpr(bloom_size, hash_count, cache_size)
                overhead_bytes = bloom_size / 8
                
                # Score: balance FPR and overhead (lower is better)
                # Weight: FPR more important (weight=10), overhead less (weight=1)
                score = 10 * fpr + overhead_bytes / 100
                
                config = {
                    'cache_size': cache_size,
                    'bloom_size': bloom_size,
                    'hash_count': hash_count,
                    'fpr': fpr,
                    'overhead_bytes': overhead_bytes,
                    'score': score
                }
                
                results['configurations'].append(config)
                
                if score < best_score:
                    best_score = score
                    best_config = config
        
        if best_config:
            results['optimal_configs'][f'cache_{cache_size}'] = best_config
    
    return results


def analyze_state_space_tradeoffs() -> Dict:
    """
    Analyze trade-offs for different state space sizes
    
    Returns:
        Dictionary with state space size analysis
    """
    analysis = {
        'state_space_sizes': {
            'minimal_5': {
                'features': ['Content cached', 'Content size', 'Remaining capacity', 'Access frequency', 'Neighbor has content'],
                'pros': 'Fastest learning, minimal computation',
                'cons': 'May miss important information (topology, semantic)',
                'recommended': False
            },
            'optimal_10': {
                'features': 'All 10 essential features (current implementation)',
                'pros': 'Balanced: includes all critical info without redundancy',
                'cons': 'Slightly more computation than minimal',
                'recommended': True
            },
            'original_18': {
                'features': 'Original 18 features with 8 redundant',
                'pros': 'More information (but redundant)',
                'cons': 'Slower learning, more noise, redundant features',
                'recommended': False
            }
        },
        'learning_speed': {
            '5_features': 'Fastest (smallest state space)',
            '10_features': 'Optimal (current)',
            '18_features': 'Slowest (largest state space, redundant info)'
        },
        'recommendation': '10 features optimal: includes all critical information (especially Bloom filter neighbor awareness) without redundancy'
    }
    
    return analysis


def analyze_communication_overhead() -> Dict:
    """
    Analyze communication overhead of Bloom filter propagation
    
    Returns:
        Dictionary with overhead analysis
    """
    analysis = {
        'bloom_filter_overhead': {
            'size_bits': 2000,
            'size_bytes': 250,
            'update_frequency': 'Every N cache insertions (configurable)',
            'per_neighbor_cost': '250 bytes per update',
            'total_cost': '250 bytes * num_neighbors * update_frequency'
        },
        'alternatives': {
            'exact_cache_list': {
                'overhead': 'Variable: depends on cache size (could be KBs)',
                'pros': 'No false positives, exact information',
                'cons': 'High overhead, privacy concerns (exposes exact content)',
                'comparison': 'Bloom filter: 250 bytes vs exact list: potentially KBs'
            },
            'cache_summary_stats': {
                'overhead': 'Small (few bytes: hit rate, utilization)',
                'pros': 'Very low overhead',
                'cons': 'No content-specific information, can\'t check if neighbor has specific content',
                'comparison': 'Bloom filter provides content-specific info with low overhead'
            },
            'no_coordination': {
                'overhead': 0,
                'pros': 'No communication cost',
                'cons': 'No coordination, high redundancy, poor cache diversity',
                'comparison': 'Bloom filter enables coordination with minimal overhead'
            }
        },
        'scalability': {
            'overhead_per_router': '250 bytes * avg_neighbors * updates_per_round',
            'network_total': 'Overhead scales linearly with number of routers',
            'efficiency': 'Much lower than exact cache lists, enables scalable coordination'
        }
    }
    
    return analysis


def generate_tradeoff_report(output_file: str = "tradeoff_analysis_report.json") -> Dict:
    """
    Generate comprehensive trade-off analysis report
    
    Args:
        output_file: Path to save report
    
    Returns:
        Complete trade-off analysis dictionary
    """
    report = {
        'bloom_filter_tradeoffs': analyze_bloom_filter_tradeoffs(),
        'state_space_tradeoffs': analyze_state_space_tradeoffs(),
        'communication_overhead': analyze_communication_overhead(),
        'recommendations': {
            'bloom_filter': {
                'size': 2000,
                'hash_count': 4,
                'reason': 'Balances false positive rate (~1%) and communication overhead (250 bytes)'
            },
            'state_space': {
                'size': 10,
                'reason': 'Includes all critical features (especially Bloom filter neighbor awareness) without redundancy'
            },
            'update_frequency': {
                'recommendation': 'Every 10-20 cache insertions',
                'reason': 'Balances accuracy (frequent updates) and overhead (less frequent updates)'
            }
        }
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("TRADE-OFF ANALYSIS SUMMARY")
    print("="*80)
    print(f"Bloom Filter: 2000 bits, 4 hashes, ~1% FPR, 250 bytes overhead")
    print(f"State Space: 10 features (optimal, removed 8 redundant)")
    print(f"Communication: 250 bytes per neighbor update (much less than exact cache lists)")
    print(f"\nFull report saved to: {output_file}")
    
    return report


if __name__ == "__main__":
    generate_tradeoff_report()

