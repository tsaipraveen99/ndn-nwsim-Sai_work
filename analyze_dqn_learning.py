"""
Analyze DQN learning curves from simulation
Extracts learning curve data from routers and provides analysis
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List
from collections import defaultdict

def analyze_dqn_learning_curves(G, output_file: str = None):
    """
    Extract and analyze DQN learning curves from all routers
    
    Args:
        G: Network graph with routers
        output_file: Optional file to save aggregated learning curves
    """
    all_curves = {}
    router_curves = {}
    
    # Collect learning curves from all routers
    for node, data in G.nodes(data=True):
        if 'router' in data:
            router = data['router']
            if hasattr(router, 'content_store'):
                cs = router.content_store
                if hasattr(cs, 'dqn_agent') and cs.dqn_agent is not None:
                    agent = cs.dqn_agent
                    curve = agent.get_learning_curve()
                    if curve:
                        router_curves[router.router_id] = curve
                        all_curves[f"Router_{router.router_id}"] = curve
    
    if not all_curves:
        print("⚠️  No DQN learning curves found. Make sure DQN was enabled and simulation completed.")
        return None
    
    # Aggregate statistics across all routers
    print(f"\n{'='*80}")
    print(f"DQN LEARNING CURVE ANALYSIS")
    print(f"{'='*80}")
    print(f"Found learning curves from {len(router_curves)} routers\n")
    
    # Find common rounds across all routers
    all_rounds = set()
    for curve in router_curves.values():
        all_rounds.update(curve.keys())
    all_rounds = sorted(all_rounds)
    
    if not all_rounds:
        print("⚠️  No learning curve data found in routers")
        return None
    
    # Aggregate metrics across routers
    aggregated = defaultdict(list)
    for round_num in all_rounds:
        round_hit_rates = []
        round_losses = []
        round_rewards = []
        round_epsilons = []
        
        for router_id, curve in router_curves.items():
            if round_num in curve:
                round_hit_rates.append(curve[round_num].get('hit_rate', 0))
                round_losses.append(curve[round_num].get('loss', 0))
                round_rewards.append(curve[round_num].get('reward', 0))
                round_epsilons.append(curve[round_num].get('epsilon', 1.0))
        
        if round_hit_rates:
            aggregated[round_num] = {
                'hit_rate_mean': np.mean(round_hit_rates),
                'hit_rate_std': np.std(round_hit_rates),
                'loss_mean': np.mean(round_losses) if round_losses else 0,
                'reward_mean': np.mean(round_rewards) if round_rewards else 0,
                'epsilon_mean': np.mean(round_epsilons) if round_epsilons else 1.0,
                'num_routers': len(round_hit_rates)
            }
    
    # Print summary
    print(f"Rounds tracked: {len(all_rounds)} (from {min(all_rounds)} to {max(all_rounds)})")
    print(f"\n{'Round':<8} {'Hit Rate':<15} {'Loss':<12} {'Reward':<12} {'Epsilon':<12} {'Routers':<10}")
    print("-" * 80)
    
    # Show every 5th round for readability
    sample_rounds = all_rounds[::max(1, len(all_rounds) // 20)] + [all_rounds[-1]]
    sample_rounds = sorted(set(sample_rounds))
    
    for round_num in sample_rounds:
        if round_num in aggregated:
            stats = aggregated[round_num]
            print(f"{round_num:<8} "
                  f"{stats['hit_rate_mean']*100:>6.2f}% ± {stats['hit_rate_std']*100:>5.2f}  "
                  f"{stats['loss_mean']:>10.4f}  "
                  f"{stats['reward_mean']:>10.4f}  "
                  f"{stats['epsilon_mean']:>10.4f}  "
                  f"{stats['num_routers']:<10}")
    
    # Calculate learning trends
    if len(all_rounds) >= 10:
        early_rounds = all_rounds[:len(all_rounds)//3]
        late_rounds = all_rounds[-len(all_rounds)//3:]
        
        early_hit_rate = np.mean([aggregated[r]['hit_rate_mean'] for r in early_rounds if r in aggregated])
        late_hit_rate = np.mean([aggregated[r]['hit_rate_mean'] for r in late_rounds if r in aggregated])
        improvement = late_hit_rate - early_hit_rate
        
        print(f"\n{'='*80}")
        print(f"LEARNING TREND ANALYSIS")
        print(f"{'='*80}")
        print(f"Early rounds (1-{len(early_rounds)}): {early_hit_rate*100:.2f}% hit rate")
        print(f"Late rounds ({late_rounds[0]}-{late_rounds[-1]}): {late_hit_rate*100:.2f}% hit rate")
        print(f"Improvement: {improvement*100:+.2f}%")
        
        if improvement > 0.01:
            print("✅ DQN is learning and improving!")
        elif improvement > 0:
            print("⚠️  DQN is learning slowly")
        else:
            print("❌ DQN is not improving - may need more training or hyperparameter tuning")
    
    # Save aggregated data
    if output_file:
        output_data = {
            'aggregated': aggregated,
            'per_router': all_curves,
            'summary': {
                'num_routers': len(router_curves),
                'num_rounds': len(all_rounds),
                'round_range': (min(all_rounds), max(all_rounds))
            }
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Learning curve data saved to: {output_file}")
    
    return aggregated, all_curves


if __name__ == '__main__':
    # This would be called after a simulation
    print("To analyze DQN learning curves, run this after a simulation with DQN enabled.")
    print("The graph G with routers should be passed to analyze_dqn_learning_curves(G)")





