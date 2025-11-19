"""
Learning Curve Plot Generator
Specialized script for plotting DQN learning curves
"""

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

import json
import os
import sys
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def plot_dqn_learning_curve(learning_curve_data: Dict[int, Dict], output_file: str, 
                           baseline_hit_rate: float = None):
    """
    Plot DQN learning curve with optional baseline comparison
    
    Args:
        learning_curve_data: Dictionary mapping round numbers to metrics
        output_file: Path to save figure
        baseline_hit_rate: Optional baseline hit rate to plot as horizontal line
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, cannot create plots")
        return
    
    rounds = sorted(learning_curve_data.keys())
    hit_rates = [learning_curve_data[r].get('hit_rate', 0) for r in rounds]
    losses = [learning_curve_data[r].get('loss', 0) for r in rounds]
    rewards = [learning_curve_data[r].get('reward', 0) for r in rounds]
    epsilons = [learning_curve_data[r].get('epsilon', 1.0) for r in rounds]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Hit Rate
    ax1 = axes[0, 0]
    ax1.plot(rounds, hit_rates, 'b-o', linewidth=2, markersize=5, label='DQN')
    if baseline_hit_rate is not None:
        ax1.axhline(y=baseline_hit_rate, color='r', linestyle='--', linewidth=2, label='Baseline')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Hit Rate', fontsize=12)
    ax1.set_title('Cache Hit Rate Over Rounds', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Loss
    ax2 = axes[0, 1]
    ax2.plot(rounds, losses, 'g-s', linewidth=2, markersize=5)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('DQN Training Loss', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Reward
    ax3 = axes[1, 0]
    ax3.plot(rounds, rewards, 'm-^', linewidth=2, markersize=5)
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Average Reward', fontsize=12)
    ax3.set_title('DQN Average Reward', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Epsilon
    ax4 = axes[1, 1]
    ax4.plot(rounds, epsilons, 'c-d', linewidth=2, markersize=5)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Epsilon', fontsize=12)
    ax4.set_title('Exploration Rate (Epsilon Decay)', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Learning curve plot saved to {output_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_learning_curves.py <learning_curve_json> [baseline_hit_rate]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    baseline = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    try:
        with open(json_file, 'r') as f:
            learning_curve_data = json.load(f)
        
        output_file = json_file.replace('.json', '_learning_curve.png')
        plot_dqn_learning_curve(learning_curve_data, output_file, baseline)
        print(f"Learning curve plot saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error plotting learning curve: {e}")

