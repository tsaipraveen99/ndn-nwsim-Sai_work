"""
Comparison Visualization Script
Generates bar charts, learning curves, and other visualizations for paper
"""

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

import numpy as np
from typing import Dict, List, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)


def plot_comparison_bar_chart(results: Dict[str, Dict], metric_name: str, output_file: str, 
                              ylabel: str = None, title: str = None):
    """
    Create bar chart with error bars comparing algorithms
    
    Args:
        results: Dictionary mapping algorithm names to statistics dicts
        metric_name: Name of metric to plot
        output_file: Path to save figure
        ylabel: Y-axis label
        title: Chart title
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, cannot create plots")
        return
    
    algorithms = []
    means = []
    errors = []
    
    for alg_name, stats in results.items():
        if metric_name in stats:
            algorithms.append(alg_name)
            means.append(stats[metric_name].get('mean', 0))
            # Use half of confidence interval as error bar
            ci_lower = stats[metric_name].get('ci_95_lower', 0)
            ci_upper = stats[metric_name].get('ci_95_upper', 0)
            error = (ci_upper - ci_lower) / 2
            errors.append(error)
    
    if not algorithms:
        logger.warning(f"No data found for metric {metric_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(algorithms))
    
    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel(ylabel or metric_name, fontsize=12)
    ax.set_title(title or f'{metric_name} Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + errors[i],
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Bar chart saved to {output_file}")


def plot_learning_curves(learning_curves: Dict[str, Dict[int, Dict]], output_file: str):
    """
    Plot learning curves for DQN and baselines
    
    Args:
        learning_curves: Dictionary mapping algorithm names to {round: metrics} dicts
        output_file: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, cannot create plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Hit Rate over rounds
    ax1 = axes[0, 0]
    for alg_name, curve_data in learning_curves.items():
        rounds = sorted(curve_data.keys())
        hit_rates = [curve_data[r].get('hit_rate', 0) for r in rounds]
        ax1.plot(rounds, hit_rates, marker='o', label=alg_name, linewidth=2, markersize=4)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Hit Rate', fontsize=11)
    ax1.set_title('Cache Hit Rate Over Rounds', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Loss over rounds (DQN only)
    ax2 = axes[0, 1]
    for alg_name, curve_data in learning_curves.items():
        if 'DQN' in alg_name:
            rounds = sorted([r for r in curve_data.keys() if 'loss' in curve_data[r]])
            losses = [curve_data[r].get('loss', 0) for r in rounds]
            if losses:
                ax2.plot(rounds, losses, marker='s', label=alg_name, linewidth=2, markersize=4)
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('DQN Training Loss', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Reward over rounds (DQN only)
    ax3 = axes[1, 0]
    for alg_name, curve_data in learning_curves.items():
        if 'DQN' in alg_name:
            rounds = sorted([r for r in curve_data.keys() if 'reward' in curve_data[r]])
            rewards = [curve_data[r].get('reward', 0) for r in rounds]
            if rewards:
                ax3.plot(rounds, rewards, marker='^', label=alg_name, linewidth=2, markersize=4)
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('Average Reward', fontsize=11)
    ax3.set_title('DQN Average Reward', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Epsilon decay (DQN only)
    ax4 = axes[1, 1]
    for alg_name, curve_data in learning_curves.items():
        if 'DQN' in alg_name:
            rounds = sorted([r for r in curve_data.keys() if 'epsilon' in curve_data[r]])
            epsilons = [curve_data[r].get('epsilon', 1.0) for r in rounds]
            if epsilons:
                ax4.plot(rounds, epsilons, marker='d', label=alg_name, linewidth=2, markersize=4)
    ax4.set_xlabel('Round', fontsize=11)
    ax4.set_ylabel('Epsilon', fontsize=11)
    ax4.set_title('DQN Exploration Rate (Epsilon)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Learning curves saved to {output_file}")


def plot_learning_curves_from_json(json_file: str, output_file: str):
    """Load learning curve data from JSON and plot"""
    try:
        with open(json_file, 'r') as f:
            learning_curves = json.load(f)
        plot_learning_curves(learning_curves, output_file)
    except Exception as e:
        logger.error(f"Error loading/plotting learning curves: {e}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python visualize_comparison.py <results_json> <output_dir>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(results_file, 'r') as f:
            comparison_data = json.load(f)
        
        # Plot bar charts for each metric
        if 'statistics' in comparison_data:
            for metric_name, stats_report in comparison_data['statistics'].items():
                if 'algorithms' in stats_report:
                    output_path = os.path.join(output_dir, f'{metric_name}_comparison.png')
                    plot_comparison_bar_chart(
                        stats_report['algorithms'],
                        metric_name,
                        output_path,
                        ylabel=metric_name.replace('_', ' ').title()
                    )
        
        print(f"Visualizations saved to {output_dir}")
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

