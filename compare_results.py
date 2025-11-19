"""
Comparison Results Aggregator
Parses log files from multiple runs and generates comparison tables
"""

import os
import re
import json
import csv
from typing import Dict, List, Optional
from pathlib import Path
import logging
from statistical_analysis import generate_statistics_report, format_statistics_table

logger = logging.getLogger(__name__)


def parse_log_file(log_file: str) -> Dict:
    """
    Parse a simulation log file to extract metrics
    
    Args:
        log_file: Path to log file
    
    Returns:
        Dictionary with extracted metrics
    """
    metrics = {
        'hit_rate': 0.0,
        'cache_hits': 0,
        'cache_insertions': 0,
        'total_requests': 0,
        'latency_mean': 0.0,
        'cache_utilization': 0.0
    }
    
    if not os.path.exists(log_file):
        logger.warning(f"Log file not found: {log_file}")
        return metrics
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract cache hit rate
            hit_rate_match = re.search(r'hit rate[:\s]+([\d.]+)', content, re.IGNORECASE)
            if hit_rate_match:
                metrics['hit_rate'] = float(hit_rate_match.group(1))
            
            # Extract cache hits
            hits_match = re.search(r'cache hits?[:\s]+(\d+)', content, re.IGNORECASE)
            if hits_match:
                metrics['cache_hits'] = int(hits_match.group(1))
            
            # Extract cache insertions
            insertions_match = re.search(r'cache insertion[:\s]+(\d+)', content, re.IGNORECASE)
            if insertions_match:
                metrics['cache_insertions'] = int(insertions_match.group(1))
            
            # Extract latency from metrics
            latency_match = re.search(r'latency.*mean[:\s]+([\d.]+)', content, re.IGNORECASE)
            if latency_match:
                metrics['latency_mean'] = float(latency_match.group(1))
            
            # Extract cache utilization
            utilization_match = re.search(r'cache utilization[:\s]+([\d.]+)', content, re.IGNORECASE)
            if utilization_match:
                metrics['cache_utilization'] = float(utilization_match.group(1))
    
    except Exception as e:
        logger.error(f"Error parsing log file {log_file}: {e}")
    
    return metrics


def aggregate_results(results_dir: str, algorithm_name: str) -> List[Dict]:
    """
    Aggregate results from multiple runs of the same algorithm
    
    Args:
        results_dir: Directory containing result files
        algorithm_name: Name of the algorithm
    
    Returns:
        List of metric dictionaries from all runs
    """
    results = []
    results_path = Path(results_dir)
    
    # Look for log files matching pattern: {algorithm}_run{number}.log
    pattern = f"{algorithm_name}_run*.log"
    log_files = list(results_path.glob(pattern))
    
    if not log_files:
        # Try alternative pattern: {algorithm}/run{number}/*.log
        alg_dir = results_path / algorithm_name
        if alg_dir.exists():
            log_files = list(alg_dir.glob("**/*.log"))
    
    for log_file in sorted(log_files):
        metrics = parse_log_file(str(log_file))
        results.append(metrics)
        logger.info(f"Parsed {log_file}: hit_rate={metrics['hit_rate']:.4f}")
    
    return results


def load_published_results(paper_name: str) -> Optional[Dict]:
    """
    Load published results from state-of-the-art papers for comparison
    
    Args:
        paper_name: Name of paper (e.g., 'Fei_Wang_ICC2023')
    
    Returns:
        Dictionary with published results or None if not available
    """
    # Placeholder for published results
    # In practice, these would be loaded from a file or database
    published_results = {
        'Fei_Wang_ICC2023': {
            'hit_rate': 0.15,  # Example: 15% hit rate
            'network_size': 50,
            'notes': 'Multi-agent DQN with neighbor state (exact, not Bloom filters)'
        },
        # Add more published results as needed
    }
    
    return published_results.get(paper_name)


def compare_algorithms(results_dir: str, algorithms: List[str], output_file: Optional[str] = None, 
                       include_published: bool = True) -> Dict:
    """
    Compare multiple algorithms across multiple runs
    
    Args:
        results_dir: Directory containing result files
        algorithms: List of algorithm names to compare
        output_file: Optional path to save comparison report
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'algorithms': {},
        'statistics': {}
    }
    
    # Aggregate results for each algorithm
    for alg in algorithms:
        results = aggregate_results(results_dir, alg)
        if results:
            comparison['algorithms'][alg] = results
    
    # Extract metrics for statistical analysis
    metrics_to_analyze = ['hit_rate', 'latency_mean', 'cache_utilization']
    
    for metric in metrics_to_analyze:
        metric_results = {}
        for alg, runs in comparison['algorithms'].items():
            values = [run.get(metric, 0.0) for run in runs]
            if values:
                metric_results[alg] = values
        
        if metric_results:
            stats_report = generate_statistics_report(metric_results, metric_name=metric)
            comparison['statistics'][metric] = stats_report
    
    # Add published results if requested
    if include_published:
        published = load_published_results('Fei_Wang_ICC2023')
        if published:
            comparison['published_results'] = {
                'Fei_Wang_ICC2023': published
            }
    
    # Generate formatted report
    report_lines = []
    report_lines.append("# Algorithm Comparison Report")
    report_lines.append("")
    
    # Add published results section
    if include_published and 'published_results' in comparison:
        report_lines.append("## Published Results (State-of-the-Art)")
        report_lines.append("")
        for paper, results in comparison['published_results'].items():
            report_lines.append(f"### {paper}")
            report_lines.append(f"- Hit Rate: {results.get('hit_rate', 0):.4f}%")
            report_lines.append(f"- Network Size: {results.get('network_size', 'N/A')}")
            report_lines.append(f"- Notes: {results.get('notes', '')}")
            report_lines.append("")
    
    for metric, stats_report in comparison['statistics'].items():
        report_lines.append(format_statistics_table(stats_report))
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        logger.info(f"Comparison report saved to {output_file}")
    
    # Also save as JSON
    if output_file:
        json_file = output_file.replace('.md', '.json').replace('.txt', '.json')
        with open(json_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison data saved to {json_file}")
    
    return comparison


def export_to_csv(comparison: Dict, output_file: str):
    """
    Export comparison results to CSV format
    
    Args:
        comparison: Comparison dictionary
        output_file: Path to CSV file
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Algorithm', 'Run', 'Hit Rate', 'Cache Hits', 'Cache Insertions', 
                        'Latency Mean', 'Cache Utilization'])
        
        # Write data
        for alg, runs in comparison['algorithms'].items():
            for i, run in enumerate(runs):
                writer.writerow([
                    alg,
                    i + 1,
                    run.get('hit_rate', 0.0),
                    run.get('cache_hits', 0),
                    run.get('cache_insertions', 0),
                    run.get('latency_mean', 0.0),
                    run.get('cache_utilization', 0.0)
                ])
    
    logger.info(f"CSV export saved to {output_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <results_dir> <algorithm1> [algorithm2] ...")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    algorithms = sys.argv[2:]
    
    output_file = os.path.join(results_dir, 'comparison_report.md')
    comparison = compare_algorithms(results_dir, algorithms, output_file)
    
    csv_file = os.path.join(results_dir, 'comparison_results.csv')
    export_to_csv(comparison, csv_file)
    
    print(f"\nComparison complete! Results saved to:")
    print(f"  - {output_file}")
    print(f"  - {csv_file}")

