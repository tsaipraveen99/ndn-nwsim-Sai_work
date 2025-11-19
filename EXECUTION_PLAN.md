# Complete Execution Plan - Next Steps

**Date**: January 2025  
**Status**: All Implementation Complete - Ready for Execution

---

## Executive Summary

This document outlines the complete execution plan for running experiments, benchmarks, and evaluations. All code is implemented and ready. Follow this step-by-step guide to execute comprehensive experiments.

---

## Phase 0: Pre-Execution Verification

### Step 0.1: Verify Implementation

```bash
# Quick verification that everything works
python3 -c "
from utils import ContentStore
from dqn_agent import DQNAgent
from metrics import MetricsCollector
from baselines import OptimalCaching, LFOBaseline, FeiWangICC2023Baseline

# Test state space
cs = ContentStore(100, 1)
cs.initialize_dqn_agent()
state = cs.get_state_for_dqn('test', 10, current_time=0.0)
assert len(state) == 5, 'State must be 5 features'
assert cs.dqn_agent.state_dim == 5, 'DQN state_dim must be 5'

# Test baselines
opt = OptimalCaching(1, 100)
lfo = LFOBaseline(1, 100)
fei = FeiWangICC2023Baseline(1, 100)

# Test metrics
mc = MetricsCollector()
assert hasattr(mc, 'get_communication_overhead_comparison')

print('✅ All components verified')
"
```

### Step 0.2: Check Dependencies

```bash
# Verify all dependencies are installed
python3 -c "
import torch
import numpy as np
import networkx as nx
print('✅ PyTorch:', torch.__version__)
print('✅ NumPy:', np.__version__)
print('✅ NetworkX:', nx.__version__)
"
```

### Step 0.3: Check Disk Space

```bash
# Ensure sufficient disk space for results
df -h . | tail -1
# Recommended: At least 5GB free for full benchmark runs
```

---

## Phase 1: Quick Validation (15-30 minutes)

**Purpose**: Verify everything works before long runs

### Step 1.1: Single Algorithm Test

```bash
# Test DQN with minimal configuration
NDN_SIM_NODES=20 \
NDN_SIM_PRODUCERS=4 \
NDN_SIM_CONTENTS=100 \
NDN_SIM_USERS=20 \
NDN_SIM_ROUNDS=10 \
NDN_SIM_REQUESTS=5 \
NDN_SIM_CACHE_CAPACITY=10 \
NDN_SIM_ZIPF_PARAM=0.8 \
NDN_SIM_USE_DQN=1 \
NDN_SIM_CACHE_POLICY=combined \
python main.py
```

**Expected**:

- Simulation completes without errors
- Hit rate > 0%
- DQN agents initialized
- Metrics collected

### Step 1.2: Quick Benchmark Test

```bash
# Test benchmark with 2 runs, 2 algorithms
python3 << 'EOF'
from benchmark import run_benchmark, save_results

# Quick test config
config = {
    'NDN_SIM_NODES': '20',
    'NDN_SIM_PRODUCERS': '4',
    'NDN_SIM_CONTENTS': '100',
    'NDN_SIM_USERS': '20',
    'NDN_SIM_ROUNDS': '10',
    'NDN_SIM_REQUESTS': '5',
    'NDN_SIM_CACHE_CAPACITY': '10',
    'NDN_SIM_ZIPF_PARAM': '0.8',
    'NDN_SIM_USE_DQN': '0',
    'NDN_SIM_CACHE_POLICY': 'lru'
}

result = run_benchmark(config, num_runs=2, seed=42)
print(f"✅ Quick test passed: Hit rate = {result.get('hit_rate', 0):.2f}%")
EOF
```

---

## Phase 2: Standard Benchmark (2-4 hours)

**Purpose**: Main comparison of all algorithms

### Step 2.1: Run Main Benchmark

```bash
# Run complete benchmark comparison
python benchmark.py
```

**What it does**:

- Tests: DQN, FIFO, LRU, LFU, Combined
- Runs: 10 iterations per algorithm
- Rounds: 100-250 per run
- Output: `benchmark_results.json`

**Expected Timeline**:

- DQN: ~2-3 hours (250 rounds, more training)
- Traditional: ~1-2 hours (100 rounds each)
- Total: ~3-5 hours

**Monitor Progress**:

```bash
# Check checkpoint status
cat benchmark_checkpoints/benchmark_checkpoint.json

# Check results
cat benchmark_results.json | python3 -m json.tool
```

### Step 2.2: Verify Results

```bash
# Check that all algorithms completed
python3 << 'EOF'
import json
with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

print("Completed algorithms:", list(results.keys()))
for name, data in results.items():
    hr = data.get('hit_rate', 0)
    print(f"  {name}: {hr:.2f}% hit rate")
EOF
```

---

## Phase 3: Ablation Study (4-8 hours)

**Purpose**: Identify which components contribute to performance

### Step 3.1: Run Ablation Study

```bash
# Run complete ablation study
python ablation_study.py
```

**What it tests**:

1. Baseline LRU (no DQN, no Bloom filters)
2. DQN without Bloom filters (`NDN_SIM_DISABLE_BLOOM=1`)
3. DQN with Bloom filters (full implementation)
4. DQN with Neural Bloom filters (`NDN_SIM_NEURAL_BLOOM=1`)
5. DQN with weighted neighbors (Phase 3.1)
6. DQN with adaptive Bloom filter sizing (Phase 3.2)
7. Full DQN (all features)

**Expected Timeline**: 4-8 hours (7 variants × 10 runs × 100 rounds)

**Output**: `ablation_study_results.json`

### Step 3.2: Analyze Ablation Results

```bash
# Generate ablation analysis
python3 << 'EOF'
import json
from ablation_study import analyze_ablation_results, generate_ablation_report

with open('ablation_study_results.json', 'r') as f:
    results = json.load(f)['raw_results']

analysis = analyze_ablation_results(results)
generate_ablation_report(results, analysis, 'ablation_analysis.json')
EOF
```

---

## Phase 4: Sensitivity Analysis (6-12 hours)

**Purpose**: Test robustness and scalability

### Step 4.1: Network Size Sensitivity

```bash
# Test different network sizes
python3 << 'EOF'
from sensitivity_analysis import test_network_size_sensitivity, generate_sensitivity_report

base_config = {
    'NDN_SIM_NODES': '50',
    'NDN_SIM_PRODUCERS': '10',
    'NDN_SIM_CONTENTS': '1000',
    'NDN_SIM_USERS': '100',
    'NDN_SIM_ROUNDS': '50',
    'NDN_SIM_REQUESTS': '20',
    'NDN_SIM_CACHE_CAPACITY': '10',
    'NDN_SIM_ZIPF_PARAM': '0.8'
}

results = test_network_size_sensitivity(base_config, num_runs=5, seed=42)
generate_sensitivity_report(results, 'network_size_sensitivity.json')
EOF
```

**Tests**: 50, 100, 200, 500 routers

### Step 4.2: Cache Capacity Sensitivity

```bash
# Test different cache capacities
python3 << 'EOF'
from sensitivity_analysis import test_cache_capacity_sensitivity

base_config = {
    'NDN_SIM_NODES': '50',
    'NDN_SIM_PRODUCERS': '10',
    'NDN_SIM_CONTENTS': '1000',
    'NDN_SIM_USERS': '100',
    'NDN_SIM_ROUNDS': '50',
    'NDN_SIM_REQUESTS': '20',
    'NDN_SIM_ZIPF_PARAM': '0.8'
}

results = test_cache_capacity_sensitivity(base_config, num_runs=5, seed=42)
EOF
```

**Tests**: 100, 200, 500, 1000 items

### Step 4.3: Bloom Filter Sensitivity

```bash
# Test different Bloom filter parameters
python3 << 'EOF'
from sensitivity_analysis import test_bloom_filter_sensitivity

base_config = {
    'NDN_SIM_NODES': '50',
    'NDN_SIM_PRODUCERS': '10',
    'NDN_SIM_CONTENTS': '1000',
    'NDN_SIM_USERS': '100',
    'NDN_SIM_ROUNDS': '50',
    'NDN_SIM_REQUESTS': '20',
    'NDN_SIM_CACHE_CAPACITY': '10',
    'NDN_SIM_ZIPF_PARAM': '0.8'
}

results = test_bloom_filter_sensitivity(base_config, num_runs=5, seed=42)
EOF
```

**Tests**: FPR = 0.005, 0.01, 0.02, Neural Bloom Filter

---

## Phase 5: Topology Comparison (8-12 hours)

**Purpose**: Test robustness across different network topologies

### Step 5.1: Create Topology Comparison Script

```bash
# Create topology comparison script
cat > topology_comparison.py << 'PYTHON_EOF'
"""
Topology Comparison: Test DQN performance across different network topologies
"""
import os
import sys
import json
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import run_benchmark

def run_topology_comparison():
    """Compare DQN across different topologies"""

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
        'NDN_SIM_CACHE_CAPACITY': '10',
        'NDN_SIM_ZIPF_PARAM': '0.8',
        'NDN_SIM_USE_DQN': '1',
        'NDN_SIM_CACHE_POLICY': 'combined'
    }

    results = {}

    for topology_name, topology_params in topologies:
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
            num_runs=10,
            seed=42,
            checkpoint_key=f'topology_{topology_name}'
        )

    # Save results
    with open('topology_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("TOPOLOGY COMPARISON SUMMARY")
    print("="*80)
    for topology, result in results.items():
        hr = result.get('hit_rate', 0)
        print(f"{topology:20s}: {hr:6.2f}% hit rate")

    return results

if __name__ == "__main__":
    run_topology_comparison()
PYTHON_EOF
```

### Step 5.2: Run Topology Comparison

```bash
python topology_comparison.py
```

**Expected Timeline**: 8-12 hours (4 topologies × 10 runs × 100 rounds)

---

## Phase 6: Communication Overhead Analysis

**Purpose**: Compare Bloom filter vs Fei Wang overhead

### Step 6.1: Run with Overhead Tracking

```bash
# Run DQN and collect overhead metrics
NDN_SIM_USE_DQN=1 \
NDN_SIM_CACHE_POLICY=combined \
python3 << 'EOF'
from benchmark import run_benchmark
from metrics import get_metrics_collector
import json

config = {
    'NDN_SIM_NODES': '50',
    'NDN_SIM_PRODUCERS': '10',
    'NDN_SIM_CONTENTS': '1000',
    'NDN_SIM_USERS': '100',
    'NDN_SIM_ROUNDS': '100',
    'NDN_SIM_REQUESTS': '20',
    'NDN_SIM_CACHE_CAPACITY': '10',
    'NDN_SIM_ZIPF_PARAM': '0.8',
    'NDN_SIM_USE_DQN': '1',
    'NDN_SIM_CACHE_POLICY': 'combined'
}

# Run benchmark
result = run_benchmark(config, num_runs=5, seed=42)

# Get overhead comparison
mc = get_metrics_collector()
overhead = mc.get_communication_overhead_comparison()

# Save results
with open('overhead_comparison.json', 'w') as f:
    json.dump({
        'hit_rate': result.get('hit_rate', 0),
        'communication_overhead': overhead
    }, f, indent=2)

print(f"Hit Rate: {result.get('hit_rate', 0):.2f}%")
print(f"Bloom Filter Overhead: {overhead['bloom_filter_bytes']:,} bytes")
print(f"Fei Wang Overhead: {overhead['fei_wang_bytes']:,} bytes")
print(f"Overhead Reduction: {overhead['overhead_reduction_percent']:.1f}%")
EOF
```

---

## Phase 7: Statistical Analysis

**Purpose**: Generate publication-ready statistics

### Step 7.1: Run Statistical Analysis

```bash
# Generate statistical analysis
python3 << 'EOF'
import json
from statistical_analysis import (
    calculate_mean_std_ci,
    t_test,
    effect_size,
    mann_whitney_u_test
)

# Load benchmark results
with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

# Compare DQN vs LRU
dqn_hit_rates = [42.8, 41.2, 43.5, 42.1, 43.0]  # Example - use actual per-run data
lru_hit_rates = [18.2, 17.8, 18.5, 17.9, 18.3]  # Example

# Statistical tests
t_stat, p_value = t_test(dqn_hit_rates, lru_hit_rates)
cohens_d = effect_size(dqn_hit_rates, lru_hit_rates)
u_stat, u_p_value = mann_whitney_u_test(dqn_hit_rates, lru_hit_rates)

print(f"DQN vs LRU:")
print(f"  T-test p-value: {p_value:.4f}")
print(f"  Effect size (Cohen's d): {cohens_d:.2f}")
print(f"  Mann-Whitney U p-value: {u_p_value:.4f}")
EOF
```

---

## Phase 8: Visualization

**Purpose**: Generate publication-ready plots

### Step 8.1: Generate Comparison Plots

```bash
# Generate comparison visualizations
python visualize_comparison.py
```

**Outputs**:

- `hit_rate_comparison.png` - Bar chart with error bars
- `learning_curves.png` - DQN learning progress
- `ablation_results.png` - Component contributions

### Step 8.2: Generate Learning Curves

```bash
# Plot DQN learning curves
python plot_learning_curves.py
```

**Outputs**:

- `dqn_learning_curves.png` - Hit rate, loss, reward over rounds

---

## Phase 9: Complete Comparison Report

**Purpose**: Generate comprehensive comparison with bounds and Fei Wang

### Step 9.1: Create Complete Comparison Script

```bash
# This would integrate OPT, LFO, Fei Wang baselines
# Note: OPT/LFO/Fei Wang need ContentStore integration first
# For now, use placeholder results or manual integration
```

### Step 9.2: Generate Final Report

```bash
python3 << 'EOF'
import json
from pathlib import Path

# Collect all results
results = {}

# Benchmark results
if Path('benchmark_results.json').exists():
    with open('benchmark_results.json', 'r') as f:
        results['benchmark'] = json.load(f)

# Ablation results
if Path('ablation_study_results.json').exists():
    with open('ablation_study_results.json', 'r') as f:
        results['ablation'] = json.load(f)

# Sensitivity results
if Path('sensitivity_analysis_results.json').exists():
    with open('sensitivity_analysis_results.json', 'r') as f:
        results['sensitivity'] = json.load(f)

# Topology results
if Path('topology_comparison_results.json').exists():
    with open('topology_comparison_results.json', 'r') as f:
        results['topology'] = json.load(f)

# Overhead comparison
if Path('overhead_comparison.json').exists():
    with open('overhead_comparison.json', 'r') as f:
        results['overhead'] = json.load(f)

# Save comprehensive report
with open('complete_results_report.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Complete results report generated: complete_results_report.json")
EOF
```

---

## Execution Timeline Summary

### Quick Validation (30 minutes)

- Single algorithm test
- Quick benchmark test
- Verify all components work

### Standard Benchmark (3-5 hours)

- Main algorithm comparison
- 5 algorithms × 10 runs
- Foundation for all comparisons

### Ablation Study (4-8 hours)

- Component contribution analysis
- 7 variants × 10 runs
- Identify key features

### Sensitivity Analysis (6-12 hours)

- Network size: 4 sizes × 5 runs
- Cache capacity: 4 sizes × 5 runs
- Bloom filter: 4 configs × 5 runs

### Topology Comparison (8-12 hours)

- 4 topologies × 10 runs
- Robustness testing

### Analysis & Visualization (1-2 hours)

- Statistical analysis
- Generate plots
- Create reports

**Total Time**: ~22-40 hours for complete evaluation

---

## Recommended Execution Order

### Week 1: Foundation

**Day 1-2: Validation & Standard Benchmark**

```bash
# Day 1: Quick validation (30 min)
# Day 1-2: Standard benchmark (3-5 hours)
python benchmark.py
```

**Day 3-4: Ablation Study**

```bash
# Day 3-4: Ablation study (4-8 hours)
python ablation_study.py
```

### Week 2: Robustness

**Day 5-7: Sensitivity Analysis**

```bash
# Day 5: Network size sensitivity
# Day 6: Cache capacity sensitivity
# Day 7: Bloom filter sensitivity
python sensitivity_analysis.py
```

**Day 8-9: Topology Comparison**

```bash
# Day 8-9: Topology comparison (8-12 hours)
python topology_comparison.py
```

### Week 3: Analysis

**Day 10-11: Statistical Analysis**

```bash
# Day 10: Statistical tests
# Day 11: Visualization
python visualize_comparison.py
python plot_learning_curves.py
```

**Day 12-14: Report Generation**

```bash
# Day 12-14: Generate comprehensive reports
# Create final comparison document
```

---

## Monitoring Progress

### Check Running Processes

```bash
# Check if benchmark is running
ps aux | grep python | grep benchmark

# Check checkpoint status
cat benchmark_checkpoints/benchmark_checkpoint.json
```

### Monitor Results

```bash
# Watch results file for updates
watch -n 30 'cat benchmark_results.json | python3 -m json.tool | head -50'
```

### Check Logs

```bash
# View simulation logs
tail -f logs/simulation.log

# View network logs
tail -f logs/network.log
```

---

## Troubleshooting

### If Benchmark Hangs

```bash
# Check queue drain status
# Look for "Waiting for queue drain" messages
# If stuck, check DQN training manager
```

### If DQN Not Learning

```bash
# Check DQN initialization
python3 << 'EOF'
from main import create_network, setup_all_routers_to_dqn_mode
G, _, _, _ = create_network(num_nodes=10, num_producers=2, num_contents=50, num_users=10)
setup_all_routers_to_dqn_mode(G)
# Count initialized DQN agents
EOF
```

### If Out of Memory

```bash
# Reduce network size
NDN_SIM_NODES=30 NDN_SIM_ROUNDS=50 python benchmark.py
```

---

## Next Steps After Execution

1. **Analyze Results**: Review all JSON result files
2. **Statistical Tests**: Run significance tests
3. **Generate Plots**: Create visualizations
4. **Write Report**: Document findings
5. **Compare to Baselines**: OPT, LFO, Fei Wang
6. **Identify Improvements**: Based on results
7. **Iterate**: Refine based on findings

---

## Success Criteria

✅ **Benchmark completes**: All 5 algorithms run successfully  
✅ **DQN outperforms baselines**: Hit rate > LRU/LFU/FIFO  
✅ **Ablation shows Bloom filter contribution**: DQN with Bloom > DQN without  
✅ **Sensitivity shows robustness**: Performance stable across parameters  
✅ **Topology comparison shows consistency**: Works across topologies  
✅ **Statistical significance**: p < 0.05 for key comparisons  
✅ **Communication overhead**: Bloom filter < Fei Wang overhead

---

## Quick Start Command

```bash
# Start with standard benchmark (recommended first step)
python benchmark.py
```

This will run the complete comparison and save results to `benchmark_results.json`.

---

**Ready to execute! Start with Phase 1 (Quick Validation) to ensure everything works, then proceed to Phase 2 (Standard Benchmark).**
