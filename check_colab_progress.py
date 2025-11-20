#!/usr/bin/env python3
"""
Quick diagnostic script to check benchmark progress in Colab
Run this in a separate cell while benchmark.py is running
"""

import json
import os
from pathlib import Path
from datetime import datetime

print("="*70)
print("BENCHMARK PROGRESS CHECK")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check checkpoint file
checkpoint_file = Path('benchmark_checkpoints/benchmark_checkpoint.json')
if checkpoint_file.exists():
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        print("📊 CURRENT STATUS:")
        print("-"*70)
        algorithm = checkpoint.get('algorithm', 'unknown')
        completed = checkpoint.get('completed_runs', 0)
        total = checkpoint.get('total_runs', 0)
        progress_pct = (completed * 100 // total) if total > 0 else 0
        
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Progress: {completed}/{total} runs ({progress_pct}%)")
        
        # Check if there are partial results
        if 'results' in checkpoint:
            results = checkpoint['results']
            if algorithm in results:
                algo_results = results[algorithm]
                if 'hit_rates' in algo_results and len(algo_results['hit_rates']) > 0:
                    avg_hit_rate = sum(algo_results['hit_rates']) / len(algo_results['hit_rates'])
                    print(f"Current Average Hit Rate: {avg_hit_rate:.2f}%")
                    print(f"Completed Runs: {len(algo_results['hit_rates'])}")
        
        # Check last update time
        if 'last_update' in checkpoint:
            print(f"Last Update: {checkpoint['last_update']}")
        
        print()
    except Exception as e:
        print(f"⚠️  Error reading checkpoint: {e}")
        print()
else:
    print("⚠️  No checkpoint file found. Benchmark may not have started yet.")
    print()

# Check results file
results_file = Path('benchmark_checkpoints/benchmark_results.json')
if results_file.exists():
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("✅ FINAL RESULTS:")
        print("-"*70)
        print(f"{'Policy':<20} {'Hit Rate':<15} {'Std Dev':<15} {'Runs':<10}")
        print("-"*70)
        
        for name, result in results.items():
            hit_rate = result.get('hit_rate', 0)
            std = result.get('hit_rate_std', 0)
            runs = result.get('num_runs', 0)
            print(f"{name:<20} {hit_rate:6.2f}%      {std:6.2f}%      {runs:<10}")
        
        print()
        print("🎉 BENCHMARK COMPLETED!")
    except Exception as e:
        print(f"⚠️  Error reading results: {e}")
        print()
else:
    print("ℹ️  Results file not found yet. Benchmark still running or not started.")
    print()

# Check for log files that might indicate issues
print("🔍 CHECKING FOR ISSUES:")
print("-"*70)

# Check if there are any recent error patterns we can detect
# (This is a simple check - in Colab you'd need to check the output)

print("💡 TIPS:")
print("  - If progress is stuck, check the main output for 'TIMEOUT' messages")
print("  - If you see 'Queue drain: TIMEOUT', workers may be stuck")
print("  - If you see 'Message processing TIMEOUT', specific messages are taking too long")
print("  - Try reducing network size (NDN_SIM_NODES) or rounds (NDN_SIM_ROUNDS)")
print("  - Check GPU memory usage if using DQN")
print()

print("="*70)

