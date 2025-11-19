#!/usr/bin/env python3
"""
Quick script to check benchmark progress in Colab
Run this in a separate cell while the benchmark is running
"""

import json
import os
import time
from datetime import datetime

def check_progress():
    """Check current benchmark progress"""
    checkpoint_file = 'benchmark_checkpoints/benchmark_checkpoint.json'
    results_file = 'benchmark_checkpoints/benchmark_results.json'
    
    print("="*80)
    print("BENCHMARK PROGRESS CHECKER")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if benchmark completed
    if os.path.exists(results_file):
        print("✅ Benchmark COMPLETED!")
        print(f"\n📊 Loading results from {results_file}...\n")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"{'Policy':<20} {'Hit Rate':<15} {'Cache Hits':<15}")
        print("-"*80)
        
        for name, result in results.items():
            if 'hit_rate' in result:
                hit_rate = result['hit_rate']
                cache_hits = result.get('cache_hits', 0)
                print(f"{name:<20} {hit_rate:.2f}%      {cache_hits:<15.0f}")
        
        print("\n" + "="*80)
        return True
    
    # Check if benchmark is running
    if os.path.exists(checkpoint_file):
        print("⏳ Benchmark is RUNNING...\n")
        
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        algorithm = checkpoint.get('algorithm', 'unknown')
        completed = checkpoint.get('completed_runs', 0)
        total = checkpoint.get('total_runs', 0)
        
        print(f"Current Algorithm: {algorithm}")
        print(f"Progress: {completed}/{total} runs ({completed*100//total if total > 0 else 0}%)")
        
        # Estimate time remaining (rough)
        if completed > 0:
            print(f"\n💡 Tip: Each run takes time. Be patient!")
        
        print("\n" + "="*80)
        return False
    
    # No checkpoint or results found
    print("❌ No benchmark found.")
    print("   Make sure you've started the benchmark first.")
    print("\n" + "="*80)
    return None

if __name__ == "__main__":
    check_progress()

