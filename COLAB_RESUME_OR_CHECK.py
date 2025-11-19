# ============================================================================
# RESUME OR CHECK BENCHMARK STATUS IN COLAB
# Run this cell to check if benchmark can be resumed or view results
# ============================================================================

import os
import json
import time
from pathlib import Path

print("="*70)
print("BENCHMARK STATUS CHECK")
print("="*70)

# Navigate to project directory
if not os.path.exists('benchmark_checkpoints'):
    if os.path.exists('/content/ndn-nwsim-Sai_work'):
        os.chdir('/content/ndn-nwsim-Sai_work')
        print(f"📁 Changed to: {os.getcwd()}\n")
    elif os.path.exists('/content'):
        # Try to find the directory
        for item in os.listdir('/content'):
            if 'ndn-nwsim' in item:
                os.chdir(f'/content/{item}')
                print(f"📁 Changed to: {os.getcwd()}\n")
                break

results_file = Path('benchmark_checkpoints/benchmark_results.json')
checkpoint_file = Path('benchmark_checkpoints/benchmark_checkpoint.json')

# Check 1: Is benchmark complete?
if results_file.exists():
    print("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n{'Policy':<20} {'Hit Rate':<15} {'Std Dev':<15} {'Runs':<10}")
    print("-"*70)
    
    for name, result in sorted(results.items()):
        hit_rate = result.get('hit_rate', 0)
        std = result.get('hit_rate_std', 0)
        runs = result.get('num_runs', 0)
        print(f"{name:<20} {hit_rate:6.2f}%      {std:6.2f}%      {runs:<10}")
    
    print("\n" + "="*70)
    print("🎉 All done! You can download results or analyze them further.")
    print("="*70)

# Check 2: Is there a checkpoint to resume from?
elif checkpoint_file.exists():
    print("⏳ BENCHMARK WAS INTERRUPTED - CAN RESUME")
    print("="*70)
    
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    algorithm = checkpoint.get('algorithm', 'unknown')
    completed = checkpoint.get('completed_runs', 0)
    total = checkpoint.get('total_runs', 0)
    timestamp = checkpoint.get('timestamp', 0)
    
    progress_pct = (completed * 100 // total) if total > 0 else 0
    
    print(f"\n📊 Status:")
    print(f"   Algorithm: {algorithm.upper()}")
    print(f"   Progress: {completed}/{total} runs ({progress_pct}%)")
    
    if timestamp:
        elapsed = time.time() - timestamp
        print(f"   Last update: {int(elapsed // 60)}m {int(elapsed % 60)}s ago")
    
    # Check partial results
    if 'results' in checkpoint and 'partial_results' in checkpoint['results']:
        partial = checkpoint['results']['partial_results']
        if partial:
            hit_rates = [r.get('hit_rate', 0) for r in partial]
            avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
            print(f"   Partial avg hit rate: {avg_hit_rate:.2f}% ({len(partial)} runs)")
    
    print("\n" + "="*70)
    print("💡 TO RESUME:")
    print("="*70)
    print("The benchmark.py script supports resume functionality.")
    print("However, for simplicity, you can:")
    print("  1. Re-run the benchmark cell (it will start fresh)")
    print("  2. Or manually resume by modifying benchmark.py to load checkpoint")
    print("\n⚠️  Note: Re-running will start from the beginning of the current algorithm.")
    print("="*70)

# Check 3: No checkpoint found
else:
    print("⚠️  NO BENCHMARK FOUND")
    print("="*70)
    print("Either:")
    print("  1. Benchmark hasn't started yet")
    print("  2. Benchmark crashed before creating any checkpoint")
    print("  3. You're in the wrong directory")
    
    print("\n📁 Current directory:", os.getcwd())
    print("\n💡 TO START BENCHMARK:")
    print("  1. Make sure runtime is connected (check top right)")
    print("  2. Run your benchmark setup cell")
    print("  3. Run the benchmark execution cell")
    print("="*70)

# Additional diagnostics
print("\n" + "="*70)
print("RUNTIME STATUS")
print("="*70)
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU - will use CPU")
except:
    print("⚠️  Could not check GPU status")

print(f"\n📁 Working directory: {os.getcwd()}")
print(f"📂 Checkpoint dir exists: {Path('benchmark_checkpoints').exists()}")
print("="*70)

