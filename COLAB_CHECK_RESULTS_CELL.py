# ============================================================================
# CHECK BENCHMARK RESULTS IN COLAB
# Run this cell anytime to check if benchmark completed and view results
# ============================================================================

import os
import json
import time
from pathlib import Path
from datetime import datetime

print("="*70)
print("CHECKING BENCHMARK STATUS")
print("="*70)

# Navigate to project directory if needed
if not os.path.exists('benchmark_checkpoints'):
    if os.path.exists('/content/ndn-nwsim-Sai_work'):
        os.chdir('/content/ndn-nwsim-Sai_work')
        print(f"📁 Changed to: {os.getcwd()}")
    else:
        print("⚠️  Project directory not found. Make sure you've cloned the repo.")
        print("   Run: !git clone https://github.com/tsaipraveen99/ndn-nwsim-Sai_work.git")
        print("   Then: %cd ndn-nwsim-Sai_work")

results_file = Path('benchmark_checkpoints/benchmark_results.json')
checkpoint_file = Path('benchmark_checkpoints/benchmark_checkpoint.json')

# Check if benchmark completed
if results_file.exists():
    print("\n✅ BENCHMARK COMPLETED!")
    print("="*70)
    
    try:
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
        print("DETAILED STATISTICS")
        print("="*70)
        
        for name, result in sorted(results.items()):
            print(f"\n📊 {name.upper()}:")
            print(f"   Hit Rate: {result.get('hit_rate', 0):.2f}% ± {result.get('hit_rate_std', 0):.2f}%")
            print(f"   Cache Hits: {result.get('cache_hits', 0):.0f}")
            print(f"   Cached Items: {result.get('cached_items', 0):.0f}")
            print(f"   Avg Cache Utilization: {result.get('avg_cache_utilization', 0):.2f}%")
            print(f"   Latency: {result.get('latency_mean', 0):.4f}")
            print(f"   Runs Completed: {result.get('num_runs', 0)}")
            
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        import traceback
        traceback.print_exc()

# Check if still running
elif checkpoint_file.exists():
    print("\n⏳ BENCHMARK STILL RUNNING...")
    print("="*70)
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        algorithm = checkpoint.get('algorithm', 'unknown')
        completed = checkpoint.get('completed_runs', 0)
        total = checkpoint.get('total_runs', 0)
        timestamp = checkpoint.get('timestamp', 0)
        
        if timestamp:
            elapsed = time.time() - timestamp
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            elapsed_str = "unknown"
        
        progress_pct = (completed * 100 // total) if total > 0 else 0
        
        print(f"\n📈 Current Algorithm: {algorithm.upper()}")
        print(f"📊 Progress: {completed}/{total} runs ({progress_pct}%)")
        print(f"⏱️  Elapsed: {elapsed_str}")
        
        # Show partial results if available
        if 'results' in checkpoint and 'partial_results' in checkpoint['results']:
            partial = checkpoint['results']['partial_results']
            if partial:
                print(f"\n📋 Partial Results ({len(partial)} runs):")
                hit_rates = [r.get('hit_rate', 0) for r in partial]
                avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
                print(f"   Average Hit Rate (so far): {avg_hit_rate:.2f}%")
        
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\n⚠️  NO BENCHMARK FOUND")
    print("="*70)
    print("Either:")
    print("  1. Benchmark hasn't started yet")
    print("  2. Benchmark crashed before creating checkpoint")
    print("  3. You're in the wrong directory")
    print("\n💡 Make sure you're in the project directory:")
    print("   %cd /content/ndn-nwsim-Sai_work")

print("\n" + "="*70)
print("💡 TIP: Run this cell anytime to check progress!")
print("="*70)

