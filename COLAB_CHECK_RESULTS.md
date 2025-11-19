# How to Check Benchmark Results in Colab

## 📊 Where Results Are Saved

After the benchmark completes, results are saved to:
- **`benchmark_checkpoints/benchmark_results.json`** - Final results for all algorithms
- **`benchmark_checkpoints/benchmark_checkpoint.json`** - Current progress (if still running)

---

## ✅ Method 1: View Results in Colab (Easiest)

### Step 1: Check if Benchmark Completed

```python
import os
import json

# Check if results file exists
if os.path.exists('benchmark_checkpoints/benchmark_results.json'):
    print("✅ Results file found!")
else:
    print("⏳ Benchmark may still be running...")
    # Check checkpoint for progress
    if os.path.exists('benchmark_checkpoints/benchmark_checkpoint.json'):
        with open('benchmark_checkpoints/benchmark_checkpoint.json', 'r') as f:
            checkpoint = json.load(f)
        print(f"Current algorithm: {checkpoint.get('algorithm', 'unknown')}")
        print(f"Progress: {checkpoint.get('completed_runs', 0)}/{checkpoint.get('total_runs', 0)} runs")
```

### Step 2: View Results

```python
import json

# Load and display results
with open('benchmark_checkpoints/benchmark_results.json', 'r') as f:
    results = json.load(f)

print("="*80)
print("BENCHMARK RESULTS")
print("="*80)
print(f"\n{'Policy':<15} {'Hit Rate':<15} {'Cache Hits':<15} {'Cached Items':<15}")
print("-"*80)

for name, result in results.items():
    if 'hit_rate' in result:
        hit_rate = result['hit_rate']
        cache_hits = result.get('cache_hits', 0)
        cached_items = result.get('cached_items', 0)
        print(f"{name:<15} {hit_rate:.2f}%      {cache_hits:<15.0f} {cached_items:<15.0f}")

print("\n" + "="*80)
```

### Step 3: Detailed Statistics

```python
import json

with open('benchmark_checkpoints/benchmark_results.json', 'r') as f:
    results = json.load(f)

# Show detailed stats for each algorithm
for name, result in results.items():
    print(f"\n{'='*60}")
    print(f"{name} - Detailed Results")
    print(f"{'='*60}")
    
    if 'hit_rate' in result:
        print(f"Hit Rate: {result['hit_rate']:.4f}%")
        if 'hit_rate_std' in result:
            print(f"  Standard Deviation: {result['hit_rate_std']:.4f}%")
        if 'hit_rate_ci_lower' in result:
            print(f"  95% CI: [{result['hit_rate_ci_lower']:.4f}%, {result['hit_rate_ci_upper']:.4f}%]")
    
    print(f"Cache Hits: {result.get('cache_hits', 0):.0f}")
    print(f"Nodes Traversed: {result.get('nodes_traversed', 0):.0f}")
    print(f"Cached Items: {result.get('cached_items', 0):.0f}")
    print(f"Number of Runs: {result.get('num_runs', 0)}")
```

---

## 📥 Method 2: Download Results

### Download JSON File

```python
from google.colab import files

# Download results
files.download('benchmark_checkpoints/benchmark_results.json')
print("✅ Results downloaded!")
```

### Download All Results

```python
from google.colab import files
import os

# Download results and log
if os.path.exists('benchmark_checkpoints/benchmark_results.json'):
    files.download('benchmark_checkpoints/benchmark_results.json')
if os.path.exists('benchmark_run.log'):
    files.download('benchmark_run.log')
print("✅ All files downloaded!")
```

---

## 📈 Method 3: Create Visualizations

### Plot Hit Rate Comparison

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('benchmark_checkpoints/benchmark_results.json', 'r') as f:
    results = json.load(f)

# Extract data
policies = []
hit_rates = []
errors = []

for name, result in results.items():
    if 'hit_rate' in result:
        policies.append(name)
        hit_rates.append(result['hit_rate'])
        # Use std as error bar if available
        errors.append(result.get('hit_rate_std', 0))

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(policies, hit_rates, yerr=errors, capsize=5, alpha=0.7)
plt.xlabel('Caching Policy')
plt.ylabel('Hit Rate (%)')
plt.title('Cache Hit Rate Comparison')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, rate) in enumerate(zip(bars, hit_rates)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{rate:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Save plot
plt.savefig('hit_rate_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved as hit_rate_comparison.png")
```

### Comparison Table

```python
import json
import pandas as pd

# Load results
with open('benchmark_checkpoints/benchmark_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame
df_data = []
for name, result in results.items():
    if 'hit_rate' in result:
        df_data.append({
            'Policy': name,
            'Hit Rate (%)': f"{result['hit_rate']:.2f}",
            'Cache Hits': result.get('cache_hits', 0),
            'Cached Items': result.get('cached_items', 0),
            'Nodes Traversed': result.get('nodes_traversed', 0),
            'Runs': result.get('num_runs', 0)
        })

df = pd.DataFrame(df_data)
print("\n" + "="*80)
print("RESULTS SUMMARY TABLE")
print("="*80)
print(df.to_string(index=False))
```

---

## 🔍 Method 4: Check Progress While Running

### Monitor Live Progress

```python
import json
import time
import os

# Monitor checkpoint while benchmark is running
while True:
    if os.path.exists('benchmark_checkpoints/benchmark_checkpoint.json'):
        with open('benchmark_checkpoints/benchmark_checkpoint.json', 'r') as f:
            checkpoint = json.load(f)
        
        algorithm = checkpoint.get('algorithm', 'unknown')
        completed = checkpoint.get('completed_runs', 0)
        total = checkpoint.get('total_runs', 0)
        
        print(f"\r{algorithm}: {completed}/{total} runs completed", end='')
        
        if completed >= total:
            print("\n✅ Algorithm completed!")
            break
    else:
        print("⏳ Waiting for benchmark to start...")
    
    time.sleep(5)  # Check every 5 seconds
```

### View Log File

```python
# View last 50 lines of log
!tail -50 benchmark_run.log

# Or view in real-time (if benchmark is running)
# !tail -f benchmark_run.log
```

---

## 📋 Complete Results Viewer

Copy this complete code into a Colab cell:

```python
import json
import os
import pandas as pd

print("="*80)
print("BENCHMARK RESULTS VIEWER")
print("="*80)

# Check if results exist
if not os.path.exists('benchmark_checkpoints/benchmark_results.json'):
    print("\n❌ Results file not found.")
    print("   Benchmark may still be running or hasn't started yet.")
    
    # Check checkpoint
    if os.path.exists('benchmark_checkpoints/benchmark_checkpoint.json'):
        with open('benchmark_checkpoints/benchmark_checkpoint.json', 'r') as f:
            checkpoint = json.load(f)
        print(f"\n📊 Current Progress:")
        print(f"   Algorithm: {checkpoint.get('algorithm', 'unknown')}")
        print(f"   Runs: {checkpoint.get('completed_runs', 0)}/{checkpoint.get('total_runs', 0)}")
    exit()

# Load results
with open('benchmark_checkpoints/benchmark_results.json', 'r') as f:
    results = json.load(f)

print(f"\n✅ Found results for {len(results)} algorithm(s)\n")

# Summary table
print("SUMMARY TABLE")
print("-"*80)
print(f"{'Policy':<15} {'Hit Rate':<15} {'Cache Hits':<15} {'Cached Items':<15}")
print("-"*80)

for name, result in results.items():
    if 'hit_rate' in result:
        print(f"{name:<15} {result['hit_rate']:.2f}%      "
              f"{result.get('cache_hits', 0):<15.0f} "
              f"{result.get('cached_items', 0):<15.0f}")

# Detailed view
print("\n" + "="*80)
print("DETAILED STATISTICS")
print("="*80)

for name, result in results.items():
    print(f"\n{name}:")
    if 'hit_rate' in result:
        print(f"  Hit Rate: {result['hit_rate']:.4f}%")
        if 'hit_rate_std' in result:
            print(f"    Std Dev: {result['hit_rate_std']:.4f}%")
        if 'hit_rate_ci_lower' in result:
            print(f"    95% CI: [{result['hit_rate_ci_lower']:.4f}%, "
                  f"{result['hit_rate_ci_upper']:.4f}%]")
    print(f"  Cache Hits: {result.get('cache_hits', 0):.0f}")
    print(f"  Nodes Traversed: {result.get('nodes_traversed', 0):.0f}")
    print(f"  Cached Items: {result.get('cached_items', 0):.0f}")
    print(f"  Runs: {result.get('num_runs', 0)}")

# Best performer
if results:
    best = max(results.items(), key=lambda x: x[1].get('hit_rate', 0))
    print(f"\n🏆 Best Performer: {best[0]} ({best[1].get('hit_rate', 0):.2f}% hit rate)")

print("\n" + "="*80)
```

---

## 💾 Save to Google Drive

```python
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Save results to Drive
import shutil
shutil.copy('benchmark_checkpoints/benchmark_results.json', 
            '/content/drive/MyDrive/ndn_benchmark_results.json')
print("✅ Results saved to Google Drive!")
```

---

## 🎯 Quick Commands

**View results:**
```python
!cat benchmark_checkpoints/benchmark_results.json | python -m json.tool
```

**Check if done:**
```python
import os
print("Done!" if os.path.exists('benchmark_checkpoints/benchmark_results.json') else "Still running...")
```

**Download:**
```python
from google.colab import files
files.download('benchmark_checkpoints/benchmark_results.json')
```

---

**That's it!** Use any of these methods to check your results! 📊

