# Fix Cell Blocking Issues in Colab

## 🔍 Problem: One Cell Waiting for Another

This happens when:
- Cell 1 is running the benchmark (long process)
- Cell 2 is trying to read results (waiting for Cell 1 to finish)
- Both cells are blocking each other

---

## ✅ Solution 1: Check Which Cell is Running

### See Active Cells:
- Look for **spinning icons** ⏳ next to cell numbers
- Running cells show: `[*]` instead of `[1]`, `[2]`, etc.
- Check the **Runtime** menu → **Manage sessions** to see active processes

### Stop a Cell:
- Click the **stop button** (⏹️) next to the running cell
- Or press `Ctrl+M` then `.` (period)
- Or go to **Runtime** → **Interrupt execution**

---

## ✅ Solution 2: Run Cells Independently

### If Benchmark is Running:
**Don't try to read results while benchmark is running!**

Instead, use a **separate cell** to check progress:

```python
# This cell checks progress WITHOUT blocking
import json
import os
import time

# Check every 5 seconds
for i in range(12):  # Check for 1 minute
    if os.path.exists('benchmark_checkpoints/benchmark_checkpoint.json'):
        with open('benchmark_checkpoints/benchmark_checkpoint.json', 'r') as f:
            checkpoint = json.load(f)
        algorithm = checkpoint.get('algorithm', 'unknown')
        completed = checkpoint.get('completed_runs', 0)
        total = checkpoint.get('total_runs', 0)
        print(f"{algorithm}: {completed}/{total} runs ({completed/total*100:.1f}%)")
        
        if completed >= total:
            print("✅ Algorithm completed!")
            break
    else:
        print("⏳ Waiting for benchmark to start...")
    
    time.sleep(5)  # Wait 5 seconds before next check
```

**This cell will finish quickly** and won't block the benchmark.

---

## ✅ Solution 3: Check if Benchmark is Actually Running

### Verify Benchmark Process:

```python
# Check if benchmark.py is running
import subprocess
import os

# Check running Python processes
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
if 'benchmark.py' in result.stdout:
    print("✅ Benchmark is running!")
    # Show relevant lines
    for line in result.stdout.split('\n'):
        if 'benchmark.py' in line:
            print(line)
else:
    print("❌ Benchmark is NOT running")
```

### Check Log File:

```python
# View last few lines of log (non-blocking)
!tail -20 benchmark_run.log
```

---

## ✅ Solution 4: Run Benchmark in Background (Advanced)

### Option A: Use `&` to run in background

```python
# Run benchmark in background
import subprocess
import os

# Set environment variables
os.environ['NDN_SIM_USE_DQN'] = '1'
# ... (other configs)

# Run in background
process = subprocess.Popen(['python', 'benchmark.py'], 
                          stdout=open('benchmark_run.log', 'w'),
                          stderr=subprocess.STDOUT)
print(f"Benchmark started with PID: {process.pid}")
print("You can now run other cells while benchmark runs!")
```

### Option B: Use threading (for monitoring)

```python
import threading
import subprocess
import os

def run_benchmark():
    os.environ['NDN_SIM_USE_DQN'] = '1'
    # ... (set configs)
    subprocess.run(['python', 'benchmark.py'])

# Start benchmark in background thread
thread = threading.Thread(target=run_benchmark, daemon=True)
thread.start()
print("✅ Benchmark started in background!")
print("You can now check progress in other cells")
```

---

## ✅ Solution 5: Proper Cell Organization

### Recommended Cell Structure:

**Cell 1: Install dependencies**
```python
!pip install networkx numpy torch scipy matplotlib pandas scikit-learn dill bitarray hdbscan tensorflow mmh3
```

**Cell 2: Upload files**
```python
from google.colab import files
uploaded = files.upload()
```

**Cell 3: Configure**
```python
import os
os.environ['NDN_SIM_USE_DQN'] = '1'
# ... (other configs)
```

**Cell 4: Run benchmark** (This will take 10-20 minutes)
```python
!python benchmark.py
```

**Cell 5: Check results** (Run this AFTER Cell 4 finishes)
```python
import json
with open('benchmark_checkpoints/benchmark_results.json', 'r') as f:
    results = json.load(f)
# ... (view results)
```

---

## 🚨 Common Issues and Fixes

### Issue 1: "File is locked" or "Permission denied"
**Fix**: Wait for the benchmark to finish, or stop it first

### Issue 2: Cell shows `[*]` forever
**Fix**: 
- Check if benchmark is actually running: `!ps aux | grep benchmark`
- Check log: `!tail -50 benchmark_run.log`
- If stuck, stop cell and restart

### Issue 3: Can't read results while benchmark runs
**Fix**: This is normal! Wait for benchmark to complete, or check checkpoint file instead

### Issue 4: Multiple cells trying to write same file
**Fix**: Only run one benchmark at a time. Stop other cells first.

---

## 💡 Best Practices

1. **Run benchmark in ONE cell** - Don't split it across cells
2. **Wait for completion** - Don't try to read results while running
3. **Use checkpoint file** - Check progress via checkpoint, not results file
4. **One process at a time** - Don't run multiple benchmarks simultaneously
5. **Monitor with separate cell** - Use a different cell to check progress

---

## 🔍 Quick Diagnostic

Run this to see what's happening:

```python
import os
import subprocess

print("="*60)
print("DIAGNOSTIC CHECK")
print("="*60)

# Check if benchmark is running
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
if 'benchmark.py' in result.stdout:
    print("✅ Benchmark process is running")
else:
    print("❌ No benchmark process found")

# Check files
print("\nFile Status:")
print(f"  Results file: {'✅ Exists' if os.path.exists('benchmark_checkpoints/benchmark_results.json') else '❌ Not found'}")
print(f"  Checkpoint file: {'✅ Exists' if os.path.exists('benchmark_checkpoints/benchmark_checkpoint.json') else '❌ Not found'}")
print(f"  Log file: {'✅ Exists' if os.path.exists('benchmark_run.log') else '❌ Not found'}")

# Check log tail
if os.path.exists('benchmark_run.log'):
    print("\nLast 5 lines of log:")
    !tail -5 benchmark_run.log
```

---

## 🎯 Quick Fix Summary

**If a cell is stuck:**
1. Click the **stop button** (⏹️) on the running cell
2. Check if benchmark is actually running: `!ps aux | grep benchmark`
3. If benchmark is running, **wait for it to finish** before reading results
4. Use a **separate cell** to check progress (checkpoint file, not results file)

**If you want to check progress while benchmark runs:**
- Use the checkpoint file (not results file)
- Run progress check in a **separate cell**
- The progress check cell will finish quickly and won't block

---

**Most common issue**: Trying to read `benchmark_results.json` while benchmark is still running. **Solution**: Wait for benchmark to finish, or check `benchmark_checkpoint.json` for progress instead! ✅

