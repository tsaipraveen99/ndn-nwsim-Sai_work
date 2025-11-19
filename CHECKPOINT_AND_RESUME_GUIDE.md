# Checkpoint and Resume Guide

**Status**: ✅ All Execution Scripts Support Checkpointing and Resume

---

## Overview

All execution scripts now support:
- ✅ **Checkpointing**: Save progress after each algorithm/variant completes
- ✅ **Resume Capability**: Automatically resume from last checkpoint if interrupted
- ✅ **Background Execution**: Run in background with nohup/screen/tmux
- ✅ **Incremental Results**: Results saved incrementally (not lost on interruption)

---

## Checkpoint Support by Script

### ✅ benchmark.py

**Checkpoint Level**: Per-algorithm (DQN, FIFO, LRU, etc.)

**How it works**:
- Saves checkpoint after every 2 runs (configurable)
- Saves checkpoint when algorithm completes
- Automatically resumes from last completed run
- Clears checkpoint when algorithm fully completes

**Checkpoint File**: `benchmark_checkpoints/benchmark_checkpoint.json`

**Resume**:
```bash
# Automatically resumes (default)
python benchmark.py

# Start fresh (ignore checkpoints)
python benchmark.py --no-resume

# Clear checkpoint manually
python benchmark.py --clear-checkpoint
```

**Example Checkpoint**:
```json
{
  "algorithm": "DQN",
  "completed_runs": 7,
  "total_runs": 10,
  "timestamp": 1234567890,
  "results": {
    "partial_results": [...],
    "num_completed": 7
  }
}
```

---

### ✅ ablation_study.py

**Checkpoint Level**: Per-variant (7 variants total)

**How it works**:
- Saves checkpoint after each variant completes
- Skips already-completed variants on resume
- Clears checkpoint when all 7 variants complete

**Checkpoint File**: `ablation_study_checkpoint.json`

**Resume**:
```bash
# Automatically resumes (default)
python ablation_study.py

# Start fresh
python ablation_study.py --no-resume
```

**Example Checkpoint**:
```json
{
  "results": {
    "Baseline_LRU": {...},
    "DQN_NoBloom": {...}
  },
  "completed_variants": ["Baseline_LRU", "DQN_NoBloom"]
}
```

---

### ✅ sensitivity_analysis.py

**Checkpoint Level**: Per-test-type (Network Size, Cache Capacity, Bloom Filter)

**How it works**:
- Separate checkpoints for each test type
- Saves after each parameter value completes
- Skips already-completed values on resume

**Checkpoint Files**:
- `sensitivity_network_size_checkpoint.json`
- `sensitivity_cache_capacity_checkpoint.json`
- `sensitivity_bloom_filter_checkpoint.json`

**Resume**:
```bash
# Automatically resumes (default)
python sensitivity_analysis.py

# Start fresh
python sensitivity_analysis.py --no-resume
```

---

### ✅ topology_comparison.py

**Checkpoint Level**: Per-topology (4 topologies)

**How it works**:
- Saves checkpoint after each topology completes
- Skips already-completed topologies on resume
- Clears checkpoint when all 4 topologies complete

**Checkpoint File**: `topology_comparison_checkpoint.json`

**Resume**:
```bash
# Automatically resumes (default)
python topology_comparison.py

# Start fresh
python topology_comparison.py --no-resume
```

---

### ✅ complete_comparison.py

**Checkpoint Level**: Per-algorithm (8 algorithms total)

**How it works**:
- Saves checkpoint after each algorithm completes
- Skips already-completed algorithms on resume
- Clears checkpoint when all algorithms complete

**Checkpoint File**: `complete_comparison_checkpoint.json`

**Resume**:
```bash
# Automatically resumes (default)
python complete_comparison.py

# Start fresh
python complete_comparison.py --no-resume
```

---

## Background Execution

### Method 1: Using nohup (Recommended)

```bash
# Run in background with nohup
nohup python benchmark.py > benchmark.log 2>&1 &

# Or use the helper script
./run_background.sh benchmark.py
```

**Monitor**:
```bash
# Watch output
tail -f benchmark.log

# Check if running
ps aux | grep benchmark.py

# Check PID
cat benchmark.py.pid
```

**Stop**:
```bash
# Find PID
ps aux | grep benchmark.py

# Kill process
kill <PID>

# Or use pkill
pkill -f benchmark.py
```

---

### Method 2: Using screen

```bash
# Start screen session
screen -S benchmark

# Run script
python benchmark.py

# Detach: Ctrl+A, then D

# Reattach later
screen -r benchmark
```

---

### Method 3: Using tmux

```bash
# Start tmux session
tmux new -s benchmark

# Run script
python benchmark.py

# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t benchmark
```

---

## Resume After Interruption

### Automatic Resume

All scripts automatically resume from checkpoints:

```bash
# If interrupted, just run again - it will resume automatically
python benchmark.py
```

**What happens**:
1. Script checks for checkpoint file
2. If found, loads completed runs/variants
3. Skips already-completed items
4. Continues from where it left off

---

### Manual Resume

```bash
# Check checkpoint status
cat benchmark_checkpoints/benchmark_checkpoint.json | python3 -m json.tool

# Check which algorithms completed
cat benchmark_checkpoints/benchmark_results.json | python3 -m json.tool
```

---

## Checkpoint File Locations

```
benchmark_checkpoints/
├── benchmark_checkpoint.json          # Current algorithm checkpoint
└── benchmark_results.json             # Completed algorithms

ablation_study_checkpoint.json         # Ablation study progress

sensitivity_network_size_checkpoint.json
sensitivity_cache_capacity_checkpoint.json
sensitivity_bloom_filter_checkpoint.json

topology_comparison_checkpoint.json    # Topology comparison progress

complete_comparison_checkpoint.json    # Complete comparison progress
```

---

## Example: Interrupted Benchmark

### Scenario: Benchmark interrupted after 3 algorithms

**Before interruption**:
- ✅ DQN: 10/10 runs completed
- ✅ FIFO: 10/10 runs completed
- ⏸️ LRU: 5/10 runs completed (interrupted)

**After restart**:
```bash
python benchmark.py
```

**Output**:
```
📊 Found 2 completed algorithms: DQN, FIFO
⏭️  Skipping DQN (already completed)
⏭️  Skipping FIFO (already completed)

Testing LRU...
🔄 Resuming from checkpoint: 5/10 runs already completed
📊 Loaded 5 previous run results
  Run 6/10...
  Run 7/10...
  ...
```

---

## Best Practices

### 1. Always Use Checkpoints

```bash
# Default behavior (resume enabled)
python benchmark.py
```

### 2. Monitor Long-Running Jobs

```bash
# Start in background
nohup python benchmark.py > benchmark.log 2>&1 &

# Monitor progress
tail -f benchmark.log

# Check checkpoint status
cat benchmark_checkpoints/benchmark_checkpoint.json
```

### 3. Clean Up After Completion

```bash
# Checkpoints are automatically cleared when complete
# But you can manually clear if needed
rm benchmark_checkpoints/benchmark_checkpoint.json
```

### 4. Use Screen/Tmux for Long Runs

```bash
# Start screen session
screen -S benchmark

# Run script
python benchmark.py

# Detach and let it run
# Reattach anytime to check progress
```

---

## Troubleshooting

### Issue: Checkpoint not resuming

**Solution**:
```bash
# Check checkpoint file exists
ls -la benchmark_checkpoints/benchmark_checkpoint.json

# Check checkpoint content
cat benchmark_checkpoints/benchmark_checkpoint.json | python3 -m json.tool

# If corrupted, delete and restart
rm benchmark_checkpoints/benchmark_checkpoint.json
python benchmark.py --no-resume
```

### Issue: Process killed but checkpoint not saved

**Solution**:
- Checkpoints are saved after each run completes
- If killed mid-run, that run will be re-run on resume
- This is expected behavior (atomic run completion)

### Issue: Want to restart from scratch

**Solution**:
```bash
# Clear checkpoint and start fresh
python benchmark.py --no-resume

# Or manually delete checkpoint
rm benchmark_checkpoints/benchmark_checkpoint.json
python benchmark.py
```

---

## Summary

✅ **All scripts support checkpointing**  
✅ **Automatic resume on restart**  
✅ **Background execution supported**  
✅ **Incremental results saved**  
✅ **No data loss on interruption**  

**Just run the script again after interruption - it will automatically resume!**

