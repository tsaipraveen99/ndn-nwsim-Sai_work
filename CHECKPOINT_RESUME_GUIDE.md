# Checkpoint/Resume Guide

## ✅ Checkpoint/Resume Functionality Added!

The benchmark now supports **checkpointing and resuming** from interruptions.

## How It Works

### Automatic Checkpointing

1. **After Each Algorithm**: When an algorithm completes all runs, results are saved immediately
2. **During Runs**: Checkpoint saved every 2 runs (for safety)
3. **Incremental Results**: Results are saved to `benchmark_checkpoints/benchmark_results.json` after each algorithm

### Resume Behavior

When you restart the benchmark:
- ✅ **Automatically detects** completed algorithms and skips them
- ✅ **Resumes** from the last checkpoint if an algorithm was interrupted
- ✅ **Continues** with remaining algorithms

## Files Created

- `benchmark_checkpoints/benchmark_checkpoint.json` - Current algorithm progress
- `benchmark_checkpoints/benchmark_results.json` - All completed algorithm results

## Usage

### Normal Run (with resume)
```bash
python3 benchmark.py
# or
./run_benchmark_safe.sh
```

### Start Fresh (ignore checkpoints)
```bash
python3 benchmark.py --no-resume
```

### Clear All Checkpoints and Start Fresh
```bash
python3 benchmark.py --clear-checkpoint
```

## Example Scenarios

### Scenario 1: Laptop Sleeps Mid-Run

**Before**: Lost all progress, had to restart from beginning
**Now**: 
1. Benchmark saves checkpoint every 2 runs
2. When you restart, it resumes from last checkpoint
3. Already-completed algorithms are skipped

### Scenario 2: Process Killed

**Before**: Lost all progress
**Now**:
1. Completed algorithms are saved in `benchmark_results.json`
2. In-progress algorithm has checkpoint
3. Restart continues from checkpoint

### Scenario 3: Want to Re-run Specific Algorithm

```bash
# Clear just the checkpoint (keeps completed results)
rm benchmark_checkpoints/benchmark_checkpoint.json

# Or clear everything and start fresh
python3 benchmark.py --clear-checkpoint
```

## Monitoring Progress

### Check Current Status
```bash
# View checkpoint
cat benchmark_checkpoints/benchmark_checkpoint.json | python3 -m json.tool

# View completed results
cat benchmark_checkpoints/benchmark_results.json | python3 -m json.tool
```

### Check What's Completed
```bash
python3 -c "
import json
with open('benchmark_checkpoints/benchmark_results.json', 'r') as f:
    results = json.load(f)
    print('Completed algorithms:', ', '.join(results.keys()))
"
```

## Benefits

1. ✅ **No Lost Progress**: Partial results are preserved
2. ✅ **Time Savings**: Skip already-completed algorithms
3. ✅ **Fault Tolerance**: Survives interruptions
4. ✅ **Incremental Results**: Can analyze partial results while benchmark runs

## Notes

- Checkpoints are saved every 2 runs (configurable in code)
- Results are saved after each algorithm completes
- Checkpoint file is cleared when algorithm completes
- If checkpoint file exists at end, some algorithm didn't complete

