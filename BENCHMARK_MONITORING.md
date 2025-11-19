# Benchmark Monitoring Guide

## ✅ Benchmark is Running

**Process ID**: Check `benchmark.pid` file or run:
```bash
ps aux | grep "python3 benchmark.py" | grep -v grep
```

## Monitor Progress

### View Live Log
```bash
tail -f benchmark_run.log
```

### Check Current Status
```bash
tail -50 benchmark_run.log | grep -E "(Testing|Run|RESULTS|hit_rate)"
```

### Check if Still Running
```bash
ps aux | grep "python3 benchmark.py" | grep -v grep
```

## What's Running

The benchmark tests:
1. **5 Algorithms** × **10 runs each** = 50 runs
   - FIFO
   - LRU
   - LFU
   - Combined
   - DQN

2. **Scalability Tests** = 2 runs
   - Small network
   - Medium network

**Total**: ~52 runs
**Estimated Time**: 30-60 minutes

## Expected Output

When complete, you'll see:
- Results table with hit rates
- Statistical comparison (mean, std, confidence intervals)
- Results saved to JSON files

## Important Notes

⚠️ **Laptop Sleep**: On macOS, processes are **suspended** (not killed) when the laptop sleeps. They resume when the laptop wakes up, but won't make progress while sleeping.

💡 **To Keep Laptop Awake** (optional):
```bash
caffeinate -d python3 benchmark.py
```

## Stop Benchmark

If you need to stop it:
```bash
kill $(cat benchmark.pid)
# or
pkill -f "python3 benchmark.py"
```

## After Completion

Results will be in:
- Console output (in `benchmark_run.log`)
- `results/medium_network_comparison.json` (if updated)

Then you can:
1. Review results
2. Run ablation study: `python3 ablation_study.py`
3. Run sensitivity analysis: `python3 sensitivity_analysis.py`

