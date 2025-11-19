# 🚀 Start Benchmark Now - Quick Guide

**Status**: ✅ All Code Ready - Environment Setup Required

---

## Prerequisites

Make sure your Python environment is activated:

```bash
# If using virtual environment
source venv/bin/activate  # or your venv path

# If using conda
conda activate your_env_name

# Verify dependencies
python -c "import torch, numpy, networkx; print('✅ All dependencies available')"
```

---

## Start Benchmark

### Method 1: Interactive Menu (Recommended)

```bash
./start_benchmark.sh
```

Choose:
- **1** = Foreground (see output, Ctrl+C to stop)
- **2** = Background with nohup (detached)
- **3** = Background with screen (detachable)

### Method 2: Direct Start

```bash
# Foreground (see output)
python benchmark.py

# Background
nohup python benchmark.py > benchmark.log 2>&1 &
```

### Method 3: Using Helper Script

```bash
./run_background.sh benchmark.py
```

---

## What Happens

The benchmark will:

1. **Test 5 algorithms**:
   - DQN (with Bloom filters) - 250 rounds, 10 runs
   - FIFO - 100 rounds, 10 runs
   - LRU - 100 rounds, 10 runs
   - LFU - 100 rounds, 10 runs
   - Combined - 100 rounds, 10 runs

2. **Save checkpoints** after every 2 runs

3. **Save results incrementally** to `benchmark_checkpoints/benchmark_results.json`

4. **Take 3-5 hours** total

---

## Monitor Progress

### If Running in Foreground
- Output appears in terminal
- Press Ctrl+C to stop (checkpoint saved)

### If Running in Background

```bash
# View output
tail -f benchmark.log

# Check if running
ps aux | grep benchmark.py

# Check checkpoint
cat benchmark_checkpoints/benchmark_checkpoint.json | python3 -m json.tool

# Check results
cat benchmark_checkpoints/benchmark_results.json | python3 -m json.tool
```

---

## Resume After Interruption

If interrupted, just run again:

```bash
python benchmark.py
```

It will automatically resume from the last checkpoint.

---

## Expected Results

After completion, you'll have:

- `benchmark_checkpoints/benchmark_results.json` - Complete results
- Hit rate comparison for all 5 algorithms
- Statistical metrics (mean, std dev, confidence intervals)

**Expected hit rates**:
- DQN: 40-50%
- LRU: 18-25%
- LFU: 15-22%
- FIFO: 12-18%
- Combined: 20-28%

---

## Next Steps After Benchmark

1. **Review Results**: Check `benchmark_checkpoints/benchmark_results.json`
2. **Run Ablation Study**: `python ablation_study.py`
3. **Run Sensitivity Analysis**: `python sensitivity_analysis.py`
4. **Run Topology Comparison**: `python topology_comparison.py`

---

## Troubleshooting

### "Module not found" error
- Activate your virtual environment or conda environment
- Install dependencies: `pip install torch numpy networkx`

### Benchmark hangs
- Check logs: `tail -f benchmark.log`
- Check checkpoint: `cat benchmark_checkpoints/benchmark_checkpoint.json`
- May need to wait for DQN training to complete

### Want to start fresh
```bash
python benchmark.py --clear-checkpoint
```

---

## Ready to Start!

1. Activate your Python environment
2. Run: `python benchmark.py` or `./start_benchmark.sh`
3. Monitor progress
4. Review results when complete

**Good luck! 🚀**

