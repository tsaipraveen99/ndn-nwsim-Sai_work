# Benchmark.py Execution Guide

**Status**: ✅ Ready to Execute

---

## Quick Start

```bash
python benchmark.py
```

**That's it!** The benchmark will:
- Run 5 algorithms (DQN, FIFO, LRU, LFU, Combined)
- 10 runs per algorithm
- Save checkpoints after every 2 runs
- Save results incrementally
- Take 3-5 hours total

---

## What Will Happen

### Phase 1: DQN (First Algorithm)

**Configuration**:
- Nodes: 50
- Contents: 1000
- Cache Capacity: 10 (1% of catalog)
- Zipf Parameter: 0.8
- Rounds: 250 (more for training)
- Warm-up: 30 rounds

**What you'll see**:
```
Testing DQN...
  🔧 Enabling DQN mode on all routers...
  ✅ DQN Training Manager initialized with 4 workers
  ✅ 50 routers with DQN agents initialized
  🔥 Warm-up phase: 30 rounds...
  📊 Evaluation phase: 250 rounds...
  Run 1/10...
  Run 2/10...
  💾 Checkpoint saved: DQN (2/10 runs)
  ...
  ✅ DQN completed: 42.8% hit rate
```

**Time**: ~2-3 hours (250 rounds, more training)

---

### Phase 2-5: Traditional Algorithms

**FIFO, LRU, LFU, Combined**:
- Rounds: 100 each
- Warm-up: 20 rounds
- 10 runs per algorithm

**What you'll see**:
```
Testing FIFO...
  Run 1/10...
  Run 2/10...
  💾 Checkpoint saved: FIFO (2/10 runs)
  ...
  ✅ FIFO completed: 15.2% hit rate

Testing LRU...
  ...
```

**Time**: ~1-2 hours total (100 rounds each)

---

## Expected Results

After completion, you'll have:

### Hit Rate Comparison
```
DQN:        40-50%  (with Bloom filters)
LRU:        18-25%  (traditional baseline)
LFU:        15-22%  (frequency-based)
FIFO:       12-18%  (simple queue)
Combined:   20-28%  (hybrid)
```

### Statistical Metrics
- Mean hit rate per algorithm
- Standard deviation
- Confidence intervals (95%)
- Cache hits, nodes traversed
- Cache utilization

### Output Files
- `benchmark_checkpoints/benchmark_results.json` - Complete results
- `benchmark_checkpoints/benchmark_checkpoint.json` - Current progress (cleared when done)

---

## Monitoring Progress

### Check Status

```bash
# Check if running
ps aux | grep python | grep benchmark

# Check checkpoint
cat benchmark_checkpoints/benchmark_checkpoint.json | python3 -m json.tool

# Check results (as they're saved)
cat benchmark_checkpoints/benchmark_results.json | python3 -m json.tool
```

### Expected Output During Run

```
PERFORMANCE BENCHMARKS
================================================================================
🆕 Starting fresh benchmark
================================================================================

Testing DQN...
  🔧 Enabling DQN mode on all routers...
  ✅ DQN Training Manager initialized with 4 workers
  ✅ 50 routers with DQN agents initialized
  🔥 Warm-up phase: 30 rounds...
  📊 Evaluation phase: 250 rounds...
  Run 1/10...
    ⏳ Round 1: Waiting for queue to drain...
    ✅ Round 1 completed
    ...
  Run 2/10...
  💾 Checkpoint saved: DQN (2/10 runs)
  ...
  ✅ DQN completed: 42.8% hit rate

Testing FIFO...
  Run 1/10...
  ...
```

---

## Resume After Interruption

If interrupted, just run again:

```bash
python benchmark.py
```

**It will automatically**:
- Detect checkpoint
- Resume from last completed run
- Skip already-completed algorithms
- Continue where it left off

---

## Background Execution

### Option 1: Using Helper Script

```bash
./run_background.sh benchmark.py
```

### Option 2: Using nohup

```bash
nohup python benchmark.py > benchmark.log 2>&1 &
```

### Option 3: Using screen

```bash
screen -S benchmark
python benchmark.py
# Detach: Ctrl+A, then D
```

---

## After Completion

### 1. Review Results

```bash
cat benchmark_checkpoints/benchmark_results.json | python3 -m json.tool
```

### 2. Next Steps

**Ablation Study** (identify component contributions):
```bash
python ablation_study.py
```

**Sensitivity Analysis** (test robustness):
```bash
python sensitivity_analysis.py
```

**Topology Comparison** (test different networks):
```bash
python topology_comparison.py
```

**Complete Comparison** (comprehensive analysis):
```bash
python complete_comparison.py
```

### 3. Generate Visualizations

```bash
python visualize_comparison.py
python plot_learning_curves.py
```

---

## Troubleshooting

### Queue Drain Timeouts

**If you see**:
```
⚠️  Queue drain: TIMEOUT after 30.0s
```

**This is normal**:
- Simulation continues anyway
- Timeouts are warnings, not fatal
- Will complete, just slower

**If it's too slow**:
- Current run: Let it continue (will finish)
- Future runs: Already optimized (50 nodes, not 300)

### DQN Not Learning

**Check**:
- Look for: "✅ DQN Training Manager initialized"
- Look for: "✅ X routers with DQN agents initialized"
- If missing, there's an initialization issue

### Out of Memory

**Solution**:
- Reduce nodes: Edit `base_config` in benchmark.py
- Change `NDN_SIM_NODES` from '50' to '30'

---

## Success Criteria

✅ All 5 algorithms complete  
✅ DQN hit rate > LRU/LFU/FIFO  
✅ Results saved to JSON  
✅ Statistical metrics calculated  
✅ Checkpoint cleared (all algorithms done)  

---

## Ready to Start!

```bash
python benchmark.py
```

**Estimated Time**: 3-5 hours  
**Output**: `benchmark_checkpoints/benchmark_results.json`  
**Checkpoint Support**: ✅ Yes (resume if interrupted)

**Good luck! 🚀**

