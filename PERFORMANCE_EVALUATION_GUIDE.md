# Performance Evaluation Guide

## Quick Summary

### ✅ GPU Status
- **MPS (Metal) GPU**: Available ✅
- **CUDA GPU**: Not available
- **DQN Training**: Will use GPU automatically

### ⏱️ Time Estimates

| Configuration | Network Size | Estimated Time | Notes |
|--------------|--------------|----------------|-------|
| Quick Benchmark | 30 nodes, 10 rounds | ~20 seconds | Fast comparison |
| Full Simulation | 300 nodes, 20 rounds | **~5-10 minutes** | Combined eviction |
| Full Simulation + DQN | 300 nodes, 20 rounds | **~10-15 minutes** | With GPU acceleration |

### 🚀 How to Run

#### Option 1: Quick Benchmark (Fastest)
```bash
python benchmark.py
```
**Time**: ~20 seconds  
**Purpose**: Compare FIFO, LRU, Combined policies

#### Option 2: Full Simulation - Combined Eviction
```bash
./run_full_simulation.sh
# OR
python main.py
```
**Time**: ~5-10 minutes  
**Configuration**: 300 nodes, 20 rounds, combined eviction, 500 cache capacity

#### Option 3: Full Simulation - DQN Enabled
```bash
./run_dqn_simulation.sh
# OR
export NDN_SIM_USE_DQN=1 && python main.py
```
**Time**: ~10-15 minutes  
**Configuration**: Same as above + DQN caching with GPU acceleration

---

## GPU Benefits

### Current Setup
- **MPS (Metal) GPU**: Available on your Mac
- **DQN Training**: Automatically uses GPU when available
- **Speedup**: ~2-3x faster than CPU for DQN training

### What GPU Accelerates:
1. ✅ **DQN Neural Network Training**
   - Forward passes (action selection)
   - Backward passes (gradient computation)
   - Experience replay batch processing
   - **Speedup**: 2-3x faster

2. ✅ **Semantic Encoder (CNN)**
   - Convolutional operations
   - Embedding generation
   - **Speedup**: 3-5x faster (if using CNN mode)

3. ✅ **Neural Bloom Filter**
   - Neural network inference
   - False positive prediction
   - **Speedup**: 2-3x faster

### What GPU Does NOT Accelerate:
- ❌ Network simulation (CPU-bound)
- ❌ Packet routing (CPU-bound)
- ❌ Cache operations (CPU-bound)
- ❌ Metrics collection (CPU-bound)

### Overall Impact:
- **Without DQN**: GPU not used (simulation is CPU-bound)
- **With DQN**: GPU accelerates training by 2-3x, reducing total time from ~15-20 min to ~10-15 min

### If You Get a CUDA GPU:
- **Better Performance**: CUDA typically faster than MPS
- **More Memory**: Can handle larger batch sizes
- **Speedup**: Potentially 3-5x for DQN training
- **Total Time**: Could reduce to ~8-12 minutes with DQN

---

## Expected Results

### Cache Hit Rate Targets:
- **Baseline (FIFO)**: 0.01-0.05%
- **Combined Eviction**: 0.05-0.15% (5-15x improvement)
- **DQN Caching**: 0.15-0.50% (15-50x improvement)

### Metrics to Check:
1. **Cache Hit Rate**: Should be > 5% with combined, > 15% with DQN
2. **Cache Insertions**: Should be > 1000 (vs 37 before)
3. **Routers with Cache**: Should be > 50 (vs 3 before)
4. **Latency**: Should decrease with better caching
5. **Redundancy**: Content should be cached in multiple routers

---

## Running Performance Evaluation

### Step 1: Quick Benchmark (Recommended First)
```bash
python benchmark.py
```
**Time**: ~20 seconds  
**Output**: Comparison of caching policies

### Step 2: Full Simulation - Baseline
```bash
./run_full_simulation.sh
```
**Time**: ~5-10 minutes  
**Output**: Full results with combined eviction

### Step 3: Full Simulation - DQN (Optional)
```bash
./run_dqn_simulation.sh
```
**Time**: ~10-15 minutes  
**Output**: Full results with DQN caching

### Step 4: Check Results
```bash
# View simulation results
cat logs/simulation_results.log

# View comprehensive metrics
grep -A 50 "COMPREHENSIVE EVALUATION METRICS" logs/simulation_results.log
```

---

## Performance Optimization Tips

### To Reduce Runtime:
1. **Reduce Network Size**: Set `NDN_SIM_NODES=100` (faster, less accurate)
2. **Reduce Rounds**: Set `NDN_SIM_ROUNDS=10` (faster, less warm-up)
3. **Reduce Users**: Set `NDN_SIM_USERS=500` (faster, less traffic)

### To Improve Accuracy:
1. **Increase Rounds**: Set `NDN_SIM_ROUNDS=50` (slower, better warm-up)
2. **Increase Warm-up**: Set `NDN_SIM_WARMUP_ROUNDS=10` (slower, better cache)
3. **Use DQN**: Enable DQN for better decisions (slower, better results)

### GPU Optimization:
- **Current**: MPS GPU automatically used for DQN
- **If CUDA Available**: Will automatically use CUDA (faster)
- **No Action Needed**: GPU usage is automatic

---

## Troubleshooting

### If Simulation Takes Too Long:
- Reduce `NDN_SIM_NODES` to 100-150
- Reduce `NDN_SIM_ROUNDS` to 10
- Reduce `NDN_SIM_USERS` to 500-1000

### If Hit Rate Still Low:
- Check `logs/simulation_results.log` for cache statistics
- Verify cache capacity is 500 (not 100)
- Check that combined eviction is being used
- Ensure warm-up phase completed

### If DQN Not Working:
- Check `NDN_SIM_USE_DQN=1` is set
- Verify PyTorch installed: `python -c "import torch; print(torch.__version__)"`
- Check GPU available: `python -c "import torch; print(torch.backends.mps.is_available())"`

---

## Example Commands

```bash
# Quick test (30 nodes, ~20 seconds)
python benchmark.py

# Full simulation - Combined (300 nodes, ~5-10 min)
./run_full_simulation.sh

# Full simulation - DQN (300 nodes, ~10-15 min)
./run_dqn_simulation.sh

# Custom configuration
export NDN_SIM_NODES=200
export NDN_SIM_ROUNDS=15
export NDN_SIM_USE_DQN=1
python main.py
```

---

## Results Location

After running simulations, check:
- `logs/simulation_results.log` - Final statistics
- `logs/network_setup.log` - Network configuration
- `logs/trace.log` - Detailed router state changes
- Console output - Real-time progress

---

## Next Steps

1. ✅ Run quick benchmark (done - ~20s)
2. ⏳ Run full simulation - Combined eviction (~5-10 min)
3. ⏳ Run full simulation - DQN enabled (~10-15 min)
4. ⏳ Analyze results and compare with baseline

**Total Time for Complete Evaluation**: ~20-30 minutes

