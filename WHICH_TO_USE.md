# Which File to Use: main.py vs benchmark.py

## Quick Answer

**For Research/Publication**: Use `benchmark.py` ✅

**For Quick Testing**: Use `main.py` (but set environment variables)

---

## Detailed Comparison

### main.py Issues

**Current Defaults** (lines 155, 167, 770-776):
- ❌ `NDN_SIM_CACHE_CAPACITY`: Default = `"500"` (should be `"10"`)
- ❌ `NDN_SIM_ZIPF_PARAM`: Default = `'1.2'` (should be `'0.8'`)
- ❌ `NDN_SIM_CONTENTS`: Default = `"6000"` (should be `"1000"`)
- ⚠️ `NDN_SIM_NODES`: Default = `"300"` (benchmark.py uses `"50"`)

**What main.py has**:
- ✅ DQNTrainingManager initialization
- ✅ DQN checkpointing setup
- ✅ Visualization (network_topology.png)
- ✅ Detailed logging
- ✅ Single simulation run

### benchmark.py Status

**Current Configuration**:
- ✅ `NDN_SIM_CACHE_CAPACITY`: `'10'` (correct)
- ✅ `NDN_SIM_ZIPF_PARAM`: `'0.8'` (correct)
- ✅ `NDN_SIM_CONTENTS`: `'1000'` (correct)
- ✅ `NDN_SIM_NODES`: `'50'` (correct)
- ✅ DQNTrainingManager initialization (just fixed)
- ✅ Checkpoint/resume support
- ✅ Statistical analysis
- ✅ Multiple algorithms comparison

---

## Recommendation

### Option 1: Use benchmark.py (Recommended) ✅

**Why**:
- All parameters already fixed
- Comprehensive comparison (5 algorithms)
- Statistical analysis (10 runs per algorithm)
- Checkpoint/resume support
- Ready for publication

**Command**:
```bash
python benchmark.py
```

**Output**: `benchmark_checkpoints/benchmark_results.json`

---

### Option 2: Use main.py (After Setting Environment Variables)

**Why**:
- Faster for quick testing
- Good for debugging
- Has visualization

**Command** (with correct parameters):
```bash
NDN_SIM_NODES=50 \
NDN_SIM_PRODUCERS=10 \
NDN_SIM_CONTENTS=1000 \
NDN_SIM_USERS=100 \
NDN_SIM_ROUNDS=100 \
NDN_SIM_REQUESTS=20 \
NDN_SIM_CACHE_CAPACITY=10 \
NDN_SIM_ZIPF_PARAM=0.8 \
NDN_SIM_USE_DQN=1 \
python main.py
```

**Note**: You MUST set environment variables, otherwise it uses old defaults!

---

## What Needs to Be Changed

### If You Want to Fix main.py Defaults

Update these 3 lines in `main.py`:

**Line 155** (cache capacity):
```python
# OLD:
router_capacity = int(os.environ.get("NDN_SIM_CACHE_CAPACITY", "500"))

# NEW:
router_capacity = int(os.environ.get("NDN_SIM_CACHE_CAPACITY", "10"))
```

**Line 167** (Zipf parameter):
```python
# OLD:
zipf_param = float(os.environ.get('NDN_SIM_ZIPF_PARAM', '1.2'))

# NEW:
zipf_param = float(os.environ.get('NDN_SIM_ZIPF_PARAM', '0.8'))
```

**Line 770-776** (default parameters):
```python
# OLD:
params = {
    'num_nodes': int(os.environ.get("NDN_SIM_NODES", "300")),
    'num_producers': int(os.environ.get("NDN_SIM_PRODUCERS", "60")),
    'num_contents': int(os.environ.get("NDN_SIM_CONTENTS", "6000")),
    'num_users': int(os.environ.get("NDN_SIM_USERS", "2000")),
    'num_rounds': int(os.environ.get("NDN_SIM_ROUNDS", "20")),
    'num_requests': int(os.environ.get("NDN_SIM_REQUESTS", "5")),
    ...
}

# NEW:
params = {
    'num_nodes': int(os.environ.get("NDN_SIM_NODES", "50")),
    'num_producers': int(os.environ.get("NDN_SIM_PRODUCERS", "10")),
    'num_contents': int(os.environ.get("NDN_SIM_CONTENTS", "1000")),
    'num_users': int(os.environ.get("NDN_SIM_USERS", "100")),
    'num_rounds': int(os.environ.get("NDN_SIM_ROUNDS", "100")),
    'num_requests': int(os.environ.get("NDN_SIM_REQUESTS", "20")),
    ...
}
```

---

## Summary

| Aspect | main.py | benchmark.py |
|--------|---------|--------------|
| **Ready to Use?** | ⚠️ Needs env vars or fixes | ✅ Ready |
| **Parameters** | ❌ Old defaults | ✅ Fixed |
| **DQNTrainingManager** | ✅ Yes | ✅ Yes |
| **Multiple Algorithms** | ❌ No | ✅ Yes |
| **Statistical Analysis** | ❌ No | ✅ Yes |
| **Checkpoint Support** | ✅ Yes (DQN only) | ✅ Yes (full) |
| **Best For** | Quick testing | Research/publication |

---

## Final Answer

**Use `benchmark.py`** - It's already fixed and ready for research.

If you want to use `main.py` for quick testing, either:
1. Set all environment variables (see Option 2 above), OR
2. Fix the 3 default values in main.py (see "What Needs to Be Changed" above)

**Recommended**: Use `benchmark.py` for your research work.

