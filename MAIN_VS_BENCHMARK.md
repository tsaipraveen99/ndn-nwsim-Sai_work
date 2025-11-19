# main.py vs benchmark.py - Comparison Guide

## Purpose

### `main.py` - Single Simulation Run

**Use for**: Quick testing, validation, debugging, single algorithm runs

**What it does**:

- Runs ONE simulation with ONE algorithm
- Uses environment variables for configuration
- Good for quick validation (5-10 minutes)
- Has visualization (network_topology.png)
- Has detailed logging

### `benchmark.py` - Comprehensive Comparison

**Use for**: Research evaluation, algorithm comparison, publication results

**What it does**:

- Runs MULTIPLE algorithms (DQN, FIFO, LRU, LFU, Combined)
- 10 runs per algorithm for statistical significance
- Saves results to JSON
- Has checkpoint/resume support
- Has statistical analysis (mean, std dev, confidence intervals)

---

## Configuration Comparison

### main.py Configuration

**Source**: Environment variables (defaults shown)

```python
NDN_SIM_NODES=300          # Default: 300 nodes
NDN_SIM_CONTENTS=6000     # Default: 6000 contents
NDN_SIM_CACHE_CAPACITY=500 # Default: 500 items
NDN_SIM_ZIPF_PARAM=1.2    # Default: 1.2 (OLD - not updated!)
NDN_SIM_ROUNDS=20         # Default: 20 rounds
NDN_SIM_REQUESTS=5        # Default: 5 requests
```

**Issues**:

- ❌ Still uses old Zipf parameter (1.2 instead of 0.8)
- ❌ Still uses old cache capacity (500 instead of 10)
- ❌ Still uses old contents (6000 instead of 1000)
- ✅ Has DQNTrainingManager initialization
- ✅ Has DQN checkpointing setup

### benchmark.py Configuration

**Source**: Hardcoded in `base_config` (FIXED values)

```python
NDN_SIM_NODES=50          # ✅ Fixed: 50 nodes
NDN_SIM_CONTENTS=1000     # ✅ Fixed: 1000 contents
NDN_SIM_CACHE_CAPACITY=10 # ✅ Fixed: 10 items (1% of catalog)
NDN_SIM_ZIPF_PARAM=0.8    # ✅ Fixed: 0.8 (realistic)
NDN_SIM_ROUNDS=100        # ✅ Fixed: 100 rounds (250 for DQN)
NDN_SIM_REQUESTS=20       # ✅ Fixed: 20 requests
```

**Status**:

- ✅ All parameters fixed (Zipf 0.8, Contents 1000, Cache 10)
- ✅ Has DQNTrainingManager initialization (just fixed)
- ✅ Has checkpoint/resume support
- ✅ Has statistical analysis

---

## What Needs to Be Changed

### Option 1: Fix main.py (Recommended if you want to use it)

**Changes needed in `main.py`**:

1. Update default Zipf parameter from 1.2 to 0.8
2. Update default cache capacity from 500 to 10
3. Update default contents from 6000 to 1000
4. Update default nodes from 300 to 50 (optional, for consistency)

**Location**: `main.py` lines 770-776

### Option 2: Use benchmark.py (Recommended for research)

**Status**: ✅ Already fixed and ready

- All parameters correct
- DQNTrainingManager initialized
- Checkpoint support
- Statistical analysis

---

## Recommendation

### For Research/Publication: Use `benchmark.py`

**Why**:

- ✅ All parameters fixed correctly
- ✅ Multiple algorithms compared
- ✅ Statistical significance (10 runs)
- ✅ Results saved to JSON
- ✅ Checkpoint/resume support
- ✅ Ready for publication

**Command**:

```bash
python benchmark.py
```

### For Quick Testing: Use `main.py` (after fixing)

**Why**:

- ✅ Faster (single run)
- ✅ Good for debugging
- ✅ Has visualization
- ⚠️ Needs parameter fixes first

**After fixing, command**:

```bash
NDN_SIM_NODES=50 \
NDN_SIM_CONTENTS=1000 \
NDN_SIM_CACHE_CAPACITY=10 \
NDN_SIM_ZIPF_PARAM=0.8 \
NDN_SIM_ROUNDS=100 \
NDN_SIM_REQUESTS=20 \
NDN_SIM_USE_DQN=1 \
python main.py
```

---

## Quick Fix for main.py

If you want to use `main.py`, update these defaults:

```python
# In main.py, around line 770
params = {
    'num_nodes': int(os.environ.get("NDN_SIM_NODES", "50")),        # Changed from 300
    'num_producers': int(os.environ.get("NDN_SIM_PRODUCERS", "10")), # Changed from 60
    'num_contents': int(os.environ.get("NDN_SIM_CONTENTS", "1000")), # Changed from 6000
    'num_users': int(os.environ.get("NDN_SIM_USERS", "100")),      # Changed from 2000
    'num_rounds': int(os.environ.get("NDN_SIM_ROUNDS", "100")),   # Changed from 20
    'num_requests': int(os.environ.get("NDN_SIM_REQUESTS", "20")), # Changed from 5
    'cache_policy': os.environ.get("NDN_SIM_CACHE_POLICY", "combined"),
    'use_dqn_cache': os.environ.get("NDN_SIM_USE_DQN", "0") == "1"
}

# Also update Zipf default around line 167
zipf_param = float(os.environ.get('NDN_SIM_ZIPF_PARAM', '0.8'))  # Changed from 1.2

# Also update cache capacity default around line 155
router_capacity = int(os.environ.get("NDN_SIM_CACHE_CAPACITY", "10"))  # Changed from 500
```

---

## Summary

| Feature                  | main.py           | benchmark.py               |
| ------------------------ | ----------------- | -------------------------- |
| **Purpose**              | Single run        | Multi-algorithm comparison |
| **Zipf Parameter**       | ❌ 1.2 (old)      | ✅ 0.8 (fixed)             |
| **Contents**             | ❌ 6000 (old)     | ✅ 1000 (fixed)            |
| **Cache Capacity**       | ❌ 500 (old)      | ✅ 10 (fixed)              |
| **DQNTrainingManager**   | ✅ Yes            | ✅ Yes (just fixed)        |
| **Checkpoint Support**   | ✅ Yes (DQN only) | ✅ Yes (full)              |
| **Statistical Analysis** | ❌ No             | ✅ Yes                     |
| **Multiple Algorithms**  | ❌ No             | ✅ Yes                     |
| **Best For**             | Quick testing     | Research/publication       |

---

## Final Recommendation

**For your research**: Use `benchmark.py` ✅

**Reasons**:

1. All parameters already fixed
2. Comprehensive comparison (5 algorithms)
3. Statistical analysis included
4. Checkpoint/resume support
5. Ready for publication

**Command**:

```bash
python benchmark.py
```

**If you want to use `main.py`**: Fix the defaults first (see "Quick Fix" above)
