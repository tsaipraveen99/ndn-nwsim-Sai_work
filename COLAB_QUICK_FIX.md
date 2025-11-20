# Quick Fix for Colab Queue Drain Timeouts

## Problem
The simulation is experiencing persistent "Message processing TIMEOUT" and "Queue drain: TIMEOUT" errors, particularly for `data` messages. Workers are timing out after 10 seconds when processing data packets.

## Immediate Solutions

### Option 1: Reduce Network Size (Fastest Fix)
The current configuration may be too large for Colab's resources. Try this smaller configuration:

```python
# In your Colab cell, before running benchmark.py:
import os

# REDUCE NETWORK SIZE
os.environ['NDN_SIM_NODES'] = '20'              # Reduced from 30
os.environ['NDN_SIM_PRODUCERS'] = '3'           # Reduced from 5
os.environ['NDN_SIM_USERS'] = '30'              # Reduced from 50
os.environ['NDN_SIM_CONTENTS'] = '200'          # Reduced from 300

# REDUCE SIMULATION LENGTH
os.environ['NDN_SIM_ROUNDS'] = '20'             # Reduced from 50
os.environ['NDN_SIM_REQUESTS'] = '20'           # Reduced from 30
os.environ['NDN_SIM_WARMUP_ROUNDS'] = '3'       # Reduced from 5

# KEEP OTHER SETTINGS
os.environ['NDN_SIM_CACHE_CAPACITY'] = '500'
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'
os.environ['NDN_SIM_TOPOLOGY'] = 'watts_strogatz'
os.environ['NDN_SIM_USE_DQN'] = '1'
os.environ['NDN_SIM_QUIET'] = '0'               # Show progress
os.environ['NDN_SIM_SKIP_DELAYS'] = '1'
```

### Option 2: Increase Worker Timeout (Easiest)
If data processing is legitimately slow (e.g., DQN inference), increase the timeout via environment variable:

```python
# In your Colab cell, before running benchmark.py:
os.environ['NDN_SIM_WORKER_TIMEOUT'] = '30.0'  # Increase from default 10.0 seconds
```

This is now configurable without modifying code!

### Option 3: Disable DQN Temporarily (For Testing)
If DQN is causing slowdowns, test without it first:

```python
os.environ['NDN_SIM_USE_DQN'] = '0'  # Disable DQN
os.environ['NDN_SIM_CACHE_POLICY'] = 'lru'  # Use LRU instead
```

## Check Progress

Run this in a separate Colab cell while the benchmark is running:

```python
!python check_colab_progress.py
```

Or manually check:

```python
import json
from pathlib import Path

# Check checkpoint
checkpoint_file = Path('benchmark_checkpoints/benchmark_checkpoint.json')
if checkpoint_file.exists():
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    print(f"Algorithm: {checkpoint.get('algorithm', 'unknown')}")
    print(f"Progress: {checkpoint.get('completed_runs', 0)}/{checkpoint.get('total_runs', 0)} runs")
```

## Root Cause Analysis

The timeouts are happening because:

1. **Data message processing is slow**: `handle_data()` involves:
   - DQN agent inference (if enabled) - can be slow on CPU
   - Cache operations
   - PIT lookups and forwarding
   - Multiple `forward_data()` calls

2. **Worker timeout is too short**: 10 seconds may not be enough for complex operations, especially with DQN.

3. **Queue flooding**: Too many messages in the queue can cause workers to be overwhelmed.

## Long-term Fixes (Already Implemented)

The codebase already includes:
- ✅ Asynchronous FIB propagation (prevents blocking)
- ✅ Worker timeout protection (prevents indefinite hangs)
- ✅ Queue drain timeout (120 seconds)
- ✅ Task done tracking (prevents queue blocking)

But the worker-level timeout (10s) may still be too short for DQN operations.

## Recommended Approach

1. **Start with Option 1** (reduce network size) - this is the safest and fastest fix
2. **If still timing out**, try Option 2 (increase worker timeout)
3. **If DQN is the issue**, try Option 3 (disable DQN) to confirm
4. **Once working**, gradually increase network size

## Expected Behavior

After applying fixes, you should see:
- ✅ No "Message processing TIMEOUT" messages
- ✅ No "Queue drain: TIMEOUT" messages
- ✅ Steady progress through rounds
- ✅ Checkpoint file updating regularly
- ✅ Final results file created when complete

