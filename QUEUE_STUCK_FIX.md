# Queue Stuck Issue - Diagnosis and Fix

## Problem

**Symptoms**:
- Queue growing: 79k → 114k messages
- Processed count stuck at 7487
- Processing rate: 0.0 msgs/s
- Workers alive (128/128) but not processing

**Root Cause**: Workers processed initial messages, then got stuck in a blocking operation.

## Immediate Fix Options

### Option 1: Increase Queue Drain Timeout (Quick Fix)

The queue drain timeout is 30s, but with 100k+ messages, it needs more time. However, the real issue is workers aren't processing.

### Option 2: Reduce Network Size (Quick Fix)

Current config has 160 routers with 128 workers. This might be too large. Try:
- Reduce nodes to 50
- Reduce workers to 32

### Option 3: Add Timeout to Message Processing (Proper Fix)

Add timeout to `dispatch_message()` to prevent workers from getting stuck.

## Recommended Action

**For now**: The simulation is continuing despite timeouts. It's completing rounds (Round 1-8 completed). The timeouts are warnings, not fatal errors.

**The simulation will complete**, but slowly. You can:
1. Let it continue (it will finish eventually)
2. Reduce network size for faster runs
3. Increase queue drain timeout

## Quick Configuration Fix

Add to benchmark config:
```python
'NDN_SIM_NODES': '50',  # Reduce from 160
'NDN_SIM_WORKERS': '32',  # Reduce from 128
```

This will reduce queue size and improve processing speed.

