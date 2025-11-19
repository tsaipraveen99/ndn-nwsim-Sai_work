# Debug Logging Added

## Changes Made

### 1. Fixed Hardcoded Timeout in `main.py` ✅
- Changed `timeout=30.0` → `timeout=120.0` in `run_simulation()`
- Now uses the 120s timeout we set in `router.py`

### 2. Added Worker Loop Debugging ✅
- Logs every 100th message processed
- Logs if 10 seconds pass without processing
- Shows worker ID, message type, and router ID
- Tracks messages processed per worker

### 3. Added Processing Rate Calculation ✅
- Calculates rate from processed count (not just queue size)
- Shows "⚠️ WORKERS STUCK" if processed_rate = 0
- Helps identify if workers are actually processing

### 4. Added Dispatch Message Timing ✅
- Logs if `dispatch_message()` takes > 1 second
- Helps identify slow message handlers
- Shows which message types are slow

---

## What to Look For

### In Worker Logs:
```
Worker 0: Processing interest to router 5 (msg #100)
Worker 1: Processing data to router 10 (msg #200)
```

### In Queue Drain Logs:
```
Queue drain: ... processed_rate=15.2 msgs/s  ← Good!
Queue drain: ... processed_rate=0.0 msgs/s ⚠️ WORKERS STUCK  ← Bad!
```

### In Dispatch Logs:
```
Router 5: dispatch_message(interest) took 2.34s  ← Slow handler!
```

---

## Next Steps

1. **Restart benchmark** with new logging:
   ```bash
   python benchmark.py 2>&1 | tee benchmark_debug.log
   ```

2. **Monitor for**:
   - Worker processing messages (should see "Worker X: Processing...")
   - Processing rate > 0 (should see "processed_rate=X.X msgs/s")
   - Slow messages (should see "dispatch_message(...) took X.Xs")

3. **If workers still stuck**:
   - Check logs for which message types are slow
   - Check if specific routers are slow
   - Check if `handle_interest()` or `handle_data()` are blocking

---

## Expected Output

### Good (Workers Processing):
```
Worker 0: Processing interest to router 5 (msg #100)
Queue drain: ... processed_rate=15.2 msgs/s
```

### Bad (Workers Stuck):
```
Queue drain: ... processed_rate=0.0 msgs/s ⚠️ WORKERS STUCK
```

### Slow Handler:
```
Router 5: dispatch_message(interest) took 2.34s
```

