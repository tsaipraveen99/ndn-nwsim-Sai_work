# Queue Drain Diagnostics Enhancement

## Problem
The simulation was experiencing queue drain timeouts - the queue wasn't draining within the 30-second timeout period, causing warnings and potentially incomplete message processing.

## Root Cause Analysis
The original `wait_for_queue_drain()` method had several limitations:
1. **Unreliable queue size checking**: Using `queue.empty()` is not thread-safe and can give false results
2. **No visibility into worker activity**: Couldn't tell if workers were processing messages or stuck
3. **No processing rate metrics**: Couldn't see if messages were being processed slowly or not at all
4. **Limited diagnostics**: Only showed elapsed time, not what was actually happening

## Enhancements Made

### 1. Enhanced Queue Size Estimation
- Added `_estimate_queue_size()` method that uses `queue.qsize()` (approximate but useful)
- Falls back gracefully if `qsize()` is unavailable
- Returns -1 if queue is not empty but exact size is unknown

### 2. Worker Activity Tracking
- Added `worker_processed_count` list to track messages processed by each worker
- Thread-safe counter using `worker_lock`
- Shows total messages processed across all workers
- Helps identify if workers are active or stuck

### 3. Improved Diagnostics Output
The new diagnostics show:
- **Initial state**: Queue size estimate, number of alive workers, total messages processed
- **Progress updates** (every 2 seconds):
  - Elapsed time and remaining timeout
  - Current queue size estimate
  - Number of alive workers
  - Total messages processed so far
  - Processing rate (messages/second) if calculable
- **Timeout state**: Final queue size, worker status, and total processed count

### 4. Better Queue Drain Detection
- Uses a separate thread to monitor queue drain status
- Double-checks queue empty state to avoid race conditions
- More reliable detection of when queue actually drains

## What the New Output Shows

### Example Output:
```
📊 Queue drain: initial_size≈150, workers_alive=8/8, processed=1250
⏳ Queue drain: still waiting... (2.0s elapsed, 28.0s remaining, queue≈120, workers=8/8, processed=1280, rate≈15.0 msgs/s)
⏳ Queue drain: still waiting... (4.0s elapsed, 26.0s remaining, queue≈90, workers=8/8, processed=1320, rate≈20.0 msgs/s)
...
⚠️  Queue drain: TIMEOUT after 30.0s (queue≈50, workers=8/8, total_processed=1500)
```

## Interpreting the Diagnostics

### Healthy Queue Drain:
- Queue size decreases over time
- Processing rate is positive and stable
- Workers are alive and processing messages
- Queue eventually drains before timeout

### Problematic Queue Drain:
- **Queue size not decreasing**: Messages aren't being processed (workers stuck?)
- **Processing rate = 0**: Workers aren't processing (deadlock? blocking?)
- **Queue size increasing**: New messages being enqueued faster than processed
- **Workers not alive**: Worker threads have died (exception? crash?)

## Next Steps for Investigation

If timeouts continue, check:

1. **Worker Status**: Are all workers alive? If not, check for exceptions in worker threads
2. **Processing Rate**: Is it 0 or very low? Messages might be stuck in processing
3. **Queue Growth**: Is queue size increasing? New messages might be enqueued during drain wait
4. **Message Types**: What types of messages are in the queue? Some might be causing blocking

## Potential Issues to Investigate

1. **Circular Message Dependencies**: Messages might be creating new messages that keep the queue non-empty
2. **Blocking Operations**: `dispatch_message()` might be blocking, preventing workers from processing more
3. **Deadlocks**: Workers might be waiting on locks that never release
4. **Exception Handling**: Exceptions in message processing might not be properly handled

## Code Changes

### Files Modified:
- `router.py`:
  - Enhanced `wait_for_queue_drain()` method
  - Added `_estimate_queue_size()` helper method
  - Added worker activity tracking in `__init__()` and `_worker_loop()`
  - Improved diagnostic output throughout

## Testing Recommendations

1. Run a simulation and observe the new diagnostic output
2. Check if queue size decreases over time
3. Verify workers are processing messages (processed count increases)
4. If timeouts persist, use the diagnostics to identify the specific issue:
   - If `processed` count doesn't increase → workers aren't processing
   - If queue size doesn't decrease → messages stuck or new ones being added
   - If workers not alive → check for exceptions/crashes

