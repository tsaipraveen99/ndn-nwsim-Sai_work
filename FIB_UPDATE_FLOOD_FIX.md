# Fix: FIB Update Queue Flooding

## Problem

The simulation was getting stuck processing only `fib_update` messages for extended periods (22+ minutes), preventing interest/data messages from being processed. This was caused by:

1. **Exponential FIB Propagation**: `install_global_prefix_routes` triggers FIB updates on every router, which propagate to all neighbors, creating an exponential explosion of FIB updates
2. **No Deduplication**: The same FIB update could be processed multiple times
3. **No Rate Limiting**: FIB updates could flood the queue without any throttling

## Root Cause

When `install_global_prefix_routes` is called:
- It calls `router.add_to_FIB(prefix, producer.router_id, G)` for every prefix on every router
- Each `add_to_FIB` call propagates to all neighbors (typically 3-5 neighbors per router)
- Each neighbor propagates to all its neighbors
- This creates O(N*M*D) FIB updates where:
  - N = number of routers
  - M = number of prefixes  
  - D = average degree of the graph

For a network with 20 routers, 3 producers, and 4 prefixes per producer:
- Initial calls: 20 routers × 12 prefixes = 240 FIB calls
- First propagation: 240 × 3 neighbors = 720 FIB updates
- Second propagation: 720 × 3 neighbors = 2,160 FIB updates
- This continues exponentially until the queue is saturated

## Fixes Applied

### 1. FIB Update Deduplication
**File**: `router.py`, lines 770-775, 828-845

Added deduplication to prevent processing the same FIB update multiple times:
```python
# Track recently processed FIB updates
self.recent_fib_updates: Set[tuple] = set()  # (content_name, next_hop) tuples

# In dispatch_message, check for duplicates:
if fib_key in self.recent_fib_updates:
    logger.debug(f"Router {self.router_id}: Skipping duplicate FIB update")
    return
```

### 2. FIB Update Rate Limiting
**File**: `router.py`, lines 175-201

Added rate limiting in `RouterRuntime.enqueue()` to throttle FIB updates:
```python
if message_type == 'fib_update':
    MAX_FIB_UPDATES_PER_SECOND = int(os.environ.get('NDN_SIM_MAX_FIB_RATE', '50'))
    # Count FIB updates in the last second
    # If exceeding limit, drop the FIB update
    if len(self.metrics['fib_update_times']) >= MAX_FIB_UPDATES_PER_SECOND:
        logger.debug(f"Dropping FIB update due to rate limit")
        return
```

### 3. Limited FIB Propagation
**File**: `router.py`, lines 962-963

Limited the number of neighbors that receive FIB propagation to prevent exponential explosion:
```python
MAX_PROPAGATION_NEIGHBORS = int(os.environ.get('NDN_SIM_MAX_FIB_PROPAGATION', '10'))
propagation_targets = propagate_targets[:MAX_PROPAGATION_NEIGHBORS]  # Limit neighbors
```

## Configuration

New environment variables (added to `COLAB_SINGLE_CELL_ENHANCED.py`):

- `NDN_SIM_MAX_FIB_RATE`: Maximum FIB updates per second (default: 50, Colab: 30)
- `NDN_SIM_MAX_FIB_PROPAGATION`: Maximum neighbors to propagate to per FIB update (default: 10, Colab: 5)

## Expected Behavior After Fix

1. ✅ FIB updates are deduplicated - same update won't be processed twice
2. ✅ FIB updates are rate-limited - queue won't be flooded
3. ✅ FIB propagation is limited - prevents exponential explosion
4. ✅ Interest/data messages can proceed - FIB updates don't block critical messages
5. ✅ Simulation progresses normally - FIB updates happen in background at controlled rate

## Testing

After applying this fix:
- FIB updates should process at a controlled rate (30 per second in Colab)
- Interest/data messages should be processed normally
- Queue should drain successfully without timeouts
- Simulation should complete in reasonable time

## Additional Recommendations

If FIB updates are still causing issues, consider:

1. **Disable Global FIB Installation**: Set `NDN_SIM_INSTALL_GLOBAL=0` to skip `install_global_prefix_routes`
2. **Reduce Network Size**: Smaller networks generate fewer FIB updates
3. **Increase Rate Limits**: If needed, increase `NDN_SIM_MAX_FIB_RATE` (but monitor queue size)

