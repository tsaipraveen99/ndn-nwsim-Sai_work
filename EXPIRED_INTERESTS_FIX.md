# Expired Interests Bug Fix

## Problem Identified

**Issue**: 2,174,612 expired Interests (99% of all Interests!)

### Root Cause:
1. **Time Mismatch**: `Interest.is_expired()` uses real time (`time.time()`), but simulation uses `router_time` (simulation time)
2. **Interest Creation**: Interests are created with `creation_time = time.time()` (real time)
3. **Expiration Check**: `is_expired()` compares `time.time() - creation_time` (real time)
4. **Simulation Time**: Routers use `router_time` which starts at `time.time()` but increments by 0.1 per step

### Why This Causes Massive Expiration:
- Simulation runs for 27 minutes in **real time**
- But Interests have `lifetime = 4.0` seconds
- When an Interest created at start is checked later, real time has advanced 27 minutes
- `27 minutes > 4 seconds` → Interest appears expired immediately!

### Example:
```
Interest created at: time.time() = 1000.0 (real time)
Interest lifetime: 4.0 seconds
Simulation runs for 27 minutes (1620 seconds)
When checked: time.time() = 2620.0
Expiration check: (2620.0 - 1000.0) > 4.0 → TRUE (expired!)
But simulation time might only be 10 seconds!
```

---

## Fix

### Solution 1: Use Simulation Time for Expiration (Recommended)

Modify `Interest.is_expired()` to accept `current_time` parameter (like `Data.is_fresh()` does):

```python
def is_expired(self, current_time: float = None) -> bool:
    """Check if the Interest packet has expired"""
    if current_time is None:
        current_time = time.time()  # Fallback to real time
    return (current_time - self.creation_time) > self.lifetime
```

Then in `router.py`, pass `router_time`:
```python
if interest.is_expired(self.router_time):
    logger.warning(...)
    return
```

### Solution 2: Set Creation Time to Simulation Time

When creating Interests, set `creation_time` to the router's current `router_time` instead of `time.time()`.

---

## Implementation

See the code changes in:
- `packet.py`: Update `is_expired()` method
- `router.py`: Pass `router_time` to `is_expired()`
- `endpoints.py`: Optionally set `creation_time` to router's `router_time` when creating Interests

---

## Expected Impact

- **Before**: 2,174,612 expired Interests (99% of all)
- **After**: Should be < 1% (only truly expired Interests)
- **Result**: Massive improvement in successful Interest processing

