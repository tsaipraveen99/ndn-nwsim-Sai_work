# NDN Simulation Compliance Analysis

## Executive Summary

The NDN simulation has **several critical deviations** from the standard NDN (Named Data Networking) specification. While the basic packet structure is present, the forwarding logic and routing behaviors do not fully comply with RFC 8569 (NDN Packet Format Specification) and the standard NDN forwarding plane semantics.

---

## Critical Issues

### 1. ❌ **Interest Aggregation Not Properly Implemented**

**Standard Behavior**: When a router receives multiple Interests for the same content name, they should be aggregated in the PIT. Only ONE Interest is forwarded downstream, and when Data arrives, it satisfies all pending requests.

**Current Implementation** (router.py:452-456):

```python
# Check PIT for existing interests
if str(interest.name) in self.PIT:
    logger.debug(f"Router {self.router_id}: Interest {interest.name} already in PIT. Aggregating request.")
    self.PIT.add_entry(str(interest.name), prev_node, self.router_time)
    return
```

**Issues**:

- After aggregating, the code simply RETURNS without creating a NEW PIT entry
- If the PIT entry already exists, it should not forward a new Interest downstream
- However, the logic doesn't prevent the SAME Interest from being sent again

**What Should Happen**:

- First Interest arrives → Create PIT entry, forward Interest downstream
- Second Interest for same name arrives → Add incoming face to existing PIT entry, DO NOT forward again
- Data arrives → Use PIT to fan out to ALL incoming faces

---

### 2. ❌ **Data Packet Cannot Be Reused Multiple Times**

**Standard Behavior**: A Data packet can satisfy multiple pending Interests. When cached, the same Data can be sent to multiple consumers.

**Current Implementation** (router.py:620-641):

```python
def forward_data(self, G, data: Data, next_node: int):
    """Forward Data packet to the next node"""
    ...
    data.current_hops += 1
    stats.update(packets=1, size=getattr(data, 'size', 0))

    self.send_message(
        next_node,
        'data',
        (G, data, self.router_id),
        priority=1
    )
```

**Issues**:

- Data packet is forwarded by reference, not cloned
- When `current_hops` is incremented (line 627), it modifies the ORIGINAL packet
- If the same Data needs to be sent to multiple faces, the hop count gets incremented multiple times
- **Standard NDN requires**: Each outgoing face should get a clone or the hops should not be modified until transmission

**What Should Happen**:

- Clone the Data packet before sending on each interface
- Only the transmitted copy should have its hops modified

---

### 3. ❌ **Cache Entry Timing Not Properly Managed**

**Standard Behavior**: Data packets have explicit lifetimes. Cached entries should expire after their `FreshnessPeriod` or after a timeout. Multiple requests to the same content should only bypass the cache if it's stale.

**Current Implementation** (utils.py): ContentStore tracks creation time but doesn't properly implement NDN freshness period semantics.

**Issues**:

- Cache entries don't have explicit freshness periods
- No distinction between "fresh" and "stale" cached content
- Older simulation output showed 0 cache hits, indicating cache wasn't being properly checked

**What Should Happen**:

- Data packets should carry `FreshnessPeriod` field
- Cached entries should include metadata about when they were cached
- Check: `current_time - cache_time < freshness_period`

---

### 4. ❌ **Interest Return Path Not Properly Tracked**

**Standard Behavior**: In NDN, Data MUST return along the reverse path of the Interest. Each router must record which face the Interest came from and send Data BACK on that face.

**Current Implementation** (router.py:538-576):

```python
def handle_data(self, G, data: Data, prev_node: int):
    """Process incoming Data packets"""
    ...
    # Forward data to interested nodes from PIT
    interested_nodes = self.PIT.get(str(data.name))
    if interested_nodes:
        logger.debug(f"Router {self.router_id}: Forwarding Data to {len(interested_nodes)} interested nodes")
        for node in interested_nodes:
            if node != prev_node:
                try:
                    self.forward_data(G, data, node)
```

**Issues**:

- The code uses `self.PIT.get()` to retrieve interested nodes
- BUT: A router's PIT should only have entries for Interests that PASSED THROUGH THIS ROUTER
- The check `if node != prev_node` is not sufficient—it doesn't ensure the Interest actually came from `prev_node`
- **Missing**: Proper PIT face tracking with incoming/outgoing faces

**What Should Happen**:

- PIT should track: `{content_name: {incoming_face_1, incoming_face_2, ...}}`
- When Interest comes in, add `prev_node` to this set
- When Interest is forwarded, also track outgoing faces
- On Data arrival, send to ALL incoming faces EXCEPT the one it arrived on

---

### 5. ❌ **Producer Directly Sends Data to Requesting Router**

**Standard Behavior**: Data MUST follow the reverse path of Interest through the network. When a producer generates Data, it goes back hop-by-hop following the PIT entries at each router.

**Current Implementation** (endpoints.py:97-101):

```python
originating_router.submit_message(
    'data',
    (G, data_packet, producer_router_id),
    priority=1
)
```

**Issues**:

- The producer directly sends Data to `originating_router` (the router where Interest came FROM)
- This BYPASSES all intermediate routers on the path
- In standard NDN, Data would be sent to the producer's connected router first, then forwarded backwards
- Intermediate routers cannot cache the Data this way
- The Data never "enters" intermediate routers for potential caching

**What Should Happen**:

- Producer sends Data to its connected router
- That router forwards Data based on PIT entries
- Each intermediate router can decide whether to cache before forwarding

---

### 6. ❌ **Interest Not Properly Added to PIT Before Forwarding**

**Standard Behavior**: An Interest should be added to the PIT BEFORE it's forwarded to the next hop. This ensures the router knows to expect Data.

**Current Implementation** (router.py:514-516):

```python
# Add to PIT and forward to neighbors
self.PIT.add_entry(str(content_name), prev_node, self.router_time)
stats.update(nodes=1)  # Count node traversal
```

**Issues**:

- PIT entry is added ONLY in the fallback case (no FIB entry)
- If a valid FIB entry exists, the code skips this step
- This means: If a router has a route, it doesn't add the Interest to PIT before forwarding!
- When Data returns, there's NO PIT entry to guide it back

**What Should Happen**:

- ALWAYS add Interest to PIT, regardless of FIB match
- Format: `PIT[content_name] = {prev_node, other_faces_with_interest}`
- Then forward to next hops based on FIB

---

### 7. ❌ **Interest Nonce Not Used for Loop Detection**

**Standard Behavior**: Each Interest has a unique `nonce` field. Routers use this to detect and prevent interest loops. If a router sees the same Interest (same name + nonce) twice, it should DROP it.

**Current Implementation**:

- Packet.py defines nonce (line 136): ✅ Present
- Router.py interest handling: ❌ **NEVER checks the nonce**

**Issues**:

- No loop detection mechanism
- A buggy network or misconfigured FIB could cause Interests to loop indefinitely
- The nonce is generated but never validated

**What Should Happen**:

- Track recently seen nonces: `{content_name: set(nonces)}`
- Before processing Interest, check if nonce was seen before
- If duplicate nonce detected, DROP the Interest (loop)

---

### 8. ❌ **Hop Limit Not Properly Checked**

**Standard Behavior**: Each Interest and Data packet has a `hop_limit`. Routers should decrement this on each hop. If hop limit reaches 0, the packet should be dropped.

**Current Implementation**:

- Packet.py defines hop_limit (line 135 for Interest, line 76 for Data): ✅ Present
- Router.py: ❌ **NEVER decrements or checks hop_limit**

**Issues**:

- Hop limit is defined but never used
- Packets could theoretically loop forever if nonce check is also missing
- No protection against runaway flooding

**What Should Happen**:

```python
# On Interest arrival:
if interest.current_hops >= interest.hop_limit:
    logger.info(f"Interest {interest.name} exceeded hop limit; dropping")
    return

# Before forwarding Interest:
interest.current_hops += 1
```

---

### 9. ❌ **Interest Lifetime Not Enforced**

**Standard Behavior**: Interests have a lifetime (default 4 seconds in your code). If an Interest expires before receiving Data, routers should stop forwarding it and can remove PIT entries.

**Current Implementation**:

- Interest.is_expired() method exists (packet.py:144-146): ✅ Present
- Router.py: ❌ **NEVER calls is_expired()**

**Issues**:

- PIT entries accumulate indefinitely
- Even if an Interest times out, the router still holds the PIT entry
- No cleanup mechanism

**What Should Happen**:

- When forwarding Interest, check if expired
- Implement PIT timeout mechanism (background thread or lazy removal)
- Remove stale PIT entries periodically

---

### 10. ❌ **Multi-Face Forwarding Not Compliant**

**Standard Behavior**: NDN routers can forward a single Interest on multiple faces simultaneously (multicast to multiple next hops). When Data returns from ANY face, it satisfies all pending Interests.

**Current Implementation** (router.py:522-534):

```python
for next_node in next_nodes:
    try:
        if G is not None and next_node in G.nodes and 'router' in G.nodes[next_node]:
            self.send_message(
                next_node,
                'interest',
                (G, interest, self.router_id),
                priority=2
            )
```

**Issues**:

- Interest packet is sent by reference, not cloned
- Same Interest object is modified when sent to multiple faces
- Could cause issues if the packet is modified during transmission

**What Should Happen**:

- Clone the Interest for each outgoing face
- Track which faces got the Interest
- Suppress duplicate Interests that return on other faces

---

## Summary of Non-Compliance

| Issue                              | Severity | RFC Section | Status |
| ---------------------------------- | -------- | ----------- | ------ |
| Interest aggregation broken        | CRITICAL | 4.1         | ❌     |
| Data packet not cloned             | CRITICAL | 4.2         | ❌     |
| Cache freshness not tracked        | HIGH     | 4.3         | ❌     |
| Data doesn't follow reverse path   | CRITICAL | 4.2         | ❌     |
| Producer bypasses network          | CRITICAL | 3.0         | ❌     |
| PIT not populated on FIB match     | CRITICAL | 4.1         | ❌     |
| Nonce-based loop detection missing | HIGH     | 5.1         | ❌     |
| Hop limit not enforced             | HIGH     | 3.1         | ❌     |
| Interest lifetime not enforced     | MEDIUM   | 3.2         | ❌     |
| Multi-face forwarding broken       | HIGH     | 4.2         | ❌     |

---

## Recommended Fixes (Priority Order)

1. **FIX: Proper PIT Management**

   - Always add Interest to PIT before forwarding
   - Track incoming and outgoing faces separately
   - Update Data forwarding to use PIT faces

2. **FIX: Producer Integration**

   - Send Data to producer's local router first
   - Let routers forward Data backwards through network

3. **FIX: Packet Cloning**

   - Clone Data and Interest when forwarding to multiple faces
   - Reset hop count for each outgoing interface

4. **FIX: Loop Detection**

   - Implement nonce-based loop detection
   - Track and drop duplicate Interests

5. **FIX: Cache Freshness**
   - Add FreshnessPeriod to Data packets
   - Check freshness before returning cached content

---

## Testing Recommendations

- ✅ Verify PIT has entries after Interest forwarding
- ✅ Confirm Data follows reverse path of Interest
- ✅ Validate cache hit ratio matches theory
- ✅ Test interest aggregation with concurrent requests
- ✅ Check that Nonce prevents loops
- ✅ Verify hop limit stops runaway packets
