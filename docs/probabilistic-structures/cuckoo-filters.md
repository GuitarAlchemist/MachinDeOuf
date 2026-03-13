# Cuckoo Filters

## The Problem

You manage a real-time session store for a chat application. When a user comes online, you add them; when they go offline, you remove them. Before routing a message, you need to check "is this user currently online?" A Bloom filter handles the add-and-check part perfectly, but it cannot handle the *remove* part --- once a bit is set, it stays set forever. You need a set-membership structure that supports **insert, lookup, and delete**, all in constant time with constant memory per item.

Other places this pattern shows up:

- **Firewall allow/deny lists with revocation.** Block an IP, then unblock it later without rebuilding the filter.
- **Feature flag rollout.** Add user IDs to a "beta" filter, then remove them when the rollout is complete.
- **Distributed deduplication with corrections.** Mark an event as processed, but retract the mark if you discover the processing failed.

---

## The Intuition

A Cuckoo filter is like an apartment building where every unit can hold up to 4 tenants (fingerprints). When a new tenant arrives:

1. Compute the tenant's **fingerprint** (a short hash) and two possible **apartment numbers** (bucket indices).
2. If either apartment has a vacancy, move in.
3. If both are full, knock on one apartment, evict a random existing tenant, and take their spot. The evicted tenant then tries *their* alternate apartment. This chain of evictions continues until someone finds a vacancy or you hit a kick limit (the building is too full).

To delete, find the tenant's fingerprint in one of the two candidate apartments and remove it.

This "cuckoo hashing" strategy --- named after the cuckoo bird, which lays eggs in other birds' nests --- gives the filter its name.

Key insight: **deletions are possible because we store fingerprints, not just bits.** The trade-off is slightly more memory per item than a Bloom filter, and the possibility that inserts fail when the filter is near capacity.

---

## How It Works

| Concept | Detail |
|---------|--------|
| Fingerprint | A 16-bit hash of the item (nonzero) |
| Bucket | A small array (capacity 4) of fingerprints |
| Bucket count | Next power of two >= `capacity / 4` |
| Alternate index | `i XOR hash(fingerprint)` (allows computing the other bucket from either one) |

### Insert

```
fp = fingerprint(item)
i1 = hash(item) % num_buckets
i2 = i1 XOR hash(fp) % num_buckets

if bucket[i1] has room -> store fp there
else if bucket[i2] has room -> store fp there
else -> kick a random entry from bucket[i1], relocate it, repeat up to max_kicks times
```

**In plain English:** try both candidate buckets. If neither has room, play musical chairs with existing fingerprints until someone finds a spot. If nobody finds a spot after 500 kicks, the filter is too full --- return `false`.

### Contains

```
fp = fingerprint(item)
i1 = hash(item) % num_buckets
i2 = i1 XOR hash(fp) % num_buckets

return bucket[i1].contains(fp) OR bucket[i2].contains(fp)
```

### Delete (remove)

```
fp = fingerprint(item)
i1 = hash(item) % num_buckets
i2 = i1 XOR hash(fp) % num_buckets

if bucket[i1] has fp -> remove it, return true
else if bucket[i2] has fp -> remove it, return true
else -> return false
```

**In plain English:** find the fingerprint in one of the two candidate buckets and remove it. If it is not found, the item was never inserted (or was already deleted).

---

## In Rust

### Session tracking with deletion

```rust
use machin_probabilistic::cuckoo::CuckooFilter;

// Create a filter sized for ~1,000 sessions
let mut sessions = CuckooFilter::new(1000);

// User comes online
assert!(sessions.insert(&"user-alice"));
assert!(sessions.insert(&"user-bob"));

// Check who is online
assert!(sessions.contains(&"user-alice"));  // true
assert!(sessions.contains(&"user-bob"));    // true
assert!(!sessions.contains(&"user-eve"));   // false (never added)

// User goes offline
assert!(sessions.remove(&"user-alice"));    // true (found and removed)
assert!(!sessions.contains(&"user-alice")); // false (no longer present)

println!("Active sessions: {}", sessions.len());          // 1
println!("Load factor: {:.2}", sessions.load_factor());   // low
```

### Handling a full filter

```rust
use machin_probabilistic::cuckoo::CuckooFilter;

let mut cf = CuckooFilter::new(100);

let mut inserted = 0;
for i in 0..200 {
    if cf.insert(&i) {
        inserted += 1;
    } else {
        println!("Filter full after {} insertions", inserted);
        break;
    }
}
// Typically accommodates 85-95% of the theoretical capacity
```

---

## When To Use This

| Situation | Cuckoo Filter | Bloom Filter | `HashSet` |
|-----------|:-------------:|:------------:|:---------:|
| Insert + lookup | Yes | Yes | Yes |
| Delete | Yes | No | Yes |
| Memory efficiency | Good | Best | Worst |
| False positives | Possible | Possible | None |
| False negatives | None | None | None |
| Merge / union | Not supported | Bitwise OR | Set union |
| Fails when full | Yes (insert returns false) | No (FP rate rises) | No (just grows) |

**Rule of thumb:** if you need deletion, use a Cuckoo filter. If you never delete and want the smallest possible memory footprint or need distributed union, use a [Bloom filter](./bloom-filters.md). If you need exactness, use a `HashSet`.

---

## Key Parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| `capacity` | Approximate number of items the filter can hold | The constructor rounds up to a power-of-two bucket count. Expect ~85-95% usable capacity |
| Bucket size | Items per bucket (hardcoded at 4) | 4 is the empirically optimal value in the original paper |
| Max kicks | Relocation attempts before declaring "full" (500) | Higher values squeeze out a few more items but slow down worst-case inserts |
| Fingerprint size | 16 bits (hardcoded) | Determines the false positive rate: ~1/(2 * bucket_size * 2^f) where f=16 |

---

## Pitfalls

1. **Deleting an item you never inserted can corrupt the filter.** If item A and item B happen to share a fingerprint in the same bucket, deleting B will remove A's fingerprint. Only delete items you are certain were inserted.

2. **Duplicate inserts cause problems.** Inserting the same item twice stores two copies of its fingerprint. A single `remove()` call only removes one copy. If you need idempotent inserts, check `contains()` before inserting.

3. **Insert can fail.** Unlike a Bloom filter (which gracefully degrades by raising FP rates), a Cuckoo filter has a hard capacity limit. When `insert()` returns `false`, you must either delete existing items or create a larger filter.

4. **No union operation.** You cannot merge two Cuckoo filters the way you can OR two Bloom filters. For distributed use cases, each node must maintain its own filter, and queries must be fanned out.

5. **Fingerprint collisions.** Two different items with the same fingerprint and the same candidate buckets are indistinguishable. This is the source of false positives. With 16-bit fingerprints and 4 entries per bucket, the theoretical FP rate is approximately 0.0012% (1 in 83,000).

---

## Going Further

- **Semi-sorting Cuckoo filters** sort fingerprints within each bucket, enabling slightly better compression and lower FP rates.
- **Vacuum Cuckoo filters** reduce the average number of kicks during insertion by maintaining auxiliary data about bucket occupancy.
- The original paper, "Cuckoo Filter: Practically Better Than Bloom" (Fan et al., 2014), provides the theoretical analysis and benchmarks comparing with Bloom filters and counting Bloom filters.
- For frequency estimation rather than membership, see the [Count-Min Sketch](./count-min-sketch.md). For cardinality estimation, see [HyperLogLog](./hyperloglog.md).
