# Count-Min Sketch

## The Problem

You operate a network monitoring system that sees 500,000 packets per second. You need to answer questions like "which source IPs are sending the most traffic?" and "is this IP suddenly spiking?" Storing an exact counter for every IP address you have ever seen would require unbounded memory --- and most IPs only appear once or twice. You need a fixed-size data structure that can estimate frequencies on the fly, with a bounded and predictable error.

Other places this pattern shows up:

- **Heavy-hitter detection.** Find the top-N most frequently invoked API endpoints without logging every single request.
- **Cache admission policies.** Only promote an item to cache after it has been requested more than *k* times. A Count-Min Sketch tracks approximate request counts in constant space.
- **NLP term frequency.** Estimate how often each word appears in a corpus too large to fit in a hash map.

---

## The Intuition

Imagine you have five rows of numbered buckets. When a packet arrives from IP `10.0.0.1`, you run five different hash functions --- one per row --- and drop a marble into the bucket each hash points to. To estimate how many packets came from that IP, you look at the five buckets and take the **minimum** marble count.

Why the minimum? Because buckets can share marbles with *other* IPs (hash collisions). The minimum is the one least inflated by collisions, so it gives the closest estimate to the true count.

Key insight: **the sketch only overcounts, never undercounts.** The true frequency is always less than or equal to the estimate.

---

## How It Works

A Count-Min Sketch is a 2D array with `depth` rows and `width` columns.

| Symbol | Meaning |
|--------|---------|
| `w` | Width (columns per row) |
| `d` | Depth (number of rows / hash functions) |

### Add

For item `x`, increment `table[row][hash_row(x) % w]` for each of the `d` rows.

### Estimate

For item `x`, return `min over all rows of table[row][hash_row(x) % w]`.

### Sizing from error bounds

Given desired error factor `epsilon` and failure probability `delta`:

```
w = ceil(e / epsilon)
d = ceil(ln(1 / delta))
```

**In plain English:** the width controls accuracy (wider = less overcounting) and the depth controls confidence (more rows = higher probability that at least one row gives a tight estimate). A sketch with `epsilon = 0.001` and `delta = 0.01` uses about 2,718 columns and 5 rows --- roughly 54 KB for `u64` counters.

### Error bound

```
estimate(x) <= true_count(x) + epsilon * total_count
```

**In plain English:** the overcount on any single item is at most a small fraction (`epsilon`) of the total number of items ever inserted, and this guarantee holds with probability at least `1 - delta`.

---

## In Rust

### Basic frequency tracking

```rust
use ix_probabilistic::count_min::CountMinSketch;

// 100 columns, 5 hash functions
let mut sketch = CountMinSketch::new(100, 5);

// Record observations
for _ in 0..1000 { sketch.add(&"GET /api/users"); }
for _ in 0..50  { sketch.add(&"GET /api/admin"); }
for _ in 0..3   { sketch.add(&"DELETE /api/users/42"); }

// Estimate frequencies (always >= true count)
println!("/api/users:  ~{}", sketch.estimate(&"GET /api/users"));    // >= 1000
println!("/api/admin:  ~{}", sketch.estimate(&"GET /api/admin"));    // >= 50
println!("total:        {}", sketch.total_count());                   // 1053
```

### Sizing from error requirements

```rust
use ix_probabilistic::count_min::CountMinSketch;

// "I want estimates within 1% of total count, 99% of the time"
let mut sketch = CountMinSketch::with_error(0.01, 0.01);

for _ in 0..10_000 { sketch.add(&"frequent"); }
for _ in 0..10     { sketch.add(&"rare"); }

let est = sketch.estimate(&"frequent");
// est >= 10_000 and est <= 10_000 + 0.01 * 10_010 (with 99% probability)
```

### Adding specific counts and merging

```rust
use ix_probabilistic::count_min::CountMinSketch;

let mut sketch = CountMinSketch::new(200, 5);
sketch.add_count(&"batch-event", 500);  // Add 500 at once

// Merge sketches from two monitoring nodes
let mut node_a = CountMinSketch::new(200, 5);
let mut node_b = CountMinSketch::new(200, 5);
node_a.add(&"error-503");
node_b.add(&"error-503");
node_b.add(&"error-503");

node_a.merge(&node_b).expect("same dimensions required");
assert!(node_a.estimate(&"error-503") >= 3);
```

---

## When To Use This

| Situation | Count-Min Sketch | `HashMap<K, u64>` | HyperLogLog |
|-----------|:----------------:|:------------------:|:-----------:|
| Estimate frequency of specific items | Yes | Yes (exact) | No |
| Fixed memory regardless of item count | Yes | No | Yes |
| Count distinct items | No | Yes | Yes |
| Supports deletion / decrement | No | Yes | No |
| Merge across distributed nodes | Yes | Expensive | Yes |
| Error is always in one direction (overcount) | Yes | N/A | N/A |

**Rule of thumb:** use a Count-Min Sketch when you care about *how many times* a specific item appeared, not just *whether* it appeared (Bloom filter) or *how many distinct items* there are (HyperLogLog).

---

## Key Parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| `width` | Columns per row; controls accuracy | Higher = less overcounting. `ceil(e / epsilon)` for formal bound |
| `depth` | Number of hash rows; controls confidence | Higher = more likely estimate is tight. `ceil(ln(1/delta))` |
| `epsilon` (via `with_error`) | Maximum error as fraction of total count | 0.01 (1%) is a solid starting point |
| `delta` (via `with_error`) | Probability that error exceeds epsilon | 0.01 (1%) means 99% confidence |

---

## Pitfalls

1. **Overcounting is proportional to total insertions.** If you have inserted 10 million items, even a rare item's estimate could be inflated by up to `epsilon * 10,000,000`. Size the sketch wide enough for your workload.

2. **You cannot decrement or delete.** The sketch only supports additive updates. If you need to track items that come and go, pair it with a separate mechanism.

3. **Merging requires identical dimensions.** Both sketches must have the same `width` and `depth`. The `merge()` method returns `Err` if they differ.

4. **Low-frequency items get drowned out.** If one item accounts for 99% of traffic, the overcount on rare items can approach the sketch's error bound. Consider combining with a separate "heavy hitters" list for the top items.

5. **Not a set membership test.** A Count-Min Sketch will return a nonzero estimate for items that were never inserted (because of hash collisions). If you need "is this item present at all?" with a guarantee on false negatives, use a [Bloom filter](./bloom-filters.md).

---

## Going Further

- **Heavy-hitter detection with Count-Min + heap:** maintain a min-heap of the top-k items. On each insertion, query the sketch and promote the item into the heap if its estimated count exceeds the current k-th largest.
- **Sliding-window sketches** maintain multiple sketches for time windows and expire old ones, giving you "requests in the last 5 minutes" estimates.
- **Conservative update** only increments the minimum counter(s) among the `d` rows, reducing overcounting at no extra memory cost. A potential future enhancement to `ix-probabilistic`.
- Pair with the [`ix-cache`](../../crates/ix-cache) embedded cache: use the sketch to implement a TinyLFU admission policy --- only admit items to cache whose estimated frequency exceeds a threshold.
