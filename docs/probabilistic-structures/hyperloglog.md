# HyperLogLog

## The Problem

You run an analytics platform and need to report the number of unique visitors to each page, every day. Your site sees 200 million page views per day. Storing every visitor ID in a `HashSet` would consume gigabytes of RAM per page. Worse, at the end of the day you need to merge per-server counts into a global total --- and set union on huge hash sets is slow. You need a data structure that can estimate the number of distinct elements in a stream using a **fixed, tiny** amount of memory, and that supports merging across nodes.

Other places this pattern shows up:

- **Unique query counting.** How many distinct search queries did your system handle today?
- **Network monitoring.** How many unique source IPs contacted this server in the last hour?
- **Database cardinality estimation.** Query planners use HLL internally to estimate the number of distinct values in a column, which drives join order decisions.

---

## The Intuition

Flip a coin repeatedly and write down the longest streak of heads you see. If you see 10 heads in a row, you can guess that you must have flipped the coin roughly 2^10 = 1,024 times. A single experiment is noisy, but if you run many experiments in parallel and average the results, the estimate tightens up.

HyperLogLog does exactly this, but with hashes instead of coins:

1. Hash each item to a random-looking 64-bit number.
2. Use the first few bits to pick one of `m` "buckets" (experiments).
3. Count the leading zeros in the remaining bits --- that is the "longest streak of heads."
4. Each bucket remembers only the maximum leading-zero count it has ever seen.

To estimate cardinality, compute the **harmonic mean** of `2^(max leading zeros)` across all buckets, then apply a correction factor.

Key insight: **each bucket stores a single byte** (the max leading-zero count, which never exceeds 64). The entire data structure for standard precision (p=14) is just 16 KB, regardless of whether you are counting 1,000 or 1 billion distinct items.

---

## How It Works

| Symbol | Meaning |
|--------|---------|
| `p` | Precision parameter (4 to 18) |
| `m = 2^p` | Number of buckets (registers) |
| `alpha_m` | Bias-correction constant |

### Add

```
hash = hash64(item)
bucket = hash & (m - 1)            // first p bits
remaining = hash >> p
leading_zeros = clz(remaining) + 1  // count leading zeros
registers[bucket] = max(registers[bucket], leading_zeros)
```

### Count (estimate cardinality)

```
raw_estimate = alpha_m * m^2 / sum(2^(-register[i]) for all i)
```

**In plain English:** take the harmonic mean of the per-bucket estimates, scale by a correction factor, and you get the approximate number of distinct items. For small estimates (less than 2.5 * m), a "small range correction" is applied using the number of zero-valued registers.

### Error rate

```
standard_error = 1.04 / sqrt(m)
```

**In plain English:** with p=14 (16,384 buckets), the typical error is about 0.81%. Doubling the number of buckets (adding one to `p`) cuts the error by roughly 30%, at the cost of doubling memory.

### Merge

To merge two HLL instances with the same precision, take the element-wise maximum of their register arrays. This works because the maximum leading-zero count for a given bucket is the same regardless of which node observed it.

---

## In Rust

### Counting unique visitors

```rust
use ix_probabilistic::hyperloglog::HyperLogLog;

// Standard precision: p=14, ~16 KB memory, ~0.81% error
let mut hll = HyperLogLog::standard();

// Simulate visitor IDs arriving from a stream
for visitor_id in 0..100_000u64 {
    hll.add(&visitor_id);
}

// Adding the same visitor again does not increase the count
for visitor_id in 0..50_000u64 {
    hll.add(&visitor_id);  // duplicates are absorbed
}

let estimate = hll.count();
println!("Estimated unique visitors: {:.0}", estimate);  // ~100,000
println!("Error rate: {:.2}%", hll.error_rate() * 100.0); // ~0.81%
println!("Memory used: {} bytes", hll.memory_bytes());     // 16,384
```

### Custom precision for different workloads

```rust
use ix_probabilistic::hyperloglog::HyperLogLog;

// Low memory (256 bytes), ~6.5% error -- good for rough estimates
let mut hll_small = HyperLogLog::new(8);

// High precision (262,144 bytes), ~0.41% error -- when accuracy matters
let mut hll_precise = HyperLogLog::new(18);
```

### Merging across distributed nodes

```rust
use ix_probabilistic::hyperloglog::HyperLogLog;

let mut server_a = HyperLogLog::new(12);
let mut server_b = HyperLogLog::new(12);

// Each server sees different visitors (with some overlap)
for i in 0..5_000u64    { server_a.add(&i); }
for i in 3_000..8_000u64 { server_b.add(&i); }

// Merge into a global count
server_a.merge(&server_b).expect("same precision required");
let total_unique = server_a.count();
println!("Total unique across both servers: {:.0}", total_unique); // ~8,000
```

---

## When To Use This

| Situation | HyperLogLog | `HashSet` | Bloom Filter | Count-Min Sketch |
|-----------|:-----------:|:---------:|:------------:|:----------------:|
| Count distinct items | Yes | Yes (exact) | No | No |
| Fixed memory regardless of cardinality | Yes | No | Yes | Yes |
| Check membership of specific item | No | Yes | Yes | No |
| Estimate frequency of specific item | No | Yes | No | Yes |
| Merge across nodes | Easy (max registers) | Expensive (set union) | Easy (bitwise OR) | Easy (add tables) |
| Sub-1% error at 16 KB | Yes | N/A | N/A | N/A |

**Rule of thumb:** use HyperLogLog when the question is "how many *different* things have I seen?" rather than "have I seen *this specific* thing?" or "how often did I see *this specific* thing?"

---

## Key Parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| `precision` (p) | Trade-off between memory and accuracy | p=14 (`standard()`) is the sweet spot for most workloads |
| Memory | `2^p` bytes | p=10 gives ~1 KB; p=14 gives ~16 KB; p=18 gives ~256 KB |
| Error rate | `1.04 / sqrt(2^p)` | p=10: ~3.25%; p=14: ~0.81%; p=18: ~0.41% |

### Quick reference

| Precision (p) | Registers | Memory | Typical Error |
|:-:|:-:|:-:|:-:|
| 4 | 16 | 16 B | 26% |
| 8 | 256 | 256 B | 6.5% |
| 10 | 1,024 | 1 KB | 3.25% |
| 12 | 4,096 | 4 KB | 1.63% |
| **14** | **16,384** | **16 KB** | **0.81%** |
| 16 | 65,536 | 64 KB | 0.41% |
| 18 | 262,144 | 256 KB | 0.20% |

---

## Pitfalls

1. **Small cardinalities are noisy.** If the true count is below roughly `2.5 * m`, the raw harmonic-mean estimate is biased. The implementation applies a "linear counting" correction using the number of empty registers, but estimates below a few hundred items will still have higher relative error.

2. **Merging requires the same precision.** `merge()` returns `Err` if the two instances have different `p` values. In a distributed system, standardize on a single precision across all nodes.

3. **You cannot query for specific items.** HLL answers "how many distinct items?" but not "is item X in the set?" For that, use a [Bloom filter](./bloom-filters.md).

4. **You cannot remove items.** Once an item's hash has updated a register, there is no way to undo it. If you need to track items joining and leaving a set, consider maintaining HLLs per time window and expiring old ones.

5. **Hash collisions on small inputs.** Items that produce the same 64-bit hash are indistinguishable. For typical workloads this is negligible (collision probability is ~1 in 2^64), but be aware when counting items drawn from a very small alphabet.

---

## Going Further

- **HyperLogLog++ (Google's improvement)** adds bias correction for small and medium cardinalities and uses sparse representation for registers that are mostly zero. This is the standard in production analytics systems.
- **Sliding-window HLL** maintains multiple HLL sketches for overlapping time windows, letting you answer "how many unique visitors in the last hour?" without storing individual timestamps.
- **Set operations.** The merge operation computes the union. Intersection cardinality can be *estimated* via the inclusion-exclusion principle: `|A inter B| ~ |A| + |B| - |A union B|`, though this estimate has high relative error when the intersection is much smaller than either set.
- Combine with a [Count-Min Sketch](./count-min-sketch.md) to answer both "how many distinct items?" and "how often does each item appear?" --- two complementary views of the same data stream.
