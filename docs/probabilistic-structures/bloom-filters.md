# Bloom Filters

## The Problem

You run a web browser that needs to warn users before they visit a malicious URL. Your blocklist contains 10 million known-bad URLs. You could store them all in a `HashSet`, but that eats hundreds of megabytes of RAM --- and the list has to be checked on every single navigation. You need a data structure that can answer "is this URL in the blocklist?" in constant time, using a fraction of the memory, and you are willing to tolerate a tiny rate of false alarms (flagging a safe site) as long as you **never** miss a truly dangerous one.

Other places this exact pattern shows up:

- **Cache hit checking.** Before querying a slow database, check whether the key *could* exist. If the Bloom filter says "no," skip the query entirely.
- **Deduplicating events.** A streaming pipeline receives millions of events per second. A Bloom filter tells you instantly whether you have already processed a given event ID.
- **Spell-checkers and username registries.** A fast first pass that says "this word is *probably* in the dictionary" or "this username is *probably* taken."

---

## The Intuition

Imagine a wall of 1,000 light switches, all starting in the OFF position. When you want to *remember* an item, you run it through three different hash functions. Each hash gives you a switch number, and you flip those three switches ON.

Later, when you want to check if an item is in the set, you hash it the same way and look at the three switches. If **any** of them is OFF, the item was definitely never added. If **all three** are ON, the item *probably* was added --- but there is a small chance some other items flipped those same switches.

Key insight: **false positives are possible, false negatives are impossible.** If the filter says "no," it means no. If it says "yes," it means "probably yes."

The math lets you trade memory for accuracy. More bits and more hash functions drive the false positive rate down to whatever level you need.

---

## How It Works

A Bloom filter has two parameters:

| Symbol | Meaning |
|--------|---------|
| `m` | Number of bits in the bit array |
| `k` | Number of independent hash functions |

### Insert

For each item, compute `k` hash values, each in the range `[0, m)`. Set those bits to 1.

### Query

Compute the same `k` hashes. If every corresponding bit is 1, return `true` (probably present). If any bit is 0, return `false` (definitely absent).

### Optimal sizing

Given `n` items and desired false positive rate `p`:

```
m = -(n * ln(p)) / (ln2)^2
k = (m / n) * ln2
```

**In plain English:** the number of bits grows linearly with the number of items you plan to store, and logarithmically with how strict you want the false positive rate. A 1% false positive rate for 1 million items costs roughly 1.2 MB.

### Estimated false positive rate at current fill

```
FP_rate ~ (fraction_of_bits_set) ^ k
```

**In plain English:** the more bits that are flipped on, the more likely a random query will hit all `k` of them by accident.

### Union

Two Bloom filters with the **same `m` and `k`** can be merged with a bitwise OR. The result contains every item from both filters. This makes Bloom filters perfect for distributed systems where each node maintains a local filter and you periodically merge them.

---

## In Rust

> Full runnable example: [`examples/probabilistic/bloom_filter.rs`](../../examples/probabilistic/bloom_filter.rs)

### Basic usage --- URL blocklist

```rust
use machin_probabilistic::bloom::BloomFilter;

// Create a filter sized for 10,000 URLs at 1% false positive rate.
// The library computes the optimal bit array size and hash count.
let mut blocklist = BloomFilter::new(10_000, 0.01);

// Populate with known-bad URLs
blocklist.insert(&"malicious-site.com");
blocklist.insert(&"phishing-page.net");
blocklist.insert(&"scam-offer.org");

// Check incoming URLs
assert!(blocklist.contains(&"malicious-site.com"));   // true  -- in the set
assert!(!blocklist.contains(&"safe-site.org"));         // false -- definitely not

// Inspect the filter
println!("Items inserted:  {}", blocklist.len());          // 3
println!("Bit array size:  {}", blocklist.bit_size());     // ~95,851
println!("Est. FP rate:    {:.6}", blocklist.estimated_fp_rate());
```

### Manual parameters

```rust
use machin_probabilistic::bloom::BloomFilter;

// When you know exactly how many bits and hashes you want
let mut bf = BloomFilter::with_params(1024, 5);
bf.insert(&42u64);
assert!(bf.contains(&42u64));
```

### Merging distributed filters

```rust
use machin_probabilistic::bloom::BloomFilter;

let mut node_a = BloomFilter::with_params(10_000, 7);
let mut node_b = BloomFilter::with_params(10_000, 7);

node_a.insert(&"event-001");
node_b.insert(&"event-002");

// Merge into a single filter that knows about both events
let merged = node_a.union(&node_b).expect("same params required");
assert!(merged.contains(&"event-001"));
assert!(merged.contains(&"event-002"));
```

---

## When To Use This

| Situation | Bloom Filter | `HashSet` | Cuckoo Filter |
|-----------|:-----------:|:---------:|:-------------:|
| Memory-constrained membership check | Best | Worst | Good |
| Need zero false negatives | Yes | Yes | Yes |
| Need zero false positives | No | Yes | No |
| Need deletion support | No | Yes | Yes |
| Distributed merge (union) | Easy | Expensive | Not supported |
| Counting frequencies | No | No | No |

**Rule of thumb:** use a Bloom filter when you need a fast, space-efficient "definitely not / probably yes" gate in front of a more expensive lookup.

---

## Key Parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| `capacity` | Expected number of items | Overestimate by 20-50% to keep FP rate low as the filter fills |
| `fp_rate` | Target false positive probability | 0.01 (1%) is a good default; 0.001 for stricter use cases |
| `size` (manual) | Exact number of bits | Use `new()` to let the library compute this for you |
| `num_hashes` (manual) | Number of hash functions | More hashes = lower FP rate but slower insert/query |

---

## Pitfalls

1. **You cannot remove items.** Once a bit is set, it stays set. If you need deletion, use a [`CuckooFilter`](./cuckoo-filters.md) instead.

2. **Overfilling destroys accuracy.** If you insert far more items than `capacity`, the false positive rate climbs sharply. The `estimated_fp_rate()` method lets you monitor this at runtime.

3. **Union requires identical parameters.** Calling `union()` on filters with different `size` or `num_hashes` returns `None`. Design your distributed system so every node creates filters with the same constructor arguments.

4. **Items are not retrievable.** A Bloom filter tells you membership, not contents. You cannot iterate over what was inserted.

5. **Hash quality matters.** The current implementation uses `DefaultHasher` with seed mixing. For cryptographic adversaries who can craft hash collisions, you would need a keyed hash. For typical ML/data-pipeline workloads this is not a concern.

---

## Going Further

- **Counting Bloom filters** replace each bit with a counter, enabling deletion at the cost of more memory. MachinDeOuf does not include one yet, but the `CuckooFilter` covers the deletion use case.
- **Scalable Bloom filters** automatically add new bit arrays as the fill level rises, maintaining a target FP rate without knowing the item count up front.
- **Count-Min Sketch** ([next doc](./count-min-sketch.md)) solves a related but different problem: estimating *how many times* an item appears, not just whether it exists.
- Bloom filters pair naturally with the [`machin-cache`](../../crates/machin-cache) embedded cache: use the filter as a fast admission gate before writing to the cache, avoiding polluting it with one-hit wonders.
