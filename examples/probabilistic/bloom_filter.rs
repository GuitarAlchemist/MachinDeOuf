//! Probabilistic Membership Testing
//!
//! Bloom filters for fast set membership with zero false negatives.
//! HyperLogLog for cardinality estimation of massive streams.
//!
//! ```bash
//! cargo run --example bloom_filter
//! ```

use machin_probabilistic::bloom::BloomFilter;
use machin_probabilistic::hyperloglog::HyperLogLog;

fn main() {
    // Bloom filter: fast "is this URL in the blocklist?"
    let mut bloom = BloomFilter::new(10_000, 0.01); // 10k items, 1% FP rate
    bloom.insert("malicious-site.com");
    bloom.insert("phishing-page.net");
    bloom.insert("scam-offer.org");

    println!("Bloom filter:");
    println!(
        "  malicious-site.com: {} (expected: true)",
        bloom.contains("malicious-site.com")
    );
    println!(
        "  safe-site.org:      {} (expected: false)",
        bloom.contains("safe-site.org")
    );
    println!(
        "  phishing-page.net:  {} (expected: true)",
        bloom.contains("phishing-page.net")
    );

    // HyperLogLog: estimate cardinality of massive streams
    let mut hll = HyperLogLog::new(14); // 2^14 = 16384 registers
    for i in 0..100_000 {
        hll.insert(&format!("item-{}", i));
    }
    println!("\nHyperLogLog:");
    println!("  Estimated unique items: ~{}", hll.count());
    println!("  Actual unique items: 100000");
}
