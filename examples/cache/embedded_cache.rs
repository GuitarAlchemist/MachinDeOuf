//! Embedded Cache with TTL
//!
//! Fast in-memory caching with expiration, Redis-style data structures, and pub/sub.
//!
//! ```bash
//! cargo run --example embedded_cache
//! ```

use machin_cache::store::{Cache, CacheConfig};
use serde_json::json;
use std::time::Duration;

fn main() {
    let cache = Cache::new(CacheConfig {
        num_shards: 16,
        max_capacity: 100_000,
        default_ttl: Some(Duration::from_secs(300)), // 5 min TTL
    });

    // Basic key-value
    cache.set("user:1234", &json!({"name": "Alice", "score": 95}));
    let user: serde_json::Value = cache.get("user:1234").unwrap();
    println!("User: {}", user);

    // Atomic counters
    cache.incr("page:views", 1);
    cache.incr("page:views", 1);
    let views: i64 = cache.get("page:views").unwrap();
    println!("Page views: {}", views);

    // Hash maps, lists, sets -- Redis-style
    cache.hset("session:abc", "token", &"xyz789".to_string());
    cache.lpush("queue:tasks", &"process-image".to_string());
    cache.sadd("tags:post:1", &"rust".to_string());

    // Pub/sub
    let sub = cache.subscribe("events");
    cache.publish("events", &"new-deployment".to_string());
    let msg = sub.recv().unwrap();
    println!("Received event: {}", msg);
}
