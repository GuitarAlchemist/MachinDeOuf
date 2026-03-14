//! Cluster Data with K-Means
//!
//! Segment data into k groups using K-Means clustering.
//!
//! ```bash
//! cargo run --example kmeans_clustering
//! ```

use ix_unsupervised::kmeans::KMeans;
use ndarray::Array2;

fn main() {
    // Generate 3 clusters of 2D points
    let data = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 0.5, 0.5, 0.2, 0.3, 0.1, 0.4, // cluster 0
            5.0, 5.0, 5.5, 5.5, 5.2, 5.3, 5.1, 5.4, // cluster 1
            10.0, 0.0, 10.5, 0.5, 10.2, 0.3, 10.1, 0.4, // cluster 2
        ],
    )
    .unwrap();

    let model = KMeans::new(3, 100, 42); // k=3, max_iter=100, seed=42
    let assignments = model.fit(&data);

    println!("Cluster assignments: {:?}", assignments);
    for c in 0..3 {
        let count = assignments.iter().filter(|&&a| a == c).count();
        println!("Cluster {}: {} points", c, count);
    }
}
