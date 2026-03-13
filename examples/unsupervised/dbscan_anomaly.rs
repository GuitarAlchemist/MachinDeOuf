//! Detect Anomalies with DBSCAN
//!
//! Find natural clusters without specifying k, and flag noise points as anomalies.
//!
//! ```bash
//! cargo run --example dbscan_anomaly
//! ```

use machin_unsupervised::dbscan::DBSCAN;
use ndarray::Array2;

fn main() {
    // Two dense clusters + some noise points
    let data = Array2::from_shape_vec(
        (14, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 0.1, 0.2, // cluster A
            5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2, 5.1, 5.2, // cluster B
            50.0, 50.0, // noise
            -20.0, 30.0, // noise
            99.0, -5.0, // noise
            10.0, 80.0, // noise
        ],
    )
    .unwrap();

    let model = DBSCAN::new(0.5, 3); // eps=0.5, min_points=3
    let labels = model.fit(&data);

    println!("Labels: {:?}", labels);
    let noise_count = labels.iter().filter(|&&l| l == 0).count();
    println!("Detected {} noise points (potential anomalies)", noise_count);
}
