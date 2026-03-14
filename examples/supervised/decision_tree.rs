//! Classify Data with a Decision Tree
//!
//! Train a CART decision tree and predict new samples.
//!
//! ```bash
//! cargo run --example decision_tree
//! ```

use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::traits::Classifier;
use ndarray::{array, Array2};

fn main() {
    let features = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 6.0, 9.0, 1.0, 0.6, 9.0, 11.0],
    )
    .unwrap();
    let labels = array![0, 0, 1, 1, 0, 1];

    let mut tree = DecisionTree::new(5).with_min_samples_split(2);
    tree.fit(&features, &labels);

    let prediction = tree.predict(&features);
    println!("Predictions: {:?}", prediction);

    let proba = tree.predict_proba(&features);
    println!("Probabilities:\n{:.4}", proba);
}
