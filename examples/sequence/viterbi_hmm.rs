//! Decode Hidden States with Viterbi
//!
//! Given observations from a weather HMM, find the most likely weather sequence.
//!
//! ```bash
//! cargo run --example viterbi_hmm
//! ```

use machin_graph::hmm::HiddenMarkovModel;
use ndarray::Array2;

fn main() {
    // States: Sunny=0, Rainy=1
    // Observations: Walk=0, Shop=1, Clean=2
    let hmm = HiddenMarkovModel::new(
        ndarray::array![0.6, 0.4], // initial: 60% sunny
        Array2::from_shape_vec(
            (2, 2),
            vec![
                0.7, 0.3, // sunny->sunny=0.7, sunny->rainy=0.3
                0.4, 0.6, // rainy->sunny=0.4, rainy->rainy=0.6
            ],
        )
        .unwrap(),
        Array2::from_shape_vec(
            (2, 3),
            vec![
                0.1, 0.4, 0.5, // sunny: walk=0.1, shop=0.4, clean=0.5
                0.6, 0.3, 0.1, // rainy: walk=0.6, shop=0.3, clean=0.1
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let observations = vec![0, 1, 2, 0]; // walk, shop, clean, walk
    let (path, log_prob) = hmm.viterbi(&observations);
    println!("Observations: walk, shop, clean, walk");
    println!("Most likely weather: {:?}", path);
    println!("Log probability: {:.4}", log_prob);

    // Learn parameters from data with Baum-Welch EM
    let trained = hmm.baum_welch(&observations, 100, 1e-6);
    println!("Trained HMM transition matrix:\n{:.4}", trained.transition);
}
