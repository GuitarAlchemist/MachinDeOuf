//! Optimize a Cost Function
//!
//! Find the minimum of the Rosenbrock function using particle swarm optimization.
//!
//! ```bash
//! cargo run --example pso_rosenbrock
//! ```

use ix_optimize::pso::pso_minimize;
use ix_optimize::traits::ClosureObjective;
use ndarray::Array1;

fn main() {
    let objective = ClosureObjective {
        f: Box::new(|x: &Array1<f64>| {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                .sum::<f64>()
        }),
        dimensions: 10,
    };

    // PSO: 30 particles, 1000 iterations
    let result = pso_minimize(&objective, 30, 1000, (-5.0, 5.0), 42);
    println!("Best position: {:?}", result.best_position);
    println!("Best cost: {:.6}", result.best_cost);
}
