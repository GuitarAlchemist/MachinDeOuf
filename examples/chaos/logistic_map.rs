//! Detect Chaos in Time Series
//!
//! Check if a dynamical system is chaotic using Lyapunov exponents.
//!
//! ```bash
//! cargo run --example logistic_map
//! ```

use ix_chaos::attractors::integrate;
use ix_chaos::bifurcation::bifurcation_diagram;
use ix_chaos::lyapunov::{classify_dynamics, mle_1d, DynamicsType};

fn main() {
    // Logistic map: is r=3.9 chaotic?
    let logistic = |x: f64, r: f64| r * x * (1.0 - x);
    let lyapunov = mle_1d(|x| logistic(x, 3.9), 0.1, 10000, 1000);
    println!("Lyapunov exponent at r=3.9: {:.4}", lyapunov);

    match classify_dynamics(lyapunov) {
        DynamicsType::Chaotic => println!("System is chaotic!"),
        DynamicsType::Periodic => println!("System is periodic"),
        DynamicsType::FixedPoint => println!("System converges to a fixed point"),
    }

    // Compare different r values
    for &r in &[2.5, 3.2, 3.5, 3.9] {
        let le = mle_1d(|x| logistic(x, r), 0.1, 10000, 1000);
        println!("r={:.1}: Lyapunov={:.4} -> {:?}", r, le, classify_dynamics(le));
    }
}
