//! Find Nash Equilibria
//!
//! Analyze the Prisoner's Dilemma using support enumeration.
//!
//! ```bash
//! cargo run --example nash_equilibrium
//! ```

use machin_game::nash::{support_enumeration, BimatrixGame};
use ndarray::array;

fn main() {
    // Prisoner's Dilemma payoffs
    let game = BimatrixGame::new(
        array![[3.0, 0.0], [5.0, 1.0]], // Player A
        array![[3.0, 5.0], [0.0, 1.0]], // Player B
    );

    let equilibria = support_enumeration(&game);
    println!("Prisoner's Dilemma — Nash Equilibria:");
    for (i, eq) in equilibria.iter().enumerate() {
        println!("  Equilibrium {}: P1={:?}, P2={:?}", i, eq.strategy_a, eq.strategy_b);
    }
    // Nash equilibrium: both defect (strategy index 1)
}
