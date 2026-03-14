//! Bridge: replicator dynamics with pre-computed fitness vectors.
//!
//! The existing [`crate::evolutionary`] module uses a full payoff matrix.
//! Grammar species have *exogenous* fitness scores (e.g. derivation reward)
//! rather than pairwise interactions, so this module accepts a fitness
//! vector directly, reusing the same replicator equation:
//!
//!   `dx_i/dt = x_i * (f_i - f_avg)`

/// One step of replicator dynamics given explicit fitness values.
///
/// `proportions` and `fitness` must have the same length and
/// `proportions` should sum to ~1.
/// Returns new proportions projected onto the simplex.
///
/// ```
/// use ix_game::grammar_replicator::replicator_step_vector;
/// let p = vec![0.5, 0.5];
/// let f = vec![1.0, 0.0];
/// let next = replicator_step_vector(&p, &f, 0.1);
/// assert!(next[0] > 0.5);
/// ```
pub fn replicator_step_vector(proportions: &[f64], fitness: &[f64], dt: f64) -> Vec<f64> {
    let avg: f64 = proportions
        .iter()
        .zip(fitness.iter())
        .map(|(p, f)| p * f)
        .sum();

    let mut new_props: Vec<f64> = proportions
        .iter()
        .zip(fitness.iter())
        .map(|(p, f)| (p + dt * p * (f - avg)).max(0.0))
        .collect();

    let total: f64 = new_props.iter().sum();
    if total > 1e-15 {
        for v in &mut new_props {
            *v /= total;
        }
    }
    new_props
}

/// Run replicator dynamics with a fixed fitness vector for `steps` steps.
///
/// Returns the full trajectory including the initial state.
///
/// ```
/// use ix_game::grammar_replicator::simulate_vector;
/// let p = vec![0.5, 0.5];
/// let f = vec![1.0, 0.0];
/// let traj = simulate_vector(&p, &f, 50, 0.05);
/// assert_eq!(traj.len(), 51);
/// assert!(traj.last().unwrap()[0] > 0.9);
/// ```
pub fn simulate_vector(
    initial_proportions: &[f64],
    fitness: &[f64],
    steps: usize,
    dt: f64,
) -> Vec<Vec<f64>> {
    let mut current = initial_proportions.to_vec();
    let mut trajectory = Vec::with_capacity(steps + 1);
    trajectory.push(current.clone());
    for _ in 0..steps {
        current = replicator_step_vector(&current, fitness, dt);
        trajectory.push(current.clone());
    }
    trajectory
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_fitter_grows() {
        let p = vec![0.5, 0.5];
        let f = vec![2.0, 0.0];
        let next = replicator_step_vector(&p, &f, 0.1);
        assert!(next[0] > 0.5);
        assert!(next[1] < 0.5);
    }

    #[test]
    fn test_step_sums_to_one() {
        let p = vec![0.3, 0.4, 0.3];
        let f = vec![1.0, 0.5, 0.2];
        let next = replicator_step_vector(&p, &f, 0.05);
        let total: f64 = next.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulate_trajectory_length() {
        let p = vec![0.5, 0.5];
        let f = vec![1.0, 0.0];
        let traj = simulate_vector(&p, &f, 10, 0.1);
        assert_eq!(traj.len(), 11);
    }

    #[test]
    fn test_simulate_dominant_wins() {
        let p = vec![0.5, 0.5];
        let f = vec![1.0, 0.0];
        let traj = simulate_vector(&p, &f, 200, 0.05);
        let last = traj.last().unwrap();
        assert!(last[0] > 0.99, "Dominant species should win: {:?}", last);
    }

    #[test]
    fn test_equal_fitness_proportions_unchanged() {
        let p = vec![0.4, 0.3, 0.3];
        let f = vec![1.0, 1.0, 1.0];
        let next = replicator_step_vector(&p, &f, 0.1);
        for (a, b) in p.iter().zip(next.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
