//! machin - Claude Code ML skill CLI.
//!
//! Exposes machin algorithms as CLI commands for use as Claude Code skills.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "machin", version, about = "ML algorithms for Claude Code skills")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Optimization algorithms
    Optimize {
        /// Algorithm: sgd, adam, annealing, pso, genetic, differential
        #[arg(long)]
        algo: String,

        /// Benchmark function: sphere, rosenbrock, rastrigin
        #[arg(long, default_value = "sphere")]
        function: String,

        /// Number of dimensions
        #[arg(long, default_value = "2")]
        dim: usize,

        /// Maximum iterations
        #[arg(long, default_value = "1000")]
        max_iter: usize,
    },

    /// Supervised learning
    Train {
        /// Model: linear, logistic, knn, naive-bayes
        #[arg(long)]
        model: String,

        /// Path to CSV data file
        #[arg(long)]
        data: Option<String>,
    },

    /// Clustering
    Cluster {
        /// Algorithm: kmeans, dbscan
        #[arg(long)]
        algo: String,

        /// Number of clusters (for kmeans)
        #[arg(long, default_value = "3")]
        k: usize,
    },

    /// Information about available algorithms
    List,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Optimize { algo, function, dim, max_iter } => {
            println!("Running {} on {} (dim={}, max_iter={})", algo, function, dim, max_iter);
            run_optimize(&algo, &function, dim, max_iter);
        }
        Commands::Train { model, data } => {
            println!("Training {} model", model);
            if let Some(path) = data {
                println!("  Data: {}", path);
            }
            println!("  (TODO: implement data loading)");
        }
        Commands::Cluster { algo, k } => {
            println!("Clustering with {} (k={})", algo, k);
            println!("  (TODO: implement data loading)");
        }
        Commands::List => {
            print_algorithms();
        }
    }
}

fn run_optimize(algo: &str, function: &str, dim: usize, max_iter: usize) {
    use machin_optimize::traits::ClosureObjective;
    use machin_math::ndarray::Array1;

    let obj: ClosureObjective<Box<dyn Fn(&Array1<f64>) -> f64>> = match function {
        "sphere" => ClosureObjective {
            f: Box::new(|x: &Array1<f64>| x.mapv(|v| v * v).sum()),
            dimensions: dim,
        },
        "rosenbrock" => ClosureObjective {
            f: Box::new(|x: &Array1<f64>| {
                (0..x.len() - 1)
                    .map(|i| {
                        let xi: f64 = x[i];
                        let xi1: f64 = x[i + 1];
                        100.0 * (xi1 - xi.powi(2)).powi(2) + (1.0 - xi).powi(2)
                    })
                    .sum::<f64>()
            }),
            dimensions: dim,
        },
        "rastrigin" => ClosureObjective {
            f: Box::new(|x: &Array1<f64>| {
                let n = x.len() as f64;
                10.0 * n + x.iter().map(|&xi: &f64| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
            }),
            dimensions: dim,
        },
        _ => {
            eprintln!("Unknown function: {}. Use: sphere, rosenbrock, rastrigin", function);
            return;
        }
    };

    match algo {
        "annealing" => {
            let sa = machin_optimize::annealing::SimulatedAnnealing::new()
                .with_max_iterations(max_iter);
            let initial = Array1::from_elem(dim, 5.0);
            let result = sa.minimize(&obj, initial);
            print_result("Simulated Annealing", &result);
        }
        "pso" => {
            let pso = machin_optimize::pso::ParticleSwarm::new()
                .with_max_iterations(max_iter);
            let result = pso.minimize(&obj);
            print_result("Particle Swarm", &result);
        }
        "genetic" => {
            let ga = machin_evolution::genetic::GeneticAlgorithm::new()
                .with_generations(max_iter);
            let result = ga.minimize(&obj.f, dim);
            println!("\n  Genetic Algorithm:");
            println!("    Best fitness: {:.6}", result.best_fitness);
            println!("    Best params:  {:?}", result.best_genes.to_vec());
            println!("    Generations:  {}", result.generations);
        }
        "differential" => {
            let de = machin_evolution::differential::DifferentialEvolution::new()
                .with_generations(max_iter);
            let result = de.minimize(&obj.f, dim);
            println!("\n  Differential Evolution:");
            println!("    Best fitness: {:.6}", result.best_fitness);
            println!("    Best params:  {:?}", result.best_genes.to_vec());
            println!("    Generations:  {}", result.generations);
        }
        "sgd" | "adam" => {
            use machin_optimize::convergence::ConvergenceCriteria;
            let criteria = ConvergenceCriteria { max_iterations: max_iter, tolerance: 1e-8 };
            let initial = Array1::from_elem(dim, 5.0);

            let result = if algo == "adam" {
                let mut opt = machin_optimize::gradient::Adam::new(0.01);
                machin_optimize::gradient::minimize(&obj, &mut opt, initial, &criteria)
            } else {
                let mut opt = machin_optimize::gradient::SGD::new(0.01);
                machin_optimize::gradient::minimize(&obj, &mut opt, initial, &criteria)
            };
            print_result(algo, &result);
        }
        _ => {
            eprintln!("Unknown algorithm: {}. Use: sgd, adam, annealing, pso, genetic, differential", algo);
        }
    }
}

fn print_result(name: &str, result: &machin_optimize::traits::OptimizeResult) {
    println!("\n  {}:", name);
    println!("    Best value:   {:.6}", result.best_value);
    println!("    Best params:  {:?}", result.best_params.to_vec());
    println!("    Iterations:   {}", result.iterations);
    println!("    Converged:    {}", result.converged);
}

fn print_algorithms() {
    println!("machin - ML algorithms for Claude Code skills\n");
    println!("OPTIMIZATION:");
    println!("  sgd            - Stochastic Gradient Descent");
    println!("  adam           - Adam optimizer");
    println!("  annealing      - Simulated Annealing");
    println!("  pso            - Particle Swarm Optimization");
    println!("  genetic        - Genetic Algorithm");
    println!("  differential   - Differential Evolution");
    println!();
    println!("SUPERVISED LEARNING:");
    println!("  linear         - Linear Regression (OLS)");
    println!("  logistic       - Logistic Regression");
    println!("  knn            - k-Nearest Neighbors");
    println!("  naive-bayes    - Gaussian Naive Bayes");
    println!("  decision-tree  - Decision Tree (CART) [TODO]");
    println!("  svm            - Linear SVM [TODO]");
    println!();
    println!("UNSUPERVISED LEARNING:");
    println!("  kmeans         - K-Means clustering");
    println!("  dbscan         - DBSCAN [TODO]");
    println!("  pca            - PCA [TODO]");
    println!();
    println!("NEURAL NETWORKS:");
    println!("  dense          - Dense layer + backprop");
    println!();
    println!("REINFORCEMENT LEARNING:");
    println!("  epsilon-greedy - Epsilon-Greedy bandit");
    println!("  ucb1           - Upper Confidence Bound");
    println!("  thompson       - Thompson Sampling");
}
