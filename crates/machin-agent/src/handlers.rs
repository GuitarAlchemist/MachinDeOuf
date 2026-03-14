//! Tool handler implementations — each parses JSON params, calls the underlying crate, returns JSON.

use ndarray::{Array1, Array2};
use serde_json::{json, Value};

use machin_cache::{Cache, CacheConfig};

use std::sync::OnceLock;

/// Global cache instance shared across tool calls.
fn global_cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Cache::new(CacheConfig::default()))
}

// ── helpers ────────────────────────────────────────────────────

fn parse_f64_array(val: &Value, field: &str) -> Result<Vec<f64>, String> {
    val.get(field)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("Missing or invalid field '{}'", field))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| format!("Non-numeric value in '{}'", field)))
        .collect()
}

fn parse_f64_matrix(val: &Value, field: &str) -> Result<Vec<Vec<f64>>, String> {
    val.get(field)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("Missing or invalid field '{}'", field))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| format!("Non-array row in '{}'", field))?
                .iter()
                .map(|v| v.as_f64().ok_or_else(|| format!("Non-numeric value in '{}'", field)))
                .collect()
        })
        .collect()
}

fn vec_to_array1(v: &[f64]) -> Array1<f64> {
    Array1::from_vec(v.to_vec())
}

fn vecs_to_array2(rows: &[Vec<f64>]) -> Result<Array2<f64>, String> {
    if rows.is_empty() {
        return Err("Empty matrix".into());
    }
    let ncols = rows[0].len();
    if rows.iter().any(|r| r.len() != ncols) {
        return Err("Inconsistent row lengths in matrix".into());
    }
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), ncols), flat)
        .map_err(|e| format!("Matrix shape error: {}", e))
}

fn parse_usize(val: &Value, field: &str) -> Result<usize, String> {
    val.get(field)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| format!("Missing or invalid field '{}'", field))
}

fn parse_str<'a>(val: &'a Value, field: &str) -> Result<&'a str, String> {
    val.get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("Missing or invalid field '{}'", field))
}

// ── machin_stats ───────────────────────────────────────────────

pub fn stats(params: Value) -> Result<Value, String> {
    let data = parse_f64_array(&params, "data")?;
    if data.is_empty() {
        return Err("data must not be empty".into());
    }
    let arr = vec_to_array1(&data);

    let mean = machin_math::stats::mean(&arr).map_err(|e| format!("{}", e))?;
    let std_dev = machin_math::stats::std_dev(&arr).map_err(|e| format!("{}", e))?;
    let median = machin_math::stats::median(&arr).map_err(|e| format!("{}", e))?;
    let (min, max) = machin_math::stats::min_max(&arr).map_err(|e| format!("{}", e))?;
    let variance = machin_math::stats::variance(&arr).map_err(|e| format!("{}", e))?;

    Ok(json!({
        "mean": mean,
        "std_dev": std_dev,
        "variance": variance,
        "median": median,
        "min": min,
        "max": max,
        "count": data.len(),
    }))
}

// ── machin_distance ────────────────────────────────────────────

pub fn distance(params: Value) -> Result<Value, String> {
    let a = parse_f64_array(&params, "a")?;
    let b = parse_f64_array(&params, "b")?;
    let metric = parse_str(&params, "metric")?;

    let arr_a = vec_to_array1(&a);
    let arr_b = vec_to_array1(&b);

    let (distance, metric_name) = match metric {
        "euclidean" => {
            let d = machin_math::distance::euclidean(&arr_a, &arr_b)
                .map_err(|e| format!("{}", e))?;
            (d, "euclidean")
        }
        "cosine" => {
            let d = machin_math::distance::cosine_distance(&arr_a, &arr_b)
                .map_err(|e| format!("{}", e))?;
            (d, "cosine")
        }
        "manhattan" => {
            let d = machin_math::distance::manhattan(&arr_a, &arr_b)
                .map_err(|e| format!("{}", e))?;
            (d, "manhattan")
        }
        _ => return Err(format!("Unknown metric: {}", metric)),
    };

    Ok(json!({
        "distance": distance,
        "metric": metric_name,
    }))
}

// ── machin_optimize ────────────────────────────────────────────

pub fn optimize(params: Value) -> Result<Value, String> {
    let func_name = parse_str(&params, "function")?;
    let dimensions = parse_usize(&params, "dimensions")?;
    let method = parse_str(&params, "method")?;
    let max_iter = parse_usize(&params, "max_iter")?;

    if dimensions == 0 {
        return Err("dimensions must be >= 1".into());
    }

    #[allow(clippy::type_complexity)]
    let objective: machin_optimize::traits::ClosureObjective<Box<dyn Fn(&Array1<f64>) -> f64>> =
        match func_name {
            "sphere" => machin_optimize::traits::ClosureObjective {
                f: Box::new(|x: &Array1<f64>| x.mapv(|v| v * v).sum()),
                dimensions,
            },
            "rosenbrock" => machin_optimize::traits::ClosureObjective {
                f: Box::new(|x: &Array1<f64>| {
                    (0..x.len() - 1)
                        .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                        .sum()
                }),
                dimensions,
            },
            "rastrigin" => machin_optimize::traits::ClosureObjective {
                f: Box::new(|x: &Array1<f64>| {
                    let n = x.len() as f64;
                    10.0 * n
                        + x.iter()
                            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                            .sum::<f64>()
                }),
                dimensions,
            },
            _ => return Err(format!("Unknown function: {}", func_name)),
        };

    let result = match method {
        "sgd" => {
            let mut opt = machin_optimize::gradient::SGD::new(0.01);
            let criteria = machin_optimize::convergence::ConvergenceCriteria {
                max_iterations: max_iter,
                tolerance: 1e-8,
            };
            let init = Array1::from_elem(dimensions, 5.0);
            machin_optimize::gradient::minimize(&objective, &mut opt, init, &criteria)
        }
        "adam" => {
            let mut opt = machin_optimize::gradient::Adam::new(0.01);
            let criteria = machin_optimize::convergence::ConvergenceCriteria {
                max_iterations: max_iter,
                tolerance: 1e-8,
            };
            let init = Array1::from_elem(dimensions, 5.0);
            machin_optimize::gradient::minimize(&objective, &mut opt, init, &criteria)
        }
        "pso" => {
            let pso = machin_optimize::pso::ParticleSwarm::new()
                .with_max_iterations(max_iter)
                .with_bounds(-10.0, 10.0)
                .with_seed(42);
            pso.minimize(&objective)
        }
        "annealing" => {
            let sa = machin_optimize::annealing::SimulatedAnnealing::new()
                .with_max_iterations(max_iter)
                .with_seed(42);
            let init = Array1::from_elem(dimensions, 5.0);
            sa.minimize(&objective, init)
        }
        _ => return Err(format!("Unknown method: {}", method)),
    };

    Ok(json!({
        "best_params": result.best_params.to_vec(),
        "best_value": result.best_value,
        "iterations": result.iterations,
        "converged": result.converged,
        "method": method,
        "function": func_name,
    }))
}

// ── machin_linear_regression ───────────────────────────────────

pub fn linear_regression(params: Value) -> Result<Value, String> {
    use machin_supervised::traits::Regressor;

    let x_rows = parse_f64_matrix(&params, "x")?;
    let y_data = parse_f64_array(&params, "y")?;

    let x = vecs_to_array2(&x_rows)?;
    let y = vec_to_array1(&y_data);

    if x.nrows() != y.len() {
        return Err(format!(
            "x has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        ));
    }

    let mut model = machin_supervised::linear_regression::LinearRegression::new();
    model.fit(&x, &y);

    let predictions = model.predict(&x);
    let weights = model.weights.as_ref().map(|w| w.to_vec()).unwrap_or_default();

    Ok(json!({
        "weights": weights,
        "bias": model.bias,
        "predictions": predictions.to_vec(),
    }))
}

// ── machin_kmeans ──────────────────────────────────────────────

pub fn kmeans(params: Value) -> Result<Value, String> {
    use machin_unsupervised::traits::Clusterer;

    let data_rows = parse_f64_matrix(&params, "data")?;
    let k = parse_usize(&params, "k")?;
    let max_iter = parse_usize(&params, "max_iter")?;

    let data = vecs_to_array2(&data_rows)?;

    let mut km = machin_unsupervised::kmeans::KMeans::new(k);
    km.max_iterations = max_iter;
    km.seed = 42;

    let labels = km.fit_predict(&data);
    let centroids: Vec<Vec<f64>> = km
        .centroids
        .as_ref()
        .map(|c| (0..c.nrows()).map(|i| c.row(i).to_vec()).collect())
        .unwrap_or_default();

    let inertia = km.centroids.as_ref().map(|c| {
        machin_unsupervised::kmeans::inertia(&data, &labels, c)
    }).unwrap_or(0.0);

    Ok(json!({
        "labels": labels.to_vec(),
        "centroids": centroids,
        "inertia": inertia,
        "k": k,
    }))
}

// ── machin_fft ─────────────────────────────────────────────────

pub fn fft(params: Value) -> Result<Value, String> {
    let signal = parse_f64_array(&params, "signal")?;
    if signal.is_empty() {
        return Err("signal must not be empty".into());
    }

    let spectrum = machin_signal::fft::rfft(&signal);
    let magnitudes = machin_signal::fft::magnitude_spectrum(&spectrum);
    let n = spectrum.len();
    // Frequency bins assuming sample_rate=1.0 (normalized)
    let frequencies: Vec<f64> = (0..n).map(|k| k as f64 / n as f64).collect();

    Ok(json!({
        "frequencies": frequencies,
        "magnitudes": magnitudes,
        "fft_size": n,
    }))
}

// ── machin_markov ──────────────────────────────────────────────

pub fn markov(params: Value) -> Result<Value, String> {
    let tm_rows = parse_f64_matrix(&params, "transition_matrix")?;
    let steps = parse_usize(&params, "steps")?;

    let tm = vecs_to_array2(&tm_rows)?;
    let chain = machin_graph::markov::MarkovChain::new(tm)?;

    let stationary = chain.stationary_distribution(steps, 1e-10);
    let is_ergodic = chain.is_ergodic(100);

    Ok(json!({
        "stationary_distribution": stationary.to_vec(),
        "n_states": chain.n_states(),
        "is_ergodic": is_ergodic,
    }))
}

// ── machin_viterbi ─────────────────────────────────────────────

pub fn viterbi(params: Value) -> Result<Value, String> {
    let initial_data = parse_f64_array(&params, "initial")?;
    let transition_rows = parse_f64_matrix(&params, "transition")?;
    let emission_rows = parse_f64_matrix(&params, "emission")?;
    let observations: Vec<usize> = params
        .get("observations")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Missing or invalid field 'observations'".to_string())?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| "Non-integer in observations".to_string()).map(|n| n as usize))
        .collect::<Result<Vec<_>, _>>()?;

    let initial = vec_to_array1(&initial_data);
    let transition = vecs_to_array2(&transition_rows)?;
    let emission = vecs_to_array2(&emission_rows)?;

    let hmm = machin_graph::hmm::HiddenMarkovModel::new(initial, transition, emission)?;
    let (path, log_probability) = hmm.viterbi(&observations);

    Ok(json!({
        "path": path,
        "log_probability": log_probability,
    }))
}

// ── machin_search ──────────────────────────────────────────────

pub fn search_info(params: Value) -> Result<Value, String> {
    let algorithm = parse_str(&params, "algorithm")?;

    let info = match algorithm {
        "astar" => json!({
            "algorithm": "A*",
            "description": "A* is a best-first graph search algorithm that finds the shortest path using a heuristic function h(n) to estimate cost to goal. Combines Dijkstra's optimality with greedy best-first speed.",
            "time_complexity": "O(b^d) worst case, often much better with good heuristic",
            "space_complexity": "O(b^d)",
            "optimal": true,
            "complete": true,
            "requires_heuristic": true,
            "properties": ["optimal if h is admissible", "complete if branching factor is finite"]
        }),
        "bfs" => json!({
            "algorithm": "Breadth-First Search",
            "description": "BFS explores all neighbors at the current depth before moving to nodes at the next depth level. Guarantees shortest path in unweighted graphs.",
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V)",
            "optimal": "for unweighted graphs",
            "complete": true,
            "requires_heuristic": false,
            "properties": ["finds shortest path in unweighted graphs", "level-order traversal"]
        }),
        "dfs" => json!({
            "algorithm": "Depth-First Search",
            "description": "DFS explores as far as possible along each branch before backtracking. Memory efficient but does not guarantee shortest path.",
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V) worst case, O(d) with depth limit",
            "optimal": false,
            "complete": "only if graph is finite and no cycles (or with cycle detection)",
            "requires_heuristic": false,
            "properties": ["memory efficient", "good for topological sort", "can detect cycles"]
        }),
        _ => return Err(format!("Unknown algorithm: {}", algorithm)),
    };

    Ok(info)
}

// ── machin_game_nash ───────────────────────────────────────────

pub fn game_nash(params: Value) -> Result<Value, String> {
    let pa_rows = parse_f64_matrix(&params, "payoff_a")?;
    let pb_rows = parse_f64_matrix(&params, "payoff_b")?;

    let payoff_a = vecs_to_array2(&pa_rows)?;
    let payoff_b = vecs_to_array2(&pb_rows)?;

    let game = machin_game::nash::BimatrixGame::new(payoff_a, payoff_b);
    let equilibria = game.support_enumeration();

    let eq_json: Vec<Value> = equilibria
        .iter()
        .map(|e| {
            json!({
                "player_a": e.player_a.to_vec(),
                "player_b": e.player_b.to_vec(),
                "expected_payoff_a": e.expected_payoff_a(&game),
                "expected_payoff_b": e.expected_payoff_b(&game),
            })
        })
        .collect();

    Ok(json!({
        "equilibria": eq_json,
        "count": eq_json.len(),
    }))
}

// ── machin_chaos_lyapunov ──────────────────────────────────────

pub fn chaos_lyapunov(params: Value) -> Result<Value, String> {
    let map = parse_str(&params, "map")?;
    let parameter = params
        .get("parameter")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "Missing or invalid field 'parameter'".to_string())?;
    let iterations = parse_usize(&params, "iterations")?;

    match map {
        "logistic" => {
            let r = parameter;
            let f = move |x: f64| r * x * (1.0 - x);
            let df = move |x: f64| r * (1.0 - 2.0 * x);
            let mle = machin_chaos::lyapunov::mle_1d(f, df, 0.1, iterations, 1000);
            let dynamics = machin_chaos::lyapunov::classify_dynamics(mle, 0.01);

            Ok(json!({
                "lyapunov_exponent": mle,
                "dynamics": format!("{:?}", dynamics),
                "map": "logistic",
                "parameter": r,
                "iterations": iterations,
            }))
        }
        _ => Err(format!("Unknown map: {}", map)),
    }
}

// ── machin_adversarial_fgsm ────────────────────────────────────

pub fn adversarial_fgsm(params: Value) -> Result<Value, String> {
    let input = parse_f64_array(&params, "input")?;
    let gradient = parse_f64_array(&params, "gradient")?;
    let epsilon = params
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "Missing or invalid field 'epsilon'".to_string())?;

    let input_arr = vec_to_array1(&input);
    let grad_arr = vec_to_array1(&gradient);

    let adversarial = machin_adversarial::evasion::fgsm(&input_arr, &grad_arr, epsilon);
    let perturbation = &adversarial - &input_arr;
    let l_inf = perturbation.mapv(f64::abs).iter().cloned().fold(0.0_f64, f64::max);

    Ok(json!({
        "adversarial_input": adversarial.to_vec(),
        "perturbation": perturbation.to_vec(),
        "l_inf_norm": l_inf,
        "epsilon": epsilon,
    }))
}

// ── machin_bloom_filter ────────────────────────────────────────

pub fn bloom_filter(params: Value) -> Result<Value, String> {
    let items: Vec<String> = params
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Missing or invalid field 'items'".to_string())?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();

    let fp_rate = params
        .get("false_positive_rate")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "Missing or invalid field 'false_positive_rate'".to_string())?;

    let mut bf = machin_probabilistic::bloom::BloomFilter::new(items.len().max(1), fp_rate);
    for item in &items {
        bf.insert(item);
    }

    let operation = params
        .get("operation")
        .and_then(|v| v.as_str())
        .unwrap_or("create");

    match operation {
        "check" => {
            let query = parse_str(&params, "query")?;
            let probably_contains = bf.contains(&query.to_string());
            Ok(json!({
                "query": query,
                "probably_contains": probably_contains,
                "items_count": items.len(),
                "estimated_fp_rate": bf.estimated_fp_rate(),
                "bit_size": bf.bit_size(),
            }))
        }
        _ => {
            Ok(json!({
                "created": true,
                "items_count": items.len(),
                "bit_size": bf.bit_size(),
                "estimated_fp_rate": bf.estimated_fp_rate(),
            }))
        }
    }
}

// ── machin_grammar_weights ─────────────────────────────────────

pub fn grammar_weights(params: Value) -> Result<Value, String> {
    use machin_grammar::weighted::{bayesian_update, softmax, WeightedRule};

    // Parse rules array
    let rules_val = params
        .get("rules")
        .and_then(|v| v.as_array())
        .ok_or("Missing or invalid field 'rules'")?;

    let mut rules: Vec<WeightedRule> = rules_val
        .iter()
        .map(|r| {
            let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let alpha = r.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let beta = r.get("beta").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let weight = r.get("weight").and_then(|v| v.as_f64()).unwrap_or(alpha / (alpha + beta));
            let level = r.get("level").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let source = r.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string();
            WeightedRule { id, alpha, beta, weight, level, source }
        })
        .collect();

    // Apply Bayesian update if observation provided
    if let Some(obs) = params.get("observation") {
        let rule_id = obs.get("rule_id").and_then(|v| v.as_str())
            .ok_or("observation.rule_id required")?;
        let success = obs.get("success").and_then(|v| v.as_bool())
            .ok_or("observation.success required")?;

        for rule in &mut rules {
            if rule.id == rule_id {
                *rule = bayesian_update(rule, success);
                break;
            }
        }
    }

    let temperature = params.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let probs = softmax(&rules, temperature);

    let rules_json: Vec<Value> = rules
        .iter()
        .map(|r| json!({
            "id": r.id,
            "alpha": r.alpha,
            "beta": r.beta,
            "weight": r.weight,
            "level": r.level,
            "source": r.source,
        }))
        .collect();

    let probs_json: Vec<Value> = probs
        .iter()
        .map(|(id, p)| json!({ "rule_id": id, "probability": p }))
        .collect();

    Ok(json!({
        "updated_rules": rules_json,
        "probabilities": probs_json,
        "temperature": temperature,
    }))
}

// ── machin_grammar_evolve ──────────────────────────────────────

pub fn grammar_evolve(params: Value) -> Result<Value, String> {
    use machin_grammar::replicator::{simulate, GrammarSpecies};

    let species_val = params
        .get("species")
        .and_then(|v| v.as_array())
        .ok_or("Missing or invalid field 'species'")?;

    let species: Vec<GrammarSpecies> = species_val
        .iter()
        .map(|s| {
            let id = s.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let proportion = s.get("proportion").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let fitness = s.get("fitness").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let is_stable = s.get("is_stable").and_then(|v| v.as_bool()).unwrap_or(false);
            GrammarSpecies { id, proportion, fitness, is_stable }
        })
        .collect();

    let steps = parse_usize(&params, "steps")?;
    let dt = params.get("dt").and_then(|v| v.as_f64()).unwrap_or(0.05);
    let prune_threshold = params.get("prune_threshold").and_then(|v| v.as_f64()).unwrap_or(1e-6);

    let result = simulate(&species, steps, dt, prune_threshold);

    let species_to_json = |s: &GrammarSpecies| json!({
        "id": s.id,
        "proportion": s.proportion,
        "fitness": s.fitness,
        "is_stable": s.is_stable,
    });

    // Return trajectory sampled at most every 10 steps to keep payload manageable
    let sample_rate = (steps / 100).max(1);
    let trajectory_json: Vec<Value> = result.trajectory
        .iter()
        .step_by(sample_rate)
        .map(|step| step.iter().map(species_to_json).collect::<Vec<_>>().into())
        .collect();

    Ok(json!({
        "final_species": result.final_species.iter().map(species_to_json).collect::<Vec<_>>(),
        "trajectory": trajectory_json,
        "ess": result.ess.iter().map(species_to_json).collect::<Vec<_>>(),
        "steps": steps,
        "dt": dt,
    }))
}

// ── machin_grammar_search ──────────────────────────────────────

pub fn grammar_search(params: Value) -> Result<Value, String> {
    use machin_grammar::constrained::{search_derivation, EbnfGrammar};

    let grammar_str = parse_str(&params, "grammar_ebnf")?;
    let grammar = EbnfGrammar::from_str(grammar_str)?;

    let max_iterations = params.get("max_iterations").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
    let exploration = params.get("exploration").and_then(|v| v.as_f64()).unwrap_or(1.41);
    let max_depth = params.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

    let result = search_derivation(grammar, max_iterations, exploration, max_depth, seed);

    let derivation_json: Vec<Value> = result
        .best_derivation
        .iter()
        .map(|(nt, alt)| json!({ "nonterminal": nt, "alternative": alt }))
        .collect();

    Ok(json!({
        "best_derivation": derivation_json,
        "reward": result.reward,
        "iterations": result.iterations,
    }))
}

// ── machin_rotation ────────────────────────────────────────────

pub fn rotation(params: Value) -> Result<Value, String> {
    use machin_rotation::quaternion::Quaternion;

    let op = parse_str(&params, "operation")?;

    match op {
        "quaternion" => {
            let axis = parse_f64_array(&params, "axis")?;
            let angle = params.get("angle").and_then(|v| v.as_f64())
                .ok_or("Missing 'angle'")?;
            if axis.len() != 3 { return Err("axis must have 3 elements".into()); }
            let q = Quaternion::from_axis_angle([axis[0], axis[1], axis[2]], angle);
            Ok(json!({ "w": q.w, "x": q.x, "y": q.y, "z": q.z, "norm": q.norm() }))
        }
        "slerp" => {
            use machin_rotation::slerp::slerp;
            let axis1 = parse_f64_array(&params, "axis")?;
            let angle1 = params.get("angle").and_then(|v| v.as_f64()).ok_or("Missing 'angle'")?;
            let axis2 = parse_f64_array(&params, "axis2")?;
            let angle2 = params.get("angle2").and_then(|v| v.as_f64()).ok_or("Missing 'angle2'")?;
            let t = params.get("t").and_then(|v| v.as_f64()).ok_or("Missing 't'")?;
            if axis1.len() != 3 || axis2.len() != 3 { return Err("axes must have 3 elements".into()); }
            let q0 = Quaternion::from_axis_angle([axis1[0], axis1[1], axis1[2]], angle1);
            let q1 = Quaternion::from_axis_angle([axis2[0], axis2[1], axis2[2]], angle2);
            let r = slerp(&q0, &q1, t);
            Ok(json!({ "w": r.w, "x": r.x, "y": r.y, "z": r.z }))
        }
        "euler_to_quat" => {
            use machin_rotation::euler::{to_quaternion, EulerOrder};
            let roll = params.get("roll").and_then(|v| v.as_f64()).ok_or("Missing 'roll'")?;
            let pitch = params.get("pitch").and_then(|v| v.as_f64()).ok_or("Missing 'pitch'")?;
            let yaw = params.get("yaw").and_then(|v| v.as_f64()).ok_or("Missing 'yaw'")?;
            let q = to_quaternion(roll, pitch, yaw, EulerOrder::XYZ);
            Ok(json!({ "w": q.w, "x": q.x, "y": q.y, "z": q.z, "norm": q.norm() }))
        }
        "quat_to_euler" => {
            use machin_rotation::euler::{from_quaternion, EulerOrder, gimbal_lock_check};
            let qv = parse_f64_array(&params, "quaternion")?;
            if qv.len() != 4 { return Err("quaternion must have 4 elements [w,x,y,z]".into()); }
            let q = Quaternion::new(qv[0], qv[1], qv[2], qv[3]);
            let (roll, pitch, yaw) = from_quaternion(&q, EulerOrder::XYZ);
            Ok(json!({ "roll": roll, "pitch": pitch, "yaw": yaw, "gimbal_lock": gimbal_lock_check(pitch) }))
        }
        "rotate_point" => {
            let axis = parse_f64_array(&params, "axis")?;
            let angle = params.get("angle").and_then(|v| v.as_f64()).ok_or("Missing 'angle'")?;
            let point = parse_f64_array(&params, "point")?;
            if axis.len() != 3 || point.len() != 3 { return Err("axis and point must have 3 elements".into()); }
            let q = Quaternion::from_axis_angle([axis[0], axis[1], axis[2]], angle);
            let rotated = q.rotate_point([point[0], point[1], point[2]]);
            Ok(json!({ "rotated_point": rotated }))
        }
        "rotation_matrix" => {
            use machin_rotation::rotation_matrix::{from_quaternion, is_rotation_matrix};
            let axis = parse_f64_array(&params, "axis")?;
            let angle = params.get("angle").and_then(|v| v.as_f64()).ok_or("Missing 'angle'")?;
            if axis.len() != 3 { return Err("axis must have 3 elements".into()); }
            let q = Quaternion::from_axis_angle([axis[0], axis[1], axis[2]], angle);
            let m = from_quaternion(&q);
            let valid = is_rotation_matrix(&m, 1e-8);
            Ok(json!({ "matrix": m, "valid": valid, "quaternion": { "w": q.w, "x": q.x, "y": q.y, "z": q.z } }))
        }
        _ => Err(format!("Unknown rotation operation: {}", op)),
    }
}

// ── machin_number_theory ──────────────────────────────────────

pub fn number_theory(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;

    match op {
        "sieve" => {
            let limit = parse_usize(&params, "limit")?;
            let primes = machin_number_theory::sieve::sieve_of_eratosthenes(limit);
            Ok(json!({ "primes": primes, "count": primes.len(), "limit": limit }))
        }
        "is_prime" => {
            let n = params.get("n").and_then(|v| v.as_u64()).ok_or("Missing 'n'")?;
            let trial = machin_number_theory::primality::is_prime_trial(n);
            let miller_rabin = machin_number_theory::primality::is_prime_miller_rabin(n, 10);
            Ok(json!({ "n": n, "is_prime_trial": trial, "is_prime_miller_rabin": miller_rabin }))
        }
        "mod_pow" => {
            let base = params.get("base").and_then(|v| v.as_u64()).ok_or("Missing 'base'")?;
            let exp = params.get("exp").and_then(|v| v.as_u64()).ok_or("Missing 'exp'")?;
            let modulus = params.get("modulus").and_then(|v| v.as_u64()).ok_or("Missing 'modulus'")?;
            let result = machin_number_theory::modular::mod_pow(base, exp, modulus);
            Ok(json!({ "result": result, "expression": format!("{}^{} mod {}", base, exp, modulus) }))
        }
        "gcd" => {
            let a = params.get("a").and_then(|v| v.as_u64()).ok_or("Missing 'a'")?;
            let b = params.get("b").and_then(|v| v.as_u64()).ok_or("Missing 'b'")?;
            let g = machin_number_theory::modular::gcd(a, b);
            Ok(json!({ "gcd": g, "a": a, "b": b }))
        }
        "lcm" => {
            let a = params.get("a").and_then(|v| v.as_u64()).ok_or("Missing 'a'")?;
            let b = params.get("b").and_then(|v| v.as_u64()).ok_or("Missing 'b'")?;
            let l = machin_number_theory::modular::lcm(a, b);
            Ok(json!({ "lcm": l, "a": a, "b": b }))
        }
        "mod_inverse" => {
            let a = params.get("a").and_then(|v| v.as_u64()).ok_or("Missing 'a'")?;
            let modulus = params.get("modulus").and_then(|v| v.as_u64()).ok_or("Missing 'modulus'")?;
            let inv = machin_number_theory::modular::mod_inverse(a, modulus);
            Ok(json!({ "inverse": inv, "a": a, "modulus": modulus, "exists": inv.is_some() }))
        }
        "prime_gaps" => {
            let limit = parse_usize(&params, "limit")?;
            let primes = machin_number_theory::sieve::sieve_of_eratosthenes(limit);
            let gaps: Vec<usize> = primes.windows(2).map(|w| w[1] - w[0]).collect();
            let max_gap = gaps.iter().copied().max().unwrap_or(0);
            let avg_gap = if gaps.is_empty() { 0.0 } else { gaps.iter().sum::<usize>() as f64 / gaps.len() as f64 };
            Ok(json!({ "prime_count": primes.len(), "max_gap": max_gap, "avg_gap": avg_gap, "first_10_gaps": &gaps[..gaps.len().min(10)] }))
        }
        _ => Err(format!("Unknown number theory operation: {}", op)),
    }
}

// ── machin_fractal ────────────────────────────────────────────

pub fn fractal(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;

    match op {
        "takagi" => {
            let n_points = parse_usize(&params, "n_points")?;
            let terms = parse_usize(&params, "terms")?;
            let curve = machin_fractal::takagi::takagi_series(n_points, terms);
            let step = 1.0 / (n_points - 1).max(1) as f64;
            let points: Vec<[f64; 2]> = curve.iter().enumerate()
                .map(|(i, &y)| [i as f64 * step, y]).collect();
            Ok(json!({ "points": points, "n_points": n_points, "terms": terms }))
        }
        "hilbert" => {
            let order = params.get("order").and_then(|v| v.as_u64()).ok_or("Missing 'order'")? as u32;
            let points = machin_fractal::space_filling::hilbert_curve(order);
            Ok(json!({ "points": points, "order": order, "n_points": points.len() }))
        }
        "peano" => {
            let order = params.get("order").and_then(|v| v.as_u64()).ok_or("Missing 'order'")? as u32;
            let points = machin_fractal::space_filling::peano_curve(order);
            Ok(json!({ "points": points, "order": order, "n_points": points.len() }))
        }
        "morton_encode" => {
            let x = params.get("x").and_then(|v| v.as_u64()).ok_or("Missing 'x'")? as u32;
            let y = params.get("y").and_then(|v| v.as_u64()).ok_or("Missing 'y'")? as u32;
            let z = machin_fractal::space_filling::morton_encode(x, y);
            Ok(json!({ "z_order": z, "x": x, "y": y }))
        }
        "morton_decode" => {
            let z = params.get("z").and_then(|v| v.as_u64()).ok_or("Missing 'z'")?;
            let (x, y) = machin_fractal::space_filling::morton_decode(z);
            Ok(json!({ "x": x, "y": y, "z_order": z }))
        }
        _ => Err(format!("Unknown fractal operation: {}", op)),
    }
}

// ── machin_sedenion ───────────────────────────────────────────

pub fn sedenion(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;
    let a_vec = parse_f64_array(&params, "a")?;

    match op {
        "multiply" => {
            let b_vec = parse_f64_array(&params, "b")?;
            if a_vec.len() != b_vec.len() { return Err("a and b must have same length".into()); }
            let product = machin_sedenion::cayley_dickson::double_multiply(&a_vec, &b_vec);
            Ok(json!({ "product": product, "dimension": a_vec.len() }))
        }
        "conjugate" => {
            let conj = machin_sedenion::cayley_dickson::double_conjugate(&a_vec);
            Ok(json!({ "conjugate": conj, "dimension": a_vec.len() }))
        }
        "norm" => {
            let n = machin_sedenion::cayley_dickson::double_norm(&a_vec);
            Ok(json!({ "norm": n, "dimension": a_vec.len() }))
        }
        "cayley_dickson_multiply" => {
            let b_vec = parse_f64_array(&params, "b")?;
            let product = machin_sedenion::cayley_dickson::double_multiply(&a_vec, &b_vec);
            Ok(json!({ "product": product, "dimension": a_vec.len() }))
        }
        _ => Err(format!("Unknown sedenion operation: {}", op)),
    }
}

// ── machin_topo ───────────────────────────────────────────────

pub fn topo(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;
    let points_raw = parse_f64_matrix(&params, "points")?;

    let max_dim = params.get("max_dim").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let max_radius = params.get("max_radius").and_then(|v| v.as_f64()).unwrap_or(2.0);

    match op {
        "persistence" => {
            let diagrams = machin_topo::pointcloud::persistence_from_points(&points_raw, max_dim, max_radius);
            let diag_json: Vec<Value> = diagrams.iter().enumerate().map(|(dim, d)| {
                let pairs: Vec<Value> = d.pairs.iter().map(|p| json!({ "birth": p.0, "death": p.1 })).collect();
                json!({ "dimension": dim, "pairs": pairs })
            }).collect();
            Ok(json!({ "diagrams": diag_json, "max_dim": max_dim, "max_radius": max_radius }))
        }
        "betti_at_radius" => {
            let radius = params.get("radius").and_then(|v| v.as_f64()).ok_or("Missing 'radius'")?;
            let betti = machin_topo::pointcloud::betti_at_radius(&points_raw, max_dim, radius);
            Ok(json!({ "betti_numbers": betti, "radius": radius }))
        }
        "betti_curve" => {
            let n_steps = params.get("n_steps").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
            let curve = machin_topo::pointcloud::betti_curve(&points_raw, max_dim, n_steps);
            let curve_json: Vec<Value> = curve.iter().map(|(r, b)| json!({ "radius": r, "betti": b })).collect();
            Ok(json!({ "curve": curve_json, "n_steps": n_steps }))
        }
        _ => Err(format!("Unknown topo operation: {}", op)),
    }
}

// ── machin_category ───────────────────────────────────────────

pub fn category(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;

    match op {
        "monad_laws" => {
            use machin_category::monad::{OptionMonad, Monad};
            let a = params.get("value").and_then(|v| v.as_i64()).unwrap_or(5) as i32;
            let f = |x: i32| -> Option<i32> { Some(x + 1) };
            let g = |x: i32| -> Option<i32> { Some(x * 2) };

            // Left unit: bind(unit(a), f) == f(a)
            let lhs_left: Option<i32> = OptionMonad::bind(OptionMonad::unit(a), f);
            let rhs_left = f(a);
            let left_ok = lhs_left == rhs_left;

            // Right unit: bind(m, unit) == m
            let m = OptionMonad::unit(a);
            let lhs_right: Option<i32> = OptionMonad::bind(m, OptionMonad::unit);
            let right_ok = lhs_right == m;

            // Associativity
            let bind_m_f: Option<i32> = OptionMonad::bind(m, f);
            let lhs_assoc: Option<i32> = match bind_m_f {
                Some(v) => OptionMonad::bind(Some(v), g),
                None => None,
            };
            let rhs_assoc: Option<i32> = OptionMonad::bind(m, |x| {
                match f(x) { Some(v) => OptionMonad::bind(Some(v), g), None => None }
            });
            let assoc_ok = lhs_assoc == rhs_assoc;

            Ok(json!({
                "value": a,
                "left_unit": { "pass": left_ok, "lhs": format!("{:?}", lhs_left), "rhs": format!("{:?}", rhs_left) },
                "right_unit": { "pass": right_ok, "lhs": format!("{:?}", lhs_right), "rhs": format!("{:?}", m) },
                "associativity": { "pass": assoc_ok, "lhs": format!("{:?}", lhs_assoc), "rhs": format!("{:?}", rhs_assoc) },
                "all_pass": left_ok && right_ok && assoc_ok,
            }))
        }
        "free_forgetful" => {
            use machin_category::monad::FreeForgetfulAdj;
            let elements: Vec<i32> = params.get("elements")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|n| n as i32)).collect())
                .unwrap_or_else(|| vec![1, 2, 3]);
            let free = FreeForgetfulAdj::free(&elements);
            let forget = FreeForgetfulAdj::forget(&free);
            let round_trip_ok = elements == forget;
            Ok(json!({
                "input": elements,
                "free": free,
                "forget": forget,
                "round_trip_ok": round_trip_ok,
            }))
        }
        _ => Err(format!("Unknown category operation: {}", op)),
    }
}

// ── machin_nn_forward ─────────────────────────────────────────

pub fn nn_forward(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;

    match op {
        "dense_forward" => {
            use machin_nn::layer::Layer;
            let input_rows = parse_f64_matrix(&params, "input")?;
            let output_size = parse_usize(&params, "output_size")?;
            let input = vecs_to_array2(&input_rows)?;
            let input_size = input.ncols();
            let mut layer = machin_nn::layer::Dense::new(input_size, output_size);
            let output = layer.forward(&input);
            let output_rows: Vec<Vec<f64>> = (0..output.nrows())
                .map(|i| output.row(i).to_vec()).collect();
            Ok(json!({ "output": output_rows, "input_size": input_size, "output_size": output_size }))
        }
        "mse_loss" => {
            let pred_rows = parse_f64_matrix(&params, "input")?;
            let target_rows = parse_f64_matrix(&params, "target")?;
            let pred = vecs_to_array2(&pred_rows)?;
            let target = vecs_to_array2(&target_rows)?;
            let loss = machin_nn::loss::mse_loss(&pred, &target);
            Ok(json!({ "mse_loss": loss }))
        }
        "bce_loss" => {
            let pred_rows = parse_f64_matrix(&params, "input")?;
            let target_rows = parse_f64_matrix(&params, "target")?;
            let pred = vecs_to_array2(&pred_rows)?;
            let target = vecs_to_array2(&target_rows)?;
            let loss = machin_nn::loss::binary_cross_entropy(&pred, &target);
            Ok(json!({ "bce_loss": loss }))
        }
        "sinusoidal_encoding" => {
            let max_len = parse_usize(&params, "max_len")?;
            let d_model = parse_usize(&params, "d_model")?;
            let enc = machin_nn::positional::sinusoidal_encoding(max_len, d_model);
            let rows: Vec<Vec<f64>> = (0..enc.nrows())
                .map(|i| enc.row(i).to_vec()).collect();
            Ok(json!({ "encoding": rows, "max_len": max_len, "d_model": d_model }))
        }
        _ => Err(format!("Unknown nn operation: {}", op)),
    }
}

// ── machin_bandit ─────────────────────────────────────────────

pub fn bandit(params: Value) -> Result<Value, String> {
    use rand::SeedableRng;
    use rand_distr::{Normal, Distribution};

    let algo = parse_str(&params, "algorithm")?;
    let true_means = parse_f64_array(&params, "true_means")?;
    let rounds = parse_usize(&params, "rounds")?;
    let n_arms = true_means.len();

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut total_reward = 0.0;
    let mut arm_pulls = vec![0usize; n_arms];

    match algo {
        "epsilon_greedy" => {
            let epsilon = params.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(0.1);
            let mut bandit = machin_rl::bandit::EpsilonGreedy::new(n_arms, epsilon, 42);
            for _ in 0..rounds {
                let arm = bandit.select_arm();
                let normal = Normal::new(true_means[arm], 1.0).unwrap();
                let reward = normal.sample(&mut rng);
                bandit.update(arm, reward);
                total_reward += reward;
                arm_pulls[arm] += 1;
            }
            Ok(json!({
                "algorithm": "epsilon_greedy",
                "q_values": bandit.q_values,
                "arm_pulls": arm_pulls,
                "total_reward": total_reward,
                "avg_reward": total_reward / rounds as f64,
            }))
        }
        "ucb1" => {
            let mut bandit = machin_rl::bandit::UCB1::new(n_arms);
            for _ in 0..rounds {
                let arm = bandit.select_arm();
                let normal = Normal::new(true_means[arm], 1.0).unwrap();
                let reward = normal.sample(&mut rng);
                bandit.update(arm, reward);
                total_reward += reward;
                arm_pulls[arm] += 1;
            }
            Ok(json!({
                "algorithm": "ucb1",
                "q_values": bandit.q_values,
                "arm_pulls": arm_pulls,
                "total_reward": total_reward,
                "avg_reward": total_reward / rounds as f64,
            }))
        }
        "thompson" => {
            let mut bandit = machin_rl::bandit::ThompsonSampling::new(n_arms, 42);
            for _ in 0..rounds {
                let arm = bandit.select_arm();
                let normal = Normal::new(true_means[arm], 1.0).unwrap();
                let reward = normal.sample(&mut rng);
                bandit.update(arm, reward);
                total_reward += reward;
                arm_pulls[arm] += 1;
            }
            Ok(json!({
                "algorithm": "thompson",
                "means": bandit.means,
                "arm_pulls": arm_pulls,
                "total_reward": total_reward,
                "avg_reward": total_reward / rounds as f64,
            }))
        }
        _ => Err(format!("Unknown bandit algorithm: {}", algo)),
    }
}

// ── machin_evolution ──────────────────────────────────────────

pub fn evolution(params: Value) -> Result<Value, String> {
    let algo = parse_str(&params, "algorithm")?;
    let func_name = parse_str(&params, "function")?;
    let dimensions = parse_usize(&params, "dimensions")?;
    let generations = parse_usize(&params, "generations")?;
    let pop_size = params.get("population_size").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

    #[allow(clippy::type_complexity)]
    let fitness_fn: Box<dyn Fn(&Array1<f64>) -> f64> = match func_name {
        "sphere" => Box::new(|x: &Array1<f64>| x.mapv(|v| v * v).sum()),
        "rosenbrock" => Box::new(|x: &Array1<f64>| {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                .sum()
        }),
        "rastrigin" => Box::new(|x: &Array1<f64>| {
            let n = x.len() as f64;
            10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
        }),
        _ => return Err(format!("Unknown function: {}", func_name)),
    };

    let result = match algo {
        "genetic" => {
            let mutation_rate = params.get("mutation_rate").and_then(|v| v.as_f64()).unwrap_or(0.1);
            let ga = machin_evolution::genetic::GeneticAlgorithm::new()
                .with_population_size(pop_size)
                .with_generations(generations)
                .with_mutation_rate(mutation_rate)
                .with_bounds(-10.0, 10.0)
                .with_seed(42);
            ga.minimize(&fitness_fn, dimensions)
        }
        "differential" => {
            let de = machin_evolution::differential::DifferentialEvolution::new()
                .with_population_size(pop_size)
                .with_generations(generations)
                .with_bounds(-10.0, 10.0)
                .with_seed(42);
            de.minimize(&fitness_fn, dimensions)
        }
        _ => return Err(format!("Unknown evolution algorithm: {}", algo)),
    };

    Ok(json!({
        "algorithm": algo,
        "function": func_name,
        "best_params": result.best_genes.to_vec(),
        "best_fitness": result.best_fitness,
        "generations": result.generations,
        "fitness_history_len": result.fitness_history.len(),
    }))
}

// ── machin_random_forest ──────────────────────────────────────

pub fn random_forest(params: Value) -> Result<Value, String> {
    use machin_ensemble::traits::EnsembleClassifier;

    let x_train_rows = parse_f64_matrix(&params, "x_train")?;
    let y_train_raw: Vec<usize> = params.get("y_train")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'y_train'")?
        .iter()
        .map(|v| v.as_u64().ok_or("Non-integer in y_train").map(|n| n as usize))
        .collect::<Result<Vec<_>, _>>()?;
    let x_test_rows = parse_f64_matrix(&params, "x_test")?;

    let x_train = vecs_to_array2(&x_train_rows)?;
    let y_train = Array1::from_vec(y_train_raw);
    let x_test = vecs_to_array2(&x_test_rows)?;

    let n_trees = params.get("n_trees").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let max_depth = params.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    let mut rf = machin_ensemble::random_forest::RandomForest::new(n_trees, max_depth).with_seed(42);
    rf.fit(&x_train, &y_train);
    let predictions = rf.predict(&x_test);
    let probas = rf.predict_proba(&x_test);
    let proba_rows: Vec<Vec<f64>> = (0..probas.nrows())
        .map(|i| probas.row(i).to_vec()).collect();

    Ok(json!({
        "predictions": predictions.to_vec(),
        "probabilities": proba_rows,
        "n_trees": n_trees,
        "max_depth": max_depth,
    }))
}

// ── machin_cache ───────────────────────────────────────────────

pub fn cache_op(params: Value) -> Result<Value, String> {
    let operation = parse_str(&params, "operation")?;
    let cache = global_cache();

    match operation {
        "set" => {
            let key = parse_str(&params, "key")?;
            let value = params
                .get("value")
                .ok_or_else(|| "Missing field 'value'".to_string())?;
            cache.set(key, value);
            Ok(json!({ "ok": true, "key": key }))
        }
        "get" => {
            let key = parse_str(&params, "key")?;
            let value: Option<Value> = cache.get(key);
            match value {
                Some(v) => Ok(json!({ "key": key, "value": v, "found": true })),
                None => Ok(json!({ "key": key, "value": null, "found": false })),
            }
        }
        "delete" => {
            let key = parse_str(&params, "key")?;
            let deleted = cache.delete(key);
            Ok(json!({ "key": key, "deleted": deleted }))
        }
        "keys" => {
            let pattern = params
                .get("key")
                .and_then(|v| v.as_str())
                .unwrap_or("*");
            let keys = cache.keys(pattern);
            Ok(json!({ "keys": keys, "count": keys.len() }))
        }
        _ => Err(format!("Unknown cache operation: {}", operation)),
    }
}
