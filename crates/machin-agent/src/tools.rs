//! Tool registry — defines MCP tools and dispatches calls.

use serde_json::{json, Value};

use crate::handlers;

/// An MCP tool definition.
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
    pub handler: fn(Value) -> Result<Value, String>,
}

/// Registry of all available tools.
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut reg = Self { tools: Vec::new() };
        reg.register_all();
        reg
    }

    /// List all tools as MCP tool definitions.
    pub fn list(&self) -> Value {
        let tools: Vec<Value> = self
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                })
            })
            .collect();
        json!({ "tools": tools })
    }

    /// Call a tool by name with the given arguments.
    pub fn call(&self, name: &str, arguments: Value) -> Result<Value, String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("Unknown tool: {}", name))?;
        (tool.handler)(arguments)
    }

    fn register_all(&mut self) {
        self.tools.push(Tool {
            name: "machin_stats",
            description: "Compute statistics (mean, std, min, max, median) on a list of numbers.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "List of numbers to compute statistics on"
                    }
                },
                "required": ["data"]
            }),
            handler: handlers::stats,
        });

        self.tools.push(Tool {
            name: "machin_distance",
            description: "Compute distance between two vectors (euclidean, cosine, or manhattan).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "a": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "First vector"
                    },
                    "b": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Second vector"
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["euclidean", "cosine", "manhattan"],
                        "description": "Distance metric to use"
                    }
                },
                "required": ["a", "b", "metric"]
            }),
            handler: handlers::distance,
        });

        self.tools.push(Tool {
            name: "machin_optimize",
            description: "Minimize a benchmark function (sphere, rosenbrock, rastrigin) using SGD, Adam, PSO, or simulated annealing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "function": {
                        "type": "string",
                        "enum": ["sphere", "rosenbrock", "rastrigin"],
                        "description": "Benchmark function to minimize"
                    },
                    "dimensions": {
                        "type": "integer",
                        "description": "Number of dimensions",
                        "minimum": 1
                    },
                    "method": {
                        "type": "string",
                        "enum": ["sgd", "adam", "pso", "annealing"],
                        "description": "Optimization method"
                    },
                    "max_iter": {
                        "type": "integer",
                        "description": "Maximum iterations",
                        "minimum": 1
                    }
                },
                "required": ["function", "dimensions", "method", "max_iter"]
            }),
            handler: handlers::optimize,
        });

        self.tools.push(Tool {
            name: "machin_linear_regression",
            description: "Fit ordinary least squares linear regression and return weights, bias, and predictions.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Feature matrix (rows=samples, cols=features)"
                    },
                    "y": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Target values"
                    }
                },
                "required": ["x", "y"]
            }),
            handler: handlers::linear_regression,
        });

        self.tools.push(Tool {
            name: "machin_kmeans",
            description: "K-Means clustering with K-Means++ initialization.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Data matrix (rows=samples, cols=features)"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of clusters",
                        "minimum": 1
                    },
                    "max_iter": {
                        "type": "integer",
                        "description": "Maximum iterations",
                        "minimum": 1
                    }
                },
                "required": ["data", "k", "max_iter"]
            }),
            handler: handlers::kmeans,
        });

        self.tools.push(Tool {
            name: "machin_fft",
            description: "Compute the Fast Fourier Transform of a real-valued signal. Returns frequency bins and magnitudes.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Real-valued signal samples"
                    }
                },
                "required": ["signal"]
            }),
            handler: handlers::fft,
        });

        self.tools.push(Tool {
            name: "machin_markov",
            description: "Analyze a Markov chain: compute stationary distribution after a number of steps.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "transition_matrix": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Row-stochastic transition matrix (rows sum to 1)"
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of power-iteration steps for stationary distribution",
                        "minimum": 1
                    }
                },
                "required": ["transition_matrix", "steps"]
            }),
            handler: handlers::markov,
        });

        self.tools.push(Tool {
            name: "machin_viterbi",
            description: "HMM Viterbi decoding: find the most likely hidden state sequence given observations.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "initial": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Initial state distribution (sums to 1)"
                    },
                    "transition": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "State transition matrix (row-stochastic)"
                    },
                    "emission": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Emission probability matrix (row-stochastic)"
                    },
                    "observations": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Observation sequence (indices into emission columns)"
                    }
                },
                "required": ["initial", "transition", "emission", "observations"]
            }),
            handler: handlers::viterbi,
        });

        self.tools.push(Tool {
            name: "machin_search",
            description: "Get information about search algorithms (A*, BFS, DFS) including descriptions and complexity.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["astar", "bfs", "dfs"],
                        "description": "Search algorithm to describe"
                    },
                    "description": {
                        "type": "boolean",
                        "description": "Whether to include a description"
                    }
                },
                "required": ["algorithm"]
            }),
            handler: handlers::search_info,
        });

        self.tools.push(Tool {
            name: "machin_game_nash",
            description: "Find Nash equilibria of a 2-player bimatrix game via support enumeration.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "payoff_a": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Player A payoff matrix"
                    },
                    "payoff_b": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Player B payoff matrix"
                    }
                },
                "required": ["payoff_a", "payoff_b"]
            }),
            handler: handlers::game_nash,
        });

        self.tools.push(Tool {
            name: "machin_chaos_lyapunov",
            description: "Compute the maximal Lyapunov exponent of the logistic map for a given parameter r.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "map": {
                        "type": "string",
                        "enum": ["logistic"],
                        "description": "Map type (currently only 'logistic')"
                    },
                    "parameter": {
                        "type": "number",
                        "description": "Map parameter (r for logistic map, 0 < r <= 4)"
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "Number of iterations for Lyapunov computation",
                        "minimum": 1
                    }
                },
                "required": ["map", "parameter", "iterations"]
            }),
            handler: handlers::chaos_lyapunov,
        });

        self.tools.push(Tool {
            name: "machin_adversarial_fgsm",
            description: "Fast Gradient Sign Method: compute adversarial perturbation of an input given its loss gradient.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Original input vector"
                    },
                    "gradient": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Loss gradient w.r.t. input"
                    },
                    "epsilon": {
                        "type": "number",
                        "description": "Perturbation magnitude"
                    }
                },
                "required": ["input", "gradient", "epsilon"]
            }),
            handler: handlers::adversarial_fgsm,
        });

        self.tools.push(Tool {
            name: "machin_bloom_filter",
            description: "Create a Bloom filter from items and check membership. Returns whether query items are (probably) in the set.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "check"],
                        "description": "Operation: 'create' inserts items, 'check' tests membership"
                    },
                    "items": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Items to insert into the Bloom filter"
                    },
                    "query": {
                        "type": "string",
                        "description": "Item to check membership for (used with 'check')"
                    },
                    "false_positive_rate": {
                        "type": "number",
                        "description": "Desired false positive rate (e.g. 0.01)",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["items", "false_positive_rate"]
            }),
            handler: handlers::bloom_filter,
        });

        self.tools.push(Tool {
            name: "machin_grammar_weights",
            description: "Bayesian (Beta-Binomial) update of grammar rule weights and softmax probability query. \
                Returns updated rule weights and selection probabilities.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "rules": {
                        "type": "array",
                        "description": "Grammar rules with weights",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":     { "type": "string" },
                                "alpha":  { "type": "number" },
                                "beta":   { "type": "number" },
                                "weight": { "type": "number" },
                                "level":  { "type": "integer" },
                                "source": { "type": "string" }
                            },
                            "required": ["id"]
                        }
                    },
                    "observation": {
                        "type": "object",
                        "description": "Optional: apply a Bayesian update to one rule",
                        "properties": {
                            "rule_id": { "type": "string" },
                            "success": { "type": "boolean" }
                        },
                        "required": ["rule_id", "success"]
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Softmax temperature (default 1.0)",
                        "minimum": 0
                    }
                },
                "required": ["rules"]
            }),
            handler: handlers::grammar_weights,
        });

        self.tools.push(Tool {
            name: "machin_grammar_evolve",
            description: "Simulate grammar rule competition via replicator dynamics. \
                Returns the final species proportions, full trajectory, and detected ESS.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "species": {
                        "type": "array",
                        "description": "Initial grammar species",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":          { "type": "string" },
                                "proportion":  { "type": "number" },
                                "fitness":     { "type": "number" },
                                "is_stable":   { "type": "boolean" }
                            },
                            "required": ["id", "proportion", "fitness"]
                        }
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of simulation steps",
                        "minimum": 1
                    },
                    "dt": {
                        "type": "number",
                        "description": "Time step (default 0.05)",
                        "minimum": 0
                    },
                    "prune_threshold": {
                        "type": "number",
                        "description": "Proportion below which species are pruned (default 1e-6)",
                        "minimum": 0
                    }
                },
                "required": ["species", "steps"]
            }),
            handler: handlers::grammar_evolve,
        });

        self.tools.push(Tool {
            name: "machin_grammar_search",
            description: "Grammar-guided MCTS derivation search. \
                Finds the best sentence derivation from an EBNF grammar using Monte Carlo Tree Search.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "grammar_ebnf": {
                        "type": "string",
                        "description": "Grammar in EBNF notation (one rule per line: name ::= alt | alt)"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "MCTS iterations (default 500)",
                        "minimum": 1
                    },
                    "exploration": {
                        "type": "number",
                        "description": "UCB1 exploration constant (default 1.41)",
                        "minimum": 0
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max grammar expansion depth (default 20)",
                        "minimum": 1
                    },
                    "seed": {
                        "type": "integer",
                        "description": "RNG seed for reproducibility (default 42)",
                        "minimum": 0
                    }
                },
                "required": ["grammar_ebnf"]
            }),
            handler: handlers::grammar_search,
        });

        self.tools.push(Tool {
            name: "machin_cache",
            description: "In-memory cache operations: set, get, delete, or list keys.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["set", "get", "delete", "keys"],
                        "description": "Cache operation to perform"
                    },
                    "key": {
                        "type": "string",
                        "description": "Cache key (required for set/get/delete)"
                    },
                    "value": {
                        "description": "Value to store (required for set, any JSON value)"
                    }
                },
                "required": ["operation"]
            }),
            handler: handlers::cache_op,
        });
    }
}
