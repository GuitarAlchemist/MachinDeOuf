//! L-system grammar expansion and turtle graphics interpretation.
//!
//! An L-system is a parallel rewriting system consisting of an axiom (initial string)
//! and a set of production rules. The `interpret` function converts the resulting
//! string into a 2D path using turtle graphics commands.
//!
//! # Examples
//!
//! ```
//! use machin_fractal::lsystem;
//!
//! let lsys = lsystem::dragon_curve();
//! let expanded = lsys.expand(3);
//! assert!(!expanded.is_empty());
//!
//! let path = lsystem::interpret(&expanded, 90.0, 1.0);
//! assert!(path.len() > 1);
//! ```

use std::collections::HashMap;

/// An L-system defined by an axiom and production rules.
#[derive(Debug, Clone)]
pub struct LSystem {
    /// The initial string (seed).
    pub axiom: String,
    /// Production rules: each character maps to a replacement string.
    pub rules: HashMap<char, String>,
}

impl LSystem {
    /// Expand the L-system for the given number of iterations.
    ///
    /// Each iteration replaces every character in the current string with its
    /// production rule (or the character itself if no rule exists).
    pub fn expand(&self, iterations: usize) -> String {
        let mut current = self.axiom.clone();

        for _ in 0..iterations {
            let mut next = String::with_capacity(current.len() * 2);
            for ch in current.chars() {
                if let Some(replacement) = self.rules.get(&ch) {
                    next.push_str(replacement);
                } else {
                    next.push(ch);
                }
            }
            current = next;
        }

        current
    }
}

/// 2D turtle state for interpreting L-system strings.
#[derive(Debug, Clone)]
pub struct TurtleState {
    /// Current x position.
    pub x: f64,
    /// Current y position.
    pub y: f64,
    /// Current heading angle in degrees.
    pub angle: f64,
}

/// Interpret an L-system string as turtle graphics commands.
///
/// Commands:
/// - `F` or `A` or `B`: move forward by `step_size` in the current direction, recording the point
/// - `+`: turn left by `angle_delta` degrees
/// - `-`: turn right by `angle_delta` degrees
/// - `[`: push current state onto stack
/// - `]`: pop state from stack
///
/// Returns a vector of `[x, y]` points representing the path (starting from the origin).
pub fn interpret(commands: &str, angle_delta: f64, step_size: f64) -> Vec<[f64; 2]> {
    let mut state = TurtleState {
        x: 0.0,
        y: 0.0,
        angle: 0.0,
    };
    let mut stack: Vec<TurtleState> = Vec::new();
    let mut path = vec![[0.0, 0.0]];

    let deg_to_rad = std::f64::consts::PI / 180.0;

    for ch in commands.chars() {
        match ch {
            'F' | 'A' | 'B' => {
                let rad = state.angle * deg_to_rad;
                state.x += step_size * rad.cos();
                state.y += step_size * rad.sin();
                path.push([state.x, state.y]);
            }
            '+' => {
                state.angle += angle_delta;
            }
            '-' => {
                state.angle -= angle_delta;
            }
            '[' => {
                stack.push(state.clone());
            }
            ']' => {
                if let Some(saved) = stack.pop() {
                    state = saved;
                }
            }
            _ => {}
        }
    }

    path
}

/// Dragon curve L-system.
///
/// Axiom: `"F"`, Rules: `F -> F+G`, `G -> F-G`, Angle: 90 degrees.
pub fn dragon_curve() -> LSystem {
    let mut rules = HashMap::new();
    rules.insert('F', "F+G".to_string());
    rules.insert('G', "F-G".to_string());
    LSystem {
        axiom: "F".to_string(),
        rules,
    }
}

/// Sierpinski arrowhead curve L-system.
///
/// Axiom: `"A"`, Rules: `A -> B-A-B`, `B -> A+B+A`, Angle: 60 degrees.
pub fn sierpinski_arrowhead() -> LSystem {
    let mut rules = HashMap::new();
    rules.insert('A', "B-A-B".to_string());
    rules.insert('B', "A+B+A".to_string());
    LSystem {
        axiom: "A".to_string(),
        rules,
    }
}

/// Koch curve L-system.
///
/// Axiom: `"F"`, Rules: `F -> F+F--F+F`, Angle: 60 degrees.
pub fn koch_curve() -> LSystem {
    let mut rules = HashMap::new();
    rules.insert('F', "F+F--F+F".to_string());
    LSystem {
        axiom: "F".to_string(),
        rules,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_no_iterations() {
        let lsys = dragon_curve();
        let result = lsys.expand(0);
        assert_eq!(result, "F");
    }

    #[test]
    fn test_dragon_curve_one_iteration() {
        let lsys = dragon_curve();
        let result = lsys.expand(1);
        assert_eq!(result, "F+G");
    }

    #[test]
    fn test_dragon_curve_two_iterations() {
        let lsys = dragon_curve();
        let result = lsys.expand(2);
        // F -> F+G, G -> F-G
        // "F+G" -> "F+G" + "+" + "F-G" = "F+G+F-G"
        assert_eq!(result, "F+G+F-G");
    }

    #[test]
    fn test_expansion_length_grows() {
        let lsys = koch_curve();
        let len0 = lsys.expand(0).len();
        let len1 = lsys.expand(1).len();
        let len2 = lsys.expand(2).len();
        assert!(len1 > len0, "iteration 1 should be longer than 0");
        assert!(len2 > len1, "iteration 2 should be longer than 1");
    }

    #[test]
    fn test_sierpinski_arrowhead_one_iteration() {
        let lsys = sierpinski_arrowhead();
        let result = lsys.expand(1);
        assert_eq!(result, "B-A-B");
    }

    #[test]
    fn test_koch_curve_one_iteration() {
        let lsys = koch_curve();
        let result = lsys.expand(1);
        assert_eq!(result, "F+F--F+F");
    }

    #[test]
    fn test_interpret_single_forward() {
        let path = interpret("F", 90.0, 1.0);
        assert_eq!(path.len(), 2);
        assert!((path[0][0] - 0.0).abs() < 1e-10);
        assert!((path[0][1] - 0.0).abs() < 1e-10);
        assert!((path[1][0] - 1.0).abs() < 1e-10);
        assert!((path[1][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpret_turn_and_forward() {
        // Turn left 90 degrees then move forward
        let path = interpret("+F", 90.0, 1.0);
        assert_eq!(path.len(), 2);
        assert!((path[1][0] - 0.0).abs() < 1e-10);
        assert!((path[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpret_square() {
        // Draw a square: F+F+F+F with 90 degree turns
        let path = interpret("F+F+F+F", 90.0, 1.0);
        assert_eq!(path.len(), 5);
        // Should return close to origin
        assert!((path[4][0] - 0.0).abs() < 1e-10);
        assert!((path[4][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_path_generation_non_empty() {
        let lsys = dragon_curve();
        let expanded = lsys.expand(5);
        let path = interpret(&expanded, 90.0, 1.0);
        assert!(path.len() > 1, "path should have more than just the origin");
    }

    #[test]
    fn test_bracket_push_pop() {
        // Forward, push, turn, forward, pop, forward
        // Should result in: (0,0) -> (1,0) -> (1,1) back to (1,0) -> (2,0)
        let path = interpret("F[+F]F", 90.0, 1.0);
        assert_eq!(path.len(), 4); // origin + 3 forwards
        assert!((path[3][0] - 2.0).abs() < 1e-10);
        assert!((path[3][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_string() {
        let path = interpret("", 90.0, 1.0);
        assert_eq!(path.len(), 1); // just origin
    }
}
