//! Search with A* and Q*
//!
//! Compare hand-crafted vs learned heuristic search on a grid.
//!
//! ```bash
//! cargo run --example astar_qstar
//! ```

use machin_search::astar::{astar, SearchState};
use machin_search::qstar::{compare_qstar_vs_astar, qstar_search, TabularQ};

// A simple grid state for demonstration
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GridPos {
    x: i32,
    y: i32,
    goal_x: i32,
    goal_y: i32,
}

impl GridPos {
    fn manhattan_distance(&self) -> f64 {
        ((self.x - self.goal_x).abs() + (self.y - self.goal_y).abs()) as f64
    }
}

impl SearchState for GridPos {
    fn is_goal(&self) -> bool {
        self.x == self.goal_x && self.y == self.goal_y
    }

    fn successors(&self) -> Vec<(Self, f64)> {
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        dirs.iter()
            .filter_map(|&(dx, dy)| {
                let nx = self.x + dx;
                let ny = self.y + dy;
                if nx >= 0 && nx <= self.goal_x && ny >= 0 && ny <= self.goal_y {
                    Some((
                        GridPos {
                            x: nx,
                            y: ny,
                            goal_x: self.goal_x,
                            goal_y: self.goal_y,
                        },
                        1.0,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}

fn main() {
    let start = GridPos {
        x: 0,
        y: 0,
        goal_x: 10,
        goal_y: 10,
    };

    // A* with Manhattan distance heuristic
    let a_result = astar(start.clone(), |s| s.manhattan_distance());
    match a_result {
        Some(r) => println!("A*: path length={}, nodes expanded={}", r.path.len(), r.nodes_expanded),
        None => println!("A*: no path found"),
    }

    // Q* with learned heuristic
    let q = TabularQ::new(10.0);
    let q_result = qstar_search(start.clone(), &q);
    match q_result {
        Some(r) => println!("Q*: path length={}, nodes expanded={}", r.path.len(), r.nodes_expanded),
        None => println!("Q*: no path found"),
    }
}
