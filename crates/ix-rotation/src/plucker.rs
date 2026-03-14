//! Plücker line coordinates for 3D line geometry.

/// A line in 3D space represented in Plücker coordinates.
///
/// A line through two points P1 and P2 has:
/// - `direction`: P2 - P1 (unnormalized direction vector)
/// - `moment`: P1 × P2 (cross product of the two points)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PluckerLine {
    /// Direction vector of the line.
    pub direction: [f64; 3],
    /// Moment vector (P1 × direction for a line through P1).
    pub moment: [f64; 3],
}

impl PluckerLine {
    /// Create a Plücker line from two points.
    pub fn from_two_points(p1: [f64; 3], p2: [f64; 3]) -> Self {
        let direction = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let moment = cross(&p1, &p2);
        Self { direction, moment }
    }

    /// Reciprocal product of two Plücker lines.
    ///
    /// This is zero when the lines are coplanar (intersect or are parallel).
    pub fn reciprocal_product(l1: &PluckerLine, l2: &PluckerLine) -> f64 {
        dot(&l1.direction, &l2.moment) + dot(&l2.direction, &l1.moment)
    }

    /// Check if two lines intersect (are coplanar) within a tolerance.
    pub fn lines_intersect(l1: &PluckerLine, l2: &PluckerLine, tol: f64) -> bool {
        Self::reciprocal_product(l1, l2).abs() < tol
    }

    /// Compute the closest distance between two lines.
    ///
    /// For intersecting lines this is 0. For parallel lines, computes the
    /// perpendicular distance.
    pub fn closest_distance(l1: &PluckerLine, l2: &PluckerLine) -> f64 {
        let d_cross = cross(&l1.direction, &l2.direction);
        let d_cross_norm = norm(&d_cross);

        if d_cross_norm < 1e-12 {
            // Lines are parallel: distance = |moment_diff × direction| / |direction|^2
            // More precisely, pick a point on each line and compute perpendicular distance.
            let d_norm = norm(&l1.direction);
            if d_norm < 1e-12 {
                return 0.0;
            }
            // For parallel lines, distance = |l1.moment - (l1.direction·l2.direction/|l1.direction|^2)*l2.moment| type formula
            // Use: dist = |l1.moment × l1.direction - l2.moment × l2.direction| ... is complex.
            // Simpler: the reciprocal product is 0 for parallel lines.
            // Distance between parallel lines l1 (through origin + moment/|d|^2 direction) and l2:
            // Use the formula: d = |d1 × (p2 - p1)| / |d1|
            // where p1 is a point on l1 and p2 a point on l2.

            // Point on line: p = (direction × moment) / |direction|^2
            let d1_sq = d_norm * d_norm;
            let p1 = cross_div(&l1.direction, &l1.moment, d1_sq);
            let d2_norm = norm(&l2.direction);
            if d2_norm < 1e-12 {
                return 0.0;
            }
            let d2_sq = d2_norm * d2_norm;
            let p2 = cross_div(&l2.direction, &l2.moment, d2_sq);

            let diff = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
            let c = cross(&l1.direction, &diff);
            return norm(&c) / d_norm;
        }

        // General case: distance = |reciprocal_product| / |d1 × d2|
        Self::reciprocal_product(l1, l2).abs() / d_cross_norm
    }
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Compute (a × b) / divisor component-wise.
fn cross_div(a: &[f64; 3], b: &[f64; 3], divisor: f64) -> [f64; 3] {
    let c = cross(a, b);
    [c[0] / divisor, c[1] / divisor, c[2] / divisor]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_coplanar_intersecting_lines() {
        // Two lines in the XY plane that intersect at the origin
        let l1 = PluckerLine::from_two_points([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let l2 = PluckerLine::from_two_points([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(PluckerLine::lines_intersect(&l1, &l2, 1e-10));
        assert!(approx_eq(PluckerLine::closest_distance(&l1, &l2), 0.0));
    }

    #[test]
    fn test_parallel_lines() {
        // Two parallel lines along X, separated by 1 in Y
        let l1 = PluckerLine::from_two_points([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let l2 = PluckerLine::from_two_points([0.0, 1.0, 0.0], [1.0, 1.0, 0.0]);
        // Parallel lines are coplanar
        assert!(PluckerLine::lines_intersect(&l1, &l2, 1e-10));
        let d = PluckerLine::closest_distance(&l1, &l2);
        assert!(approx_eq(d, 1.0), "expected 1.0, got {}", d);
    }

    #[test]
    fn test_skew_lines() {
        // Line along X-axis at z=0, line along Y-axis at z=1
        let l1 = PluckerLine::from_two_points([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let l2 = PluckerLine::from_two_points([0.0, 0.0, 1.0], [0.0, 1.0, 1.0]);
        assert!(!PluckerLine::lines_intersect(&l1, &l2, 1e-10));
        let d = PluckerLine::closest_distance(&l1, &l2);
        assert!(approx_eq(d, 1.0), "expected 1.0, got {}", d);
    }

    #[test]
    fn test_reciprocal_product_zero_for_intersecting() {
        // Lines that meet at (1,1,0)
        let l1 = PluckerLine::from_two_points([0.0, 0.0, 0.0], [1.0, 1.0, 0.0]);
        let l2 = PluckerLine::from_two_points([2.0, 0.0, 0.0], [1.0, 1.0, 0.0]);
        let rp = PluckerLine::reciprocal_product(&l1, &l2);
        assert!(approx_eq(rp, 0.0));
    }

    #[test]
    fn test_from_two_points_direction() {
        let l = PluckerLine::from_two_points([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert!(approx_eq(l.direction[0], 3.0));
        assert!(approx_eq(l.direction[1], 3.0));
        assert!(approx_eq(l.direction[2], 3.0));
    }
}
