//! Axis-angle representation and conversions.

use crate::quaternion::Quaternion;

/// Convert an axis and angle (radians) to a quaternion.
/// The axis does not need to be pre-normalized.
pub fn to_quaternion(axis: [f64; 3], angle: f64) -> Quaternion {
    Quaternion::from_axis_angle(axis, angle)
}

/// Extract the axis and angle from a quaternion.
///
/// Returns `(axis, angle)` where `axis` is a unit vector and `angle` is in `[0, 2π)`.
/// For identity quaternion, returns `([1, 0, 0], 0)`.
pub fn from_quaternion(q: &Quaternion) -> ([f64; 3], f64) {
    let q = q.normalize();
    // Ensure w is non-negative for consistent angle range [0, π]
    let (w, x, y, z) = if q.w < 0.0 {
        (-q.w, -q.x, -q.y, -q.z)
    } else {
        (q.w, q.x, q.y, q.z)
    };

    let angle = 2.0 * w.clamp(-1.0, 1.0).acos();
    let sin_half = (1.0 - w * w).sqrt();

    if sin_half < 1e-12 {
        // Near identity: axis is arbitrary
        return ([1.0, 0.0, 0.0], 0.0);
    }

    let axis = [x / sin_half, y / sin_half, z / sin_half];
    (axis, angle)
}

/// Extract axis and angle from a 3×3 rotation matrix.
///
/// Uses Rodrigues' formula inversion.
pub fn from_rotation_matrix(m: &[[f64; 3]; 3]) -> ([f64; 3], f64) {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();

    if angle.abs() < 1e-12 {
        // Identity rotation
        return ([1.0, 0.0, 0.0], 0.0);
    }

    if (angle - std::f64::consts::PI).abs() < 1e-6 {
        // 180° rotation: extract axis from (M + I) / 2
        // The column with the largest diagonal element gives the axis
        let mut best = 0;
        if m[1][1] > m[best][best] {
            best = 1;
        }
        if m[2][2] > m[best][best] {
            best = 2;
        }

        let mut axis = [0.0; 3];
        axis[best] = ((m[best][best] + 1.0) / 2.0).sqrt();
        let denom = 2.0 * axis[best];
        for i in 0..3 {
            if i != best {
                axis[i] = m[i][best] / denom;
            }
        }
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if len > 1e-12 {
            axis[0] /= len;
            axis[1] /= len;
            axis[2] /= len;
        }
        return (axis, std::f64::consts::PI);
    }

    // General case
    let sin_angle = angle.sin();
    let axis = [
        (m[2][1] - m[1][2]) / (2.0 * sin_angle),
        (m[0][2] - m[2][0]) / (2.0 * sin_angle),
        (m[1][0] - m[0][1]) / (2.0 * sin_angle),
    ];
    (axis, angle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-8;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_identity() {
        let q = to_quaternion([1.0, 0.0, 0.0], 0.0);
        let (_, angle) = from_quaternion(&q);
        assert!(approx_eq(angle, 0.0));
    }

    #[test]
    fn test_180_rotation() {
        let q = to_quaternion([0.0, 0.0, 1.0], PI);
        let (axis, angle) = from_quaternion(&q);
        assert!(approx_eq(angle, PI));
        assert!(approx_eq(axis[2].abs(), 1.0));
    }

    #[test]
    fn test_round_trip() {
        let original_axis = [1.0, 0.0, 0.0];
        let original_angle = 1.23;
        let q = to_quaternion(original_axis, original_angle);
        let (axis, angle) = from_quaternion(&q);
        assert!(approx_eq(angle, original_angle));
        assert!(approx_eq(axis[0], original_axis[0]));
        assert!(approx_eq(axis[1], original_axis[1]));
        assert!(approx_eq(axis[2], original_axis[2]));
    }

    #[test]
    fn test_from_rotation_matrix_identity() {
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (_, angle) = from_rotation_matrix(&m);
        assert!(approx_eq(angle, 0.0));
    }

    #[test]
    fn test_from_rotation_matrix_round_trip() {
        let q = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 0.8);
        let m = q.to_rotation_matrix();
        let (axis, angle) = from_rotation_matrix(&m);
        assert!(approx_eq(angle, 0.8));
        assert!(approx_eq(axis[1], 1.0));
        assert!(approx_eq(axis[0], 0.0));
        assert!(approx_eq(axis[2], 0.0));
    }

    #[test]
    fn test_180_from_matrix() {
        let q = Quaternion::from_axis_angle([0.0, 1.0, 0.0], PI);
        let m = q.to_rotation_matrix();
        let (axis, angle) = from_rotation_matrix(&m);
        assert!(approx_eq(angle, PI));
        assert!(approx_eq(axis[1].abs(), 1.0));
    }
}
