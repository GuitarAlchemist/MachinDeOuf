//! Spherical linear interpolation (SLERP) for quaternions.

use crate::quaternion::Quaternion;

/// Spherical linear interpolation between two unit quaternions.
///
/// `t` ranges from 0.0 (returns `q0`) to 1.0 (returns `q1`).
/// Handles the near-parallel case by falling back to normalized lerp.
pub fn slerp(q0: &Quaternion, q1: &Quaternion, t: f64) -> Quaternion {
    let mut dot = q0.dot(q1);

    // If the dot product is negative, negate one quaternion to take the short path.
    let mut q1_adj = *q1;
    if dot < 0.0 {
        q1_adj = Quaternion::new(-q1.w, -q1.x, -q1.y, -q1.z);
        dot = -dot;
    }

    // Clamp for numerical stability
    if dot > 1.0 {
        dot = 1.0;
    }

    // Near-parallel: fall back to normalized linear interpolation
    if dot > 0.9995 {
        let result = Quaternion::new(
            q0.w + t * (q1_adj.w - q0.w),
            q0.x + t * (q1_adj.x - q0.x),
            q0.y + t * (q1_adj.y - q0.y),
            q0.z + t * (q1_adj.z - q0.z),
        );
        return result.normalize();
    }

    let theta = dot.acos();
    let sin_theta = theta.sin();
    let s0 = ((1.0 - t) * theta).sin() / sin_theta;
    let s1 = (t * theta).sin() / sin_theta;

    Quaternion::new(
        s0 * q0.w + s1 * q1_adj.w,
        s0 * q0.x + s1 * q1_adj.x,
        s0 * q0.y + s1 * q1_adj.y,
        s0 * q0.z + s1 * q1_adj.z,
    )
}

/// Sample `n` evenly-spaced quaternions along the SLERP path from `q0` to `q1`.
///
/// Returns a vector of `n` quaternions at t = 0, 1/(n-1), 2/(n-1), ..., 1.
/// If `n` is 0, returns an empty vector. If `n` is 1, returns `q0`.
pub fn slerp_array(q0: &Quaternion, q1: &Quaternion, n: usize) -> Vec<Quaternion> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![*q0];
    }
    let step = 1.0 / (n - 1) as f64;
    (0..n).map(|i| slerp(q0, q1, i as f64 * step)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn quat_approx_eq(a: &Quaternion, b: &Quaternion) -> bool {
        // Quaternions q and -q represent the same rotation
        let same = approx_eq(a.w, b.w)
            && approx_eq(a.x, b.x)
            && approx_eq(a.y, b.y)
            && approx_eq(a.z, b.z);
        let negated = approx_eq(a.w, -b.w)
            && approx_eq(a.x, -b.x)
            && approx_eq(a.y, -b.y)
            && approx_eq(a.z, -b.z);
        same || negated
    }

    #[test]
    fn test_slerp_t0_gives_q0() {
        let q0 = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 0.5);
        let q1 = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 1.0);
        let r = slerp(&q0, &q1, 0.0);
        assert!(quat_approx_eq(&r, &q0));
    }

    #[test]
    fn test_slerp_t1_gives_q1() {
        let q0 = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 0.5);
        let q1 = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 1.0);
        let r = slerp(&q0, &q1, 1.0);
        assert!(quat_approx_eq(&r, &q1));
    }

    #[test]
    fn test_slerp_unit_norm_throughout() {
        let q0 = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 0.0);
        let q1 = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI);
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let r = slerp(&q0, &q1, t);
            assert!(
                (r.norm() - 1.0).abs() < 1e-10,
                "norm at t={}: {}",
                t,
                r.norm()
            );
        }
    }

    #[test]
    fn test_slerp_midpoint() {
        let q0 = Quaternion::from_axis_angle([0.0, 0.0, 1.0], 0.0);
        let q1 = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI);
        let mid = slerp(&q0, &q1, 0.5);
        // Midpoint should be 90° around Z
        let expected = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        assert!(quat_approx_eq(&mid, &expected));
    }

    #[test]
    fn test_slerp_array_endpoints() {
        let q0 = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 0.3);
        let q1 = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 1.5);
        let arr = slerp_array(&q0, &q1, 5);
        assert_eq!(arr.len(), 5);
        assert!(quat_approx_eq(&arr[0], &q0));
        assert!(quat_approx_eq(&arr[4], &q1));
    }
}
