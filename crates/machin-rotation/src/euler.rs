//! Euler angle conversions with gimbal lock detection.

use crate::quaternion::Quaternion;

/// Euler rotation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EulerOrder {
    XYZ,
    XZY,
    YXZ,
    YZX,
    ZXY,
    ZYX,
    ZXZ,
    XYX,
    YZY,
}

/// Convert Euler angles (roll, pitch, yaw) to a quaternion.
///
/// - `roll`: rotation about the first axis
/// - `pitch`: rotation about the second axis
/// - `yaw`: rotation about the third axis
///
/// The resulting quaternion applies rotations in the intrinsic order specified.
pub fn to_quaternion(roll: f64, pitch: f64, yaw: f64, order: EulerOrder) -> Quaternion {
    let qx = Quaternion::from_axis_angle([1.0, 0.0, 0.0], roll);
    let qy = Quaternion::from_axis_angle([0.0, 1.0, 0.0], pitch);
    let qz = Quaternion::from_axis_angle([0.0, 0.0, 1.0], yaw);

    match order {
        // Intrinsic XYZ = extrinsic ZYX: q = qx * qy * qz (rightmost applied first)
        // We compute intrinsic order: first axis applied first in multiplication chain
        EulerOrder::XYZ => qx.mul(&qy).mul(&qz),
        EulerOrder::XZY => qx.mul(&qz).mul(&qy),
        EulerOrder::YXZ => qy.mul(&qx).mul(&qz),
        EulerOrder::YZX => qy.mul(&qz).mul(&qx),
        EulerOrder::ZXY => qz.mul(&qx).mul(&qy),
        EulerOrder::ZYX => qz.mul(&qy).mul(&qx),
        EulerOrder::ZXZ => {
            let qz2 = Quaternion::from_axis_angle([0.0, 0.0, 1.0], yaw);
            let qz1 = Quaternion::from_axis_angle([0.0, 0.0, 1.0], roll);
            qz1.mul(&qx).mul(&qz2)
        }
        EulerOrder::XYX => {
            let qx2 = Quaternion::from_axis_angle([1.0, 0.0, 0.0], yaw);
            let qx1 = Quaternion::from_axis_angle([1.0, 0.0, 0.0], roll);
            qx1.mul(&qy).mul(&qx2)
        }
        EulerOrder::YZY => {
            let qy2 = Quaternion::from_axis_angle([0.0, 1.0, 0.0], yaw);
            let qy1 = Quaternion::from_axis_angle([0.0, 1.0, 0.0], roll);
            qy1.mul(&qz).mul(&qy2)
        }
    }
}

/// Convert a quaternion to Euler angles (roll, pitch, yaw) for a given order.
///
/// Returns `(roll, pitch, yaw)` in radians.
pub fn from_quaternion(q: &Quaternion, order: EulerOrder) -> (f64, f64, f64) {
    let m = q.to_rotation_matrix();
    match order {
        EulerOrder::XYZ => {
            // pitch = asin(m[0][2])
            let pitch = clamp(m[0][2], -1.0, 1.0).asin();
            if m[0][2].abs() < 0.99999 {
                let roll = (-m[1][2]).atan2(m[2][2]);
                let yaw = (-m[0][1]).atan2(m[0][0]);
                (roll, pitch, yaw)
            } else {
                // Gimbal lock
                let roll = m[1][0].atan2(m[1][1]);
                (roll, pitch, 0.0)
            }
        }
        EulerOrder::ZYX => {
            // pitch = asin(-m[2][0])
            let pitch = clamp(-m[2][0], -1.0, 1.0).asin();
            if m[2][0].abs() < 0.99999 {
                let roll = m[2][1].atan2(m[2][2]);
                let yaw = m[1][0].atan2(m[0][0]);
                (roll, pitch, yaw)
            } else {
                let roll = (-m[0][1]).atan2(m[0][2]);
                (roll, pitch, 0.0)
            }
        }
        EulerOrder::ZXZ => {
            // Euler ZXZ: m[2][2] = cos(pitch)
            let pitch = clamp(m[2][2], -1.0, 1.0).acos();
            if pitch.sin().abs() > 1e-6 {
                let roll = m[0][2].atan2(-m[1][2]);
                let yaw = m[2][0].atan2(m[2][1]);
                (roll, pitch, yaw)
            } else {
                let roll = m[0][0].atan2(m[0][1]);
                (roll, pitch, 0.0)
            }
        }
        // For other orders, convert through rotation matrix with appropriate decomposition.
        // Use a general approach: convert to ZYX, then note we can use the matrix.
        _ => {
            // Fallback: decompose via round-trip through matrix for less common orders.
            // We rebuild by trying all angles; for correctness in the common Tait-Bryan cases:
            from_quaternion_general(q, order)
        }
    }
}

/// General Euler decomposition for less common orders via matrix analysis.
fn from_quaternion_general(q: &Quaternion, order: EulerOrder) -> (f64, f64, f64) {
    let m = q.to_rotation_matrix();
    match order {
        EulerOrder::YXZ => {
            let pitch = clamp(-m[1][2], -1.0, 1.0).asin();
            if m[1][2].abs() < 0.99999 {
                let roll = m[0][2].atan2(m[2][2]);
                let yaw = m[1][0].atan2(m[1][1]);
                (roll, pitch, yaw)
            } else {
                let roll = (-m[2][0]).atan2(m[0][0]);
                (roll, pitch, 0.0)
            }
        }
        EulerOrder::YZX => {
            let pitch = clamp(m[1][0], -1.0, 1.0).asin();
            if m[1][0].abs() < 0.99999 {
                let roll = (-m[2][0]).atan2(m[0][0]);
                let yaw = (-m[1][2]).atan2(m[1][1]);
                (roll, pitch, yaw)
            } else {
                let roll = m[0][2].atan2(m[2][2]);
                (roll, pitch, 0.0)
            }
        }
        EulerOrder::XZY => {
            let pitch = clamp(-m[0][1], -1.0, 1.0).asin();
            if m[0][1].abs() < 0.99999 {
                let roll = m[2][1].atan2(m[1][1]);
                let yaw = m[0][2].atan2(m[0][0]);
                (roll, pitch, yaw)
            } else {
                let roll = (-m[1][2]).atan2(m[2][2]);
                (roll, pitch, 0.0)
            }
        }
        EulerOrder::ZXY => {
            let pitch = clamp(m[2][1], -1.0, 1.0).asin();
            if m[2][1].abs() < 0.99999 {
                let roll = (-m[0][1]).atan2(m[1][1]);
                let yaw = (-m[2][0]).atan2(m[2][2]);
                (roll, pitch, yaw)
            } else {
                let roll = m[1][0].atan2(m[0][0]);
                (roll, pitch, 0.0)
            }
        }
        EulerOrder::XYX => {
            let pitch = clamp(m[0][0], -1.0, 1.0).acos();
            if pitch.sin().abs() > 1e-6 {
                let roll = m[1][0].atan2(-m[2][0]);
                let yaw = m[0][1].atan2(m[0][2]);
                (roll, pitch, yaw)
            } else {
                let roll = m[1][1].atan2(m[1][2]);
                (roll, pitch, 0.0)
            }
        }
        EulerOrder::YZY => {
            let pitch = clamp(m[1][1], -1.0, 1.0).acos();
            if pitch.sin().abs() > 1e-6 {
                let roll = m[2][1].atan2(-m[0][1]);
                let yaw = m[1][2].atan2(m[1][0]);
                (roll, pitch, yaw)
            } else {
                let roll = m[2][2].atan2(m[2][0]);
                (roll, pitch, 0.0)
            }
        }
        // Already handled in from_quaternion
        _ => (0.0, 0.0, 0.0),
    }
}

fn clamp(val: f64, min: f64, max: f64) -> f64 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

/// Check if a pitch angle is near gimbal lock (±π/2).
pub fn gimbal_lock_check(pitch: f64) -> bool {
    let half_pi = std::f64::consts::FRAC_PI_2;
    (pitch.abs() - half_pi).abs() < 1e-4
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    const EPS: f64 = 1e-8;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_identity_euler() {
        let q = to_quaternion(0.0, 0.0, 0.0, EulerOrder::XYZ);
        let id = Quaternion::identity();
        assert!(approx_eq(q.w, id.w));
        assert!(approx_eq(q.x, id.x));
        assert!(approx_eq(q.y, id.y));
        assert!(approx_eq(q.z, id.z));
    }

    #[test]
    fn test_round_trip_xyz() {
        let (r, p, y) = (0.3, 0.5, -0.7);
        let q = to_quaternion(r, p, y, EulerOrder::XYZ);
        let (r2, p2, y2) = from_quaternion(&q, EulerOrder::XYZ);
        assert!(approx_eq(r, r2), "roll: {} vs {}", r, r2);
        assert!(approx_eq(p, p2), "pitch: {} vs {}", p, p2);
        assert!(approx_eq(y, y2), "yaw: {} vs {}", y, y2);
    }

    #[test]
    fn test_round_trip_zyx() {
        let (r, p, y) = (0.2, -0.4, 0.6);
        let q = to_quaternion(r, p, y, EulerOrder::ZYX);
        let (r2, p2, y2) = from_quaternion(&q, EulerOrder::ZYX);
        assert!(approx_eq(r, r2), "roll: {} vs {}", r, r2);
        assert!(approx_eq(p, p2), "pitch: {} vs {}", p, p2);
        assert!(approx_eq(y, y2), "yaw: {} vs {}", y, y2);
    }

    #[test]
    fn test_gimbal_lock_check() {
        assert!(gimbal_lock_check(FRAC_PI_2));
        assert!(gimbal_lock_check(-FRAC_PI_2));
        assert!(!gimbal_lock_check(0.0));
        assert!(!gimbal_lock_check(1.0));
    }

    #[test]
    fn test_gimbal_lock_at_half_pi() {
        // At gimbal lock, yaw becomes degenerate, but we should still get a valid quaternion
        let q = to_quaternion(0.3, FRAC_PI_2, 0.5, EulerOrder::XYZ);
        assert!((q.norm() - 1.0).abs() < 1e-10);
    }
}
