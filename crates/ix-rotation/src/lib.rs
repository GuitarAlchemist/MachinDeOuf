//! # ix-rotation
//!
//! 3D rotation mathematics: quaternions, dual quaternions, SLERP,
//! Euler angles, rotation matrices, axis-angle, and Plücker coordinates.
//!
//! ## Modules
//!
//! - **quaternion**: Unit quaternion representation and operations
//! - **dual_quaternion**: Rigid body transformations (rotation + translation)
//! - **slerp**: Spherical linear interpolation
//! - **euler**: Euler angle conversions with gimbal lock detection
//! - **axis_angle**: Axis-angle ↔ quaternion ↔ matrix conversions
//! - **rotation_matrix**: SO(3) rotation matrix utilities
//! - **plucker**: Plücker line coordinates for 3D geometry

pub mod quaternion;
pub mod dual_quaternion;
pub mod slerp;
pub mod euler;
pub mod axis_angle;
pub mod rotation_matrix;
pub mod plucker;
