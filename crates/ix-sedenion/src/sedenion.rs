use std::ops::{Add, Sub, Mul, Neg};
use crate::octonion::Octonion;

/// A sedenion: 16-dimensional hypercomplex number built via Cayley-Dickson
/// construction from octonion pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sedenion {
    pub components: [f64; 16],
}

impl Sedenion {
    /// Create a sedenion from 16 components.
    pub fn new(components: [f64; 16]) -> Self {
        Self { components }
    }

    /// The zero sedenion.
    pub fn zero() -> Self {
        Self { components: [0.0; 16] }
    }

    /// The multiplicative identity (1, 0, 0, ..., 0).
    pub fn one() -> Self {
        let mut c = [0.0; 16];
        c[0] = 1.0;
        Self { components: c }
    }

    /// The i-th basis element e_i (0-indexed). e_0 = 1.
    pub fn basis(i: usize) -> Self {
        assert!(i < 16, "Sedenion basis index must be 0..15");
        let mut c = [0.0; 16];
        c[i] = 1.0;
        Self { components: c }
    }

    /// Extract the first octonion half.
    fn first_oct(&self) -> Octonion {
        let mut c = [0.0; 8];
        c.copy_from_slice(&self.components[0..8]);
        Octonion::new(c)
    }

    /// Extract the second octonion half.
    fn second_oct(&self) -> Octonion {
        let mut c = [0.0; 8];
        c.copy_from_slice(&self.components[8..16]);
        Octonion::new(c)
    }

    /// Build a sedenion from two octonion halves.
    fn from_oct_pair(first: &Octonion, second: &Octonion) -> Sedenion {
        let mut c = [0.0; 16];
        c[0..8].copy_from_slice(&first.components);
        c[8..16].copy_from_slice(&second.components);
        Sedenion { components: c }
    }

    /// Component-wise addition.
    pub fn add(&self, other: &Sedenion) -> Sedenion {
        let mut c = [0.0; 16];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.components[i] + other.components[i];
        }
        Sedenion { components: c }
    }

    /// Component-wise subtraction.
    pub fn sub(&self, other: &Sedenion) -> Sedenion {
        let mut c = [0.0; 16];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.components[i] - other.components[i];
        }
        Sedenion { components: c }
    }

    /// Cayley-Dickson multiplication using octonion pairs.
    /// (a, b) * (c, d) = (a*c - conj(d)*b, d*a + b*conj(c))
    /// where a, b, c, d are octonions.
    pub fn mul(&self, other: &Sedenion) -> Sedenion {
        let a = self.first_oct();
        let b = self.second_oct();
        let c = other.first_oct();
        let d = other.second_oct();

        let conj_d = d.conjugate();
        let conj_c = c.conjugate();

        // first half: a*c - conj(d)*b
        let ac = Octonion::mul(&a, &c);
        let conj_d_b = Octonion::mul(&conj_d, &b);
        let first = Octonion::sub(&ac, &conj_d_b);

        // second half: d*a + b*conj(c)
        let da = Octonion::mul(&d, &a);
        let b_conj_c = Octonion::mul(&b, &conj_c);
        let second = Octonion::add(&da, &b_conj_c);

        Self::from_oct_pair(&first, &second)
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Sedenion {
        let mut c = [0.0; 16];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.components[i] * s;
        }
        Sedenion { components: c }
    }

    /// Sedenion conjugate: negate all imaginary parts (indices 1..15).
    pub fn conjugate(&self) -> Sedenion {
        let mut c = self.components;
        for ci in c.iter_mut().skip(1) {
            *ci = -*ci;
        }
        Sedenion { components: c }
    }

    /// Squared norm: sum of squares of all components.
    pub fn norm_squared(&self) -> f64 {
        self.components.iter().map(|x| x * x).sum()
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Multiplicative inverse: conj(x) / norm²(x).
    pub fn inverse(&self) -> Sedenion {
        let ns = self.norm_squared();
        assert!(ns > 1e-15, "Cannot invert zero sedenion");
        self.conjugate().scale(1.0 / ns)
    }
}

impl Add for Sedenion {
    type Output = Sedenion;
    fn add(self, rhs: Sedenion) -> Sedenion {
        Sedenion::add(&self, &rhs)
    }
}

impl Sub for Sedenion {
    type Output = Sedenion;
    fn sub(self, rhs: Sedenion) -> Sedenion {
        Sedenion::sub(&self, &rhs)
    }
}

impl Mul for Sedenion {
    type Output = Sedenion;
    fn mul(self, rhs: Sedenion) -> Sedenion {
        Sedenion::mul(&self, &rhs)
    }
}

impl Neg for Sedenion {
    type Output = Sedenion;
    fn neg(self) -> Sedenion {
        self.scale(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: &Sedenion, b: &Sedenion) -> bool {
        a.components.iter().zip(b.components.iter())
            .all(|(x, y)| (x - y).abs() < EPS)
    }

    #[test]
    fn test_unit_times_unit_is_unit() {
        let one = Sedenion::one();
        assert!(approx_eq(&Sedenion::mul(&one, &one), &one));
    }

    #[test]
    fn test_unit_multiply() {
        let one = Sedenion::one();
        let s = Sedenion::new([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                               9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]);
        assert!(approx_eq(&Sedenion::mul(&one, &s), &s));
        assert!(approx_eq(&Sedenion::mul(&s, &one), &s));
    }

    #[test]
    fn test_basis_squared_is_minus_one() {
        let one = Sedenion::one();
        let neg_one = -one;
        for i in 1..16 {
            let ei = Sedenion::basis(i);
            let sq = Sedenion::mul(&ei, &ei);
            assert!(approx_eq(&sq, &neg_one),
                "e_{}^2 should be -1, got {:?}", i, sq.components);
        }
    }

    #[test]
    fn test_conjugate_times_self_is_norm_squared() {
        let s = Sedenion::new([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                               9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]);
        let conj = s.conjugate();
        let product = Sedenion::mul(&conj, &s);
        let ns = s.norm_squared();
        assert!((product.components[0] - ns).abs() < EPS,
            "real part should be norm², got {} vs {}", product.components[0], ns);
        for i in 1..16 {
            assert!(product.components[i].abs() < EPS,
                "imaginary part {} should be 0, got {}", i, product.components[i]);
        }
    }

    #[test]
    fn test_power_associativity() {
        // Sedenions are power-associative: x*(x*x) = (x*x)*x
        let x = Sedenion::new([1.0,0.5,0.3,0.1,0.2,0.4,0.6,0.8,
                               0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]);
        let xx = Sedenion::mul(&x, &x);
        let lhs = Sedenion::mul(&x, &xx);
        let rhs = Sedenion::mul(&xx, &x);
        assert!(approx_eq(&lhs, &rhs),
            "Power associativity failed:\nlhs={:?}\nrhs={:?}", lhs.components, rhs.components);
    }

    #[test]
    fn test_scalar_multiplication() {
        let s = Sedenion::new([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                               9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]);
        let scaled = s.scale(2.0);
        for i in 0..16 {
            assert!((scaled.components[i] - 2.0 * s.components[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_zero() {
        let z = Sedenion::zero();
        assert!((z.norm() - 0.0).abs() < EPS);
        let one = Sedenion::one();
        assert!(approx_eq(&(one + z), &one));
    }

    #[test]
    fn test_inverse() {
        let s = Sedenion::new([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                               0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]);
        let inv = s.inverse();
        let product = Sedenion::mul(&s, &inv);
        assert!(approx_eq(&product, &Sedenion::one()),
            "s * s^-1 should be 1, got {:?}", product.components);
    }

    #[test]
    fn test_add_sub() {
        let a = Sedenion::new([1.0;16]);
        let b = Sedenion::new([2.0;16]);
        let sum = a + b;
        for i in 0..16 {
            assert!((sum.components[i] - 3.0).abs() < EPS);
        }
        let diff = b - a;
        for i in 0..16 {
            assert!((diff.components[i] - 1.0).abs() < EPS);
        }
    }

    #[test]
    fn test_neg() {
        let a = Sedenion::new([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                               9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]);
        let neg_a = -a;
        for i in 0..16 {
            assert!((neg_a.components[i] + a.components[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_norm() {
        let one = Sedenion::one();
        assert!((one.norm() - 1.0).abs() < EPS);

        let e5 = Sedenion::basis(5);
        assert!((e5.norm() - 1.0).abs() < EPS);
    }
}
