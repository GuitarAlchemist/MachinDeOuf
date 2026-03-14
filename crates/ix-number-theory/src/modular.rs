/// Modular exponentiation: base^exp mod modulus, using exponentiation by squaring.
///
/// Uses u128 internally to avoid overflow on multiplication.
pub fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus == 0 {
        panic!("modulus must be non-zero");
    }
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u128;
    let mut base = (base % modulus) as u128;
    let modulus_128 = modulus as u128;
    let mut exp = exp;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus_128;
        }
        exp >>= 1;
        base = base * base % modulus_128;
    }
    result as u64
}

/// Extended Euclidean algorithm: returns (gcd, x, y) such that a*x + b*y = gcd.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        // gcd = |a|, but we keep sign convention: gcd(a, 0) = a for a >= 0
        if a >= 0 {
            return (a, 1, 0);
        } else {
            return (-a, -1, 0);
        }
    }
    let (g, x1, y1) = extended_gcd(b, a % b);
    (g, y1, x1 - (a / b) * y1)
}

/// Greatest common divisor using the Euclidean algorithm.
pub fn gcd(a: u64, b: u64) -> u64 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Least common multiple.
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd(a, b) * b
}

/// Modular inverse of a mod m, if it exists (i.e., gcd(a, m) == 1).
///
/// Returns `Some(x)` where `a * x ≡ 1 (mod m)`, or `None` if no inverse exists.
pub fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    if m == 0 {
        return None;
    }
    if m == 1 {
        return Some(0);
    }
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g != 1 {
        return None;
    }
    // Normalize x into [0, m)
    Some(((x % m as i64 + m as i64) % m as i64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(2, 10, 1024), 0);
        assert_eq!(mod_pow(3, 0, 7), 1);
        assert_eq!(mod_pow(0, 0, 5), 1); // 0^0 = 1 by convention
        assert_eq!(mod_pow(5, 3, 13), 8); // 125 % 13 = 8
    }

    #[test]
    fn test_mod_pow_modulus_1() {
        assert_eq!(mod_pow(100, 100, 1), 0);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
        assert_eq!(gcd(0, 0), 0);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(0, 5), 0);
        assert_eq!(lcm(7, 1), 7);
    }

    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(12, 8);
        assert_eq!(g, 4);
        assert_eq!(12 * x + 8 * y, 4);

        let (g, x, y) = extended_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }

    #[test]
    fn test_mod_inverse() {
        // 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!(mod_inverse(3, 7), Some(5));
        // 2 has no inverse mod 4
        assert_eq!(mod_inverse(2, 4), None);
        // edge
        assert_eq!(mod_inverse(1, 1), Some(0));
    }

    #[test]
    fn test_mod_inverse_larger() {
        let inv = mod_inverse(17, 43).unwrap();
        assert_eq!((17 * inv) % 43, 1);
    }
}
