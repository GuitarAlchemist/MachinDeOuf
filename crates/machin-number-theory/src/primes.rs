use crate::sieve::sieve_of_eratosthenes;

/// Returns the gaps between consecutive primes up to `limit`.
///
/// If there are fewer than 2 primes up to `limit`, returns an empty vector.
pub fn prime_gaps(limit: usize) -> Vec<usize> {
    let primes = sieve_of_eratosthenes(limit);
    if primes.len() < 2 {
        return Vec::new();
    }
    primes.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Returns all twin prime pairs (p, p+2) where both are prime and ≤ `limit`.
pub fn twin_primes(limit: usize) -> Vec<(usize, usize)> {
    let primes = sieve_of_eratosthenes(limit);
    primes
        .windows(2)
        .filter_map(|w| {
            if w[1] - w[0] == 2 {
                Some((w[0], w[1]))
            } else {
                None
            }
        })
        .collect()
}

/// Returns prime triplets up to `limit`.
///
/// A prime triplet is (p, p+2, p+6) or (p, p+4, p+6) where all three are prime.
pub fn prime_triplets(limit: usize) -> Vec<(usize, usize, usize)> {
    if limit < 7 {
        return Vec::new();
    }
    let mut is_prime = vec![false; limit + 1];
    for p in sieve_of_eratosthenes(limit) {
        is_prime[p] = true;
    }
    let mut result = Vec::new();
    for p in 2..=limit {
        if !is_prime[p] {
            continue;
        }
        // Form (p, p+2, p+6)
        if p + 6 <= limit && is_prime[p + 2] && is_prime[p + 6] {
            result.push((p, p + 2, p + 6));
        }
        // Form (p, p+4, p+6)
        if p + 6 <= limit && is_prime[p + 4] && is_prime[p + 6] {
            result.push((p, p + 4, p + 6));
        }
    }
    result
}

/// Prime counting function π(n): the number of primes ≤ n.
pub fn prime_counting(n: usize) -> usize {
    sieve_of_eratosthenes(n).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twin_primes_small() {
        let twins = twin_primes(20);
        assert!(twins.contains(&(3, 5)));
        assert!(twins.contains(&(5, 7)));
        assert!(twins.contains(&(11, 13)));
        assert!(twins.contains(&(17, 19)));
    }

    #[test]
    fn test_prime_counting_100() {
        assert_eq!(prime_counting(100), 25);
    }

    #[test]
    fn test_prime_counting_10() {
        assert_eq!(prime_counting(10), 4);
    }

    #[test]
    fn test_prime_counting_edge() {
        assert_eq!(prime_counting(0), 0);
        assert_eq!(prime_counting(1), 0);
        assert_eq!(prime_counting(2), 1);
    }

    #[test]
    fn test_prime_gaps() {
        let gaps = prime_gaps(20);
        // primes: 2,3,5,7,11,13,17,19 → gaps: 1,2,2,4,2,4,2
        assert_eq!(gaps, vec![1, 2, 2, 4, 2, 4, 2]);
    }

    #[test]
    fn test_prime_gaps_empty() {
        assert_eq!(prime_gaps(1), Vec::<usize>::new());
        assert_eq!(prime_gaps(2), Vec::<usize>::new()); // only [2], < 2 primes for a gap
    }

    #[test]
    fn test_prime_triplets() {
        let trips = prime_triplets(50);
        // (5, 7, 11) is (p, p+2, p+6) with p=5
        assert!(trips.contains(&(5, 7, 11)));
        // (7, 11, 13) is (p, p+4, p+6) with p=7
        assert!(trips.contains(&(7, 11, 13)));
    }

    #[test]
    fn test_twin_primes_empty() {
        assert_eq!(twin_primes(2), Vec::<(usize, usize)>::new());
    }
}
