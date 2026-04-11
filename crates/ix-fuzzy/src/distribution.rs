//! [`FuzzyDistribution`] — generic fuzzy membership over discrete
//! variants.
//!
//! Backed by `BTreeMap<T, f64>` so iteration is deterministic and
//! serialization is bit-identical across processes — the governance
//! instrument contract.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::FuzzyError;

/// Absolute tolerance for the `sum == 1.0` invariant. Documented in
/// `governance/demerzel/logic/fuzzy-membership.md`.
pub const SUM_TOLERANCE: f64 = 0.01;

/// A fuzzy distribution over variants of type `T`.
///
/// Invariants (checked on construction):
/// 1. Every membership is a finite value in `[0.0, 1.0]`.
/// 2. The sum of all memberships is `1.0 ± SUM_TOLERANCE`.
/// 3. At least one variant is present.
///
/// Iteration order is always sorted by `T` — `BTreeMap` under the
/// hood — so two distributions with the same logical content produce
/// bit-identical JSON regardless of construction order. This is the
/// same determinism guarantee `ix_agent_core::ReadContext` honors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FuzzyDistribution<T>
where
    T: Ord + Clone,
{
    memberships: BTreeMap<T, f64>,
}

impl<T> FuzzyDistribution<T>
where
    T: Ord + Clone,
{
    /// Construct from an iterator of `(variant, membership)` pairs.
    /// Validates invariants 1-3 and returns `FuzzyError` on any
    /// violation.
    ///
    /// Duplicate keys are summed — the last write does NOT win,
    /// because summing lets callers accumulate evidence by repeated
    /// insertion and validate the total in one step.
    pub fn new<I>(pairs: I) -> Result<Self, FuzzyError>
    where
        I: IntoIterator<Item = (T, f64)>,
    {
        let mut memberships: BTreeMap<T, f64> = BTreeMap::new();
        for (k, v) in pairs {
            if !v.is_finite() {
                return Err(FuzzyError::NotFinite { value: v });
            }
            if !(0.0..=1.0).contains(&v) {
                return Err(FuzzyError::OutOfRange { value: v });
            }
            *memberships.entry(k).or_insert(0.0) += v;
        }
        if memberships.is_empty() {
            return Err(FuzzyError::Empty);
        }
        let sum: f64 = memberships.values().sum();
        if (sum - 1.0).abs() > SUM_TOLERANCE {
            return Err(FuzzyError::BadSum { sum });
        }
        Ok(Self { memberships })
    }

    /// Construct a *sharp* distribution — one variant with mass 1.0,
    /// every other supplied variant with mass 0.0. `variants` must
    /// include `selected`, otherwise the selected variant is added
    /// implicitly.
    ///
    /// This is the `Pure(v)` constructor from the Demerzel spec.
    pub fn pure<I>(selected: T, variants: I) -> Result<Self, FuzzyError>
    where
        I: IntoIterator<Item = T>,
    {
        let mut memberships: BTreeMap<T, f64> = BTreeMap::new();
        for v in variants {
            memberships.entry(v).or_insert(0.0);
        }
        memberships.insert(selected, 1.0);
        Self::new(memberships)
    }

    /// Construct a *uniform* distribution — every supplied variant
    /// gets mass `1/N`. This is the `Uniform(variants)` constructor
    /// from the Demerzel spec.
    pub fn uniform<I>(variants: I) -> Result<Self, FuzzyError>
    where
        I: IntoIterator<Item = T>,
    {
        let variants: Vec<T> = variants.into_iter().collect();
        if variants.is_empty() {
            return Err(FuzzyError::Empty);
        }
        let mass = 1.0 / variants.len() as f64;
        Self::new(variants.into_iter().map(|v| (v, mass)))
    }

    /// Borrow the membership for a variant, or `0.0` if the variant
    /// is absent from the distribution.
    pub fn get(&self, variant: &T) -> f64 {
        self.memberships.get(variant).copied().unwrap_or(0.0)
    }

    /// Iterate `(variant, membership)` pairs in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&T, f64)> {
        self.memberships.iter().map(|(k, v)| (k, *v))
    }

    /// Number of variants in this distribution.
    pub fn len(&self) -> usize {
        self.memberships.len()
    }

    /// `true` iff the distribution has no variants. Cannot be
    /// constructed through the public API — present because clippy
    /// complains if `len` exists without `is_empty`.
    pub fn is_empty(&self) -> bool {
        self.memberships.is_empty()
    }

    /// The variant with the highest membership. Returns the first
    /// key in sorted order when there is a tie — callers that need
    /// domain-specific tiebreak (e.g., `C > U > D > P > T > F` for
    /// [`crate::HexavalentDistribution`]) should use their own
    /// variant of this function.
    pub fn argmax(&self) -> &T {
        let (key, _) = self
            .memberships
            .iter()
            .max_by(|a, b| {
                // `partial_cmp` is fine because all values are
                // finite by construction.
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // Stable tiebreak: higher key wins so ties are
                    // deterministic and `>= argmax` makes sense.
                    .then_with(|| a.0.cmp(b.0))
            })
            .expect("distribution non-empty by invariant");
        key
    }

    /// The membership value of the argmax variant.
    pub fn argmax_mass(&self) -> f64 {
        self.get(self.argmax())
    }

    /// `true` iff the argmax membership exceeds `threshold`.
    /// Callers that want to collapse to a discrete value should
    /// check `is_sharp` first, then take `argmax()`.
    pub fn is_sharp(&self, threshold: f64) -> bool {
        self.argmax_mass() > threshold
    }

    /// Renormalize so memberships sum to `1.0` exactly. Idempotent.
    /// Returns [`FuzzyError::BadSum`] only if every membership is
    /// zero (no way to rescale).
    pub fn renormalize(&mut self) -> Result<(), FuzzyError> {
        let sum: f64 = self.memberships.values().sum();
        if sum <= 0.0 {
            return Err(FuzzyError::BadSum { sum });
        }
        for v in self.memberships.values_mut() {
            *v /= sum;
        }
        Ok(())
    }

    /// Raw access to the underlying map. Exposed for
    /// per-variant ops modules in this crate.
    pub(crate) fn memberships(&self) -> &BTreeMap<T, f64> {
        &self.memberships
    }

    /// Mutable raw access — test-only so renormalize can be
    /// exercised from a deliberately-poisoned interior map. Not
    /// part of the public API.
    #[cfg(test)]
    pub(crate) fn memberships_mut(&mut self) -> &mut BTreeMap<T, f64> {
        &mut self.memberships
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_validates_range() {
        let err = FuzzyDistribution::new(vec![("a", 1.5), ("b", -0.5)]).unwrap_err();
        assert!(matches!(err, FuzzyError::OutOfRange { .. }));
    }

    #[test]
    fn new_validates_finite() {
        let err = FuzzyDistribution::new(vec![("a", f64::NAN), ("b", 1.0)]).unwrap_err();
        assert!(matches!(err, FuzzyError::NotFinite { .. }));
    }

    #[test]
    fn new_validates_sum() {
        let err = FuzzyDistribution::new(vec![("a", 0.3), ("b", 0.3)]).unwrap_err();
        assert!(matches!(err, FuzzyError::BadSum { .. }));
    }

    #[test]
    fn new_accepts_exact_sum() {
        let d = FuzzyDistribution::new(vec![("a", 0.5), ("b", 0.5)]).unwrap();
        assert_eq!(d.get(&"a"), 0.5);
    }

    #[test]
    fn new_accepts_sum_within_tolerance() {
        FuzzyDistribution::new(vec![("a", 0.995), ("b", 0.005)]).unwrap();
    }

    #[test]
    fn pure_places_all_mass_on_selected() {
        let d = FuzzyDistribution::pure("x", vec!["x", "y", "z"]).unwrap();
        assert_eq!(d.get(&"x"), 1.0);
        assert_eq!(d.get(&"y"), 0.0);
    }

    #[test]
    fn uniform_distributes_evenly() {
        let d = FuzzyDistribution::uniform(vec!["a", "b", "c", "d"]).unwrap();
        for v in &["a", "b", "c", "d"] {
            assert!((d.get(v) - 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn uniform_empty_errors() {
        let err = FuzzyDistribution::<&str>::uniform(Vec::<&str>::new()).unwrap_err();
        assert!(matches!(err, FuzzyError::Empty));
    }

    #[test]
    fn argmax_and_is_sharp() {
        let d = FuzzyDistribution::new(vec![("a", 0.1), ("b", 0.2), ("c", 0.7)]).unwrap();
        assert_eq!(*d.argmax(), "c");
        assert!(d.is_sharp(0.5));
        assert!(!d.is_sharp(0.8));
    }

    #[test]
    fn argmax_tie_breaks_by_sorted_key() {
        let d = FuzzyDistribution::new(vec![("a", 0.5), ("b", 0.5)]).unwrap();
        assert_eq!(*d.argmax(), "b");
    }

    #[test]
    fn renormalize_scales_mass_to_unity() {
        // Construct a legal distribution then renormalize a
        // manually-poisoned interior map so we exercise the reset
        // path from a non-unit sum.
        let mut d = FuzzyDistribution::new(vec![("a", 0.6), ("b", 0.4)]).unwrap();
        for v in d.memberships_mut().values_mut() {
            *v *= 2.0;
        }
        d.renormalize().unwrap();
        let sum: f64 = d.iter().map(|(_, v)| v).sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn iteration_is_deterministic_across_inserts() {
        // Insert in two different orders, assert JSON is identical.
        let a = FuzzyDistribution::new(vec![("a", 0.3), ("b", 0.3), ("c", 0.4)]).unwrap();
        let b = FuzzyDistribution::new(vec![("c", 0.4), ("a", 0.3), ("b", 0.3)]).unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }
}
