//! Neo-Riemannian P, L, R operators on consonant triads.
//!
//! Partial functions defined only on the 24 consonant triads (12 major + 12 minor).
//! Each returns `None` for other PC sets. All three are involutions.

use crate::pc_set::PcSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriadKind {
    Major,
    Minor,
}

/// Classify `x` as a consonant triad if cardinality is 3 and some pitch-class `r`
/// makes `{r, r+4, r+7}` (major) or `{r, r+3, r+7}` (minor) the exact set.
pub fn classify_triad(x: PcSet) -> Option<(u8, TriadKind)> {
    if x.cardinality() != 3 {
        return None;
    }
    for r in 0..12u8 {
        if !x.contains(r) {
            continue;
        }
        if x.contains((r + 4) % 12) && x.contains((r + 7) % 12) {
            return Some((r, TriadKind::Major));
        }
        if x.contains((r + 3) % 12) && x.contains((r + 7) % 12) {
            return Some((r, TriadKind::Minor));
        }
    }
    None
}

/// Parallel: swaps major ↔ minor keeping the root.
/// C major {0,4,7} ↔ C minor {0,3,7}.
pub fn p(x: PcSet) -> Option<PcSet> {
    let (root, kind) = classify_triad(x)?;
    let (old_third, new_third) = match kind {
        TriadKind::Major => ((root + 4) % 12, (root + 3) % 12),
        TriadKind::Minor => ((root + 3) % 12, (root + 4) % 12),
    };
    Some(x.remove(old_third).insert(new_third))
}

/// Leading-tone exchange.
/// C major {0,4,7} ↔ E minor {4,7,11}: major loses its root, gaining a new
/// note a semitone below; minor loses its fifth, gaining a note a semitone above.
pub fn l(x: PcSet) -> Option<PcSet> {
    let (root, kind) = classify_triad(x)?;
    match kind {
        TriadKind::Major => Some(x.remove(root).insert((root + 11) % 12)),
        TriadKind::Minor => Some(x.remove((root + 7) % 12).insert((root + 8) % 12)),
    }
}

/// Relative: major ↔ relative minor.
/// C major {0,4,7} ↔ A minor {0,4,9}: major loses its fifth, gains a tone above;
/// minor loses its root, gains a tone above root.
pub fn r(x: PcSet) -> Option<PcSet> {
    let (root, kind) = classify_triad(x)?;
    match kind {
        TriadKind::Major => Some(x.remove((root + 7) % 12).insert((root + 9) % 12)),
        TriadKind::Minor => Some(x.remove(root).insert((root + 10) % 12)),
    }
}

/// Slide: L ∘ P ∘ R applied left-to-right. Swaps triads sharing a common third.
/// C major {0,4,7} ↔ C# minor {1,4,8}.
pub fn s(x: PcSet) -> Option<PcSet> {
    l(p(r(x)?)?)
}

/// Nebenverwandt: R ∘ L ∘ P. C major ↔ F minor.
pub fn n(x: PcSet) -> Option<PcSet> {
    r(l(p(x)?)?)
}

/// Hexatonic pole: L ∘ P ∘ L. C major ↔ Ab minor.
pub fn h(x: PcSet) -> Option<PcSet> {
    l(p(l(x)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c_major() -> PcSet {
        PcSet::from_pcs([0, 4, 7])
    }
    fn c_minor() -> PcSet {
        PcSet::from_pcs([0, 3, 7])
    }
    fn a_minor() -> PcSet {
        PcSet::from_pcs([0, 4, 9])
    }
    fn e_minor() -> PcSet {
        PcSet::from_pcs([4, 7, 11])
    }

    #[test]
    fn classify_distinguishes_major_minor_and_returns_root() {
        assert_eq!(classify_triad(c_major()), Some((0, TriadKind::Major)));
        assert_eq!(classify_triad(c_minor()), Some((0, TriadKind::Minor)));
        assert_eq!(classify_triad(e_minor()), Some((4, TriadKind::Minor)));
    }

    #[test]
    fn classify_rejects_non_consonant() {
        assert!(classify_triad(PcSet::from_pcs([0, 4])).is_none()); // dyad
        assert!(classify_triad(PcSet::from_pcs([0, 4, 7, 10])).is_none()); // dom7
        assert!(classify_triad(PcSet::from_pcs([0, 4, 8])).is_none()); // augmented
        assert!(classify_triad(PcSet::from_pcs([0, 3, 6])).is_none()); // diminished
    }

    #[test]
    fn p_swaps_parallel() {
        assert_eq!(p(c_major()), Some(c_minor()));
        assert_eq!(p(c_minor()), Some(c_major()));
    }

    #[test]
    fn l_swaps_major_with_mediant_minor() {
        assert_eq!(l(c_major()), Some(e_minor()));
        assert_eq!(l(e_minor()), Some(c_major()));
    }

    #[test]
    fn r_swaps_major_with_relative_minor() {
        assert_eq!(r(c_major()), Some(a_minor()));
        assert_eq!(r(a_minor()), Some(c_major()));
    }

    #[test]
    fn p_l_r_are_involutions_on_all_24_consonant_triads() {
        for root in 0..12u8 {
            let maj = PcSet::from_pcs([root, root + 4, root + 7]);
            let min = PcSet::from_pcs([root, root + 3, root + 7]);
            for t in [maj, min] {
                assert_eq!(p(p(t).unwrap()), Some(t), "P∘P ≠ id on {t:?}");
                assert_eq!(l(l(t).unwrap()), Some(t), "L∘L ≠ id on {t:?}");
                assert_eq!(r(r(t).unwrap()), Some(t), "R∘R ≠ id on {t:?}");
            }
        }
    }

    #[test]
    fn slide_maps_c_major_to_csharp_minor() {
        // S = L ∘ P ∘ R: C major → A minor → A major → C# minor
        // {0,4,7} → {0,4,9} → {1,4,9} → {1,4,8}
        let s_result = s(c_major()).unwrap();
        assert_eq!(s_result, PcSet::from_pcs([1, 4, 8]));
    }

    #[test]
    fn hexatonic_pole_is_involution() {
        for root in 0..12u8 {
            let maj = PcSet::from_pcs([root, root + 4, root + 7]);
            assert_eq!(h(h(maj).unwrap()), Some(maj));
        }
    }

    #[test]
    fn plr_returns_none_on_non_triads() {
        let aug = PcSet::from_pcs([0, 4, 8]);
        assert_eq!(p(aug), None);
        assert_eq!(l(aug), None);
        assert_eq!(r(aug), None);
    }

    #[test]
    fn plr_preserves_consonant_triad_cardinality() {
        for root in 0..12u8 {
            let maj = PcSet::from_pcs([root, root + 4, root + 7]);
            assert_eq!(p(maj).unwrap().cardinality(), 3);
            assert_eq!(l(maj).unwrap().cardinality(), 3);
            assert_eq!(r(maj).unwrap().cardinality(), 3);
        }
    }
}
