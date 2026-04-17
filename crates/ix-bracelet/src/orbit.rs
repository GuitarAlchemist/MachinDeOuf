//! Orbit enumeration: all D₁₂ images of a set, and the 224 canonical orbits.

use crate::action::Action;
use crate::dihedral::DihedralElement;
use crate::pc_set::PcSet;
use crate::prime_form::bracelet_prime_form;

use std::collections::BTreeSet;
use std::sync::OnceLock;

/// Return all 24 dihedral images of `x` (with duplicates when the stabilizer is non-trivial).
/// Order: `(rotation, reflected)` iterates rotations 0..12 inner, reflections outer.
pub fn orbit(x: PcSet) -> [PcSet; 24] {
    let mut out = [PcSet::empty(); 24];
    let mut i = 0;
    for reflected in [false, true] {
        for rotation in 0..12u8 {
            let g = DihedralElement::from_tn_tni(rotation, reflected);
            out[i] = g.apply(x);
            i += 1;
        }
    }
    out
}

/// Deduplicated orbit — each element appears exactly once, sorted by raw mask.
pub fn orbit_unique(x: PcSet) -> Vec<PcSet> {
    let mut seen: BTreeSet<u16> = BTreeSet::new();
    for y in orbit(x) {
        seen.insert(y.raw());
    }
    seen.into_iter().map(PcSet::new).collect()
}

/// All 224 distinct D₁₂-orbits on subsets of Z/12, each represented by its prime form.
/// The count 224 is the Burnside/Pólya total for D₁₂ acting on 2¹² subsets and matches
/// the Forte set-class count (when the empty set, singletons, and chromatic aggregate
/// are included).
pub fn all_prime_forms() -> &'static [PcSet] {
    static CACHE: OnceLock<Vec<PcSet>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut set: BTreeSet<u16> = BTreeSet::new();
        for mask in 0u16..=0x0FFF {
            set.insert(bracelet_prime_form(PcSet::new(mask)).raw());
        }
        set.into_iter().map(PcSet::new).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_of_empty_set_is_all_empty() {
        let o = orbit(PcSet::empty());
        for p in o {
            assert_eq!(p, PcSet::empty());
        }
        assert_eq!(orbit_unique(PcSet::empty()), vec![PcSet::empty()]);
    }

    #[test]
    fn orbit_of_chromatic_is_all_chromatic() {
        let o = orbit(PcSet::chromatic());
        for p in o {
            assert_eq!(p, PcSet::chromatic());
        }
    }

    #[test]
    fn orbit_contains_x_and_has_size_dividing_24() {
        for mask in [0x091u16, 0x089, 0x111, 0x025, 0x249] {
            let x = PcSet::new(mask);
            let uniq = orbit_unique(x);
            assert!(uniq.contains(&x), "orbit of {mask:03X} missing x itself");
            assert!(
                24 % uniq.len() == 0,
                "orbit size {} does not divide 24 (x = {mask:03X})",
                uniq.len()
            );
        }
    }

    #[test]
    fn orbit_members_share_bracelet_prime_form() {
        let x = PcSet::new(0x091); // {0, 4, 7} — major triad
        let pf = bracelet_prime_form(x);
        for y in orbit_unique(x) {
            assert_eq!(bracelet_prime_form(y), pf);
        }
    }

    #[test]
    fn there_are_exactly_224_distinct_prime_forms() {
        let pfs = all_prime_forms();
        assert_eq!(pfs.len(), 224);
    }

    #[test]
    fn every_prime_form_is_its_own_bracelet_prime_form() {
        for &pf in all_prime_forms() {
            assert_eq!(bracelet_prime_form(pf), pf);
        }
    }

    #[test]
    fn major_and_minor_triad_share_a_bracelet_orbit() {
        let major = PcSet::new(0x091); // {0, 4, 7}
        let minor = PcSet::new(0x089); // {0, 3, 7}
        assert_eq!(bracelet_prime_form(major), bracelet_prime_form(minor));
        let uniq = orbit_unique(major);
        assert!(uniq.contains(&major));
        assert!(uniq.contains(&minor));
    }

    #[test]
    fn augmented_triad_orbit_has_size_4() {
        // {0, 4, 8} is T₄-symmetric and inversion-symmetric → stabilizer of order 6 → orbit 24/6 = 4.
        let aug = PcSet::from_pcs([0, 4, 8]);
        assert_eq!(orbit_unique(aug).len(), 4);
    }

    #[test]
    fn diminished_seventh_orbit_has_size_3() {
        // {0, 3, 6, 9} has stabilizer of order 8 → orbit size 24/8 = 3.
        let dim7 = PcSet::from_pcs([0, 3, 6, 9]);
        assert_eq!(orbit_unique(dim7).len(), 3);
    }
}
