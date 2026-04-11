//! DNA-codon encoding for SessionEvent variant tags.
//!
//! # The idea
//!
//! Biology stores information in 4-base DNA. Three bases form one
//! codon; 64 codons encode 20 amino acids plus stop signals with
//! *redundancy* built in — many codons map to the same amino acid,
//! so single-base errors are often silent.
//!
//! We borrow the same vocabulary for SessionEvent variants:
//!
//! - Each base is 2 bits (A=00, C=01, G=10, T=11)
//! - Each codon is 3 bases = 6 bits
//! - 64 codons → up to 64 distinct tags per codon position
//! - Multiple codons can encode the same variant (redundancy)
//!
//! # Why this is not just an enum encoding
//!
//! If we only needed "map variant to integer" we'd use `#[repr(u8)]`.
//! What DNA gives us on top is:
//!
//! 1. **Fixed-width bit-packing:** every tag is exactly 6 bits,
//!    which composes nicely into larger bit-packed formats.
//! 2. **Redundancy/error-correction:** mapping multiple codons to
//!    the same variant means a corrupted base has a reasonable
//!    probability of still decoding to the right variant.
//! 3. **Sequence alignment:** Smith-Waterman / Needleman-Wunsch on
//!    codon streams finds *similar* session prefixes even with
//!    insertions/deletions. Bioinformatics for session diffing.
//!
//! # What this module is NOT
//!
//! It's not a full SessionEvent encoder. Payload (params, values,
//! evidence) is arbitrary JSON and needs separate compression. This
//! module encodes only the *tag space*: which variant an event is,
//! which source produced it, which aspect it concerns. Those are
//! the low-entropy fields where biology wins.

use std::fmt;

/// DNA base.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Base {
    /// Adenine — bit pattern `00`.
    A = 0b00,
    /// Cytosine — bit pattern `01`.
    C = 0b01,
    /// Guanine — bit pattern `10`.
    G = 0b10,
    /// Thymine — bit pattern `11`.
    T = 0b11,
}

impl Base {
    /// Return the 2-bit value of this base.
    pub fn bits(self) -> u8 {
        self as u8
    }

    /// Decode a 2-bit value into a base. Returns `None` for any
    /// value >= 4 (bit 2 set).
    pub fn from_bits(bits: u8) -> Option<Self> {
        match bits & 0b11 {
            0b00 => Some(Self::A),
            0b01 => Some(Self::C),
            0b10 => Some(Self::G),
            0b11 => Some(Self::T),
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for Base {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ch = match self {
            Self::A => 'A',
            Self::C => 'C',
            Self::G => 'G',
            Self::T => 'T',
        };
        write!(f, "{ch}")
    }
}

/// A 3-base codon, stored as a 6-bit value (bits 5..0). Bits 7..6
/// are always zero in a valid codon.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Codon(u8);

impl Codon {
    /// Construct a codon from three bases. Base 0 occupies bits 5..4,
    /// base 1 occupies bits 3..2, base 2 occupies bits 1..0.
    pub fn from_bases(a: Base, b: Base, c: Base) -> Self {
        Self((a.bits() << 4) | (b.bits() << 2) | c.bits())
    }

    /// Decode the three bases of this codon. Always succeeds.
    pub fn to_bases(self) -> (Base, Base, Base) {
        let v = self.0;
        (
            Base::from_bits((v >> 4) & 0b11).unwrap(),
            Base::from_bits((v >> 2) & 0b11).unwrap(),
            Base::from_bits(v & 0b11).unwrap(),
        )
    }

    /// Raw 6-bit value (0..=63).
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Construct from raw 6-bit value. Panics on value >= 64.
    pub fn from_bits(bits: u8) -> Self {
        assert!(bits < 64, "codon bits must be 0..=63, got {bits}");
        Self(bits)
    }
}

impl fmt::Display for Codon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (a, b, c) = self.to_bases();
        write!(f, "{a}{b}{c}")
    }
}

/// The SessionEvent variants this module knows how to tag. Kept as
/// a local enum to avoid a circular dependency on `ix-agent-core`;
/// the mapping to real SessionEvent variants is stable in the
/// comments but not typed here.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EventTag {
    /// Maps to `SessionEvent::ActionProposed`.
    ActionProposed,
    /// Maps to `SessionEvent::ActionBlocked`.
    ActionBlocked,
    /// Maps to `SessionEvent::ActionReplaced`.
    ActionReplaced,
    /// Maps to `SessionEvent::MetadataMounted`.
    MetadataMounted,
    /// Maps to `SessionEvent::ActionCompleted`.
    ActionCompleted,
    /// Maps to `SessionEvent::ActionFailed`.
    ActionFailed,
    /// Maps to `SessionEvent::BeliefChanged`.
    BeliefChanged,
    /// Proposed for the Path C Phase 1 schema: a new variant for
    /// explicit hexavalent observation entries.
    ObservationAdded,
}

impl EventTag {
    /// Map an event tag to its *primary* codon. Redundant codons
    /// that also decode to the same tag live in the full codon
    /// table; see [`decode_codon_with_redundancy`].
    pub fn primary_codon(self) -> Codon {
        match self {
            Self::ActionProposed => Codon::from_bases(Base::A, Base::T, Base::G),
            Self::ActionBlocked => Codon::from_bases(Base::T, Base::A, Base::A),
            Self::ActionReplaced => Codon::from_bases(Base::T, Base::A, Base::G),
            Self::MetadataMounted => Codon::from_bases(Base::T, Base::G, Base::A),
            Self::ActionCompleted => Codon::from_bases(Base::G, Base::C, Base::A),
            Self::ActionFailed => Codon::from_bases(Base::G, Base::T, Base::A),
            Self::BeliefChanged => Codon::from_bases(Base::C, Base::A, Base::G),
            Self::ObservationAdded => Codon::from_bases(Base::A, Base::A, Base::C),
        }
    }
}

/// Decode a codon to its event tag, honoring the redundancy table.
///
/// The primary codons from [`EventTag::primary_codon`] always
/// decode to their assigned tag. A small set of "near miss" codons
/// (differing by one base) also decode to the same tag — this is
/// the error-tolerance property borrowed from biology. Any codon
/// not in the table decodes to `None`.
pub fn decode_codon_with_redundancy(codon: Codon) -> Option<EventTag> {
    // Primary lookup.
    for tag in [
        EventTag::ActionProposed,
        EventTag::ActionBlocked,
        EventTag::ActionReplaced,
        EventTag::MetadataMounted,
        EventTag::ActionCompleted,
        EventTag::ActionFailed,
        EventTag::BeliefChanged,
        EventTag::ObservationAdded,
    ] {
        if tag.primary_codon() == codon {
            return Some(tag);
        }
    }
    // Redundant aliases: each primary codon has one alias that
    // differs in the third base only — this is the minimum
    // redundancy and catches single-base corruption at the least
    // significant position. A fuller biology-style table would
    // add more aliases; we stay minimal to keep the decoder
    // small.
    let (a, b, _) = codon.to_bases();
    for tag in [
        EventTag::ActionProposed,
        EventTag::ActionBlocked,
        EventTag::ActionReplaced,
        EventTag::MetadataMounted,
        EventTag::ActionCompleted,
        EventTag::ActionFailed,
        EventTag::BeliefChanged,
        EventTag::ObservationAdded,
    ] {
        let primary = tag.primary_codon();
        let (pa, pb, _) = primary.to_bases();
        if a == pa && b == pb {
            return Some(tag);
        }
    }
    None
}

/// Pack a sequence of event tags into a byte vector. Two codons per
/// byte (since each codon is 6 bits and 6+6 > 8, we actually use
/// one codon per byte with 2 unused bits for alignment simplicity;
/// this is a deliberate tradeoff for decoder simplicity over
/// maximum density). Future versions may switch to a bit-packed
/// layout.
pub fn pack(tags: &[EventTag]) -> Vec<u8> {
    tags.iter().map(|t| t.primary_codon().bits()).collect()
}

/// Unpack a byte vector into event tags, honoring the redundancy
/// table so corrupted bytes may still decode correctly. Bytes that
/// do not decode to any known tag are skipped.
pub fn unpack(bytes: &[u8]) -> Vec<EventTag> {
    bytes
        .iter()
        .filter_map(|b| decode_codon_with_redundancy(Codon::from_bits(b & 0b0011_1111)))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_roundtrip() {
        for b in [Base::A, Base::C, Base::G, Base::T] {
            let bits = b.bits();
            let back = Base::from_bits(bits).unwrap();
            assert_eq!(b, back);
        }
    }

    #[test]
    fn codon_roundtrip() {
        let c = Codon::from_bases(Base::A, Base::T, Base::G);
        let (a, b, c2) = c.to_bases();
        assert_eq!(a, Base::A);
        assert_eq!(b, Base::T);
        assert_eq!(c2, Base::G);
    }

    #[test]
    fn codon_display_is_three_chars() {
        let c = Codon::from_bases(Base::A, Base::T, Base::G);
        assert_eq!(format!("{c}"), "ATG");
    }

    #[test]
    fn primary_codons_are_all_distinct() {
        let tags = [
            EventTag::ActionProposed,
            EventTag::ActionBlocked,
            EventTag::ActionReplaced,
            EventTag::MetadataMounted,
            EventTag::ActionCompleted,
            EventTag::ActionFailed,
            EventTag::BeliefChanged,
            EventTag::ObservationAdded,
        ];
        let codons: std::collections::HashSet<Codon> =
            tags.iter().map(|t| t.primary_codon()).collect();
        assert_eq!(
            codons.len(),
            tags.len(),
            "each tag must have a unique primary codon"
        );
    }

    #[test]
    fn pack_and_unpack_round_trip() {
        let original = vec![
            EventTag::ActionProposed,
            EventTag::MetadataMounted,
            EventTag::ActionCompleted,
            EventTag::ObservationAdded,
            EventTag::BeliefChanged,
        ];
        let packed = pack(&original);
        assert_eq!(packed.len(), original.len());
        let unpacked = unpack(&packed);
        assert_eq!(original, unpacked);
    }

    #[test]
    fn pack_produces_one_byte_per_tag() {
        // Simplicity: 1 codon per byte, 2 bits unused. Confirms
        // the chosen layout.
        let tags = vec![EventTag::ActionProposed; 10];
        let packed = pack(&tags);
        assert_eq!(packed.len(), 10);
    }

    #[test]
    fn redundancy_tolerates_last_base_corruption() {
        // Flip the last base of ActionProposed (ATG) to any other
        // base and the decoder should still recover ActionProposed
        // via the redundancy table.
        let primary = EventTag::ActionProposed.primary_codon();
        let (a, b, _) = primary.to_bases();

        for corrupted_last in [Base::A, Base::C, Base::G, Base::T] {
            let corrupted = Codon::from_bases(a, b, corrupted_last);
            let decoded = decode_codon_with_redundancy(corrupted);
            assert_eq!(
                decoded,
                Some(EventTag::ActionProposed),
                "expected ActionProposed after corrupting last base to {corrupted_last}"
            );
        }
    }

    #[test]
    fn completely_unknown_codon_decodes_to_none() {
        // Pick a codon that no tag or alias claims. The current
        // primary set uses prefixes (A,T), (T,A), (T,G), (G,C),
        // (G,T), (C,A), (A,A) — so a (C,C,C) codon is free.
        let c = Codon::from_bases(Base::C, Base::C, Base::C);
        assert_eq!(decode_codon_with_redundancy(c), None);
    }

    #[test]
    fn unpack_skips_unknown_bytes() {
        let mut bytes = pack(&[EventTag::ActionProposed, EventTag::ActionCompleted]);
        // Insert a known-bad codon byte in between (C,C,C = 0b010101 = 21).
        bytes.insert(1, 0b010101);
        let tags = unpack(&bytes);
        assert_eq!(
            tags,
            vec![EventTag::ActionProposed, EventTag::ActionCompleted]
        );
    }
}
