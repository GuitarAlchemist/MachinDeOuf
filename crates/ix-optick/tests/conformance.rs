//! Golden vector conformance tests for the OPTIC-K v3 binary format.
//!
//! These tests validate that the Rust reader (`ix-optick`) correctly reads index
//! files written in the same byte layout as the C# `OptickIndexWriter`. A test
//! helper builds a minimal valid OPTK v3 binary with 5 golden voicings, and each
//! test case asserts a specific correctness property of the reader.
//!
//! Cross-language float drift (FMA, rounding-mode differences) can introduce
//! ~1e-7 per dimension; the tolerance constants below account for that.

use std::io::Write;
use std::path::PathBuf;

use ix_optick::{compute_schema_hash, OptickIndex};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DIM: usize = 228;
const NUM_INSTRUMENTS: usize = 3;

/// Cross-language float tolerance (~1e-5 allows for FMA accumulation over 228 dims).
const FLOAT_TOL: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Partition-weight spec (matches the GA OptickIndexWriter defaults)
// ---------------------------------------------------------------------------

/// OPTIC-K v3 partition layout:
///   dims  0..5   -> PITCH         weight = sqrt(0.30)
///   dims  6..29  -> STRUCTURE     weight = sqrt(0.45)
///   dims 30..41  -> INTERVAL      weight = sqrt(0.10)
///   dims 42..53  -> REGISTER      weight = sqrt(0.03)
///   dims 54..65  -> PLAYABILITY   weight = sqrt(0.02)
///   dims 66..77  -> SYMBOLIC      weight = sqrt(0.10)
///   dims 78..227 -> RESERVED      weight = 0.0
fn canonical_partition_weights() -> [f32; DIM] {
    let mut w = [0.0f32; DIM];
    let sqrt = |x: f32| x.sqrt();
    for d in 0..6 {
        w[d] = sqrt(0.30);
    }
    for d in 6..30 {
        w[d] = sqrt(0.45);
    }
    for d in 30..42 {
        w[d] = sqrt(0.10);
    }
    for d in 42..54 {
        w[d] = sqrt(0.03);
    }
    for d in 54..66 {
        w[d] = sqrt(0.02);
    }
    for d in 66..78 {
        w[d] = sqrt(0.10);
    }
    // dims 78..228 stay 0.0
    w
}

// ---------------------------------------------------------------------------
// Golden voicing definitions
// ---------------------------------------------------------------------------

/// A golden voicing with hand-crafted embedding and known metadata.
struct GoldenVoicing {
    diagram: &'static str,
    instrument: &'static str,
    midi_notes: &'static [i32],
    quality_inferred: Option<&'static str>,
    /// Raw (pre-weight, pre-normalize) embedding: which dims are nonzero and their value.
    raw_pattern: Vec<(usize, f32)>,
}

/// Build the 5 golden voicings described in the spec.
fn golden_voicings() -> Vec<GoldenVoicing> {
    vec![
        // V0 (guitar): STRUCTURE dims 6..30 set to 0.5
        GoldenVoicing {
            diagram: "x-0-2-2-1-0",
            instrument: "guitar",
            midi_notes: &[40, 47, 52, 57, 59, 64],
            quality_inferred: Some("minor"),
            raw_pattern: (6..30).map(|d| (d, 0.5)).collect(),
        },
        // V1 (guitar): SYMBOLIC dims 66..78 set to 1.0
        GoldenVoicing {
            diagram: "3-2-0-0-0-3",
            instrument: "guitar",
            midi_notes: &[43, 47, 52, 55, 59, 67],
            quality_inferred: Some("major"),
            raw_pattern: (66..78).map(|d| (d, 1.0)).collect(),
        },
        // V2 (guitar): uniform 1/sqrt(228) across all dims
        GoldenVoicing {
            diagram: "x-x-0-2-3-2",
            instrument: "guitar",
            midi_notes: &[52, 57, 64, 67],
            quality_inferred: None,
            raw_pattern: (0..DIM).map(|d| (d, 1.0 / (DIM as f32).sqrt())).collect(),
        },
        // V3 (bass): same pattern as V0 (STRUCTURE)
        GoldenVoicing {
            diagram: "0-2-2-x",
            instrument: "bass",
            midi_notes: &[28, 33, 40],
            quality_inferred: Some("power"),
            raw_pattern: (6..30).map(|d| (d, 0.5)).collect(),
        },
        // V4 (ukulele): same pattern as V1 (SYMBOLIC)
        GoldenVoicing {
            diagram: "0-2-3-2",
            instrument: "ukulele",
            midi_notes: &[60, 64, 67, 72],
            quality_inferred: Some("major"),
            raw_pattern: (66..78).map(|d| (d, 1.0)).collect(),
        },
    ]
}

/// Apply partition weights and L2-normalize a raw pattern into a 228-dim f32 vector.
fn embed_golden(raw: &[(usize, f32)], weights: &[f32; DIM]) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    for &(d, val) in raw {
        v[d] = weights[d] * val;
    }
    // L2 normalize
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Binary builder -- assembles a valid OPTK v3 file byte-by-byte
// ---------------------------------------------------------------------------

/// Write a complete OPTK v3 binary file containing the given golden voicings.
/// Returns the path to the written file inside `dir`.
fn write_golden_index(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("golden.optk");
    let mut f = std::fs::File::create(&path).expect("create golden.optk");

    let voicings = golden_voicings();
    let weights = canonical_partition_weights();

    // Compute instrument counts: guitar=3, bass=1, ukulele=1
    // (must match ordering in golden_voicings)
    let inst_counts: [usize; 3] = [3, 1, 1];

    // Pre-compute embedded vectors
    let vectors: Vec<[f32; DIM]> = voicings
        .iter()
        .map(|gv| embed_golden(&gv.raw_pattern, &weights))
        .collect();

    // Pre-serialize metadata as msgpack
    let mut meta_buf: Vec<u8> = Vec::new();
    for gv in &voicings {
        let meta = serde_json::json!({
            "diagram": gv.diagram,
            "instrument": gv.instrument,
            "midiNotes": gv.midi_notes,
            "quality_inferred": gv.quality_inferred,
        });
        let packed = rmp_serde::to_vec(&meta).expect("msgpack serialize");
        meta_buf.extend_from_slice(&packed);
    }

    // Compute layout sizes
    let vectors_byte_len = voicings.len() * DIM * 4;

    // --- Header ---
    let mut hdr = Vec::new();
    // magic
    hdr.extend_from_slice(b"OPTK");
    // version
    hdr.extend_from_slice(&3u32.to_le_bytes());
    // header_size placeholder (offset 8)
    let header_size_pos = hdr.len();
    hdr.extend_from_slice(&0u32.to_le_bytes());
    // schema_hash
    hdr.extend_from_slice(&compute_schema_hash().to_le_bytes());
    // endian marker
    hdr.extend_from_slice(&0xFEFFu16.to_le_bytes());
    // reserved
    hdr.extend_from_slice(&0u16.to_le_bytes());
    // dimension
    hdr.extend_from_slice(&(DIM as u32).to_le_bytes());
    // count
    hdr.extend_from_slice(&(voicings.len() as u64).to_le_bytes());
    // instruments
    hdr.push(NUM_INSTRUMENTS as u8);
    // padding (7 bytes)
    hdr.extend_from_slice(&[0u8; 7]);

    // instrument slices: byte_offset (relative to vectors start) + count
    let mut running = 0usize;
    for &c in &inst_counts {
        let byte_off = (running * DIM * 4) as u64;
        hdr.extend_from_slice(&byte_off.to_le_bytes());
        hdr.extend_from_slice(&(c as u64).to_le_bytes());
        running += c;
    }

    // vectors_offset placeholder
    let vec_off_pos = hdr.len();
    hdr.extend_from_slice(&0u64.to_le_bytes());
    // metadata_offset placeholder
    let meta_off_pos = hdr.len();
    hdr.extend_from_slice(&0u64.to_le_bytes());
    // metadata_length placeholder
    let meta_len_pos = hdr.len();
    hdr.extend_from_slice(&0u64.to_le_bytes());

    // partition weights
    for &w in weights.iter() {
        hdr.extend_from_slice(&w.to_le_bytes());
    }

    // Patch header_size
    let header_size = hdr.len() as u32;
    hdr[header_size_pos..header_size_pos + 4].copy_from_slice(&header_size.to_le_bytes());

    // Patch vectors_offset
    let vectors_offset = header_size as u64;
    hdr[vec_off_pos..vec_off_pos + 8].copy_from_slice(&vectors_offset.to_le_bytes());

    // Patch metadata_offset
    let metadata_offset = vectors_offset + vectors_byte_len as u64;
    hdr[meta_off_pos..meta_off_pos + 8].copy_from_slice(&metadata_offset.to_le_bytes());

    // Patch metadata_length
    let metadata_length = meta_buf.len() as u64;
    hdr[meta_len_pos..meta_len_pos + 8].copy_from_slice(&metadata_length.to_le_bytes());

    // --- Write everything ---
    f.write_all(&hdr).expect("write header");

    // Vectors (little-endian f32)
    for vec in &vectors {
        for &x in vec.iter() {
            f.write_all(&x.to_le_bytes()).expect("write vector element");
        }
    }

    // Metadata
    f.write_all(&meta_buf).expect("write metadata");

    f.flush().expect("flush");
    drop(f);

    path
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

#[test]
fn test_golden_header_roundtrip() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_golden_index(&dir);
    let index = OptickIndex::open(&path).expect("open golden index");

    let hdr = index.header();
    assert_eq!(hdr.version, 3);
    assert_eq!(hdr.dimension, 228);
    assert_eq!(hdr.count, 5);
    assert_eq!(hdr.instruments, 3);
    assert_eq!(hdr.schema_hash, compute_schema_hash());

    // Instrument slice counts
    assert_eq!(hdr.instrument_slices[0].count, 3, "guitar count");
    assert_eq!(hdr.instrument_slices[1].count, 1, "bass count");
    assert_eq!(hdr.instrument_slices[2].count, 1, "ukulele count");

    // Instrument byte offsets are sequential
    assert_eq!(hdr.instrument_slices[0].byte_offset, 0);
    assert_eq!(
        hdr.instrument_slices[1].byte_offset,
        3 * DIM as u64 * 4,
        "bass offset = 3 guitar voicings * 228 dims * 4 bytes"
    );
    assert_eq!(
        hdr.instrument_slices[2].byte_offset,
        4 * DIM as u64 * 4,
        "ukulele offset = (3+1) voicings * 228 dims * 4 bytes"
    );

    // vectors_offset comes right after the header
    assert!(
        hdr.vectors_offset > 0,
        "vectors_offset must be nonzero"
    );
    assert!(
        hdr.metadata_offset > hdr.vectors_offset,
        "metadata follows vectors"
    );
    assert!(hdr.metadata_length > 0, "metadata is nonempty");
}

#[test]
fn test_golden_search_top1() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_golden_index(&dir);
    let index = OptickIndex::open(&path).expect("open golden index");

    // Build a query that matches V0 exactly: STRUCTURE dims 6..30 at 0.5,
    // weighted and normalized the same way the golden builder does.
    let weights = canonical_partition_weights();
    let raw_v0: Vec<(usize, f32)> = (6..30).map(|d| (d, 0.5)).collect();
    let embedded = embed_golden(&raw_v0, &weights);

    let results = index
        .search(&embedded, None, 1)
        .expect("search should succeed");

    assert_eq!(results.len(), 1);
    // V0 is at global index 0
    assert_eq!(results[0].index, 0, "top-1 should be V0");
    assert!(
        results[0].score > 0.99,
        "score should be > 0.99 for an exact-pattern match, got {}",
        results[0].score
    );
    assert_eq!(results[0].metadata.diagram, "x-0-2-2-1-0");
}

#[test]
fn test_golden_metadata_integrity() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_golden_index(&dir);
    let index = OptickIndex::open(&path).expect("open golden index");

    // Retrieve all 5 voicings by searching with a uniform query
    let query = [1.0f32 / (DIM as f32).sqrt(); DIM];
    let results = index
        .search(&query, None, 5)
        .expect("search all 5");
    assert_eq!(results.len(), 5, "should return all 5 voicings");

    // Collect results by index for deterministic checking
    let mut by_index: Vec<_> = results.iter().collect();
    by_index.sort_by_key(|r| r.index);

    // V0
    assert_eq!(by_index[0].metadata.diagram, "x-0-2-2-1-0");
    assert_eq!(by_index[0].metadata.instrument, "guitar");
    assert_eq!(by_index[0].metadata.midi_notes, vec![40, 47, 52, 57, 59, 64]);
    assert_eq!(
        by_index[0].metadata.quality_inferred.as_deref(),
        Some("minor")
    );

    // V1
    assert_eq!(by_index[1].metadata.diagram, "3-2-0-0-0-3");
    assert_eq!(by_index[1].metadata.instrument, "guitar");
    assert_eq!(by_index[1].metadata.midi_notes, vec![43, 47, 52, 55, 59, 67]);
    assert_eq!(
        by_index[1].metadata.quality_inferred.as_deref(),
        Some("major")
    );

    // V2
    assert_eq!(by_index[2].metadata.diagram, "x-x-0-2-3-2");
    assert_eq!(by_index[2].metadata.instrument, "guitar");
    assert_eq!(by_index[2].metadata.midi_notes, vec![52, 57, 64, 67]);
    assert!(by_index[2].metadata.quality_inferred.is_none());

    // V3
    assert_eq!(by_index[3].metadata.diagram, "0-2-2-x");
    assert_eq!(by_index[3].metadata.instrument, "bass");
    assert_eq!(by_index[3].metadata.midi_notes, vec![28, 33, 40]);
    assert_eq!(
        by_index[3].metadata.quality_inferred.as_deref(),
        Some("power")
    );

    // V4
    assert_eq!(by_index[4].metadata.diagram, "0-2-3-2");
    assert_eq!(by_index[4].metadata.instrument, "ukulele");
    assert_eq!(by_index[4].metadata.midi_notes, vec![60, 64, 67, 72]);
    assert_eq!(
        by_index[4].metadata.quality_inferred.as_deref(),
        Some("major")
    );
}

#[test]
fn test_golden_instrument_counts() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_golden_index(&dir);
    let index = OptickIndex::open(&path).expect("open golden index");

    // Use a uniform query so all voicings get a nonzero score
    let query = [1.0f32 / (DIM as f32).sqrt(); DIM];

    let guitar = index
        .search(&query, Some("guitar"), 10)
        .expect("guitar search");
    assert_eq!(guitar.len(), 3, "guitar partition should have 3 voicings");

    let bass = index
        .search(&query, Some("bass"), 10)
        .expect("bass search");
    assert_eq!(bass.len(), 1, "bass partition should have 1 voicing");

    let ukulele = index
        .search(&query, Some("ukulele"), 10)
        .expect("ukulele search");
    assert_eq!(ukulele.len(), 1, "ukulele partition should have 1 voicing");

    // Sum of instrument counts equals total
    assert_eq!(
        guitar.len() + bass.len() + ukulele.len(),
        index.count() as usize,
        "instrument counts should sum to total"
    );
}

#[test]
fn test_golden_partition_weights() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_golden_index(&dir);
    let index = OptickIndex::open(&path).expect("open golden index");

    let expected = canonical_partition_weights();
    let actual = &index.header().partition_weights;

    for (d, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (exp - act).abs() < FLOAT_TOL,
            "partition weight mismatch at dim {d}: expected {exp}, got {act}"
        );
    }

    // Spot-check specific partition boundaries
    assert!(
        (actual[0] - 0.30f32.sqrt()).abs() < FLOAT_TOL,
        "dim 0 (PITCH) weight should be sqrt(0.30)"
    );
    assert!(
        (actual[6] - 0.45f32.sqrt()).abs() < FLOAT_TOL,
        "dim 6 (STRUCTURE) weight should be sqrt(0.45)"
    );
    assert!(
        (actual[66] - 0.10f32.sqrt()).abs() < FLOAT_TOL,
        "dim 66 (SYMBOLIC) weight should be sqrt(0.10)"
    );
    assert_eq!(
        actual[78], 0.0,
        "dim 78 (RESERVED) weight should be 0.0"
    );
    assert_eq!(
        actual[227], 0.0,
        "dim 227 (last RESERVED) weight should be 0.0"
    );
}

#[test]
fn test_golden_cross_instrument_isolation() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_golden_index(&dir);
    let index = OptickIndex::open(&path).expect("open golden index");

    // Build a query that exactly matches the bass voicing (V3 = STRUCTURE pattern).
    let weights = canonical_partition_weights();
    let raw_bass: Vec<(usize, f32)> = (6..30).map(|d| (d, 0.5)).collect();
    let bass_query = embed_golden(&raw_bass, &weights);

    // Search bass-only: should return exactly 1 result
    let bass_results = index
        .search(&bass_query, Some("bass"), 10)
        .expect("bass search");
    assert_eq!(bass_results.len(), 1, "bass partition has exactly 1 voicing");
    assert_eq!(bass_results[0].metadata.instrument, "bass");
    assert_eq!(bass_results[0].metadata.diagram, "0-2-2-x");
    assert_eq!(bass_results[0].index, 3, "bass voicing is at global index 3");

    // The bass result should NOT contain any guitar voicings
    for r in &bass_results {
        assert_ne!(
            r.metadata.instrument, "guitar",
            "bass search must not return guitar voicings"
        );
        assert_ne!(
            r.metadata.instrument, "ukulele",
            "bass search must not return ukulele voicings"
        );
    }

    // Search ukulele-only: should return exactly 1 result
    let ukulele_results = index
        .search(&bass_query, Some("ukulele"), 10)
        .expect("ukulele search");
    assert_eq!(
        ukulele_results.len(),
        1,
        "ukulele partition has exactly 1 voicing"
    );
    assert_eq!(ukulele_results[0].metadata.instrument, "ukulele");
    assert_eq!(ukulele_results[0].index, 4, "ukulele voicing is at global index 4");

    // Guitar search with the bass query: V0 should be top-1 (same STRUCTURE pattern)
    let guitar_results = index
        .search(&bass_query, Some("guitar"), 3)
        .expect("guitar search");
    assert_eq!(guitar_results.len(), 3);
    assert_eq!(
        guitar_results[0].index, 0,
        "V0 (same pattern as bass) should be top-1 in guitar"
    );
    assert!(
        guitar_results[0].score > 0.99,
        "V0 should near-perfectly match the STRUCTURE query, got {}",
        guitar_results[0].score
    );
}
