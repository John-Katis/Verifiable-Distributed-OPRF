use sha2::{Digest, Sha256};
use vdoprf_field::Fp;

/// Hash arbitrary bytes to a 32-byte digest (SHA-256).
pub fn hash_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Hash a commitment: H(data || salt).
pub fn hash_commitment(data: &[u8], salt: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.update(salt);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Hash a list of field elements.
///
/// Each element is length-prefixed (4-byte big-endian byte count) before
/// its own big-endian bytes, mirroring `Transcript::append_field_element`.
/// Without this, `BigUint::to_bytes_be()`'s variable-width, leading-zero-
/// stripped encoding lets two structurally different element vectors
/// concatenate to the identical byte string — e.g. `[Fp(1), Fp(2)]` (bytes
/// `01 02`) would otherwise collide with `[Fp(0x0102)]` (also `01 02`).
pub fn hash_field_elements(elements: &[Fp]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for elem in elements {
        let bytes = elem.value.to_bytes_be();
        hasher.update((bytes.len() as u32).to_be_bytes());
        hasher.update(&bytes);
    }
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Hash two 32-byte values (for Merkle tree internal nodes).
pub fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    #[test]
    fn test_hash_deterministic() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_field_elements() {
        let p = BigUint::from(113u32);
        let a = Fp::new(BigUint::from(42u32), &p);
        let b = Fp::new(BigUint::from(43u32), &p);
        let h = hash_field_elements(&[a, b]);
        assert_eq!(h.len(), 32);
    }

    /// Without length-framing, `[Fp(1), Fp(2)]` (bytes `01 || 02`) and
    /// `[Fp(0x0102)]` (bytes `01 02`) concatenate to the identical byte
    /// string and would hash identically — a real commitment-framing break.
    /// The 4-byte length prefix per element must disambiguate them.
    #[test]
    fn test_hash_field_elements_no_cross_element_ambiguity() {
        // Large enough modulus that 0x0102 = 258 doesn't wrap.
        let p = BigUint::from(1_000_000u32);
        let one = Fp::new(BigUint::from(1u32), &p);
        let two = Fp::new(BigUint::from(2u32), &p);
        let combined = Fp::new(BigUint::from(0x0102u32), &p);

        let h_split = hash_field_elements(&[one, two]);
        let h_combined = hash_field_elements(&[combined]);
        assert_ne!(
            h_split, h_combined,
            "length-framing must prevent [1,2] from colliding with [0x0102]",
        );
    }
}
