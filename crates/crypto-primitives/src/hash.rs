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
pub fn hash_field_elements(elements: &[Fp]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for elem in elements {
        hasher.update(elem.value.to_bytes_be());
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
}
