use num_bigint::BigUint;
use sha2::{Digest, Sha256};
use vdoprf_field::Fp;

/// Fiat-Shamir transcript for non-interactive proofs.
/// Accumulates commitments and derives deterministic challenges.
#[derive(Clone, Debug)]
pub struct Transcript {
    hasher: Sha256,
}

impl Transcript {
    pub fn new(domain_separator: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain_separator);
        Transcript { hasher }
    }

    /// Append raw bytes to the transcript.
    pub fn append_bytes(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    /// Append a field element to the transcript.
    pub fn append_field_element(&mut self, elem: &Fp) {
        let bytes = elem.value.to_bytes_be();
        // Prefix with length for domain separation
        self.hasher.update((bytes.len() as u32).to_be_bytes());
        self.hasher.update(&bytes);
    }

    /// Append a 32-byte hash/commitment to the transcript.
    pub fn append_commitment(&mut self, commitment: &[u8; 32]) {
        self.hasher.update(commitment);
    }

    /// Derive a challenge field element from the current transcript state.
    /// Does not consume the transcript — further appends are possible.
    pub fn challenge(&self, modulus: &BigUint) -> Fp {
        let hash = self.hasher.clone().finalize();
        let val = BigUint::from_bytes_be(&hash) % modulus;
        Fp::new(val, modulus)
    }

    /// Derive a challenge integer in [0, bound).
    pub fn challenge_index(&self, bound: usize) -> usize {
        let hash = self.hasher.clone().finalize();
        let val = BigUint::from_bytes_be(&hash);
        let bound_big = BigUint::from(bound);
        let idx = val % bound_big;
        let digits = idx.to_u64_digits();
        if digits.is_empty() {
            0
        } else {
            digits[0] as usize
        }
    }

    /// Derive multiple challenge field elements.
    pub fn challenge_vec(&self, count: usize, modulus: &BigUint) -> Vec<Fp> {
        let mut challenges = Vec::with_capacity(count);
        let mut t = self.clone();
        for i in 0..count {
            t.append_bytes(&(i as u64).to_be_bytes());
            challenges.push(t.challenge(modulus));
        }
        challenges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_deterministic() {
        let p = BigUint::from(113u32);
        let mut t1 = Transcript::new(b"test");
        let mut t2 = Transcript::new(b"test");
        t1.append_bytes(b"hello");
        t2.append_bytes(b"hello");
        assert_eq!(t1.challenge(&p), t2.challenge(&p));
    }

    #[test]
    fn test_transcript_different_inputs() {
        let p = BigUint::from(113u32);
        let mut t1 = Transcript::new(b"test");
        let mut t2 = Transcript::new(b"test");
        t1.append_bytes(b"hello");
        t2.append_bytes(b"world");
        assert_ne!(t1.challenge(&p), t2.challenge(&p));
    }

    #[test]
    fn test_challenge_in_range() {
        let p = BigUint::from(113u32);
        let t = Transcript::new(b"range_test");
        let c = t.challenge(&p);
        assert!(c.value < p);
    }
}
