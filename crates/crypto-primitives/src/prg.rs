use aes::cipher::{KeyIvInit, StreamCipher};
use num_bigint::BigUint;
use vdoprf_field::Fp;

type Aes128Ctr = ctr::Ctr64BE<aes::Aes128>;

/// Pairwise pseudorandom generator shared between two parties.
/// Uses AES-128-CTR to expand (seed, counter) into field elements.
#[derive(Clone, Debug)]
pub struct PairwisePrg {
    seed: [u8; 16],
}

impl PairwisePrg {
    pub fn new(seed: [u8; 16]) -> Self {
        PairwisePrg { seed }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut seed = [0u8; 16];
        let len = bytes.len().min(16);
        seed[..len].copy_from_slice(&bytes[..len]);
        PairwisePrg { seed }
    }

    /// Generate a deterministic field element from (seed, counter).
    /// Uses AES-CTR mode with the seed as key and counter-derived nonce.
    pub fn generate(&self, counter: u64, modulus: &BigUint) -> Fp {
        let bit_len = modulus.bits() as usize;
        let bytes_needed = (bit_len + 7) / 8 + 8; // extra bytes for rejection sampling

        // Use counter as the IV
        let mut iv = [0u8; 16];
        iv[8..16].copy_from_slice(&counter.to_be_bytes());

        let mut cipher = Aes128Ctr::new((&self.seed).into(), (&iv).into());
        let mut buf = vec![0u8; bytes_needed];
        cipher.apply_keystream(&mut buf);

        // Rejection sampling: interpret as BigUint and reduce mod p
        let val = BigUint::from_bytes_be(&buf) % modulus;
        Fp::new(val, modulus)
    }

    /// Generate raw bytes from (seed, counter).
    pub fn generate_bytes(&self, counter: u64, num_bytes: usize) -> Vec<u8> {
        let mut iv = [0u8; 16];
        iv[8..16].copy_from_slice(&counter.to_be_bytes());

        let mut cipher = Aes128Ctr::new((&self.seed).into(), (&iv).into());
        let mut buf = vec![0u8; num_bytes];
        cipher.apply_keystream(&mut buf);
        buf
    }

    /// Generate a child seed (for GGM tree expansion).
    pub fn expand_seed(&self, counter: u64) -> [u8; 16] {
        let bytes = self.generate_bytes(counter, 16);
        let mut seed = [0u8; 16];
        seed.copy_from_slice(&bytes);
        seed
    }
}

/// Replicated PRF: keyed PRF shared among parties not in a subset T.
/// Essentially a wrapper around PairwisePrg with a specific key.
#[derive(Clone, Debug)]
pub struct ReplicatedPrf {
    prg: PairwisePrg,
}

impl ReplicatedPrf {
    pub fn new(key: [u8; 16]) -> Self {
        ReplicatedPrf {
            prg: PairwisePrg::new(key),
        }
    }

    pub fn evaluate(&self, counter: u64, modulus: &BigUint) -> Fp {
        self.prg.generate(counter, modulus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    #[test]
    fn test_prg_deterministic() {
        let prg = PairwisePrg::new([1u8; 16]);
        let p = BigUint::from(113u32);
        let a = prg.generate(0, &p);
        let b = prg.generate(0, &p);
        assert_eq!(a, b);
    }

    #[test]
    fn test_prg_different_counters() {
        let prg = PairwisePrg::new([1u8; 16]);
        let p = BigUint::from(113u32);
        let a = prg.generate(0, &p);
        let b = prg.generate(1, &p);
        // Very unlikely to be equal
        assert_ne!(a, b);
    }

    #[test]
    fn test_prg_different_seeds() {
        let prg1 = PairwisePrg::new([1u8; 16]);
        let prg2 = PairwisePrg::new([2u8; 16]);
        let p = BigUint::from(113u32);
        let a = prg1.generate(0, &p);
        let b = prg2.generate(0, &p);
        assert_ne!(a, b);
    }

    #[test]
    fn test_replicated_prf() {
        let prf = ReplicatedPrf::new([42u8; 16]);
        let p = BigUint::from(113u32);
        let a = prf.evaluate(0, &p);
        let b = prf.evaluate(0, &p);
        assert_eq!(a, b);
    }

    #[test]
    fn test_expand_seed() {
        let prg = PairwisePrg::new([1u8; 16]);
        let s1 = prg.expand_seed(0);
        let s2 = prg.expand_seed(1);
        assert_ne!(s1, s2);
        assert_eq!(s1, prg.expand_seed(0)); // deterministic
    }
}
