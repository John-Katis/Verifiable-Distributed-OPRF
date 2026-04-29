use num_bigint::BigUint;
use num_traits::{One, Zero};
use vdoprf_field::Fp;

use crate::ntt;

/// Reed–Solomon encoder over F_p with two construction modes:
///
/// * [`ReedSolomon::new`] uses the simple integer domain `{1, …, n_c}`. This
///   is used only by low-dependency unit tests; it cannot exploit NTT and
///   must fall back to Horner evaluation.
/// * [`ReedSolomon::with_root_of_unity`] uses the subgroup domain
///   `{ω^0, …, ω^{n_c−1}}`, caches ω, and precomputes the divided-difference
///   inverses needed for Newton interpolation. When `n_c` is a power of two,
///   [`encode`] dispatches to a radix-2 NTT (`O(n_c log n_c)`).
#[derive(Clone, Debug)]
pub struct ReedSolomon {
    /// Codeword length.
    pub n_c: usize,
    /// Message length (degree bound k−1, so k coefficients).
    pub k: usize,
    /// Evaluation domain points `[d_0, d_1, …, d_{n_c−1}]`.
    pub domain: Vec<Fp>,
    /// Field modulus.
    pub modulus: BigUint,
    /// Primitive `n_c`-th root of unity when constructed via
    /// [`with_root_of_unity`]; unset for the integer-domain constructor.
    omega: Option<Fp>,
    /// `diff_inv[i][j]` = `(domain[i] − domain[i − (j+1)])^{-1}` for
    /// `0 ≤ j < i < k`. Indexed as a ragged table. Built once at
    /// construction using Montgomery batched inversion (O(k²) multiplications
    /// plus a single modular inverse), removing the per-interpolation
    /// inversion bottleneck.
    diff_inv: Vec<Vec<Fp>>,
}

impl ReedSolomon {
    /// Integer-domain constructor (testing only).
    pub fn new(n_c: usize, k: usize, modulus: &BigUint) -> Self {
        assert!(k <= n_c, "message length must be <= codeword length");
        let domain: Vec<Fp> = (1..=n_c)
            .map(|i| Fp::new(BigUint::from(i), modulus))
            .collect();
        let diff_inv = build_diff_inv(&domain, k, modulus);
        ReedSolomon {
            n_c,
            k,
            domain,
            modulus: modulus.clone(),
            omega: None,
            diff_inv,
        }
    }

    /// Roots-of-unity constructor. Requires `n_c | (p − 1)` so that `ω`
    /// generates a subgroup of order `n_c`.
    pub fn with_root_of_unity(n_c: usize, k: usize, omega: Fp) -> Self {
        let modulus = omega.modulus().clone();
        // domain[i] = ω · domain[i-1] avoids n_c × O(log n_c) BigUint pow calls.
        let mut domain: Vec<Fp> = Vec::with_capacity(n_c);
        domain.push(Fp::one(&modulus));
        for _ in 1..n_c {
            let next = domain.last().unwrap() * &omega;
            domain.push(next);
        }
        let diff_inv = build_diff_inv(&domain, k, &modulus);
        ReedSolomon {
            n_c,
            k,
            domain,
            modulus,
            omega: Some(omega),
            diff_inv,
        }
    }

    /// Encode a message of up to `k` coefficients into a length-`n_c`
    /// codeword. Uses NTT when the domain is a subgroup of roots of unity
    /// and `n_c` is a power of two; otherwise falls back to Horner evaluation.
    pub fn encode(&self, message: &[Fp]) -> Vec<Fp> {
        assert!(message.len() <= self.k, "message too long");
        if let Some(omega) = &self.omega {
            if self.n_c.is_power_of_two() {
                let mut padded: Vec<Fp> = Vec::with_capacity(self.n_c);
                padded.extend_from_slice(message);
                padded.resize(self.n_c, Fp::zero(&self.modulus));
                ntt::ntt(&mut padded, omega);
                return padded;
            }
        }
        self.domain
            .iter()
            .map(|x| self.eval_poly(message, x))
            .collect()
    }

    /// Evaluate a polynomial at `x` using iterated powers (Horner-like).
    fn eval_poly(&self, coeffs: &[Fp], x: &Fp) -> Fp {
        let mut result = Fp::zero(&self.modulus);
        let mut x_pow = Fp::one(&self.modulus);
        for coeff in coeffs {
            result = &result + &(coeff * &x_pow);
            x_pow = &x_pow * x;
        }
        result
    }

    /// Proximity test: does `word` lie on an RS codeword of degree `< k`?
    ///
    /// When built via [`with_root_of_unity`] with a power-of-two `n_c`, this
    /// dispatches to an iNTT and checks that the top `n_c − k` coefficients
    /// vanish — O(n_c log n_c). This matches the paper's claimed verifier
    /// cost (§3a-zkp-Approach.tex L222–223) and is the interleaved proximity
    /// test required by Π_Lig (appendix.tex L490(ii)).
    ///
    /// The integer-domain fallback uses Lagrange interpolation (O((n_c−k)·k²))
    /// and is retained only for the low-dependency unit tests in this module.
    pub fn is_valid_codeword(&self, word: &[Fp]) -> bool {
        if word.len() != self.n_c {
            return false;
        }
        if let Some(omega) = &self.omega {
            if self.n_c.is_power_of_two() {
                let mut work = word.to_vec();
                ntt::intt(&mut work, omega);
                for coeff in &work[self.k..] {
                    if !coeff.is_zero() {
                        return false;
                    }
                }
                return true;
            }
        }
        let points: Vec<(&Fp, &Fp)> = self.domain[..self.k]
            .iter()
            .zip(word[..self.k].iter())
            .collect();
        for i in self.k..self.n_c {
            let expected = self.lagrange_interpolate(&points, &self.domain[i]);
            if expected != word[i] {
                return false;
            }
        }
        true
    }

    fn lagrange_interpolate(&self, points: &[(&Fp, &Fp)], x: &Fp) -> Fp {
        let mut result = Fp::zero(&self.modulus);
        for (i, (xi, yi)) in points.iter().enumerate() {
            let mut basis = Fp::one(&self.modulus);
            for (j, (xj, _)) in points.iter().enumerate() {
                if i != j {
                    let num = x - *xj;
                    let den = (*xi - *xj).inv().unwrap();
                    basis = &basis * &(&num * &den);
                }
            }
            result = &result + &(*yi * &basis);
        }
        result
    }

    /// Given `values[0..k]` interpreted as evaluations of a polynomial
    /// `P` of degree `< k` at `domain[0..k]`, return the `k` monomial
    /// coefficients of `P`. Uses Newton's divided differences with the
    /// precomputed inverse table, so the hot loop has no modular inverses.
    pub fn interpolate_coefficients(&self, values: &[Fp]) -> Vec<Fp> {
        assert!(values.len() >= self.k);
        let n = self.k;

        // Newton divided differences using cached (domain[i] − domain[i − j])⁻¹.
        let mut dd: Vec<Fp> = values[..n].to_vec();
        for j in 1..n {
            for i in (j..n).rev() {
                let num = &dd[i] - &dd[i - 1];
                // diff_inv[i][j-1] stores 1 / (domain[i] − domain[i - j]).
                dd[i] = &num * &self.diff_inv[i][j - 1];
            }
        }

        // Newton basis → monomial basis.
        let mut coeffs = vec![Fp::zero(&self.modulus); n];
        coeffs[0] = dd[n - 1].clone();
        for i in (0..n - 1).rev() {
            for j in (1..n).rev() {
                let term = &coeffs[j - 1] - &(&self.domain[i] * &coeffs[j]);
                coeffs[j] = term;
            }
            coeffs[0] = &(-&self.domain[i]) * &coeffs[0];
            coeffs[0] = &coeffs[0] + &dd[i];
        }
        coeffs
    }
}

/// Batched inversion of `{ domain[i] − domain[i − j] : 1 ≤ j ≤ i < k }` using
/// the Montgomery trick: one modular inverse plus O(#entries) multiplications.
fn build_diff_inv(domain: &[Fp], k: usize, modulus: &BigUint) -> Vec<Vec<Fp>> {
    if k <= 1 {
        return (0..k).map(|_| Vec::new()).collect();
    }

    // Flatten the (i, j) pairs into a single array, invert in one pass,
    // then scatter back to the ragged per-row layout.
    let mut flat: Vec<Fp> = Vec::with_capacity(k * (k - 1) / 2);
    for i in 1..k {
        for j in 1..=i {
            flat.push(&domain[i] - &domain[i - j]);
        }
    }
    let flat_inv = batch_inverse(&flat, modulus);

    let mut out: Vec<Vec<Fp>> = Vec::with_capacity(k);
    out.push(Vec::new());
    let mut cursor = 0usize;
    for i in 1..k {
        let row: Vec<Fp> = flat_inv[cursor..cursor + i].to_vec();
        out.push(row);
        cursor += i;
    }
    out
}

/// Montgomery batched inverse. Given nonzero `[a_0, …, a_{m-1}]`, returns
/// `[a_0^{-1}, …, a_{m-1}^{-1}]` using `3m − 2` multiplications and one
/// modular inverse.
fn batch_inverse(values: &[Fp], modulus: &BigUint) -> Vec<Fp> {
    let m = values.len();
    if m == 0 {
        return Vec::new();
    }
    // prefix[i] = values[0] · … · values[i]
    let mut prefix: Vec<Fp> = Vec::with_capacity(m);
    prefix.push(values[0].clone());
    for i in 1..m {
        let p = prefix.last().unwrap() * &values[i];
        prefix.push(p);
    }
    let total_inv = prefix[m - 1]
        .inv()
        .expect("batch_inverse: zero denominator (domain collision)");
    let mut inv = vec![Fp::zero(modulus); m];
    // Running suffix inverse: inv_of_prefix[i] = 1 / prefix[i].
    let mut running = total_inv;
    for i in (1..m).rev() {
        inv[i] = &running * &prefix[i - 1];
        running = &running * &values[i];
    }
    inv[0] = running;
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modulus() -> BigUint {
        BigUint::from(113u32)
    }

    #[test]
    fn test_encode_constant() {
        let p = modulus();
        let rs = ReedSolomon::new(5, 2, &p);
        let msg = vec![Fp::new(BigUint::from(42u32), &p)];
        let codeword = rs.encode(&msg);
        assert_eq!(codeword.len(), 5);
        for v in &codeword {
            assert_eq!(v.value, BigUint::from(42u32));
        }
    }

    #[test]
    fn test_encode_linear() {
        let p = modulus();
        let rs = ReedSolomon::new(5, 2, &p);
        let msg = vec![
            Fp::new(BigUint::from(1u32), &p),
            Fp::new(BigUint::from(2u32), &p),
        ];
        let codeword = rs.encode(&msg);
        assert_eq!(codeword[0].value, BigUint::from(3u32));
        assert_eq!(codeword[1].value, BigUint::from(5u32));
        assert_eq!(codeword[2].value, BigUint::from(7u32));
        assert_eq!(codeword[3].value, BigUint::from(9u32));
        assert_eq!(codeword[4].value, BigUint::from(11u32));
    }

    #[test]
    fn test_valid_codeword() {
        let p = modulus();
        let rs = ReedSolomon::new(5, 2, &p);
        let msg = vec![
            Fp::new(BigUint::from(1u32), &p),
            Fp::new(BigUint::from(2u32), &p),
        ];
        let codeword = rs.encode(&msg);
        assert!(rs.is_valid_codeword(&codeword));
    }

    #[test]
    fn test_invalid_codeword() {
        let p = modulus();
        let rs = ReedSolomon::new(5, 2, &p);
        let msg = vec![
            Fp::new(BigUint::from(1u32), &p),
            Fp::new(BigUint::from(2u32), &p),
        ];
        let mut codeword = rs.encode(&msg);
        codeword[3] = Fp::new(BigUint::from(99u32), &p);
        assert!(!rs.is_valid_codeword(&codeword));
    }

    #[test]
    fn test_interpolate_coefficients() {
        let p = modulus();
        let rs = ReedSolomon::new(5, 3, &p);
        let msg = vec![
            Fp::new(BigUint::from(5u32), &p),
            Fp::new(BigUint::from(3u32), &p),
            Fp::new(BigUint::from(2u32), &p),
        ];
        let codeword = rs.encode(&msg);
        let recovered = rs.interpolate_coefficients(&codeword);
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0].value, BigUint::from(5u32));
        assert_eq!(recovered[1].value, BigUint::from(3u32));
        assert_eq!(recovered[2].value, BigUint::from(2u32));
    }

    #[test]
    fn test_with_root_of_unity_roundtrip() {
        // p = 113, n_c = 7 (odd) → no NTT path; hits Horner fallback.
        let p = modulus();
        let n_c = 7usize;
        let k = 3usize;
        let omega = Fp::new(BigUint::from(49u32), &p);
        assert_eq!(omega.pow(&BigUint::from(n_c as u32)).value, BigUint::from(1u32));
        assert_ne!(omega.value, BigUint::from(1u32));

        let rs = ReedSolomon::with_root_of_unity(n_c, k, omega);
        let msg = vec![
            Fp::new(BigUint::from(5u32), &p),
            Fp::new(BigUint::from(3u32), &p),
            Fp::new(BigUint::from(2u32), &p),
        ];
        let codeword = rs.encode(&msg);
        assert_eq!(codeword.len(), n_c);
        assert!(rs.is_valid_codeword(&codeword));

        let recovered = rs.interpolate_coefficients(&codeword);
        assert_eq!(recovered.len(), k);
        for (got, exp) in recovered.iter().zip(msg.iter()) {
            assert_eq!(got.value, exp.value);
        }
    }

    #[test]
    fn test_root_of_unity_ntt_path() {
        // p = 65537, n_c = 16 (pow of 2) → NTT path. Matches naive encode.
        let p = BigUint::from(65537u32);
        let n_c = 16usize;
        let k = 5usize;
        // Generator 3 of F_65537^*; primitive 16th root = 3^(65536/16) = 3^4096.
        let omega = Fp::new(BigUint::from(3u32), &p).pow(&BigUint::from(4096u32));
        assert_eq!(omega.pow(&BigUint::from(n_c as u32)).value, BigUint::from(1u32));

        let rs_rou = ReedSolomon::with_root_of_unity(n_c, k, omega.clone());
        let msg = vec![
            Fp::new(BigUint::from(7u32), &p),
            Fp::new(BigUint::from(11u32), &p),
            Fp::new(BigUint::from(13u32), &p),
            Fp::new(BigUint::from(17u32), &p),
            Fp::new(BigUint::from(19u32), &p),
        ];
        let ntt_cw = rs_rou.encode(&msg);
        assert_eq!(ntt_cw.len(), n_c);

        // Naive reference: evaluate polynomial directly at each domain point.
        for j in 0..n_c {
            let x = omega.pow(&BigUint::from(j as u32));
            let mut val = Fp::zero(&p);
            let mut x_pow = Fp::one(&p);
            for c in &msg {
                val = &val + &(c * &x_pow);
                x_pow = &x_pow * &x;
            }
            assert_eq!(ntt_cw[j], val, "position {j}");
        }

        // Round-trip: the interpolator must recover the original coefficients.
        let recovered = rs_rou.interpolate_coefficients(&ntt_cw);
        assert_eq!(recovered.len(), k);
        for (got, exp) in recovered.iter().zip(msg.iter()) {
            assert_eq!(got.value, exp.value);
        }
    }

    #[test]
    fn test_batch_inverse_matches_elementwise() {
        let p = BigUint::from(65537u32);
        let vals: Vec<Fp> = [2u32, 3, 5, 7, 11, 13, 100, 99]
            .iter()
            .map(|v| Fp::new(BigUint::from(*v), &p))
            .collect();
        let got = batch_inverse(&vals, &p);
        for (v, g) in vals.iter().zip(got.iter()) {
            assert_eq!((v * g).value, BigUint::from(1u32));
        }
    }
}
