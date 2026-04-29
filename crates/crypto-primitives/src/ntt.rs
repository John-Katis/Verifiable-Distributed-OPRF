//! Radix-2 Cooley–Tukey number-theoretic transform over F_p.
//!
//! Given a primitive n-th root of unity ω ∈ F_p with n a power of two,
//! [`ntt`] computes the DFT
//!
//!     A_j = Σ_{i=0}^{n-1} a_i · ω^{i·j}   (j = 0..n)
//!
//! in-place in O(n log n) field operations. [`intt`] is the same butterfly
//! network driven by ω⁻¹ followed by a scale by n⁻¹.
//!
//! Convention matches Reed–Solomon encoding: if `a = (c_0, …, c_{k-1}, 0, …, 0)`
//! are polynomial coefficients padded to length n, then after [`ntt`] the
//! vector holds `(P(ω^0), P(ω^1), …, P(ω^{n-1}))` where
//! `P(x) = c_0 + c_1 x + … + c_{k-1} x^{k-1}`.

use num_bigint::BigUint;
use vdoprf_field::Fp;

/// In-place forward NTT. `omega` must be a primitive `vec.len()`-th root of
/// unity and `vec.len()` must be a power of two. Panics otherwise.
pub fn ntt(vec: &mut [Fp], omega: &Fp) {
    let n = vec.len();
    assert!(n.is_power_of_two(), "NTT length must be a power of 2");
    if n <= 1 {
        return;
    }
    let modulus = omega.modulus().clone();
    let log_n = n.trailing_zeros() as usize;

    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            vec.swap(i, j);
        }
    }

    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let w_step = omega.pow(&BigUint::from((n / size) as u64));
        let mut start = 0;
        while start < n {
            let mut w = Fp::one(&modulus);
            for j in 0..half {
                let t = &vec[start + j + half] * &w;
                let u = vec[start + j].clone();
                vec[start + j] = &u + &t;
                vec[start + j + half] = &u - &t;
                w = &w * &w_step;
            }
            start += size;
        }
        size *= 2;
    }
}

/// In-place inverse NTT. Pass the *forward* ω; this inverts it internally
/// and scales by n⁻¹.
pub fn intt(vec: &mut [Fp], omega: &Fp) {
    let n = vec.len();
    assert!(n.is_power_of_two(), "NTT length must be a power of 2");
    if n == 0 {
        return;
    }
    let modulus = omega.modulus().clone();
    let omega_inv = omega.inv().expect("ω must be a nonzero primitive root");
    ntt(vec, &omega_inv);
    if n == 1 {
        return;
    }
    let n_inv = Fp::new(BigUint::from(n as u64), &modulus)
        .inv()
        .expect("n must be invertible in F_p (n | p-1)");
    for v in vec.iter_mut() {
        *v = &*v * &n_inv;
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_p() -> BigUint {
        BigUint::from(65537u32)
    }

    fn primitive_n_th_root(n: usize, p: &BigUint) -> Fp {
        // p = 65537 = 2^16 + 1 → p − 1 = 2^16, and 3 is a generator of F_p^*.
        let g = Fp::new(BigUint::from(3u32), p);
        let exp = BigUint::from(65536u32) / BigUint::from(n as u64);
        g.pow(&exp)
    }

    #[test]
    fn test_ntt_roundtrip() {
        let p = test_p();
        let n = 8;
        let omega = primitive_n_th_root(n, &p);
        let a: Vec<Fp> = (0..n)
            .map(|i| Fp::new(BigUint::from((i * 7 + 3) as u32), &p))
            .collect();
        let mut work = a.clone();
        ntt(&mut work, &omega);
        intt(&mut work, &omega);
        assert_eq!(work, a);
    }

    #[test]
    fn test_ntt_matches_direct_dft() {
        let p = test_p();
        let n = 16;
        let omega = primitive_n_th_root(n, &p);
        let a: Vec<Fp> = (0..n)
            .map(|i| Fp::new(BigUint::from((i + 1) as u32), &p))
            .collect();
        let mut expected = vec![Fp::zero(&p); n];
        for j in 0..n {
            let mut sum = Fp::zero(&p);
            let mut w_pow = Fp::one(&p);
            let omega_j = omega.pow(&BigUint::from(j as u64));
            for item in a.iter() {
                sum = &sum + &(item * &w_pow);
                w_pow = &w_pow * &omega_j;
            }
            expected[j] = sum;
        }
        let mut work = a.clone();
        ntt(&mut work, &omega);
        assert_eq!(work, expected);
    }

    #[test]
    fn test_ntt_evaluates_padded_polynomial() {
        // NTT of coefficients padded with zeros = evaluations at ω^i.
        let p = test_p();
        let n_c = 16;
        let k = 5;
        let omega = primitive_n_th_root(n_c, &p);
        let coeffs: Vec<Fp> = (1..=k)
            .map(|i| Fp::new(BigUint::from(i as u32), &p))
            .collect();
        let mut padded: Vec<Fp> = coeffs.clone();
        padded.resize(n_c, Fp::zero(&p));
        ntt(&mut padded, &omega);
        for j in 0..n_c {
            let x = omega.pow(&BigUint::from(j as u64));
            let mut val = Fp::zero(&p);
            let mut x_pow = Fp::one(&p);
            for c in &coeffs {
                val = &val + &(c * &x_pow);
                x_pow = &x_pow * &x;
            }
            assert_eq!(padded[j], val, "position {j}");
        }
    }
}
