//! Approach III-b: Ligero-based Dual-Share ZKP (Pi_Lig), "balanced layout"
//! fixed protocol (Z4/Z5/Z6/Z7 fixes).
//!
//! ## Layout
//!
//! `B` instances are packed into `l = ceil(B/c)` segments of `c` instances
//! each (the last segment padded with dummy instances `M=1, a=0, delta=1`,
//! which trivially satisfy every constraint). A segment has four rows of
//! length `L = c*N`: `M`, `a`, running-product `p`, running-sum `s`. Instance
//! `nu` (0-indexed within its segment) occupies local positions
//! `[nu*N, nu*N+N-1)`. `S` = instance starts, `E` = instance ends, `W` =
//! `[0, L)`.
//!
//! ## Domains
//!
//! `D = <omega>` (size `n_c`), `H = <eta>` (size `k'`), `eta = omega^(n_c/k')`,
//! `H <= D`. Each row polynomial has degree `< k'`: it matches the real data
//! on `H`'s first `L` points and FRESH UNIFORM RANDOM values on the rest of
//! `H` (the Z4 zero-knowledge fix). `K* = 2k' - 1`.
//!
//! ## Constraints (per segment, with public `Delta_sigma`)
//!
//!   h1 = f_p(eta x) - f_p(x) f_M(eta x)      vanishes on W \ E
//!   h2 = f_s(eta x) - f_s(x) - f_a(eta x)     vanishes on W \ E
//!   h3 = f_p(x) - f_M(x)                      vanishes on S
//!   h4 = f_s(x) - f_a(x)                      vanishes on S
//!   h5 = f_p(x) - f_s(x) - Delta_sigma(x)     vanishes on E
//!
//! `N_sigma = Z_E*(h1 + alpha*h2) + Z_{W\S}*(alpha^2*h3 + alpha^3*h4)
//!            + alpha^4 * Z_{W\E} * h5`, batched across segments as
//! `N = sum_sigma alpha^{5*sigma} * N_sigma`, `C_sigma = N_sigma / Z_W`
//! (accumulated per-segment, matching linearity of division).
//!
//! ## Row commitment
//!
//! All `4l + n_parties + 1` rows (segment data rows, one mask row `Z_j` per
//! party, one shared blinding row `Z_0`) are RS-encoded over `D` and
//! Merkle-committed with a FLAT single-level leaf `H(column values)`. This
//! replaces the old two-level share/private split, which was itself the Z7
//! vulnerability's mechanism (Merkle membership with no codeword binding).
//!
//! ## Per-party binding polynomial (the Z7 fix)
//!
//! Each party `j` gets `q_j = sum_sigma Lambda_{j,sigma} * (gamma_{j,sigma,M}
//! * f_{M,sigma} + gamma_{j,sigma,a} * f_{a,sigma}) + Z_j`, sent P2P only
//! (never broadcast — broadcasting would leak a linear relation on the
//! hidden witness at positions the recipient doesn't hold). `Lambda_{j,sigma}`
//! is the degree-`<k'` polynomial equal to `beta_{j,i}` at each local
//! position `i` party `j` holds in that segment, and `0` elsewhere on `H`.
//! Verification splits into [`ligero_verify`] (shared: proximity/constraint/
//! consistency, every party runs identically) and [`ligero_verify_as_party`]
//! (per-party: binds `q_j` to the party's own locally-known shares).
//!
//! ## Sampling order (the Z6 fix)
//!
//! Every `h_j = H(q_j)` is committed into the transcript BEFORE the `q`
//! query points are sampled, so a prover cannot re-roll `q_j` to steer
//! sampling away from a corrupted column. Repetition/query counts are sized
//! for the computational parameter `lambda` (grinding resistance), not a
//! smaller statistical parameter.

use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use vdoprf_crypto::hash::hash_field_elements;
use vdoprf_crypto::merkle::MerkleTree;
use vdoprf_crypto::ntt;
use vdoprf_crypto::transcript::Transcript;
use vdoprf_field::Fp;

/// Upper bound on how far above `min_n_c` the divisor search is willing to
/// walk before declaring infeasibility.
const DIVISOR_SEARCH_EXTRA: usize = 1 << 20;
/// Safety cap on how many times the layout search doubles `k'` before giving up.
const MAX_K_PRIME_DOUBLINGS: u32 = 40;
/// Safety cap on how many times the layout search doubles `n_c` (for a fixed
/// `k'`) before moving on to a bigger `k'`.
const MAX_N_C_DOUBLINGS: u32 = 40;

/// Domain + degree-bound parameters for one Ligero proof.
#[derive(Clone, Debug)]
pub struct LigeroParams {
    /// `|D|`, a power of two dividing `p-1`.
    pub n_c: usize,
    /// `|H|`, a power of two, `k' | n_c`.
    pub k_prime: usize,
    /// `2*k_prime - 1`: the degree bound for mask rows, `u`, `C`, and `q_j`.
    pub k_star: usize,
    /// `|Q|`: number of sampled points in `D \ H`.
    pub num_queries: usize,
    /// Generator of `D`.
    pub omega: Fp,
    /// Generator of `H` (`omega^(n_c/k_prime)`).
    pub eta: Fp,
    /// Computational security parameter driving the soundness/grinding bound.
    pub lambda: u32,
}

/// Segment/instance layout for `B` instances packed into `l` segments of `c`.
#[derive(Clone, Debug)]
pub struct LigeroLayout {
    /// `N`: wires per instance.
    pub n_wires: usize,
    /// Instances per segment.
    pub c: usize,
    /// Number of segments, `ceil(B/c)`.
    pub l: usize,
    /// `B`: real (unpadded) instance count.
    pub b_total: usize,
    /// `L = c * n_wires`: length of a segment's data rows.
    pub l_len: usize,
}

impl LigeroLayout {
    pub fn instance_start(&self, nu: usize) -> usize {
        nu * self.n_wires
    }

    pub fn instance_end(&self, nu: usize) -> usize {
        nu * self.n_wires + self.n_wires - 1
    }

    pub fn num_padded_instances(&self) -> usize {
        self.l * self.c
    }
}

/// One Pi_Lig instance: a dealer's `(m_T, a_T)` witness plus, for every party
/// `j`, the local wire positions (`K_j` subset of `[0, N)`) that party
/// independently holds. This is public protocol structure (derivable from
/// the subset family), not secret data.
#[derive(Clone, Debug)]
pub struct LigeroInstance {
    pub m_values: Vec<Fp>,
    pub a_values: Vec<Fp>,
    pub delta: Fp,
    pub verifier_positions: Vec<Vec<usize>>,
}

impl LigeroInstance {
    pub fn new(
        m_values: Vec<Fp>,
        a_values: Vec<Fp>,
        delta: Fp,
        verifier_positions: Vec<Vec<usize>>,
    ) -> Self {
        assert_eq!(m_values.len(), a_values.len(), "m/a wire counts must match");
        LigeroInstance { m_values, a_values, delta, verifier_positions }
    }

    pub fn compute_delta(m_values: &[Fp], a_values: &[Fp], modulus: &BigUint) -> Fp {
        let mut product = Fp::one(modulus);
        for m in m_values {
            product = &product * m;
        }
        let mut sum = Fp::zero(modulus);
        for a in a_values {
            sum = &sum + a;
        }
        &product - &sum
    }
}

// ============================================================================
// Layout / parameter search.
// ============================================================================

/// Find the smallest sound `(k', q, c, l, n_c)` layout for `n_wires`-wire
/// instances batching `b_total` of them, sized for computational parameter
/// `lambda`. Returns `None` if no layout is found within the search bounds.
///
/// Simplification vs. a fully paper-faithful search: this returns the FIRST
/// feasible `k'` (smallest power of two clearing the structural bounds) and,
/// for it, the smallest `q` meeting the soundness bound with `e` fixed at its
/// paper-allowed maximum — it does not additionally search across `k'` for
/// the byte-minimal layout (which would also require the target party count
/// `n`, not part of this function's signature). This costs some proof-size
/// optimality, never soundness: every returned layout still satisfies the
/// full bound below.
pub fn try_new_layout(
    n_wires: usize,
    b_total: usize,
    lambda: u32,
    modulus: &BigUint,
) -> Option<(LigeroParams, LigeroLayout)> {
    if n_wires == 0 || b_total == 0 {
        return None;
    }
    let p_minus_1 = modulus - BigUint::one();
    let mut k_prime = n_wires.max(4).next_power_of_two();

    for _ in 0..MAX_K_PRIME_DOUBLINGS {
        let k_star = 2 * k_prime - 1;
        // `n_c >= 4k'` is necessary but nowhere near sufficient: the second
        // soundness term's ratio `(2K*+c-2+2e)/T` starts near 4/3 at the
        // minimal n_c (since T ~ 3k' there while 2K* alone is ~4k'), which
        // is > 1 and only gets worse under a higher power `q` — no `q` can
        // ever satisfy the bound until `n_c` grows enough to push `T` well
        // past `2K*+c`. So also search over growing `n_c`, not just the
        // smallest structurally-valid divisor.
        let mut min_n_c = (4 * k_prime).max(k_star + 1);
        for _ in 0..MAX_N_C_DOUBLINGS {
            let search_cap = min_n_c.saturating_add(DIVISOR_SEARCH_EXTRA);
            let found = find_smallest_pow2_divisor_ge(&p_minus_1, min_n_c, search_cap).and_then(|n_c| {
                if n_c % k_prime != 0 {
                    return None;
                }
                let omega = find_primitive_root_of_unity(n_c, modulus)?;
                let s = n_c / k_prime;
                let eta = omega.pow(&BigUint::from(s as u64));
                let t = n_c - k_prime;
                if t <= k_star {
                    return None;
                }
                let e = (t - k_star) / 4;
                let (q, c, l) = smallest_sound_q(n_wires, b_total, k_prime, t, k_star, e, lambda)?;
                Some((n_c, omega, eta, q, c, l))
            });
            if let Some((n_c, omega, eta, q, c, l)) = found {
                let layout = LigeroLayout { n_wires, c, l, b_total, l_len: c * n_wires };
                let params = LigeroParams { n_c, k_prime, k_star, num_queries: q, omega, eta, lambda };
                return Some((params, layout));
            }
            min_n_c = min_n_c.saturating_mul(2);
        }
        k_prime = k_prime.saturating_mul(2);
    }
    None
}

/// Smallest `q >= 1` (with the resulting `c = min(B, (k'-2q)/N) > 0`) meeting
/// the two-term soundness bound `(1-e/T)^q <= 2^-lambda` and
/// `((2K*+c-2+2e)/T)^q <= 2^-lambda`. Returns `(q, c, l)`.
fn smallest_sound_q(
    n_wires: usize,
    b_total: usize,
    k_prime: usize,
    t: usize,
    k_star: usize,
    e: usize,
    lambda: u32,
) -> Option<(usize, usize, usize)> {
    let bound = 2f64.powi(-(lambda as i32));
    let t_f = t as f64;
    let mut q = 1usize;
    loop {
        if 2 * q >= k_prime {
            return None;
        }
        let c = b_total.min((k_prime - 2 * q) / n_wires);
        if c == 0 {
            return None;
        }
        let l = b_total.div_ceil(c);
        let term1 = (1.0 - e as f64 / t_f).powi(q as i32);
        let term2 = ((2 * k_star + c).saturating_sub(2) as f64 + 2.0 * e as f64) / t_f;
        let term2 = term2.powi(q as i32);
        if term1 <= bound && term2 <= bound {
            return Some((q, c, l));
        }
        q += 1;
        if q > k_prime {
            return None;
        }
    }
}

fn find_smallest_pow2_divisor_ge(n: &BigUint, min_val: usize, max_val: usize) -> Option<usize> {
    let mut d = min_val.max(1).next_power_of_two();
    while d <= max_val {
        let d_big = BigUint::from(d);
        if &d_big > n {
            return None;
        }
        if (n % &d_big).is_zero() {
            return Some(d);
        }
        d = d.checked_mul(2)?;
    }
    None
}

fn find_primitive_root_of_unity(n: usize, modulus: &BigUint) -> Option<Fp> {
    let p_minus_1 = modulus - BigUint::one();
    let n_big = BigUint::from(n);
    if !(&p_minus_1 % &n_big).is_zero() {
        return None;
    }
    let exp = &p_minus_1 / &n_big;
    let factors = prime_factors(n);
    for g_val in 2u32..10_000 {
        let g_big = BigUint::from(g_val);
        if g_big >= *modulus {
            break;
        }
        let g = Fp::new(g_big, modulus);
        let w = g.pow(&exp);
        if is_primitive_n_root(&w, n, &factors, modulus) {
            return Some(w);
        }
    }
    None
}

fn is_primitive_n_root(w: &Fp, n: usize, factors: &[usize], modulus: &BigUint) -> bool {
    if w.is_zero() {
        return false;
    }
    if w.pow(&BigUint::from(n)) != Fp::one(modulus) {
        return false;
    }
    for &q in factors {
        if w.pow(&BigUint::from(n / q)) == Fp::one(modulus) {
            return false;
        }
    }
    true
}

fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut p = 2;
    while p * p <= n {
        if n % p == 0 {
            factors.push(p);
            while n % p == 0 {
                n /= p;
            }
        }
        p += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

// ============================================================================
// Polynomial arithmetic helpers.
// ============================================================================

fn poly_add(a: &[Fp], b: &[Fp], modulus: &BigUint) -> Vec<Fp> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let ai = a.get(i).cloned().unwrap_or_else(|| Fp::zero(modulus));
            let bi = b.get(i).cloned().unwrap_or_else(|| Fp::zero(modulus));
            &ai + &bi
        })
        .collect()
}

fn poly_sub(a: &[Fp], b: &[Fp], modulus: &BigUint) -> Vec<Fp> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let ai = a.get(i).cloned().unwrap_or_else(|| Fp::zero(modulus));
            let bi = b.get(i).cloned().unwrap_or_else(|| Fp::zero(modulus));
            &ai - &bi
        })
        .collect()
}

fn poly_mul(a: &[Fp], b: &[Fp], modulus: &BigUint) -> Vec<Fp> {
    if a.is_empty() || b.is_empty() {
        return vec![Fp::zero(modulus)];
    }
    let mut result = vec![Fp::zero(modulus); a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] = &result[i + j] + &(&a[i] * &b[j]);
        }
    }
    result
}

fn poly_scale(p: &[Fp], s: &Fp) -> Vec<Fp> {
    p.iter().map(|c| c * s).collect()
}

/// Coefficients of `p(eta * x)`: scale coefficient `i` by `eta^i`.
fn poly_shift(p: &[Fp], eta: &Fp, modulus: &BigUint) -> Vec<Fp> {
    let mut eta_pow = Fp::one(modulus);
    let mut out = Vec::with_capacity(p.len());
    for c in p {
        out.push(c * &eta_pow);
        eta_pow = &eta_pow * eta;
    }
    out
}

fn poly_divide_exact(num: &[Fp], div: &[Fp], modulus: &BigUint) -> Vec<Fp> {
    let n = num.len();
    let d = div.len();
    if n < d {
        return vec![Fp::zero(modulus)];
    }
    let mut rem: Vec<Fp> = num.to_vec();
    let lead_inv = div[d - 1].inv().expect("divisor leading coeff non-invertible");
    let mut quot = vec![Fp::zero(modulus); n - d + 1];
    for k in (0..(n - d + 1)).rev() {
        let q = &rem[d - 1 + k] * &lead_inv;
        for j in 0..d {
            rem[k + j] = &rem[k + j] - &(&q * &div[j]);
        }
        quot[k] = q;
    }
    quot
}

fn poly_eval(p: &[Fp], x: &Fp, modulus: &BigUint) -> Fp {
    let mut result = Fp::zero(modulus);
    for c in p.iter().rev() {
        result = &(&result * x) + c;
    }
    result
}

/// `eta^0, eta^1, ..., eta^{count-1}`.
fn eta_powers(eta: &Fp, count: usize, modulus: &BigUint) -> Vec<Fp> {
    let mut v = Vec::with_capacity(count);
    let mut cur = Fp::one(modulus);
    for _ in 0..count {
        v.push(cur.clone());
        cur = &cur * eta;
    }
    v
}

/// Vanishing polynomial of `{eta^0, ..., eta^{count-1}}` (a contiguous prefix).
fn vanishing_prefix(eta_pows: &[Fp], count: usize, modulus: &BigUint) -> Vec<Fp> {
    let mut result: Vec<Fp> = vec![Fp::one(modulus)];
    for r in &eta_pows[..count] {
        result = poly_mul(&result, &[-r, Fp::one(modulus)], modulus);
    }
    result
}

/// Vanishing polynomial of `{eta^i : i in indices}` (an arbitrary index set).
fn vanishing_from_indices(eta_pows: &[Fp], indices: &[usize], modulus: &BigUint) -> Vec<Fp> {
    let mut result: Vec<Fp> = vec![Fp::one(modulus)];
    for &i in indices {
        result = poly_mul(&result, &[-&eta_pows[i], Fp::one(modulus)], modulus);
    }
    result
}

/// Pointwise evaluation of the vanishing polynomial of `{eta^0..eta^{count-1}}` at `x`.
fn eval_vanishing_prefix(x: &Fp, eta_pows: &[Fp], count: usize, modulus: &BigUint) -> Fp {
    let mut result = Fp::one(modulus);
    for r in &eta_pows[..count] {
        result = &result * &(x - r);
    }
    result
}

/// Pointwise evaluation of the vanishing polynomial of `{eta^i : i in indices}` at `x`.
fn eval_vanishing_from_indices(x: &Fp, eta_pows: &[Fp], indices: &[usize], modulus: &BigUint) -> Fp {
    let mut result = Fp::one(modulus);
    for &i in indices {
        result = &result * &(x - &eta_pows[i]);
    }
    result
}

/// Interpolate `values` given at ALL `k'` points of `H = <eta>` back to the
/// unique degree-`<k'` polynomial's coefficients. `values.len()` must equal
/// `eta`'s order (a power of two). This is a full-domain interpolation,
/// exactly an inverse NTT (`O(k' log k')`), not the `O(k'^2)` Newton method
/// `ReedSolomon::interpolate_coefficients` would use at full length.
fn interpolate_over_full_domain(values: &[Fp], eta: &Fp) -> Vec<Fp> {
    let mut work = values.to_vec();
    ntt::intt(&mut work, eta);
    work
}

/// General Newton-divided-difference interpolation through arbitrary
/// (not necessarily domain-prefix) `(x_i, y_i)` pairs. `O(count^2)`, fine
/// since `count` (a segment size `c`) is expected to be small.
fn interpolate_at_points(points: &[(Fp, Fp)], modulus: &BigUint) -> Vec<Fp> {
    let n = points.len();
    if n == 0 {
        return vec![Fp::zero(modulus)];
    }
    let xs: Vec<Fp> = points.iter().map(|(x, _)| x.clone()).collect();
    let mut dd: Vec<Fp> = points.iter().map(|(_, y)| y.clone()).collect();
    for j in 1..n {
        for i in (j..n).rev() {
            let num = &dd[i] - &dd[i - 1];
            let den = (&xs[i] - &xs[i - j])
                .inv()
                .expect("interpolate_at_points: duplicate x coordinate");
            dd[i] = &num * &den;
        }
    }
    let mut coeffs = vec![Fp::zero(modulus); n];
    coeffs[0] = dd[n - 1].clone();
    for i in (0..n - 1).rev() {
        for j in (1..n).rev() {
            coeffs[j] = &coeffs[j - 1] - &(&xs[i] * &coeffs[j]);
        }
        coeffs[0] = &(-&xs[i]) * &coeffs[0];
        coeffs[0] = &coeffs[0] + &dd[i];
    }
    coeffs
}

/// Pad `coeffs` to length `n_c` and forward-NTT: the `D`-evaluation codeword.
fn encode_domain(coeffs: &[Fp], n_c: usize, omega: &Fp, modulus: &BigUint) -> Vec<Fp> {
    let mut padded = coeffs.to_vec();
    assert!(padded.len() <= n_c, "coefficient vector longer than n_c");
    padded.resize(n_c, Fp::zero(modulus));
    ntt::ntt(&mut padded, omega);
    padded
}

/// Is `word` (length `n_c`) a valid RS codeword of degree `< degree_bound`?
fn is_valid_low_degree(word: &[Fp], degree_bound: usize, omega: &Fp) -> bool {
    let mut work = word.to_vec();
    ntt::intt(&mut work, omega);
    work[degree_bound..].iter().all(|c| c.is_zero())
}

fn domain_point(omega: &Fp, index: usize) -> Fp {
    omega.pow(&BigUint::from(index as u64))
}

// ============================================================================
// Constraint system (h1..h5, batched per-segment numerator).
// ============================================================================

/// Build `N_sigma(x)` (coefficient form) for one segment from its four row
/// polynomials and public `Delta_sigma`. See the module doc for the exact
/// vanishing-set assignment (this is the highest-risk piece of the rewrite —
/// see `test_constraint_channels_vanish_on_correct_sets`).
#[allow(clippy::too_many_arguments)]
fn build_segment_numerator(
    coeffs_m: &[Fp],
    coeffs_a: &[Fp],
    coeffs_p: &[Fp],
    coeffs_s: &[Fp],
    delta_sigma_coeffs: &[Fp],
    alpha: &Fp,
    eta: &Fp,
    z_e: &[Fp],
    z_w_minus_s: &[Fp],
    z_w_minus_e: &[Fp],
    modulus: &BigUint,
) -> Vec<Fp> {
    let m_shift = poly_shift(coeffs_m, eta, modulus);
    let p_shift = poly_shift(coeffs_p, eta, modulus);
    let s_shift = poly_shift(coeffs_s, eta, modulus);
    let a_shift = poly_shift(coeffs_a, eta, modulus);

    let h1 = poly_sub(&p_shift, &poly_mul(coeffs_p, &m_shift, modulus), modulus);
    let h2 = poly_sub(&poly_sub(&s_shift, coeffs_s, modulus), &a_shift, modulus);
    let h3 = poly_sub(coeffs_p, coeffs_m, modulus);
    let h4 = poly_sub(coeffs_s, coeffs_a, modulus);
    let h5 = poly_sub(&poly_sub(coeffs_p, coeffs_s, modulus), delta_sigma_coeffs, modulus);

    let alpha_sq = alpha * alpha;
    let alpha_3 = &alpha_sq * alpha;
    let alpha_4 = &alpha_3 * alpha;

    let h1_plus_alpha_h2 = poly_add(&h1, &poly_scale(&h2, alpha), modulus);
    let term1 = poly_mul(z_e, &h1_plus_alpha_h2, modulus);

    let inner2 = poly_add(&poly_scale(&h3, &alpha_sq), &poly_scale(&h4, &alpha_3), modulus);
    let term2 = poly_mul(z_w_minus_s, &inner2, modulus);

    let term3 = poly_scale(&poly_mul(z_w_minus_e, &h5, modulus), &alpha_4);

    poly_add(&poly_add(&term1, &term2, modulus), &term3, modulus)
}

/// Verifier-side pointwise recomputation of `N_sigma(x)` from opened row
/// values at `x` and `eta*x` plus `Delta_sigma(x)`.
#[allow(clippy::too_many_arguments)]
fn recompute_segment_numerator_at(
    m_x: &Fp,
    a_x: &Fp,
    p_x: &Fp,
    s_x: &Fp,
    m_ex: &Fp,
    a_ex: &Fp,
    p_ex: &Fp,
    s_ex: &Fp,
    delta_sigma_x: &Fp,
    alpha: &Fp,
    z_e_x: &Fp,
    z_w_minus_s_x: &Fp,
    z_w_minus_e_x: &Fp,
) -> Fp {
    let h1 = p_ex - &(p_x * m_ex);
    let h2 = &(s_ex - s_x) - a_ex;
    let h3 = p_x - m_x;
    let h4 = s_x - a_x;
    let h5 = &(p_x - s_x) - delta_sigma_x;

    let alpha_sq = alpha * alpha;
    let alpha_3 = &alpha_sq * alpha;
    let alpha_4 = &alpha_3 * alpha;

    let term1 = z_e_x * &(&h1 + &(alpha * &h2));
    let term2 = z_w_minus_s_x * &(&(&alpha_sq * &h3) + &(&alpha_3 * &h4));
    let term3 = &(&alpha_4 * z_w_minus_e_x) * &h5;
    &(&term1 + &term2) + &term3
}

// ============================================================================
// Proof / verifier-material types.
// ============================================================================

/// A Pi_Lig proof over `B >= 1` instances (the "balanced layout" fixed
/// protocol). Broadcast material only — per-party binding polynomials are
/// returned separately by [`ligero_prove`] and must be sent P2P, never
/// broadcast (see module doc).
#[derive(Clone, Debug)]
pub struct LigeroProof {
    pub rt_w: [u8; 32],
    pub rt_c: [u8; 32],
    pub rt_u: [u8; 32],
    /// Full `n_c`-length consistency codeword, sent in full so its own
    /// proximity (`is_valid_low_degree`) covers every committed row.
    pub u: Vec<Fp>,
    /// `h_j = H(q_j)` for every party `j`, committed BEFORE `query_positions`
    /// are sampled (the Z6 fix).
    pub h: Vec<[u8; 32]>,
    pub query_positions: Vec<usize>,
    /// `[query][row]` opened committed-row values at `x`.
    pub opened_columns_x: Vec<Vec<Fp>>,
    /// `[query][row]` opened committed-row values at `eta*x`.
    pub opened_columns_etax: Vec<Vec<Fp>>,
    pub merkle_paths_x: Vec<Vec<[u8; 32]>>,
    pub merkle_paths_etax: Vec<Vec<[u8; 32]>>,
    pub opened_composition_x: Vec<Fp>,
    pub composition_merkle_paths: Vec<Vec<[u8; 32]>>,
    pub num_instances: usize,
    pub num_segments: usize,
}

impl LigeroProof {
    /// Wire-byte size of the BROADCAST material only; a party's own `q_j` is
    /// accounted separately via [`LigeroVerifierMaterial::wire_bytes`] since
    /// it is sent P2P.
    pub fn wire_bytes(&self, feb: usize) -> usize {
        let mut bytes = self.rt_w.len() + self.rt_c.len() + self.rt_u.len();
        bytes += self.u.len() * feb;
        bytes += self.h.len() * 32;
        for col in &self.opened_columns_x {
            bytes += col.len() * feb;
        }
        for col in &self.opened_columns_etax {
            bytes += col.len() * feb;
        }
        bytes += self.opened_composition_x.len() * feb;
        for path in &self.merkle_paths_x {
            bytes += path.len() * 32;
        }
        for path in &self.merkle_paths_etax {
            bytes += path.len() * 32;
        }
        for path in &self.composition_merkle_paths {
            bytes += path.len() * 32;
        }
        bytes
    }
}

/// One party's private binding-polynomial material (P7's P2P send).
#[derive(Clone, Debug)]
pub struct LigeroVerifierMaterial {
    pub party_id: usize,
    pub q_j_coeffs: Vec<Fp>,
}

impl LigeroVerifierMaterial {
    pub fn wire_bytes(&self, feb: usize) -> usize {
        self.q_j_coeffs.len() * feb
    }
}

fn total_row_count(layout: &LigeroLayout, n_parties: usize) -> usize {
    4 * layout.l + n_parties + 1
}

fn mask_row_index(layout: &LigeroLayout, party: usize) -> usize {
    4 * layout.l + party
}

// ============================================================================
// Fiat-Shamir transcript replay (shared verbatim by prove/verify/verify_as_party).
// ============================================================================

fn derive_alpha(rt_w: &[u8; 32], deltas: &[Fp], modulus: &BigUint) -> (Transcript, Fp) {
    let mut transcript = Transcript::new(b"Ligero");
    transcript.append_commitment(rt_w);
    for d in deltas {
        transcript.append_field_element(d);
    }
    let alpha = transcript.challenge(modulus);
    (transcript, alpha)
}

fn derive_post_rtc(
    transcript: &mut Transcript,
    rt_c: &[u8; 32],
    total_rows: usize,
    n_parties: usize,
    layout: &LigeroLayout,
    modulus: &BigUint,
) -> (Vec<Fp>, Vec<Fp>, Vec<Fp>) {
    transcript.append_commitment(rt_c);
    transcript.append_bytes(b"v");
    let v = transcript.challenge_vec(total_rows + 1, modulus);
    transcript.append_bytes(b"gamma");
    let gamma_flat = transcript.challenge_vec(n_parties * layout.l * 2, modulus);
    transcript.append_bytes(b"beta");
    let beta_flat = transcript.challenge_vec(n_parties * layout.l_len, modulus);
    (v, gamma_flat, beta_flat)
}

fn derive_query_positions(
    transcript: &mut Transcript,
    rt_u: &[u8; 32],
    h_all: &[[u8; 32]],
    params: &LigeroParams,
) -> Vec<usize> {
    transcript.append_commitment(rt_u);
    for h in h_all {
        transcript.append_commitment(h);
    }
    let s = params.n_c / params.k_prime;
    let mut query_positions: Vec<usize> = Vec::with_capacity(params.num_queries);
    for qi in 0..params.num_queries {
        loop {
            let idx = transcript.challenge_index(params.n_c);
            transcript.append_bytes(&(qi as u64).to_be_bytes());
            if idx % s == 0 {
                continue;
            }
            if query_positions.contains(&idx) {
                continue;
            }
            query_positions.push(idx);
            break;
        }
    }
    query_positions
}

// ============================================================================
// Prover.
// ============================================================================

struct SegmentPolys {
    coeffs_m: Vec<Fp>,
    coeffs_a: Vec<Fp>,
    coeffs_p: Vec<Fp>,
    coeffs_s: Vec<Fp>,
    cw_m: Vec<Fp>,
    cw_a: Vec<Fp>,
    cw_p: Vec<Fp>,
    cw_s: Vec<Fp>,
    delta_sigma_coeffs: Vec<Fp>,
}

fn build_segments(
    instances: &[LigeroInstance],
    layout: &LigeroLayout,
    params: &LigeroParams,
    eta_pows: &[Fp],
    modulus: &BigUint,
    rng: &mut impl rand::Rng,
) -> Vec<SegmentPolys> {
    let n = layout.n_wires;
    let c = layout.c;
    let l_len = layout.l_len;
    let k_prime = params.k_prime;
    let b_total = layout.b_total;

    (0..layout.l)
        .map(|sigma| {
            let mut vals_m = vec![Fp::zero(modulus); k_prime];
            let mut vals_a = vec![Fp::zero(modulus); k_prime];
            let mut vals_p = vec![Fp::zero(modulus); k_prime];
            let mut vals_s = vec![Fp::zero(modulus); k_prime];
            let mut delta_points: Vec<(Fp, Fp)> = Vec::with_capacity(c);

            for nu in 0..c {
                let g = sigma * c + nu;
                let start = layout.instance_start(nu);
                let (m_row, a_row, delta_val) = if g < b_total {
                    (
                        instances[g].m_values.clone(),
                        instances[g].a_values.clone(),
                        instances[g].delta.clone(),
                    )
                } else {
                    (vec![Fp::one(modulus); n], vec![Fp::zero(modulus); n], Fp::one(modulus))
                };
                let mut p_prev = Fp::zero(modulus);
                let mut s_prev = Fp::zero(modulus);
                for k in 0..n {
                    let idx = start + k;
                    vals_m[idx] = m_row[k].clone();
                    vals_a[idx] = a_row[k].clone();
                    if k == 0 {
                        vals_p[idx] = m_row[0].clone();
                        vals_s[idx] = a_row[0].clone();
                    } else {
                        vals_p[idx] = &p_prev * &m_row[k];
                        vals_s[idx] = &s_prev + &a_row[k];
                    }
                    p_prev = vals_p[idx].clone();
                    s_prev = vals_s[idx].clone();
                }
                delta_points.push((eta_pows[layout.instance_end(nu)].clone(), delta_val));
            }
            for idx in l_len..k_prime {
                vals_m[idx] = Fp::random(modulus, rng);
                vals_a[idx] = Fp::random(modulus, rng);
                vals_p[idx] = Fp::random(modulus, rng);
                vals_s[idx] = Fp::random(modulus, rng);
            }

            let coeffs_m = interpolate_over_full_domain(&vals_m, &params.eta);
            let coeffs_a = interpolate_over_full_domain(&vals_a, &params.eta);
            let coeffs_p = interpolate_over_full_domain(&vals_p, &params.eta);
            let coeffs_s = interpolate_over_full_domain(&vals_s, &params.eta);
            let cw_m = encode_domain(&coeffs_m, params.n_c, &params.omega, modulus);
            let cw_a = encode_domain(&coeffs_a, params.n_c, &params.omega, modulus);
            let cw_p = encode_domain(&coeffs_p, params.n_c, &params.omega, modulus);
            let cw_s = encode_domain(&coeffs_s, params.n_c, &params.omega, modulus);
            let delta_sigma_coeffs = interpolate_at_points(&delta_points, modulus);

            SegmentPolys { coeffs_m, coeffs_a, coeffs_p, coeffs_s, cw_m, cw_a, cw_p, cw_s, delta_sigma_coeffs }
        })
        .collect()
}

/// Generate a Pi_Lig proof over `instances.len() = B` instances for
/// `n_parties` verifiers, per `params`/`layout` (from [`try_new_layout`]).
/// Returns the broadcast proof plus, for every party, its private `q_j`
/// material (to be sent P2P — never broadcast).
pub fn ligero_prove(
    instances: &[LigeroInstance],
    n_parties: usize,
    params: &LigeroParams,
    layout: &LigeroLayout,
    modulus: &BigUint,
) -> (LigeroProof, Vec<LigeroVerifierMaterial>) {
    let b_total = instances.len();
    assert_eq!(b_total, layout.b_total, "instance count must match layout");
    assert!(b_total >= 1, "ligero_prove requires at least one instance");
    for inst in instances {
        assert_eq!(inst.verifier_positions.len(), n_parties, "one position-set per party");
    }

    let c = layout.c;
    let l = layout.l;
    let l_len = layout.l_len;
    let k_prime = params.k_prime;
    let k_star = params.k_star;
    let n_c = params.n_c;
    let omega = &params.omega;
    let eta = &params.eta;
    let s = n_c / k_prime;
    let total_rows = total_row_count(layout, n_parties);

    let mut rng = rand::thread_rng();
    let eta_pows = eta_powers(eta, k_prime, modulus);
    let deltas: Vec<Fp> = instances.iter().map(|i| i.delta.clone()).collect();

    let start_positions: Vec<usize> = (0..c).map(|nu| layout.instance_start(nu)).collect();
    let end_positions: Vec<usize> = (0..c).map(|nu| layout.instance_end(nu)).collect();
    let z_s = vanishing_from_indices(&eta_pows, &start_positions, modulus);
    let z_e = vanishing_from_indices(&eta_pows, &end_positions, modulus);
    let z_w = vanishing_prefix(&eta_pows, l_len, modulus);
    let z_w_minus_s = poly_divide_exact(&z_w, &z_s, modulus);
    let z_w_minus_e = poly_divide_exact(&z_w, &z_e, modulus);

    let segments = build_segments(instances, layout, params, &eta_pows, modulus, &mut rng);

    // Mask rows (one per party) + shared blinding row.
    let mask_deg = k_star.saturating_sub(l_len);
    let mask_g: Vec<Vec<Fp>> = (0..n_parties)
        .map(|_| (0..mask_deg).map(|_| Fp::random(modulus, &mut rng)).collect())
        .collect();
    let mask_coeffs: Vec<Vec<Fp>> = mask_g.iter().map(|g| poly_mul(&z_w, g, modulus)).collect();
    let blinding_coeffs: Vec<Fp> = (0..k_star).map(|_| Fp::random(modulus, &mut rng)).collect();
    let mask_cw: Vec<Vec<Fp>> = mask_coeffs.iter().map(|c| encode_domain(c, n_c, omega, modulus)).collect();
    let blinding_cw = encode_domain(&blinding_coeffs, n_c, omega, modulus);

    let mut all_codewords: Vec<&Vec<Fp>> = Vec::with_capacity(total_rows);
    for seg in &segments {
        all_codewords.push(&seg.cw_m);
        all_codewords.push(&seg.cw_a);
        all_codewords.push(&seg.cw_p);
        all_codewords.push(&seg.cw_s);
    }
    for mc in &mask_cw {
        all_codewords.push(mc);
    }
    all_codewords.push(&blinding_cw);
    debug_assert_eq!(all_codewords.len(), total_rows);

    let columns: Vec<Vec<Fp>> = (0..n_c)
        .map(|j| all_codewords.iter().map(|r| r[j].clone()).collect())
        .collect();
    let column_leaves: Vec<[u8; 32]> = columns.iter().map(|col| hash_field_elements(col)).collect();
    let merkle_w = MerkleTree::new(column_leaves);
    let rt_w = merkle_w.root();

    let (mut transcript, alpha) = derive_alpha(&rt_w, &deltas, modulus);
    let alpha_5 = alpha.pow(&BigUint::from(5u32));

    let mut c_batch: Vec<Fp> = vec![Fp::zero(modulus); l_len.max(1)];
    let mut alpha_pow = Fp::one(modulus);
    for seg in &segments {
        let n_sigma = build_segment_numerator(
            &seg.coeffs_m, &seg.coeffs_a, &seg.coeffs_p, &seg.coeffs_s,
            &seg.delta_sigma_coeffs, &alpha, eta, &z_e, &z_w_minus_s, &z_w_minus_e, modulus,
        );
        let c_sigma = poly_divide_exact(&n_sigma, &z_w, modulus);
        debug_assert!(
            c_sigma.iter().skip(k_star).all(|c| c.is_zero()) || c_sigma.len() <= k_star,
            "C_sigma degree must stay below K*"
        );
        for (j, cv) in c_sigma.iter().enumerate() {
            if j >= c_batch.len() {
                c_batch.resize(j + 1, Fp::zero(modulus));
            }
            c_batch[j] = &c_batch[j] + &(&alpha_pow * cv);
        }
        alpha_pow = &alpha_pow * &alpha_5;
    }
    let composition_codeword = encode_domain(&c_batch, n_c, omega, modulus);
    let composition_leaves: Vec<[u8; 32]> = composition_codeword
        .iter()
        .map(|v| hash_field_elements(std::slice::from_ref(v)))
        .collect();
    let merkle_c = MerkleTree::new(composition_leaves);
    let rt_c = merkle_c.root();

    let (v, gamma_flat, beta_flat) = derive_post_rtc(&mut transcript, &rt_c, total_rows, n_parties, layout, modulus);

    let mut u = vec![Fp::zero(modulus); n_c];
    for (j, u_j) in u.iter_mut().enumerate() {
        let mut acc = Fp::zero(modulus);
        for (r, cw) in all_codewords.iter().enumerate() {
            acc = &acc + &(&v[r] * &cw[j]);
        }
        acc = &acc + &(&v[total_rows] * &composition_codeword[j]);
        *u_j = acc;
    }
    let rt_u = hash_field_elements(&u);

    let mut q_all: Vec<Vec<Fp>> = Vec::with_capacity(n_parties);
    let mut h_all: Vec<[u8; 32]> = Vec::with_capacity(n_parties);
    for j in 0..n_parties {
        let mut q_j = mask_coeffs[j].clone();
        for (sigma, seg) in segments.iter().enumerate() {
            let gm = &gamma_flat[(j * l + sigma) * 2];
            let ga = &gamma_flat[(j * l + sigma) * 2 + 1];
            let mut lambda_vals = vec![Fp::zero(modulus); k_prime];
            for nu in 0..c {
                let g = sigma * c + nu;
                if g >= b_total {
                    continue;
                }
                let start = layout.instance_start(nu);
                for &pos in &instances[g].verifier_positions[j] {
                    let local_i = start + pos;
                    lambda_vals[local_i] = beta_flat[j * l_len + local_i].clone();
                }
            }
            let lambda_coeffs = interpolate_over_full_domain(&lambda_vals, eta);
            let combo = poly_add(&poly_scale(&seg.coeffs_m, gm), &poly_scale(&seg.coeffs_a, ga), modulus);
            let term = poly_mul(&lambda_coeffs, &combo, modulus);
            q_j = poly_add(&q_j, &term, modulus);
        }
        h_all.push(hash_field_elements(&q_j));
        q_all.push(q_j);
    }

    let query_positions = derive_query_positions(&mut transcript, &rt_u, &h_all, params);

    let opened_columns_x: Vec<Vec<Fp>> = query_positions.iter().map(|&j| columns[j].clone()).collect();
    let opened_columns_etax: Vec<Vec<Fp>> =
        query_positions.iter().map(|&j| columns[(j + s) % n_c].clone()).collect();
    let merkle_paths_x: Vec<Vec<[u8; 32]>> =
        query_positions.iter().map(|&j| merkle_w.authentication_path(j)).collect();
    let merkle_paths_etax: Vec<Vec<[u8; 32]>> = query_positions
        .iter()
        .map(|&j| merkle_w.authentication_path((j + s) % n_c))
        .collect();
    let opened_composition_x: Vec<Fp> =
        query_positions.iter().map(|&j| composition_codeword[j].clone()).collect();
    let composition_merkle_paths: Vec<Vec<[u8; 32]>> =
        query_positions.iter().map(|&j| merkle_c.authentication_path(j)).collect();

    let proof = LigeroProof {
        rt_w,
        rt_c,
        rt_u,
        u,
        h: h_all,
        query_positions,
        opened_columns_x,
        opened_columns_etax,
        merkle_paths_x,
        merkle_paths_etax,
        opened_composition_x,
        composition_merkle_paths,
        num_instances: b_total,
        num_segments: l,
    };
    let materials: Vec<LigeroVerifierMaterial> = q_all
        .into_iter()
        .enumerate()
        .map(|(party_id, q_j_coeffs)| LigeroVerifierMaterial { party_id, q_j_coeffs })
        .collect();
    (proof, materials)
}

// ============================================================================
// Shared verifier (proximity / constraint / consistency).
// ============================================================================

/// Verify the shared (party-independent) checks: proximity of `u`, the
/// batched constraint identity, and interleaved consistency, at every
/// sampled point. Every party runs this identically.
pub fn ligero_verify(
    proof: &LigeroProof,
    deltas: &[Fp],
    n_parties: usize,
    params: &LigeroParams,
    layout: &LigeroLayout,
    modulus: &BigUint,
) -> bool {
    let b_total = proof.num_instances;
    if deltas.len() != b_total || b_total == 0 || proof.num_segments != layout.l {
        return false;
    }
    let l = layout.l;
    let c = layout.c;
    let l_len = layout.l_len;
    let total_rows = total_row_count(layout, n_parties);
    let n_c = params.n_c;
    let k_prime = params.k_prime;
    let k_star = params.k_star;
    let omega = &params.omega;
    let eta = &params.eta;
    let s = n_c / k_prime;
    let q = params.num_queries;

    if proof.u.len() != n_c {
        return false;
    }
    if hash_field_elements(&proof.u) != proof.rt_u {
        return false;
    }
    if proof.h.len() != n_parties {
        return false;
    }
    if proof.query_positions.len() != q
        || proof.opened_columns_x.len() != q
        || proof.opened_columns_etax.len() != q
        || proof.merkle_paths_x.len() != q
        || proof.merkle_paths_etax.len() != q
        || proof.opened_composition_x.len() != q
        || proof.composition_merkle_paths.len() != q
    {
        return false;
    }
    for i in 0..q {
        if proof.opened_columns_x[i].len() != total_rows || proof.opened_columns_etax[i].len() != total_rows {
            return false;
        }
    }

    for i in 0..q {
        let j = proof.query_positions[i];
        if j >= n_c {
            return false;
        }
        let jx = (j + s) % n_c;
        let leaf_x = hash_field_elements(&proof.opened_columns_x[i]);
        if !MerkleTree::verify_path(&proof.rt_w, &leaf_x, j, n_c, &proof.merkle_paths_x[i]) {
            return false;
        }
        let leaf_ex = hash_field_elements(&proof.opened_columns_etax[i]);
        if !MerkleTree::verify_path(&proof.rt_w, &leaf_ex, jx, n_c, &proof.merkle_paths_etax[i]) {
            return false;
        }
        let leaf_c = hash_field_elements(std::slice::from_ref(&proof.opened_composition_x[i]));
        if !MerkleTree::verify_path(&proof.rt_c, &leaf_c, j, n_c, &proof.composition_merkle_paths[i]) {
            return false;
        }
    }

    let (mut transcript, alpha) = derive_alpha(&proof.rt_w, deltas, modulus);
    let (v, _gamma_flat, _beta_flat) =
        derive_post_rtc(&mut transcript, &proof.rt_c, total_rows, n_parties, layout, modulus);
    let expected_positions = derive_query_positions(&mut transcript, &proof.rt_u, &proof.h, params);
    if expected_positions != proof.query_positions {
        return false;
    }

    if !is_valid_low_degree(&proof.u, k_star, omega) {
        return false;
    }

    let eta_pows = eta_powers(eta, k_prime, modulus);
    let start_positions: Vec<usize> = (0..c).map(|nu| layout.instance_start(nu)).collect();
    let end_positions: Vec<usize> = (0..c).map(|nu| layout.instance_end(nu)).collect();
    let alpha_5 = alpha.pow(&BigUint::from(5u32));

    let delta_sigma_coeffs: Vec<Vec<Fp>> = (0..l)
        .map(|sigma| {
            let points: Vec<(Fp, Fp)> = (0..c)
                .map(|nu| {
                    let g = sigma * c + nu;
                    let d = if g < b_total { deltas[g].clone() } else { Fp::one(modulus) };
                    (eta_pows[layout.instance_end(nu)].clone(), d)
                })
                .collect();
            interpolate_at_points(&points, modulus)
        })
        .collect();

    for i in 0..q {
        let j = proof.query_positions[i];
        let x = domain_point(omega, j);
        let col_x = &proof.opened_columns_x[i];
        let col_ex = &proof.opened_columns_etax[i];
        let c_x = &proof.opened_composition_x[i];

        let z_e_x = eval_vanishing_from_indices(&x, &eta_pows, &end_positions, modulus);
        let z_s_x = eval_vanishing_from_indices(&x, &eta_pows, &start_positions, modulus);
        let z_w_x = eval_vanishing_prefix(&x, &eta_pows, l_len, modulus);
        let z_w_minus_s_x = &z_w_x * &z_s_x.inv().expect("Z_S(x) nonzero for x outside H");
        let z_w_minus_e_x = &z_w_x * &z_e_x.inv().expect("Z_E(x) nonzero for x outside H");

        let mut n_batch = Fp::zero(modulus);
        let mut alpha_pow = Fp::one(modulus);
        for sigma in 0..l {
            let base = 4 * sigma;
            let delta_sigma_x = poly_eval(&delta_sigma_coeffs[sigma], &x, modulus);
            let n_sigma_x = recompute_segment_numerator_at(
                &col_x[base], &col_x[base + 1], &col_x[base + 2], &col_x[base + 3],
                &col_ex[base], &col_ex[base + 1], &col_ex[base + 2], &col_ex[base + 3],
                &delta_sigma_x, &alpha, &z_e_x, &z_w_minus_s_x, &z_w_minus_e_x,
            );
            n_batch = &n_batch + &(&alpha_pow * &n_sigma_x);
            alpha_pow = &alpha_pow * &alpha_5;
        }
        let lhs = &z_w_x * c_x;
        if lhs != n_batch {
            return false;
        }

        let mut u_expected = Fp::zero(modulus);
        for (r, vr) in v.iter().enumerate().take(total_rows) {
            u_expected = &u_expected + &(vr * &col_x[r]);
        }
        u_expected = &u_expected + &(&v[total_rows] * c_x);
        if u_expected != proof.u[j] {
            return false;
        }
    }

    true
}

/// Verify party `party_id`'s own binding-polynomial checks (V1/V3/V4 — the
/// Z7 fix): its `q_j` digest matches, `q_j` is consistent with the shared
/// opened columns at every sampled point (V3), and `q_j` matches this
/// party's own locally-known share values at every position it holds (V4).
///
/// `local_shares[g]` maps wire position `k` (within instance `g`'s `N`
/// wires) to `(m_k, a_k)` for every position `party_id` independently holds
/// in instance `g`; empty if it holds none. Must have length `>= B`.
pub fn ligero_verify_as_party(
    party_id: usize,
    proof: &LigeroProof,
    material: &LigeroVerifierMaterial,
    local_shares: &[BTreeMap<usize, (Fp, Fp)>],
    deltas: &[Fp],
    n_parties: usize,
    params: &LigeroParams,
    layout: &LigeroLayout,
    modulus: &BigUint,
) -> bool {
    if material.party_id != party_id || party_id >= n_parties || party_id >= proof.h.len() {
        return false;
    }
    let k_star = params.k_star;
    if material.q_j_coeffs.len() > k_star {
        return false;
    }
    if hash_field_elements(&material.q_j_coeffs) != proof.h[party_id] {
        return false;
    }

    let b_total = proof.num_instances;
    if deltas.len() != b_total || local_shares.len() < b_total {
        return false;
    }
    let l = layout.l;
    let c = layout.c;
    let total_rows = total_row_count(layout, n_parties);
    let q = params.num_queries;
    if proof.query_positions.len() != q || proof.opened_columns_x.len() != q {
        return false;
    }
    for i in 0..q {
        if proof.opened_columns_x[i].len() != total_rows {
            return false;
        }
    }

    let (mut transcript, _alpha) = derive_alpha(&proof.rt_w, deltas, modulus);
    let (_v, gamma_flat, beta_flat) =
        derive_post_rtc(&mut transcript, &proof.rt_c, total_rows, n_parties, layout, modulus);
    let expected_positions = derive_query_positions(&mut transcript, &proof.rt_u, &proof.h, params);
    if expected_positions != proof.query_positions {
        return false;
    }

    let eta = &params.eta;
    let eta_pows = eta_powers(eta, params.k_prime, modulus);
    let omega = &params.omega;
    let mask_idx = mask_row_index(layout, party_id);

    // Lambda_{party_id,sigma} coefficients, built once per segment (reused
    // across every sampled point in V3) from this party's own held positions
    // — the same public `beta` table + held-position structure the prover used.
    let lambda_coeffs_per_segment: Vec<Vec<Fp>> = (0..l)
        .map(|sigma| {
            let mut lambda_vals = vec![Fp::zero(modulus); params.k_prime];
            for nu in 0..c {
                let g = sigma * c + nu;
                if g >= b_total {
                    continue;
                }
                if let Some(map) = local_shares.get(g) {
                    let start = layout.instance_start(nu);
                    for &pos in map.keys() {
                        let local_i = start + pos;
                        lambda_vals[local_i] = beta_flat[party_id * layout.l_len + local_i].clone();
                    }
                }
            }
            interpolate_over_full_domain(&lambda_vals, eta)
        })
        .collect();

    // V3: q_j(x) at every sampled point, using the shared opened columns.
    for (i, &j) in proof.query_positions.iter().enumerate() {
        let x = domain_point(omega, j);
        let q_j_x = poly_eval(&material.q_j_coeffs, &x, modulus);
        let z_j_x = proof.opened_columns_x[i][mask_idx].clone();
        let mut rhs = z_j_x;
        for sigma in 0..l {
            let base = 4 * sigma;
            let m_x = &proof.opened_columns_x[i][base];
            let a_x = &proof.opened_columns_x[i][base + 1];
            let gm = &gamma_flat[(party_id * l + sigma) * 2];
            let ga = &gamma_flat[(party_id * l + sigma) * 2 + 1];
            let lambda_x = poly_eval(&lambda_coeffs_per_segment[sigma], &x, modulus);
            rhs = &rhs + &(&lambda_x * &(&(gm * m_x) + &(ga * a_x)));
        }
        if q_j_x != rhs {
            return false;
        }
    }

    // V4: q_j at every LOCAL position this party actually holds, against its
    // own locally-known share values — the actual Z7 fix (input binding).
    // Every segment shares the same domain H, so a single local index `i`
    // can coincide with a held position in MULTIPLE segments (e.g. the party
    // holds wire `pos` of the instance sitting at that local slot in more
    // than one segment); q_j(eta^i) is the SUM of every such segment's
    // contribution, not just one, per the spec: "the sum, over the segments
    // sigma in which S_j holds position i, of gamma_{j,sigma,M}*M + ...".
    let n_wires = layout.n_wires;
    for local_i in 0..layout.l_len {
        let nu = local_i / n_wires;
        let pos = local_i % n_wires;
        let mut sum = Fp::zero(modulus);
        let mut held_any = false;
        for sigma in 0..l {
            let g = sigma * c + nu;
            if g >= b_total {
                continue;
            }
            if let Some((m_val, a_val)) = local_shares[g].get(&pos) {
                held_any = true;
                let gm = &gamma_flat[(party_id * l + sigma) * 2];
                let ga = &gamma_flat[(party_id * l + sigma) * 2 + 1];
                sum = &sum + &(&(gm * m_val) + &(ga * a_val));
            }
        }
        if !held_any {
            continue;
        }
        let x = eta_pows[local_i].clone();
        let q_j_x = poly_eval(&material.q_j_coeffs, &x, modulus);
        let beta_val = &beta_flat[party_id * layout.l_len + local_i];
        let expected = beta_val * &sum;
        if q_j_x != expected {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    fn test_modulus() -> BigUint {
        BigUint::from(65537u32)
    }

    fn fp(v: u32, p: &BigUint) -> Fp {
        Fp::new(BigUint::from(v), p)
    }

    /// Small, deterministic (not soundness-searched) layout/params for fast
    /// unit tests: n_wires N, batch size b_total, c instances/segment.
    fn small_setup(n_wires: usize, b_total: usize, c: usize, modulus: &BigUint) -> (LigeroParams, LigeroLayout) {
        let l_len = c * n_wires;
        let k_prime = (l_len + 8).next_power_of_two().max(8);
        let k_star = 2 * k_prime - 1;
        let n_c = (4 * k_prime).next_power_of_two().max(k_star + 1).next_power_of_two();
        let omega = find_primitive_root_of_unity(n_c, modulus).expect("root of unity must exist");
        let s = n_c / k_prime;
        let eta = omega.pow(&BigUint::from(s as u64));
        let l = b_total.div_ceil(c);
        let params = LigeroParams { n_c, k_prime, k_star, num_queries: 2, omega, eta, lambda: 8 };
        let layout = LigeroLayout { n_wires, c, l, b_total, l_len };
        (params, layout)
    }

    fn make_instance(
        m_vals: &[u32],
        a_vals: &[u32],
        n_parties: usize,
        holder_of: impl Fn(usize) -> usize,
        p: &BigUint,
    ) -> LigeroInstance {
        let m: Vec<Fp> = m_vals.iter().map(|v| fp(*v, p)).collect();
        let a: Vec<Fp> = a_vals.iter().map(|v| fp(*v, p)).collect();
        let delta = LigeroInstance::compute_delta(&m, &a, p);
        let mut verifier_positions = vec![Vec::new(); n_parties];
        for pos in 0..m.len() {
            verifier_positions[holder_of(pos)].push(pos);
        }
        LigeroInstance::new(m, a, delta, verifier_positions)
    }

    #[test]
    fn test_interpolate_over_full_domain_roundtrip() {
        let p = test_modulus();
        let eta = find_primitive_root_of_unity(8, &p).unwrap();
        let coeffs: Vec<Fp> = (1..=8u32).map(|v| fp(v, &p)).collect();
        let eta_pows = eta_powers(&eta, 8, &p);
        let values: Vec<Fp> = eta_pows.iter().map(|x| poly_eval(&coeffs, x, &p)).collect();
        let recovered = interpolate_over_full_domain(&values, &eta);
        assert_eq!(recovered, coeffs);
    }

    #[test]
    fn test_interpolate_at_points_roundtrip() {
        let p = test_modulus();
        let coeffs = vec![fp(5, &p), fp(3, &p), fp(2, &p)];
        let xs = [fp(10, &p), fp(20, &p), fp(30, &p)];
        let points: Vec<(Fp, Fp)> = xs.iter().map(|x| (x.clone(), poly_eval(&coeffs, x, &p))).collect();
        let recovered = interpolate_at_points(&points, &p);
        assert_eq!(recovered, coeffs);
    }

    #[test]
    fn test_vanishing_polynomials_correctness() {
        let p = test_modulus();
        let eta = find_primitive_root_of_unity(16, &p).unwrap();
        let eta_pows = eta_powers(&eta, 16, &p);
        let n_wires = 3;
        let c = 2;
        let l_len = n_wires * c;
        let starts: Vec<usize> = (0..c).map(|nu| nu * n_wires).collect();
        let ends: Vec<usize> = (0..c).map(|nu| nu * n_wires + n_wires - 1).collect();

        let z_w = vanishing_prefix(&eta_pows, l_len, &p);
        for i in 0..l_len {
            assert!(poly_eval(&z_w, &eta_pows[i], &p).is_zero(), "Z_W must vanish on W at {i}");
        }
        assert!(!poly_eval(&z_w, &eta_pows[l_len], &p).is_zero(), "Z_W must not vanish outside W");

        let z_s = vanishing_from_indices(&eta_pows, &starts, &p);
        for &i in &starts {
            assert!(poly_eval(&z_s, &eta_pows[i], &p).is_zero());
        }
        for &i in &ends {
            if !starts.contains(&i) {
                assert!(!poly_eval(&z_s, &eta_pows[i], &p).is_zero());
            }
        }

        let z_e = vanishing_from_indices(&eta_pows, &ends, &p);
        for &i in &ends {
            assert!(poly_eval(&z_e, &eta_pows[i], &p).is_zero());
        }
        for &i in &starts {
            if !ends.contains(&i) {
                assert!(!poly_eval(&z_e, &eta_pows[i], &p).is_zero());
            }
        }
    }

    #[test]
    fn test_layout_search_satisfies_constraints() {
        let p = test_modulus();
        for (n_wires, b_total) in [(3usize, 2usize), (6, 5), (4, 20)] {
            let (params, layout) = try_new_layout(n_wires, b_total, 8, &p)
                .unwrap_or_else(|| panic!("expected feasible layout for N={n_wires} B={b_total}"));
            assert!(params.k_prime >= layout.l_len + 2 * params.num_queries);
            assert!(params.n_c >= 4 * params.k_prime);
            assert_eq!(params.n_c % params.k_prime, 0);
            assert_eq!(params.k_star, 2 * params.k_prime - 1);
            assert!(layout.c >= 1 && layout.l >= 1);
            assert!(layout.l * layout.c >= b_total);
            assert_eq!(layout.l_len, layout.c * layout.n_wires);
        }
    }

    #[test]
    fn test_constraint_channels_vanish_on_correct_sets() {
        // c = 2 instances of N = 3 wires each, built by hand so h1/h3/h5 are
        // demonstrably nonzero where they must NOT be required to vanish.
        let p = test_modulus();
        let n_wires = 3;
        let c = 2;
        let l_len = n_wires * c;
        let k_prime = 16usize;
        let eta = find_primitive_root_of_unity(k_prime, &p).unwrap();
        let eta_pows = eta_powers(&eta, k_prime, &p);

        let m0 = [3u32, 5, 7];
        let a0 = [10u32, 20, 30];
        let m1 = [2u32, 4, 6];
        let a1 = [1u32, 2, 3];

        let mut vals_m = vec![Fp::zero(&p); k_prime];
        let mut vals_a = vec![Fp::zero(&p); k_prime];
        let mut vals_p = vec![Fp::zero(&p); k_prime];
        let mut vals_s = vec![Fp::zero(&p); k_prime];
        let mut delta_points = Vec::new();
        for (nu, (mv, av)) in [(m0, a0), (m1, a1)].iter().enumerate() {
            let start = nu * n_wires;
            let mut p_prev = Fp::zero(&p);
            let mut s_prev = Fp::zero(&p);
            for k in 0..n_wires {
                let idx = start + k;
                vals_m[idx] = fp(mv[k], &p);
                vals_a[idx] = fp(av[k], &p);
                if k == 0 {
                    vals_p[idx] = fp(mv[0], &p);
                    vals_s[idx] = fp(av[0], &p);
                } else {
                    vals_p[idx] = &p_prev * &fp(mv[k], &p);
                    vals_s[idx] = &s_prev + &fp(av[k], &p);
                }
                p_prev = vals_p[idx].clone();
                s_prev = vals_s[idx].clone();
            }
            let delta = LigeroInstance::compute_delta(
                &mv.iter().map(|v| fp(*v, &p)).collect::<Vec<_>>(),
                &av.iter().map(|v| fp(*v, &p)).collect::<Vec<_>>(),
                &p,
            );
            delta_points.push((eta_pows[start + n_wires - 1].clone(), delta));
        }
        for idx in l_len..k_prime {
            vals_m[idx] = fp(99, &p);
            vals_a[idx] = fp(98, &p);
            vals_p[idx] = fp(97, &p);
            vals_s[idx] = fp(96, &p);
        }

        let coeffs_m = interpolate_over_full_domain(&vals_m, &eta);
        let coeffs_a = interpolate_over_full_domain(&vals_a, &eta);
        let coeffs_p = interpolate_over_full_domain(&vals_p, &eta);
        let coeffs_s = interpolate_over_full_domain(&vals_s, &eta);
        let delta_sigma_coeffs = interpolate_at_points(&delta_points, &p);

        let m_shift = poly_shift(&coeffs_m, &eta, &p);
        let p_shift = poly_shift(&coeffs_p, &eta, &p);
        let s_shift = poly_shift(&coeffs_s, &eta, &p);
        let a_shift = poly_shift(&coeffs_a, &eta, &p);
        let h1 = poly_sub(&p_shift, &poly_mul(&coeffs_p, &m_shift, &p), &p);
        let h2 = poly_sub(&poly_sub(&s_shift, &coeffs_s, &p), &a_shift, &p);
        let h3 = poly_sub(&coeffs_p, &coeffs_m, &p);
        let h4 = poly_sub(&coeffs_s, &coeffs_a, &p);
        let h5 = poly_sub(&poly_sub(&coeffs_p, &coeffs_s, &p), &delta_sigma_coeffs, &p);

        let starts = [0usize, n_wires];
        let ends = [n_wires - 1, 2 * n_wires - 1];

        // h1/h2 must vanish everywhere in W \ E (interior + starts), and are
        // demonstrably NONZERO at at least one end (recurrence breaks there).
        for i in 0..l_len {
            if !ends.contains(&i) {
                assert!(poly_eval(&h1, &eta_pows[i], &p).is_zero(), "h1 must vanish at {i}");
                assert!(poly_eval(&h2, &eta_pows[i], &p).is_zero(), "h2 must vanish at {i}");
            }
        }
        assert!(
            !poly_eval(&h1, &eta_pows[ends[0]], &p).is_zero(),
            "h1 must be free (generically nonzero) at an end position"
        );

        // h3/h4 vanish at starts; demonstrably nonzero at an interior position.
        for &i in &starts {
            assert!(poly_eval(&h3, &eta_pows[i], &p).is_zero(), "h3 must vanish at start {i}");
            assert!(poly_eval(&h4, &eta_pows[i], &p).is_zero(), "h4 must vanish at start {i}");
        }
        assert!(
            !poly_eval(&h3, &eta_pows[1], &p).is_zero(),
            "h3 must be free (generically nonzero) at an interior position"
        );

        // h5 vanishes at ends; demonstrably nonzero at a start (delta only
        // constrains the instance's own end, not its start).
        for &i in &ends {
            assert!(poly_eval(&h5, &eta_pows[i], &p).is_zero(), "h5 must vanish at end {i}");
        }
        assert!(
            !poly_eval(&h5, &eta_pows[0], &p).is_zero(),
            "h5 must be free (generically nonzero) at a start position"
        );

        // Exact-division sanity: N_sigma built from these h's must vanish on
        // ALL of W (not just be "close"), i.e. divide Z_W with zero remainder.
        let z_s = vanishing_from_indices(&eta_pows, &starts, &p);
        let z_e = vanishing_from_indices(&eta_pows, &ends, &p);
        let z_w = vanishing_prefix(&eta_pows, l_len, &p);
        let z_w_minus_s = poly_divide_exact(&z_w, &z_s, &p);
        let z_w_minus_e = poly_divide_exact(&z_w, &z_e, &p);
        let alpha = fp(7, &p);
        let n_sigma = build_segment_numerator(
            &coeffs_m, &coeffs_a, &coeffs_p, &coeffs_s, &delta_sigma_coeffs,
            &alpha, &eta, &z_e, &z_w_minus_s, &z_w_minus_e, &p,
        );
        for i in 0..l_len {
            assert!(
                poly_eval(&n_sigma, &eta_pows[i], &p).is_zero(),
                "N_sigma must vanish on all of W at {i} for an honest witness"
            );
        }
        let c_sigma = poly_divide_exact(&n_sigma, &z_w, &p);
        let reconstructed = poly_mul(&z_w, &c_sigma, &p);
        for i in 0..l_len.min(reconstructed.len()) {
            assert_eq!(
                poly_eval(&n_sigma, &eta_pows[i], &p),
                poly_eval(&reconstructed, &eta_pows[i], &p)
            );
        }
    }

    fn local_shares_for(
        instances: &[LigeroInstance],
        party: usize,
    ) -> Vec<BTreeMap<usize, (Fp, Fp)>> {
        instances
            .iter()
            .map(|inst| {
                inst.verifier_positions[party]
                    .iter()
                    .map(|&pos| (pos, (inst.m_values[pos].clone(), inst.a_values[pos].clone())))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_ligero_prove_verify_honest_round_trip() {
        let p = test_modulus();
        let n_parties = 3;
        let inst0 = make_instance(&[3, 5, 7], &[10, 20, 30], n_parties, |pos| pos % n_parties, &p);
        let inst1 = make_instance(&[2, 4, 6], &[1, 2, 3], n_parties, |pos| (pos + 1) % n_parties, &p);
        let instances = vec![inst0, inst1];
        let (params, layout) = small_setup(3, 2, 1, &p);

        let (proof, materials) = ligero_prove(&instances, n_parties, &params, &layout, &p);
        let deltas: Vec<Fp> = instances.iter().map(|i| i.delta.clone()).collect();
        assert!(ligero_verify(&proof, &deltas, n_parties, &params, &layout, &p));

        for party in 0..n_parties {
            let shares = local_shares_for(&instances, party);
            assert!(
                ligero_verify_as_party(
                    party, &proof, &materials[party], &shares, &deltas, n_parties, &params, &layout, &p,
                ),
                "party {party} must accept an honest proof"
            );
        }
    }

    #[test]
    fn test_ligero_multi_segment_round_trip() {
        let p = test_modulus();
        let n_parties = 4;
        let n_wires = 3;
        let c = 2;
        let instances: Vec<LigeroInstance> = (0..5u32)
            .map(|i| {
                make_instance(
                    &[3 + i, 5 + i, 7 + i],
                    &[10 + i, 20 + i, 30 + i],
                    n_parties,
                    move |pos| (pos + i as usize) % n_parties,
                    &p,
                )
            })
            .collect();
        let (params, layout) = small_setup(n_wires, instances.len(), c, &p);
        assert!(layout.l > 1, "test needs multiple segments");

        let (proof, materials) = ligero_prove(&instances, n_parties, &params, &layout, &p);
        let deltas: Vec<Fp> = instances.iter().map(|i| i.delta.clone()).collect();
        assert!(ligero_verify(&proof, &deltas, n_parties, &params, &layout, &p));
        for party in 0..n_parties {
            let shares = local_shares_for(&instances, party);
            assert!(ligero_verify_as_party(
                party, &proof, &materials[party], &shares, &deltas, n_parties, &params, &layout, &p,
            ));
        }
    }

    #[test]
    fn test_ligero_rejects_wrong_delta() {
        let p = test_modulus();
        let n_parties = 3;
        let inst = make_instance(&[3, 5, 7], &[10, 20, 30], n_parties, |pos| pos % n_parties, &p);
        let instances = vec![inst];
        let (params, layout) = small_setup(3, 1, 1, &p);
        let (proof, _materials) = ligero_prove(&instances, n_parties, &params, &layout, &p);
        let wrong = vec![fp(44, &p)];
        assert!(!ligero_verify(&proof, &wrong, n_parties, &params, &layout, &p));
    }

    #[test]
    fn test_ligero_rejects_tampered_row() {
        let p = test_modulus();
        let n_parties = 3;
        let inst = make_instance(&[3, 5, 7], &[10, 20, 30], n_parties, |pos| pos % n_parties, &p);
        let instances = vec![inst];
        let (params, layout) = small_setup(3, 1, 1, &p);
        let (mut proof, _materials) = ligero_prove(&instances, n_parties, &params, &layout, &p);
        let deltas: Vec<Fp> = instances.iter().map(|i| i.delta.clone()).collect();
        // Tamper one opened row value at the first query.
        if !proof.opened_columns_x[0].is_empty() {
            proof.opened_columns_x[0][0] = &proof.opened_columns_x[0][0] + &Fp::one(&p);
        }
        assert!(!ligero_verify(&proof, &deltas, n_parties, &params, &layout, &p));
    }

    #[test]
    fn test_z6_tampering_h_changes_query_positions_and_rejects() {
        let p = test_modulus();
        let n_parties = 3;
        let inst = make_instance(&[3, 5, 7], &[10, 20, 30], n_parties, |pos| pos % n_parties, &p);
        let instances = vec![inst];
        let (params, layout) = small_setup(3, 1, 1, &p);
        let (mut proof, _materials) = ligero_prove(&instances, n_parties, &params, &layout, &p);
        let deltas: Vec<Fp> = instances.iter().map(|i| i.delta.clone()).collect();
        assert!(ligero_verify(&proof, &deltas, n_parties, &params, &layout, &p));

        let original_positions = proof.query_positions.clone();
        proof.h[0] = [0xABu8; 32];
        let total_rows = total_row_count(&layout, n_parties);
        let (mut transcript, _alpha) = derive_alpha(&proof.rt_w, &deltas, &p);
        let (_v, _g, _b) = derive_post_rtc(&mut transcript, &proof.rt_c, total_rows, n_parties, &layout, &p);
        let tampered_positions = derive_query_positions(&mut transcript, &proof.rt_u, &proof.h, &params);
        assert_ne!(
            tampered_positions, original_positions,
            "sampled query positions must depend on h_j (the Z6 fix)"
        );
        assert!(!ligero_verify(&proof, &deltas, n_parties, &params, &layout, &p));
    }

    /// Z7 flagship regression: a corrupted dealer swaps wire 0's value while
    /// keeping the (witness, delta) pair internally consistent, so the
    /// shared `ligero_verify` has no way to detect it — exactly the "shift
    /// the sharing of M^i arbitrarily" vulnerability. A party who
    /// independently derived the TRUE wire-0 value must catch it via
    /// `ligero_verify_as_party`.
    #[test]
    fn test_z7_regression_dealer_cannot_shift_witness_undetected() {
        let p = test_modulus();
        let n_parties = 3;
        let true_m = [3u32, 5, 7];
        let a = [10u32, 20, 30];
        // Tampered: wire 0 uses a different m value; delta recomputed to
        // stay internally consistent (so ligero_verify alone can't tell).
        let tampered_m = [4u32, 5, 7];

        let holder = |pos: usize| pos % n_parties; // party (pos%3) holds wire `pos`
        let tampered_inst = make_instance(&tampered_m, &a, n_parties, holder, &p);
        let instances = vec![tampered_inst];
        let (params, layout) = small_setup(3, 1, 1, &p);

        let (proof, materials) = ligero_prove(&instances, n_parties, &params, &layout, &p);
        let deltas: Vec<Fp> = instances.iter().map(|i| i.delta.clone()).collect();

        // The shared check alone accepts: nothing here contradicts itself.
        assert!(
            ligero_verify(&proof, &deltas, n_parties, &params, &layout, &p),
            "shared verify must accept a self-consistent but shifted witness"
        );

        // Party holding wire 0 independently knows the TRUE m0, not the
        // tampered one baked into the codewords.
        let holder_of_wire0 = holder(0);
        let true_m_fp: Vec<Fp> = true_m.iter().map(|v| fp(*v, &p)).collect();
        let a_fp: Vec<Fp> = a.iter().map(|v| fp(*v, &p)).collect();
        let mut true_shares = vec![BTreeMap::new(); 1];
        for pos in 0..3 {
            if holder(pos) == holder_of_wire0 {
                true_shares[0].insert(pos, (true_m_fp[pos].clone(), a_fp[pos].clone()));
            }
        }

        assert!(
            !ligero_verify_as_party(
                holder_of_wire0,
                &proof,
                &materials[holder_of_wire0],
                &true_shares,
                &deltas,
                n_parties,
                &params,
                &layout,
                &p,
            ),
            "the true share-holder must reject a proof whose committed wire-0 value disagrees with its own"
        );

        // Negative control: an HONEST proof (true_m throughout) must be
        // accepted by the same party using the same true shares.
        let honest_inst = make_instance(&true_m, &a, n_parties, holder, &p);
        let honest_instances = vec![honest_inst];
        let (honest_proof, honest_materials) =
            ligero_prove(&honest_instances, n_parties, &params, &layout, &p);
        let honest_deltas: Vec<Fp> = honest_instances.iter().map(|i| i.delta.clone()).collect();
        assert!(ligero_verify(&honest_proof, &honest_deltas, n_parties, &params, &layout, &p));
        assert!(ligero_verify_as_party(
            holder_of_wire0,
            &honest_proof,
            &honest_materials[holder_of_wire0],
            &true_shares,
            &honest_deltas,
            n_parties,
            &params,
            &layout,
            &p,
        ));
    }

    #[test]
    fn test_ligero_stress_1_and_40_instances() {
        let p = test_modulus();
        let n_parties = 5;
        let n_wires = 3;
        let c = 4;
        for instance_count in [1usize, 40usize] {
            let instances: Vec<LigeroInstance> = (0..instance_count as u32)
                .map(|i| {
                    make_instance(
                        &[3 + i % 10, 5 + i % 7, 7 + i % 11],
                        &[10 + i % 10, 20 + i % 7, 30 + i % 11],
                        n_parties,
                        move |pos| (pos + i as usize) % n_parties,
                        &p,
                    )
                })
                .collect();
            let (params, layout) = small_setup(n_wires, instance_count, c.min(instance_count.max(1)), &p);
            let (proof, materials) = ligero_prove(&instances, n_parties, &params, &layout, &p);
            let deltas: Vec<Fp> = instances.iter().map(|i| i.delta.clone()).collect();
            assert!(
                ligero_verify(&proof, &deltas, n_parties, &params, &layout, &p),
                "honest batch of {instance_count} instances must verify"
            );
            let shares = local_shares_for(&instances, 0);
            assert!(ligero_verify_as_party(
                0, &proof, &materials[0], &shares, &deltas, n_parties, &params, &layout, &p,
            ));
        }
    }

    #[test]
    fn test_ligero_verify_rejects_malformed_proof_without_panic() {
        let p = test_modulus();
        let n_parties = 3;
        let inst = make_instance(&[3, 5, 7], &[10, 20, 30], n_parties, |pos| pos % n_parties, &p);
        let instances = vec![inst];
        let (params, layout) = small_setup(3, 1, 1, &p);
        let (mut proof, _materials) = ligero_prove(&instances, n_parties, &params, &layout, &p);
        let deltas: Vec<Fp> = instances.iter().map(|i| i.delta.clone()).collect();
        assert!(proof.opened_columns_x.len() >= 1);
        proof.opened_columns_x.pop();
        assert!(!ligero_verify(&proof, &deltas, n_parties, &params, &layout, &p));
    }

    #[test]
    fn test_layout_search_smoke() {
        let p = test_modulus();
        assert!(try_new_layout(0, 5, 8, &p).is_none());
        assert!(try_new_layout(3, 0, 8, &p).is_none());
        let _ = thread_rng(); // keep rand import exercised across configurations
    }
}
