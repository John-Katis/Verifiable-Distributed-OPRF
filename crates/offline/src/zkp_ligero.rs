//! Approach III-b: Ligero-based Dual-Share ZKP (Π_Lig), paper-accurate
//! multi-instance batched implementation matching Appendix E.
//!
//! ## Row layout
//!
//! For `B ≥ 1` instances, the interleaved witness matrix has `4B + 2` rows:
//!
//!   rows [4b .. 4b+4) for b ∈ 0..B: instance b's data rows
//!         [m^{(b)}, a^{(b)}, p^{(b)}, s^{(b)}]
//!   rows [4B .. 4B+2): two blinding rows, shared across all instances.
//!
//! Each row is RS-encoded over the multiplicative domain
//! `D = {1, ω, …, ω^{n_c-1}}` where ω is a primitive n_c-th root of unity.
//! Instance `b`'s composition contribution is scaled by `α^{5b}` in the
//! batched numerator `N(x)`, so all `5B` constraint slots use distinct
//! powers of the single Fiat-Shamir challenge `α`.
//!
//! ## Two-level column hash (Appendix E Stage 4)
//!
//! For each column j ∈ 0..n_c, the Merkle leaf is
//!   leaf[j] = hash_pair(share_hash(col[j]), private_hash(col[j]))
//! where share entries are the m- and a-row values for every instance and
//! private entries are the p-, s-rows plus the two shared blinding rows.
//! This split lets each distributed verifier `S_j` check its share values
//! at witness positions against `rt_W` WITHOUT ever seeing the computation
//! rows, via `ligero_verify_partial_opening`.
//!
//! ## Constraints (per instance b, with δ_b)
//!
//!   h_1^{(b)}(x) = f_p^{(b)}(ωx) − f_p^{(b)}(x)·f_m^{(b)}(ωx)       on T
//!   h_2^{(b)}(x) = f_s^{(b)}(ωx) − f_s^{(b)}(x) − f_a^{(b)}(ωx)     on T
//!   h_3^{(b)}(x) = f_p^{(b)}(x) − f_m^{(b)}(x)                      at x=1
//!   h_4^{(b)}(x) = f_s^{(b)}(x) − f_a^{(b)}(x)                      at x=1
//!   h_5^{(b)}(x) = f_p^{(b)}(x) − f_s^{(b)}(x) − δ_b                at x=ω^{N-1}
//!
//! With Z_T(x) = Π_{k=0}^{N-2}(x − ω^k), Z_T*(x) = Z_T(x)/(x−1),
//! Z_W(x) = Π_{k=0}^{N-1}(x − ω^k), the batched numerator is
//!
//!   N(x) = Σ_{b=0}^{B-1} α^{5b} · [
//!            (x − ω^{N-1})·(h_1^{(b)} + α h_2^{(b)})
//!          + Z_T*(x)·(x − ω^{N-1})·(α² h_3^{(b)} + α³ h_4^{(b)})
//!          + α⁴·Z_T(x)·h_5^{(b)}
//!          ]
//!
//! and C(x) = N(x)/Z_W(x), deg C ≤ N−1, the RS-encoded composition codeword.

use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use vdoprf_crypto::hash::{hash_field_elements, hash_pair};
use vdoprf_crypto::merkle::MerkleTree;
use vdoprf_crypto::reed_solomon::ReedSolomon;
use vdoprf_crypto::transcript::Transcript;
use vdoprf_field::Fp;
use vdoprf_ss::SubsetT;

/// Parameters for Ligero.
#[derive(Clone, Debug)]
pub struct LigeroParams {
    pub n_c: usize,
    pub n_k: usize,
    pub kappa: usize,
    pub num_queries: usize,
    pub omega: Fp,
}

/// Upper bound on how far above `min_n_c` the divisor search is willing to
/// walk before declaring infeasibility. Keeps pathological cases from hanging
/// when `min_n_c` is astronomically large (small N at high κ).
const DIVISOR_SEARCH_EXTRA: usize = 1 << 20;

impl LigeroParams {
    /// Construct parameters or panic if the paper's κ-soundness / ZK bounds
    /// cannot be met for this (n_k, κ, p). Call sites that expect to skip
    /// infeasible configurations should use [`LigeroParams::try_new`] instead.
    pub fn new(n_k: usize, kappa: usize, modulus: &BigUint) -> Self {
        Self::try_new(n_k, kappa, modulus).unwrap_or_else(|| {
            panic!(
                "Ligero infeasible: n_k = {n_k}, κ = {kappa} — no divisor of p−1 \
                 satisfies the paper's soundness bound n_c ≥ N + (2N−1)·2^(κ/⌊(N−1)/2⌋) \
                 while respecting the ZK cap |Q| ≤ ⌊(N−1)/2⌋"
            )
        })
    }

    /// Paper-faithful parameter selection. Returns `None` when the tight
    /// bounds from Appendix E cannot be satisfied:
    ///
    /// * Zero-knowledge requires `|Q| ≤ ⌊(N−1)/2⌋`, so `N ≥ 3`.
    /// * Soundness requires `n_c ≥ N + (2N−1) · 2^{κ/|Q|}` (with `|Q|` chosen
    ///   as large as ZK allows to minimize n_c).
    /// * `n_c` must be a divisor of `p−1` (for a primitive `n_c`-th root of
    ///   unity) and reachable by the bounded trial-division search.
    pub fn try_new(n_k: usize, kappa: usize, modulus: &BigUint) -> Option<Self> {
        // ZK cap: opening 2|Q| extension columns, each pair {j, ωj}, must not
        // exceed N−1 so the degree-(N−1) row polynomials stay un-interpolated.
        if n_k < 3 {
            return None;
        }
        let q_max = (n_k - 1) / 2;

        // Soundness bound. Pick |Q| = q_max to minimize the required n_c.
        let exp = kappa as f64 / q_max as f64;
        // Guard against overflow of `2^exp`: >60 already implies n_c > 2^60,
        // which no bench configuration can afford.
        if !exp.is_finite() || exp > 60.0 {
            return None;
        }
        let blowup = exp.exp2();
        let min_n_c_f = n_k as f64 + (2.0 * n_k as f64 - 1.0) * blowup;
        if !min_n_c_f.is_finite() || min_n_c_f >= usize::MAX as f64 {
            return None;
        }
        // (3N−1) < n_c is also required so the degree-(2N−1) error polynomial
        // cannot vanish on every extension point.
        let min_n_c = (min_n_c_f.ceil() as usize).max(3 * n_k + 1);

        let p_minus_1 = modulus - BigUint::one();
        // Require n_c to be a power of two so `ReedSolomon::encode` uses the
        // O(n_c log n_c) radix-2 NTT path (paper §3a L221 claims O(BN log N)
        // prover work, which presumes NTT-based encoding).
        let search_cap = min_n_c.saturating_add(DIVISOR_SEARCH_EXTRA);
        let n_c = find_smallest_pow2_divisor_ge(&p_minus_1, min_n_c, search_cap)?;

        let omega = find_primitive_root_of_unity(n_c, modulus)?;

        // Actual query count: as few as soundness permits, capped by the ZK
        // budget q_max and by the number of disjoint extension pairs.
        let ext_pairs = (n_c - n_k) / 2;
        let ratio = (2.0 * n_k as f64 - 1.0) / (n_c - n_k) as f64;
        // With min_n_c enforced above, ratio < 1 always; guard anyway.
        let num_queries = if ratio < 1.0 {
            let needed = (kappa as f64 / -ratio.log2()).ceil() as usize;
            needed.max(1).min(q_max).min(ext_pairs)
        } else {
            q_max.min(ext_pairs)
        };
        if num_queries == 0 {
            return None;
        }

        Some(LigeroParams { n_c, n_k, kappa, num_queries, omega })
    }
}

/// Smallest power-of-two divisor of `n` in `[min_val, max_val]`. The search
/// starts at `min_val.next_power_of_two()` and doubles; since every doubled
/// value of a power of two either divides `n` (because `2^j | n`) or does
/// not, the loop terminates in `O(log₂ max_val)` steps.
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

pub const NUM_DATA_ROWS: usize = 4;
pub const NUM_SHARED_BLINDING: usize = 2;

// Within one instance's 4 data rows:
const ROW_M: usize = 0;
const ROW_A: usize = 1;
const ROW_P: usize = 2;
const ROW_S: usize = 3;

/// Witness matrix for a single Π_Lig instance. Holds only the 4 data rows;
/// blinding is generated inside `ligero_prove` and shared across the batch.
#[derive(Clone, Debug)]
pub struct WitnessMatrix {
    pub m_row: Vec<Fp>,
    pub a_row: Vec<Fp>,
    pub p_row: Vec<Fp>,
    pub s_row: Vec<Fp>,
}

impl WitnessMatrix {
    pub fn new(m_values: Vec<Fp>, a_values: Vec<Fp>, _modulus: &BigUint) -> Self {
        let n = m_values.len();
        assert_eq!(n, a_values.len());
        assert!(n >= 1, "need at least one subset");
        let mut p_row = Vec::with_capacity(n);
        p_row.push(m_values[0].clone());
        for i in 1..n {
            p_row.push(&p_row[i - 1] * &m_values[i]);
        }
        let mut s_row = Vec::with_capacity(n);
        s_row.push(a_values[0].clone());
        for i in 1..n {
            s_row.push(&s_row[i - 1] + &a_values[i]);
        }
        WitnessMatrix { m_row: m_values, a_row: a_values, p_row, s_row }
    }

    pub fn rows(&self) -> [&Vec<Fp>; NUM_DATA_ROWS] {
        [&self.m_row, &self.a_row, &self.p_row, &self.s_row]
    }

    pub fn num_rows(&self) -> usize {
        NUM_DATA_ROWS
    }

    pub fn num_cols(&self) -> usize {
        self.m_row.len()
    }

    pub fn compute_delta(&self, _modulus: &BigUint) -> Fp {
        let product = self.p_row.last().unwrap();
        let sum = self.s_row.last().unwrap();
        product - sum
    }
}

/// Share-row indices within an interleaved column (the `m` and `a` rows of
/// each instance). Used for the two-level column hash.
fn share_indices(num_instances: usize) -> Vec<usize> {
    let mut v = Vec::with_capacity(2 * num_instances);
    for b in 0..num_instances {
        v.push(4 * b + ROW_M);
        v.push(4 * b + ROW_A);
    }
    v
}

/// Private-row indices within an interleaved column (p and s per instance,
/// plus the two shared blinding rows at the tail of the matrix).
fn private_indices(num_instances: usize) -> Vec<usize> {
    let mut v = Vec::with_capacity(2 * num_instances + NUM_SHARED_BLINDING);
    for b in 0..num_instances {
        v.push(4 * b + ROW_P);
        v.push(4 * b + ROW_S);
    }
    let blinding_base = 4 * num_instances;
    for k in 0..NUM_SHARED_BLINDING {
        v.push(blinding_base + k);
    }
    v
}

/// Compute the two-level column leaf used for the Merkle tree over `rt_w`.
///
/// Deviates from the paper's literal formula `H(Ŵ_share,j ‖ H(Ŵ_priv,j))`
/// (raw share values concatenated with the private-row hash) — this instead
/// computes `H(H(share) ‖ H(priv))`, i.e. it additionally hashes the share
/// values before combining. Intentional and still sound: collision
/// resistance is unaffected, and a verifier holding only the share values
/// can still independently recompute `share_hash` and check it against the
/// leaf without ever seeing the private rows (see
/// `ligero_verify_partial_opening` and its test).
fn two_level_leaf(column: &[Fp], num_instances: usize) -> [u8; 32] {
    let share_vals: Vec<Fp> = share_indices(num_instances)
        .iter()
        .map(|&i| column[i].clone())
        .collect();
    let priv_vals: Vec<Fp> = private_indices(num_instances)
        .iter()
        .map(|&i| column[i].clone())
        .collect();
    let share_hash = hash_field_elements(&share_vals);
    let private_hash = hash_field_elements(&priv_vals);
    hash_pair(&share_hash, &private_hash)
}

/// Extract the private-row values from one column and hash them.
fn column_private_hash(column: &[Fp], num_instances: usize) -> [u8; 32] {
    let priv_vals: Vec<Fp> = private_indices(num_instances)
        .iter()
        .map(|&i| column[i].clone())
        .collect();
    hash_field_elements(&priv_vals)
}

/// A Ligero proof over B ≥ 1 instances.
///
/// `rt_c` commits the composition codeword `Ĉ` as a Merkle tree over
/// per-entry leaves (appendix.tex L451); only `Ĉ_j` at the `|Q|` sampled
/// pair-start positions is broadcast, with authentication paths. The
/// consistency codeword `u` is still broadcast in full (appendix.tex L461),
/// and its codeword-ness carries proximity for `Ĉ` via the interleaved
/// proximity lemma over `Ŵ' = (Ŵ; Ĉ)`.
#[derive(Clone, Debug)]
pub struct LigeroProof {
    pub rt_w: [u8; 32],
    pub rt_c: [u8; 32],
    pub rt_u: [u8; 32],
    pub consistency_codeword: Vec<Fp>,
    /// Each entry has length `4B + 2` (the full column).
    pub opened_columns_j: Vec<Vec<Fp>>,
    pub opened_columns_next: Vec<Vec<Fp>>,
    pub merkle_paths_j: Vec<Vec<[u8; 32]>>,
    pub merkle_paths_next: Vec<Vec<[u8; 32]>>,
    /// `Ĉ[j]` at each sampled pair-start position, aligned with
    /// `query_positions`. The constraint check `Z_W(j)·Ĉ_j = N(j)` only
    /// evaluates at `j`, not `j+1`, so one opening per pair suffices.
    pub opened_composition: Vec<Fp>,
    pub composition_merkle_paths: Vec<Vec<[u8; 32]>>,
    pub query_positions: Vec<usize>,
    pub num_instances: usize,
    /// For each witness column k ∈ 0..N, the precomputed hash of that
    /// column's private-row entries. Enables `ligero_verify_partial_opening`
    /// without leaking p, s, or blinding values.
    pub witness_private_hashes: Vec<[u8; 32]>,
    /// Merkle authentication paths from witness-column leaves to `rt_w`.
    pub witness_merkle_paths: Vec<Vec<[u8; 32]>>,
}

/// Merkle-leaf hash for a single composition codeword entry.
fn composition_leaf(c: &Fp) -> [u8; 32] {
    hash_field_elements(std::slice::from_ref(c))
}

impl LigeroProof {
    /// Wire-byte size as placed on the network by the broadcast accounting in
    /// `approach_iii::gen_zkp` (Ligero branch). Matches the exact byte stream
    /// broadcast by the dealer: three Merkle roots, both sets of opened
    /// columns, both codewords, all Merkle paths, witness private hashes, and
    /// witness paths. `query_positions` and `num_instances` are re-derived
    /// from the transcript and not sent on the wire.
    pub fn wire_bytes(&self, feb: usize) -> usize {
        let mut bytes = self.rt_w.len() + self.rt_c.len() + self.rt_u.len();
        for col in &self.opened_columns_j {
            bytes += col.len() * feb;
        }
        for col in &self.opened_columns_next {
            bytes += col.len() * feb;
        }
        bytes += self.opened_composition.len() * feb;
        bytes += self.consistency_codeword.len() * feb;
        for path in &self.merkle_paths_j {
            bytes += path.len() * 32;
        }
        for path in &self.merkle_paths_next {
            bytes += path.len() * 32;
        }
        for path in &self.composition_merkle_paths {
            bytes += path.len() * 32;
        }
        bytes += self.witness_private_hashes.len() * 32;
        for path in &self.witness_merkle_paths {
            bytes += path.len() * 32;
        }
        bytes
    }
}

// ============================================================================
// Polynomial arithmetic helpers (unchanged from previous version).
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

fn poly_shift_omega(p: &[Fp], omega: &Fp, modulus: &BigUint) -> Vec<Fp> {
    let mut omega_pow = Fp::one(modulus);
    let mut out = Vec::with_capacity(p.len());
    for c in p {
        out.push(c * &omega_pow);
        omega_pow = &omega_pow * omega;
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

fn build_vanishing(omega: &Fp, count: usize, modulus: &BigUint) -> Vec<Fp> {
    let mut result: Vec<Fp> = vec![Fp::one(modulus)];
    let mut omega_k = Fp::one(modulus);
    for _ in 0..count {
        let factor = vec![-&omega_k, Fp::one(modulus)];
        result = poly_mul(&result, &factor, modulus);
        omega_k = &omega_k * omega;
    }
    result
}

/// Evaluate the per-instance numerator `N_b(ω^j)` from 4 column entries
/// `[m, a, p, s]` at j and the corresponding 4 entries at (j+1) mod n_c.
#[allow(clippy::too_many_arguments)]
fn recompute_n_at(
    col_j: &[Fp],
    col_next: &[Fp],
    x: &Fp,
    alpha: &Fp,
    delta: &Fp,
    omega_n_minus_1: &Fp,
    z_t_at_x: &Fp,
    z_t_star_at_x: &Fp,
    _modulus: &BigUint,
) -> Fp {
    let p_at_x = &col_j[ROW_P];
    let m_at_x = &col_j[ROW_M];
    let s_at_x = &col_j[ROW_S];
    let a_at_x = &col_j[ROW_A];
    let p_at_ox = &col_next[ROW_P];
    let m_at_ox = &col_next[ROW_M];
    let s_at_ox = &col_next[ROW_S];
    let a_at_ox = &col_next[ROW_A];

    let h1 = p_at_ox - &(p_at_x * m_at_ox);
    let h2 = &(s_at_ox - s_at_x) - a_at_ox;
    let h3 = p_at_x - m_at_x;
    let h4 = s_at_x - a_at_x;
    let h5 = &(p_at_x - s_at_x) - delta;

    let x_minus_last = x - omega_n_minus_1;
    let alpha_sq = alpha * alpha;
    let alpha_3 = &alpha_sq * alpha;
    let alpha_4 = &alpha_3 * alpha;

    let term_a = &x_minus_last * &(&h1 + &(alpha * &h2));
    let term_b = &(z_t_star_at_x * &x_minus_last) * &(&(&alpha_sq * &h3) + &(&alpha_3 * &h4));
    let term_c = &(&alpha_4 * z_t_at_x) * &h5;
    &(&term_a + &term_b) + &term_c
}

/// Build the per-instance numerator polynomial `N_b(x)` (prover-side,
/// coefficient form) using α^0..α^4 powers internally.
#[allow(clippy::too_many_arguments)]
fn build_numerator_poly(
    coeffs_m: &[Fp],
    coeffs_a: &[Fp],
    coeffs_p: &[Fp],
    coeffs_s: &[Fp],
    alpha: &Fp,
    delta: &Fp,
    omega: &Fp,
    z_t: &[Fp],
    z_t_star: &[Fp],
    omega_n_minus_1: &Fp,
    modulus: &BigUint,
) -> Vec<Fp> {
    let coeffs_m_shift = poly_shift_omega(coeffs_m, omega, modulus);
    let coeffs_p_shift = poly_shift_omega(coeffs_p, omega, modulus);
    let coeffs_s_shift = poly_shift_omega(coeffs_s, omega, modulus);
    let coeffs_a_shift = poly_shift_omega(coeffs_a, omega, modulus);

    let h1 = poly_sub(
        &coeffs_p_shift,
        &poly_mul(coeffs_p, &coeffs_m_shift, modulus),
        modulus,
    );
    let h2 = poly_sub(
        &poly_sub(&coeffs_s_shift, coeffs_s, modulus),
        &coeffs_a_shift,
        modulus,
    );
    let h3 = poly_sub(coeffs_p, coeffs_m, modulus);
    let h4 = poly_sub(coeffs_s, coeffs_a, modulus);
    let mut h5 = poly_sub(coeffs_p, coeffs_s, modulus);
    h5[0] = &h5[0] - delta;

    let x_minus_last = vec![-omega_n_minus_1, Fp::one(modulus)];
    let alpha_sq = alpha * alpha;
    let alpha_3 = &alpha_sq * alpha;
    let alpha_4 = &alpha_3 * alpha;

    let h1_plus_alpha_h2 = poly_add(&h1, &poly_scale(&h2, alpha), modulus);
    let term_a = poly_mul(&x_minus_last, &h1_plus_alpha_h2, modulus);

    let inner_b = poly_add(&poly_scale(&h3, &alpha_sq), &poly_scale(&h4, &alpha_3), modulus);
    let z_t_star_times_xlast = poly_mul(z_t_star, &x_minus_last, modulus);
    let term_b = poly_mul(&z_t_star_times_xlast, &inner_b, modulus);

    let term_c = poly_scale(&poly_mul(z_t, &h5, modulus), &alpha_4);
    poly_add(&poly_add(&term_a, &term_b, modulus), &term_c, modulus)
}

/// Generate a Ligero proof over `B = witnesses.len()` instances using a
/// single Fiat-Shamir challenge α with per-instance offset α^{5b}. Shared
/// blinding: two random rows of length n_k, committed alongside the data
/// rows in the single column Merkle tree under the two-level column leaf.
pub fn ligero_prove(
    witnesses: &[&WitnessMatrix],
    deltas: &[Fp],
    params: &LigeroParams,
    modulus: &BigUint,
) -> LigeroProof {
    let m = witnesses.len();
    assert!(m >= 1, "ligero_prove requires at least one instance");
    assert_eq!(m, deltas.len(), "one delta per instance");
    for w in witnesses {
        assert_eq!(w.num_cols(), params.n_k, "witness column count must equal n_k");
    }

    let n_k = params.n_k;
    let n_c = params.n_c;
    let omega = &params.omega;
    let rs = ReedSolomon::with_root_of_unity(n_c, n_k, omega.clone());

    // Build interleaved row list: 4B data rows followed by 2 shared blinding rows.
    let mut rng = rand::thread_rng();
    let blinding_rows: Vec<Vec<Fp>> = (0..NUM_SHARED_BLINDING)
        .map(|_| (0..n_k).map(|_| Fp::random(modulus, &mut rng)).collect())
        .collect();

    let mut row_refs: Vec<&Vec<Fp>> = Vec::with_capacity(m * NUM_DATA_ROWS + NUM_SHARED_BLINDING);
    for w in witnesses {
        for r in w.rows().iter() {
            row_refs.push(*r);
        }
    }
    for br in &blinding_rows {
        row_refs.push(br);
    }
    debug_assert_eq!(row_refs.len(), m * NUM_DATA_ROWS + NUM_SHARED_BLINDING);

    let row_coeffs: Vec<Vec<Fp>> = row_refs
        .iter()
        .map(|row| rs.interpolate_coefficients(row))
        .collect();
    let encoded_rows: Vec<Vec<Fp>> = row_coeffs.iter().map(|c| rs.encode(c)).collect();

    // Column-wise matrix with two-level leaves.
    let columns: Vec<Vec<Fp>> = (0..n_c)
        .map(|j| encoded_rows.iter().map(|row| row[j].clone()).collect())
        .collect();
    let column_leaves: Vec<[u8; 32]> = columns.iter().map(|c| two_level_leaf(c, m)).collect();
    let merkle_w = MerkleTree::new(column_leaves);
    let rt_w = merkle_w.root();

    // Fiat-Shamir: α bound to rt_w and all deltas.
    let mut transcript = Transcript::new(b"Ligero");
    transcript.append_commitment(&rt_w);
    for d in deltas {
        transcript.append_field_element(d);
    }
    let alpha = transcript.challenge(modulus);
    let alpha_5 = alpha.pow(&BigUint::from(5u32));

    // Shared vanishing polynomials and ω^{N-1}.
    let z_t = build_vanishing(omega, n_k.saturating_sub(1), modulus);
    let z_w = build_vanishing(omega, n_k, modulus);
    let one_p = Fp::one(modulus);
    let x_minus_one = vec![-&one_p, Fp::one(modulus)];
    let z_t_star = if n_k >= 2 {
        poly_divide_exact(&z_t, &x_minus_one, modulus)
    } else {
        vec![Fp::one(modulus)]
    };
    let omega_n_minus_1 = omega.pow(&BigUint::from(n_k.saturating_sub(1)));

    // Build batched C(x) = [Σ_b α^{5b} · N_b(x)] / Z_W(x). We accumulate each
    // per-instance quotient scaled by α^{5b} into a running c_batch poly.
    let mut c_batch: Vec<Fp> = vec![Fp::zero(modulus); n_k];
    let mut alpha_offset = Fp::one(modulus);
    for b in 0..m {
        let base = b * NUM_DATA_ROWS;
        let coeffs_m = &row_coeffs[base + ROW_M];
        let coeffs_a = &row_coeffs[base + ROW_A];
        let coeffs_p = &row_coeffs[base + ROW_P];
        let coeffs_s = &row_coeffs[base + ROW_S];
        let n_poly_b = build_numerator_poly(
            coeffs_m,
            coeffs_a,
            coeffs_p,
            coeffs_s,
            &alpha,
            &deltas[b],
            omega,
            &z_t,
            &z_t_star,
            &omega_n_minus_1,
            modulus,
        );
        let c_b = poly_divide_exact(&n_poly_b, &z_w, modulus);
        for (j, cb_j) in c_b.iter().enumerate() {
            if j >= c_batch.len() {
                c_batch.resize(j + 1, Fp::zero(modulus));
            }
            c_batch[j] = &c_batch[j] + &(&alpha_offset * cb_j);
        }
        alpha_offset = &alpha_offset * &alpha_5;
    }

    let composition_codeword: Vec<Fp> = rs.encode(&c_batch);
    let composition_leaves: Vec<[u8; 32]> =
        composition_codeword.iter().map(composition_leaf).collect();
    let merkle_c = MerkleTree::new(composition_leaves);
    let rt_c = merkle_c.root();

    transcript.append_commitment(&rt_c);
    let total_witness_rows = m * NUM_DATA_ROWS + NUM_SHARED_BLINDING;
    let v_coeffs = transcript.challenge_vec(total_witness_rows + 1, modulus);

    let consistency_codeword: Vec<Fp> = (0..n_c)
        .map(|j| {
            let mut u_j = Fp::zero(modulus);
            for (i, row) in encoded_rows.iter().enumerate() {
                u_j = &u_j + &(&v_coeffs[i] * &row[j]);
            }
            u_j = &u_j + &(&v_coeffs[total_witness_rows] * &composition_codeword[j]);
            u_j
        })
        .collect();
    let rt_u = hash_field_elements(&consistency_codeword);

    transcript.append_commitment(&rt_u);
    // Paper Appendix E: partition the `n_c − N` extension positions into
    // ⌊(n_c−N)/2⌋ disjoint pairs {j, ωj} and sample |Q| of them via FS. We
    // index pairs directly so (pos, pos+1) never wraps into a witness column.
    let num_pairs = (n_c - n_k) / 2;
    assert!(num_pairs >= params.num_queries);
    // Resample on a duplicate `pair_idx` (rather than silently dropping it)
    // so `query_positions.len()` always equals exactly `params.num_queries`
    // — dropping duplicates would sample fewer than the nominal parameter,
    // weakening the soundness margin below what `params.num_queries` was
    // chosen to guarantee. `ligero_verify`'s sampling loop below must stay
    // structurally identical (same challenge_index + append_bytes sequence
    // per retry) so both sides derive the same positions.
    let mut query_positions: Vec<usize> = Vec::with_capacity(params.num_queries);
    for q in 0..params.num_queries {
        loop {
            let pair_idx = transcript.challenge_index(num_pairs);
            let pos = n_k + 2 * pair_idx;
            transcript.append_bytes(&(q as u64).to_be_bytes());
            if query_positions.contains(&pos) {
                continue;
            }
            query_positions.push(pos);
            break;
        }
    }

    let opened_columns_j: Vec<Vec<Fp>> =
        query_positions.iter().map(|&j| columns[j].clone()).collect();
    let opened_columns_next: Vec<Vec<Fp>> = query_positions
        .iter()
        .map(|&j| columns[j + 1].clone())
        .collect();
    let merkle_paths_j: Vec<Vec<[u8; 32]>> = query_positions
        .iter()
        .map(|&j| merkle_w.authentication_path(j))
        .collect();
    let merkle_paths_next: Vec<Vec<[u8; 32]>> = query_positions
        .iter()
        .map(|&j| merkle_w.authentication_path(j + 1))
        .collect();
    let opened_composition: Vec<Fp> = query_positions
        .iter()
        .map(|&j| composition_codeword[j].clone())
        .collect();
    let composition_merkle_paths: Vec<Vec<[u8; 32]>> = query_positions
        .iter()
        .map(|&j| merkle_c.authentication_path(j))
        .collect();

    // Stage-4 witness-position partial-opening material: private-row hashes
    // and Merkle paths for each k ∈ 0..n_k.
    let witness_private_hashes: Vec<[u8; 32]> = (0..n_k)
        .map(|k| column_private_hash(&columns[k], m))
        .collect();
    let witness_merkle_paths: Vec<Vec<[u8; 32]>> = (0..n_k)
        .map(|k| merkle_w.authentication_path(k))
        .collect();

    LigeroProof {
        rt_w,
        rt_c,
        rt_u,
        consistency_codeword,
        opened_columns_j,
        opened_columns_next,
        merkle_paths_j,
        merkle_paths_next,
        opened_composition,
        composition_merkle_paths,
        query_positions,
        num_instances: m,
        witness_private_hashes,
        witness_merkle_paths,
    }
}

/// Verify a Ligero proof (stages 1-3: soundness + proximity).
pub fn ligero_verify(
    proof: &LigeroProof,
    deltas: &[Fp],
    params: &LigeroParams,
    modulus: &BigUint,
) -> bool {
    let m = proof.num_instances;
    if deltas.len() != m || m == 0 {
        return false;
    }
    let n_k = params.n_k;
    let n_c = params.n_c;
    let omega = &params.omega;
    let expected_row_count = m * NUM_DATA_ROWS + NUM_SHARED_BLINDING;

    if proof.consistency_codeword.len() != n_c {
        return false;
    }
    if hash_field_elements(&proof.consistency_codeword) != proof.rt_u {
        return false;
    }
    if proof.opened_composition.len() != proof.query_positions.len()
        || proof.composition_merkle_paths.len() != proof.query_positions.len()
    {
        return false;
    }
    if proof.witness_private_hashes.len() != n_k
        || proof.witness_merkle_paths.len() != n_k
    {
        return false;
    }
    if proof.opened_columns_j.len() != proof.query_positions.len()
        || proof.opened_columns_next.len() != proof.query_positions.len()
        || proof.merkle_paths_j.len() != proof.query_positions.len()
        || proof.merkle_paths_next.len() != proof.query_positions.len()
    {
        // A malformed/adversarial proof with fewer entries than
        // `query_positions` must be rejected here, not reach the indexing
        // loop below (which would panic with an out-of-bounds index).
        return false;
    }

    // Merkle verification for queried column pairs (two-level leaf for Ŵ;
    // single-leaf hash for Ĉ). Pairs are always (j, j+1) with j+1 < n_c by
    // construction of the sampling.
    for (i, &j) in proof.query_positions.iter().enumerate() {
        if j + 1 >= n_c
            || proof.opened_columns_j[i].len() != expected_row_count
            || proof.opened_columns_next[i].len() != expected_row_count
        {
            return false;
        }
        let leaf_j = two_level_leaf(&proof.opened_columns_j[i], m);
        if !MerkleTree::verify_path(&proof.rt_w, &leaf_j, j, n_c, &proof.merkle_paths_j[i]) {
            return false;
        }
        let leaf_next = two_level_leaf(&proof.opened_columns_next[i], m);
        if !MerkleTree::verify_path(
            &proof.rt_w,
            &leaf_next,
            j + 1,
            n_c,
            &proof.merkle_paths_next[i],
        ) {
            return false;
        }
        let leaf_c = composition_leaf(&proof.opened_composition[i]);
        if !MerkleTree::verify_path(
            &proof.rt_c,
            &leaf_c,
            j,
            n_c,
            &proof.composition_merkle_paths[i],
        ) {
            return false;
        }
    }

    // Re-derive FS challenges (single-α path).
    let mut transcript = Transcript::new(b"Ligero");
    transcript.append_commitment(&proof.rt_w);
    for d in deltas {
        transcript.append_field_element(d);
    }
    let alpha = transcript.challenge(modulus);
    let alpha_5 = alpha.pow(&BigUint::from(5u32));

    transcript.append_commitment(&proof.rt_c);
    let v_coeffs = transcript.challenge_vec(expected_row_count + 1, modulus);

    transcript.append_commitment(&proof.rt_u);
    let num_pairs = (n_c - n_k) / 2;
    if num_pairs < params.num_queries {
        return false;
    }
    // Must stay structurally identical to `ligero_prove`'s sampling loop
    // (resample on duplicate, same challenge_index + append_bytes sequence
    // per retry) so both sides derive the same `params.num_queries`
    // positions.
    let mut expected_positions: Vec<usize> = Vec::with_capacity(params.num_queries);
    for q in 0..params.num_queries {
        loop {
            let pair_idx = transcript.challenge_index(num_pairs);
            let pos = n_k + 2 * pair_idx;
            transcript.append_bytes(&(q as u64).to_be_bytes());
            if expected_positions.contains(&pos) {
                continue;
            }
            expected_positions.push(pos);
            break;
        }
    }
    if expected_positions != proof.query_positions {
        return false;
    }

    let z_t = build_vanishing(omega, n_k.saturating_sub(1), modulus);
    let z_w = build_vanishing(omega, n_k, modulus);
    let one_p = Fp::one(modulus);
    let x_minus_one = vec![-&one_p, Fp::one(modulus)];
    let z_t_star = if n_k >= 2 {
        poly_divide_exact(&z_t, &x_minus_one, modulus)
    } else {
        vec![Fp::one(modulus)]
    };
    let omega_n_minus_1 = omega.pow(&BigUint::from(n_k.saturating_sub(1)));

    let rs = ReedSolomon::with_root_of_unity(n_c, n_k, omega.clone());

    for (i, &j) in proof.query_positions.iter().enumerate() {
        let col_j = &proof.opened_columns_j[i];
        let col_next = &proof.opened_columns_next[i];
        let x = &rs.domain[j];
        let z_w_at_x = poly_eval(&z_w, x, modulus);
        let z_t_at_x = poly_eval(&z_t, x, modulus);
        let z_t_star_at_x = poly_eval(&z_t_star, x, modulus);

        // N_batch(ω^j) = Σ_b α^{5b} · N_b(ω^j).
        let mut n_batch = Fp::zero(modulus);
        let mut alpha_offset = Fp::one(modulus);
        for b in 0..m {
            let base = b * NUM_DATA_ROWS;
            let slice_j = &col_j[base..base + NUM_DATA_ROWS];
            let slice_next = &col_next[base..base + NUM_DATA_ROWS];
            let n_b = recompute_n_at(
                slice_j,
                slice_next,
                x,
                &alpha,
                &deltas[b],
                &omega_n_minus_1,
                &z_t_at_x,
                &z_t_star_at_x,
                modulus,
            );
            n_batch = &n_batch + &(&alpha_offset * &n_b);
            alpha_offset = &alpha_offset * &alpha_5;
        }

        let c_at_j = &proof.opened_composition[i];
        let lhs = &z_w_at_x * c_at_j;
        if lhs != n_batch {
            return false;
        }

        // Interleaved consistency: u(j) == Σ v_i · col_j[i] + v_last · C_batch(j).
        let mut u_expected = Fp::zero(modulus);
        for (row_idx, v_i) in v_coeffs.iter().take(expected_row_count).enumerate() {
            u_expected = &u_expected + &(v_i * &col_j[row_idx]);
        }
        u_expected = &u_expected + &(&v_coeffs[expected_row_count] * c_at_j);
        if u_expected != proof.consistency_codeword[j] {
            return false;
        }
    }

    // Paper appendix.tex L490(ii): the interleaved consistency codeword `u`
    // must be a valid RS codeword of degree ≤ N−1. Since `u` is formed over
    // Ŵ' = (Ŵ; Ĉ), this single proximity check covers every witness row
    // AND the composition codeword via the interleaved-code proximity lemma,
    // so `Ĉ` does not need a separate codeword test. Implemented via iNTT
    // (O(n_c log n_c)) inside `ReedSolomon::is_valid_codeword`, matching the
    // paper's O(κ log N) hashes + O(Bκ) verifier-cost claim.
    if !rs.is_valid_codeword(&proof.consistency_codeword) {
        return false;
    }

    true
}

/// Stage-4 partial-opening verification (paper Appendix E "our contribution").
///
/// Each distributed verifier `S_j` holds the share values (`m^{(b)}_k`,
/// `a^{(b)}_k`) for every instance `b` at every witness column `k` whose
/// subset does not contain `j`. For each such k, the verifier calls this
/// function with `witness_column = k` and `known_share_values` laid out as
///
///   [m^{(0)}_k, a^{(0)}_k, m^{(1)}_k, a^{(1)}_k, …, m^{(B-1)}_k, a^{(B-1)}_k]
///
/// (length `2B`). The function checks that these values are consistent with
/// `proof.rt_w` *without* revealing the private rows (p, s, blinding).
pub fn ligero_verify_partial_opening(
    proof: &LigeroProof,
    witness_column: usize,
    known_share_values: &[Fp],
    params: &LigeroParams,
) -> bool {
    let m = proof.num_instances;
    if witness_column >= params.n_k {
        return false;
    }
    if known_share_values.len() != 2 * m {
        return false;
    }
    if witness_column >= proof.witness_private_hashes.len()
        || witness_column >= proof.witness_merkle_paths.len()
    {
        return false;
    }

    let share_hash = hash_field_elements(known_share_values);
    let private_hash = &proof.witness_private_hashes[witness_column];
    let leaf = hash_pair(&share_hash, private_hash);

    MerkleTree::verify_path(
        &proof.rt_w,
        &leaf,
        witness_column,
        params.n_c,
        &proof.witness_merkle_paths[witness_column],
    )
}

// Keep BTreeMap and SubsetT referenced so callers that still import the
// module's public re-exports don't break. These are used by call sites in
// approach_iii.rs indirectly via shared imports.
#[allow(dead_code)]
fn _unused(_m: BTreeMap<SubsetT, (Fp, Fp)>) {
    let _ = <BigUint as Zero>::zero();
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests use κ = 8 with N = 3, which under the paper's bound
    //   n_c ≥ N + (2N−1)·2^{κ/|Q|}   (with |Q| = ⌊(N−1)/2⌋ = 1)
    // requires n_c ≥ 1283. We pick p = 65537 so p − 1 = 2^16 has a divisor
    // (2048) in that range. The small field keeps tests fast while still
    // exercising the real soundness path.
    fn test_modulus() -> BigUint {
        BigUint::from(65537u32)
    }

    fn make_witness(m_vals: &[u32], a_vals: &[u32], modulus: &BigUint) -> WitnessMatrix {
        let m: Vec<Fp> = m_vals.iter().map(|v| Fp::new(BigUint::from(*v), modulus)).collect();
        let a: Vec<Fp> = a_vals.iter().map(|v| Fp::new(BigUint::from(*v), modulus)).collect();
        WitnessMatrix::new(m, a, modulus)
    }

    #[test]
    fn test_root_of_unity_small() {
        let p = test_modulus();
        let omega = find_primitive_root_of_unity(16, &p).expect("should find");
        let one = Fp::one(&p);
        assert_eq!(omega.pow(&BigUint::from(16u32)), one);
        assert_ne!(omega.pow(&BigUint::from(8u32)), one);
    }

    #[test]
    fn test_poly_divide_exact() {
        let p = test_modulus();
        let num = vec![
            Fp::new(BigUint::from(2u32), &p),
            -&Fp::new(BigUint::from(3u32), &p),
            Fp::one(&p),
        ];
        let div = vec![-&Fp::one(&p), Fp::one(&p)];
        let q = poly_divide_exact(&num, &div, &p);
        assert_eq!(q.len(), 2);
        assert_eq!(q[0], -&Fp::new(BigUint::from(2u32), &p));
        assert_eq!(q[1], Fp::one(&p));
    }

    #[test]
    fn test_witness_matrix() {
        let p = test_modulus();
        let w = make_witness(&[3, 5, 7], &[10, 20, 30], &p);
        assert_eq!(w.p_row[2].value, BigUint::from(105u32) % &p);
        assert_eq!(w.s_row[2].value, BigUint::from(60u32));
        assert_eq!(w.compute_delta(&p).value, BigUint::from(45u32));
        assert_eq!(w.num_rows(), NUM_DATA_ROWS);
    }

    #[test]
    fn test_share_private_indices_partition() {
        // For B = 3 instances, share = 6 entries, private = 8 entries = 4B+2 total.
        let s = share_indices(3);
        let p = private_indices(3);
        assert_eq!(s.len(), 6);
        assert_eq!(p.len(), 8);
        // disjoint union = {0..14}
        let mut all: Vec<usize> = s.iter().chain(p.iter()).copied().collect();
        all.sort();
        assert_eq!(all, (0..14).collect::<Vec<_>>());
    }

    #[test]
    fn test_ligero_prove_verify_single() {
        let p = test_modulus();
        let w = make_witness(&[3, 5, 7], &[10, 20, 30], &p);
        let delta = w.compute_delta(&p);
        let params = LigeroParams::new(3, 8, &p);
        let proof = ligero_prove(&[&w], std::slice::from_ref(&delta), &params, &p);
        assert_eq!(proof.num_instances, 1);
        // Opened columns have 4*1 + 2 = 6 entries.
        assert_eq!(proof.opened_columns_j[0].len(), NUM_DATA_ROWS + NUM_SHARED_BLINDING);
        assert!(ligero_verify(&proof, std::slice::from_ref(&delta), &params, &p));
    }

    /// A malformed/adversarial proof with fewer `opened_columns_j` entries
    /// than `query_positions` must be rejected with `false`, not panic with
    /// an out-of-bounds index inside the verification loop.
    #[test]
    fn test_ligero_verify_rejects_short_opened_columns_without_panic() {
        let p = test_modulus();
        let w = make_witness(&[3, 5, 7], &[10, 20, 30], &p);
        let delta = w.compute_delta(&p);
        let params = LigeroParams::new(3, 8, &p);
        let mut proof = ligero_prove(&[&w], std::slice::from_ref(&delta), &params, &p);
        assert!(proof.query_positions.len() >= 1, "test needs at least one query");

        // Truncate opened_columns_j so it's shorter than query_positions.
        proof.opened_columns_j.pop();
        assert!(!ligero_verify(&proof, std::slice::from_ref(&delta), &params, &p));
    }

    /// Regression guard for "dedup can silently under-sample": force
    /// `num_queries` close to `num_pairs` (a small, hand-built RS domain
    /// bypassing `LigeroParams::new`'s soundness-driven `n_c` search) so a
    /// `pair_idx` collision is virtually certain across a handful of
    /// witnesses, and confirm `query_positions` always has *exactly*
    /// `num_queries` distinct entries — never fewer, regardless of
    /// collisions during sampling.
    #[test]
    fn test_ligero_query_positions_exactly_num_queries_under_forced_collisions() {
        let p = test_modulus();
        let n_c = 16;
        let n_k = 3;
        let omega = find_primitive_root_of_unity(n_c, &p).expect("root of unity must exist");
        let num_pairs = (n_c - n_k) / 2;
        let params = LigeroParams {
            n_c,
            n_k,
            kappa: 8,
            num_queries: num_pairs,
            omega,
        };

        let mut saw_len_below_num_queries_ever = false;
        for seed in 0u32..20 {
            let w = make_witness(
                &[3 + seed, 5 + seed, 7 + seed],
                &[10 + seed, 20 + seed, 30 + seed],
                &p,
            );
            let delta = w.compute_delta(&p);
            let proof = ligero_prove(&[&w], std::slice::from_ref(&delta), &params, &p);

            assert_eq!(
                proof.query_positions.len(),
                params.num_queries,
                "seed {seed}: query_positions must never be shorter than num_queries",
            );
            let mut sorted = proof.query_positions.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                params.num_queries,
                "seed {seed}: query_positions must contain no duplicates",
            );
            if proof.query_positions.len() < params.num_queries {
                saw_len_below_num_queries_ever = true;
            }
            assert!(ligero_verify(&proof, std::slice::from_ref(&delta), &params, &p));
        }
        assert!(
            !saw_len_below_num_queries_ever,
            "fix regressed: some run produced fewer than num_queries positions",
        );
    }

    #[test]
    fn test_ligero_single_rejects_wrong_witness() {
        let p = test_modulus();
        let w_honest = make_witness(&[3, 5, 7], &[10, 20, 30], &p);
        let delta = w_honest.compute_delta(&p);
        let mut w_bad = w_honest.clone();
        w_bad.p_row[2] = Fp::new(BigUint::from(99u32), &p);
        let params = LigeroParams::new(3, 8, &p);
        let proof = ligero_prove(&[&w_bad], std::slice::from_ref(&delta), &params, &p);
        assert!(!ligero_verify(&proof, std::slice::from_ref(&delta), &params, &p));
    }

    #[test]
    fn test_ligero_single_rejects_wrong_delta() {
        let p = test_modulus();
        let w = make_witness(&[3, 5, 7], &[10, 20, 30], &p);
        let honest_delta = w.compute_delta(&p);
        let params = LigeroParams::new(3, 8, &p);
        let proof = ligero_prove(&[&w], std::slice::from_ref(&honest_delta), &params, &p);
        let wrong = Fp::new(BigUint::from(44u32), &p);
        assert!(!ligero_verify(&proof, std::slice::from_ref(&wrong), &params, &p));
    }

    #[test]
    fn test_ligero_batch_prove_verify_3_instances() {
        let p = test_modulus();
        let w0 = make_witness(&[3, 5, 7], &[10, 20, 30], &p);
        let w1 = make_witness(&[2, 4, 6], &[1, 2, 3], &p);
        let w2 = make_witness(&[11, 13, 17], &[7, 8, 9], &p);
        let deltas = vec![
            w0.compute_delta(&p),
            w1.compute_delta(&p),
            w2.compute_delta(&p),
        ];
        let witnesses = [&w0, &w1, &w2];
        let params = LigeroParams::new(3, 8, &p);
        let proof = ligero_prove(&witnesses, &deltas, &params, &p);
        assert_eq!(proof.num_instances, 3);
        // 3 instances × 4 + 2 shared blinding = 14 entries per column.
        for col in &proof.opened_columns_j {
            assert_eq!(col.len(), 3 * NUM_DATA_ROWS + NUM_SHARED_BLINDING);
        }
        assert!(ligero_verify(&proof, &deltas, &params, &p));
    }

    #[test]
    fn test_ligero_batch_rejects_tampered_instance() {
        let p = test_modulus();
        let w0 = make_witness(&[3, 5, 7], &[10, 20, 30], &p);
        let mut w1 = make_witness(&[2, 4, 6], &[1, 2, 3], &p);
        let w2 = make_witness(&[11, 13, 17], &[7, 8, 9], &p);
        let deltas = vec![
            w0.compute_delta(&p),
            w1.compute_delta(&p),
            w2.compute_delta(&p),
        ];
        w1.p_row[2] = Fp::new(BigUint::from(99u32), &p);
        let witnesses = [&w0, &w1, &w2];
        let params = LigeroParams::new(3, 8, &p);
        let proof = ligero_prove(&witnesses, &deltas, &params, &p);
        assert!(!ligero_verify(&proof, &deltas, &params, &p));
    }

    #[test]
    fn test_ligero_batch_rejects_wrong_delta_for_one_instance() {
        let p = test_modulus();
        let w0 = make_witness(&[3, 5, 7], &[10, 20, 30], &p);
        let w1 = make_witness(&[2, 4, 6], &[1, 2, 3], &p);
        let w2 = make_witness(&[11, 13, 17], &[7, 8, 9], &p);
        let honest = vec![
            w0.compute_delta(&p),
            w1.compute_delta(&p),
            w2.compute_delta(&p),
        ];
        let witnesses = [&w0, &w1, &w2];
        let params = LigeroParams::new(3, 8, &p);
        let proof = ligero_prove(&witnesses, &honest, &params, &p);
        let mut wrong = honest.clone();
        wrong.swap(0, 2);
        assert!(!ligero_verify(&proof, &wrong, &params, &p));
    }

    /// Stress: 1 instance and 100 instances must both accept and produce
    /// structurally correct proofs.
    #[test]
    fn test_ligero_batch_1_and_100_instances() {
        let p = test_modulus();
        let params = LigeroParams::new(3, 8, &p);

        for instance_count in [1usize, 100usize] {
            let witnesses_owned: Vec<WitnessMatrix> = (0..instance_count)
                .map(|i| {
                    let a = ((i as u32) % 10) + 1;
                    let b = ((i as u32) % 7) + 2;
                    let c = ((i as u32) % 11) + 3;
                    make_witness(
                        &[a, b, c],
                        &[a + 10, b + 20, c + 30],
                        &p,
                    )
                })
                .collect();
            let deltas: Vec<Fp> = witnesses_owned.iter().map(|w| w.compute_delta(&p)).collect();
            let witness_refs: Vec<&WitnessMatrix> = witnesses_owned.iter().collect();

            let proof = ligero_prove(&witness_refs, &deltas, &params, &p);
            assert_eq!(proof.num_instances, instance_count);
            for col in &proof.opened_columns_j {
                assert_eq!(
                    col.len(),
                    instance_count * NUM_DATA_ROWS + NUM_SHARED_BLINDING,
                    "opened column length at B = {instance_count}"
                );
            }
            assert!(
                ligero_verify(&proof, &deltas, &params, &p),
                "honest batch of {instance_count} instances must verify"
            );
        }
    }

    #[test]
    fn test_ligero_partial_opening_at_witness_position() {
        // Paper Stage-4: distributed verifier checks its known shares at a
        // witness column without seeing private rows.
        let p = test_modulus();
        let w0 = make_witness(&[3, 5, 7], &[10, 20, 30], &p);
        let w1 = make_witness(&[2, 4, 6], &[1, 2, 3], &p);
        let w2 = make_witness(&[11, 13, 17], &[7, 8, 9], &p);
        let deltas = vec![
            w0.compute_delta(&p),
            w1.compute_delta(&p),
            w2.compute_delta(&p),
        ];
        let witnesses = [&w0, &w1, &w2];
        let params = LigeroParams::new(3, 8, &p);
        let proof = ligero_prove(&witnesses, &deltas, &params, &p);

        // Verifier knows (m^(b)_k, a^(b)_k) at witness column k=0 for all b.
        // Layout: [m^(0)_0, a^(0)_0, m^(1)_0, a^(1)_0, m^(2)_0, a^(2)_0].
        let shares_at_0: Vec<Fp> = vec![
            w0.m_row[0].clone(),
            w0.a_row[0].clone(),
            w1.m_row[0].clone(),
            w1.a_row[0].clone(),
            w2.m_row[0].clone(),
            w2.a_row[0].clone(),
        ];
        assert!(ligero_verify_partial_opening(&proof, 0, &shares_at_0, &params));

        // Tamper any one share → rejection.
        let mut bad = shares_at_0.clone();
        bad[2] = Fp::new(BigUint::from(42u32), &p);
        assert!(!ligero_verify_partial_opening(&proof, 0, &bad, &params));

        // Out-of-range column → rejection.
        assert!(!ligero_verify_partial_opening(&proof, 99, &shares_at_0, &params));

        // Wrong share vector length → rejection.
        let truncated = shares_at_0[..4].to_vec();
        assert!(!ligero_verify_partial_opening(&proof, 0, &truncated, &params));

        // Column k = 1 also works.
        let shares_at_1: Vec<Fp> = vec![
            w0.m_row[1].clone(),
            w0.a_row[1].clone(),
            w1.m_row[1].clone(),
            w1.a_row[1].clone(),
            w2.m_row[1].clone(),
            w2.a_row[1].clone(),
        ];
        assert!(ligero_verify_partial_opening(&proof, 1, &shares_at_1, &params));
    }
}
