//! Approach II: Degenerate Encoding Multiplication (Π_Gen^deg) with
//! cross-phase batched DZKP.
//!
//! `gen_degenerate(&[e_0, e_1, …], …)` runs `N ≥ 1` phases and emits a
//! single DZKP covering every RSS multiplication from every phase. For
//! `N = 1` the result matches the original single-phase behavior.
//!
//! Per-phase steps (unchanged from the single-instance version):
//!   1. Derive multiplicative subset shares `m_T` from PRG seeds.
//!   2. Local exponentiation: `m'_T = m_T^e` wrapped as degenerate encoding.
//!   3. Sequential Π_DegMul fold: `⟦q_1⟧ = ⟨M_{T_1}⟩`, then
//!      `⟦q_j⟧ = ⟦q_{j-1}⟧ · ⟨M_{T_j}⟩` for `j = 2..N`. Every step's right
//!      operand is a degenerate encoding, so every step is a `Π_DegMul`
//!      (full-RSS × degenerate → full-RSS) — full `Π_RSS.Mul` is never
//!      invoked, unlike the binary-tree fan-in `pub_base_exp_semi_honest`
//!      uses for Π_exp's degenerate-encoding aggregation.
//!
//! All `MulRecord`s from every phase are collected and fed into one call
//! to `dzkp_compute_batch` at the end.

use num_bigint::BigUint;
use vdoprf_field::Fp;
use vdoprf_ss::{DegenerateEncoding, SubsetFamily, SubsetT, RssShare, get_party_share};
use crate::double_rand::{generate_double_sharing_degenerate, DoubleShareLocal};
use vdoprf_network::CommStats;
use crate::dzkp::{dzkp_compute_batch, DzkpResult};
use crate::rss_mul::{deg_mul, MulRecord};
use crate::PreSharedMaterial;
use std::collections::BTreeMap;

/// Result of Approach II, dynamic-batched across any number of phases.
///
/// `result_shares[i]` holds the RSS shares of `α_i^{e_i}` for phase `i`.
/// `verdict` is the combined server-verified DZKP outcome covering every
/// RSS multiplication from every phase.
pub struct ApproachIIResult {
    pub result_shares: Vec<Vec<RssShare>>,
    pub verdict: DzkpResult,
    pub comm: CommStats,
}

/// Per-phase counter namespace. Phase `k` uses `PHASE_STRIDE * k + base`.
/// `gen_degenerate` only touches counters in ranges [1000, 1001] (α derive)
/// and [2000, 2000 + num_muls_per_phase]; 10_000 is plenty of headroom.
const PHASE_STRIDE: u64 = 10_000;
const COUNTER_M_VALUES: u64 = 1_000;
const COUNTER_TREE_BASE: u64 = 2_000;
/// Maximum number of counter offsets consulted when rejection-sampling away
/// a PRG output of zero. For cryptographic `p`, this is never hit.
const REJECTION_RETRY_BUDGET: usize = 256;

/// Execute Approach II for `N = exponents.len()` phases with a single
/// combined DZKP. Returns `(alpha_es, result)` where `alpha_es[i]` is the
/// plaintext `α_i^{e_i}` for phase `i` (used by tests to check correctness).
pub fn gen_degenerate(
    exponents: &[BigUint],
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Vec<Fp>, ApproachIIResult) {
    assert!(!exponents.is_empty(), "gen_degenerate requires at least one exponent");

    let n = family.n;
    let mut mul_records: Vec<MulRecord> = Vec::new();
    let mut comm = CommStats::default();
    let mut alpha_es: Vec<Fp> = Vec::with_capacity(exponents.len());
    let mut result_shares_per_phase: Vec<Vec<RssShare>> = Vec::with_capacity(exponents.len());

    for (phase_idx, e) in exponents.iter().enumerate() {
        let phase_off = PHASE_STRIDE * (phase_idx as u64);
        let mut counter = COUNTER_TREE_BASE + phase_off;
        // Each phase is its own Π_Gen^deg invocation. Only DZKP truly batches
        // (one proof, sublinear bytes); m parallel DegGen calls are not a
        // batch — rounds add across phases. Bytes still sum.
        let mut phase_comm = CommStats::default();

        // Step 1: PRG-derive m_T for each subset. Phase-offset the counter so
        // different phases get independent multiplicative shares. Rejection-
        // sample on m_T = 0 (matches approach_i.rs::find_nonzero_counter
        // pattern) to preserve the uniform distribution the paper assumes.
        let m_derive_counter_base = COUNTER_M_VALUES + phase_off;
        let mut m_values: BTreeMap<SubsetT, Fp> = BTreeMap::new();
        for subset in &family.subsets {
            let party = (0..n).find(|p| !subset.contains(p)).unwrap();
            let mut m_t = Fp::zero(modulus);
            for offset in 0..REJECTION_RETRY_BUDGET as u64 {
                let candidate = pre_shared[party].prf_keys[subset]
                    .evaluate(m_derive_counter_base + offset, modulus);
                if !candidate.is_zero() {
                    m_t = candidate;
                    break;
                }
            }
            assert!(!m_t.is_zero(), "rejection sampler exhausted budget for m_T");
            m_values.insert(subset.clone(), m_t);
        }

        let mut alpha = Fp::one(modulus);
        for v in m_values.values() {
            alpha = &alpha * v;
        }
        let alpha_e = alpha.pow(e);
        alpha_es.push(alpha_e.clone());

        // Step 2: Local exponentiation + bundling into degenerate encodings.
        let m_prime_encodings: Vec<DegenerateEncoding> = family
            .subsets
            .iter()
            .map(|subset| DegenerateEncoding {
                target_subset: subset.clone(),
                value: m_values[subset].pow(e),
            })
            .collect();

        // Step 3: sequential Π_DegMul fold (paper "Sequential multiplication"):
        //   ⟦q_1⟧ = ⟨M_{T_1}⟩_{T_1}
        //   ⟦q_j⟧ = ⟦q_{j-1}⟧ · ⟨M_{T_j}⟩_{T_j}   for j = 2..N
        // Every step's right operand is a degenerate encoding, so every step
        // is a Π_DegMul (full-RSS × degenerate → full-RSS) — never a full
        // Π_RSS.Mul. The N−1 multiplications form a dependency chain so we
        // `merge` (sum rounds) rather than `merge_parallel`. Per-phase comm:
        // 2(t+1)(N−1) field elements; per-record `party_pairs` size ≈ N
        // (vs N²/n for full RSS.Mul) — keeps `mul_records` small at large m.
        let mut current: Vec<RssShare> = {
            let sharing = m_prime_encodings[0].to_replicated_sharing(family, modulus);
            (0..n).map(|i| get_party_share(&sharing, i, family)).collect()
        };
        for j in 1..m_prime_encodings.len() {
            let encoding = &m_prime_encodings[j];
            let t_j = &encoding.target_subset;
            let double_shares: Vec<DoubleShareLocal> = (0..n)
                .map(|p| {
                    generate_double_sharing_degenerate(
                        p,
                        counter,
                        t_j,
                        &pre_shared[p],
                        family,
                        modulus,
                    )
                })
                .collect();
            counter += 1;

            let (next, record, mul_comm) =
                deg_mul(&current, encoding, &double_shares, family, modulus);
            phase_comm.merge(&mul_comm);
            mul_records.push(record);
            current = next;
        }
        let result_shares = current;
        result_shares_per_phase.push(result_shares);
        comm.merge_parallel(&phase_comm);
    }

    // Single server-verified DZKP over all phases' multiplications.
    let (verdict, dzkp_comm) = dzkp_compute_batch(&mul_records, family, modulus, pre_shared);
    comm.merge(&dzkp_comm);

    (
        alpha_es,
        ApproachIIResult {
            result_shares: result_shares_per_phase,
            verdict,
            comm,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_pre_shared;
    use vdoprf_ss::ReplicatedSharing;

    #[test]
    fn test_approach_ii_small_exponent() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (alpha_es, result) = gen_degenerate(&[BigUint::from(4u32)], &pre_shared, &family, &modulus);
        assert_eq!(result.verdict, DzkpResult::Accept);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    #[test]
    fn test_approach_ii_larger_n() {
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (alpha_es, result) = gen_degenerate(&[BigUint::from(4u32)], &pre_shared, &family, &modulus);
        assert_eq!(result.verdict, DzkpResult::Accept);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    #[test]
    fn test_approach_ii_n3_t1_e2() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (alpha_es, result) = gen_degenerate(&[BigUint::from(2u32)], &pre_shared, &family, &modulus);
        assert_eq!(result.verdict, DzkpResult::Accept);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    #[test]
    fn test_approach_ii_n3_t1_e1() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (alpha_es, result) = gen_degenerate(&[BigUint::from(1u32)], &pre_shared, &family, &modulus);
        assert_eq!(result.verdict, DzkpResult::Accept);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    #[test]
    fn test_approach_ii_non_power_of_2_e3() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (alpha_es, result) = gen_degenerate(&[BigUint::from(3u32)], &pre_shared, &family, &modulus);
        assert_eq!(result.verdict, DzkpResult::Accept);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    #[test]
    fn test_approach_ii_non_power_of_2_e5() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (alpha_es, result) = gen_degenerate(&[BigUint::from(5u32)], &pre_shared, &family, &modulus);
        assert_eq!(result.verdict, DzkpResult::Accept);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    #[test]
    fn test_approach_ii_n7_t3() {
        let n = 7;
        let t = 3;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (alpha_es, result) = gen_degenerate(&[BigUint::from(4u32)], &pre_shared, &family, &modulus);
        assert_eq!(result.verdict, DzkpResult::Accept);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    #[test]
    fn test_approach_ii_comm_nonzero() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (_, result) = gen_degenerate(&[BigUint::from(4u32)], &pre_shared, &family, &modulus);
        assert!(result.comm.total_bytes() > 0);
    }

    #[test]
    fn test_approach_ii_comm_scales_with_n() {
        let modulus = BigUint::from(113u32);
        let family_small = SubsetFamily::new(3, 1);
        let pre_shared_small = setup_pre_shared(3, 1, &modulus);
        let (_, result_small) =
            gen_degenerate(&[BigUint::from(4u32)], &pre_shared_small, &family_small, &modulus);

        let family_large = SubsetFamily::new(5, 2);
        let pre_shared_large = setup_pre_shared(5, 2, &modulus);
        let (_, result_large) =
            gen_degenerate(&[BigUint::from(4u32)], &pre_shared_large, &family_large, &modulus);

        assert!(result_large.comm.total_bytes() > result_small.comm.total_bytes());
    }

    #[test]
    fn test_approach_ii_larger_modulus() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (alpha_es, result) = gen_degenerate(&[BigUint::from(4u32)], &pre_shared, &family, &modulus);
        assert_eq!(result.verdict, DzkpResult::Accept);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    /// Stress: `N = 1` and `N = 100` offline phases both produce correct
    /// `α^e` per phase and an accepting combined DZKP.
    #[test]
    fn test_approach_ii_1_and_100_phases() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        for phase_count in [1usize, 100usize] {
            let exponents: Vec<BigUint> =
                (1..=phase_count).map(|i| BigUint::from(i as u32)).collect();
            let (alpha_es, result) = gen_degenerate(&exponents, &pre_shared, &family, &modulus);

            assert_eq!(alpha_es.len(), phase_count);
            assert_eq!(result.result_shares.len(), phase_count);
            assert_eq!(
                result.verdict,
                DzkpResult::Accept,
                "batched DZKP must accept at N = {phase_count}"
            );
            for i in 0..phase_count {
                let rec = ReplicatedSharing::reconstruct_from_party_shares(
                    &result.result_shares[i],
                    &modulus,
                );
                assert_eq!(
                    rec.value,
                    alpha_es[i].value,
                    "N={phase_count} phase {i} correctness"
                );
            }
        }
    }

    /// Per-phase offline rounds are constant in N (parallel composition);
    /// only the batched DZKP's recursion depth grows logarithmically in N
    /// because Boyle 3.3 charges one F_coin per fold step.
    #[test]
    fn test_approach_ii_rounds_grow_logarithmically_in_phases() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (_, r1) = gen_degenerate(&[BigUint::from(4u32)], &pre_shared, &family, &modulus);
        let exps: Vec<BigUint> = (0..100).map(|_| BigUint::from(4u32)).collect();
        let (_, r100) = gen_degenerate(&exps, &pre_shared, &family, &modulus);
        // Linear growth would give r100 ≈ 100 · r1; log growth keeps r100
        // under r1 + 2·log2(100·N) ≈ r1 + 18 for N ≤ 8.
        let log_bound = r1.comm.rounds + 20;
        assert!(
            r100.comm.rounds <= log_bound,
            "rounds must grow at most logarithmically across N (got {} at N=1 vs {} at N=100; bound {})",
            r1.comm.rounds, r100.comm.rounds, log_bound,
        );
    }

    /// Cross-phase batching: run 3 phases, one combined DZKP.
    #[test]
    fn test_approach_ii_cross_phase_batched() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let exponents = vec![BigUint::from(2u32), BigUint::from(3u32), BigUint::from(4u32)];
        let (alpha_es, result) = gen_degenerate(&exponents, &pre_shared, &family, &modulus);

        assert_eq!(alpha_es.len(), 3);
        assert_eq!(result.result_shares.len(), 3);
        assert_eq!(
            result.verdict,
            DzkpResult::Accept
        );

        for i in 0..3 {
            let rec = ReplicatedSharing::reconstruct_from_party_shares(
                &result.result_shares[i],
                &modulus,
            );
            assert_eq!(rec.value, alpha_es[i].value, "phase {i} correctness");
        }
    }

}
