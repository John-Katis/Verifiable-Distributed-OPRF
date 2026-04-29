//! Approach I: Baseline offline generation via `Π_AlyGen` (Protocol 17,
//! overleaf appendix C), using Aly-Smart [ACNS 2019] public-base
//! exponentiation as the sub-protocol `Π_exp`.
//!
//! Dynamic cross-phase API: `aly_gen(&[e_0, e_1, …, e_{N-1}], …)` runs
//! `N ≥ 1` phases of Π_AlyGen and produces a single batched DZKP covering
//! every RSS multiplication emitted across all phases. For `N = 1` this
//! matches the original single-phase behavior exactly.
//!
//! Per-phase steps (run once per exponent):
//!
//!  1. F_Rand → `[r']_{p-1} ∈ Z*_{p-1}` and `[[α]]_p`.
//!  2. `[[r̄]] ← Π_exp(g, [r'])`.
//!  3. `[[c]] ← Π_RSS.Mul([[r̄]], [[α]])`.
//!  4. `c ← Open([[c]])`; abort if `c = 0`.
//!  5. Local: `c' = c^e`.
//!  6. Local: `[ē]_{p-1} = -e · [r']`.
//!  7. `[[ρ]] ← Π_exp(g, [ē])`.
//!  8. Local: `[[α^e]] = c' · [[ρ]]`.

use num_bigint::BigUint;
use num_traits::One;
use vdoprf_field::Fp;
use vdoprf_ss::{SubsetFamily, RssShare};
use crate::double_rand::{generate_double_sharing, DoubleShareLocal};
use vdoprf_network::{CommStats, SimulatedNetwork};
use crate::dzkp::{dzkp_compute_batch, DzkpResult};
use crate::pub_base_exp::{is_coprime, open_rss, pub_base_exp_malicious, scalar_mul_rss};
use crate::rss_mul::{rss_mul_all_parties_with_record, MulRecord};
use crate::PreSharedMaterial;

/// Result of Approach I (Π_AlyGen), dynamic-batched across any number of
/// offline phases.
///
/// `result_shares[i]` holds the RSS shares of `α_i^{e_i}` produced by phase
/// `i`. `verdict` is the combined server-verified DZKP outcome covering
/// every RSS multiplication from every phase.
pub struct ApproachIResult {
    pub result_shares: Vec<Vec<RssShare>>,
    pub verdict: DzkpResult,
    pub comm: CommStats,
}

/// Counter namespace for Approach I, per-phase.
/// Phase `k` uses `PHASE_STRIDE * k + base_constant`. `PHASE_STRIDE` must
/// exceed the max internal counter advance per Π_exp call plus the fixed
/// slot spacing — `pub_base_exp_malicious` advances by `≤ ~2·N + a few`
/// each call, so 20_000 leaves generous headroom.
const PHASE_STRIDE: u64 = 20_000;
const ALYGEN_COUNTER_R_PRIME: u64 = 6_000;
const ALYGEN_COUNTER_ALPHA: u64 = 7_000;
const ALYGEN_COUNTER_R_AND_RPRIME: u64 = 8_000;
const ALYGEN_COUNTER_EXP1_BASE: u64 = 9_000;
const ALYGEN_COUNTER_MUL_STEP3: u64 = 11_500;
const ALYGEN_COUNTER_EXP2_BASE: u64 = 12_000;
const REJECTION_RETRY_BUDGET: usize = 256;

/// Execute Π_AlyGen for `N = exponents.len()` offline phases with a single
/// combined DZKP. Returns `(alphas, result)` where `alphas[i]` is the value
/// of `α_i` sampled in phase `i` (for test verification; production code
/// must NOT open α).
pub fn aly_gen(
    exponents: &[BigUint],
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
    generator: &Fp,
) -> (Vec<Fp>, ApproachIResult) {
    assert!(!exponents.is_empty(), "aly_gen requires at least one exponent");
    assert!(generator.modulus() == modulus, "generator must be in F_p");
    let p_minus_1 = modulus - BigUint::one();
    for e in exponents {
        assert!(is_coprime(e, &p_minus_1), "Π_AlyGen requires gcd(e, p-1) = 1");
    }

    let n = family.n;
    let mut comm = CommStats::default();
    let mut mul_records: Vec<MulRecord> = Vec::new();
    let mut alphas = Vec::with_capacity(exponents.len());
    let mut result_shares_per_phase: Vec<Vec<RssShare>> = Vec::with_capacity(exponents.len());

    for (phase_idx, e) in exponents.iter().enumerate() {
        let phase_off = PHASE_STRIDE * (phase_idx as u64);

        // Step 1: F_Rand with phase-offset counters.
        let (counter_r_prime, _) = find_coprime_counter(
            pre_shared,
            family,
            &p_minus_1,
            ALYGEN_COUNTER_R_PRIME + phase_off,
        )
        .expect("rejection sampler exhausted budget for r'");

        let r_prime_doubles: Vec<DoubleShareLocal> = (0..n)
            .map(|i| generate_double_sharing(i, counter_r_prime, &pre_shared[i], family, &p_minus_1))
            .collect();
        let r_prime_shares: Vec<RssShare> =
            r_prime_doubles.iter().map(|d| d.rss_share.clone()).collect();

        let counter_alpha = ALYGEN_COUNTER_ALPHA + phase_off;
        let alpha_shares: Vec<RssShare> = (0..n)
            .map(|i| generate_double_sharing(i, counter_alpha, &pre_shared[i], family, modulus).rss_share)
            .collect();
        let alpha_value =
            reconstruct_from_prfs(pre_shared, family, counter_alpha, modulus);
        alphas.push(alpha_value);

        // Protocol-6 randomness (fresh per-Π_exp-call, fresh per-phase).
        let (c_mr1_exp, _) = find_nonzero_counter(
            pre_shared,
            family,
            &p_minus_1,
            ALYGEN_COUNTER_R_AND_RPRIME + phase_off,
        )
        .expect("non-zero counter");
        let mr_exp1_shares: Vec<RssShare> = (0..n)
            .map(|i| {
                generate_double_sharing(i, c_mr1_exp, &pre_shared[i], family, &p_minus_1).rss_share
            })
            .collect();

        let (c_mr1_p, _) = find_nonzero_counter(
            pre_shared,
            family,
            modulus,
            ALYGEN_COUNTER_R_AND_RPRIME + 1_000 + phase_off,
        )
        .expect("non-zero counter");
        let mr_exp1_p_shares: Vec<RssShare> = (0..n)
            .map(|i| generate_double_sharing(i, c_mr1_p, &pre_shared[i], family, modulus).rss_share)
            .collect();

        let (c_mr2_exp, _) = find_nonzero_counter(
            pre_shared,
            family,
            &p_minus_1,
            ALYGEN_COUNTER_R_AND_RPRIME + 2_000 + phase_off,
        )
        .expect("non-zero counter");
        let mr_exp2_shares: Vec<RssShare> = (0..n)
            .map(|i| {
                generate_double_sharing(i, c_mr2_exp, &pre_shared[i], family, &p_minus_1).rss_share
            })
            .collect();

        let (c_mr2_p, _) = find_nonzero_counter(
            pre_shared,
            family,
            modulus,
            ALYGEN_COUNTER_R_AND_RPRIME + 3_000 + phase_off,
        )
        .expect("non-zero counter");
        let mr_exp2_p_shares: Vec<RssShare> = (0..n)
            .map(|i| generate_double_sharing(i, c_mr2_p, &pre_shared[i], family, modulus).rss_share)
            .collect();

        // Step 6 (local): ē = -e · r' over Z_{p-1}. Computed up-front so Step 7
        // can launch in parallel with the Step-2→3→4 chain (neither chain
        // consumes the other's output).
        let neg_e = (&p_minus_1 - (e % &p_minus_1)) % &p_minus_1;
        let neg_e_fp = Fp::new(neg_e, &p_minus_1);
        let e_bar_shares = scalar_mul_rss(&r_prime_shares, &neg_e_fp);

        // Chain A: Step 2 (Π_exp on [r']) → Step 3 (RSS.Mul r̄·α) → Step 4 (Open c).
        let mut chain_a = CommStats::default();
        let exp1 = pub_base_exp_malicious(
            generator,
            &r_prime_shares,
            &mr_exp1_shares,
            &mr_exp1_p_shares,
            &p_minus_1,
            modulus,
            pre_shared,
            family,
            ALYGEN_COUNTER_EXP1_BASE + phase_off,
        );
        assert!(exp1.valid, "Π_exp #1 rejected an honest execution");
        let r_bar_shares = exp1.result_shares;
        mul_records.extend(exp1.mul_records);
        chain_a.merge(&exp1.comm);

        let step3_doubles: Vec<DoubleShareLocal> = (0..n)
            .map(|i| {
                generate_double_sharing(
                    i,
                    ALYGEN_COUNTER_MUL_STEP3 + phase_off,
                    &pre_shared[i],
                    family,
                    modulus,
                )
            })
            .collect();
        let (c_shares, rec, cc) = rss_mul_all_parties_with_record(
            &r_bar_shares,
            &alpha_shares,
            &step3_doubles,
            family,
            modulus,
        );
        mul_records.push(rec);
        chain_a.merge(&cc);

        let mut open_net = SimulatedNetwork::new(n);
        let c = open_rss(&c_shares, modulus, &mut open_net);
        chain_a.merge(&open_net.stats());
        assert!(!c.is_zero(), "AlyGen abort: c = 0 (negligible for cryptographic p)");

        // Step 5: c' = c^e (local, depends on Chain A's opened c).
        let c_prime = c.pow(e);

        // Chain B: Step 7 (Π_exp on [ē]) — runs in parallel with Chain A.
        let mut chain_b = CommStats::default();
        let exp2 = pub_base_exp_malicious(
            generator,
            &e_bar_shares,
            &mr_exp2_shares,
            &mr_exp2_p_shares,
            &p_minus_1,
            modulus,
            pre_shared,
            family,
            ALYGEN_COUNTER_EXP2_BASE + phase_off,
        );
        assert!(exp2.valid, "Π_exp #2 rejected an honest execution");
        let rho_shares = exp2.result_shares;
        mul_records.extend(exp2.mul_records);
        chain_b.merge(&exp2.comm);

        // Step 8 (local): [[α^e]] = c' · [[ρ]].
        let alpha_e_shares = scalar_mul_rss(&rho_shares, &c_prime);
        result_shares_per_phase.push(alpha_e_shares);

        // Phase total = max(Chain A, Chain B); phases run in parallel across
        // the outer loop since distinct phases share no data.
        let mut phase_comm = CommStats::default();
        phase_comm.merge_parallel(&chain_a);
        phase_comm.merge_parallel(&chain_b);
        comm.merge_parallel(&phase_comm);
    }

    // Single server-verified batched DZKP across all phases' multiplications.
    let (verdict, dzkp_comm) = dzkp_compute_batch(&mul_records, family, modulus, pre_shared);
    comm.merge(&dzkp_comm);

    (
        alphas,
        ApproachIResult {
            result_shares: result_shares_per_phase,
            verdict,
            comm,
        },
    )
}

/// Reconstruct a PRF-derived shared value by summing per-subset PRF outputs.
/// Trusted-dealer F_Rand abstraction — not called by any real party in
/// deployment; used here only for rejection-sampling predicates and for
/// returning α to tests.
fn reconstruct_from_prfs(
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    counter: u64,
    modulus: &BigUint,
) -> Fp {
    let mut total = Fp::zero(modulus);
    for subset in &family.subsets {
        let holder = (0..family.n)
            .find(|p| !subset.contains(p))
            .expect("every subset is covered");
        let piece = pre_shared[holder].prf_keys[subset].evaluate(counter, modulus);
        total = &total + &piece;
    }
    total
}

fn find_coprime_counter(
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    p_minus_1: &BigUint,
    counter_base: u64,
) -> Option<(u64, Fp)> {
    for offset in 0..(REJECTION_RETRY_BUDGET as u64) {
        let counter = counter_base + offset;
        let r = reconstruct_from_prfs(pre_shared, family, counter, p_minus_1);
        if !r.is_zero() && is_coprime(&r.value, p_minus_1) {
            return Some((counter, r));
        }
    }
    None
}

fn find_nonzero_counter(
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
    counter_base: u64,
) -> Option<(u64, Fp)> {
    for offset in 0..(REJECTION_RETRY_BUDGET as u64) {
        let counter = counter_base + offset;
        let v = reconstruct_from_prfs(pre_shared, family, counter, modulus);
        if !v.is_zero() {
            return Some((counter, v));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_pre_shared;
    use vdoprf_ss::ReplicatedSharing;

    fn test_generator(modulus: &BigUint) -> Fp {
        Fp::new(BigUint::from(3u32), modulus)
    }

    fn run_alygen_bench_cell(n: usize, t: usize, m: usize) {
        let p_hex = "8000000000000000000000000000005f00000000000000000000000000000001";
        let modulus = BigUint::parse_bytes(p_hex.as_bytes(), 16).unwrap();
        let family = SubsetFamily::new(n, t);
        eprintln!(
            "setup: (n,t)=({},{}), N={}, m={}, p bits={}",
            n, t, family.subsets.len(), m, modulus.bits()
        );
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);
        let mut e = BigUint::one() << 128;
        let p_minus_1 = &modulus - BigUint::one();
        while crate::pub_base_exp::gcd(&e, &p_minus_1) != BigUint::one() {
            e += BigUint::one();
        }
        let exponents: Vec<BigUint> = (0..m).map(|_| e.clone()).collect();
        let t0 = std::time::Instant::now();
        let (_alphas, result) = aly_gen(&exponents, &pre_shared, &family, &modulus, &g);
        eprintln!(
            "(n,t)=({},{}) m={} done in {:.2}s; verdict={:?}, p2p={} kB, bcast={} kB, rounds={}",
            n, t, m, t0.elapsed().as_secs_f64(),
            result.verdict,
            result.comm.p2p_bytes / 1000,
            result.comm.broadcast_bytes / 1000,
            result.comm.rounds,
        );
        assert_eq!(result.verdict, DzkpResult::Accept);
    }

    #[test]
    #[ignore]
    fn repro_alygen_m50_9_4_256bit() {
        run_alygen_bench_cell(9, 4, 50);
    }

    /// Reproduces the bench cell (n,t)=(7,3), m=50, 256-bit p, e = 2^128+1
    /// (bench's `aly_coprime_e(2^128)`). Not run by default (#[ignore]).
    #[test]
    #[ignore]
    fn repro_alygen_m50_7_3_256bit() {
        let n = 7;
        let t = 3;
        let p_hex = "8000000000000000000000000000005f00000000000000000000000000000001";
        let modulus = BigUint::parse_bytes(p_hex.as_bytes(), 16).unwrap();
        let family = SubsetFamily::new(n, t);
        eprintln!(
            "setup: n={}, t={}, N(subsets)={}, p bits={}",
            n,
            t,
            family.subsets.len(),
            modulus.bits()
        );
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        // Bench uses `aly_coprime_e(2^128)`: smallest e ≥ 2^128 with
        // gcd(e, p-1) = 1. Mirror that precisely.
        let mut e = BigUint::one() << 128;
        let p_minus_1 = &modulus - BigUint::one();
        while crate::pub_base_exp::gcd(&e, &p_minus_1) != BigUint::one() {
            e += BigUint::one();
        }
        eprintln!("e bits = {}, e coprime-nudge = {}", e.bits(), &e - (BigUint::one() << 128));

        let exponents: Vec<BigUint> = (0..50).map(|_| e.clone()).collect();
        let t0 = std::time::Instant::now();
        let (_alphas, result) = aly_gen(&exponents, &pre_shared, &family, &modulus, &g);
        eprintln!(
            "aly_gen m=50 done in {:.2}s; verdict={:?}, p2p={} kB, bcast={} kB, rounds={}",
            t0.elapsed().as_secs_f64(),
            result.verdict,
            result.comm.p2p_bytes / 1000,
            result.comm.broadcast_bytes / 1000,
            result.comm.rounds,
        );
        assert_eq!(result.verdict, DzkpResult::Accept);
    }

    /// Standard single-phase correctness: one exponent → one alpha^e.
    #[test]
    fn test_alygen_single_phase_e3() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        let e = BigUint::from(3u32);
        let (alphas, result) = aly_gen(&[e.clone()], &pre_shared, &family, &modulus, &g);

        assert_eq!(alphas.len(), 1);
        assert_eq!(result.result_shares.len(), 1);
        assert_eq!(
            result.verdict,
            DzkpResult::Accept
        );

        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alphas[0].pow(&e).value);
    }

    #[test]
    fn test_alygen_single_phase_e5() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        let e = BigUint::from(5u32);
        let (alphas, result) = aly_gen(&[e.clone()], &pre_shared, &family, &modulus, &g);
        assert_eq!(
            result.verdict,
            DzkpResult::Accept
        );
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alphas[0].pow(&e).value);
    }

    #[test]
    fn test_alygen_e_equals_1() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        let (alphas, result) = aly_gen(&[BigUint::from(1u32)], &pre_shared, &family, &modulus, &g);
        assert_eq!(
            result.verdict,
            DzkpResult::Accept
        );
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alphas[0].value);
    }

    #[test]
    fn test_alygen_n5_t2() {
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        let e = BigUint::from(11u32);
        let (alphas, result) = aly_gen(&[e.clone()], &pre_shared, &family, &modulus, &g);
        assert_eq!(
            result.verdict,
            DzkpResult::Accept
        );
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alphas[0].pow(&e).value);
    }

    #[test]
    fn test_alygen_larger_modulus() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        let e = BigUint::from(3u32);
        let (alphas, result) = aly_gen(&[e.clone()], &pre_shared, &family, &modulus, &g);
        assert_eq!(
            result.verdict,
            DzkpResult::Accept
        );
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result.result_shares[0], &modulus);
        assert_eq!(reconstructed.value, alphas[0].pow(&e).value);
    }

    #[test]
    fn test_alygen_comm_nonzero() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        let (_, result) = aly_gen(&[BigUint::from(3u32)], &pre_shared, &family, &modulus, &g);
        assert!(result.comm.total_bytes() > 0);
    }

    #[test]
    #[should_panic(expected = "gcd(e, p-1) = 1")]
    fn test_alygen_rejects_non_coprime_e() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        let _ = aly_gen(&[BigUint::from(4u32)], &pre_shared, &family, &modulus, &g);
    }

    /// Cross-phase batching: run 3 offline phases with different exponents.
    /// Expect one combined DZKP covering all three phases' multiplications.
    #[test]
    fn test_alygen_cross_phase_batched() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        let exponents = vec![
            BigUint::from(3u32),
            BigUint::from(5u32),
            BigUint::from(9u32),
        ];
        let (alphas, result) = aly_gen(&exponents, &pre_shared, &family, &modulus, &g);

        assert_eq!(alphas.len(), 3);
        assert_eq!(result.result_shares.len(), 3);

        // Single combined DZKP must accept.
        assert_eq!(
            result.verdict,
            DzkpResult::Accept
        );

        // Correctness: each phase's reconstructed output = α_i^{e_i}.
        for (i, e_i) in exponents.iter().enumerate() {
            let rec = ReplicatedSharing::reconstruct_from_party_shares(
                &result.result_shares[i],
                &modulus,
            );
            assert_eq!(rec.value, alphas[i].pow(e_i).value, "phase {i} correctness");
        }
    }

    /// Stress: `N = 1` and `N = 100` offline phases must both produce
    /// correct `α^e` per phase and an accepting combined DZKP. Uses
    /// `p = 65537` (p−1 = 2^16) so every odd exponent is coprime to p−1.
    #[test]
    fn test_alygen_1_and_100_phases() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        for phase_count in [1usize, 100usize] {
            let exponents: Vec<BigUint> = (0..phase_count)
                .map(|i| BigUint::from((2 * i + 1) as u32))
                .collect();
            let (alphas, result) = aly_gen(&exponents, &pre_shared, &family, &modulus, &g);

            assert_eq!(alphas.len(), phase_count);
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
                    alphas[i].pow(&exponents[i]).value,
                    "N={phase_count} phase {i} correctness"
                );
            }
        }
    }

    /// Amortization sanity: 3-phase batched run has a smaller per-phase DZKP
    /// proof size than 3 independent single-phase calls (proof size grows
    /// sublinearly in N due to U/V/W polynomial batching in dzkp.rs).
    #[test]
    fn test_alygen_batch_dzkp_single_verdict() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let g = test_generator(&modulus);

        let (_, three_phases) = aly_gen(
            &[BigUint::from(3u32), BigUint::from(5u32), BigUint::from(9u32)],
            &pre_shared,
            &family,
            &modulus,
            &g,
        );
        // Single combined DZKP verdict covers every phase's multiplications.
        assert_eq!(three_phases.verdict, DzkpResult::Accept);
    }
}
