//! Approach III: Dual-Share ZKP (Π_Gen)
//!
//! Common protocol for both III-a (VOLEitH) and III-b (Ligero).
//! Uses t+1 designated dealers, each proving dual-share consistency.
//!
//! Steps:
//! 1. Each dealer i derives (m_T, a_T) pairs from PRG seeds.
//! 2. Each dealer computes M^i = Π_T m_T and A^i = Σ_T a_T.
//! 3. Each dealer broadcasts δ_i = M^i - A^i and proves correctness via ZKP.
//! 4. Every server independently re-runs the public verifier on each dealer
//!    proof; the combined verdict is the AND of all per-server, per-proof
//!    verifications plus the server-verified DZKP verdict.
//! 5. After verification, servers multiply all dealer contributions.

use num_bigint::BigUint;
use std::collections::BTreeMap;
use vdoprf_field::Fp;
use vdoprf_network::CommStats;
use vdoprf_ss::{SubsetFamily, SubsetT, RssShare};
use crate::double_rand::{generate_double_sharing, DoubleShareLocal};
use crate::dzkp::{dzkp_compute_batch, DzkpResult};
use crate::rss_mul::{rss_mul_all_parties_with_record, MulRecord};
use crate::zkp_vith::{self, VitHParams, ExtendedWitness};
use crate::zkp_ligero::{self, LigeroParams, WitnessMatrix};
use crate::PreSharedMaterial;

/// ZKP variant selection.
#[derive(Clone, Debug)]
pub enum ZkpVariant {
    VitH(VitHParams),
    Ligero(LigeroParams),
}

/// Result of Approach III across any number of offline phases.
///
/// `result_shares[k]` holds the RSS shares of `α_k^{e_k}` produced by phase
/// `k`. `verdict` is the combined server-verified outcome: every dealer's
/// VitH or Ligero proof is re-run by every server, and the cross-phase DZKP
/// is likewise server-verified. All must accept for `verdict = Accept`.
pub struct ApproachIIIResult {
    pub result_shares: Vec<Vec<RssShare>>,
    pub verdict: DzkpResult,
    pub comm: CommStats,
}

/// Per-phase counter namespace. Phase `k` uses `PHASE_STRIDE * k + base`.
/// `10_000` leaves generous headroom around the 3000/4000 m/a PRF slots and
/// the 5000 tree-multiplication base.
const PHASE_STRIDE: u64 = 10_000;
const COUNTER_M_BASE: u64 = 3_000;
const COUNTER_A_BASE: u64 = 4_000;
const COUNTER_TREE_BASE: u64 = 5_000;
/// Rejection-sampling retry budget for `m_T = 0`. Matches the pattern in
/// `approach_i.rs::find_nonzero_counter`.
const REJECTION_RETRY_BUDGET: usize = 256;

/// Execute Approach III for `N = exponents.len()` phases. Produces per-phase
/// `α^e` outputs, server-verifies every dealer proof (VitH per-dealer or one
/// batched Π_Lig covering every dealer in every phase), and server-verifies
/// the single combined DZKP covering every binary-tree RSS multiplication.
pub fn gen_zkp(
    exponents: &[BigUint],
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
    variant: &ZkpVariant,
) -> (Vec<Fp>, ApproachIIIResult) {
    assert!(!exponents.is_empty(), "gen_zkp requires at least one exponent");

    let n = family.n;
    let t = family.t;
    let num_dealers = t + 1;
    let feb = ((modulus.bits() + 7) / 8) as usize;
    let mut zkp_net = vdoprf_network::SimulatedNetwork::new(n);
    let dealers: Vec<usize> = (0..num_dealers).collect();

    let mut alpha_es: Vec<Fp> = Vec::with_capacity(exponents.len());
    let mut result_shares_per_phase: Vec<Vec<RssShare>> = Vec::with_capacity(exponents.len());
    // Accumulated across all phases for cross-phase Ligero batching.
    let mut all_ligero_witnesses: Vec<WitnessMatrix> = Vec::new();
    let mut all_ligero_deltas: Vec<Fp> = Vec::new();
    let mut all_mul_records: Vec<MulRecord> = Vec::new();
    let mut tree_counter = COUNTER_TREE_BASE;
    // Tree-multiplication rounds for phase k accumulate into `tree_phase_comm`
    // and fold into `tree_comm` with `merge_parallel` — phases share no data,
    // so under parallel composition the wall-clock round count is max over
    // phases, not sum.
    let mut tree_comm = CommStats::default();
    // Combined verdict: AND of every dealer-proof re-verification across
    // every server and the final DZKP verdict.
    let mut verdict = DzkpResult::Accept;

    for (phase_idx, e) in exponents.iter().enumerate() {
        let phase_off = PHASE_STRIDE * (phase_idx as u64);
        let mut dealer_products: Vec<Fp> = Vec::with_capacity(num_dealers);
        let mut dealer_sharings = Vec::with_capacity(num_dealers);

        for &dealer_id in &dealers {
            // Step 1: derive (m_T, a_T) for every subset T in the dealer's
            // RSS view (T̄ = [n] \ T contains dealer, i.e., dealer ∉ T). The
            // paper's "N pairs" (§3a L53) reduces to (n-1 choose t) real
            // wires per dealer under the standard RSS PRF schema — the
            // remaining t subsets are held by disjoint parties that do not
            // share a seed with the dealer, so no real witness exists for
            // them. Rejection-sample m_T = 0 to keep the distribution uniform.
            let dealer_subsets: Vec<SubsetT> = family
                .subsets_not_containing(dealer_id)
                .into_iter()
                .cloned()
                .collect();

            let m_counter = COUNTER_M_BASE + dealer_id as u64 + phase_off;
            let a_counter = COUNTER_A_BASE + dealer_id as u64 + phase_off;

            let mut m_values: Vec<Fp> = Vec::with_capacity(dealer_subsets.len());
            let mut a_values: Vec<Fp> = Vec::with_capacity(dealer_subsets.len());
            for subset in &dealer_subsets {
                let prf = pre_shared[dealer_id]
                    .prf_keys
                    .get(subset)
                    .expect("dealer holds PRF key for every subset in its RSS view");
                let mut m_t = Fp::zero(modulus);
                for offset in 0..REJECTION_RETRY_BUDGET as u64 {
                    let cand = prf.evaluate(m_counter + offset, modulus);
                    if !cand.is_zero() {
                        m_t = cand;
                        break;
                    }
                }
                assert!(!m_t.is_zero(), "rejection sampler exhausted budget for m_T");
                m_values.push(m_t.pow(e));
                a_values.push(prf.evaluate(a_counter, modulus));
            }

            // Step 2-4: M^i, A^i, δ_i.
            let mut m_product = Fp::one(modulus);
            for m in &m_values {
                m_product = &m_product * m;
            }
            let mut a_sum = Fp::zero(modulus);
            for a in &a_values {
                a_sum = &a_sum + a;
            }
            let delta = &m_product - &a_sum;
            zkp_net.broadcast(dealer_id, delta.value.to_bytes_be());

            // Step 5: ZKP — VitH proves + is re-verified by every server here;
            // Ligero accumulates witnesses for a single batched proof covering
            // all phases, verified once at the end.
            match variant {
                ZkpVariant::VitH(params) => {
                    let witness = ExtendedWitness::new(m_values.clone(), a_values.clone(), modulus);
                    let proof = zkp_vith::vith_prove(&witness, &delta, params, modulus);

                    let mut broadcast_data = Vec::new();
                    broadcast_data.extend_from_slice(&proof.commitment);
                    for rep in 0..params.repetitions {
                        for mw in &proof.masked_witnesses[rep] {
                            broadcast_data.extend_from_slice(&mw.value.to_bytes_be());
                        }
                        let (ref a0t, ref a1t) = proof.check_values[rep];
                        broadcast_data.extend_from_slice(&a0t.value.to_bytes_be());
                        broadcast_data.extend_from_slice(&a1t.value.to_bytes_be());
                    }
                    for copath in &proof.copaths {
                        for seed in copath {
                            broadcast_data.extend_from_slice(seed);
                        }
                    }
                    zkp_net.broadcast(dealer_id, broadcast_data);

                    let r = params.repetitions;
                    let log_tau = (params.tau as f64).log2().ceil() as usize;
                    for v in 0..n {
                        if v != dealer_id {
                            let path_data = vec![0u8; r * log_tau * 32];
                            zkp_net.send_p2p(dealer_id, v, path_data);
                        }
                    }

                    // Every server runs vith_verify independently in a real
                    // deployment. vith_verify is deterministic in (proof,
                    // delta, params), so all honest servers reach the same
                    // verdict. Simulate once here — the wall-clock of a single
                    // call equals the per-server CPU cost under the paper's
                    // parallelism model (all servers verify concurrently).
                    let local_shares: BTreeMap<SubsetT, (Fp, Fp)> = BTreeMap::new();
                    if !zkp_vith::vith_verify(&proof, &delta, &local_shares, params, modulus) {
                        verdict = DzkpResult::Abort;
                    }
                }
                ZkpVariant::Ligero(_) => {
                    let witness = WitnessMatrix::new(m_values.clone(), a_values.clone(), modulus);
                    all_ligero_witnesses.push(witness);
                    all_ligero_deltas.push(delta.clone());
                }
            }

            dealer_products.push(m_product.clone());

            // Step 3 of Protocol 12: construct ⟨M^i⟩ from the a_T witness and
            // the broadcast δ. Every party S_j, for each subset T in j's RSS
            // view, sets its additive component for T to a_T when dealer ∉ T
            // (every honest party in T̄ derives the same a_T from the shared
            // PRF key). When dealer ∈ T, S_j has no witness for T and sets
            // its component to 0. A single canonical slot T* (the first
            // subset in the dealer's view) carries the δ correction so that
            //   Σ_T component_T = Σ_{dealer∉T} a_T + δ = A^i + δ = M^i.
            // This ties the RSS shares the downstream tree-mul consumes
            // directly to the witness the ZKP proves — a malicious dealer
            // cannot emit an arbitrary δ + RSS unrelated to {a_T}.
            let t_star: SubsetT = dealer_subsets
                .first()
                .expect("dealer has at least one subset in its view")
                .clone();
            let dealer_shares: Vec<RssShare> = (0..n)
                .map(|party_j| {
                    let mut shares: BTreeMap<SubsetT, Fp> = BTreeMap::new();
                    for subset in family.subsets_not_containing(party_j) {
                        let base = if subset.contains(&dealer_id) {
                            Fp::zero(modulus)
                        } else {
                            // Party j in T̄ and dealer in T̄ both share the
                            // PRF key for T; they derive the same a_T.
                            pre_shared[party_j].prf_keys[subset].evaluate(a_counter, modulus)
                        };
                        let val = if subset == &t_star {
                            &base + &delta
                        } else {
                            base
                        };
                        shares.insert(subset.clone(), val);
                    }
                    RssShare { party_id: party_j, shares }
                })
                .collect();

            // Sanity check in debug: the honest-dealer construction must
            // reconstruct to M^i (every a_T matches between the dealer and
            // other holders by PRF determinism). Catches bugs in `a_counter`
            // plumbing or canonical-slot choice.
            debug_assert_eq!(
                vdoprf_ss::ReplicatedSharing::reconstruct_from_party_shares(
                    &dealer_shares,
                    modulus,
                ),
                m_product,
                "honest ⟨M^i⟩ construction must reconstruct to M^i"
            );

            dealer_sharings.push(dealer_shares);
        }

        // Step 6 (per phase): binary-tree multiplication of dealer contributions.
        let mut current_layer: Vec<Vec<RssShare>> = dealer_sharings;
        let mut tree_phase_comm = CommStats::default();
        while current_layer.len() > 1 {
            let mut next_layer = Vec::new();
            let mut layer_comm = CommStats::default();
            let mut i = 0;
            while i + 1 < current_layer.len() {
                let double_shares: Vec<DoubleShareLocal> = (0..n)
                    .map(|p| generate_double_sharing(p, tree_counter, &pre_shared[p], family, modulus))
                    .collect();
                tree_counter += 1;
                let (result, record, mul_comm) = rss_mul_all_parties_with_record(
                    &current_layer[i],
                    &current_layer[i + 1],
                    &double_shares,
                    family,
                    modulus,
                );
                layer_comm.merge_parallel(&mul_comm);
                all_mul_records.push(record);
                next_layer.push(result);
                i += 2;
            }
            if i < current_layer.len() {
                next_layer.push(current_layer[i].clone());
            }
            tree_phase_comm.merge(&layer_comm);
            current_layer = next_layer;
        }
        let phase_result = current_layer.into_iter().next().unwrap();
        result_shares_per_phase.push(phase_result);
        tree_comm.merge_parallel(&tree_phase_comm);

        let mut phase_alpha_e = Fp::one(modulus);
        for mp in &dealer_products {
            phase_alpha_e = &phase_alpha_e * mp;
        }
        alpha_es.push(phase_alpha_e);
    }

    // After all phases: for Ligero, produce ONE batched proof over all
    // N × (t+1) dealer instances, then re-verify at every server.
    if let ZkpVariant::Ligero(params) = variant {
        let witness_refs: Vec<&WitnessMatrix> = all_ligero_witnesses.iter().collect();
        let proof = zkp_ligero::ligero_prove(&witness_refs, &all_ligero_deltas, params, modulus);

        let mut broadcast_data = Vec::new();
        broadcast_data.extend_from_slice(&proof.rt_w);
        broadcast_data.extend_from_slice(&proof.rt_c);
        broadcast_data.extend_from_slice(&proof.rt_u);
        for col in &proof.opened_columns_j {
            for fe in col {
                broadcast_data.extend_from_slice(&fe.value.to_bytes_be());
            }
        }
        for col in &proof.opened_columns_next {
            for fe in col {
                broadcast_data.extend_from_slice(&fe.value.to_bytes_be());
            }
        }
        for fe in &proof.opened_composition {
            broadcast_data.extend_from_slice(&fe.value.to_bytes_be());
        }
        for fe in &proof.consistency_codeword {
            broadcast_data.extend_from_slice(&fe.value.to_bytes_be());
        }
        for path in &proof.merkle_paths_j {
            for hash in path {
                broadcast_data.extend_from_slice(hash);
            }
        }
        for path in &proof.merkle_paths_next {
            for hash in path {
                broadcast_data.extend_from_slice(hash);
            }
        }
        for path in &proof.composition_merkle_paths {
            for hash in path {
                broadcast_data.extend_from_slice(hash);
            }
        }
        for h in &proof.witness_private_hashes {
            broadcast_data.extend_from_slice(h);
        }
        for path in &proof.witness_merkle_paths {
            for hash in path {
                broadcast_data.extend_from_slice(hash);
            }
        }
        zkp_net.broadcast(0, broadcast_data);

        let num_queries = params.num_queries;
        for v in 0..n {
            if v != 0 {
                let partial_data = vec![0u8; num_queries * 2 * feb];
                zkp_net.send_p2p(0, v, partial_data);
            }
        }

        // Every server runs ligero_verify independently in a real deployment.
        // Stages 1–3 are deterministic in (proof, deltas, params); one
        // simulated call equals the per-server CPU cost under parallelism.
        if !zkp_ligero::ligero_verify(&proof, &all_ligero_deltas, params, modulus) {
            verdict = DzkpResult::Abort;
        }
    }

    // Aggregate ZKP + phase communication + single server-verified DZKP.
    let mut comm = zkp_net.stats();
    comm.merge(&tree_comm);
    let (dzkp_verdict, dzkp_comm) = dzkp_compute_batch(&all_mul_records, family, modulus, pre_shared);
    comm.merge(&dzkp_comm);
    if dzkp_verdict != DzkpResult::Accept {
        verdict = DzkpResult::Abort;
    }

    (
        alpha_es,
        ApproachIIIResult {
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
    fn test_approach_iii_vith() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let e = BigUint::from(4u32);
        let variant = ZkpVariant::VitH(VitHParams::new(4, 8));
        let (alpha_es, result) = gen_zkp(&[e], &pre_shared, &family, &modulus, &variant);

        assert_eq!(result.verdict, DzkpResult::Accept);

        let reconstructed = ReplicatedSharing::reconstruct_from_party_shares(
            &result.result_shares[0], &modulus,
        );
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    #[test]
    fn test_approach_iii_ligero() {
        // Ligero's ZK bound `|Q| ≤ ⌊(N−1)/2⌋` requires per-dealer wire count
        // N ≥ 3 (zkp_ligero::LigeroParams::try_new L97–99). After Fix C the
        // dealer wire count is (n−1 choose t), so the smallest feasible
        // configuration is (n=5, t=2) → (4 choose 2) = 6 wires per dealer.
        // This matches the paper's own stance (§3-Offline.tex L79: Ligero
        // skipped for small N).
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let e = BigUint::from(4u32);
        let n_k = family.subsets_not_containing(0).len();
        let variant = ZkpVariant::Ligero(LigeroParams::new(n_k, 8, &modulus));
        let (alpha_es, result) = gen_zkp(&[e], &pre_shared, &family, &modulus, &variant);

        assert_eq!(result.verdict, DzkpResult::Accept);

        let reconstructed = ReplicatedSharing::reconstruct_from_party_shares(
            &result.result_shares[0], &modulus,
        );
        assert_eq!(reconstructed.value, alpha_es[0].value);
    }

    /// Tree-mul rounds fold with `merge_parallel` (phases are independent),
    /// but the batched DZKP's recursion depth grows logarithmically in N
    /// because Boyle 3.3 charges one F_coin per fold step.
    #[test]
    fn test_approach_iii_vith_rounds_grow_logarithmically_in_phases() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let variant = ZkpVariant::VitH(VitHParams::new(4, 8));

        let (_, r1) = gen_zkp(&[BigUint::from(2u32)], &pre_shared, &family, &modulus, &variant);
        let exps: Vec<BigUint> = (0..100).map(|_| BigUint::from(2u32)).collect();
        let (_, r100) = gen_zkp(&exps, &pre_shared, &family, &modulus, &variant);
        let log_bound = r1.comm.rounds + 20;
        assert!(
            r100.comm.rounds <= log_bound,
            "rounds must grow at most logarithmically across N (got {} at N=1 vs {} at N=100; bound {})",
            r1.comm.rounds, r100.comm.rounds, log_bound,
        );
    }

    /// Cross-phase batching (VitH variant): 3 offline phases, combined DZKP.
    #[test]
    fn test_approach_iii_vith_cross_phase() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let variant = ZkpVariant::VitH(VitHParams::new(4, 8));

        let exponents = vec![BigUint::from(2u32), BigUint::from(3u32), BigUint::from(5u32)];
        let (alpha_es, result) = gen_zkp(&exponents, &pre_shared, &family, &modulus, &variant);
        assert_eq!(alpha_es.len(), 3);
        assert_eq!(result.result_shares.len(), 3);
        assert_eq!(result.verdict, DzkpResult::Accept);
        for i in 0..3 {
            let rec = ReplicatedSharing::reconstruct_from_party_shares(
                &result.result_shares[i],
                &modulus,
            );
            assert_eq!(rec.value, alpha_es[i].value, "phase {i}");
        }
    }

    /// Cross-phase batching (Ligero variant): 3 offline phases, ONE combined
    /// Π_Lig proof covering all 3 × (t+1) dealer instances.
    #[test]
    fn test_approach_iii_ligero_cross_phase() {
        // See test_approach_iii_ligero comment on (n, t) choice.
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let n_k = family.subsets_not_containing(0).len();
        let variant = ZkpVariant::Ligero(LigeroParams::new(n_k, 8, &modulus));

        let exponents = vec![BigUint::from(2u32), BigUint::from(3u32), BigUint::from(5u32)];
        let (alpha_es, result) = gen_zkp(&exponents, &pre_shared, &family, &modulus, &variant);
        assert_eq!(alpha_es.len(), 3);
        assert_eq!(result.result_shares.len(), 3);
        assert_eq!(result.verdict, DzkpResult::Accept);
        for i in 0..3 {
            let rec = ReplicatedSharing::reconstruct_from_party_shares(
                &result.result_shares[i],
                &modulus,
            );
            assert_eq!(rec.value, alpha_es[i].value, "phase {i}");
        }
    }

    /// Stress (Ligero variant): `N = 1` and `N = 100` offline phases. For
    /// `N = 100` the single batched `Π_Lig` proof covers `100 × (t+1) = 200`
    /// dealer instances.
    #[test]
    fn test_approach_iii_1_and_100_phases_ligero() {
        // See test_approach_iii_ligero comment on (n, t) choice.
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let n_k = family.subsets_not_containing(0).len();
        let variant = ZkpVariant::Ligero(LigeroParams::new(n_k, 8, &modulus));

        for phase_count in [1usize, 100usize] {
            let exponents: Vec<BigUint> =
                (1..=phase_count).map(|i| BigUint::from(i as u32)).collect();
            let (alpha_es, result) = gen_zkp(&exponents, &pre_shared, &family, &modulus, &variant);

            assert_eq!(alpha_es.len(), phase_count);
            assert_eq!(result.result_shares.len(), phase_count);
            assert_eq!(
                result.verdict,
                DzkpResult::Accept,
                "batched Ligero + DZKP must accept at N = {phase_count}"
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

    /// Stress (VitH variant): `N = 1` and `N = 100` offline phases. VitH has
    /// no cross-instance batching, so there are `N × (t+1)` independent
    /// proofs — the user accepts that scaling cost.
    #[test]
    fn test_approach_iii_1_and_100_phases_vith() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let variant = ZkpVariant::VitH(VitHParams::new(4, 8));

        for phase_count in [1usize, 100usize] {
            let exponents: Vec<BigUint> =
                (1..=phase_count).map(|i| BigUint::from(i as u32)).collect();
            let (alpha_es, result) = gen_zkp(&exponents, &pre_shared, &family, &modulus, &variant);

            assert_eq!(alpha_es.len(), phase_count);
            assert_eq!(result.result_shares.len(), phase_count);
            assert_eq!(
                result.verdict,
                DzkpResult::Accept,
                "all VitH proofs and the combined DZKP must accept at N = {phase_count}"
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
}
