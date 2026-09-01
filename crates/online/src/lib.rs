//! Online phase of the v-dOPRF protocol.
//!
//! Given preprocessed `rss{k}` (PRF key share) and `rss{α_k^e}` (one per
//! client input, from `F_RandExp` in the §4 protocol) produced by the
//! offline phase, and a batch of client inputs `x_1, …, x_m ∈ F_p`, the
//! servers jointly compute `c_j = (x_j + k) · α_j^e` for each input and
//! open the results to the client. Correctness is proven via the **Π_VIP**
//! (Verifiable Inner Product) primitive in [`vip`], which unifies
//! multiplication and opening.
//!
//! Three variants match the paper's Section 4 performance table:
//! - [`compute::compute`]: sequential per-input `Π_VIP^Prl` (one proof per
//!   input, rounds stack).
//! - [`compute_parallel::compute_parallel`]: same but accounted as parallel
//!   network composition (rounds = max, bytes sum).
//! - [`compute_batch::compute_batch`]: full `Π_dVOPRF` — zero-sharing +
//!   commit-then-hash + ε_k aggregation → **one** `Π_VIP^Prl` over the
//!   aggregated relation. Paper Protocol `fig:doprf_protocol`.

pub mod compute;
pub mod compute_batch;
pub mod compute_boyle;
pub mod compute_parallel;
pub mod input;
pub mod vip;

use num_bigint::BigUint;
use vdoprf_field::Fp;
use vdoprf_network::CommStats;
use vdoprf_ss::{
    cross_multiply_with_pairs_and_subsets, get_party_share, share, ReplicatedSharing, RssShare,
    SubsetFamily, SubsetT,
};

use crate::vip::VipParallelOutput;

/// Material produced by the offline phase, fed into the online phase.
///
/// Paper §4-Online line 253: "Servers invoke F_RandExp and F_Zero a total
/// of `m` times". We carry one `α_k^e` sharing per client input. The
/// `k_sharing` is the long-lived PRF key share.
///
/// Benches simulate these with fresh random sharings (we do NOT run the
/// offline phase); for the bench the random fill is indistinguishable from
/// honest offline output.
#[derive(Clone, Debug)]
pub struct OnlinePreprocessed {
    /// Replicated sharing of the VDOPRF key `k`.
    pub k_sharing: ReplicatedSharing,
    /// Per-input replicated sharings of `α_k^e` (from F_RandExp; one per
    /// client input). For backward-compatible single-α setups the bench
    /// can replicate one sharing across all m entries.
    pub alpha_e_sharings: Vec<ReplicatedSharing>,
}

/// What the servers send to the client so the client can verify.
///
/// - `Components(Vec<VipParallelOutput>)`: one VIP^Prl output per client
///   input. Used by `compute` and `compute_parallel`. For `m = 1` the vec
///   has length 1.
/// - `Batched(VipDvoprfOutput)`: full Π_dVOPRF output — one VIP^Prl over
///   the aggregated relation plus the per-server `ṽ_i^{(k)}` additive
///   openings that let the client reconstruct `v^{(k)}` = Σ_i ṽ_i^{(k)}.
#[derive(Debug)]
pub enum OnlineProof {
    Components(Vec<VipParallelOutput>),
    Batched(VipDvoprfOutput),
}

/// Output of Π_dVOPRF (compute_batch): the single aggregated VIP^Prl proof
/// plus the per-server additive openings `ṽ_i^{(k)}` that the client sums
/// to recover `v^{(k)}`.
#[derive(Clone, Debug)]
pub struct VipDvoprfOutput {
    pub vip: VipParallelOutput,
    /// `tilde_v[i][k]` = server i's ṽ for input k.
    pub tilde_v: Vec<Vec<Fp>>,
}

/// Output of one online-phase invocation over `m` client inputs.
#[derive(Debug)]
pub struct OnlineResult {
    /// Reconstructed client outputs `c_j = (x_j + k) · α_j^e`.
    pub client_outputs: Vec<Fp>,
    /// Proof artifacts the client will verify.
    pub proof: OnlineProof,
    /// Aggregated server↔server + server→client communication.
    pub comm: CommStats,
}

/// Generate random preprocessed material — used by benches that don't actually
/// run the offline phase. `m` fresh α^e sharings are produced. For
/// backward compatibility a single-α helper is also exposed.
pub fn setup_random_preprocessed_m(
    m: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
) -> OnlinePreprocessed {
    let mut rng = rand::thread_rng();
    let k = Fp::random(modulus, &mut rng);
    let alpha_e_sharings: Vec<ReplicatedSharing> = (0..m.max(1))
        .map(|_| {
            let alpha_e = Fp::random_nonzero(modulus, &mut rng);
            share(&alpha_e, family, modulus, &mut rng)
        })
        .collect();
    OnlinePreprocessed {
        k_sharing: share(&k, family, modulus, &mut rng),
        alpha_e_sharings,
    }
}

/// Legacy helper: single-α preprocessed material. Kept so the end-to-end
/// bench can install a single `α^e` sharing from an offline run and have
/// the online helpers replicate it across all `m` inputs.
pub fn setup_random_preprocessed(
    family: &SubsetFamily,
    modulus: &BigUint,
) -> OnlinePreprocessed {
    setup_random_preprocessed_m(1, family, modulus)
}

/// Build an `OnlinePreprocessed` that reuses a single `α^e` sharing across
/// all `m` inputs. Used by the end-to-end bench, which runs one offline
/// α^e generation and then fans it out across the batched online queries.
pub fn preprocessed_from_single_alpha(
    k_sharing: ReplicatedSharing,
    alpha_e_sharing: ReplicatedSharing,
    m: usize,
) -> OnlinePreprocessed {
    OnlinePreprocessed {
        k_sharing,
        alpha_e_sharings: vec![alpha_e_sharing; m.max(1)],
    }
}

/// Per-input data extracted from the client's VSS input and the offline
/// preprocessed material — the cross-product pairs each server must feed
/// into VIP.
///
/// Per §5-Online.tex:242 ("Servers locally obtain ⟦a_i^(j,ℓ)⟧ and
/// ⟦b_i^(j,ℓ)⟧ via local share conversion"), each plaintext additive-share
/// pair `(addss{a}_u, addss{b}_v)` is conceptually re-encoded as a
/// degenerate RSS sharing whose only nonzero subset is the originating
/// `T_u` (resp. `T_v`). We do not materialise the full per-pair sharings
/// here; storing the originating subsets is enough for `vip_single` to
/// reconstruct the residual `(a^{(1)}, b^{(1)})` shares directly from the
/// fold weights at the end of the recursion. This keeps the working set
/// at O(N²·m) field elements (the actual information content) instead of
/// O(N²·m·n·S) — paper §5-Online line 232 — see `vip_single` for the
/// weighted-residual construction.
#[derive(Clone, Debug)]
pub(crate) struct PerInputPairs {
    /// `party_pairs[i]` = prover i's assigned (a_u, b_v) additive cross
    /// products for this input. Plaintext known to the prover; used by the
    /// prover to compute `q(1), q(2), q(3)` per VIP iteration.
    pub party_pairs: Vec<Vec<(Fp, Fp)>>,
    /// `party_targets[i][ℓ] = (T_u, T_v)`: originating subsets of the ℓ-th
    /// pair's `a_val` / `b_val` for prover i. The planned `vip_single`
    /// refactor will use these (instead of materialised RSS sharings) to
    /// reconstruct the residual a^{(1)}/b^{(1)} from fold weights at the
    /// end of the recursion — see the docstring above.
    pub party_targets: Vec<Vec<(SubsetT, SubsetT)>>,
    /// Per-party additive-share `v_i = Σ a·b` (local cross-product sum).
    pub party_cp: Vec<Fp>,
}

/// Driver: share each `x_j`, locally add the key, and extract the per-party
/// cross-product pairs that VIP will consume. **No RSS multiplication** —
/// VIP combines multiplication and opening. The only round-charging comm
/// in this stage is the client→servers VSS input distribution, which the
/// bench charges separately via `client_to_servers_input`; we return
/// `CommStats::default()` here.
///
/// Unverified client-input path: the client is treated as an ad-hoc local
/// dealer (`vdoprf_ss::share`) with no consistency check — see
/// [`crate::input`] for the verifiable replacement (`Π_Input`, Protocol 9),
/// used by `compute_batch::compute_batch_with_verified_input`. This
/// function is left as-is deliberately, so `compute_batch::compute_batch`
/// remains available as the "unverified input" comparison baseline.
pub(crate) fn share_add_and_extract_pairs(
    xs: &[Fp],
    pre: &OnlinePreprocessed,
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Vec<PerInputPairs>, CommStats) {
    let n = family.n;
    let mut rng = rand::thread_rng();

    let x_shares_per_input: Vec<Vec<RssShare>> = xs
        .iter()
        .map(|x| {
            // Client VSS-shares x_j; each server pulls its piece.
            let x_sharing = share(x, family, modulus, &mut rng);
            (0..n)
                .map(|i| get_party_share(&x_sharing, i, family))
                .collect()
        })
        .collect();

    (
        extract_pairs_from_x_shares(&x_shares_per_input, pre, family, modulus),
        CommStats::default(),
    )
}

/// Given each input's already-obtained per-party RSS shares of `x_j`
/// (however they were produced — the unverified `share()` dealer above, or
/// `Π_Input`'s verifiable mask-and-broadcast in [`crate::input`]), add the
/// key and extract the per-party cross-product pairs VIP will consume.
/// Factored out of `share_add_and_extract_pairs` so both client-input
/// paths share this identical downstream logic.
pub(crate) fn extract_pairs_from_x_shares(
    x_shares_per_input: &[Vec<RssShare>],
    pre: &OnlinePreprocessed,
    family: &SubsetFamily,
    _modulus: &BigUint,
) -> Vec<PerInputPairs> {
    let n = family.n;

    let k_shares: Vec<RssShare> = (0..n)
        .map(|i| get_party_share(&pre.k_sharing, i, family))
        .collect();
    assert!(
        !pre.alpha_e_sharings.is_empty(),
        "OnlinePreprocessed must carry at least one α^e sharing"
    );

    let mut out: Vec<PerInputPairs> = Vec::with_capacity(x_shares_per_input.len());

    for (j, x_party_shares) in x_shares_per_input.iter().enumerate() {
        let alpha_sharing = &pre.alpha_e_sharings[j % pre.alpha_e_sharings.len()];
        let alpha_party_shares: Vec<RssShare> = (0..n)
            .map(|i| get_party_share(alpha_sharing, i, family))
            .collect();

        // [a_j] = [x_j + k]  (local RSS addition).
        let a_shares: Vec<RssShare> = (0..n)
            .map(|i| x_party_shares[i].local_add(&k_shares[i]))
            .collect();

        // Extract per-party cross-product pairs and local sum. The originating
        // subsets `(T_u, T_v)` are kept alongside each `(a_val, b_val)` so
        // `vip_single` can reconstruct the residual a^{(1)}/b^{(1)} sharings
        // directly from fold weights — no per-pair `DegenerateEncoding`
        // materialisation, no `n × C(n-1,t)`-Fp explosion (§5-Online.tex:242).
        let mut party_pairs: Vec<Vec<(Fp, Fp)>> = Vec::with_capacity(n);
        let mut party_targets: Vec<Vec<(SubsetT, SubsetT)>> = Vec::with_capacity(n);
        let mut party_cp: Vec<Fp> = Vec::with_capacity(n);
        for i in 0..n {
            let (sum, pairs_with_subsets) = cross_multiply_with_pairs_and_subsets(
                &a_shares[i],
                &alpha_party_shares[i],
                family,
            );
            let mut pairs_plain: Vec<(Fp, Fp)> = Vec::with_capacity(pairs_with_subsets.len());
            let mut targets: Vec<(SubsetT, SubsetT)> =
                Vec::with_capacity(pairs_with_subsets.len());
            for (a_val, b_val, t_u, t_v) in pairs_with_subsets {
                pairs_plain.push((a_val, b_val));
                targets.push((t_u, t_v));
            }
            party_pairs.push(pairs_plain);
            party_targets.push(targets);
            party_cp.push(sum);
        }

        out.push(PerInputPairs {
            party_pairs,
            party_targets,
            party_cp,
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use vdoprf_offline::{setup_pre_shared, PreSharedMaterial};

    fn small_setup() -> (
        usize,
        usize,
        BigUint,
        SubsetFamily,
        Vec<PreSharedMaterial>,
        OnlinePreprocessed,
    ) {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let pre = setup_random_preprocessed_m(4, &family, &modulus);
        (n, t, modulus, family, pre_shared, pre)
    }

    fn expected_outputs(xs: &[Fp], pre: &OnlinePreprocessed, modulus: &BigUint) -> Vec<Fp> {
        let k = pre.k_sharing.reconstruct(modulus);
        xs.iter()
            .enumerate()
            .map(|(j, x)| {
                let alpha_e = pre.alpha_e_sharings[j % pre.alpha_e_sharings.len()]
                    .reconstruct(modulus);
                &(x + &k) * &alpha_e
            })
            .collect()
    }

    #[test]
    fn compute_correctness() {
        let (_n, _t, modulus, family, pre_shared, pre) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..4).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let result = compute::compute(&xs, &pre, &pre_shared, &family, &modulus);
        let expected = expected_outputs(&xs, &pre, &modulus);
        assert!(matches!(result.proof, OnlineProof::Components(ref v) if v.len() == xs.len()));
        for (a, b) in result.client_outputs.iter().zip(expected.iter()) {
            assert_eq!(a.value, b.value);
        }
    }

    #[test]
    fn compute_parallel_correctness() {
        let (_n, _t, modulus, family, pre_shared, pre) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..4).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let result =
            compute_parallel::compute_parallel(&xs, &pre, &pre_shared, &family, &modulus);
        let expected = expected_outputs(&xs, &pre, &modulus);
        assert!(matches!(result.proof, OnlineProof::Components(ref v) if v.len() == xs.len()));
        for (a, b) in result.client_outputs.iter().zip(expected.iter()) {
            assert_eq!(a.value, b.value);
        }
    }

    #[test]
    fn compute_batch_correctness() {
        let (_n, _t, modulus, family, pre_shared, pre) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..4).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let result = compute_batch::compute_batch(&xs, &pre, &pre_shared, &family, &modulus);
        let expected = expected_outputs(&xs, &pre, &modulus);
        assert!(matches!(result.proof, OnlineProof::Batched(_)));
        for (a, b) in result.client_outputs.iter().zip(expected.iter()) {
            assert_eq!(a.value, b.value);
        }
    }

    #[test]
    fn compute_single_input() {
        let (_n, _t, modulus, family, pre_shared, pre) = small_setup();
        let mut rng = rand::thread_rng();
        let xs = vec![Fp::random(&modulus, &mut rng)];
        let result = compute::compute(&xs, &pre, &pre_shared, &family, &modulus);
        let expected = expected_outputs(&xs, &pre, &modulus);
        assert!(matches!(result.proof, OnlineProof::Components(_)));
        assert_eq!(result.client_outputs[0].value, expected[0].value);
    }

    // --- share_add_and_extract_pairs ---

    #[test]
    fn share_add_and_extract_pairs_cp_reconstructs_to_expected_product() {
        let (_n, _t, modulus, family, _pre_shared, pre) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..3).map(|_| Fp::random(&modulus, &mut rng)).collect();

        let (per_input, _comm) =
            share_add_and_extract_pairs(&xs, &pre, &family, &modulus);

        let k = pre.k_sharing.reconstruct(&modulus);
        for (j, x) in xs.iter().enumerate() {
            let alpha_e = pre.alpha_e_sharings[j].reconstruct(&modulus);
            let expected = &(x + &k) * &alpha_e;
            // Σ_i party_cp[i] should equal (x + k)·α_e (cross-multiplication
            // property: the per-party additive shares of a·b sum to a·b).
            let mut sum = Fp::zero(&modulus);
            for cp in &per_input[j].party_cp {
                sum = &sum + cp;
            }
            assert_eq!(sum.value, expected.value);
        }
    }

    #[test]
    fn share_add_and_extract_pairs_pairs_reproduce_cp() {
        let (n, _t, modulus, family, _pre_shared, pre) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..2).map(|_| Fp::random(&modulus, &mut rng)).collect();

        let (per_input, _comm) =
            share_add_and_extract_pairs(&xs, &pre, &family, &modulus);

        for input in &per_input {
            assert_eq!(input.party_pairs.len(), n);
            assert_eq!(input.party_cp.len(), n);
            for i in 0..n {
                let mut sum = Fp::zero(&modulus);
                for (a, b) in &input.party_pairs[i] {
                    sum = &sum + &(a * b);
                }
                assert_eq!(sum.value, input.party_cp[i].value);
            }
        }
    }

    #[test]
    fn share_add_and_extract_pairs_empty() {
        let (_n, _t, modulus, family, _pre_shared, pre) = small_setup();
        let (out, comm) = share_add_and_extract_pairs(&[], &pre, &family, &modulus);
        assert!(out.is_empty());
        assert_eq!(comm.total_bytes(), 0);
        assert_eq!(comm.rounds, 0);
    }

    // --- VIP verification wiring ---

    #[test]
    fn compute_batch_client_verify_accepts_honest() {
        use crate::vip::{client_verify_vip_parallel, VipResult};
        let (_n, _t, modulus, family, pre_shared, pre) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..3).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let result = compute_batch::compute_batch(&xs, &pre, &pre_shared, &family, &modulus);
        match result.proof {
            OnlineProof::Batched(ref out) => {
                assert_eq!(
                    client_verify_vip_parallel(&out.vip, &modulus),
                    VipResult::Accept,
                );
            }
            _ => panic!("compute_batch must emit Batched"),
        }
    }

    #[test]
    fn compute_client_verify_accepts_honest() {
        use crate::vip::{client_verify_vip_parallel, VipResult};
        let (_n, _t, modulus, family, pre_shared, pre) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..3).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let result = compute::compute(&xs, &pre, &pre_shared, &family, &modulus);
        match result.proof {
            OnlineProof::Components(ref vs) => {
                for v in vs {
                    assert_eq!(client_verify_vip_parallel(v, &modulus), VipResult::Accept);
                }
            }
            _ => panic!("compute must emit Components"),
        }
    }
}
