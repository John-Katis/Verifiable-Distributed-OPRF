//! Classic "Boyle" online path: RSS.Mul + DZKP (server-verified) + Open.
//!
//! This is the baseline to compare against our VIP-based online (`compute*`).
//! Matches the standard Boyle/BGIN20 pipeline used throughout the offline
//! phase: each server generates an RSS share of `c = (x + k) · α^e` via
//! `Π_RSS.Mul`, the servers collectively verify the multiplication by
//! running `Π_DZKP` among themselves (designated-verifier = other servers;
//! the client gets only the opened scalar), and then the servers open
//! `\rss{c}` to the client additively.
//!
//! Timing boundary per user request: timer covers RSS.Mul + DZKP prove +
//! DZKP verify + server→client open. Apples-to-apples with VIP, which
//! includes its own client-verify call in its timer.

use num_bigint::BigUint;
use vdoprf_field::Fp;
use vdoprf_network::CommStats;
use vdoprf_offline::double_rand::{generate_double_sharing, DoubleShareLocal};
use vdoprf_offline::dzkp::{dzkp_compute_batch, DzkpResult};
use vdoprf_offline::rss_mul::{
    rss_mul_all_parties_with_record, rss_mul_batched_all_parties_with_record,
};
use vdoprf_offline::PreSharedMaterial;
use vdoprf_ss::{get_party_share, share, RssShare, SubsetFamily};

use crate::OnlinePreprocessed;

/// Proof artefacts of a single Boyle-style online evaluation. DZKP is
/// server-verified so no proof material crosses to the client — the
/// `verdict` is the Accept/Abort decision and `open_shares[i]` is party
/// i's full RSS share of `c` that the servers forwarded to the client
/// once the DZKP returned Accept. The client reconstructs c from the
/// union of unique subset values across parties.
#[derive(Clone, Debug)]
pub struct BoyleProof {
    pub verdict: DzkpResult,
    pub open_shares: Vec<RssShare>,
}

/// Same as `BoyleProof` but over `m` inputs.
#[derive(Clone, Debug)]
pub struct BoyleBatchProof {
    pub verdict: DzkpResult,
    /// `open_shares[j][i]` = party i's RSS share of c_j, as produced by
    /// `rss_mul_batched_all_parties_with_record` and forwarded to the
    /// client once the DZKP accepted.
    pub open_shares: Vec<Vec<RssShare>>,
}


/// Single-input Boyle online path.
///
/// Steps:
/// 1. Client VSS-shares `x` (accounted by the caller / bench layer — this
///    function only returns server↔server + server→client comm).
/// 2. Each server locally computes `\rss{a} = \rss{x + k}`.
/// 3. RSS.Mul with `\rss{b} = \rss{α^e}` → `\rss{c}` + `MulRecord`.
/// 4. DZKP over the multiplication (server-verified). Must Accept.
/// 5. Servers open `\rss{c}` to the client additively (`rss_to_additive`).
/// 6. Client sums the per-server shares to recover `c`.
pub fn compute_boyle_single(
    x: &Fp,
    pre: &OnlinePreprocessed,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Fp, BoyleProof, CommStats) {
    let n = family.n;
    let mut rng = rand::thread_rng();
    let mut rand_counter = 10_000u64;
    let mut comm = CommStats::default();

    // 1. Client VSS → per-party share of x. (Client↔server bytes accounted externally.)
    let x_sharing = share(x, family, modulus, &mut rng);
    let x_party: Vec<RssShare> = (0..n).map(|i| get_party_share(&x_sharing, i, family)).collect();

    // 2. Pull k and α^e RSS shares.
    let k_party: Vec<RssShare> = (0..n)
        .map(|i| get_party_share(&pre.k_sharing, i, family))
        .collect();
    assert!(!pre.alpha_e_sharings.is_empty(), "need one α^e sharing for single-input Boyle");
    let alpha = &pre.alpha_e_sharings[0];
    let alpha_party: Vec<RssShare> =
        (0..n).map(|i| get_party_share(alpha, i, family)).collect();

    // 3. \rss{a} = \rss{x + k} via local RSS add.
    let a_shares: Vec<RssShare> =
        (0..n).map(|i| x_party[i].local_add(&k_party[i])).collect();
    let b_shares = alpha_party;

    // 4. Double sharings for A2T.
    let doubles: Vec<DoubleShareLocal> = (0..n)
        .map(|i| generate_double_sharing(i, rand_counter, &pre_shared[i], family, modulus))
        .collect();
    rand_counter += 1;

    // 5. RSS.Mul → \rss{c} + MulRecord.
    let (c_shares, record, mul_comm) =
        rss_mul_all_parties_with_record(&a_shares, &b_shares, &doubles, family, modulus);
    comm.merge(&mul_comm);

    // 6. DZKP verify (server-side). `dzkp_compute_batch` handles m ≥ 1.
    let records = [record];
    let (verdict, dzkp_comm) = dzkp_compute_batch(&records, family, modulus, pre_shared);
    comm.merge(&dzkp_comm);
    assert_eq!(
        verdict,
        DzkpResult::Accept,
        "Boyle single: DZKP must Accept on honest execution",
    );
    // Silence unused-mut warning in the case the counter is never advanced further.
    let _ = rand_counter;

    // Return raw RSS shares of c and the DZKP verdict. The bench simulates
    // server→client delivery (each server ships its full RSS share) and
    // the client-side reconstruction. `c` is reconstructed locally for
    // test convenience.
    let c =
        vdoprf_ss::ReplicatedSharing::reconstruct_from_party_shares(&c_shares, modulus);
    (c, BoyleProof { verdict, open_shares: c_shares }, comm)
}

/// Batched Boyle online path over `m` inputs.
///
/// Same flow as `compute_boyle_single` but uses
/// `rss_mul_batched_all_parties_with_record` for the `m` multiplications
/// (2 rounds regardless of `m`) and `dzkp_compute_batch` for the batched
/// DZKP (one U/V/W-compressed check).
pub fn compute_boyle_batch(
    xs: &[Fp],
    pre: &OnlinePreprocessed,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Vec<Fp>, BoyleBatchProof, CommStats) {
    let n = family.n;
    let m = xs.len();
    let mut rng = rand::thread_rng();
    let mut rand_counter = 20_000u64;
    let mut comm = CommStats::default();

    if m == 0 {
        return (
            Vec::new(),
            BoyleBatchProof {
                verdict: DzkpResult::Accept,
                open_shares: Vec::new(),
            },
            comm,
        );
    }

    let k_party: Vec<RssShare> = (0..n)
        .map(|i| get_party_share(&pre.k_sharing, i, family))
        .collect();
    assert!(!pre.alpha_e_sharings.is_empty(), "need at least one α^e sharing");

    // Per-input: VSS(x), extract party shares, local add with k, fetch α^e share.
    let mut a_per_input: Vec<Vec<RssShare>> = Vec::with_capacity(m);
    let mut b_per_input: Vec<Vec<RssShare>> = Vec::with_capacity(m);
    for (j, x) in xs.iter().enumerate() {
        let x_sharing = share(x, family, modulus, &mut rng);
        let x_party: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&x_sharing, i, family)).collect();
        let a_shares: Vec<RssShare> =
            (0..n).map(|i| x_party[i].local_add(&k_party[i])).collect();

        let alpha = &pre.alpha_e_sharings[j % pre.alpha_e_sharings.len()];
        let b_shares: Vec<RssShare> =
            (0..n).map(|i| get_party_share(alpha, i, family)).collect();

        a_per_input.push(a_shares);
        b_per_input.push(b_shares);
    }

    // Per-input double sharings (each multiplication needs its own fresh r).
    let doubles_per_input: Vec<Vec<DoubleShareLocal>> = (0..m)
        .map(|_| {
            let row: Vec<DoubleShareLocal> = (0..n)
                .map(|i| {
                    generate_double_sharing(i, rand_counter, &pre_shared[i], family, modulus)
                })
                .collect();
            rand_counter += 1;
            row
        })
        .collect();

    // Batched RSS.Mul: one 2-round network pass for all m inputs.
    let (c_per_input, records, mul_comm) = rss_mul_batched_all_parties_with_record(
        &a_per_input,
        &b_per_input,
        &doubles_per_input,
        family,
        modulus,
    );
    comm.merge(&mul_comm);

    // Batched DZKP (server-verified, U/V/W compression collapses n proofs → 4 scalars).
    let (verdict, dzkp_comm) = dzkp_compute_batch(&records, family, modulus, pre_shared);
    comm.merge(&dzkp_comm);
    assert_eq!(
        verdict,
        DzkpResult::Accept,
        "Boyle batch: DZKP must Accept on honest execution",
    );

    // Return the raw RSS shares `c_per_input` and the DZKP verdict. The
    // bench is responsible for simulating server→client delivery (each
    // server ships its *full* RSS share, redundantly, so the client can
    // detect a lying holder by cross-checking) and the client-side
    // reconstruction. `outputs` is still computed locally for test
    // convenience — production callers would ignore it.
    let mut outputs: Vec<Fp> = Vec::with_capacity(m);
    for c_shares in &c_per_input {
        outputs.push(
            vdoprf_ss::ReplicatedSharing::reconstruct_from_party_shares(c_shares, modulus),
        );
    }

    (
        outputs,
        BoyleBatchProof { verdict, open_shares: c_per_input },
        comm,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{setup_random_preprocessed_m, compute};
    use vdoprf_offline::setup_pre_shared;

    fn small_setup(m: usize) -> (
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
        let pre = setup_random_preprocessed_m(m.max(1), &family, &modulus);
        (n, modulus, family, pre_shared, pre)
    }

    /// `α^e · (x + k)` is the paper's PRF output; honest Boyle reconstructs it.
    fn expected_c(xs: &[Fp], pre: &OnlinePreprocessed, modulus: &BigUint) -> Vec<Fp> {
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
    fn compute_boyle_single_honest_correctness() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(1);
        let mut rng = rand::thread_rng();
        let x = Fp::random(&modulus, &mut rng);
        let (c, proof, _comm) =
            compute_boyle_single(&x, &pre, &pre_shared, &family, &modulus);
        let expected = &expected_c(&[x], &pre, &modulus)[0];
        assert_eq!(c.value, expected.value);
        assert_eq!(proof.verdict, DzkpResult::Accept);
    }

    #[test]
    fn compute_boyle_batch_honest_correctness() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(4);
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..4).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let (cs, proof, _comm) =
            compute_boyle_batch(&xs, &pre, &pre_shared, &family, &modulus);
        let expected = expected_c(&xs, &pre, &modulus);
        for (a, e) in cs.iter().zip(expected.iter()) {
            assert_eq!(a.value, e.value);
        }
        assert_eq!(proof.verdict, DzkpResult::Accept);
    }

    /// Batch with `m = 1` must behave like `compute_boyle_single` on the
    /// same `(x, k, α^e)`. Proves the batched code path is a faithful
    /// generalisation, not a divergent protocol.
    #[test]
    fn compute_boyle_batch_m1_matches_single() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(1);
        let mut rng = rand::thread_rng();
        let x = Fp::random(&modulus, &mut rng);

        let (c_single, _, _) =
            compute_boyle_single(&x, &pre, &pre_shared, &family, &modulus);
        let (cs_batch, _, _) =
            compute_boyle_batch(&[x.clone()], &pre, &pre_shared, &family, &modulus);

        assert_eq!(cs_batch.len(), 1);
        assert_eq!(cs_batch[0].value, c_single.value);
    }

    /// Parity with VIP: same `(x, k, α^e)` feeds both paths; client outputs
    /// match. Non-negotiable — divergence would mean one path is wrong.
    #[test]
    fn compute_boyle_single_parity_with_vip_compute() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(1);
        let mut rng = rand::thread_rng();
        let x = Fp::random(&modulus, &mut rng);

        let (c_boyle, _, _) =
            compute_boyle_single(&x, &pre, &pre_shared, &family, &modulus);
        let vip = compute::compute(&[x.clone()], &pre, &pre_shared, &family, &modulus);
        assert_eq!(vip.client_outputs.len(), 1);
        assert_eq!(vip.client_outputs[0].value, c_boyle.value);
    }

    /// Empty batch is a no-op: no outputs, Accept verdict, default comm stats.
    #[test]
    fn compute_boyle_batch_empty() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(1);
        let (cs, proof, comm) =
            compute_boyle_batch(&[], &pre, &pre_shared, &family, &modulus);
        assert!(cs.is_empty());
        assert_eq!(proof.verdict, DzkpResult::Accept);
        assert_eq!(comm.total_bytes(), 0);
        assert_eq!(comm.rounds, 0);
    }

    /// Comm stats must be non-trivial: at least the RSS.Mul 2 rounds + some
    /// DZKP rounds and bytes. Regression guard against accidentally
    /// returning zero stats.
    #[test]
    fn compute_boyle_single_comm_nonzero() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(1);
        let mut rng = rand::thread_rng();
        let x = Fp::random(&modulus, &mut rng);
        let (_, _, comm) =
            compute_boyle_single(&x, &pre, &pre_shared, &family, &modulus);
        assert!(comm.rounds >= 2, "Boyle single must take at least 2 rounds");
        assert!(comm.total_bytes() > 0, "Boyle single must exchange bytes");
    }

    /// Round-count topology pin matching the Boyle/BGIN20 paper decomposition,
    /// with §3.1.2's Fiat–Shamir collapse fusing Step 2(c) and Step 3(c) VSS:
    ///   2 × RSS.Mul (batched input triples, pre-4.2)
    /// + 1 × VSS ψ_i                                       (4.2 step 3)
    /// + 1 × loop+base VSS fused                           (3.3 step 2c+3c, FS-batched per §3.1.2)
    /// + 1 × F_coin (single vector output of ε's, r, γ_l)  (3.3 step 3e)
    /// + 2 × §3.1.3 W-extension RSS.Mul (h(j) = f1(j)·f2(j))
    /// + 1 × RSS.Open of σ_agg, U(ρ), V(ρ), W(ρ)           (3.3 step 3h batched)
    /// + 1 × RSS.Open β                                    (4.2 step 7)
    /// = 9 rounds, constant in (n,t,m).
    /// Bench harness adds 1 (input VSS from client) + 1 (output delivery) =
    /// 11 rounds total in the `Boyle-Batch` bench row.
    #[test]
    fn compute_boyle_batch_round_topology() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(3);
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..3).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let (_cs, _proof, comm) =
            compute_boyle_batch(&xs, &pre, &pre_shared, &family, &modulus);
        let expected = 2 /* RSS.Mul input triples */
                     + 1 /* VSS ψ_i (4.2 step 3) */
                     + 1 /* 3.3 loop+base VSS fused (step 2c+3c, FS-batched per §3.1.2) */
                     + 1 /* F_coin (3.3 step 3e — vector ε, r, γ_l) */
                     + 2 /* §3.1.3 W-extension RSS.Mul (n−1 batched mults) */
                     + 1 /* RSS.Open σ_agg, U(ρ), V(ρ), W(ρ) (step 3h) */
                     + 1 /* β reconstruct (4.2 step 7) */;
        assert_eq!(
            comm.rounds, expected,
            "compute_boyle_batch round topology diverges from Boyle/BGIN20 \
             paper decomposition: got {}, expected 2 (RSS.Mul) + 1 (VSS ψ) + 1 \
             (loop+base VSS fused) + 1 (F_coin) + 2 (W-extension mul) + 1 \
             (4-agg open) + 1 (β) = {}",
            comm.rounds, expected,
        );
    }
}
