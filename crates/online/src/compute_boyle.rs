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
//!
//! **Deviation from the paper's naive baseline (Protocol 20 / Figure 8
//! `Fvrfy`) — intentional.** The paper's naive-dVOPRF baseline describes
//! `Fvrfy` at the level of an ideal functionality: reconstruct `(a, b, c)`
//! and check `c = a·b`. This module instead calls through to
//! `dzkp_compute_batch`, i.e. the full Boyle et al. RSS.Mul + DZKP + Open
//! pipeline already used throughout the offline phase — a real (non-ideal)
//! instantiation of that same `Fvrfy` check, not a toy reimplementation of
//! the ideal-functionality description. Functionally equivalent (both
//! reject exactly when `c ≠ a·b`), just realized via the same machinery the
//! rest of the codebase already relies on rather than a separate naive
//! reconstruct-and-compare.

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

use crate::input::{client_input_share_standalone, InputResult};
use crate::OnlinePreprocessed;

/// Fresh, never-reused counter base for `Π_Input`'s `[r^(j)]` masks in this
/// file's verified-input path — distinct from every other counter base
/// already in use in this crate (`compute.rs`=3000, `compute_parallel.rs`
/// =2000, `compute_batch.rs`'s own `rand_counter`=4000, zero-sharing
/// =100000+, its `Π_Input` base=60000, and this file's own `rand_counter`
/// =10000/20000).
const BOYLE_INPUT_COUNTER_BASE: u64 = 70_000;

/// Outcome of [`compute_boyle_batch_with_verified_input`]: either the same
/// `(outputs, proof, comm)` `compute_boyle_batch` would produce, or an
/// `Abort` (with the communication already spent on the failed input
/// protocol) if `Π_Input` rejected the client's input.
#[derive(Debug)]
pub enum BoyleVerifiedInputResult {
    Accept(Vec<Fp>, BoyleBatchProof, CommStats),
    Abort(CommStats),
}

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

    // 6. DZKP verify (server-side). `dzkp_compute_batch` handles m ≥ 1. A
    // real Fvrfy failure is a controlled Abort verdict, not a process
    // crash — the caller decides what to do with `BoyleProof::verdict`.
    let records = [record];
    let (verdict, dzkp_comm) = dzkp_compute_batch(&records, family, modulus, pre_shared);
    comm.merge(&dzkp_comm);
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
///
/// Client input goes through the unverified `share()`-as-dealer path (same
/// as `compute_boyle_single`) — kept exactly as-is so it remains directly
/// testable/usable, but per the project's confirmed scope this is no
/// longer wired into any benchmark: only
/// [`compute_boyle_batch_with_verified_input`] is.
pub fn compute_boyle_batch(
    xs: &[Fp],
    pre: &OnlinePreprocessed,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Vec<Fp>, BoyleBatchProof, CommStats) {
    let n = family.n;
    let m = xs.len();
    if m == 0 {
        return compute_boyle_batch_inner(&[], pre, pre_shared, family, modulus, CommStats::default());
    }
    let mut rng = rand::thread_rng();

    // Per-input VSS(x), extract party shares — the unverified "client is
    // its own ad-hoc dealer" path.
    let x_shares_per_input: Vec<Vec<RssShare>> = xs
        .iter()
        .map(|x| {
            let x_sharing = share(x, family, modulus, &mut rng);
            (0..n).map(|i| get_party_share(&x_sharing, i, family)).collect()
        })
        .collect();

    compute_boyle_batch_inner(
        &x_shares_per_input,
        pre,
        pre_shared,
        family,
        modulus,
        CommStats::default(),
    )
}

/// Same as [`compute_boyle_batch`], but client input goes through the
/// *full* `Π_Input` protocol (Protocol 9, [`crate::input`]) at its
/// standalone 3-round cost first. Unlike `compute_batch_with_verified_input`
/// (which folds `Π_Input`'s step-4 echo into VIP's own existing
/// commit-then-hash round), Boyle's RSS.Mul + DZKP pipeline has no
/// equivalent pre-existing broadcast round to fold into without touching
/// its round-count-pinned internals (see `compute_boyle_batch_round_topology`
/// below), so this uses [`client_input_share_standalone`] as-is: 2
/// client-facing rounds (mask-and-open) + 1 dedicated echo round.
///
/// On `Π_Input` abort, returns `BoyleVerifiedInputResult::Abort` without
/// running the rest of the protocol.
pub fn compute_boyle_batch_with_verified_input(
    xs: &[Fp],
    pre: &OnlinePreprocessed,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> BoyleVerifiedInputResult {
    let (verdict, x_shares_per_input, input_comm) =
        client_input_share_standalone(xs, pre_shared, family, modulus, BOYLE_INPUT_COUNTER_BASE);
    if verdict == InputResult::Abort {
        return BoyleVerifiedInputResult::Abort(input_comm);
    }
    let (outputs, proof, comm) = compute_boyle_batch_inner(
        &x_shares_per_input,
        pre,
        pre_shared,
        family,
        modulus,
        input_comm,
    );
    BoyleVerifiedInputResult::Accept(outputs, proof, comm)
}

/// Shared body of both entry points above: local-add with `k`, batched
/// RSS.Mul, batched DZKP — given already-obtained per-input, per-party RSS
/// shares of `x` and however much comm the client-input stage has already
/// spent.
fn compute_boyle_batch_inner(
    x_shares_per_input: &[Vec<RssShare>],
    pre: &OnlinePreprocessed,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
    mut comm: CommStats,
) -> (Vec<Fp>, BoyleBatchProof, CommStats) {
    let n = family.n;
    let m = x_shares_per_input.len();
    let mut rand_counter = 20_000u64;

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

    // Per-input: local add x's share with k, fetch α^e share.
    let mut a_per_input: Vec<Vec<RssShare>> = Vec::with_capacity(m);
    let mut b_per_input: Vec<Vec<RssShare>> = Vec::with_capacity(m);
    for (j, x_party) in x_shares_per_input.iter().enumerate() {
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

    // Batched DZKP (server-verified, U/V/W compression collapses n proofs
    // → 4 scalars). A real Fvrfy failure is a controlled Abort verdict, not
    // a process crash — the caller decides what to do with
    // `BoyleBatchProof::verdict`.
    let (verdict, dzkp_comm) = dzkp_compute_batch(&records, family, modulus, pre_shared);
    comm.merge(&dzkp_comm);

    // Return the raw RSS shares `c_per_input` and the DZKP verdict. The
    // bench is responsible for simulating server→client delivery (each
    // server ships its *full* RSS share, redundantly, so the client can
    // detect a lying holder by cross-checking) and the client-side
    // reconstruction. `outputs` is still computed locally for test
    // convenience — production callers would ignore it. On `Abort`, these
    // per-input plaintexts are meaningless and must not be used; callers
    // must check `verdict` first.
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

    /// A real DZKP failure must return a controlled `Abort` verdict, not
    /// panic. Mirrors `compute_boyle_batch`'s exact internal pipeline (VSS
    /// x, local add with k, batched RSS.Mul, then `dzkp_compute_batch`) but
    /// tampers one party's claimed `cp` on the resulting `MulRecord` before
    /// verification — the same tamper `dzkp.rs`'s own
    /// `test_dzkp_compute_batch_tampered_cp_aborts` uses — so the DZKP must
    /// reject. Regression guard for the removed `assert_eq!(.., Accept, ..)`
    /// panics.
    #[test]
    fn compute_boyle_batch_dzkp_abort_does_not_panic() {
        let (n, modulus, family, pre_shared, pre) = small_setup(2);
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..2).map(|_| Fp::random(&modulus, &mut rng)).collect();

        let k_party: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&pre.k_sharing, i, &family)).collect();
        let mut rand_counter = 90_000u64;
        let mut a_per_input = Vec::with_capacity(xs.len());
        let mut b_per_input = Vec::with_capacity(xs.len());
        for (j, x) in xs.iter().enumerate() {
            let x_sharing = share(x, &family, &modulus, &mut rng);
            let x_party: Vec<RssShare> =
                (0..n).map(|i| get_party_share(&x_sharing, i, &family)).collect();
            let a_shares: Vec<RssShare> =
                (0..n).map(|i| x_party[i].local_add(&k_party[i])).collect();
            let alpha = &pre.alpha_e_sharings[j % pre.alpha_e_sharings.len()];
            let b_shares: Vec<RssShare> =
                (0..n).map(|i| get_party_share(alpha, i, &family)).collect();
            a_per_input.push(a_shares);
            b_per_input.push(b_shares);
        }
        let doubles_per_input: Vec<Vec<DoubleShareLocal>> = (0..xs.len())
            .map(|_| {
                let row: Vec<DoubleShareLocal> = (0..n)
                    .map(|i| generate_double_sharing(i, rand_counter, &pre_shared[i], &family, &modulus))
                    .collect();
                rand_counter += 1;
                row
            })
            .collect();
        let (_c_per_input, mut records, _mul_comm) = rss_mul_batched_all_parties_with_record(
            &a_per_input, &b_per_input, &doubles_per_input, &family, &modulus,
        );

        // Tamper: add 1 to party 0's claimed cp for the first record.
        let one = Fp::new(BigUint::from(1u32), &modulus);
        records[0].party_cp[0] = &records[0].party_cp[0] + &one;

        let (verdict, _dzkp_comm) = dzkp_compute_batch(&records, &family, &modulus, &pre_shared);
        assert_eq!(
            verdict,
            DzkpResult::Abort,
            "tampered cp must be rejected, not silently accepted",
        );
        // Building the proof struct from this verdict must not panic —
        // exactly what `compute_boyle_batch` now does internally.
        let proof = BoyleBatchProof { verdict, open_shares: Vec::new() };
        assert_eq!(proof.verdict, DzkpResult::Abort);
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

    // ---- compute_boyle_batch_with_verified_input (Π_Input, standalone) ----

    /// Honest path: verified-input and unverified `compute_boyle_batch`
    /// must reconstruct the same `c = (x + k) · α^e` for the same `xs` —
    /// only how `x`'s shares are obtained differs.
    #[test]
    fn compute_boyle_batch_with_verified_input_honest_correctness() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(3);
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..3).map(|_| Fp::random(&modulus, &mut rng)).collect();

        let expected = expected_c(&xs, &pre, &modulus);
        match compute_boyle_batch_with_verified_input(&xs, &pre, &pre_shared, &family, &modulus) {
            BoyleVerifiedInputResult::Accept(cs, proof, _comm) => {
                for (a, e) in cs.iter().zip(expected.iter()) {
                    assert_eq!(a.value, e.value);
                }
                assert_eq!(proof.verdict, DzkpResult::Accept);
            }
            BoyleVerifiedInputResult::Abort(_) => panic!("honest input must not abort"),
        }
    }

    /// `compute_boyle_batch_with_verified_input`'s communication cost must
    /// exceed `compute_boyle_batch`'s — real Π_Input overhead is now charged
    /// where the unverified path charges nothing for input at all.
    #[test]
    fn compute_boyle_batch_with_verified_input_costs_more_bytes() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(2);
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..2).map(|_| Fp::random(&modulus, &mut rng)).collect();

        let (_cs, _proof, plain_comm) =
            compute_boyle_batch(&xs, &pre, &pre_shared, &family, &modulus);
        let verified = match compute_boyle_batch_with_verified_input(&xs, &pre, &pre_shared, &family, &modulus) {
            BoyleVerifiedInputResult::Accept(_, _, comm) => comm,
            BoyleVerifiedInputResult::Abort(_) => panic!("honest input must not abort"),
        };

        assert!(
            verified.total_bytes() + verified.client_bytes
                > plain_comm.total_bytes() + plain_comm.client_bytes,
            "Π_Input's real accounting must cost more than the unverified path's untracked input step",
        );
    }

    /// Round-count pin: the verified-input path's own pipeline is
    /// byte-for-byte the same `compute_boyle_batch_inner` as the unverified
    /// path (9 rounds, see `compute_boyle_batch_round_topology`), plus
    /// `Π_Input`'s standalone 3 rounds (2 client-facing + 1 echo) charged
    /// up front = 12 rounds, constant in `(n,t,m)`.
    #[test]
    fn compute_boyle_batch_with_verified_input_round_topology() {
        let (_n, modulus, family, pre_shared, pre) = small_setup(3);
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..3).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let comm = match compute_boyle_batch_with_verified_input(&xs, &pre, &pre_shared, &family, &modulus) {
            BoyleVerifiedInputResult::Accept(_, _, comm) => comm,
            BoyleVerifiedInputResult::Abort(_) => panic!("honest input must not abort"),
        };
        assert_eq!(
            comm.rounds, 12,
            "compute_boyle_batch_with_verified_input must cost exactly 9 \
             (inner Boyle pipeline) + 3 (Π_Input standalone) = 12 rounds, got {}",
            comm.rounds,
        );
    }

    /// A server whose locally-held PRF key material for one subset
    /// disagrees with what the designated sender used (simulating a
    /// corrupted/malicious server) must be caught by `Π_Input`'s `ψ_i`
    /// check, surfacing as `BoyleVerifiedInputResult::Abort` — not silently
    /// producing a wrong result. Mirrors
    /// `compute_batch_with_verified_input_aborts_on_forged_server_key`.
    #[test]
    fn compute_boyle_batch_with_verified_input_aborts_on_forged_server_key() {
        let (n, modulus, family, mut pre_shared, pre) = small_setup(2);
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..2).map(|_| Fp::random(&modulus, &mut rng)).collect();

        let subset = family
            .subsets
            .iter()
            .find(|t| {
                let holders: Vec<usize> = (0..n).filter(|p| !t.contains(p)).collect();
                holders.len() >= 2
            })
            .copied()
            .expect("need a subset with at least 2 holders for n=3,t=1");
        let sender = vdoprf_ss::covering_policy(&subset, n);
        let victim = (0..n)
            .find(|&p| !subset.contains(&p) && p != sender)
            .expect("need a second holder distinct from the designated sender");

        let other_subset = *family
            .subsets
            .iter()
            .find(|t| **t != subset && pre_shared[victim].prf_keys.contains_key(t))
            .expect("need another subset victim also holds a key for");
        let swapped_key = pre_shared[victim].prf_keys[&other_subset].clone();
        pre_shared[victim].prf_keys.insert(subset, swapped_key);

        match compute_boyle_batch_with_verified_input(&xs, &pre, &pre_shared, &family, &modulus) {
            BoyleVerifiedInputResult::Abort(_) => {}
            BoyleVerifiedInputResult::Accept(..) => {
                panic!("forged server key must be caught by ψ_i, not accepted")
            }
        }
    }
}
