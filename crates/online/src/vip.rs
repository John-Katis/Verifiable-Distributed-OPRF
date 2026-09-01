//! Π_VIP — Verifiable Inner Product over RSS.
//!
//! Section 5 (`overleaf-protocols/Chapters/5-Online.tex`). Replaces the old
//! "RSS.Mul (A2T) → DZKP → open" pipeline with a single **verifiable
//! multiply-and-open** primitive.
//!
//! - [`vip_single`] — one prover, `L` pairs `(a^{(ℓ)}, b^{(ℓ)})`. The prover
//!   knows them in plaintext; the servers conceptually hold them as
//!   degenerate RSS sharings via TCC'05 local share conversion (paper
//!   §5-Online.tex:242). Iterates `γ = ⌈log L⌉` times, each iteration
//!   defining linear polynomials `f_j`, their pairwise product `q(x)`
//!   (degree 2), and sharing `q(1), q(2), q(3)` via VSS (5-Online.tex:131).
//!   The plaintext fold runs as in the paper. The share-side fold is *not*
//!   materialised pair-by-pair: because each pair's share is a degenerate
//!   encoding `⟨a_ℓ⟩_{T_ℓ}` (one nonzero subset, all others zero), and
//!   every iteration's transformation is linear, the residual
//!   `⟨a^{(1)}⟩, ⟨b^{(1)}⟩` after γ folds equals
//!   `Σ_ℓ w_ℓ · ⟨a_ℓ⟩_{T_ℓ}` where `w_ℓ` is determined entirely by the
//!   shared FS challenges `(r_k)`. We compute the weights `w_ℓ` upfront
//!   and accumulate one `Fp` per subset — never materialising the
//!   `L · n · S` field elements that the naive lockstep fold would
//!   require. Outputs per-server RSS shares of
//!   `(d^{(γ)}_i, a_i^{(1)}, b_i^{(1)}, σ_i, c_i)` (5-Online.tex:313).
//!
//! - [`vip_parallel`] — `Π_VIP^Prl`. Draws the fold/batch coefficients
//!   `(r_k, ε_k, ε'_i)` via a genuine coin toss (`coin_toss`, PRF-keyed
//!   RSS shares + open) *before* running any `vip_single` instance, so all
//!   n instances see the same values; the evaluation challenge `ρ` is drawn
//!   in its own later coin-toss round *after* `Π_RSS.Mul` — it is a
//!   challenge point for the committed W/U/V polynomials and must not be
//!   predictable while W's high points are still being computed. Per the
//!   paper's Appendix L, the Fiat–Shamir transform does not apply to this
//!   batched protocol — these challenges must stay real coin tosses, not
//!   transcript hashes.
//!   Aggregates `c = Σ c_i` and `Σ = Σ ε'_i · σ_i` locally, then runs the
//!   §5.1 batched triple verification (5-Online.tex:702/728, currently in
//!   `\if0` in the rendered paper but treated as the target spec): `U(x),
//!   V(x)` of degree `n−1` through `{(i, f_1^{(i)}(r))}_{i∈[n]}` and
//!   `{(i, f_2^{(i)}(r))}_{i∈[n]}`; multiplications at `j ∈ {n+1, …, 2n−1}`
//!   (n−1 triples), batched into 2 rounds via
//!   `rss_mul_batched_all_parties_with_record`; finally `W(ρ), U(ρ), V(ρ)`
//!   are interpolated at a fresh `ρ ∉ {1, …, 2n−1}`. No `(a₀, b₀)`
//!   randomization — `f_1(r), f_2(r)` are never opened during dVOPRF
//!   (5-Online.tex:280), so the soundness margin from output randomization
//!   is unnecessary.
//!
//! Output shape `(W(ρ), U(ρ), V(ρ), Σ, c)` matches the bench's existing
//! 5-RSS-values-per-server proof accountant — drop-in replacement for the
//! old `ParallelBatchOutput`.

use num_bigint::BigUint;
use vdoprf_crypto::hash::{hash_bytes, hash_field_elements};
use vdoprf_field::Fp;
use vdoprf_network::{CommStats, SimulatedNetwork};
use vdoprf_offline::double_rand::{generate_double_sharing, generate_rss_random};
use vdoprf_offline::rss_mul::rss_mul_batched_all_parties_with_record;
use vdoprf_offline::rss_share::{charge_f_coin_batch, charge_rss_share_p2p};
use vdoprf_offline::PreSharedMaterial;
use std::collections::BTreeMap;
use vdoprf_ss::{
    get_party_share, lagrange_coeffs, lagrange_eval_with_coeffs, share, ReplicatedSharing,
    RssShare, SubsetFamily, SubsetT,
};

/// Verdict from client-side VIP verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VipResult {
    Accept,
    Abort,
}

/// Realize a single F_coin output: an unbiased field element that no
/// minority of corrupt parties can influence or predict before it is
/// opened. Each party independently derives its own PRF-keyed RSS share of
/// the same secret value (no communication needed — deterministic given
/// `pre_shared` and `counter`), then the shares are opened by
/// reconstruction. This is the standard "generate from correlated
/// randomness, then open" F_coin realization for an honest-majority RSS
/// setting, and is required here: the paper (Appendix L, after Protocol 19)
/// states the Fiat–Shamir transform is *not* applicable to the batched VIP
/// protocol (`vip_parallel`), and that "FCoin at the last step in Π_VIP
/// must be called" as a genuine coin toss, not a transcript hash. Mirrors
/// `generate_double_sharing`'s existing use in this file for the `(a₀, b₀)`
/// output-randomization values, minus the unneeded additive-share half.
fn coin_toss(
    counter: u64,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> Fp {
    let shares: Vec<RssShare> = (0..family.n)
        .map(|i| generate_rss_random(i, counter, &pre_shared[i], family, modulus))
        .collect();
    ReplicatedSharing::reconstruct_from_party_shares(&shares, modulus)
}

/// Per-server RSS shares of the five values output by Π_VIP.
/// `(d_i, a_i^{(1)}, b_i^{(1)}, σ_i, c_i)` per §5-Online line 171. σ is fully
/// closed inside `vip_single` — the per-round `(d, q1, q2)` stash and the
/// Protocol 4 step 14 ε_2..ε_γ PRG-driven σ-update are internal.
#[derive(Clone, Debug)]
pub struct VipBundle {
    pub prover_id: usize,
    pub d: Vec<RssShare>,
    pub a1: Vec<RssShare>,
    pub b1: Vec<RssShare>,
    pub sigma: Vec<RssShare>,
    pub c: Vec<RssShare>,
}

/// RSS shares of the five aggregated Π_VIP^Prl outputs sent to the client.
///
/// Matches §4-Online `fig:doprf_protocol` line 260:
/// *"Client C calls F_RSS.Open to robustly reconstruct ⟨⟨W(ρ)⟩⟩, ⟨⟨U(ρ)⟩⟩,
/// ⟨⟨V(ρ)⟩⟩, ⟨⟨Σ⟩⟩, and ⟨⟨c⟩⟩."* and the efficiency-analysis cost
/// `5·c_Open + m` per server from Appendix §A.2 (`appendix.tex:1102–1103`).
///
/// Each `Vec<RssShare>` has length `n` — one RSS share per party (every
/// entry carries the party's full `C(n-1, t)` subset components). The bench
/// delivery path `client_deliver_vip_batched` serialises each party's full
/// `share.shares.values()` + a 32-byte hash, exactly as `Π_RSS.Open`
/// (`appendix.tex:111–129`) specifies; the client reconstructs via
/// [`ReplicatedSharing::reconstruct_from_party_shares`] and checks the
/// paper's `W(ρ) = U(ρ)·V(ρ)` ∧ `Σ = 0` equations (see
/// [`client_verify_vip_parallel`]).
#[derive(Clone, Debug)]
pub struct VipParallelOutput {
    pub w_rho_shares: Vec<RssShare>,
    pub u_rho_shares: Vec<RssShare>,
    pub v_rho_shares: Vec<RssShare>,
    pub sigma_shares: Vec<RssShare>,
    pub c_shares: Vec<RssShare>,
}

/// Π_VIP (single sharing-server). Each iteration halves the pair count by
/// folding linear polynomials at a Fiat-Shamir challenge `r_k`. After
/// `γ = ⌈log L⌉` iterations the L=1 endpoint `(a^{(1)}, b^{(1)})` is an RSS
/// sharing — paper §5-Online.tex:149 ("Servers use Lagrange interpolation
/// to locally compute ⟦a_i^(ℓ)⟧, ⟦b_i^(ℓ)⟧"). The plaintext fold runs
/// pair-wise as in the paper to compute `q(1), q(2), q(3)`.
///
/// **Share-side fold is folded analytically.** Each pair's share is a
/// degenerate encoding `⟨a_ℓ⟩_{T_ℓ}` (one nonzero subset, all others
/// zero), and the per-iteration share update is the linear map
/// `(2-r_k)·lo + (r_k-1)·hi`. The composition over γ iterations is a
/// pure-scalar Kronecker product, so the residual share equals
/// `Σ_ℓ w_ℓ · ⟨a_ℓ⟩_{T_ℓ}` for fold weights `w_ℓ` derived solely from
/// the FS sequence `r_ks`. We compute the `padded_len` weights upfront and
/// accumulate one `Fp` per subset T into a single `ReplicatedSharing`,
/// then extract per-party views — total transient memory is `O(N)`
/// rather than the `O(L · n · S)` the naive lockstep fold uses. This is
/// the optimisation that lets the (n=9,t=4) m=35 cell fit on a single
/// machine.
///
/// `pair_targets[ℓ] = (T_u, T_v)` are the originating subsets of the ℓ-th
/// pair's `(a_val, b_val)` — the canonical TCC'05 local-share-conversion
/// outputs from `cross_multiply_with_pairs_and_subsets`.
///
/// **Rounds.** Under Fiat–Shamir, all γ·VSS(q(1), q(2), q(3)) sends
/// concatenate into a single p2p message per (prover, recipient) pair —
/// one synchronous round from prover to each other server (all n instances
/// run this round in parallel on the shared `net`). This is a *VSS send*
/// (shares), not an *open* (reconstruction); Π_VIP^Prl never publicly
/// reconstructs q(·), a^{(1)}, b^{(1)} between servers (`5-Online.tex:280`).
///
/// `r_ks` is the pre-derived F_coin sequence `(r_k)_{k∈[γ]}` — shared across
/// all n instances when invoked from `vip_parallel` to match
/// `fig:vip_parallel_protocol` ("All instances share the same outputs r_k of
/// F_Coin"). The caller pads inputs to a common `padded_len` (a power of two)
/// so γ is the same for every instance. `eps_sigma` is the pre-derived
/// F_coin sequence `(ε_k)_{k=2..γ}` (length γ-1) for the Protocol 4 step 14
/// σ-update — also shared across all n instances, drawn once by the caller
/// via genuine F_coin (`coin_toss`), *not* sampled locally: per Appendix L
/// the Fiat–Shamir transform does not apply here, and per the note after
/// Protocol 19, "FCoin at the last step in Π_VIP must be called" as a real
/// coin toss. `vip_single` stashes the per-round `(d, q(1), q(2))` triples
/// and applies the σ-update using the caller-supplied `eps_sigma` directly.
///
/// `net` is shared with sibling instances; all prover p2p sends land on the
/// same `SimulatedNetwork.current_round`.
#[allow(clippy::too_many_arguments)]
pub fn vip_single(
    prover_id: usize,
    n: usize,
    pairs: &[(Fp, Fp)],
    pair_targets: &[(SubsetT, SubsetT)],
    padded_len: usize,
    r_ks: &[Fp],
    eps_sigma: &[Fp],
    family: &SubsetFamily,
    modulus: &BigUint,
    net: &mut SimulatedNetwork,
) -> VipBundle {
    let mut rng = rand::thread_rng();

    // Pad pairs to the caller-specified common length (a power of two, min 2).
    // A common padded length across all n provers keeps γ the same, required
    // for the shared FS challenge sequence.
    assert!(padded_len.is_power_of_two() && padded_len >= 2);
    assert!(
        pairs.len() <= padded_len,
        "pairs ({}) must fit within padded_len ({})",
        pairs.len(),
        padded_len,
    );
    assert_eq!(pairs.len(), pair_targets.len());
    let expected_gamma = (padded_len as f64).log2() as usize;
    assert_eq!(
        r_ks.len(),
        expected_gamma,
        "r_k sequence length must match γ = log2(padded_len)",
    );
    assert_eq!(
        eps_sigma.len(),
        expected_gamma.saturating_sub(1),
        "eps_sigma length must match γ-1 (one per stashed round k=2..γ)",
    );

    // Plaintext side (prover-known): used to compute q(1), q(2), q(3) per
    // iteration. The prover holds the originating (a, b) values in clear.
    let mut cur_a: Vec<Fp> = pairs.iter().map(|(a, _)| a.clone()).collect();
    let mut cur_b: Vec<Fp> = pairs.iter().map(|(_, b)| b.clone()).collect();
    while cur_a.len() < padded_len {
        cur_a.push(Fp::zero(modulus));
        cur_b.push(Fp::zero(modulus));
    }

    // σ_i starts as RSS-of-zero. Closed inside this function: after the
    // for-k loop completes, the Protocol 4 step 14 PRG-driven ε_2..ε_γ
    // σ-update accumulates the stashed (d, q1, q2) triples.
    let zero_sharing = share(&Fp::zero(modulus), family, modulus, &mut rng);
    let mut sigma_shares: Vec<RssShare> = (0..n)
        .map(|i| get_party_share(&zero_sharing, i, family))
        .collect();

    let mut c_i_shares: Option<Vec<RssShare>> = None;
    let mut d_shares: Option<Vec<RssShare>> = None;
    // Stash of per-round (d_before_update, q(1)_ps, q(2)_ps) for k ∈ [2..γ].
    // Used by `vip_parallel` to apply the post-loop σ-update once it has
    // sampled fresh ε_2..ε_γ from the PRG.
    let mut sigma_stash: Vec<(Vec<RssShare>, Vec<RssShare>, Vec<RssShare>)> =
        Vec::with_capacity(expected_gamma.saturating_sub(1));

    let one = Fp::new(BigUint::from(1u32), modulus);
    let two = Fp::new(BigUint::from(2u32), modulus);
    let three = Fp::new(BigUint::from(3u32), modulus);

    // Loop while we still have at least two values to fold. Iteration index
    // `iter` picks the shared challenge `r_k` from the caller.
    let mut iter = 0usize;
    while cur_a.len() >= 2 {
        let half = cur_a.len() / 2;

        // q(1) = Σ_{j<half} a_j·b_j; q(2) = Σ_{j≥half} a_j·b_j (prover-side
        // plaintext aggregations — the prover knows every (a_j, b_j) and can
        // compute these sums locally).
        let mut q1 = Fp::zero(modulus);
        for j in 0..half {
            q1 = &q1 + &(&cur_a[j] * &cur_b[j]);
        }
        let mut q2 = Fp::zero(modulus);
        for j in half..cur_a.len() {
            q2 = &q2 + &(&cur_a[j] * &cur_b[j]);
        }
        // Each f_j is linear with f_j(1) = lower-half value, f_j(2) = upper-half
        // value, so f_j(3) = 2·upper − lower.
        let mut q3 = Fp::zero(modulus);
        for e in 0..half {
            let fa3 = &(&cur_a[e + half] + &cur_a[e + half]) - &cur_a[e];
            let fb3 = &(&cur_b[e + half] + &cur_b[e + half]) - &cur_b[e];
            q3 = &q3 + &(&fa3 * &fb3);
        }

        // Prover RSS-shares q(1), q(2), q(3) to all servers via Π_RSS.Share
        // (appendix.tex:92–108). Under FS the log(L) per-iteration sends
        // concatenate into a single synchronous round per prover.
        let q1_sharing = share(&q1, family, modulus, &mut rng);
        let q2_sharing = share(&q2, family, modulus, &mut rng);
        let q3_sharing = share(&q3, family, modulus, &mut rng);

        charge_rss_share_p2p(prover_id, 3, family, modulus, net);

        let q1_ps: Vec<RssShare> = (0..n).map(|i| get_party_share(&q1_sharing, i, family)).collect();
        let q2_ps: Vec<RssShare> = (0..n).map(|i| get_party_share(&q2_sharing, i, family)).collect();
        let q3_ps: Vec<RssShare> = (0..n).map(|i| get_party_share(&q3_sharing, i, family)).collect();

        // First iteration only: c_i ← q(1) + q(2), d_i ← c_i (paper line 138).
        if c_i_shares.is_none() {
            let c_init: Vec<RssShare> =
                (0..n).map(|i| q1_ps[i].local_add(&q2_ps[i])).collect();
            d_shares = Some(c_init.clone());
            c_i_shares = Some(c_init);
        }

        let r_k = r_ks[iter].clone();

        // Round k=1 (iter=0) contributes 0 because d_init = q(1) + q(2);
        // skip stashing it. For k ≥ 2 we stash the d *before* this round's
        // Lagrange update so the recovered term equals the running
        // self-consistency check between successive q polynomials.
        if iter > 0 {
            if let Some(ref d_sh) = d_shares {
                sigma_stash.push((d_sh.clone(), q1_ps.clone(), q2_ps.clone()));
            }
        }

        // Update d_i ← q(r_k) shares via Lagrange through (1,q(1)),(2,q(2)),(3,q(3)).
        let d_xs = [one.clone(), two.clone(), three.clone()];
        let d_coeffs = lagrange_coeffs(&d_xs, &r_k, modulus);
        let new_d_shares: Vec<RssShare> = (0..n)
            .map(|i| {
                let shares: [&RssShare; 3] = [&q1_ps[i], &q2_ps[i], &q3_ps[i]];
                lagrange_eval_with_coeffs(&shares, &d_coeffs)
            })
            .collect();
        d_shares = Some(new_d_shares);

        // Plaintext fold ONLY: reduce (a, b) by f_j(r_k) = (2−r_k)·lo + (r_k−1)·hi.
        // The share-side fold is folded analytically below — see
        // `build_residual_shares_from_targets`.
        let coeff1 = &two - &r_k;
        let coeff2 = &r_k - &one;
        let mut new_a = Vec::with_capacity(half);
        let mut new_b = Vec::with_capacity(half);
        for e in 0..half {
            new_a.push(&(&cur_a[e] * &coeff1) + &(&cur_a[e + half] * &coeff2));
            new_b.push(&(&cur_b[e] * &coeff1) + &(&cur_b[e + half] * &coeff2));
        }
        cur_a = new_a;
        cur_b = new_b;
        iter += 1;

        if cur_a.len() == 1 {
            break;
        }
    }

    // Build a^{(1)}_i and b^{(1)}_i directly from the pre-derived fold
    // weights. Skips materialising ANY per-pair RSS sharing; total
    // intermediate state is O(N) Fps per residual (one accumulator slot
    // per subset T). 5-Online.tex:149 says the residual shares are an RSS
    // — we just compute that RSS analytically instead of folding pair by pair.
    let weights = compute_fold_weights(padded_len, r_ks, modulus);
    let (a1_shares, b1_shares) = build_residual_shares_from_targets(
        pairs,
        pair_targets,
        &weights,
        n,
        family,
        modulus,
    );

    // Protocol 4 step 14 (5-Online.tex:169): F_coin → ε_2..ε_γ, then
    //   σ_i ← σ_i + Σ_{k=2..γ} ε_k · (d^(k)_i − q^(k)(1)_i − q^(k)(2)_i)
    // `eps_sigma` is supplied by the caller (`vip_parallel`), drawn once via
    // genuine F_coin (`coin_toss`) and shared identically across all n
    // parallel Π_VIP instances — matching `fig:vip_parallel_protocol` line
    // 197 ("all instances share the same outputs (r_k, ε_k)"). Per Appendix
    // L this must stay a real coin toss, not a Fiat-Shamir transcript value.
    for ((d_sh, q1_ps, q2_ps), eps) in sigma_stash.iter().zip(eps_sigma.iter()) {
        for i in 0..n {
            let mut term = d_sh[i].clone();
            term.local_sub_assign(&q1_ps[i]);
            term.local_sub_assign(&q2_ps[i]);
            term.local_scalar_mul_assign(eps);
            sigma_shares[i].local_add_assign(&term);
        }
    }

    VipBundle {
        prover_id,
        d: d_shares.expect("VIP loop runs at least once for L >= 2"),
        a1: a1_shares,
        b1: b1_shares,
        sigma: sigma_shares,
        c: c_i_shares.expect("VIP loop runs at least once for L >= 2"),
    }
}

/// Compute the linear-combination weights `w[ℓ]` for ℓ ∈ [padded_len] such
/// that after γ = log₂(padded_len) lockstep folds with challenges `r_ks`,
/// the residual `a^{(1)}` equals `Σ_ℓ w[ℓ] · a_orig[ℓ]`.
///
/// The fold at iteration k applies coefficients `(2-r_k)` to "lo" and
/// `(r_k-1)` to "hi". After γ iterations each original index ℓ has been
/// classified γ times as either lo or hi. The bit `(γ-1-k)` of ℓ in
/// big-endian decides iteration k's classification: bit 0 ⇒ lo, bit 1 ⇒ hi.
/// `w[ℓ]` is the product of the corresponding factors.
fn compute_fold_weights(padded_len: usize, r_ks: &[Fp], modulus: &BigUint) -> Vec<Fp> {
    let one = Fp::one(modulus);
    let two = Fp::new(BigUint::from(2u32), modulus);
    let gamma = r_ks.len();
    debug_assert_eq!(padded_len, 1usize << gamma);

    let lo_coeff: Vec<Fp> = r_ks.iter().map(|r| &two - r).collect();
    let hi_coeff: Vec<Fp> = r_ks.iter().map(|r| r - &one).collect();

    let mut weights = Vec::with_capacity(padded_len);
    for ell in 0..padded_len {
        let mut w = one.clone();
        for k in 0..gamma {
            let bit = (ell >> (gamma - 1 - k)) & 1;
            if bit == 0 {
                w = &w * &lo_coeff[k];
            } else {
                w = &w * &hi_coeff[k];
            }
        }
        weights.push(w);
    }
    weights
}

/// Assemble per-party RSS shares of `a^{(1)} = Σ_ℓ w_ℓ · a_ℓ` and
/// `b^{(1)} = Σ_ℓ w_ℓ · b_ℓ` from the prover's plaintext pairs and their
/// originating subsets.
///
/// Each pair's share is the degenerate encoding `⟨a_ℓ⟩_{T_a_ℓ}` (single
/// nonzero subset). A linear combination of degenerate encodings has, at
/// each subset `T`, the weighted sum `Σ_{ℓ : T_a_ℓ = T} w_ℓ · a_ℓ` —
/// which we accumulate directly into a `BTreeMap<SubsetT, Fp>`. Per-party
/// views fall out via `get_party_share`. Padded positions ℓ ≥ pairs.len()
/// have `a_ℓ = b_ℓ = 0` so they contribute nothing regardless of weight.
fn build_residual_shares_from_targets(
    pairs: &[(Fp, Fp)],
    pair_targets: &[(SubsetT, SubsetT)],
    weights: &[Fp],
    n: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Vec<RssShare>, Vec<RssShare>) {
    let mut a_components: BTreeMap<SubsetT, Fp> = BTreeMap::new();
    let mut b_components: BTreeMap<SubsetT, Fp> = BTreeMap::new();
    for subset in &family.subsets {
        a_components.insert(*subset, Fp::zero(modulus));
        b_components.insert(*subset, Fp::zero(modulus));
    }
    for (ell, ((a_val, b_val), (t_a, t_b))) in
        pairs.iter().zip(pair_targets.iter()).enumerate()
    {
        let w = &weights[ell];
        let a_term = &(a_val * w);
        let b_term = &(b_val * w);
        let a_slot = a_components
            .get_mut(t_a)
            .expect("target subset must exist in family");
        *a_slot = &*a_slot + a_term;
        let b_slot = b_components
            .get_mut(t_b)
            .expect("target subset must exist in family");
        *b_slot = &*b_slot + b_term;
    }
    let a_sharing = ReplicatedSharing { components: a_components };
    let b_sharing = ReplicatedSharing { components: b_components };
    let a_shares: Vec<RssShare> = (0..n)
        .map(|s| get_party_share(&a_sharing, s, family))
        .collect();
    let b_shares: Vec<RssShare> = (0..n)
        .map(|s| get_party_share(&b_sharing, s, family))
        .collect();
    (a_shares, b_shares)
}

/// Π_VIP^Prl (Protocol `fig:vip_parallel_protocol`) with §5.1 batching
/// (5-Online.tex:702 — `\subsection{Batching Optimizations}`, currently in
/// `\if0` in the rendered paper but treated as the target spec per user
/// direction).
///
/// Runs `n` instances of `vip_single` in parallel under a shared set of
/// genuine F_coin challenges (`coin_toss`), then σ-batches and runs the
/// §5.1 triple-verification: U, V of degree n−1
/// through `{(i, f_1^{(i)}(r))}_{i∈[n]}` with W computed at
/// `j ∈ {n+1, ..., 2n−1}` (n−1 multiplications batched into 2 rounds).
/// Removes the prior `(a₀, b₀)` randomization since `f₁(r), f₂(r)` are
/// never opened during dVOPRF (5-Online.tex:280). Returns the aggregated
/// 5-RSS-tuple proof and a `CommStats` accounting both phases sequentially.
pub fn vip_parallel(
    per_prover_pairs: &[Vec<(Fp, Fp)>],
    per_prover_targets: &[Vec<(SubsetT, SubsetT)>],
    family: &SubsetFamily,
    modulus: &BigUint,
    pre_shared: &[PreSharedMaterial],
    rand_counter: &mut u64,
) -> (VipParallelOutput, CommStats) {
    let n = family.n;
    assert_eq!(
        per_prover_pairs.len(),
        n,
        "vip_parallel needs exactly n pair lists"
    );
    assert_eq!(per_prover_targets.len(), n);
    for (pp, pt) in per_prover_pairs.iter().zip(per_prover_targets.iter()) {
        assert_eq!(pp.len(), pt.len());
    }

    // γ is the common fold depth: `next_pow2` is derived from the max pair
    // count across all provers (balanced Λ can give slightly different
    // |Λ_i|; padding to a common length keeps the challenge sequence the
    // same for every instance).
    let max_len = per_prover_pairs
        .iter()
        .map(|p| p.len())
        .max()
        .unwrap_or(0);
    let padded_len = max_len.next_power_of_two().max(2);
    let gamma = (padded_len as f64).log2().ceil() as usize;

    // The fold/batch coefficients `r_k`/`ε_2..ε_γ`/`ε'_i` are drawn *before*
    // running any `vip_single` instance. Per Appendix L, the Fiat–Shamir
    // transform is not applicable to the batched VIP protocol, so every one
    // of these must be a genuine coin toss (`coin_toss`) rather than a
    // transcript hash. They are safe to draw here — and to open together in
    // a single `charge_f_coin_batch` round — because none of them is a
    // challenge *point* for a committed polynomial: `r_k` fold the q-polys
    // (all committed in the one VSS-send round below), and `ε_sigma`/`ε'`
    // only linearly combine already-committed σ shares across instances.
    // This matches Appendix M's round accounting ("FCoin to generate
    // ε'_1,...,ε'_n can be simultaneously called with FCoin at the last step
    // of the single VIP protocol"): one `r_Coin` round for the lot.
    //
    // `ρ` is deliberately NOT drawn here — see the note further down, right
    // before it is drawn: it is a challenge point for W/U/V and must stay
    // unpredictable until W's high points are fixed by Π_RSS.Mul, so it gets
    // its own `r_Coin` round *after* the multiplication.
    let r_ks: Vec<Fp> = (0..gamma)
        .map(|k| coin_toss(*rand_counter + k as u64, pre_shared, family, modulus))
        .collect();
    *rand_counter += gamma as u64;

    // ε_2..ε_γ: Protocol 4 step 14, shared across all n instances.
    let eps_sigma: Vec<Fp> = (0..gamma.saturating_sub(1))
        .map(|k| coin_toss(*rand_counter + k as u64, pre_shared, family, modulus))
        .collect();
    *rand_counter += eps_sigma.len() as u64;

    // ε'_i: Protocol 5 step 4 — the Σ-batch coefficients.
    let eps: Vec<Fp> = (0..n)
        .map(|i| coin_toss(*rand_counter + i as u64, pre_shared, family, modulus))
        .collect();
    *rand_counter += n as u64;

    let total_coins = gamma + eps_sigma.len() + n;
    let mut coin_net = SimulatedNetwork::new(n);
    charge_f_coin_batch(&mut coin_net, family, modulus, total_coins);
    let coin_comm = coin_net.stats();

    // Phase 1: n parallel `vip_single` invocations, all using the shared
    // (r_k, ε_sigma) coin-toss outputs above. Every prover's VSS sends land
    // on `current_round = 0` — one synchronous round regardless of n or γ.
    let mut vip_net = SimulatedNetwork::new(n);
    let mut bundles: Vec<VipBundle> = Vec::with_capacity(n);
    for prover in 0..n {
        let bundle = vip_single(
            prover,
            n,
            &per_prover_pairs[prover],
            &per_prover_targets[prover],
            padded_len,
            &r_ks,
            &eps_sigma,
            family,
            modulus,
            &mut vip_net,
        );
        bundles.push(bundle);
    }

    // Aggregated c = Σ_i c_i (local).
    let c_shares: Vec<RssShare> = (0..n)
        .map(|s| {
            let mut acc = bundles[0].c[s].clone();
            for i in 1..n {
                acc = acc.local_add(&bundles[i].c[s]);
            }
            acc
        })
        .collect();

    let mut comm = vip_net.stats();
    comm.merge(&coin_comm);

    // Σ-batch: Σ = Σ ε'_i · σ_i (local), using the F_coin-drawn ε' above.
    let sigma_shares: Vec<RssShare> = (0..n)
        .map(|s| {
            let mut acc = bundles[0].sigma[s].local_scalar_mul(&eps[0]);
            for i in 1..n {
                let term = bundles[i].sigma[s].local_scalar_mul(&eps[i]);
                acc = acc.local_add(&term);
            }
            acc
        })
        .collect();

    // §5.1 triple verification (5-Online.tex:702/728): U(x), V(x) of degree
    // n−1 through {(i, a_i)}_{i∈[n]} and {(i, b_i)}_{i∈[n]}. W(j) computed at
    // j ∈ {n+1, …, 2n−1} via secure multiplication — the n−1 mults run in
    // parallel via the batched A2T (2 rounds total). No (a₀, b₀)
    // randomization: 5-Online.tex:280 documents that a_i^(1), b_i^(1) are
    // never opened during dVOPRF, so the soundness margin from output
    // randomization is unnecessary.
    let compute_pts: Vec<usize> = (n + 1..=(2 * n - 1)).collect();

    // Interpolation abscissae for U, V: (1, 2, …, n). Constant across all
    // compute points and all parties — hoisted once.
    let uv_xs: Vec<Fp> = (0..n)
        .map(|i| Fp::new(BigUint::from((i + 1) as u32), modulus))
        .collect();

    let mut u_inputs: Vec<Vec<RssShare>> = Vec::with_capacity(compute_pts.len());
    let mut v_inputs: Vec<Vec<RssShare>> = Vec::with_capacity(compute_pts.len());
    for &j in &compute_pts {
        let j_fp = Fp::new(BigUint::from(j as u32), modulus);
        // Coefficients L_k(j) depend on (uv_xs, j_fp) — independent of party.
        let coeffs = lagrange_coeffs(&uv_xs, &j_fp, modulus);
        let u_j: Vec<RssShare> = (0..n)
            .map(|s| {
                let shares: Vec<&RssShare> =
                    (0..n).map(|i| &bundles[i].a1[s]).collect();
                lagrange_eval_with_coeffs(&shares, &coeffs)
            })
            .collect();
        let v_j: Vec<RssShare> = (0..n)
            .map(|s| {
                let shares: Vec<&RssShare> =
                    (0..n).map(|i| &bundles[i].b1[s]).collect();
                lagrange_eval_with_coeffs(&shares, &coeffs)
            })
            .collect();
        u_inputs.push(u_j);
        v_inputs.push(v_j);
    }

    // Fresh double sharings for each batched multiplication.
    let mut double_per_mul: Vec<Vec<_>> = Vec::with_capacity(compute_pts.len());
    for _ in 0..compute_pts.len() {
        let ds = (0..n)
            .map(|i| {
                generate_double_sharing(i, *rand_counter, &pre_shared[i], family, modulus)
            })
            .collect::<Vec<_>>();
        *rand_counter += 1;
        double_per_mul.push(ds);
    }

    let (w_extra_shares, mul_comm) = if compute_pts.is_empty() {
        // n=1 degenerate case: U,V,W are constants; no mults needed.
        (Vec::<Vec<RssShare>>::new(), CommStats::default())
    } else {
        let (extras, _records, mc) = rss_mul_batched_all_parties_with_record(
            &u_inputs,
            &v_inputs,
            &double_per_mul,
            family,
            modulus,
        );
        (extras, mc)
    };
    comm.merge(&mul_comm);

    // ρ: Protocol 5 step 11 — the evaluation challenge for the
    // `W(ρ) = U(ρ)·V(ρ)` check. Unlike the `r_k`/`ε_sigma`/`ε'` coefficients,
    // ρ is a challenge *point* for polynomials the parties have now finished
    // building: W is degree 2(n−1) and its high points W(n+1..2n−1) are only
    // fixed by the n−1 multiplications just completed. If a server learned ρ
    // before choosing its multiplication error δ_j it could solve the single
    // linear constraint W(ρ) = U(ρ)·V(ρ) for δ_j and pass while W ≠ U·V as
    // polynomials. So ρ is drawn — and opened — in its own genuine F_coin
    // round *after* Π_RSS.Mul (the analytical model's second `r_Coin`,
    // App.~efficiency-analysis-online). ρ ∉ {1, …, 2n−1} with overwhelming
    // probability (cryptographic prime → negligible collision).
    let rho = coin_toss(*rand_counter, pre_shared, family, modulus);
    *rand_counter += 1;
    let mut rho_coin_net = SimulatedNetwork::new(n);
    charge_f_coin_batch(&mut rho_coin_net, family, modulus, 1);
    comm.merge(&rho_coin_net.stats());

    // Build the 2n−1 known points for W(x): W(i) = d_i for i ∈ {1..n}, plus
    // W(j) computed for j ∈ {n+1, …, 2n−1}. W has degree 2(n−1).
    let mut all_w_points: Vec<(Fp, Vec<RssShare>)> = Vec::with_capacity(2 * n - 1);
    for i in 0..n {
        let xj = Fp::new(BigUint::from((i + 1) as u32), modulus);
        all_w_points.push((xj, bundles[i].d.clone()));
    }
    for (idx, &j) in compute_pts.iter().enumerate() {
        all_w_points.push((
            Fp::new(BigUint::from(j as u32), modulus),
            w_extra_shares[idx].clone(),
        ));
    }

    // W(ρ) coefficients: 2n−1 points. Hoisted once across all parties.
    let w_xs: Vec<Fp> = all_w_points.iter().map(|(x, _)| x.clone()).collect();
    let w_coeffs = lagrange_coeffs(&w_xs, &rho, modulus);
    let w_rho_shares: Vec<RssShare> = (0..n)
        .map(|s| {
            let shares: Vec<&RssShare> =
                all_w_points.iter().map(|(_, ws)| &ws[s]).collect();
            lagrange_eval_with_coeffs(&shares, &w_coeffs)
        })
        .collect();

    // U(ρ), V(ρ) share the (1, …, n) abscissae with the triple-verify Lagrange
    // above. Coefficients are the same across all parties — hoist.
    let uv_rho_coeffs = lagrange_coeffs(&uv_xs, &rho, modulus);
    let u_rho_shares: Vec<RssShare> = (0..n)
        .map(|s| {
            let shares: Vec<&RssShare> =
                (0..n).map(|i| &bundles[i].a1[s]).collect();
            lagrange_eval_with_coeffs(&shares, &uv_rho_coeffs)
        })
        .collect();
    let v_rho_shares: Vec<RssShare> = (0..n)
        .map(|s| {
            let shares: Vec<&RssShare> =
                (0..n).map(|i| &bundles[i].b1[s]).collect();
            lagrange_eval_with_coeffs(&shares, &uv_rho_coeffs)
        })
        .collect();

    // §4-Online line 260 + Appendix §A.2 line 1102: the five outputs are
    // opened to the client via F_RSS.Open. Keep the RSS structure — each
    // party contributes its full `C(n-1, t)` components; the bench delivery
    // path serialises them with the hash check of `Π_RSS.Open`.
    (
        VipParallelOutput {
            w_rho_shares,
            u_rho_shares,
            v_rho_shares,
            sigma_shares,
            c_shares,
        },
        comm,
    )
}

/// Client-side verification for `vip_parallel` (the W=UV and Σ=0 arms only).
/// Used standalone for the single-input Π_VIP^Prl variant, where the client
/// has no `ṽ_i^{(j)}` transcript to bind `c` against.
///
/// Matches §5-Online `fig:doprf_protocol` line 253 — reconstruction is RSS
/// (`ReplicatedSharing::reconstruct_from_party_shares`), not additive. For
/// the full Π_dVOPRF^m verdict (which adds the `c = Σ_j ε_j v^{(j)}` arm
/// from line 255), use [`client_verify_dvoprf`] instead.
pub fn client_verify_vip_parallel(
    out: &VipParallelOutput,
    modulus: &BigUint,
) -> VipResult {
    let w = ReplicatedSharing::reconstruct_from_party_shares(&out.w_rho_shares, modulus);
    let u = ReplicatedSharing::reconstruct_from_party_shares(&out.u_rho_shares, modulus);
    let v = ReplicatedSharing::reconstruct_from_party_shares(&out.v_rho_shares, modulus);
    let sigma = ReplicatedSharing::reconstruct_from_party_shares(&out.sigma_shares, modulus);
    if w != &u * &v {
        return VipResult::Abort;
    }
    if !sigma.is_zero() {
        return VipResult::Abort;
    }
    VipResult::Accept
}

/// Full Π_dVOPRF^m client verdict (`5-Online.tex:255`):
///   1. `W(ρ) = U(ρ)·V(ρ)`
///   2. `Σ = 0`
///   3. `c = Σ_{j=1..m} ε_j · Σ_{i=1..n} ṽ_i^{(j)}` — binds the proof to the
///      received `ṽ` transcript via the commit-then-hash ε derivation
///      (`5-Online.tex:238–240`).
///
/// `tilde_v` is shaped `n × m` (party-major). `ε_j` is rederived from
/// `tilde_v` using the same hash primitives as `compute_batch`, so a
/// post-hoc tamper of any `ṽ_i^{(j)}` shifts every `ε_j` and breaks arm 3.
pub fn client_verify_dvoprf(
    out: &VipParallelOutput,
    tilde_v: &[Vec<Fp>],
    modulus: &BigUint,
) -> VipResult {
    if client_verify_vip_parallel(out, modulus) == VipResult::Abort {
        return VipResult::Abort;
    }
    let n = tilde_v.len();
    let m = if n == 0 { 0 } else { tilde_v[0].len() };
    if m == 0 {
        return VipResult::Accept;
    }

    // ε_j = H(ρ_ε ‖ j), ρ_ε = H(h_1 ‖ … ‖ h_n), h_i = H(ṽ_i^{(1)} ‖ … ‖ ṽ_i^{(m)}).
    // Byte-identical to `compute_batch::compute_batch` (`compute_batch.rs:71–96`)
    // — same hash primitives, same big-endian `j` encoding.
    let mut buf = Vec::with_capacity(32 * n);
    for row in tilde_v {
        buf.extend_from_slice(&hash_field_elements(row));
    }
    let rho_eps = hash_bytes(&buf);
    let epsilons: Vec<Fp> = (0..m)
        .map(|j| {
            let mut b = Vec::with_capacity(40);
            b.extend_from_slice(&rho_eps);
            b.extend_from_slice(&(j as u64).to_be_bytes());
            let d = hash_bytes(&b);
            Fp::new(BigUint::from_bytes_be(&d) % modulus, modulus)
        })
        .collect();

    let mut expected_c = Fp::zero(modulus);
    for j in 0..m {
        let mut v_j = Fp::zero(modulus);
        for i in 0..n {
            v_j = &v_j + &tilde_v[i][j];
        }
        expected_c = &expected_c + &(&epsilons[j] * &v_j);
    }
    let c = ReplicatedSharing::reconstruct_from_party_shares(&out.c_shares, modulus);
    if c != expected_c {
        return VipResult::Abort;
    }
    VipResult::Accept
}

#[cfg(test)]
mod tests {
    use super::*;
    use vdoprf_offline::setup_pre_shared;
    use vdoprf_ss::{get_party_share, share, ReplicatedSharing};

    fn small_setup() -> (usize, usize, BigUint, SubsetFamily, Vec<PreSharedMaterial>) {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        (n, t, modulus, family, pre_shared)
    }

    /// Drive the full Π_VIP^Prl from a real RSS pair (a, b) and check that
    /// the aggregated c reconstructs to a·b and that the proof verifies.
    #[test]
    fn vip_parallel_on_real_rss_mul() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let mut rng = rand::thread_rng();

        let a = Fp::new(BigUint::from(7u32), &modulus);
        let b = Fp::new(BigUint::from(11u32), &modulus);
        let expected = &a * &b;

        let sa = share(&a, &family, &modulus, &mut rng);
        let sb = share(&b, &family, &modulus, &mut rng);

        let mut per_prover_pairs: Vec<Vec<(Fp, Fp)>> = Vec::with_capacity(n);
        let mut per_prover_targets: Vec<Vec<(SubsetT, SubsetT)>> = Vec::with_capacity(n);
        for i in 0..n {
            let pa = get_party_share(&sa, i, &family);
            let pb = get_party_share(&sb, i, &family);
            let (_sum, pairs_with_subsets) =
                vdoprf_ss::cross_multiply_with_pairs_and_subsets(&pa, &pb, &family);
            let mut pairs_plain: Vec<(Fp, Fp)> = Vec::with_capacity(pairs_with_subsets.len());
            let mut targets: Vec<(SubsetT, SubsetT)> =
                Vec::with_capacity(pairs_with_subsets.len());
            for (a_val, b_val, t_u, t_v) in pairs_with_subsets {
                pairs_plain.push((a_val, b_val));
                targets.push((t_u, t_v));
            }
            per_prover_pairs.push(pairs_plain);
            per_prover_targets.push(targets);
        }

        let mut rand_counter = 50_000u64;
        let (out, _comm) = vip_parallel(
            &per_prover_pairs,
            &per_prover_targets,
            &family,
            &modulus,
            &pre_shared,
            &mut rand_counter,
        );

        let c = ReplicatedSharing::reconstruct_from_party_shares(&out.c_shares, &modulus);
        assert_eq!(c.value, expected.value);
        assert_eq!(client_verify_vip_parallel(&out, &modulus), VipResult::Accept);
    }

    // ========================================================================
    // Property tests against §4-Online.tex.
    //
    // Π_VIP (`fig:vip_protocol`) per-prover invariants — lines 151 / 164-170:
    //   * c_i = Σ_{ℓ=1}^{L} a_i^{(ℓ)}·b_i^{(ℓ)}           (inner product)
    //   * d_i = a_i^{(1)}·b_i^{(1)} after γ=⌈log L⌉ folds  (compression)
    //   * σ_i = 0 under honest execution                   (consistency)
    //   * Output shape is the 5-tuple (d_i, a_i^{(1)}, b_i^{(1)}, σ_i, c_i).
    //
    // Π_VIP^Prl (`fig:vip_parallel_protocol`) — lines 192-211:
    //   * c = Σ c_i                                        (aggregation)
    //   * Σ = Σ ε'_i·σ_i = 0 when every instance is honest (batched check)
    //   * W(ρ) = U(ρ)·V(ρ) at ρ ∉ {0,…,2n}                 (triple verify)
    //   * Output shape: (W(ρ), U(ρ), V(ρ), Σ, c).
    //   * All n instances share the same F_Coin outputs.
    //
    // Client verdict (§4-Online line 274 / `client_verify_vip_parallel`):
    //   * Accept iff W(ρ) = U(ρ)·V(ρ) AND Σ = 0.
    //   * Otherwise Abort (Schwartz–Zippel soundness; line 168).
    // ========================================================================

    fn make_pairs(modulus: &BigUint, vals: &[(u32, u32)]) -> Vec<(Fp, Fp)> {
        vals.iter()
            .map(|(a, b)| {
                (
                    Fp::new(BigUint::from(*a), modulus),
                    Fp::new(BigUint::from(*b), modulus),
                )
            })
            .collect()
    }

    fn reconstruct(shares: &[RssShare], modulus: &BigUint) -> Fp {
        ReplicatedSharing::reconstruct_from_party_shares(shares, modulus)
    }

    /// Test helper: assign a target subset `(subsets[0], subsets[0])` to each
    /// test pair so it can be fed to the new `vip_single` API. The choice of
    /// subset is irrelevant to the VIP invariants we test — `c_i`, `d_i`, and
    /// `σ_i` are functions of plaintext / `q`-shares only, and the residual
    /// `a^{(1)}/b^{(1)}` reconstructs to the plaintext fold value regardless
    /// of which subset slot it was stashed under.
    fn targets_for_test_pairs(
        pairs: &[(Fp, Fp)],
        family: &SubsetFamily,
    ) -> Vec<(SubsetT, SubsetT)> {
        let t0 = family.subsets[0];
        (0..pairs.len()).map(|_| (t0, t0)).collect()
    }

    /// Test helper: build per-prover target-subset lists. Wrapper over
    /// `targets_for_test_pairs` indexed by prover.
    fn parallel_targets_for_test_pairs(
        per_prover_pairs: &[Vec<(Fp, Fp)>],
        family: &SubsetFamily,
    ) -> Vec<Vec<(SubsetT, SubsetT)>> {
        per_prover_pairs
            .iter()
            .map(|pairs| targets_for_test_pairs(pairs, family))
            .collect()
    }

    /// Test helper: derive a γ-long r_k sequence plus the γ-1 eps_sigma
    /// sequence for single-prover Π_VIP via genuine F_coin (`coin_toss`),
    /// mirroring exactly what `vip_parallel` does for the shared sequence,
    /// just applied standalone for unit tests.
    fn derive_round_coins(
        pairs_len: usize,
        counter: &mut u64,
        pre_shared: &[PreSharedMaterial],
        family: &SubsetFamily,
        modulus: &BigUint,
    ) -> (usize, Vec<Fp>, Vec<Fp>) {
        let padded_len = pairs_len.next_power_of_two().max(2);
        let gamma = (padded_len as f64).log2() as usize;
        let mut r_ks = Vec::with_capacity(gamma);
        for _ in 0..gamma {
            r_ks.push(coin_toss(*counter, pre_shared, family, modulus));
            *counter += 1;
        }
        let mut eps_sigma = Vec::with_capacity(gamma.saturating_sub(1));
        for _ in 0..gamma.saturating_sub(1) {
            eps_sigma.push(coin_toss(*counter, pre_shared, family, modulus));
            *counter += 1;
        }
        (padded_len, r_ks, eps_sigma)
    }

    /// Per-prover pair lists with a non-trivial per-prover inner product.
    fn make_parallel_pairs(
        n: usize,
        _family: &SubsetFamily,
        modulus: &BigUint,
    ) -> Vec<Vec<(Fp, Fp)>> {
        (0..n)
            .map(|i| {
                let base = (i as u32) + 1;
                vec![
                    (
                        Fp::new(BigUint::from(base), modulus),
                        Fp::new(BigUint::from(base + 1), modulus),
                    ),
                    (
                        Fp::new(BigUint::from(base + 2), modulus),
                        Fp::new(BigUint::from(base + 4), modulus),
                    ),
                ]
            })
            .collect()
    }

    /// Add 1 to one subset component of party 0's RSS share. RSS
    /// reconstruction sums one copy of every subset (`reconstruct_from_party_shares`
    /// uses first-write-wins), so a single-component bump shifts the
    /// reconstructed scalar by the same amount.
    fn bump_party0_add(shares: &mut [RssShare], modulus: &BigUint) {
        let one = Fp::one(modulus);
        // `shares` are per-party; party 0's first subset component is the
        // first write-winner for that subset.
        let (_subset, fp) = shares[0]
            .shares
            .iter_mut()
            .next()
            .expect("RssShare has at least one subset component");
        *fp = &*fp + &one;
    }

    // ---- Π_VIP single-prover correctness ----

    /// §4-Online line 151 / 170: c_i = Σ_{ℓ∈[L]} a_i^{(ℓ)}·b_i^{(ℓ)}.
    #[test]
    fn vip_single_c_equals_inner_product() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let pairs = make_pairs(&modulus, &[(3, 5), (7, 11), (13, 17)]);
        let mut expected = Fp::zero(&modulus);
        for (a, b) in &pairs {
            expected = &expected + &(a * b);
        }

        let mut net = SimulatedNetwork::new(n);
        let mut counter = 10_000u64;
        let (padded_len, r_ks, eps_sigma) =
            derive_round_coins(pairs.len(), &mut counter, &pre_shared, &family, &modulus);
        let pair_targets = targets_for_test_pairs(&pairs, &family);
        let out = vip_single(0, n, &pairs, &pair_targets, padded_len, &r_ks, &eps_sigma, &family, &modulus, &mut net);

        assert_eq!(reconstruct(&out.c, &modulus).value, expected.value);
    }

    /// §4-Online lines 165-167: after γ=⌈log L⌉ folds, (d_i, a_i^{(1)}, b_i^{(1)})
    /// satisfies d_i = a_i^{(1)}·b_i^{(1)} (the "compression" property).
    #[test]
    fn vip_single_d_equals_a1_times_b1_after_compression() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let pairs = make_pairs(&modulus, &[(2, 3), (5, 7), (11, 13), (17, 19)]);

        let mut net = SimulatedNetwork::new(n);
        let mut counter = 11_000u64;
        let (padded_len, r_ks, eps_sigma) =
            derive_round_coins(pairs.len(), &mut counter, &pre_shared, &family, &modulus);
        let pair_targets = targets_for_test_pairs(&pairs, &family);
        let out = vip_single(0, n, &pairs, &pair_targets, padded_len, &r_ks, &eps_sigma, &family, &modulus, &mut net);

        let d = reconstruct(&out.d, &modulus);
        let a1 = reconstruct(&out.a1, &modulus);
        let b1 = reconstruct(&out.b1, &modulus);
        assert_eq!(d.value, (&a1 * &b1).value);
    }

    /// §4-Online line 167: σ_i = 0 under honest execution — the ε_k-weighted
    /// (d − q(1) − q(2)) telescoping check accumulates to zero.
    #[test]
    fn vip_single_sigma_is_zero_when_honest() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let pairs = make_pairs(&modulus, &[(2, 3), (5, 7), (11, 13)]);

        let mut net = SimulatedNetwork::new(n);
        let mut counter = 12_000u64;
        let (padded_len, r_ks, eps_sigma) =
            derive_round_coins(pairs.len(), &mut counter, &pre_shared, &family, &modulus);
        let pair_targets = targets_for_test_pairs(&pairs, &family);
        let out = vip_single(0, n, &pairs, &pair_targets, padded_len, &r_ks, &eps_sigma, &family, &modulus, &mut net);

        assert!(reconstruct(&out.sigma, &modulus).is_zero());
    }

    /// §4-Online line 151: output is the 5-tuple
    /// (d_i, a_i^{(1)}, b_i^{(1)}, σ_i, c_i), with one RSS share per server.
    #[test]
    fn vip_single_output_is_five_rss_per_server() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let pairs = make_pairs(&modulus, &[(1, 2), (3, 4)]);
        let mut net = SimulatedNetwork::new(n);
        let mut counter = 13_000u64;
        let (padded_len, r_ks, eps_sigma) =
            derive_round_coins(pairs.len(), &mut counter, &pre_shared, &family, &modulus);
        let pair_targets = targets_for_test_pairs(&pairs, &family);
        let out = vip_single(0, n, &pairs, &pair_targets, padded_len, &r_ks, &eps_sigma, &family, &modulus, &mut net);

        assert_eq!(out.d.len(), n);
        assert_eq!(out.a1.len(), n);
        assert_eq!(out.b1.len(), n);
        assert_eq!(out.sigma.len(), n);
        assert_eq!(out.c.len(), n);
        assert_eq!(out.prover_id, 0);
    }

    /// Edge case L=1 (padded to 2). c = a·b, d = a^{(1)}·b^{(1)}, σ = 0 still.
    #[test]
    fn vip_single_l_equals_one() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let pairs = make_pairs(&modulus, &[(19, 23)]);

        let mut net = SimulatedNetwork::new(n);
        let mut counter = 14_000u64;
        let (padded_len, r_ks, eps_sigma) =
            derive_round_coins(pairs.len(), &mut counter, &pre_shared, &family, &modulus);
        let pair_targets = targets_for_test_pairs(&pairs, &family);
        let out = vip_single(0, n, &pairs, &pair_targets, padded_len, &r_ks, &eps_sigma, &family, &modulus, &mut net);

        assert_eq!(
            reconstruct(&out.c, &modulus).value,
            BigUint::from(19u32 * 23u32),
        );
        let d = reconstruct(&out.d, &modulus);
        let a1 = reconstruct(&out.a1, &modulus);
        let b1 = reconstruct(&out.b1, &modulus);
        assert_eq!(d.value, (&a1 * &b1).value);
        assert!(reconstruct(&out.sigma, &modulus).is_zero());
    }

    /// Non-power-of-two L (padded with (0,0)). All three invariants still hold.
    #[test]
    fn vip_single_non_power_of_two_l() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let pairs = make_pairs(&modulus, &[(1, 2), (3, 4), (5, 6)]); // L=3 pads to 4
        let mut expected = Fp::zero(&modulus);
        for (a, b) in &pairs {
            expected = &expected + &(a * b);
        }

        let mut net = SimulatedNetwork::new(n);
        let mut counter = 15_000u64;
        let (padded_len, r_ks, eps_sigma) =
            derive_round_coins(pairs.len(), &mut counter, &pre_shared, &family, &modulus);
        let pair_targets = targets_for_test_pairs(&pairs, &family);
        let out = vip_single(0, n, &pairs, &pair_targets, padded_len, &r_ks, &eps_sigma, &family, &modulus, &mut net);

        assert_eq!(reconstruct(&out.c, &modulus).value, expected.value);
        let d = reconstruct(&out.d, &modulus);
        let a1 = reconstruct(&out.a1, &modulus);
        let b1 = reconstruct(&out.b1, &modulus);
        assert_eq!(d.value, (&a1 * &b1).value);
        assert!(reconstruct(&out.sigma, &modulus).is_zero());
    }

    /// Larger L (= 8 ⇒ γ = 3 iterations). Invariants hold across multi-fold.
    #[test]
    fn vip_single_larger_l() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let pairs: Vec<(Fp, Fp)> = (1..=8u32)
            .map(|k| {
                (
                    Fp::new(BigUint::from(k), &modulus),
                    Fp::new(BigUint::from(k + 10), &modulus),
                )
            })
            .collect();
        let mut expected = Fp::zero(&modulus);
        for (a, b) in &pairs {
            expected = &expected + &(a * b);
        }

        let mut net = SimulatedNetwork::new(n);
        let mut counter = 16_000u64;
        let (padded_len, r_ks, eps_sigma) =
            derive_round_coins(pairs.len(), &mut counter, &pre_shared, &family, &modulus);
        let pair_targets = targets_for_test_pairs(&pairs, &family);
        let out = vip_single(0, n, &pairs, &pair_targets, padded_len, &r_ks, &eps_sigma, &family, &modulus, &mut net);

        assert_eq!(reconstruct(&out.c, &modulus).value, expected.value);
        let d = reconstruct(&out.d, &modulus);
        let a1 = reconstruct(&out.a1, &modulus);
        let b1 = reconstruct(&out.b1, &modulus);
        assert_eq!(d.value, (&a1 * &b1).value);
        assert!(reconstruct(&out.sigma, &modulus).is_zero());
    }

    // ---- Π_VIP^Prl aggregation + triple-verification invariants ----

    /// §4-Online line 197: c = Σ_{i∈[n]} c_i.
    #[test]
    fn vip_parallel_c_equals_sum_of_inner_products() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);
        let mut expected = Fp::zero(&modulus);
        for pairs in &per {
            for (a, b) in pairs {
                expected = &expected + &(a * b);
            }
        }

        let mut counter = 1_000u64;
        let (out, _) =
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter);

        assert_eq!(ReplicatedSharing::reconstruct_from_party_shares(&out.c_shares, &modulus).value, expected.value);
    }

    /// §4-Online line 200 + 167: Σ = Σ_i ε'_i·σ_i, and since σ_i = 0 for every
    /// honest instance, Σ reconstructs to 0.
    #[test]
    fn vip_parallel_sigma_is_zero_when_all_honest() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);

        let mut counter = 2_000u64;
        let (out, _) =
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter);

        assert!(ReplicatedSharing::reconstruct_from_party_shares(&out.sigma_shares, &modulus).is_zero());
    }

    /// §4-Online lines 204-209: W(x) of degree 2n is pinned by
    /// {W(i):=d_i}_{i∈[n]} ∪ {W(j) := U(j)V(j)}_{j∈{0,n+1,…,2n}}; at the
    /// challenge ρ ∉ {0,…,2n}, W(ρ) = U(ρ)·V(ρ) under honest execution.
    #[test]
    fn vip_parallel_w_rho_equals_u_rho_v_rho_when_honest() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);

        let mut counter = 3_000u64;
        let (out, _) =
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter);

        let w = ReplicatedSharing::reconstruct_from_party_shares(&out.w_rho_shares, &modulus);
        let u = ReplicatedSharing::reconstruct_from_party_shares(&out.u_rho_shares, &modulus);
        let v = ReplicatedSharing::reconstruct_from_party_shares(&out.v_rho_shares, &modulus);
        assert_eq!(w.value, (&u * &v).value);
    }

    /// §4-Online line 211: output is (W(ρ), U(ρ), V(ρ), Σ, c), each an
    /// additive share per server (one `Fp`).
    #[test]
    fn vip_parallel_output_is_five_rss_per_server() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);

        let mut counter = 4_000u64;
        let (out, _) =
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter);

        assert_eq!(out.w_rho_shares.len(), n);
        assert_eq!(out.u_rho_shares.len(), n);
        assert_eq!(out.v_rho_shares.len(), n);
        assert_eq!(out.sigma_shares.len(), n);
        assert_eq!(out.c_shares.len(), n);
    }

    /// §4-Online lines 192-193: "all instances share the same F_Coin outputs
    /// (r_k, ε_k)". Consequence: given the same pairs, pre-shared material,
    /// and starting counter, the reconstructed plaintexts (W(ρ), U(ρ), V(ρ),
    /// Σ, c) are fully determined (`coin_toss` is a pure function of
    /// `(counter, pre_shared)`). The per-share RSS randomness varies across
    /// runs, but the opened values agree.
    #[test]
    fn vip_parallel_reconstructions_determined_by_transcript_and_inputs() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);

        let run = || {
            let mut counter = 42u64;
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter).0
        };
        let a = run();
        let b = run();

        for (x, y) in [
            (&a.w_rho_shares, &b.w_rho_shares),
            (&a.u_rho_shares, &b.u_rho_shares),
            (&a.v_rho_shares, &b.v_rho_shares),
            (&a.sigma_shares, &b.sigma_shares),
            (&a.c_shares, &b.c_shares),
        ] {
            assert_eq!(
                ReplicatedSharing::reconstruct_from_party_shares(x, &modulus).value,
                ReplicatedSharing::reconstruct_from_party_shares(y, &modulus).value,
            );
        }
    }

    // ---- Client-side verdict (§4-Online line 274) ----

    /// Accept iff W(ρ) = U(ρ)·V(ρ) AND Σ = 0 — honest case.
    #[test]
    fn client_verify_accepts_honest_execution() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);

        let mut counter = 5_000u64;
        let (out, _) =
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter);

        assert_eq!(client_verify_vip_parallel(&out, &modulus), VipResult::Accept);
    }

    /// Tampering W(ρ) breaks the multiplicative check W(ρ) = U(ρ)·V(ρ).
    #[test]
    fn client_verify_rejects_tampered_w_rho() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);

        let mut counter = 6_000u64;
        let (mut out, _) =
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter);

        bump_party0_add(&mut out.w_rho_shares, &modulus);
        assert_eq!(client_verify_vip_parallel(&out, &modulus), VipResult::Abort);
    }

    /// Tampering U(ρ) breaks W(ρ) = U(ρ)·V(ρ).
    #[test]
    fn client_verify_rejects_tampered_u_rho() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);

        let mut counter = 7_000u64;
        let (mut out, _) =
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter);

        bump_party0_add(&mut out.u_rho_shares, &modulus);
        assert_eq!(client_verify_vip_parallel(&out, &modulus), VipResult::Abort);
    }

    /// Tampering V(ρ) breaks W(ρ) = U(ρ)·V(ρ).
    #[test]
    fn client_verify_rejects_tampered_v_rho() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);

        let mut counter = 8_000u64;
        let (mut out, _) =
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter);

        bump_party0_add(&mut out.v_rho_shares, &modulus);
        assert_eq!(client_verify_vip_parallel(&out, &modulus), VipResult::Abort);
    }

    /// §4-Online line 274: client aborts on Σ ≠ 0.
    #[test]
    fn client_verify_rejects_tampered_sigma() {
        let (n, _t, modulus, family, pre_shared) = small_setup();
        let per = make_parallel_pairs(n, &family, &modulus);
        let per_targets = parallel_targets_for_test_pairs(&per, &family);

        let mut counter = 9_000u64;
        let (mut out, _) =
            vip_parallel(&per, &per_targets, &family, &modulus, &pre_shared, &mut counter);

        bump_party0_add(&mut out.sigma_shares, &modulus);
        assert_eq!(client_verify_vip_parallel(&out, &modulus), VipResult::Abort);
    }

    // ---- Genuine F_coin realization (regression coverage for the
    // identical-r_k / thread_rng bugs this module used to have) ----

    /// The old `r_ks` derivation appended the same constant label
    /// (`b"r_k"`) every loop iteration and called `Transcript::challenge`
    /// (which does not mutate the hasher), so every `r_k` in the sequence
    /// came out bit-identical. `coin_toss` with a distinct counter per round
    /// must not reproduce that bug.
    #[test]
    fn coin_toss_r_ks_are_pairwise_distinct_across_rounds() {
        let (_n, _t, modulus, family, pre_shared) = small_setup();
        let mut counter = 20_000u64;
        let (_padded_len, r_ks, _eps_sigma) =
            derive_round_coins(5, &mut counter, &pre_shared, &family, &modulus);
        assert!(r_ks.len() >= 2, "test needs γ ≥ 2 to check pairwise distinctness");
        for i in 0..r_ks.len() {
            for j in (i + 1)..r_ks.len() {
                assert_ne!(
                    r_ks[i].value, r_ks[j].value,
                    "r_ks[{i}] and r_ks[{j}] must differ — each round must draw an independent F_coin value",
                );
            }
        }
    }

    /// `coin_toss` is a pure, deterministic function of `(counter,
    /// pre_shared)` — same counter reproduces the same value, distinct
    /// counters (with overwhelming probability) give distinct values.
    #[test]
    fn coin_toss_deterministic_and_counter_separated() {
        let (_n, _t, modulus, family, pre_shared) = small_setup();
        let a1 = coin_toss(30_000, &pre_shared, &family, &modulus);
        let a2 = coin_toss(30_000, &pre_shared, &family, &modulus);
        let b = coin_toss(30_001, &pre_shared, &family, &modulus);
        assert_eq!(a1.value, a2.value, "same counter must reproduce the same coin");
        assert_ne!(a1.value, b.value, "distinct counters must give distinct coins");
    }

    /// All of `vip_parallel`'s F_coin draws (`r_k` per round, `ε_2..ε_γ`,
    /// `ε'_i`, `ρ`) are opened in a single batched round
    /// (`charge_f_coin_batch`), matching the paper's own round-complexity
    /// accounting (Appendix M: "FCoin to generate ε'_1,...,ε'_n can be
    /// simultaneously called with FCoin at the last step of the single VIP
    /// protocol"). Consequently `comm.rounds` must not grow with γ = ⌈log
    /// L⌉ — only the byte count should. Regression guard against
    /// accidentally reverting to one `charge_f_coin` round per draw.
    #[test]
    fn vip_parallel_round_count_independent_of_gamma() {
        let (n, _t, modulus, family, pre_shared) = small_setup();

        // L = 2 pairs per prover ⇒ γ = 1.
        let small_per: Vec<Vec<(Fp, Fp)>> = (0..n)
            .map(|i| {
                let base = (i as u32) + 1;
                vec![
                    (Fp::new(BigUint::from(base), &modulus), Fp::new(BigUint::from(base + 1), &modulus)),
                    (Fp::new(BigUint::from(base + 2), &modulus), Fp::new(BigUint::from(base + 3), &modulus)),
                ]
            })
            .collect();
        let small_targets = parallel_targets_for_test_pairs(&small_per, &family);

        // L = 16 pairs per prover ⇒ γ = 4 — far more coin-toss draws
        // (r_ks alone goes from 1 to 4 values, plus 3 new ε_sigma draws),
        // but must cost the same number of rounds.
        let large_per: Vec<Vec<(Fp, Fp)>> = (0..n)
            .map(|i| {
                let base = (i as u32) * 100 + 1;
                (0..16)
                    .map(|k| {
                        (
                            Fp::new(BigUint::from(base + 2 * k), &modulus),
                            Fp::new(BigUint::from(base + 2 * k + 1), &modulus),
                        )
                    })
                    .collect()
            })
            .collect();
        let large_targets = parallel_targets_for_test_pairs(&large_per, &family);

        let mut counter_small = 40_000u64;
        let (_, comm_small) = vip_parallel(
            &small_per, &small_targets, &family, &modulus, &pre_shared, &mut counter_small,
        );
        let mut counter_large = 41_000u64;
        let (_, comm_large) = vip_parallel(
            &large_per, &large_targets, &family, &modulus, &pre_shared, &mut counter_large,
        );

        assert_eq!(
            comm_small.rounds, comm_large.rounds,
            "round count must not scale with γ once F_coin draws are batched into one round",
        );
    }
}
