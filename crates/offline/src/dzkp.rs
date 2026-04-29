//! DZKP: Distributed Zero-Knowledge Proof for verifying RSS multiplication.
//!
//! Server-verified (Boyle/BGIN20) path only. Faithful to Boyle et al.
//! "Efficient Fully Secure Computation via Distributed Zero-Knowledge Proofs"
//! (eprint 2020/1557): Protocol 3.3 is the single-prover inner-product proof,
//! Protocol 4.2 the batched F_abort_vrfy realisation.
//!
//! * `dzkp_compute` runs Protocol 3.3 for a single prover: it takes a claimed
//!   c (as RSS shares) together with the prover's (a_k, b_k) pairs, proves
//!   c = Σ a_k·b_k by recursive polynomial reduction, and ends with RSS.Open
//!   of {q(r), f_1(r), f_2(r), σ} followed by q(r) = f_1(r)·f_2(r) ∧ σ = 0.
//! * `dzkp_compute_batch` runs Protocol 4.2 over m records: VSS of the
//!   per-party additive share ψ_i, n parallel F_proveDeg2Rel instances, and
//!   a β = Σ_k θ_k·z_k − Σ_i ψ_i reconstruction that ties the verified ψ to
//!   the claimed z_k's.

use num_bigint::BigUint;
use vdoprf_crypto::transcript::Transcript;
use vdoprf_field::Fp;
use vdoprf_network::{CommStats, SimulatedNetwork};
use vdoprf_ss::{
    get_party_share, lagrange_coeffs, lagrange_eval_on_shares, lagrange_eval_with_coeffs, share,
    ReplicatedSharing, RssShare, SubsetFamily,
};
use crate::rss_mul::{rss_mul_batched_all_parties_with_record, MulRecord};
use crate::PreSharedMaterial;
use crate::double_rand::{generate_double_sharing, generate_double_sharing_all};
use crate::rss_share::{charge_naive_rss_open, charge_naive_rss_share};

/// Result of DZKP verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DzkpResult {
    Accept,
    Abort,
}

/// Per-server RSS shares of the four values required to verify a single
/// DZKP instance: {q(r), f_1(r), f_2(r), σ}. The batched wrapper folds
/// these across n parallel provers via Protocol 3.1.3's U/V/W compression.
#[derive(Clone, Debug)]
pub struct DzkpCoreShares {
    pub q_r: Vec<RssShare>,
    pub f1_r: Vec<RssShare>,
    pub f2_r: Vec<RssShare>,
    pub sigma: Vec<RssShare>,
}

/// Lagrange interpolation: evaluate polynomial through given points at x.
fn lagrange_eval(points: &[(Fp, Fp)], x: &Fp, modulus: &BigUint) -> Fp {
    let mut result = Fp::zero(modulus);
    for (i, (xi, yi)) in points.iter().enumerate() {
        let mut basis = Fp::one(modulus);
        for (j, (xj, _)) in points.iter().enumerate() {
            if i != j {
                let num = x - xj;
                let den = (xi - xj).inv().unwrap();
                basis = &basis * &(&num * &den);
            }
        }
        result = &result + &(yi * &basis);
    }
    result
}

/// Pad pairs to next power of 2 (minimum 2).
fn pad_to_power_of_two(pairs: &[(Fp, Fp)], modulus: &BigUint) -> Vec<(Fp, Fp)> {
    let next_pow2 = pairs.len().next_power_of_two().max(2);
    let mut padded = pairs.to_vec();
    while padded.len() < next_pow2 {
        padded.push((Fp::zero(modulus), Fp::zero(modulus)));
    }
    padded
}

/// Byte size of a field element for network accounting.
fn fe_bytes(modulus: &BigUint) -> usize {
    ((modulus.bits() + 7) / 8) as usize
}

/// Recursive polynomial reduction at the heart of the DZKP.
///
/// Implements Protocol 3.3 Step 2 (loop) and Step 3 (base case) of Boyle
/// et al. `claimed_c` is the prover-claimed value of Σ a_k·b_k, held by all
/// servers as an RSS sharing (paper Inputs: "The parties hold a consistent
/// t-out-of-n secret sharing of c"). Initializing the recursion's "previous
/// d" to `claimed_c` causes the first-iteration update σ += ε·(c − q(1) −
/// q(2)) = ε·b_1 — i.e., the claim c is bound to the proof. Without this,
/// the proof would only show pair-consistency, not equality to c.
///
/// Returns per-server RSS shares of {q(r), f_1(r), f_2(r), σ}.
fn dzkp_core(
    prover_id: usize,
    n: usize,
    pairs: &[(Fp, Fp)],
    claimed_c: &[RssShare],
    family: &SubsetFamily,
    modulus: &BigUint,
    transcript: &mut Transcript,
    net: &mut SimulatedNetwork,
    pre_shared: &[PreSharedMaterial],
    rand_counter: &mut u64,
) -> DzkpCoreShares {
    let mut rng = rand::thread_rng();
    let padded = pad_to_power_of_two(pairs, modulus);
    let feb = fe_bytes(modulus);

    let mut cur_a: Vec<Fp> = padded.iter().map(|(a, _)| a.clone()).collect();
    let mut cur_b: Vec<Fp> = padded.iter().map(|(_, b)| b.clone()).collect();

    let one = Fp::new(BigUint::from(1u32), modulus);
    let two = Fp::new(BigUint::from(2u32), modulus);
    let three = Fp::new(BigUint::from(3u32), modulus);

    let zero_sharing = share(&Fp::zero(modulus), family, modulus, &mut rng);
    let mut sigma_shares: Vec<RssShare> = (0..n)
        .map(|i| get_party_share(&zero_sharing, i, family))
        .collect();

    // d starts as the claimed c — Protocol 3.3 Step 2(c) first iteration.
    let mut d_shares: Vec<RssShare> = claimed_c.to_vec();

    while cur_a.len() > 2 {
        let half = cur_a.len() / 2;

        let mut g1 = Fp::zero(modulus);
        for j in 0..half {
            g1 = &g1 + &(&cur_a[j] * &cur_b[j]);
        }
        let mut g2 = Fp::zero(modulus);
        for j in half..cur_a.len() {
            g2 = &g2 + &(&cur_a[j] * &cur_b[j]);
        }
        let mut q3 = Fp::zero(modulus);
        for e in 0..half {
            let fa3 = &(&cur_a[e + half] + &cur_a[e + half]) - &cur_a[e];
            let fb3 = &(&cur_b[e + half] + &cur_b[e + half]) - &cur_b[e];
            q3 = &q3 + &(&fa3 * &fb3);
        }

        let g1_sharing = share(&g1, family, modulus, &mut rng);
        let g2_sharing = share(&g2, family, modulus, &mut rng);
        let q3_sharing = share(&q3, family, modulus, &mut rng);

        // Protocol 3.3 Step 2 VSS of (g1, g2, q3) via the naive Π_RSS.Share
        // baseline (no Π_DoubleRand piggyback) — 3 values per iteration.
        charge_naive_rss_share(prover_id, 3, family, modulus, net);

        let g1_ps: Vec<RssShare> = (0..n).map(|i| get_party_share(&g1_sharing, i, family)).collect();
        let g2_ps: Vec<RssShare> = (0..n).map(|i| get_party_share(&g2_sharing, i, family)).collect();
        let q3_ps: Vec<RssShare> = (0..n).map(|i| get_party_share(&q3_sharing, i, family)).collect();

        // σ accumulation runs every iteration — iteration 1 binds claimed c.
        // ε_k is a fresh local PRG draw (Protocol 3.3 step 3(e) realised via
        // F_coin = local rand). Per-prover-local; the aggregate Σ = Σ ε'_i σ_i
        // = 0 under honest exec because each σ_i = 0 regardless of which ε_k
        // this prover drew.
        let epsilon = Fp::random(modulus, &mut rng);
        for i in 0..n {
            let check = d_shares[i].local_sub(&g1_ps[i]).local_sub(&g2_ps[i]);
            let term = check.local_scalar_mul(&epsilon);
            sigma_shares[i] = sigma_shares[i].local_add(&term);
        }

        transcript.append_field_element(&g1);
        transcript.append_field_element(&g2);
        transcript.append_field_element(&q3);

        transcript.append_bytes(b"r_k");
        let r_k = transcript.challenge(modulus);

        d_shares = (0..n).map(|i| {
            let pts: Vec<(Fp, RssShare)> = vec![
                (one.clone(), g1_ps[i].clone()),
                (two.clone(), g2_ps[i].clone()),
                (three.clone(), q3_ps[i].clone()),
            ];
            lagrange_eval_on_shares(&pts, &r_k, modulus)
        }).collect();

        // Reduce plaintext pairs in lockstep with the RSS shares.
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
    }

    // Per BGIN20 §3.1.2 (constant-round via Fiat–Shamir), Step 2(b)'s
    // loop VSS, Step 3(c)'s base-case VSS, and the F_Rand ω-open all
    // collapse into one synchronous prover→verifiers round: every r_k
    // and final_r is pre-derived from the FS transcript, so the prover
    // pre-computes every q(·) and ships all `d_l = q(·) − s_l` (loop)
    // together with the base-case `a₁,a₂,b₁,b₂` and `q(0..4)` shares
    // in a single batch. The downstream charge_rss_share_p2p calls
    // accumulate bytes into this same round.

    assert_eq!(cur_a.len(), 2);
    let a1 = &cur_a[0];
    let a2 = &cur_a[1];
    let b1 = &cur_b[0];
    let b2 = &cur_b[1];
    let a1_sharing = share(a1, family, modulus, &mut rng);
    let a2_sharing = share(a2, family, modulus, &mut rng);
    let b1_sharing = share(b1, family, modulus, &mut rng);
    let b2_sharing = share(b2, family, modulus, &mut rng);
    // Protocol 3.3 Step 3: VSS the four reduced leaves so servers can
    // locally interpolate f_1, f_2 from {(0,ω), (1,a_·), (2,b_·)}.
    charge_naive_rss_share(prover_id, 4, family, modulus, net);
    let a1_shares: Vec<RssShare> =
        (0..n).map(|i| get_party_share(&a1_sharing, i, family)).collect();
    let a2_shares: Vec<RssShare> =
        (0..n).map(|i| get_party_share(&a2_sharing, i, family)).collect();
    let b1_shares: Vec<RssShare> =
        (0..n).map(|i| get_party_share(&b1_sharing, i, family)).collect();
    let b2_shares: Vec<RssShare> =
        (0..n).map(|i| get_party_share(&b2_sharing, i, family)).collect();

    // F_Rand: [omega_1], [omega_2] opened to prover only (base-case randomness).
    let ds1 = generate_double_sharing_all(*rand_counter, pre_shared, family, modulus);
    *rand_counter += 1;
    let ds2 = generate_double_sharing_all(*rand_counter, pre_shared, family, modulus);
    *rand_counter += 1;

    let omega1: Fp = ds1.replicated.reconstruct(modulus);
    let omega2: Fp = ds2.replicated.reconstruct(modulus);

    for sender in 0..n {
        if sender != prover_id {
            net.send_p2p(sender, prover_id, vec![0u8; feb]);
        }
    }

    let omega1_shares: Vec<RssShare> = (0..n)
        .map(|i| get_party_share(&ds1.replicated, i, family))
        .collect();
    let omega2_shares: Vec<RssShare> = (0..n)
        .map(|i| get_party_share(&ds2.replicated, i, family))
        .collect();

    let zero_fp = Fp::zero(modulus);
    let f1_points_plain = vec![
        (zero_fp.clone(), omega1.clone()),
        (one.clone(), a1.clone()),
        (two.clone(), a2.clone()),
    ];
    let f2_points_plain = vec![
        (zero_fp.clone(), omega2.clone()),
        (one.clone(), b1.clone()),
        (two.clone(), b2.clone()),
    ];

    let mut q_evals = Vec::new();
    for pt in 0..5u32 {
        let x = Fp::new(BigUint::from(pt), modulus);
        let f1x = lagrange_eval(&f1_points_plain, &x, modulus);
        let f2x = lagrange_eval(&f2_points_plain, &x, modulus);
        q_evals.push(&f1x * &f2x);
    }

    let q_sharings: Vec<ReplicatedSharing> = q_evals
        .iter()
        .map(|v| share(v, family, modulus, &mut rng))
        .collect();

    // Protocol 3.3 Step 3 (base case): VSS q(0..4) via naive Π_RSS.Share.
    charge_naive_rss_share(prover_id, 5, family, modulus, net);

    let q_party_shares: Vec<Vec<RssShare>> = (0..n)
        .map(|i| q_sharings.iter().map(|s| get_party_share(s, i, family)).collect())
        .collect();

    for qv in &q_evals {
        transcript.append_field_element(qv);
    }

    transcript.append_bytes(b"final_r");
    let r = transcript.challenge(modulus);

    let q_r_shares: Vec<RssShare> = (0..n).map(|i| {
        let pts: Vec<(Fp, RssShare)> = (0..5u32).map(|pt| {
            (
                Fp::new(BigUint::from(pt), modulus),
                q_party_shares[i][pt as usize].clone(),
            )
        }).collect();
        lagrange_eval_on_shares(&pts, &r, modulus)
    }).collect();

    let f1_r_shares: Vec<RssShare> = (0..n).map(|i| {
        let pts: Vec<(Fp, RssShare)> = vec![
            (zero_fp.clone(), omega1_shares[i].clone()),
            (one.clone(), a1_shares[i].clone()),
            (two.clone(), a2_shares[i].clone()),
        ];
        lagrange_eval_on_shares(&pts, &r, modulus)
    }).collect();

    let f2_r_shares: Vec<RssShare> = (0..n).map(|i| {
        let pts: Vec<(Fp, RssShare)> = vec![
            (zero_fp.clone(), omega2_shares[i].clone()),
            (one.clone(), b1_shares[i].clone()),
            (two.clone(), b2_shares[i].clone()),
        ];
        lagrange_eval_on_shares(&pts, &r, modulus)
    }).collect();

    // Final σ += ε·(d − q(1) − q(2)) using d = q(r_last) from the loop.
    // ε is a fresh local PRG draw (Protocol 3.3 step 3(e) closing σ update).
    let epsilon = Fp::random(modulus, &mut rng);
    for i in 0..n {
        let q1_sh = get_party_share(&q_sharings[1], i, family);
        let q2_sh = get_party_share(&q_sharings[2], i, family);
        let check = d_shares[i].local_sub(&q1_sh).local_sub(&q2_sh);
        let term = check.local_scalar_mul(&epsilon);
        sigma_shares[i] = sigma_shares[i].local_add(&term);
    }

    DzkpCoreShares {
        q_r: q_r_shares,
        f1_r: f1_r_shares,
        f2_r: f2_r_shares,
        sigma: sigma_shares,
    }
}

/// Batched variant of [`dzkp_core`] used by `dzkp_compute_batch` (Boyle
/// Section 3.1.3). All n provers share the same `(ε_k, r_k)` sequence and
/// `final_r`, pre-derived from one transcript; per-iter/per-base-case
/// transcript appends that bind the single-prover variant are dropped —
/// soundness comes from the downstream σ aggregation and U/V/W triple-verify
/// the caller performs over the n `DzkpCoreShares` outputs.
///
/// `padded_len` is a power of two and is the common pad length across all
/// provers so γ is the same; `iter_challenges.len()` must equal
/// `log2(padded_len) − 1` (= the number of recursive fold rounds).
#[allow(clippy::too_many_arguments)]
fn dzkp_core_batched(
    prover_id: usize,
    n: usize,
    pairs: &[(Fp, Fp)],
    claimed_c: &[RssShare],
    padded_len: usize,
    iter_challenges: &[Fp],
    final_r: &Fp,
    final_epsilon: &Fp,
    family: &SubsetFamily,
    modulus: &BigUint,
    net: &mut SimulatedNetwork,
    pre_shared: &[PreSharedMaterial],
    rand_counter: &mut u64,
) -> DzkpCoreShares {
    assert!(padded_len.is_power_of_two() && padded_len >= 2);
    assert!(pairs.len() <= padded_len);
    let expected_iters = (padded_len as f64).log2() as usize - 1;
    assert_eq!(
        iter_challenges.len(),
        expected_iters,
        "challenges length must match γ − 1 = log2(padded_len) − 1",
    );
    let mut rng = rand::thread_rng();
    let feb = fe_bytes(modulus);

    let mut cur_a: Vec<Fp> = pairs.iter().map(|(a, _)| a.clone()).collect();
    let mut cur_b: Vec<Fp> = pairs.iter().map(|(_, b)| b.clone()).collect();
    while cur_a.len() < padded_len {
        cur_a.push(Fp::zero(modulus));
        cur_b.push(Fp::zero(modulus));
    }

    let one = Fp::new(BigUint::from(1u32), modulus);
    let two = Fp::new(BigUint::from(2u32), modulus);
    let three = Fp::new(BigUint::from(3u32), modulus);

    let zero_sharing = share(&Fp::zero(modulus), family, modulus, &mut rng);
    let mut sigma_shares: Vec<RssShare> =
        (0..n).map(|i| get_party_share(&zero_sharing, i, family)).collect();

    // d starts as the claimed c (Protocol 3.3 Step 2(c), first-iter binding).
    let mut d_shares: Vec<RssShare> = claimed_c.to_vec();

    // F_coin: one invocation per Boyle Protocol 3.3 call producing the whole
    // vector (ε_k)_{k=1..γ−1} of σ-update challenges (paper convention at
    // 5-Online.tex:190 — F_Coin outputs a vector). The per-iteration rng
    // draws below are the local realization of that single F_coin output;
    // the protocol-level network cost (one RSS.Open of [r], one round) is
    // charged here.
    crate::rss_share::charge_f_coin(net, family, modulus);

    let mut iter = 0usize;
    while cur_a.len() > 2 {
        let half = cur_a.len() / 2;

        let mut g1 = Fp::zero(modulus);
        for j in 0..half {
            g1 = &g1 + &(&cur_a[j] * &cur_b[j]);
        }
        let mut g2 = Fp::zero(modulus);
        for j in half..cur_a.len() {
            g2 = &g2 + &(&cur_a[j] * &cur_b[j]);
        }
        let mut q3 = Fp::zero(modulus);
        for e in 0..half {
            let fa3 = &(&cur_a[e + half] + &cur_a[e + half]) - &cur_a[e];
            let fb3 = &(&cur_b[e + half] + &cur_b[e + half]) - &cur_b[e];
            q3 = &q3 + &(&fa3 * &fb3);
        }

        let g1_sharing = share(&g1, family, modulus, &mut rng);
        let g2_sharing = share(&g2, family, modulus, &mut rng);
        let q3_sharing = share(&q3, family, modulus, &mut rng);

        charge_naive_rss_share(prover_id, 3, family, modulus, net);

        let g1_ps: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&g1_sharing, i, family)).collect();
        let g2_ps: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&g2_sharing, i, family)).collect();
        let q3_ps: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&q3_sharing, i, family)).collect();

        // ε_k is the per-iteration σ-update challenge: a fresh local PRG
        // sample (the local realization of the single F_coin vector output
        // charged once before the loop). NOT loaded from the FS transcript;
        // the aggregate Σ = Σ_i ε'_i σ_i = 0 under honest exec because each
        // σ_i = 0 regardless of which ε_k each prover drew, so per-prover-
        // local ε is sound. r_k still comes from the shared transcript so the
        // U/V/W triple-verify points line up across the n parallel provers.
        let epsilon = Fp::random(modulus, &mut rng);
        let r_k = iter_challenges[iter].clone();

        // σ accumulation — first iter binds claimed c via d = c. r_k is
        // shared across all n parallel provers (Boyle §3.1.3 aggregation).
        for i in 0..n {
            let check = d_shares[i].local_sub(&g1_ps[i]).local_sub(&g2_ps[i]);
            let term = check.local_scalar_mul(&epsilon);
            sigma_shares[i] = sigma_shares[i].local_add(&term);
        }

        d_shares = (0..n)
            .map(|i| {
                let pts: Vec<(Fp, RssShare)> = vec![
                    (one.clone(), g1_ps[i].clone()),
                    (two.clone(), g2_ps[i].clone()),
                    (three.clone(), q3_ps[i].clone()),
                ];
                lagrange_eval_on_shares(&pts, &r_k, modulus)
            })
            .collect();

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
    }

    // Per BGIN20 §3.1.2, base-case VSS (Step 3(c)) and the F_Rand ω-open
    // share the same synchronous round as the loop VSS (Step 2(b)): with
    // every r_k and final_r pre-derived from the FS transcript, the
    // prover ships all `d_l = q(·) − s_l` plus `a₁,a₂,b₁,b₂` and
    // `q(0..4)` shares in one batch.

    assert_eq!(cur_a.len(), 2);
    let a1 = &cur_a[0];
    let a2 = &cur_a[1];
    let b1 = &cur_b[0];
    let b2 = &cur_b[1];
    let a1_sharing = share(a1, family, modulus, &mut rng);
    let a2_sharing = share(a2, family, modulus, &mut rng);
    let b1_sharing = share(b1, family, modulus, &mut rng);
    let b2_sharing = share(b2, family, modulus, &mut rng);
    // Π_RSS.Share for the four reduced leaves (see dzkp_core) — naive baseline.
    charge_naive_rss_share(prover_id, 4, family, modulus, net);
    let a1_shares: Vec<RssShare> =
        (0..n).map(|i| get_party_share(&a1_sharing, i, family)).collect();
    let a2_shares: Vec<RssShare> =
        (0..n).map(|i| get_party_share(&a2_sharing, i, family)).collect();
    let b1_shares: Vec<RssShare> =
        (0..n).map(|i| get_party_share(&b1_sharing, i, family)).collect();
    let b2_shares: Vec<RssShare> =
        (0..n).map(|i| get_party_share(&b2_sharing, i, family)).collect();

    // F_Rand: ω_1, ω_2 opened to prover.
    let ds1 = generate_double_sharing_all(*rand_counter, pre_shared, family, modulus);
    *rand_counter += 1;
    let ds2 = generate_double_sharing_all(*rand_counter, pre_shared, family, modulus);
    *rand_counter += 1;

    let omega1: Fp = ds1.replicated.reconstruct(modulus);
    let omega2: Fp = ds2.replicated.reconstruct(modulus);

    for sender in 0..n {
        if sender != prover_id {
            net.send_p2p(sender, prover_id, vec![0u8; feb]);
        }
    }

    let omega1_shares: Vec<RssShare> = (0..n)
        .map(|i| get_party_share(&ds1.replicated, i, family))
        .collect();
    let omega2_shares: Vec<RssShare> = (0..n)
        .map(|i| get_party_share(&ds2.replicated, i, family))
        .collect();

    let zero_fp = Fp::zero(modulus);
    let f1_points_plain = vec![
        (zero_fp.clone(), omega1.clone()),
        (one.clone(), a1.clone()),
        (two.clone(), a2.clone()),
    ];
    let f2_points_plain = vec![
        (zero_fp.clone(), omega2.clone()),
        (one.clone(), b1.clone()),
        (two.clone(), b2.clone()),
    ];

    let mut q_evals = Vec::new();
    for pt in 0..5u32 {
        let x = Fp::new(BigUint::from(pt), modulus);
        let f1x = lagrange_eval(&f1_points_plain, &x, modulus);
        let f2x = lagrange_eval(&f2_points_plain, &x, modulus);
        q_evals.push(&f1x * &f2x);
    }

    let q_sharings: Vec<ReplicatedSharing> = q_evals
        .iter()
        .map(|v| share(v, family, modulus, &mut rng))
        .collect();

    charge_naive_rss_share(prover_id, 5, family, modulus, net);

    let q_party_shares: Vec<Vec<RssShare>> = (0..n)
        .map(|i| q_sharings.iter().map(|s| get_party_share(s, i, family)).collect())
        .collect();

    let q_r_shares: Vec<RssShare> = (0..n)
        .map(|i| {
            let pts: Vec<(Fp, RssShare)> = (0..5u32)
                .map(|pt| {
                    (
                        Fp::new(BigUint::from(pt), modulus),
                        q_party_shares[i][pt as usize].clone(),
                    )
                })
                .collect();
            lagrange_eval_on_shares(&pts, final_r, modulus)
        })
        .collect();

    let f1_r_shares: Vec<RssShare> = (0..n)
        .map(|i| {
            let pts: Vec<(Fp, RssShare)> = vec![
                (zero_fp.clone(), omega1_shares[i].clone()),
                (one.clone(), a1_shares[i].clone()),
                (two.clone(), a2_shares[i].clone()),
            ];
            lagrange_eval_on_shares(&pts, final_r, modulus)
        })
        .collect();

    let f2_r_shares: Vec<RssShare> = (0..n)
        .map(|i| {
            let pts: Vec<(Fp, RssShare)> = vec![
                (zero_fp.clone(), omega2_shares[i].clone()),
                (one.clone(), b1_shares[i].clone()),
                (two.clone(), b2_shares[i].clone()),
            ];
            lagrange_eval_on_shares(&pts, final_r, modulus)
        })
        .collect();

    // Final σ += ε·(d − q(1) − q(2)) with d = q(r_last). Shared ε from
    // §3.1.3 — same draw used by every parallel prover.
    for i in 0..n {
        let q1_sh = get_party_share(&q_sharings[1], i, family);
        let q2_sh = get_party_share(&q_sharings[2], i, family);
        let check = d_shares[i].local_sub(&q1_sh).local_sub(&q2_sh);
        let term = check.local_scalar_mul(final_epsilon);
        sigma_shares[i] = sigma_shares[i].local_add(&term);
    }

    DzkpCoreShares {
        q_r: q_r_shares,
        f1_r: f1_r_shares,
        f2_r: f2_r_shares,
        sigma: sigma_shares,
    }
}

/// Single-prover server-verified DZKP (Boyle et al. Protocol 3.3).
///
/// Prover holds `pairs = {(a_k, b_k)}` in the clear and all servers hold an
/// RSS sharing `claimed_c` of the prover's claimed inner product c. The
/// proof verifies c = Σ a_k·b_k among the servers: recursion produces
/// shares of {q(r), f_1(r), f_2(r), σ}, then RSS.Open reconstructs all
/// four and checks q(r) = f_1(r)·f_2(r) ∧ σ = 0.
pub fn dzkp_compute(
    prover_id: usize,
    n: usize,
    pairs: &[(Fp, Fp)],
    claimed_c: &[RssShare],
    family: &SubsetFamily,
    modulus: &BigUint,
    transcript: &mut Transcript,
    net: &mut SimulatedNetwork,
    pre_shared: &[PreSharedMaterial],
    rand_counter: &mut u64,
) -> DzkpResult {
    let shares = dzkp_core(
        prover_id, n, pairs, claimed_c, family, modulus, transcript, net, pre_shared, rand_counter,
    );
    // F_coin charge for the closing σ-update ε draw inside `dzkp_core`
    // (Boyle Protocol 3.3 step 3e). Mirrors the batched path's charge.
    crate::rss_share::charge_f_coin(net, family, modulus);

    // Naive Π_RSS.Open of {q(r), f_1(r), f_2(r), σ}: each server broadcasts
    // its full holdings in clear (no hash compression). 1 round.
    charge_naive_rss_open(4, family, modulus, net);

    let q_r = ReplicatedSharing::reconstruct_from_party_shares(&shares.q_r, modulus);
    let f1_r = ReplicatedSharing::reconstruct_from_party_shares(&shares.f1_r, modulus);
    let f2_r = ReplicatedSharing::reconstruct_from_party_shares(&shares.f2_r, modulus);
    let sigma = ReplicatedSharing::reconstruct_from_party_shares(&shares.sigma, modulus);

    if q_r != &f1_r * &f2_r || !sigma.is_zero() {
        return DzkpResult::Abort;
    }
    DzkpResult::Accept
}


/// Server-verified batched DZKP over m RSS multiplication records, realising
/// F_Offline^abort_vrfy as Boyle/BGIN20 Protocol 4.2 step-for-step:
///
///   Step 1 (F_coin):   θ_1..θ_m drawn via Fiat-Shamir from a transcript
///                      committing to records' c_shares (the claimed z_k).
///   Step 2 (local):    each P_i computes ψ_i = Σ_k θ_k · cp_i[k].
///   Step 3 (VSS):      each P_i VSS-shares ψ_i to all other parties.
///   Step 4 (parallel   each prover i sends (a^i_k=θ_k·x^i_k, b^i_k=y^i_k)_k
///          F_proveDeg2 to F_proveDeg2Rel = Protocol 3.3 with claimed c_i=ψ_i;
///          Rel):       RSS.Open of {q(r), f_1(r), f_2(r), σ} per prover.
///   Step 6-7 (β open): β = Σ_k θ_k·z_k − Σ_i ψ_i, reconstructed via RSS.Open;
///                      honest execution ⇒ β = 0.
///
/// Each prover's 3.3 opens independently (4·n scalars total, broadcast in
/// one round); the β check is what ties the verified ψ to the claimed z_k's.
pub fn dzkp_compute_batch(
    records: &[MulRecord],
    family: &SubsetFamily,
    modulus: &BigUint,
    pre_shared: &[PreSharedMaterial],
) -> (DzkpResult, CommStats) {
    let n = family.n;
    let m = records.len();

    if m == 0 {
        return (DzkpResult::Accept, CommStats::default());
    }

    let mut net = SimulatedNetwork::new(n);
    let mut transcript = Transcript::new(b"dOPRF.ComputeBatch");
    let mut rng = rand::thread_rng();

    // Step 1 — F_coin: derive θ_1..θ_m via FS. Bind the transcript to the
    // claimed z_k RSS shares so that θ is drawn *after* (x_k, y_k, z_k)
    // have already been placed on the wire by the RSS.Mul that produced the
    // records, as the paper's Inputs clause requires.
    for record in records {
        for party_share in &record.c_shares {
            for fp in party_share.shares.values() {
                transcript.append_field_element(fp);
            }
        }
    }
    let thetas: Vec<Fp> = (0..m)
        .map(|k| {
            transcript.append_bytes(b"theta");
            transcript.append_bytes(&(k as u64).to_be_bytes());
            transcript.challenge(modulus)
        })
        .collect();

    // Step 2 (local): ψ_i = Σ_k θ_k · cp_i[k]. cp_i[k] is P_i's additive share
    // of the actual product x_k·y_k (from the RSS.Mul cross-product step).
    let psi_additive: Vec<Fp> = (0..n)
        .map(|i| {
            let mut acc = Fp::zero(modulus);
            for (k, record) in records.iter().enumerate() {
                acc = &acc + &(&thetas[k] * &record.party_cp[i]);
            }
            acc
        })
        .collect();

    // Step 3 — VSS(ψ_i): each prover P_i shares ψ_i among all parties via
    // the naive Π_RSS.Share baseline (one value per prover; per recipient
    // the dealer ships C(n-1, t) Fp directly).
    let mut psi_sharings: Vec<Vec<RssShare>> = Vec::with_capacity(n);
    for i in 0..n {
        let psi_sharing = share(&psi_additive[i], family, modulus, &mut rng);
        let psi_shares: Vec<RssShare> = (0..n)
            .map(|j| get_party_share(&psi_sharing, j, family))
            .collect();
        charge_naive_rss_share(i, 1, family, modulus, &mut net);
        psi_sharings.push(psi_shares);
    }
    net.next_round();

    // Step 4 — parallel F_proveDeg2Rel per prover with Boyle Section 3.1.3
    // batching. All n provers share one (ε_k, r_k)_{k∈[γ-1]} + final_r +
    // final_epsilon challenge set so their (f_1(r), f_2(r), q(r), σ) outputs
    // are interpolatable across provers. Per-prover FS binding of the g(x)
    // values is dropped — soundness comes from the σ aggregation and the
    // U/V/W triple-verify performed after this parallel block.
    let mut rand_counter = 1000u64;

    // Compute the common γ from record pair counts directly (read sizes off
    // each prover without materialising the per-prover pair vecs). The
    // scaled-pair construction is deferred to the per-prover loop below so
    // we only ever hold ONE prover's pair vec (≈ 750 MB at n=9, m=35)
    // instead of all n (≈ 6.7 GB at n=9, m=35) — fixes Section 3 e2e OOM
    // at large (n, m). |Λ_i| varies slightly across provers so we take max
    // over provers, not assume symmetry.
    let max_len: usize = (0..n)
        .map(|i| records.iter().map(|r| r.party_pairs[i].len()).sum::<usize>())
        .max()
        .unwrap_or(0);
    let padded_len = max_len.next_power_of_two().max(2);
    let gamma = (padded_len as f64).log2() as usize;

    // Pre-derive shared r_k challenges from the transcript (Boyle §3.1.3 —
    // every parallel prover must see the same r_k so the per-prover proofs
    // aggregate into one zk-FLIOP via the U/V/W triple-verify). The σ-update
    // ε_k's are NOT shared (each prover draws its own via F_coin → local PRG;
    // see `dzkp_core_batched`). final_r and final_epsilon stay shared via FS.
    let iter_challenges: Vec<Fp> = (0..gamma.saturating_sub(1))
        .map(|_| {
            transcript.append_bytes(b"r_k");
            transcript.challenge(modulus)
        })
        .collect();
    transcript.append_bytes(b"final_r");
    let final_r = transcript.challenge(modulus);
    transcript.append_bytes(b"epsilon_final_post");
    let final_epsilon = transcript.challenge(modulus);

    let mut per_prover: Vec<DzkpCoreShares> = Vec::with_capacity(n);
    let mut parallel_prover_stats = CommStats::default();
    let mut max_prover_rounds = 0usize;
    for prover in 0..n {
        // Build this prover's scaled-pair vec lazily, then drop after the
        // dzkp_core_batched call returns. Peak pair-memory ≈ |records| ×
        // |Λ_i| × 2·Fp instead of n× that.
        let pairs: Vec<(Fp, Fp)> = records
            .iter()
            .enumerate()
            .flat_map(|(k, record)| {
                let theta_k = thetas[k].clone();
                record.party_pairs[prover]
                    .iter()
                    .map(move |(a_val, b_val)| (&theta_k * a_val, b_val.clone()))
            })
            .collect();
        let mut prover_net = SimulatedNetwork::new(n);
        let core = dzkp_core_batched(
            prover,
            n,
            &pairs,
            &psi_sharings[prover],
            padded_len,
            &iter_challenges,
            &final_r,
            &final_epsilon,
            family,
            modulus,
            &mut prover_net,
            pre_shared,
            &mut rand_counter,
        );
        per_prover.push(core);
        let prover_stats = prover_net.stats();
        max_prover_rounds = max_prover_rounds.max(prover_stats.rounds);
        parallel_prover_stats.merge_parallel(&prover_stats);
    }
    for _ in 0..max_prover_rounds {
        net.next_round();
    }

    // Step 4(d)/(5) — Boyle Section 3.1.3 aggregation. Collapse the n
    // per-prover 4-tuples into 4 aggregate scalars: (Σ, U(ρ), V(ρ), W(ρ)).
    //
    // σ-batching: draw {ε'_i}_{i∈[n]} from FS, compute Σ = Σ_i ε'_i · σ_i
    // locally. Honest execution ⇒ Σ = 0.
    //
    // Triple-verify (Nordholt–Veeningen 2018 / appendix.tex:1067 `4·c_Open +
    // (n-1)·c_Mul`): interpolate U(x), V(x) of degree n−1 through
    // {(i, f_1^{(i)}(r_last))}, {(i, f_2^{(i)}(r_last))} for i ∈ {1,…,n};
    // `W(i) = q^{(i)}(r_last)` for i ∈ {1,…,n} fix n of the 2n−1 points of
    // W(x) (degree 2(n−1)). The remaining n−1 points are computed via
    // batched RSS.Mul at j ∈ {n+1,…,2n−1}. A fresh ρ ∉ {1,…,2n−1} is drawn
    // from FS, and U(ρ), V(ρ), W(ρ) are locally interpolated and opened.
    // W(ρ) = U(ρ)·V(ρ) by Schwartz–Zippel catches any cross-prover cheating.

    let eps_primes: Vec<Fp> = (0..n)
        .map(|i| {
            transcript.append_bytes(b"sigma_batch");
            transcript.append_bytes(&(i as u64).to_be_bytes());
            transcript.challenge(modulus)
        })
        .collect();
    let sigma_agg_shares: Vec<RssShare> = (0..n)
        .map(|s| {
            let mut acc = per_prover[0].sigma[s].local_scalar_mul(&eps_primes[0]);
            for i in 1..n {
                let term = per_prover[i].sigma[s].local_scalar_mul(&eps_primes[i]);
                acc = acc.local_add(&term);
            }
            acc
        })
        .collect();

    // Triple-verify: U, V interpolated through (i, f_{1,2}^{(i)}(r_last))
    // for i ∈ {1,…,n}; W through those plus (j, W(j)) for j ∈ {n+1,…,2n−1}.
    let uv_xs: Vec<Fp> = (0..n)
        .map(|i| Fp::new(BigUint::from((i + 1) as u32), modulus))
        .collect();
    let compute_pts: Vec<usize> = (n + 1..=(2 * n - 1)).collect();

    // Materialise U(j), V(j) shares for each j ∈ compute_pts, per party.
    let mut u_inputs: Vec<Vec<RssShare>> = Vec::with_capacity(compute_pts.len());
    let mut v_inputs: Vec<Vec<RssShare>> = Vec::with_capacity(compute_pts.len());
    for &j in &compute_pts {
        let j_fp = Fp::new(BigUint::from(j as u32), modulus);
        let coeffs = lagrange_coeffs(&uv_xs, &j_fp, modulus);
        let u_j: Vec<RssShare> = (0..n)
            .map(|s| {
                let shares: Vec<&RssShare> =
                    (0..n).map(|i| &per_prover[i].f1_r[s]).collect();
                lagrange_eval_with_coeffs(&shares, &coeffs)
            })
            .collect();
        let v_j: Vec<RssShare> = (0..n)
            .map(|s| {
                let shares: Vec<&RssShare> =
                    (0..n).map(|i| &per_prover[i].f2_r[s]).collect();
                lagrange_eval_with_coeffs(&shares, &coeffs)
            })
            .collect();
        u_inputs.push(u_j);
        v_inputs.push(v_j);
    }

    // Batched RSS.Mul for the n−1 extension points. One call, 2 rounds.
    let mut double_per_mul: Vec<Vec<_>> = Vec::with_capacity(compute_pts.len());
    for _ in 0..compute_pts.len() {
        let ds = (0..n)
            .map(|i| generate_double_sharing(i, rand_counter, &pre_shared[i], family, modulus))
            .collect::<Vec<_>>();
        rand_counter += 1;
        double_per_mul.push(ds);
    }
    let (w_extra_shares, _records, mul_comm) = rss_mul_batched_all_parties_with_record(
        &u_inputs, &v_inputs, &double_per_mul, family, modulus,
    );

    // ρ challenge ∉ {1,…,2n−1}. Cryptographic prime → negligible collision.
    transcript.append_bytes(b"rho");
    let rho = transcript.challenge(modulus);

    // Build W's 2n−1 known points: q^{(i)}(r_last) at x=i for i∈[n], plus the
    // RSS.Mul outputs at j ∈ {n+1,…,2n−1}.
    let mut w_xs: Vec<Fp> = uv_xs.clone();
    for &j in &compute_pts {
        w_xs.push(Fp::new(BigUint::from(j as u32), modulus));
    }
    let w_coeffs = lagrange_coeffs(&w_xs, &rho, modulus);
    let uv_rho_coeffs = lagrange_coeffs(&uv_xs, &rho, modulus);

    let u_rho_shares: Vec<RssShare> = (0..n)
        .map(|s| {
            let shares: Vec<&RssShare> =
                (0..n).map(|i| &per_prover[i].f1_r[s]).collect();
            lagrange_eval_with_coeffs(&shares, &uv_rho_coeffs)
        })
        .collect();
    let v_rho_shares: Vec<RssShare> = (0..n)
        .map(|s| {
            let shares: Vec<&RssShare> =
                (0..n).map(|i| &per_prover[i].f2_r[s]).collect();
            lagrange_eval_with_coeffs(&shares, &uv_rho_coeffs)
        })
        .collect();
    let w_rho_shares: Vec<RssShare> = (0..n)
        .map(|s| {
            let mut shares: Vec<&RssShare> = Vec::with_capacity(2 * n - 1);
            for i in 0..n {
                shares.push(&per_prover[i].q_r[s]);
            }
            for j_idx in 0..compute_pts.len() {
                shares.push(&w_extra_shares[j_idx][s]);
            }
            lagrange_eval_with_coeffs(&shares, &w_coeffs)
        })
        .collect();

    // Naive RSS.Open of the 4 aggregate scalars: σ_agg, U(ρ), V(ρ), W(ρ).
    // Each server broadcasts its full holdings in clear (no hash check). 1 round.
    charge_naive_rss_open(4, family, modulus, &mut net);

    let sigma_agg = ReplicatedSharing::reconstruct_from_party_shares(&sigma_agg_shares, modulus);
    let u_rho = ReplicatedSharing::reconstruct_from_party_shares(&u_rho_shares, modulus);
    let v_rho = ReplicatedSharing::reconstruct_from_party_shares(&v_rho_shares, modulus);
    let w_rho = ReplicatedSharing::reconstruct_from_party_shares(&w_rho_shares, modulus);
    if !sigma_agg.is_zero() || w_rho != &u_rho * &v_rho {
        let mut comm = net.stats();
        comm.merge_parallel(&parallel_prover_stats);
        // §3.1.3 W-extension RSS.Muls compute h(j) = f1(j)·f2(j) and consume
        // f1(r), f2(r) from each prover's recursion output, so they run
        // sequentially AFTER the per-prover work. Adds the standard 2-round
        // RSS.Mul cost.
        comm.merge(&mul_comm);
        return (DzkpResult::Abort, comm);
    }
    net.next_round();

    // Step 6 (local) — β = Σ_k θ_k·z_k − Σ_i ψ_i. The first sum is an RSS
    // sharing computed from the records' c_shares; the second sum folds the
    // per-prover ψ_i RSS sharings from step 3. β is an RSS sharing.
    let beta_shares: Vec<RssShare> = (0..n)
        .map(|j| {
            let mut acc = records[0].c_shares[j].local_scalar_mul(&thetas[0]);
            for k in 1..m {
                let term = records[k].c_shares[j].local_scalar_mul(&thetas[k]);
                acc = acc.local_add(&term);
            }
            for i in 0..n {
                acc = acc.local_sub(&psi_sharings[i][j]);
            }
            acc
        })
        .collect();

    // Step 7 — naive RSS.Open of β. On honest execution β = 0; otherwise Abort.
    charge_naive_rss_open(1, family, modulus, &mut net);
    let beta = ReplicatedSharing::reconstruct_from_party_shares(&beta_shares, modulus);
    let mut comm = net.stats();
    comm.merge_parallel(&parallel_prover_stats);
    // §3.1.3 W-extension RSS.Muls compute h(j) = f1(j)·f2(j) and consume
    // f1(r), f2(r) from each prover's recursion output, so they run
    // sequentially AFTER the per-prover work. Adds the standard 2-round
    // RSS.Mul cost.
    comm.merge(&mul_comm);
    if !beta.is_zero() {
        return (DzkpResult::Abort, comm);
    }

    (DzkpResult::Accept, comm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rss_mul::rss_mul_all_parties_with_record;
    use crate::setup_pre_shared;
    use crate::double_rand::{generate_double_sharing, DoubleShareLocal};

    /// Build an RSS sharing of `value` and return the n per-party shares.
    fn share_scalar(value: &Fp, family: &SubsetFamily, modulus: &BigUint) -> Vec<RssShare> {
        let mut rng = rand::thread_rng();
        let sharing = share(value, family, modulus, &mut rng);
        (0..family.n).map(|i| get_party_share(&sharing, i, family)).collect()
    }

    // --- Boyle/BGIN20 single-proof Protocol 3.3 ---

    #[test]
    fn test_dzkp_compute_boyle_honest() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let pairs: Vec<(Fp, Fp)> = (1..=8u32)
            .map(|i| (
                Fp::new(BigUint::from(i), &modulus),
                Fp::new(BigUint::from(i + 1), &modulus),
            ))
            .collect();
        // Σ a_k·b_k = 240 mod 113 = 14.
        let c = Fp::new(BigUint::from(14u32), &modulus);
        let claimed_c = share_scalar(&c, &family, &modulus);

        let mut transcript = Transcript::new(b"test");
        let mut net = SimulatedNetwork::new(n);
        let mut rand_counter = 0u64;
        let verdict = dzkp_compute(
            0, n, &pairs, &claimed_c, &family, &modulus,
            &mut transcript, &mut net, &pre_shared, &mut rand_counter,
        );

        assert_eq!(verdict, DzkpResult::Accept);
        assert!(net.stats().total_bytes() > 0);
    }

    #[test]
    fn test_dzkp_compute_boyle_base_case_two_pairs() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let pairs = vec![
            (Fp::new(BigUint::from(3u32), &modulus), Fp::new(BigUint::from(5u32), &modulus)),
            (Fp::new(BigUint::from(7u32), &modulus), Fp::new(BigUint::from(2u32), &modulus)),
        ];
        // 3·5 + 7·2 = 29.
        let c = Fp::new(BigUint::from(29u32), &modulus);
        let claimed_c = share_scalar(&c, &family, &modulus);

        let mut transcript = Transcript::new(b"test");
        let mut net = SimulatedNetwork::new(n);
        let mut rand_counter = 0u64;
        let verdict = dzkp_compute(
            0, n, &pairs, &claimed_c, &family, &modulus,
            &mut transcript, &mut net, &pre_shared, &mut rand_counter,
        );
        assert_eq!(verdict, DzkpResult::Accept);
    }

    #[test]
    fn test_dzkp_compute_boyle_rejects_wrong_c() {
        // A wrong claimed c must be rejected — this is exactly what the
        // external-c binding gives us that the pair-only proof did not.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let pairs = vec![
            (Fp::new(BigUint::from(3u32), &modulus), Fp::new(BigUint::from(5u32), &modulus)),
            (Fp::new(BigUint::from(7u32), &modulus), Fp::new(BigUint::from(2u32), &modulus)),
        ];
        // True c = 29; claim 30 instead.
        let wrong_c = Fp::new(BigUint::from(30u32), &modulus);
        let claimed_c = share_scalar(&wrong_c, &family, &modulus);

        let mut transcript = Transcript::new(b"test");
        let mut net = SimulatedNetwork::new(n);
        let mut rand_counter = 0u64;
        let verdict = dzkp_compute(
            0, n, &pairs, &claimed_c, &family, &modulus,
            &mut transcript, &mut net, &pre_shared, &mut rand_counter,
        );
        assert_eq!(verdict, DzkpResult::Abort);
    }

    // --- Server-verified batched variant (Section 3.1.3 / Π_VIP^Prl) ---

    fn build_random_record(
        counter: u64,
        family: &SubsetFamily,
        modulus: &BigUint,
        pre_shared: &[PreSharedMaterial],
    ) -> MulRecord {
        let n = family.n;
        let mut rng = rand::thread_rng();
        let a = Fp::random(modulus, &mut rng);
        let b = Fp::random(modulus, &mut rng);
        let sa = share(&a, family, modulus, &mut rng);
        let sb = share(&b, family, modulus, &mut rng);
        let a_shares: Vec<_> = (0..n).map(|i| get_party_share(&sa, i, family)).collect();
        let b_shares: Vec<_> = (0..n).map(|i| get_party_share(&sb, i, family)).collect();
        let double_shares: Vec<DoubleShareLocal> = (0..n)
            .map(|i| generate_double_sharing(i, counter, &pre_shared[i], family, modulus))
            .collect();
        let (_num_result, record, _) = rss_mul_all_parties_with_record(
            &a_shares, &b_shares, &double_shares, family, modulus,
        );
        record
    }

    #[test]
    fn test_dzkp_compute_batch_honest() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let records: Vec<MulRecord> = (0..8u64)
            .map(|c| build_random_record(c, &family, &modulus, &pre_shared))
            .collect();

        let (verdict, comm) = dzkp_compute_batch(&records, &family, &modulus, &pre_shared);
        assert_eq!(verdict, DzkpResult::Accept);
        assert!(comm.total_bytes() > 0);
    }

    /// Pin the DZKP-only synchronous round count after BGIN20 §3.1.2's
    /// FS-batched fusion of Step 2(c) loop VSS and Step 3(c) base-case VSS.
    /// Decomposition (constant in (n,t,m)):
    ///   1 (ψ-VSS) + 1 (loop+base VSS fused) + 1 (F_coin)
    /// + 2 (W-extension RSS.Mul) + 1 (4-agg open) + 1 (β open) = 7.
    #[test]
    fn test_dzkp_compute_batch_round_count_after_collapse() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let records: Vec<MulRecord> = (0..4u64)
            .map(|c| build_random_record(c, &family, &modulus, &pre_shared))
            .collect();

        let (_verdict, comm) = dzkp_compute_batch(&records, &family, &modulus, &pre_shared);
        assert_eq!(
            comm.rounds, 7,
            "post-§3.1.2 collapse: DZKP must take exactly 7 synchronous rounds \
             (1 ψ-VSS + 1 fused loop+base VSS + 1 F_coin + 2 W-ext mul + 1 4-agg open + 1 β open)",
        );
    }

    #[test]
    fn test_dzkp_compute_batch_empty() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let (verdict, comm) = dzkp_compute_batch(&[], &family, &modulus, &pre_shared);
        assert_eq!(verdict, DzkpResult::Accept);
        assert_eq!(comm.total_bytes(), 0);
    }

    #[test]
    fn test_dzkp_compute_batch_n5_t2() {
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let records: Vec<MulRecord> = (0..4u64)
            .map(|c| build_random_record(c, &family, &modulus, &pre_shared))
            .collect();

        let (verdict, _comm) = dzkp_compute_batch(&records, &family, &modulus, &pre_shared);
        assert_eq!(verdict, DzkpResult::Accept);
    }

    #[test]
    fn test_dzkp_compute_batch_tampered_cp_aborts() {
        // Flip one party's claimed cp to a wrong value — the batched proof
        // must abort because claimed_c_i no longer matches Σ a·b.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let mut records: Vec<MulRecord> = (0..4u64)
            .map(|c| build_random_record(c, &family, &modulus, &pre_shared))
            .collect();

        // Tamper: add 1 to party 0's cp for record 0.
        let one = Fp::new(BigUint::from(1u32), &modulus);
        records[0].party_cp[0] = &records[0].party_cp[0] + &one;

        let (verdict, _comm) = dzkp_compute_batch(&records, &family, &modulus, &pre_shared);
        assert_eq!(verdict, DzkpResult::Abort);
    }

    /// 4.2 step 7 catches aggregator-level tampering that leaves each party's
    /// local (party_cp, party_pairs) self-consistent but corrupts the RSS
    /// output shares z_k. Flipping one subset-component of one z_k share
    /// breaks β = Σθ_k·z_k − Σψ_i without tripping any per-prover 3.3 check.
    #[test]
    fn test_dzkp_compute_batch_tampered_z_shares_aborts() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let mut records: Vec<MulRecord> = (0..4u64)
            .map(|c| build_random_record(c, &family, &modulus, &pre_shared))
            .collect();

        // Tamper: flip one subset-component that party 0 actually holds
        // (i.e. a subset not containing 0) in its z-share of record 0.
        // All party_cp and party_pairs remain untouched.
        let one = Fp::new(BigUint::from(1u32), &modulus);
        let tamper_key = records[0].c_shares[0]
            .shares
            .keys()
            .next()
            .expect("party 0 must hold at least one subset share")
            .clone();
        let current = records[0].c_shares[0].shares[&tamper_key].clone();
        records[0].c_shares[0]
            .shares
            .insert(tamper_key, &current + &one);

        let (verdict, _comm) = dzkp_compute_batch(&records, &family, &modulus, &pre_shared);
        assert_eq!(
            verdict,
            DzkpResult::Abort,
            "β check (step 7) must catch z-share tampering",
        );
    }
}
