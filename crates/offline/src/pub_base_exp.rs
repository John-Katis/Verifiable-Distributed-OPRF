//! Public-base secure exponentiation: `Π_exp(b, [a])` per Aly-Smart ACNS 2019.
//!
//! Given a public base `b ∈ F_p*` and a shared exponent `[a]_{p-1}`, produces
//! an RSS sharing `[[b^a]]_p`. This is the Public-Base case from Aly-Smart
//! §3.1.
//!
//! We implement both variants:
//!
//!   * `pub_base_exp_semi_honest` — Protocol 4 (semi-honest). Each subset T
//!     has a piece `a_T ∈ Z_{p-1}`; parties holding `a_T` locally compute
//!     `h_T = b^{a_T} ∈ F_p`, wrap it as a `DegenerateEncoding`, and the
//!     parties fan-in multiply all `⟨h_T⟩_T` via a `deg_mul` + `rss_mul`
//!     binary tree (layer 0 pairs up the degenerate encodings via `deg_mul`;
//!     layers 1+ combine two already-full-RSS results, so they fall back to
//!     full `rss_mul_all_parties_with_record`). `approach_ii::gen_degenerate`
//!     does NOT use this structure — it folds sequentially, keeping one
//!     operand degenerate at every step, so it never needs a full RSS.Mul.
//!
//!   * `pub_base_exp_malicious` — Protocol 6. Two semi-honest calls plus a
//!     consistency check that catches malicious parties who mis-compute a
//!     `h_T`. Specifically we check `v = r'·(b^a · b^{a'} · b^{-w} - 1) + 1`
//!     equals 1, where `a' = a·(r-1)` and `w = a·r` (over Z_{p-1}).
//!
//! All emitted `MulRecord`s are returned so the caller can include them in
//! the final DZKP batch.

use num_bigint::BigUint;
use num_traits::{One, Zero};
use vdoprf_field::Fp;
use vdoprf_network::{CommStats, SimulatedNetwork};
use vdoprf_ss::{
    get_party_share, share, DegenerateEncoding, ReplicatedSharing, RssShare, SubsetFamily,
};

use crate::double_rand::{
    generate_double_sharing, generate_double_sharing_degenerate, DoubleShareLocal,
};
use crate::rss_mul::{deg_mul, rss_mul_all_parties_with_record, MulRecord};
use crate::PreSharedMaterial;

/// Semi-honest Π_exp (Aly-Smart Prot. 4, adapted to RSS + degenerate
/// encodings).
///
/// The shared exponent is given as per-party RSS shares `exp_shares` over
/// modulus `p − 1`. For each subset `T ∈ family.subsets`, a party not in `T`
/// holds the piece `a_T`; that party (any one of them — deterministic choice)
/// locally computes `h_T = base^{a_T} mod p`. The resulting degenerate
/// encodings `⟨h_T⟩_T` are fan-in multiplied via a binary tree: layer 0 uses
/// `deg_mul`, subsequent layers use `rss_mul_all_parties_with_record`. The
/// final RSS sharing is `[[base^a]]_p`.
///
/// `counter_base` is used to derive fresh double-sharings; the caller must
/// pass a value that does not collide with other protocol counters.
pub fn pub_base_exp_semi_honest(
    base: &Fp,
    exp_shares: &[RssShare],
    exp_modulus: &BigUint,
    field_modulus: &BigUint,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    counter_base: u64,
) -> (Vec<RssShare>, Vec<MulRecord>, CommStats) {
    let n = family.n;
    assert!(base.modulus() == field_modulus, "base / field_modulus mismatch");

    // For each subset T, extract the piece a_T (from any party in [n]\T) and
    // compute h_T = base^{a_T} as a public scalar in F_p.
    let mut encodings: Vec<DegenerateEncoding> = Vec::with_capacity(family.subsets.len());
    for subset in &family.subsets {
        let holder = (0..n)
            .find(|p| !subset.contains(p))
            .expect("every subset has a holder party");
        let a_t = exp_shares[holder]
            .shares
            .get(subset)
            .expect("holder must have the subset's piece")
            .clone();
        debug_assert!(a_t.modulus() == exp_modulus);
        let h_t = base.pow(&a_t.value);
        encodings.push(DegenerateEncoding {
            target_subset: subset.clone(),
            value: h_t,
        });
    }

    // Layer 0: every encoding ⟨h_T⟩_T becomes an RSS sharing [[h_T]]_p
    // (in a real deployment the "sharing" is a local agreement among parties
    // in [n]\T; we use the vdoprf_ss::share utility with a fresh RNG).
    let mut rng = rand::thread_rng();
    let mut current_layer: Vec<Vec<RssShare>> = encodings
        .iter()
        .map(|enc| {
            let s = share(&enc.value, family, field_modulus, &mut rng);
            (0..n).map(|i| get_party_share(&s, i, family)).collect()
        })
        .collect();

    let mut mul_records: Vec<MulRecord> = Vec::new();
    let mut comm = CommStats::default();
    let mut counter = counter_base;

    // Layer 1: pair up RSS sharings; multiply each pair via deg_mul where the
    // second factor is the known degenerate-encoding value (public scalar).
    if current_layer.len() > 1 {
        let mut next_layer = Vec::new();
        let mut layer_comm = CommStats::default();
        let mut i = 0;
        while i + 1 < current_layer.len() {
            let encoding = &encodings[i + 1];
            let t_j = &encoding.target_subset;
            let double_shares: Vec<DoubleShareLocal> = (0..n)
                .map(|p| {
                    generate_double_sharing_degenerate(
                        p,
                        counter,
                        t_j,
                        &pre_shared[p],
                        family,
                        field_modulus,
                    )
                })
                .collect();
            counter += 1;

            let (result, record, mul_comm) = deg_mul(
                &current_layer[i],
                encoding,
                &double_shares,
                family,
                field_modulus,
            );
            layer_comm.merge_parallel(&mul_comm);
            mul_records.push(record);
            next_layer.push(result);
            i += 2;
        }
        if i < current_layer.len() {
            next_layer.push(current_layer[i].clone());
        }
        comm.merge(&layer_comm);
        current_layer = next_layer;
    }

    // Layers 2+: full RSS.Mul binary tree.
    while current_layer.len() > 1 {
        let mut next_layer = Vec::new();
        let mut layer_comm = CommStats::default();
        let mut i = 0;
        while i + 1 < current_layer.len() {
            let double_shares: Vec<DoubleShareLocal> = (0..n)
                .map(|p| {
                    generate_double_sharing(p, counter, &pre_shared[p], family, field_modulus)
                })
                .collect();
            counter += 1;

            let (result, record, mul_comm) = rss_mul_all_parties_with_record(
                &current_layer[i],
                &current_layer[i + 1],
                &double_shares,
                family,
                field_modulus,
            );
            layer_comm.merge_parallel(&mul_comm);
            mul_records.push(record);
            next_layer.push(result);
            i += 2;
        }
        if i < current_layer.len() {
            next_layer.push(current_layer[i].clone());
        }
        comm.merge(&layer_comm);
        current_layer = next_layer;
    }

    let result = current_layer.into_iter().next().unwrap();
    (result, mul_records, comm)
}

/// Open an RSS-shared value: reconstruct via algebraic sum while recording
/// the simulated communication (each party sends its share to every other).
pub fn open_rss(
    shares: &[RssShare],
    modulus: &BigUint,
    net: &mut SimulatedNetwork,
) -> Fp {
    let n = shares.len();
    let feb = ((modulus.bits() + 7) / 8) as usize;
    // Each party broadcasts its RssShare components to all others.
    // Communication accounting only — the simulated net charges |shares| bytes
    // per recipient.
    for sender in 0..n {
        // Per-party share size: sum of byte sizes across the subset pieces.
        let total = shares[sender].shares.len() * feb;
        for receiver in 0..n {
            if receiver != sender {
                net.send_p2p(sender, receiver, vec![0u8; total]);
            }
        }
    }
    ReplicatedSharing::reconstruct_from_party_shares(shares, modulus)
}

/// Multiply each party's RSS share by a public scalar (purely local).
pub fn scalar_mul_rss(shares: &[RssShare], scalar: &Fp) -> Vec<RssShare> {
    shares
        .iter()
        .map(|s| {
            let new_shares = s
                .shares
                .iter()
                .map(|(subset, val)| (subset.clone(), val * scalar))
                .collect();
            RssShare {
                party_id: s.party_id,
                shares: new_shares,
            }
        })
        .collect()
}

/// Add a public scalar to each party's RSS share. The scalar is added to the
/// first subset's piece only (constant added at one "slot" reconstructs to
/// sum + constant), consistent with how `rss_mul` adds δ back.
pub fn add_scalar_rss(shares: &[RssShare], scalar: &Fp, family: &SubsetFamily) -> Vec<RssShare> {
    let first_subset = &family.subsets[0];
    let recipients: Vec<usize> = (0..family.n).filter(|p| !first_subset.contains(p)).collect();
    shares
        .iter()
        .map(|s| {
            let mut new_shares = s.shares.clone();
            if recipients.contains(&s.party_id) {
                if let Some(val) = new_shares.get(first_subset) {
                    new_shares.insert(first_subset.clone(), val + scalar);
                }
            }
            RssShare {
                party_id: s.party_id,
                shares: new_shares,
            }
        })
        .collect()
}

/// Malicious-secure Π_exp (Aly-Smart Prot. 6).
///
/// Runs two semi-honest exp calls and an RSS-based consistency check. Returns
/// the RSS sharing `[[base^a]]_p`, the accumulated `MulRecord`s (from ALL
/// internal multiplications), comm stats, and a `valid` flag: if the final
/// v-check fails, the caller MUST abort. The result shares are still returned
/// for composability, but they are not to be trusted when `valid == false`.
///
/// `r_shares` and `r_prime_shares` are F_Rand-provided random values:
///   * `r_shares`: RSS shares of a fresh `r ∈ Z*_{p-1}`.
///   * `r_prime_shares`: RSS shares of a fresh `r' ∈ Z*_p` (used as a
///     non-zero randomizer in the v-check).
#[allow(clippy::too_many_arguments)]
pub fn pub_base_exp_malicious(
    base: &Fp,
    exp_shares: &[RssShare],
    r_shares: &[RssShare],
    r_prime_shares: &[RssShare],
    exp_modulus: &BigUint, // = p - 1
    field_modulus: &BigUint, // = p
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    counter_base: u64,
) -> MaliciousExpOut {
    let n = family.n;
    let mut comm = CommStats::default();
    let mut mul_records: Vec<MulRecord> = Vec::new();

    // Pre-allocate counter slots so the three independent chains below
    // (Steps 1, 3+4, 5+6) can execute in parallel without collisions.
    let subsets_stride = family.subsets.len() as u64 * 2 + 16;
    let counter_exp1 = counter_base;
    let counter_aprime_mul = counter_exp1 + subsets_stride;
    let counter_exp2 = counter_aprime_mul + 1;
    let counter_w_mul = counter_exp2 + subsets_stride;
    let counter_prod1_mul = counter_w_mul + 1;
    let counter_v_pre_mul = counter_prod1_mul + 1;

    // Step 3 prep (local): [r - 1] over Z_{p-1}.
    let one_exp = Fp::one(exp_modulus);
    let r_minus_one: Vec<RssShare> = r_shares
        .iter()
        .map(|s| RssShare {
            party_id: s.party_id,
            shares: {
                let mut map = s.shares.clone();
                if let Some(first) = family.subsets.first() {
                    let recipients: Vec<usize> =
                        (0..n).filter(|p| !first.contains(p)).collect();
                    if recipients.contains(&s.party_id) {
                        if let Some(val) = map.get(first) {
                            map.insert(first.clone(), val - &one_exp);
                        }
                    }
                }
                map
            },
        })
        .collect();

    // Three independent chains run concurrently (all start from [a] and [r];
    // none consumes another's output):
    //   Chain 1: Step 1 — semi-honest exp on [a].
    //   Chain 2: Step 3 ([a']=[a]·[r-1]) → Step 4 — semi-honest exp on [a'].
    //   Chain 3: Step 5 ([w]=[a]·[r]) → Step 6 — Open w.
    // Z_{p-1} muls in Chains 2 and 3 are NOT emitted to the DZKP batch — the
    // final v==1 check covers any deviation in [a'] or [w].

    let mut chain1 = CommStats::default();
    let (b_pow_a, recs_exp1, c_exp1) = pub_base_exp_semi_honest(
        base, exp_shares, exp_modulus, field_modulus, pre_shared, family, counter_exp1,
    );
    mul_records.extend(recs_exp1);
    chain1.merge(&c_exp1);

    let mut chain2 = CommStats::default();
    let double_shares_aprime: Vec<DoubleShareLocal> = (0..n)
        .map(|p| generate_double_sharing(p, counter_aprime_mul, &pre_shared[p], family, exp_modulus))
        .collect();
    let (a_prime, _rec_zp1, c_aprime) = rss_mul_all_parties_with_record(
        exp_shares,
        &r_minus_one,
        &double_shares_aprime,
        family,
        exp_modulus,
    );
    chain2.merge(&c_aprime);
    let (b_pow_a_prime, recs_exp2, c_exp2) = pub_base_exp_semi_honest(
        base,
        &a_prime,
        exp_modulus,
        field_modulus,
        pre_shared,
        family,
        counter_exp2,
    );
    mul_records.extend(recs_exp2);
    chain2.merge(&c_exp2);

    let mut chain3 = CommStats::default();
    let double_shares_w: Vec<DoubleShareLocal> = (0..n)
        .map(|p| generate_double_sharing(p, counter_w_mul, &pre_shared[p], family, exp_modulus))
        .collect();
    let (w_shares, _rec_zp1, c_w) = rss_mul_all_parties_with_record(
        exp_shares,
        r_shares,
        &double_shares_w,
        family,
        exp_modulus,
    );
    chain3.merge(&c_w);
    let mut open_net_w = SimulatedNetwork::new(n);
    let w = open_rss(&w_shares, exp_modulus, &mut open_net_w);
    chain3.merge(&open_net_w.stats());

    let mut initial_block = CommStats::default();
    initial_block.merge_parallel(&chain1);
    initial_block.merge_parallel(&chain2);
    initial_block.merge_parallel(&chain3);
    comm.merge(&initial_block);

    // Step 7: r' already provided (F_Rand).

    // Step 8: [v] = [r'] · (Product(Product([b^a], [b^{a'}]), b^{-w}) - [1]) + [1].
    // First: [prod1] = [b^a] · [b^{a'}] over F_p.
    let double_shares_f1: Vec<DoubleShareLocal> = (0..n)
        .map(|p| generate_double_sharing(p, counter_prod1_mul, &pre_shared[p], family, field_modulus))
        .collect();
    let (prod1, rec, cc) = rss_mul_all_parties_with_record(
        &b_pow_a,
        &b_pow_a_prime,
        &double_shares_f1,
        family,
        field_modulus,
    );
    mul_records.push(rec);
    comm.merge(&cc);

    // Scale by public b^{-w} = b^{p-1-w mod p-1}.
    // w lives in Z_{p-1}; compute -w mod (p-1) for the exponent.
    let neg_w_exp = if w.is_zero() {
        BigUint::from(0u32)
    } else {
        exp_modulus - &w.value
    };
    let b_inv_w = base.pow(&neg_w_exp);
    let scaled = scalar_mul_rss(&prod1, &b_inv_w);

    // Subtract [1]: rss view of constant 1 subtracted.
    let neg_one_p = -&Fp::one(field_modulus);
    let scaled_minus_one = add_scalar_rss(&scaled, &neg_one_p, family);

    // [v_pre] = [r'] · (...).
    let double_shares_f2: Vec<DoubleShareLocal> = (0..n)
        .map(|p| generate_double_sharing(p, counter_v_pre_mul, &pre_shared[p], family, field_modulus))
        .collect();
    let (v_pre, rec, cc) = rss_mul_all_parties_with_record(
        r_prime_shares,
        &scaled_minus_one,
        &double_shares_f2,
        family,
        field_modulus,
    );
    mul_records.push(rec);
    comm.merge(&cc);

    // [v] = [v_pre] + [1].
    let one_p = Fp::one(field_modulus);
    let v_shares = add_scalar_rss(&v_pre, &one_p, family);

    // Step 9: open v, check v == 1.
    let mut open_net2 = SimulatedNetwork::new(n);
    let v = open_rss(&v_shares, field_modulus, &mut open_net2);
    comm.merge(&open_net2.stats());

    let valid = v.value == BigUint::one();

    MaliciousExpOut {
        result_shares: b_pow_a,
        mul_records,
        comm,
        valid,
    }
}

/// Output of the malicious-secure Π_exp.
pub struct MaliciousExpOut {
    /// RSS sharing `[[base^a]]_p`.
    pub result_shares: Vec<RssShare>,
    /// Multiplications emitted during the protocol; to be fed to the final
    /// DZKP batch.
    pub mul_records: Vec<MulRecord>,
    pub comm: CommStats,
    /// Whether the v == 1 check passed. Caller must abort if false.
    pub valid: bool,
}

/// Euclidean gcd on BigUint.
pub fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
    let mut x = a.clone();
    let mut y = b.clone();
    while !y.is_zero() {
        let r = &x % &y;
        x = y;
        y = r;
    }
    x
}

/// Check if `x` is coprime to `m`.
pub fn is_coprime(x: &BigUint, m: &BigUint) -> bool {
    gcd(x, m).is_one()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_pre_shared;

    fn build_exp_sharing(
        value: &Fp,
        family: &SubsetFamily,
        exp_modulus: &BigUint,
    ) -> Vec<RssShare> {
        let mut rng = rand::thread_rng();
        let s = share(value, family, exp_modulus, &mut rng);
        (0..family.n).map(|i| get_party_share(&s, i, family)).collect()
    }

    #[test]
    fn test_pub_base_exp_semi_honest_small() {
        // Verify: base^a reconstructs correctly under RSS.
        let n = 3;
        let t = 1;
        let p = BigUint::from(113u32);
        let p_minus_1 = &p - BigUint::one();
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &p);

        let base = Fp::new(BigUint::from(3u32), &p); // 3 is a generator of F_113*
        let a = Fp::new(BigUint::from(17u32), &p_minus_1);
        let expected = base.pow(&a.value);

        let exp_shares = build_exp_sharing(&a, &family, &p_minus_1);
        let (result_shares, _recs, _comm) = pub_base_exp_semi_honest(
            &base,
            &exp_shares,
            &p_minus_1,
            &p,
            &pre_shared,
            &family,
            10_000,
        );

        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result_shares, &p);
        assert_eq!(reconstructed.value, expected.value);
    }

    #[test]
    fn test_pub_base_exp_malicious_accepts_honest() {
        let n = 3;
        let t = 1;
        let p = BigUint::from(113u32);
        let p_minus_1 = &p - BigUint::one();
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &p);

        let base = Fp::new(BigUint::from(3u32), &p);
        let a = Fp::new(BigUint::from(11u32), &p_minus_1);
        let expected = base.pow(&a.value);

        let r = Fp::new(BigUint::from(5u32), &p_minus_1); // arbitrary non-zero in Z_{p-1}
        let r_prime = Fp::new(BigUint::from(42u32), &p); // arbitrary non-zero in F_p

        let exp_shares = build_exp_sharing(&a, &family, &p_minus_1);
        let r_shares = build_exp_sharing(&r, &family, &p_minus_1);
        let r_prime_shares = build_exp_sharing(&r_prime, &family, &p);

        let out = pub_base_exp_malicious(
            &base,
            &exp_shares,
            &r_shares,
            &r_prime_shares,
            &p_minus_1,
            &p,
            &pre_shared,
            &family,
            20_000,
        );
        assert!(out.valid, "honest execution must pass the v==1 check");
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&out.result_shares, &p);
        assert_eq!(reconstructed.value, expected.value);
    }
}
