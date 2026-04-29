//! Protocol Π_RSS.Mul: Multiplication of two RSS-shared values.
//!
//! Steps:
//! 1. Each party computes its additive share of a*b via cross_multiply.
//! 2. Use Π_A2T (additive-to-threshold) to convert the additive sharing to RSS.
//!    Π_A2T uses a pre-computed double sharing (⟨r⟩, [r]):
//!    a) Each party broadcasts δ_i = (additive share of ab) - ⟨r⟩_i
//!    b) All parties reconstruct δ = Σ δ_i = ab - r
//!    c) Each party sets [ab]_i = [r]_i + δ (local addition of public constant)

use num_bigint::BigUint;
use vdoprf_field::Fp;
use vdoprf_network::{CommStats, SimulatedNetwork};
use vdoprf_ss::{cross_multiply, cross_multiply_with_pairs, DegenerateEncoding, RssShare, SubsetFamily};
use crate::double_rand::DoubleShareLocal;

/// Perform RSS multiplication: given [a]_i and [b]_i, compute [a*b]_i.
///
/// Requires a pre-computed double sharing for the A2T conversion.
/// Uses one round of broadcast communication.
pub fn rss_mul(
    a_share: &RssShare,
    b_share: &RssShare,
    double_share: &DoubleShareLocal,
    family: &SubsetFamily,
    modulus: &BigUint,
    net: &mut SimulatedNetwork,
    round: usize,
) -> RssShare {
    let party_id = a_share.party_id;

    // Step 1: Compute additive share of a*b
    let ab_additive = cross_multiply(a_share, b_share, family);

    // Step 2: A2T conversion using double sharing
    a2t(
        &ab_additive,
        double_share,
        party_id,
        family,
        modulus,
        net,
        round,
    )
}

/// Additive-to-Threshold conversion (Protocol Π_A2T / Π_A2R).
///
/// 2-round aggregator pattern. The aggregator is expected to be a member of
/// the update set 𝒬 = [n] \ T_0; this matches the spec (Π_A2R step 3: S_q ∈ 𝒬)
/// and lets the aggregator skip sending δ to itself.
///   Round 1: Party i ≠ q sends δ_i = c_i - ⟨r⟩_i to the aggregator (P2P).
///   Round 2: Aggregator sums δ = δ_q + Σ_{i ≠ q} δ_i and sends δ to 𝒬 \ {q}.
///   Local:   Parties in 𝒬 set [c]_i.T_0 ← [r]_i.T_0 + δ. The aggregator
///            applies this locally using its own δ.
pub fn a2t(
    additive_share: &Fp,
    double_share: &DoubleShareLocal,
    party_id: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
    net: &mut SimulatedNetwork,
    aggregator: usize,
) -> RssShare {
    let first_subset = &family.subsets[0];
    let recipients: Vec<usize> = (0..family.n).filter(|p| !first_subset.contains(p)).collect();

    // Round 1: Send δ_i to aggregator (skip self-send)
    let delta_i = additive_share - &double_share.additive_share;
    if party_id != aggregator {
        net.send_p2p(party_id, aggregator, delta_i.value.to_bytes_be());
    }

    // Aggregator sums (starting from its own δ) and sends back to 𝒬 \ {q}.
    let round1 = net.current_round();
    let mut local_delta: Option<Fp> = None;
    if party_id == aggregator {
        let mut delta = delta_i.clone();
        for sender in 0..family.n {
            if sender == aggregator { continue; }
            if let Some(data) = net.get_p2p(round1, sender, aggregator) {
                delta = &delta + &Fp::new(BigUint::from_bytes_be(data), modulus);
            }
        }
        net.next_round();
        let delta_bytes = delta.value.to_bytes_be();
        for &r in &recipients {
            if r == aggregator { continue; }
            net.send_p2p(aggregator, r, delta_bytes.clone());
        }
        local_delta = Some(delta);
    }

    // Recipients read δ (or use locally-computed δ if aggregator ∈ 𝒬).
    let round2 = net.current_round();
    let mut result_shares = double_share.rss_share.shares.clone();
    if recipients.contains(&party_id) {
        let d = if party_id == aggregator {
            local_delta.expect("aggregator always has local delta")
        } else if let Some(data) = net.get_p2p(round2, aggregator, party_id) {
            Fp::new(BigUint::from_bytes_be(data), modulus)
        } else {
            Fp::zero(modulus)
        };
        if let Some(val) = result_shares.get(first_subset) {
            result_shares.insert(first_subset.clone(), val + &d);
        }
    }

    RssShare {
        party_id,
        shares: result_shares,
    }
}

/// Perform RSS multiplication for all parties (simulated).
/// Uses aggregator pattern for A2T: 2 rounds of communication.
///   Round 1: Each party ≠ q sends δ_i to the aggregator q (P2P).
///   Round 2: Aggregator sends δ = Σ δ_i back to 𝒬 \ {q} (P2P, not broadcast).
/// The aggregator is chosen inside 𝒬 = [n] \ T_0 to match Π_A2R and save one
/// P2P message (the aggregator applies δ to its own share locally).
pub fn rss_mul_all_parties(
    a_shares: &[RssShare],
    b_shares: &[RssShare],
    double_shares: &[DoubleShareLocal],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Vec<RssShare>, CommStats) {
    let n = family.n;
    let first_subset = &family.subsets[0];
    // Parties that hold T_0: those not in T_0 — this is 𝒬 in Π_A2R.
    let recipients: Vec<usize> = (0..n).filter(|p| !first_subset.contains(p)).collect();
    // Aggregator q ∈ 𝒬 so it can apply δ locally.
    let aggregator = recipients[0];
    let mut net = SimulatedNetwork::new(n);

    // Round 1: Each non-aggregator sends δ_i to aggregator; aggregator keeps its own
    let mut delta_local = Fp::zero(modulus);
    for i in 0..n {
        let ab_add = cross_multiply(&a_shares[i], &b_shares[i], family);
        let delta_i = &ab_add - &double_shares[i].additive_share;
        if i == aggregator {
            delta_local = delta_i;
        } else {
            net.send_p2p(i, aggregator, delta_i.value.to_bytes_be());
        }
    }

    // Aggregator computes δ = own δ + Σ received δ_i
    let round1 = net.current_round();
    let mut delta = delta_local;
    for sender in 0..n {
        if sender == aggregator { continue; }
        if let Some(data) = net.get_p2p(round1, sender, aggregator) {
            delta = &delta + &Fp::new(BigUint::from_bytes_be(data), modulus);
        }
    }

    // Round 2: Aggregator sends δ to recipients (P2P, skip self)
    net.next_round();
    let delta_bytes = delta.value.to_bytes_be();
    for &r in &recipients {
        if r != aggregator {
            net.send_p2p(aggregator, r, delta_bytes.clone());
        }
    }

    // Recipients update shares; aggregator uses its local δ
    let round2 = net.current_round();
    let mut results = Vec::new();
    for i in 0..n {
        let mut result_shares = double_shares[i].rss_share.shares.clone();
        if recipients.contains(&i) {
            let d = if i == aggregator {
                delta.clone()
            } else {
                net.get_p2p(round2, aggregator, i)
                    .map(|data| Fp::new(BigUint::from_bytes_be(data), modulus))
                    .unwrap_or_else(|| Fp::zero(modulus))
            };
            if let Some(val) = result_shares.get(first_subset) {
                result_shares.insert(first_subset.clone(), val + &d);
            }
        }
        results.push(RssShare {
            party_id: i,
            shares: result_shares,
        });
    }

    (results, net.stats())
}

/// Π_DegMul: Multiply RSS sharing `[q]` by a degenerate encoding `⟨v⟩_{T'}`.
///
/// The encoding's value is public to the active set `A = [n] \ T'`.
/// Communication: 2 rounds among `|A| = n-t` parties (much less than full RSS.Mul).
///   Round 1: `|A|-1` active non-aggregator parties send δ_i to aggregator (P2P)
///   Round 2: aggregator sends δ back to the subset holding T_0 within A (P2P)
///
/// Returns (result_shares, record, comm_stats).
pub fn deg_mul(
    q_shares: &[RssShare],              // [q]_i for all n parties
    encoding: &DegenerateEncoding,      // ⟨v⟩_{T'} (value + target subset T')
    double_shares: &[DoubleShareLocal], // target-matched double sharing
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Vec<RssShare>, MulRecord, CommStats) {
    let n = family.n;
    let active_set = encoding.active_set(n);
    let scalar = &encoding.value;
    let first_subset = &family.subsets[0];
    let recipients: Vec<usize> = (0..n).filter(|p| !first_subset.contains(p)).collect();
    // Aggregator must be in 𝒜 (Π_DegMul). Prefer a choice also in 𝒬 = [n]\T_0
    // so the aggregator applies δ to its own T_0 component locally and saves
    // one P2P message. |𝒜 ∩ [n]\T_0| = n - |T' ∪ T_0| ≥ n - 2t ≥ 1 under
    // honest majority, so the preferred choice always exists.
    let aggregator = active_set
        .iter()
        .find(|&&p| !first_subset.contains(&p))
        .copied()
        .unwrap_or(active_set[0]);
    let mut net = SimulatedNetwork::new(n);

    // Each active party computes its additive share of q * m':
    // Since m' is a public scalar, party i's share = m' × Σ_{T assigned to i} q_T
    // The assignment is: each component q_T is assigned to the party via the
    // standard balanced assignment, but the "b" value is always the same scalar m'.
    let mut party_pairs = vec![Vec::new(); n];
    let mut party_cp = vec![Fp::zero(modulus); n];

    // Assign each component q_T to an active party that holds it.
    // For each T, find the first active party not in T.
    for t_sub in &family.subsets {
        // Find assigned active party: smallest active party not in T
        let assigned = active_set.iter().find(|&&p| !t_sub.contains(&p));
        if let Some(&party) = assigned {
            if let Some(q_t) = q_shares[party].shares.get(t_sub) {
                let product = q_t * scalar;
                party_pairs[party].push((q_t.clone(), scalar.clone()));
                party_cp[party] = &party_cp[party] + &product;
            }
        }
    }

    // A2T with only active parties
    // Round 1: active non-aggregator parties send δ_i to aggregator
    let mut delta_local = Fp::zero(modulus);
    for &i in &active_set {
        let delta_i = &party_cp[i] - &double_shares[i].additive_share;
        if i == aggregator {
            delta_local = delta_i;
        } else {
            net.send_p2p(i, aggregator, delta_i.value.to_bytes_be());
        }
    }

    // Aggregator sums
    let round1 = net.current_round();
    let mut delta = delta_local;
    for &sender in &active_set {
        if sender == aggregator { continue; }
        if let Some(data) = net.get_p2p(round1, sender, aggregator) {
            delta = &delta + &Fp::new(BigUint::from_bytes_be(data), modulus);
        }
    }

    // Round 2: aggregator sends δ to recipients (skip self)
    net.next_round();
    let delta_bytes = delta.value.to_bytes_be();
    for &r in &recipients {
        if r != aggregator {
            net.send_p2p(aggregator, r, delta_bytes.clone());
        }
    }

    // All parties update shares
    let round2 = net.current_round();
    let mut results = Vec::new();
    for i in 0..n {
        let mut result_shares = double_shares[i].rss_share.shares.clone();
        if recipients.contains(&i) {
            let d = if i == aggregator {
                delta.clone()
            } else {
                net.get_p2p(round2, aggregator, i)
                    .map(|data| Fp::new(BigUint::from_bytes_be(data), modulus))
                    .unwrap_or_else(|| Fp::zero(modulus))
            };
            if let Some(val) = result_shares.get(first_subset) {
                result_shares.insert(first_subset.clone(), val + &d);
            }
        }
        results.push(RssShare {
            party_id: i,
            shares: result_shares,
        });
    }

    let record = MulRecord {
        party_pairs,
        party_cp,
        a_shares: q_shares.to_vec(),
        b_shares: q_shares.to_vec(), // placeholder — b is scalar, not RSS
        c_shares: results.clone(),
    };

    (results, record, net.stats())
}

/// Record of a single multiplication, capturing per-party cross-product inputs
/// for DZKP verification.
#[derive(Clone, Debug)]
pub struct MulRecord {
    /// For each party i: the (a_val, b_val) pairs assigned to that party's cross-product.
    pub party_pairs: Vec<Vec<(Fp, Fp)>>,
    /// Per-party cross-product result cp_i.
    pub party_cp: Vec<Fp>,
    /// RSS shares of a (per-party).
    pub a_shares: Vec<RssShare>,
    /// RSS shares of b (per-party).
    pub b_shares: Vec<RssShare>,
    /// RSS shares of c = a*b (per-party).
    pub c_shares: Vec<RssShare>,
}

/// Like `rss_mul_all_parties` but also returns a `MulRecord` for DZKP verification.
/// Uses the same aggregator-based 2-round A2T pattern with P2P back to subset,
/// with the aggregator chosen inside 𝒬 = [n] \ T_0 (see `rss_mul_all_parties`).
pub fn rss_mul_all_parties_with_record(
    a_shares: &[RssShare],
    b_shares: &[RssShare],
    double_shares: &[DoubleShareLocal],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Vec<RssShare>, MulRecord, CommStats) {
    let n = family.n;
    let first_subset = &family.subsets[0];
    let recipients: Vec<usize> = (0..n).filter(|p| !first_subset.contains(p)).collect();
    let aggregator = recipients[0];
    let mut net = SimulatedNetwork::new(n);

    // Round 1: Each party computes cross-products; non-aggregators send δ_i to aggregator
    let mut party_pairs = Vec::new();
    let mut party_cp = Vec::new();
    let mut delta_local = Fp::zero(modulus);
    for i in 0..n {
        let (ab_add, pairs) = cross_multiply_with_pairs(&a_shares[i], &b_shares[i], family);
        let delta_i = &ab_add - &double_shares[i].additive_share;
        if i == aggregator {
            delta_local = delta_i;
        } else {
            net.send_p2p(i, aggregator, delta_i.value.to_bytes_be());
        }
        party_pairs.push(pairs);
        party_cp.push(ab_add);
    }

    // Aggregator computes δ = own δ + Σ received δ_i
    let round1 = net.current_round();
    let mut delta = delta_local;
    for sender in 0..n {
        if sender == aggregator { continue; }
        if let Some(data) = net.get_p2p(round1, sender, aggregator) {
            delta = &delta + &Fp::new(BigUint::from_bytes_be(data), modulus);
        }
    }

    // Round 2: Aggregator sends δ to recipients (P2P, skip self)
    net.next_round();
    let delta_bytes = delta.value.to_bytes_be();
    for &r in &recipients {
        if r != aggregator {
            net.send_p2p(aggregator, r, delta_bytes.clone());
        }
    }

    // Recipients update shares; aggregator uses its local δ
    let round2 = net.current_round();
    let mut results = Vec::new();
    for i in 0..n {
        let mut result_shares = double_shares[i].rss_share.shares.clone();
        if recipients.contains(&i) {
            let d = if i == aggregator {
                delta.clone()
            } else {
                net.get_p2p(round2, aggregator, i)
                    .map(|data| Fp::new(BigUint::from_bytes_be(data), modulus))
                    .unwrap_or_else(|| Fp::zero(modulus))
            };
            if let Some(val) = result_shares.get(first_subset) {
                result_shares.insert(first_subset.clone(), val + &d);
            }
        }
        results.push(RssShare {
            party_id: i,
            shares: result_shares,
        });
    }

    let record = MulRecord {
        party_pairs,
        party_cp,
        a_shares: a_shares.to_vec(),
        b_shares: b_shares.to_vec(),
        c_shares: results.clone(),
    };

    (results, record, net.stats())
}

/// Batched version of `rss_mul_all_parties_with_record`: runs `m` independent
/// RSS multiplications against a single `SimulatedNetwork` in exactly **2 rounds**
/// regardless of `m`. Every pair (sender, receiver) exchanges one concatenated
/// message per round, carrying all `m` δ values packed at fixed field width.
///
/// This mirrors how a synchronous network would carry `m` parallel A2T
/// conversions — the multiplications are data-independent, so they can be
/// multiplexed over the same 2-round aggregator pattern.
pub fn rss_mul_batched_all_parties_with_record(
    a_shares_per_input: &[Vec<RssShare>],
    b_shares_per_input: &[Vec<RssShare>],
    double_shares_per_input: &[Vec<DoubleShareLocal>],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> (Vec<Vec<RssShare>>, Vec<MulRecord>, CommStats) {
    let m = a_shares_per_input.len();
    assert_eq!(b_shares_per_input.len(), m);
    assert_eq!(double_shares_per_input.len(), m);

    let n = family.n;
    let first_subset = &family.subsets[0];
    let recipients: Vec<usize> = (0..n).filter(|p| !first_subset.contains(p)).collect();
    let aggregator = recipients[0];
    let mut net = SimulatedNetwork::new(n);

    if m == 0 {
        // No work to do; still report the two-round topology with zero bytes.
        net.next_round();
        return (Vec::new(), Vec::new(), net.stats());
    }

    // Fixed field-element byte width so m values can be packed/unpacked.
    let elem_bytes = ((modulus.bits() as usize + 7) / 8).max(1);
    let pad_be = |v: &Fp| -> Vec<u8> {
        let bytes = v.value.to_bytes_be();
        debug_assert!(bytes.len() <= elem_bytes);
        let mut padded = vec![0u8; elem_bytes];
        padded[elem_bytes - bytes.len()..].copy_from_slice(&bytes);
        padded
    };
    let unpad = |buf: &[u8], k: usize| -> Fp {
        let slice = &buf[k * elem_bytes..(k + 1) * elem_bytes];
        Fp::new(BigUint::from_bytes_be(slice), modulus)
    };

    // Local pre-work: compute per-(input, party) cross-product additive share
    // and δ_i, plus the pairs kept for the per-input MulRecord.
    let mut party_pairs_per_input: Vec<Vec<Vec<(Fp, Fp)>>> = Vec::with_capacity(m);
    let mut party_cp_per_input: Vec<Vec<Fp>> = Vec::with_capacity(m);
    let mut delta_vals: Vec<Vec<Fp>> = Vec::with_capacity(m);
    for j in 0..m {
        let a_shares = &a_shares_per_input[j];
        let b_shares = &b_shares_per_input[j];
        let double_shares = &double_shares_per_input[j];
        let mut pairs_j = Vec::with_capacity(n);
        let mut cp_j = Vec::with_capacity(n);
        let mut delta_j = Vec::with_capacity(n);
        for i in 0..n {
            let (ab_add, pairs) = cross_multiply_with_pairs(&a_shares[i], &b_shares[i], family);
            let delta_i = &ab_add - &double_shares[i].additive_share;
            pairs_j.push(pairs);
            cp_j.push(ab_add);
            delta_j.push(delta_i);
        }
        party_pairs_per_input.push(pairs_j);
        party_cp_per_input.push(cp_j);
        delta_vals.push(delta_j);
    }

    // Round 1: each non-aggregator party packs its m δ values into a single
    // P2P message to the aggregator (m × elem_bytes per pair, one send per pair).
    for i in 0..n {
        if i == aggregator { continue; }
        let mut buf = Vec::with_capacity(m * elem_bytes);
        for j in 0..m {
            buf.extend_from_slice(&pad_be(&delta_vals[j][i]));
        }
        net.send_p2p(i, aggregator, buf);
    }

    // Aggregator reconstructs per-input δ = own δ_j + Σ received δ_j.
    let round1 = net.current_round();
    let mut deltas: Vec<Fp> = (0..m).map(|j| delta_vals[j][aggregator].clone()).collect();
    for sender in 0..n {
        if sender == aggregator { continue; }
        if let Some(buf) = net.get_p2p(round1, sender, aggregator) {
            for j in 0..m {
                deltas[j] = &deltas[j] + &unpad(buf, j);
            }
        }
    }

    // Round 2: aggregator packs all m δ into one buffer, sends one P2P per recipient.
    net.next_round();
    let mut delta_buf = Vec::with_capacity(m * elem_bytes);
    for j in 0..m {
        delta_buf.extend_from_slice(&pad_be(&deltas[j]));
    }
    for &r in &recipients {
        if r == aggregator { continue; }
        net.send_p2p(aggregator, r, delta_buf.clone());
    }

    // Each recipient updates its T_0 component with δ_j for every input j.
    let round2 = net.current_round();
    let mut c_shares_per_input: Vec<Vec<RssShare>> = Vec::with_capacity(m);
    for j in 0..m {
        let double_shares = &double_shares_per_input[j];
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let mut result_shares = double_shares[i].rss_share.shares.clone();
            if recipients.contains(&i) {
                let d = if i == aggregator {
                    deltas[j].clone()
                } else {
                    net.get_p2p(round2, aggregator, i)
                        .map(|buf| unpad(buf, j))
                        .unwrap_or_else(|| Fp::zero(modulus))
                };
                if let Some(val) = result_shares.get(first_subset) {
                    result_shares.insert(first_subset.clone(), val + &d);
                }
            }
            results.push(RssShare { party_id: i, shares: result_shares });
        }
        c_shares_per_input.push(results);
    }

    // Build per-input MulRecord from the per-input intermediate state.
    let mut records: Vec<MulRecord> = Vec::with_capacity(m);
    for j in 0..m {
        records.push(MulRecord {
            party_pairs: std::mem::take(&mut party_pairs_per_input[j]),
            party_cp: std::mem::take(&mut party_cp_per_input[j]),
            a_shares: a_shares_per_input[j].clone(),
            b_shares: b_shares_per_input[j].clone(),
            c_shares: c_shares_per_input[j].clone(),
        });
    }

    (c_shares_per_input, records, net.stats())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::double_rand::{generate_double_sharing, generate_double_sharing_degenerate};
    use crate::setup_pre_shared;
    use vdoprf_ss::{share, get_party_share, ReplicatedSharing};

    // ---- helpers ----------------------------------------------------------

    /// Build RSS shares and matching double sharings for every party.
    fn make_party_material(
        a: &Fp,
        b: &Fp,
        family: &SubsetFamily,
        modulus: &BigUint,
        pre_shared: &[crate::PreSharedMaterial],
        counter: u64,
    ) -> (Vec<RssShare>, Vec<RssShare>, Vec<DoubleShareLocal>) {
        let mut rng = rand::thread_rng();
        let sa = share(a, family, modulus, &mut rng);
        let sb = share(b, family, modulus, &mut rng);
        let a_shares: Vec<RssShare> = (0..family.n).map(|i| get_party_share(&sa, i, family)).collect();
        let b_shares: Vec<RssShare> = (0..family.n).map(|i| get_party_share(&sb, i, family)).collect();
        let double_shares: Vec<DoubleShareLocal> = (0..family.n)
            .map(|i| generate_double_sharing(i, counter, &pre_shared[i], family, modulus))
            .collect();
        (a_shares, b_shares, double_shares)
    }

    /// Drive `a2t` across all parties. `a2t` does round-1 send unconditionally,
    /// round-2 send only when called by the aggregator, and round-2 read inline.
    /// In a sequential simulation, non-aggregator recipients' returned shares
    /// miss the round-2 δ update. We run the aggregator last and then apply
    /// the round-2 update from the network state for each recipient.
    fn drive_a2t_sequential(
        additive_shares: &[Fp],
        double_shares: &[DoubleShareLocal],
        family: &SubsetFamily,
        modulus: &BigUint,
        aggregator: usize,
    ) -> Vec<RssShare> {
        let n = family.n;
        let mut net = SimulatedNetwork::new(n);
        let mut results: Vec<Option<RssShare>> = vec![None; n];

        // Non-aggregators first: each sends δ_i in round 1.
        for i in 0..n {
            if i != aggregator {
                results[i] = Some(a2t(
                    &additive_shares[i], &double_shares[i], i,
                    family, modulus, &mut net, aggregator,
                ));
            }
        }
        // Aggregator last: reads round-1 δ_i's, advances round, sends δ out.
        results[aggregator] = Some(a2t(
            &additive_shares[aggregator], &double_shares[aggregator], aggregator,
            family, modulus, &mut net, aggregator,
        ));

        // Apply the round-2 δ update to the T_0 component of every recipient
        // that ran before the aggregator (they missed it).
        let first_subset = &family.subsets[0];
        let recipients: Vec<usize> = (0..n).filter(|p| !first_subset.contains(p)).collect();
        for &r in &recipients {
            if r == aggregator { continue; }
            if let Some(bytes) = net.get_p2p(1, aggregator, r) {
                let delta = Fp::new(BigUint::from_bytes_be(bytes), modulus);
                let share = results[r].as_mut().unwrap();
                if let Some(val) = share.shares.get(first_subset).cloned() {
                    share.shares.insert(first_subset.clone(), &val + &delta);
                }
            }
        }

        results.into_iter().map(Option::unwrap).collect()
    }

    // ---- rss_mul_all_parties: correctness and algebraic properties --------

    #[test]
    fn test_rss_mul_correctness() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let a = Fp::new(BigUint::from(7u32), &modulus);
        let b = Fp::new(BigUint::from(11u32), &modulus);
        let expected = &a * &b; // 77

        let (a_s, b_s, ds) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 0);
        let (result_shares, _comm) =
            rss_mul_all_parties(&a_s, &b_s, &ds, &family, &modulus);

        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result_shares, &modulus);
        assert_eq!(reconstructed.value, expected.value);
    }

    #[test]
    fn test_rss_mul_commutativity() {
        // a*b and b*a must reconstruct to the same value.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let a = Fp::new(BigUint::from(19u32), &modulus);
        let b = Fp::new(BigUint::from(23u32), &modulus);
        let (a_s, b_s, ds1) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 0);
        let (_, _, ds2) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 1);

        let (r_ab, _) = rss_mul_all_parties(&a_s, &b_s, &ds1, &family, &modulus);
        let (r_ba, _) = rss_mul_all_parties(&b_s, &a_s, &ds2, &family, &modulus);

        let ab = ReplicatedSharing::reconstruct_from_party_shares(&r_ab, &modulus);
        let ba = ReplicatedSharing::reconstruct_from_party_shares(&r_ba, &modulus);
        assert_eq!(ab.value, ba.value);
    }

    #[test]
    fn test_rss_mul_zero_times_x_is_zero() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let zero = Fp::zero(&modulus);
        let x = Fp::new(BigUint::from(42u32), &modulus);
        let (a_s, b_s, ds) = make_party_material(&zero, &x, &family, &modulus, &pre_shared, 0);

        let (r, _) = rss_mul_all_parties(&a_s, &b_s, &ds, &family, &modulus);
        let val = ReplicatedSharing::reconstruct_from_party_shares(&r, &modulus);
        assert_eq!(val.value, BigUint::from(0u32));
    }

    #[test]
    fn test_rss_mul_one_times_x_is_x() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let one = Fp::new(BigUint::from(1u32), &modulus);
        let x = Fp::new(BigUint::from(59u32), &modulus);
        let (a_s, b_s, ds) = make_party_material(&one, &x, &family, &modulus, &pre_shared, 0);

        let (r, _) = rss_mul_all_parties(&a_s, &b_s, &ds, &family, &modulus);
        let val = ReplicatedSharing::reconstruct_from_party_shares(&r, &modulus);
        assert_eq!(val.value, x.value);
    }

    #[test]
    fn test_rss_mul_larger_parameters_n5_t2() {
        // Exercise a non-minimal (n,t) so the subset family has N=10 subsets.
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(257u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let a = Fp::new(BigUint::from(73u32), &modulus);
        let b = Fp::new(BigUint::from(137u32), &modulus);
        let expected = &a * &b;
        let (a_s, b_s, ds) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 0);

        let (r, _) = rss_mul_all_parties(&a_s, &b_s, &ds, &family, &modulus);
        let val = ReplicatedSharing::reconstruct_from_party_shares(&r, &modulus);
        assert_eq!(val.value, expected.value);
    }

    #[test]
    fn test_rss_mul_comm_pattern_two_rounds_p2p_only() {
        // Π_RSS.Mul / Π_A2R per spec: 2 rounds, point-to-point only (no broadcast).
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let a = Fp::new(BigUint::from(2u32), &modulus);
        let b = Fp::new(BigUint::from(3u32), &modulus);
        let (a_s, b_s, ds) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 0);

        let (_, comm) = rss_mul_all_parties(&a_s, &b_s, &ds, &family, &modulus);
        assert_eq!(comm.rounds, 2, "Π_A2R uses 2 synchronous rounds");
        assert_eq!(comm.broadcast_bytes, 0, "Π_A2R is P2P-only");
        assert!(comm.p2p_bytes > 0, "Π_A2R must send some P2P bytes");
    }

    #[test]
    fn test_rss_mul_aggregator_inside_update_set() {
        // The aggregator must be placed inside 𝒬 = [n]\T_0 (per Π_A2R) so it
        // applies δ locally. Under canonical subset ordering T_0 = {0..t-1},
        // so the aggregator must NOT be party 0 (and the message count is one
        // below the naive t+1).
        //   round 1: n-1 parties send to aggregator → n-1 messages
        //   round 2: aggregator sends to |𝒬|-1 = t → t messages
        //   total p2p messages = n - 1 + t = 3t
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let a = Fp::new(BigUint::from(2u32), &modulus);
        let b = Fp::new(BigUint::from(3u32), &modulus);
        let (a_s, b_s, ds) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 0);

        let (_, comm) = rss_mul_all_parties(&a_s, &b_s, &ds, &family, &modulus);
        let expected_msgs = (n - 1) + t; // n-1 to agg + t from agg
        assert_eq!(comm.p2p_messages, expected_msgs,
            "expected {expected_msgs} P2P messages (agg ∈ 𝒬), got {}",
            comm.p2p_messages);
    }

    // ---- rss_mul / a2t orchestrated tests ---------------------------------

    #[test]
    fn test_rss_mul_single_party_api_matches_all_parties() {
        // Orchestrate the single-party `rss_mul` API across n parties and
        // assert the reconstructed product matches both the true product
        // and the `rss_mul_all_parties` helper. rss_mul's returned share is
        // "stale" for non-aggregator recipients — this test drives the
        // protocol manually to exercise the single-party code path.
        let n = 3;
        let t = 1;
        let aggregator = 0usize;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let a = Fp::new(BigUint::from(17u32), &modulus);
        let b = Fp::new(BigUint::from(29u32), &modulus);
        let expected = &a * &b;

        let (a_s, b_s, ds) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 7);

        let mut net = SimulatedNetwork::new(n);
        let mut shares: Vec<Option<RssShare>> = vec![None; n];
        for i in 0..n {
            if i == aggregator { continue; }
            shares[i] = Some(rss_mul(&a_s[i], &b_s[i], &ds[i], &family, &modulus, &mut net, aggregator));
        }
        shares[aggregator] = Some(rss_mul(
            &a_s[aggregator], &b_s[aggregator], &ds[aggregator],
            &family, &modulus, &mut net, aggregator,
        ));

        // Apply round-2 update manually to non-aggregator recipients.
        let first_subset = &family.subsets[0];
        let recipients: Vec<usize> = (0..n).filter(|p| !first_subset.contains(p)).collect();
        for &r in &recipients {
            if r == aggregator { continue; }
            if let Some(bytes) = net.get_p2p(1, aggregator, r) {
                let delta = Fp::new(BigUint::from_bytes_be(bytes), &modulus);
                let s = shares[r].as_mut().unwrap();
                if let Some(val) = s.shares.get(first_subset).cloned() {
                    s.shares.insert(first_subset.clone(), &val + &delta);
                }
            }
        }

        let collected: Vec<RssShare> = shares.into_iter().map(Option::unwrap).collect();
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&collected, &modulus);
        assert_eq!(reconstructed.value, expected.value);
    }

    #[test]
    fn test_a2t_reconstructs_additive_sum() {
        // Π_A2T: given additive shares ⟨c⟩ = {c_i} summing to c, and a
        // pre-computed double sharing (⟨r⟩, [r]), produce [c]. We sample
        // arbitrary additive shares of a chosen c and check the result
        // reconstructs to c.
        let n = 3;
        let t = 1;
        let aggregator = 0usize;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let c = Fp::new(BigUint::from(91u32), &modulus);
        // Pick random additive shares of c: fresh n-1 random, last = c - Σ.
        let mut rng = rand::thread_rng();
        let mut additive: Vec<Fp> = (0..n - 1).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let mut running = Fp::zero(&modulus);
        for x in &additive { running = &running + x; }
        additive.push(&c - &running);

        let ds: Vec<DoubleShareLocal> = (0..n)
            .map(|i| generate_double_sharing(i, 11, &pre_shared[i], &family, &modulus))
            .collect();

        let result = drive_a2t_sequential(&additive, &ds, &family, &modulus, aggregator);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result, &modulus);
        assert_eq!(reconstructed.value, c.value);
    }

    #[test]
    fn test_a2t_of_zero_additive_reconstructs_r() {
        // Sanity: additive zero ⟹ result [c=0], but since "result" is
        // expressed as [r] + (0 - r) = [0]. Verifies A2T preserves additive
        // semantics with zero input.
        let n = 3;
        let t = 1;
        let aggregator = 0usize;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let zeros: Vec<Fp> = (0..n).map(|_| Fp::zero(&modulus)).collect();
        let ds: Vec<DoubleShareLocal> = (0..n)
            .map(|i| generate_double_sharing(i, 13, &pre_shared[i], &family, &modulus))
            .collect();

        let result = drive_a2t_sequential(&zeros, &ds, &family, &modulus, aggregator);
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result, &modulus);
        assert_eq!(reconstructed.value, BigUint::from(0u32));
    }

    // ---- rss_mul_all_parties_with_record ---------------------------------

    #[test]
    fn test_rss_mul_with_record_matches_plain() {
        // The "with record" variant must produce the same output shares as
        // `rss_mul_all_parties` given identical inputs (same counter → same r).
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let a = Fp::new(BigUint::from(41u32), &modulus);
        let b = Fp::new(BigUint::from(53u32), &modulus);
        let (a_s, b_s, ds) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 5);

        let (plain, _) = rss_mul_all_parties(&a_s, &b_s, &ds, &family, &modulus);
        let (withrec, _, _) =
            rss_mul_all_parties_with_record(&a_s, &b_s, &ds, &family, &modulus);

        let p = ReplicatedSharing::reconstruct_from_party_shares(&plain, &modulus);
        let w = ReplicatedSharing::reconstruct_from_party_shares(&withrec, &modulus);
        assert_eq!(p.value, w.value);
    }

    #[test]
    fn test_rss_mul_with_record_cp_sum_equals_ab() {
        // Invariant: Σ_i record.party_cp[i] = a·b (the additive shares of
        // ab sum to ab). This is what cross_multiply is supposed to produce.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let a = Fp::new(BigUint::from(9u32), &modulus);
        let b = Fp::new(BigUint::from(37u32), &modulus);
        let (a_s, b_s, ds) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 0);

        let (_, record, _) =
            rss_mul_all_parties_with_record(&a_s, &b_s, &ds, &family, &modulus);

        let mut sum = Fp::zero(&modulus);
        for c in &record.party_cp { sum = &sum + c; }
        let expected = &a * &b;
        assert_eq!(sum.value, expected.value);
    }

    #[test]
    fn test_rss_mul_with_record_pairs_reproduce_cp() {
        // Each party's stored (a_val, b_val) pairs must multiply-and-sum back
        // to that party's cp — this is what DZKP relies on.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let a = Fp::new(BigUint::from(15u32), &modulus);
        let b = Fp::new(BigUint::from(4u32), &modulus);
        let (a_s, b_s, ds) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 0);

        let (_, record, _) =
            rss_mul_all_parties_with_record(&a_s, &b_s, &ds, &family, &modulus);

        for i in 0..n {
            let mut sum = Fp::zero(&modulus);
            for (a_val, b_val) in &record.party_pairs[i] {
                sum = &sum + &(a_val * b_val);
            }
            assert_eq!(
                sum.value, record.party_cp[i].value,
                "party {i}: pairs must reconstruct cp",
            );
        }
    }

    #[test]
    fn test_rss_mul_with_record_ab_shares_stored() {
        // The record should echo the input RSS shares verbatim (DZKP uses
        // them as the claim being proved).
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let a = Fp::new(BigUint::from(1u32), &modulus);
        let b = Fp::new(BigUint::from(1u32), &modulus);
        let (a_s, b_s, ds) = make_party_material(&a, &b, &family, &modulus, &pre_shared, 0);

        let (c_shares, record, _) =
            rss_mul_all_parties_with_record(&a_s, &b_s, &ds, &family, &modulus);

        assert_eq!(record.a_shares.len(), n);
        assert_eq!(record.b_shares.len(), n);
        assert_eq!(record.c_shares.len(), n);
        for i in 0..n {
            assert_eq!(record.a_shares[i].party_id, a_s[i].party_id);
            assert_eq!(record.a_shares[i].shares, a_s[i].shares);
            assert_eq!(record.b_shares[i].shares, b_s[i].shares);
            assert_eq!(record.c_shares[i].shares, c_shares[i].shares);
        }
    }

    // ---- deg_mul ----------------------------------------------------------

    #[test]
    fn test_deg_mul_matches_plain_product() {
        // Π_DegMul: given [q] and ⟨v⟩_{T'}, output [q · v].
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let mut rng = rand::thread_rng();

        let q = Fp::new(BigUint::from(13u32), &modulus);
        let v = Fp::new(BigUint::from(17u32), &modulus);
        let expected = &q * &v;

        let q_sharing = share(&q, &family, &modulus, &mut rng);
        let q_shares: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&q_sharing, i, &family)).collect();

        let target_subset = family.subsets[1].clone();
        let encoding = DegenerateEncoding { target_subset: target_subset.clone(), value: v };
        let double_shares: Vec<DoubleShareLocal> = (0..n)
            .map(|p| {
                generate_double_sharing_degenerate(
                    p, 0, &target_subset, &pre_shared[p], &family, &modulus,
                )
            })
            .collect();

        let (result_shares, _record, _comm) =
            deg_mul(&q_shares, &encoding, &double_shares, &family, &modulus);

        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&result_shares, &modulus);
        assert_eq!(reconstructed.value, expected.value);
    }

    #[test]
    fn test_deg_mul_correct_for_every_target_subset() {
        // The identity q·v must hold regardless of which T' ∈ T is the
        // target of the degenerate encoding.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let mut rng = rand::thread_rng();

        let q = Fp::new(BigUint::from(21u32), &modulus);
        let v = Fp::new(BigUint::from(5u32), &modulus);
        let expected = &q * &v;
        let q_sharing = share(&q, &family, &modulus, &mut rng);
        let q_shares: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&q_sharing, i, &family)).collect();

        for (idx, target) in family.subsets.iter().enumerate() {
            let target = target.clone();
            let encoding =
                DegenerateEncoding { target_subset: target.clone(), value: v.clone() };
            let double_shares: Vec<DoubleShareLocal> = (0..n)
                .map(|p| {
                    generate_double_sharing_degenerate(
                        p, idx as u64, &target, &pre_shared[p], &family, &modulus,
                    )
                })
                .collect();
            let (result_shares, _r, _c) =
                deg_mul(&q_shares, &encoding, &double_shares, &family, &modulus);
            let reconstructed =
                ReplicatedSharing::reconstruct_from_party_shares(&result_shares, &modulus);
            assert_eq!(
                reconstructed.value, expected.value,
                "deg_mul failed with target T' = {:?}", target,
            );
        }
    }

    #[test]
    fn test_deg_mul_by_zero_is_zero() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let mut rng = rand::thread_rng();

        let q = Fp::new(BigUint::from(40u32), &modulus);
        let q_sharing = share(&q, &family, &modulus, &mut rng);
        let q_shares: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&q_sharing, i, &family)).collect();

        let target = family.subsets[1].clone();
        let encoding = DegenerateEncoding { target_subset: target.clone(), value: Fp::zero(&modulus) };
        let ds: Vec<DoubleShareLocal> = (0..n)
            .map(|p| generate_double_sharing_degenerate(p, 0, &target, &pre_shared[p], &family, &modulus))
            .collect();
        let (result, _, _) = deg_mul(&q_shares, &encoding, &ds, &family, &modulus);
        let r = ReplicatedSharing::reconstruct_from_party_shares(&result, &modulus);
        assert_eq!(r.value, BigUint::from(0u32));
    }

    #[test]
    fn test_deg_mul_comm_pattern_two_rounds_p2p_only() {
        // Π_DegMul per spec: 2 rounds, P2P only.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let mut rng = rand::thread_rng();
        let q = Fp::new(BigUint::from(8u32), &modulus);
        let v = Fp::new(BigUint::from(3u32), &modulus);
        let q_sharing = share(&q, &family, &modulus, &mut rng);
        let q_shares: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&q_sharing, i, &family)).collect();
        let target = family.subsets[0].clone();
        let encoding = DegenerateEncoding { target_subset: target.clone(), value: v };
        let ds: Vec<DoubleShareLocal> = (0..n)
            .map(|p| generate_double_sharing_degenerate(p, 0, &target, &pre_shared[p], &family, &modulus))
            .collect();

        let (_, _, comm) = deg_mul(&q_shares, &encoding, &ds, &family, &modulus);
        assert_eq!(comm.rounds, 2, "Π_DegMul uses 2 synchronous rounds");
        assert_eq!(comm.broadcast_bytes, 0, "Π_DegMul is P2P-only");
        assert!(comm.p2p_bytes > 0);
    }

    #[test]
    fn test_deg_mul_record_cp_sum_equals_qv() {
        // Analogous to rss_mul: Σ_i record.party_cp[i] should equal q·v.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let mut rng = rand::thread_rng();

        let q = Fp::new(BigUint::from(11u32), &modulus);
        let v = Fp::new(BigUint::from(7u32), &modulus);
        let expected = &q * &v;

        let q_sharing = share(&q, &family, &modulus, &mut rng);
        let q_shares: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&q_sharing, i, &family)).collect();
        let target = family.subsets[1].clone();
        let encoding = DegenerateEncoding { target_subset: target.clone(), value: v };
        let ds: Vec<DoubleShareLocal> = (0..n)
            .map(|p| generate_double_sharing_degenerate(p, 0, &target, &pre_shared[p], &family, &modulus))
            .collect();

        let (_, record, _) = deg_mul(&q_shares, &encoding, &ds, &family, &modulus);
        let mut sum = Fp::zero(&modulus);
        for c in &record.party_cp { sum = &sum + c; }
        assert_eq!(sum.value, expected.value);
    }

    #[test]
    fn test_deg_mul_record_pairs_reproduce_cp() {
        // deg_mul's record pairs are (q_T, scalar) for each assignment; their
        // product-sum must equal the stored cp.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let mut rng = rand::thread_rng();

        let q = Fp::new(BigUint::from(6u32), &modulus);
        let v = Fp::new(BigUint::from(4u32), &modulus);
        let q_sharing = share(&q, &family, &modulus, &mut rng);
        let q_shares: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&q_sharing, i, &family)).collect();
        let target = family.subsets[2].clone();
        let encoding = DegenerateEncoding { target_subset: target.clone(), value: v };
        let ds: Vec<DoubleShareLocal> = (0..n)
            .map(|p| generate_double_sharing_degenerate(p, 0, &target, &pre_shared[p], &family, &modulus))
            .collect();

        let (_, record, _) = deg_mul(&q_shares, &encoding, &ds, &family, &modulus);
        for i in 0..n {
            let mut sum = Fp::zero(&modulus);
            for (a_val, b_val) in &record.party_pairs[i] {
                sum = &sum + &(a_val * b_val);
            }
            assert_eq!(
                sum.value, record.party_cp[i].value,
                "deg_mul party {i}: pairs must reconstruct cp",
            );
        }
    }

    #[test]
    fn test_deg_mul_only_active_parties_compute() {
        // In Π_DegMul only 𝒜 = [n]\T' participates in cross-multiplication;
        // parties in T' must have empty pair lists (cp = 0).
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let mut rng = rand::thread_rng();

        let q = Fp::new(BigUint::from(3u32), &modulus);
        let v = Fp::new(BigUint::from(9u32), &modulus);
        let q_sharing = share(&q, &family, &modulus, &mut rng);
        let q_shares: Vec<RssShare> =
            (0..n).map(|i| get_party_share(&q_sharing, i, &family)).collect();
        let target = family.subsets[1].clone(); // {1}, so 𝒜 = {0, 2}
        let encoding = DegenerateEncoding { target_subset: target.clone(), value: v };
        let ds: Vec<DoubleShareLocal> = (0..n)
            .map(|p| generate_double_sharing_degenerate(p, 0, &target, &pre_shared[p], &family, &modulus))
            .collect();

        let (_, record, _) = deg_mul(&q_shares, &encoding, &ds, &family, &modulus);
        for i in 0..n {
            if target.contains(&i) {
                assert!(record.party_pairs[i].is_empty(),
                    "party {i} is in T' and must not compute cross-products");
                assert_eq!(record.party_cp[i].value, BigUint::from(0u32));
            }
        }
    }
}
