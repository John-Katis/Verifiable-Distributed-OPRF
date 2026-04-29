//! Byte accounting for `Π_RSS.Share` (Appendix Protocol `fig:rss_share`,
//! `overleaf-protocols/Chapters/appendix.tex:92–108`).
//!
//! The protocol realises VSS of a secret `s` at amortised cost
//! `c_Share ≈ (3t+1)/(2t+1) ≈ 1.5` field elements per party, by piggybacking
//! on a pre-computed double sharing `(⟨r⟩, ⟦r⟧)` from `Π_DoubleRand`:
//!
//! 1. Each server `S_i` ships its `⟨r⟩_i` to the dealer `D` (1 Fp per server).
//! 2. `D` reconstructs `r = Σ_i ⟨r⟩_i`, computes `δ = s − r`, and sends `δ`
//!    to each of the `t+1` servers in a canonical subset `Q ⊂ [n]`.
//!
//! The total wire cost per share is `(n−1) + (t+1) = n+t` field elements —
//! i.e. `(3t+1)/(2t+1)` per server when amortised across all n parties
//! (amortisation matches `appendix.tex:183`).
//!
//! This module provides bench-time byte accounting for the protocol. The
//! actual `ReplicatedSharing` produced by the VSS is still constructed via
//! `vdoprf_ss::share` + `get_party_share` — in the simulated network model
//! all the secrets and shares are materialised locally, so we only need to
//! charge the correct byte totals on the `SimulatedNetwork` to reflect the
//! protocol's wire cost. Call `charge_rss_share_p2p` once per logical VSS
//! (batching `num_values` shares under one invocation packs one payload per
//! (sender, recipient) pair, matching what a real implementation would
//! pack).

use num_bigint::BigUint;
use vdoprf_network::SimulatedNetwork;
use vdoprf_ss::SubsetFamily;

fn fe_bytes(modulus: &BigUint) -> usize {
    ((modulus.bits() + 7) / 8) as usize
}

/// Canonical `Q ⊂ [n]` of size `t+1` — the holders of the first subset
/// `T_0 ∈ 𝒯`. Every RSS instance has such a canonical set, so we fix on it
/// to avoid threading an extra parameter through every call site.
fn canonical_q(family: &SubsetFamily) -> Vec<usize> {
    (0..family.n)
        .filter(|i| !family.subsets[0].contains(i))
        .collect()
}

/// Server-dealer VSS: charge the p2p bytes of a single `Π_RSS.Share`
/// instance rooted at `dealer_id` over `num_values` shared values.
///
/// `num_values` lets the caller batch multiple VSSes into one invocation —
/// e.g. `Π_VIP`'s `γ` fold iterations share `q(1), q(2), q(3)` each, so a
/// whole VIP instance's VSSes can be charged as one call with
/// `num_values = 3·γ + 2` (the final `+2` is for the end-of-loop `a₁, b₁`).
/// The payload size on every (sender, recipient) p2p is `num_values · feb`.
pub fn charge_rss_share_p2p(
    dealer_id: usize,
    num_values: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
    net: &mut SimulatedNetwork,
) {
    if num_values == 0 {
        return;
    }
    let feb = fe_bytes(modulus);
    let payload_bytes = num_values * feb;
    let n = family.n;

    // Round 1: each non-dealer server ships its ⟨r⟩_i to the dealer. One Fp
    // per shared value.
    for sender in 0..n {
        if sender != dealer_id {
            net.send_p2p(sender, dealer_id, vec![0u8; payload_bytes]);
        }
    }

    // Round 2: dealer ships δ = s − r to each member of the canonical Q that
    // isn't itself. One Fp per shared value.
    for recipient in canonical_q(family) {
        if recipient != dealer_id {
            net.send_p2p(dealer_id, recipient, vec![0u8; payload_bytes]);
        }
    }
}

/// Charge one `F_coin` invocation per `appendix.tex:1052–1056`.
///
/// `F_coin` is realised by `F_Rand → RSS.Open` of one preprocessed `[r]`:
/// every server broadcasts its RSS share (`N' = C(n-1, t)` field elements)
/// and the call advances one synchronous round. Total wire cost is
/// `n · N' · feb` bytes broadcast.
///
/// Used at the two protocol-level F_coin draws that survive the Fiat–Shamir
/// transform: `Π_VIP^Prl`'s shared σ-update ε's (`5-Online.tex:169, 190`)
/// and Boyle's `Π_proveDeg2Rel`'s closing σ-update ε (Protocol 3.3 step 3e).
pub fn charge_f_coin(net: &mut SimulatedNetwork, family: &SubsetFamily, modulus: &BigUint) {
    let n_prime = family.subsets_not_containing(0).len();
    let feb = fe_bytes(modulus);
    let payload_bytes = n_prime * feb;
    for s in 0..family.n {
        net.broadcast(s, vec![0u8; payload_bytes]);
    }
    net.next_round();
}

/// Naive RSS.Share — textbook baseline (no `Π_DoubleRand` piggyback).
///
/// Dealer directly p2p-sends each subset share to every holder of that
/// subset, with no preprocessed `(⟨r⟩, ⟦r⟧)` trick. 1 round.
///
/// Per recipient, the dealer ships the `C(n-1, t)` shares the recipient
/// holds (subsets not containing `recipient`), one Fp per shared value, so
/// the per-(dealer→recipient) p2p payload is
/// `num_values · C(n-1, t) · feb`. Dealer-to-self traffic is skipped.
///
/// Used as the `Π_RSS.Share` cost model for the Boyle/BGIN20 baseline so
/// the comparison against vDOPRF reflects unoptimised primitives.
pub fn charge_naive_rss_share(
    dealer_id: usize,
    num_values: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
    net: &mut SimulatedNetwork,
) {
    if num_values == 0 {
        return;
    }
    let feb = fe_bytes(modulus);
    let n = family.n;
    for recipient in 0..n {
        if recipient == dealer_id {
            continue;
        }
        let holds = family.subsets_not_containing(recipient).len();
        let payload_bytes = num_values * holds * feb;
        if payload_bytes > 0 {
            net.send_p2p(dealer_id, recipient, vec![0u8; payload_bytes]);
        }
    }
}

/// Naive RSS.Open — textbook baseline (no hash compression).
///
/// Each of the `n` servers broadcasts its full holdings (`C(n-1, t)` field
/// elements per opened value) in clear; the receiver collects the multiple
/// copies of every subset share it does not hold and aborts on
/// disagreement. 1 round, no hashes — strictly more wire bytes than the
/// `C(n-1, t-1) + (n-1)` hash-check variant for typical (n, t).
///
/// Per server cost: `num_values · C(n-1, t) · feb` broadcast bytes.
pub fn charge_naive_rss_open(
    num_values: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
    net: &mut SimulatedNetwork,
) {
    if num_values == 0 {
        return;
    }
    let feb = fe_bytes(modulus);
    let n = family.n;
    let holds = family.subsets_not_containing(0).len();
    let payload_bytes = num_values * holds * feb;
    for s in 0..n {
        net.broadcast(s, vec![0u8; payload_bytes]);
    }
}

/// Client-dealer variant of `charge_rss_share_p2p` for client→server input
/// distribution. The client is external to the numbered parties, so all
/// traffic is charged to `client_bytes` via `send_to_client`.
///
/// Shape mirrors the server-dealer protocol: servers ship ⟨r⟩_i to the
/// client (round 1), client ships δ to the `t+1` members of Q (round 2).
/// Per-shared-value wire cost: `(n + t + 1)·feb` total, `≈ 1.5·feb` per
/// party after amortisation — matches `appendix.tex:1099` for client VSS.
pub fn charge_client_rss_share(
    num_values: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
    net: &mut SimulatedNetwork,
) {
    if num_values == 0 {
        return;
    }
    let feb = fe_bytes(modulus);
    let payload_bytes = num_values * feb;
    let n = family.n;

    // Round 1: each server ships ⟨r⟩_i to the client.
    for sender in 0..n {
        net.send_to_client(sender, vec![0u8; payload_bytes]);
    }

    // Round 2: client ships δ to each member of Q. `client_buf` is the
    // bidirectional bucket (both directions land in `client_bytes`), so we
    // reuse `send_to_client` with arbitrary sender tags; the bench reports
    // the sum, not per-direction.
    for recipient in canonical_q(family) {
        net.send_to_client(recipient, vec![0u8; payload_bytes]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_rss_share_p2p_per_party_is_near_one_and_a_half() {
        // (n, t) = (9, 4) → 3t+1 = 13 total Fp per VSS (paper formula),
        // rather n+t wire Fp in our model since Q is t+1 and the dealer is
        // usually not in Q for an arbitrary choice. Check the exact total
        // for a fixed dealer outside Q.
        let n = 9;
        let t = 4;
        let family = SubsetFamily::new(n, t);
        // 128-bit prime, feb = 16.
        let modulus = BigUint::parse_bytes(b"170141183460469231731687303715884105727", 10).unwrap();
        let feb = fe_bytes(&modulus);

        let q: Vec<usize> = (0..n).filter(|i| !family.subsets[0].contains(i)).collect();
        // Pick a dealer NOT in Q (dealer ∈ T_0): matches the (3t+1)/(2t+1)
        // formula from the appendix.
        let dealer = family.subsets[0].iter().next().unwrap();
        assert!(!q.contains(&dealer));

        let mut net = SimulatedNetwork::new(n);
        charge_rss_share_p2p(dealer, 1, &family, &modulus, &mut net);

        // Expected: (n-1) incoming to dealer + (t+1) outgoing from dealer.
        let expected_fp = (n - 1) + (t + 1);
        assert_eq!(net.stats().p2p_bytes, expected_fp * feb);
    }

    #[test]
    fn charge_rss_share_p2p_batched_scales_linearly() {
        let family = SubsetFamily::new(5, 2);
        let modulus = BigUint::from(65537u32);
        let feb = fe_bytes(&modulus);

        let mut net1 = SimulatedNetwork::new(5);
        charge_rss_share_p2p(0, 1, &family, &modulus, &mut net1);
        let mut net3 = SimulatedNetwork::new(5);
        charge_rss_share_p2p(0, 3, &family, &modulus, &mut net3);

        assert_eq!(net3.stats().p2p_bytes, 3 * net1.stats().p2p_bytes);
        let _ = feb;
    }

    #[test]
    fn charge_client_rss_share_uses_client_bucket() {
        let family = SubsetFamily::new(5, 2);
        let modulus = BigUint::from(65537u32);
        let feb = fe_bytes(&modulus);

        let mut net = SimulatedNetwork::new(5);
        charge_client_rss_share(1, &family, &modulus, &mut net);

        let stats = net.stats();
        assert_eq!(stats.p2p_bytes, 0);
        assert_eq!(stats.broadcast_bytes, 0);
        // (n + t + 1) = 5 + 2 + 1 = 8 Fp.
        assert_eq!(stats.client_bytes, 8 * feb);
    }
}
