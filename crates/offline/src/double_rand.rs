//! Protocol Π_DoubleRand: Non-interactive generation of double sharings (⟨r⟩, [r]).
//!
//! Each party locally computes its shares using pairwise PRG seeds,
//! with no communication required.

use num_bigint::BigUint;
use std::collections::BTreeMap;
use vdoprf_field::Fp;
use vdoprf_ss::{covering_policy, SubsetFamily, SubsetT, RssShare, ReplicatedSharing};
use crate::PreSharedMaterial;

/// Result of DoubleRand for one party: its additive share and RSS share of the same random r.
#[derive(Clone, Debug)]
pub struct DoubleShareLocal {
    /// Party's additive share of r: ⟨r⟩_i
    pub additive_share: Fp,
    /// Party's RSS share of r: [r]_i
    pub rss_share: RssShare,
}

/// Generate a double sharing for a given counter.
///
/// Protocol Π_DoubleRand (Chapters/2-Preliminaries.tex):
/// 1. For each subset T, the PRF key k_T generates addss_T = PRF(k_T, counter).
///    All parties not in T know this value (replicated).
/// 2. The additive share for party i is:
///      ⟨r⟩_i = Σ_{T: P(T)=i} addss_T  +  Σ_{j ≠ i} (-1)^𝟙[j<i] · PRG(s_{i,j}, counter)
///    where P is the covering policy and s_{i,j} is the pairwise PRG seed
///    between parties i and j. The pairwise term re-randomizes the otherwise-
///    deterministic PRF-based share while keeping the zero-sum invariant
///    (each pair (i,j) contributes once with + and once with −).
/// 3. The RSS share for party i is: [r]_i = {addss_T : i ∉ T}.
///
/// The secret r = Σ_T addss_T is the same for both the additive and RSS forms.
pub fn generate_double_sharing(
    party_id: usize,
    counter: u64,
    pre_shared: &PreSharedMaterial,
    family: &SubsetFamily,
    modulus: &BigUint,
) -> DoubleShareLocal {
    // Step 1: Compute all addss_T values that this party knows
    // Party i knows addss_T for all T where i ∉ T (since it holds key k_T).
    let mut all_known_components: BTreeMap<SubsetT, Fp> = BTreeMap::new();

    for (subset, prf) in &pre_shared.prf_keys {
        let addss_t = prf.evaluate(counter, modulus);
        all_known_components.insert(subset.clone(), addss_t);
    }

    // Step 2a: PRF-based additive contribution ⟨r⟩_i = Σ_{T: P(T) = i} addss_T
    let mut additive_share = Fp::zero(modulus);
    for subset in &family.subsets {
        let p_t = covering_policy(subset, family.n);
        if p_t == party_id {
            // Party i is the designated holder for this T
            // It must know addss_T, which means i ∉ T (guaranteed by covering policy)
            let addss_t = &all_known_components[subset];
            additive_share = &additive_share + addss_t;
        }
    }

    // Step 2b: Pairwise-PRG re-randomization:
    //   Σ_{j ≠ i} (-1)^𝟙[j<i] · PRG(s_{i,j}, counter)
    // Seeds are stored once per unordered pair under key (min, max).
    for j in 0..family.n {
        if j == party_id { continue; }
        let key = if party_id < j { (party_id, j) } else { (j, party_id) };
        let prg_term = pre_shared.prg_seeds[&key].generate(counter, modulus);
        if j < party_id {
            additive_share = &additive_share - &prg_term;
        } else {
            additive_share = &additive_share + &prg_term;
        }
    }

    // Step 3: RSS share [r]_i = {addss_T : i ∉ T}
    let rss_shares: BTreeMap<SubsetT, Fp> = all_known_components;

    DoubleShareLocal {
        additive_share,
        rss_share: RssShare {
            party_id,
            shares: rss_shares,
        },
    }
}

/// Functionality 𝓕_Rand (§4-Online line 203 / line 357): RSS-only random
/// values, preprocessable in the offline phase.
///
/// The paper's `Π_VIP^Prl` calls `𝓕_Rand` to obtain `\rss{a_0}, \rss{b_0}`.
/// Unlike `𝓕_DoubleRand` (Preliminaries) there is no additive half and no
/// pairwise-PRG re-randomisation is needed — consumers use only the RSS
/// form, so we return just that.
///
/// Implementation: party i evaluates its replicated PRF keys (the ones for
/// subsets `T` with `i ∉ T`) at `counter`. No communication; deterministic
/// under the shared PRF seeds, so a single counter stream across servers
/// agrees on the same underlying secret.
pub fn generate_rss_random(
    party_id: usize,
    counter: u64,
    pre_shared: &PreSharedMaterial,
    _family: &SubsetFamily,
    modulus: &BigUint,
) -> RssShare {
    let mut shares: BTreeMap<SubsetT, Fp> = BTreeMap::new();
    for (subset, prf) in &pre_shared.prf_keys {
        shares.insert(subset.clone(), prf.evaluate(counter, modulus));
    }
    RssShare {
        party_id,
        shares,
    }
}

/// Additive sharing of zero (Functionality 𝓕_Zero).
///
/// Each party derives `z_i` from pairwise PRG keys so that `Σ_i z_i = 0`:
///   z_i = Σ_{j: j > i} PRG(s_{i,j}, counter)  −  Σ_{j: j < i} PRG(s_{j,i}, counter)
/// Each ordered pair (i, j) contributes once with `+` and once with `−`, so
/// the sum over all parties telescopes to zero. No communication needed.
pub fn generate_zero_additive_sharing(
    party_id: usize,
    counter: u64,
    pre_shared: &PreSharedMaterial,
    family: &SubsetFamily,
    modulus: &BigUint,
) -> Fp {
    let mut z = Fp::zero(modulus);
    for j in 0..family.n {
        if j == party_id { continue; }
        let key = if party_id < j { (party_id, j) } else { (j, party_id) };
        let term = pre_shared.prg_seeds[&key].generate(counter, modulus);
        if j < party_id {
            z = &z - &term;
        } else {
            z = &z + &term;
        }
    }
    z
}

/// Full double sharing result (for testing: combines all parties' views).
#[derive(Clone, Debug)]
pub struct DoubleSharing {
    /// Additive sharing of r.
    pub additive_shares: Vec<Fp>,
    /// Replicated sharing of r.
    pub replicated: ReplicatedSharing,
}

/// Generate double sharings for all parties (testing helper).
pub fn generate_double_sharing_all(
    counter: u64,
    pre_shared_all: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> DoubleSharing {
    let n = family.n;
    let locals: Vec<DoubleShareLocal> = (0..n)
        .map(|i| {
            generate_double_sharing(i, counter, &pre_shared_all[i], family, modulus)
        })
        .collect();

    let additive_shares: Vec<Fp> = locals.iter().map(|l| l.additive_share.clone()).collect();

    // The replicated sharing components are the same as the PRF outputs.
    // Extract from party 0's view (all components it knows) plus party-0's missing ones
    // from other parties.
    let mut components: BTreeMap<SubsetT, Fp> = BTreeMap::new();
    for subset in &family.subsets {
        // Find a party that knows this component (any party not in T)
        for i in 0..n {
            if !subset.contains(&i) {
                components.insert(subset.clone(), locals[i].rss_share.shares[subset].clone());
                break;
            }
        }
    }

    DoubleSharing {
        additive_shares,
        replicated: ReplicatedSharing { components },
    }
}

/// Target-matched variant Π_DoubleRand_deg for Approach II.
/// Generates a double sharing where the additive part is a degenerate encoding
/// targeted at a specific subset T'.
///
/// Only parties in A = [n] \ T' contribute an additive share.
pub fn generate_double_sharing_degenerate(
    party_id: usize,
    counter: u64,
    target_subset: &SubsetT,
    pre_shared: &PreSharedMaterial,
    family: &SubsetFamily,
    modulus: &BigUint,
) -> DoubleShareLocal {
    // The active parties are A = [n] \ target_subset
    let active: Vec<usize> = (0..family.n)
        .filter(|i| !target_subset.contains(i))
        .collect();

    // Compute all known RSS components
    let mut all_known_components: BTreeMap<SubsetT, Fp> = BTreeMap::new();
    for (subset, prf) in &pre_shared.prf_keys {
        let addss_t = prf.evaluate(counter, modulus);
        all_known_components.insert(subset.clone(), addss_t);
    }

    // Additive share: only active parties get a share.
    // Use a modified covering policy that only assigns to active parties.
    // Analog of Π_DoubleRand^deg (appendix): the additive contribution is
    //   ⟨r⟩_i = Σ_{T: P_𝒜(T)=i} addss_T  +  Σ_{ℓ ∈ 𝒜, ℓ ≠ i} (-1)^𝟙[ℓ<i] · PRG(s_{i,ℓ}, ctr)
    // where pairwise PRG re-randomization ranges over active parties only,
    // preserving the zero-sum invariant within 𝒜.
    let mut additive_share = Fp::zero(modulus);
    if active.contains(&party_id) {
        for subset in &family.subsets {
            // Modified covering: assign to smallest active party not in T
            let assigned = active
                .iter()
                .find(|&&a| !subset.contains(&a))
                .copied();
            if let Some(a) = assigned {
                if a == party_id {
                    if let Some(addss_t) = all_known_components.get(subset) {
                        additive_share = &additive_share + addss_t;
                    }
                }
            }
        }
        // Pairwise-PRG re-randomization restricted to 𝒜.
        for &ell in &active {
            if ell == party_id { continue; }
            let key = if party_id < ell { (party_id, ell) } else { (ell, party_id) };
            let prg_term = pre_shared.prg_seeds[&key].generate(counter, modulus);
            if ell < party_id {
                additive_share = &additive_share - &prg_term;
            } else {
                additive_share = &additive_share + &prg_term;
            }
        }
    }

    DoubleShareLocal {
        additive_share,
        rss_share: RssShare {
            party_id,
            shares: all_known_components,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_pre_shared;

    #[test]
    fn test_double_sharing_consistency() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let ds = generate_double_sharing_all(0, &pre_shared, &family, &modulus);

        // The sum of additive shares should equal the RSS reconstruction
        let mut add_sum = Fp::zero(&modulus);
        for share in &ds.additive_shares {
            add_sum = &add_sum + share;
        }

        let rss_sum = ds.replicated.reconstruct(&modulus);
        assert_eq!(add_sum.value, rss_sum.value);
    }

    #[test]
    fn test_double_sharing_different_counters() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let ds1 = generate_double_sharing_all(0, &pre_shared, &family, &modulus);
        let ds2 = generate_double_sharing_all(1, &pre_shared, &family, &modulus);

        let r1 = ds1.replicated.reconstruct(&modulus);
        let r2 = ds2.replicated.reconstruct(&modulus);
        // Should be different random values (with overwhelming probability)
        assert_ne!(r1.value, r2.value);
    }

    #[test]
    fn test_additive_share_includes_prg_rerandomization() {
        // The spec mandates a pairwise-PRG zero-sharing summand that
        // re-randomizes the otherwise-deterministic PRF-based additive
        // share. Individual parties' additive shares must therefore depend
        // on pairwise seeds, not just PRF outputs. Compare an honest run
        // against a hypothetical PRF-only additive share: they must differ.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let local = generate_double_sharing(0, 42, &pre_shared[0], &family, &modulus);

        // Reconstruct the "PRF-only" share that the pre-patch code would
        // produce: Σ_{T: P(T)=0} addss_T.
        let mut prf_only = Fp::zero(&modulus);
        for subset in &family.subsets {
            if covering_policy(subset, n) == 0 {
                let addss_t = pre_shared[0].prf_keys[subset].evaluate(42, &modulus);
                prf_only = &prf_only + &addss_t;
            }
        }
        // Under a negligible-collision assumption the PRG term is non-zero.
        assert_ne!(local.additive_share.value, prf_only.value,
            "additive share must include pairwise-PRG re-randomization");
    }

    #[test]
    fn test_zero_additive_sharing_sums_to_zero() {
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(257u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let mut sum = Fp::zero(&modulus);
        for i in 0..n {
            let z = generate_zero_additive_sharing(i, 7, &pre_shared[i], &family, &modulus);
            sum = &sum + &z;
        }
        assert_eq!(sum.value, BigUint::from(0u32));
    }

    #[test]
    fn test_zero_additive_sharing_different_counters() {
        // Different counters must yield different per-party shares (they are
        // PRG-derived) — but every counter still sums to zero.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let z0 = generate_zero_additive_sharing(0, 0, &pre_shared[0], &family, &modulus);
        let z1 = generate_zero_additive_sharing(0, 1, &pre_shared[0], &family, &modulus);
        assert_ne!(z0.value, z1.value);
    }

    /// `generate_rss_random` matches the RSS half of `generate_double_sharing`
    /// on the same counter / pre-shared material — confirms F_Rand and
    /// F_DoubleRand produce the same underlying replicated sharing.
    #[test]
    fn test_rss_random_matches_double_sharing_rss_half() {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        for counter in [0u64, 7, 42, 999] {
            for i in 0..n {
                let via_dbl =
                    generate_double_sharing(i, counter, &pre_shared[i], &family, &modulus).rss_share;
                let via_rand =
                    generate_rss_random(i, counter, &pre_shared[i], &family, &modulus);
                assert_eq!(via_dbl.party_id, via_rand.party_id);
                assert_eq!(via_dbl.shares.len(), via_rand.shares.len());
                for (k, v) in &via_dbl.shares {
                    assert_eq!(via_rand.shares[k].value, v.value);
                }
            }
        }
    }

    /// Per-party RSS shares from `generate_rss_random` reassemble to a
    /// consistent secret: Party i holds exactly the components for subsets
    /// not containing i, and they agree with the other parties that hold
    /// the same subset.
    #[test]
    fn test_rss_random_reconstructs_consistently() {
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(257u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let shares: Vec<RssShare> = (0..n)
            .map(|i| generate_rss_random(i, 17, &pre_shared[i], &family, &modulus))
            .collect();

        // Each subset is held by every party not in it, and they all agree.
        for subset in &family.subsets {
            let holders: Vec<&RssShare> = shares
                .iter()
                .filter(|rs| !subset.contains(&rs.party_id))
                .collect();
            let first = &holders[0].shares[subset];
            for h in &holders[1..] {
                assert_eq!(h.shares[subset].value, first.value);
            }
        }
        // And the reassembled sharing is complete (all N subsets covered).
        let rss = ReplicatedSharing::from_party_shares(&shares);
        assert_eq!(rss.components.len(), family.subsets.len());
    }

    #[test]
    fn test_additive_share_degenerate_includes_prg_rerandomization() {
        // Same property for the degenerate-encoding variant: pairwise-PRG
        // term ranges over 𝒜 only, but must still re-randomize the share.
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(113u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);

        let target = family.subsets[1].clone(); // T' = {1}, 𝒜 = {0, 2}
        let local = generate_double_sharing_degenerate(
            0, 42, &target, &pre_shared[0], &family, &modulus,
        );

        // Hypothetical PRF-only additive: Σ over T assigned to 0 under the
        // degenerate covering policy (smallest active party not in T = 0).
        let active: Vec<usize> = (0..n).filter(|i| !target.contains(i)).collect();
        let mut prf_only = Fp::zero(&modulus);
        for subset in &family.subsets {
            let assigned = active.iter().find(|&&a| !subset.contains(&a)).copied();
            if assigned == Some(0) {
                if let Some(prf) = pre_shared[0].prf_keys.get(subset) {
                    prf_only = &prf_only + &prf.evaluate(42, &modulus);
                }
            }
        }
        assert_ne!(local.additive_share.value, prf_only.value,
            "degenerate additive share must include pairwise-PRG re-randomization");
    }
}
