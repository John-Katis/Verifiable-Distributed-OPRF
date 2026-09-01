//! Π_Input (Protocol 9, Appendix E.1): verifiable `(t,n)`-RSS sharing of a
//! client input.
//!
//! Replaces the ad-hoc "client generates its own RSS shares and sends them
//! directly" pattern (`vdoprf_ss::share` used as a client dealer — still
//! what `share_add_and_extract_pairs` does today) with a mask-and-broadcast
//! protocol that actually detects a malicious client sending inconsistent
//! shares to different servers, or a malicious server reporting a wrong
//! mask component. See the paper's Lemma E.1 (consistency, input
//! integrity, extractability, privacy).
//!
//! ```text
//! Input: C holds (x^(1),...,x^(m)); servers hold pre-computed RSS
//! sharings [r^(1)],...,[r^(m)] of independent uniform r^(j), each used
//! exactly once. A public assignment P:[N]→[n] with P(ℓ) ∉ T_ℓ for every
//! ℓ∈[N].
//! 1. Each server S_i sends {[r^(j)]_ℓ | j∈[m], ℓ∈[N], P(ℓ)=i} and
//!    ψ_i ← H([r^(1)]_i ‖ ... ‖ [r^(m)]_i) to C.
//! 2. C assembles all components, recomputes each server's local list, and
//!    verifies ψ_i for all i — abort on any mismatch.
//! 3. C computes r^(j) ← Σ_ℓ [r^(j)]_ℓ and u^(j) ← r^(j) + x^(j), sends
//!    (u^(1),...,u^(m)) to all n servers.
//! 4. Each server computes χ_i ← H(u^(1)‖...‖u^(m)) and echoes it to every
//!    other server — abort if any disagree.
//! 5. Each server sets [x^(j)]_i ← u^(j) − [r^(j)]_i.
//! ```
//!
//! `P: [N]→[n]` is realized by [`covering_policy`] (smallest party index
//! not in the subset) — already used for exactly this "one designated
//! party per subset" role by `generate_double_sharing`/`rss_to_additive`.
//! `[r^(j)]` is realized by [`generate_rss_random`] (deterministic,
//! PRF-keyed, no communication needed) — the same primitive `vip.rs`'s
//! `coin_toss` uses for genuine F_coin realization elsewhere in this
//! codebase.
//!
//! Two variants are exposed:
//! - [`client_input_share`] — steps 1-3 and 5 (NOT step 4's echo). This is
//!   the "folded" core: a caller that can piggyback step 4's echo onto an
//!   existing hash-exchange round (like `compute_batch`) uses this
//!   directly and folds the echo in itself.
//! - [`client_input_share_standalone`] — the FULL protocol at the paper's
//!   stated standalone 3-round cost, composing [`client_input_share`] with
//!   [`verify_echo_standalone`]. Not called by any production code path in
//!   this codebase (only `compute_batch_with_verified_input` uses the
//!   folded 2-round variant) — implemented and tested for paper-parity.

use num_bigint::BigUint;
use std::collections::BTreeMap;
use vdoprf_crypto::hash::hash_field_elements;
use vdoprf_field::Fp;
use vdoprf_network::CommStats;
use vdoprf_offline::double_rand::generate_rss_random;
use vdoprf_offline::rss_share::{charge_input_client_facing, charge_input_echo_standalone};
use vdoprf_offline::PreSharedMaterial;
use vdoprf_ss::{covering_policy, RssShare, SubsetFamily, SubsetT};

/// Verdict from Π_Input (client input sharing) verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputResult {
    Accept,
    Abort,
}

/// Step 1: assemble every additive component `[r^(j)]_ℓ` for every input
/// `j`, each shipped once by its single designated sender
/// (`covering_policy(T_ℓ, n)`). In this simulated network all parties'
/// PRF-keyed material is available in-process, so this directly produces
/// exactly what a real client would receive from the `N` designated
/// senders — `assembled[j][&T]` = the value of component `T` for input
/// `j`.
fn assemble_components(
    m: usize,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
    counter_base: u64,
) -> Vec<BTreeMap<SubsetT, Fp>> {
    let mut assembled: Vec<BTreeMap<SubsetT, Fp>> = vec![BTreeMap::new(); m];
    for subset in &family.subsets {
        let sender = covering_policy(subset, family.n);
        for j in 0..m {
            let share = generate_rss_random(
                sender,
                counter_base + j as u64,
                &pre_shared[sender],
                family,
                modulus,
            );
            let value = share
                .shares
                .get(subset)
                .cloned()
                .expect("designated sender must hold the component it's assigned to send");
            assembled[j].insert(*subset, value);
        }
    }
    assembled
}

/// One server's full local list `{[r^(j)]_ℓ | j∈[m], i∉T_ℓ}`, in the fixed
/// `family.subsets` order — used both to recompute a server's expected
/// list from `assembled` (client side) and to compute a server's own
/// independent hash (server side).
fn local_list(assembled: &[BTreeMap<SubsetT, Fp>], party: usize, family: &SubsetFamily) -> Vec<Fp> {
    let mut list = Vec::new();
    for components in assembled {
        for subset in &family.subsets {
            if !subset.contains(&party) {
                list.push(components[subset].clone());
            }
        }
    }
    list
}

/// Step 2: verify `ψ_i = H([r^(1)]_i ‖ ... ‖ [r^(m)]_i)` for every server
/// `i`, by comparing the hash of each server's list as *recomputed from
/// `assembled`* (what the client sees) against that server's own
/// *independently PRF-derived* list (what the server actually holds).
/// These match by construction under honest execution (same PRF, same
/// counters) — the check is still real, testable code: see this module's
/// tests, which tamper one `assembled` entry to prove the mismatch is
/// caught, not just structurally guaranteed to pass.
fn check_psi(
    assembled: &[BTreeMap<SubsetT, Fp>],
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
    counter_base: u64,
) -> InputResult {
    let m = assembled.len();
    for i in 0..family.n {
        let expected_list = local_list(assembled, i, family);
        let expected_psi = hash_field_elements(&expected_list);

        let mut own_list = Vec::new();
        for j in 0..m {
            let share_i =
                generate_rss_random(i, counter_base + j as u64, &pre_shared[i], family, modulus);
            for subset in &family.subsets {
                if !subset.contains(&i) {
                    own_list.push(share_i.shares[subset].clone());
                }
            }
        }
        let own_psi = hash_field_elements(&own_list);

        if expected_psi != own_psi {
            return InputResult::Abort;
        }
    }
    InputResult::Accept
}

/// Step 3: `r^(j) ← Σ_ℓ [r^(j)]_ℓ`, `u^(j) ← r^(j) + x^(j)`.
fn compute_u(assembled: &[BTreeMap<SubsetT, Fp>], xs: &[Fp], modulus: &BigUint) -> Vec<Fp> {
    assembled
        .iter()
        .zip(xs.iter())
        .map(|(components, x)| {
            let mut r = Fp::zero(modulus);
            for v in components.values() {
                r = &r + v;
            }
            &r + x
        })
        .collect()
}

/// Step 5: `[x^(j)]_i ← u^(j) − [r^(j)]_i`. Each server re-derives its own
/// `[r^(j)]_i` (same PRF, same counter as step 1) and subtracts it from
/// the public `u^(j)`: `u - r = -(r - u)`, computed via
/// `RssShare::local_sub_public` (which gives `r - u`) negated component-
/// wise.
fn derive_shares(
    u: &[Fp],
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
    counter_base: u64,
) -> Vec<Vec<RssShare>> {
    let neg_one = &Fp::zero(modulus) - &Fp::one(modulus);
    u.iter()
        .enumerate()
        .map(|(j, u_j)| {
            (0..family.n)
                .map(|i| {
                    let r_share_i = generate_rss_random(
                        i,
                        counter_base + j as u64,
                        &pre_shared[i],
                        family,
                        modulus,
                    );
                    // r_share_i.local_sub_public(u_j) = [r]_i - u; negate to get u - [r]_i.
                    r_share_i
                        .local_sub_public(u_j, family)
                        .local_scalar_mul(&neg_one)
                })
                .collect()
        })
        .collect()
}

/// Core of Π_Input — steps 1-3 (mask-and-open with abort) and step 5 (each
/// server locally derives its share of `x`). Does **not** perform step 4's
/// echo check — that's the caller's responsibility (see module docs: fold
/// it into an existing round, or call [`verify_echo_standalone`]).
///
/// `counter_base` must be a fresh, never-reused base across all calls that
/// share the same `pre_shared`: each `[r^(j)]` must be consumed by exactly
/// one input (reusing a mask leaks `u^(1)-u^(2) = x^(1)-x^(2)` to every
/// server, per the paper's own warning). Uses one PRF counter per input
/// index (`counter_base + j`).
///
/// Returns `(verdict, shares, u, comm)`: `shares[j][i]` is party `i`'s RSS
/// share of `xs[j]`; `u[j]` is the client's broadcast value for input `j`
/// (needed by callers folding or standalone-echoing step 4); `comm` covers
/// only the client-facing rounds (steps 1-3). On `Abort`, `shares`/`u` are
/// empty.
pub fn client_input_share(
    xs: &[Fp],
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
    counter_base: u64,
) -> (InputResult, Vec<Vec<RssShare>>, Vec<Fp>, CommStats) {
    let m = xs.len();
    let comm = charge_input_client_facing(m, family, modulus);
    if m == 0 {
        return (InputResult::Accept, Vec::new(), Vec::new(), comm);
    }

    let assembled = assemble_components(m, pre_shared, family, modulus, counter_base);
    if check_psi(&assembled, pre_shared, family, modulus, counter_base) == InputResult::Abort {
        return (InputResult::Abort, Vec::new(), Vec::new(), comm);
    }

    let u = compute_u(&assembled, xs, modulus);
    let shares = derive_shares(&u, pre_shared, family, modulus, counter_base);
    (InputResult::Accept, shares, u, comm)
}

/// Step 4, standalone: each of the `n` servers independently computes
/// `χ_i = H(u_i)` from its own received copy of the broadcast `u` vector
/// and compares against every other server's `χ`; abort if any two
/// disagree. `per_party_u[i]` is server `i`'s own view of `u` — under
/// honest execution every entry is identical; a malicious client sending
/// different `u` to different servers is exactly what this catches.
pub fn verify_echo_standalone(
    per_party_u: &[Vec<Fp>],
    family: &SubsetFamily,
) -> (InputResult, CommStats) {
    let comm = charge_input_echo_standalone(family);
    if per_party_u.is_empty() {
        return (InputResult::Accept, comm);
    }
    let reference = hash_field_elements(&per_party_u[0]);
    for u_i in per_party_u.iter().skip(1) {
        if hash_field_elements(u_i) != reference {
            return (InputResult::Abort, comm);
        }
    }
    (InputResult::Accept, comm)
}

/// The FULL Π_Input protocol at the paper's stated standalone 3-round
/// cost: [`client_input_share`] (rounds 1-3) composed with
/// [`verify_echo_standalone`] (round 4, using `n` identical copies of `u`
/// — every server sees the same broadcast in this in-process simulation).
/// Implemented and tested for completeness/paper-parity, but per the
/// project's confirmed scope this is dead code from the benchmark's
/// perspective: nothing in `compute.rs`, `compute_parallel.rs`, or
/// `compute_batch.rs` calls this — only the folded [`client_input_share`]
/// (used by `compute_batch_with_verified_input`) is wired into anything
/// benchmarked.
pub fn client_input_share_standalone(
    xs: &[Fp],
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
    counter_base: u64,
) -> (InputResult, Vec<Vec<RssShare>>, CommStats) {
    let (verdict, shares, u, mut comm) =
        client_input_share(xs, pre_shared, family, modulus, counter_base);
    if verdict == InputResult::Abort {
        return (InputResult::Abort, Vec::new(), comm);
    }

    let per_party_u: Vec<Vec<Fp>> = (0..family.n).map(|_| u.clone()).collect();
    let (echo_verdict, echo_comm) = verify_echo_standalone(&per_party_u, family);
    comm.merge(&echo_comm);
    if echo_verdict == InputResult::Abort {
        return (InputResult::Abort, Vec::new(), comm);
    }
    (InputResult::Accept, shares, comm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vdoprf_offline::setup_pre_shared;
    use vdoprf_ss::ReplicatedSharing;

    fn small_setup() -> (SubsetFamily, BigUint, Vec<PreSharedMaterial>) {
        let n = 5;
        let t = 2;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        (family, modulus, pre_shared)
    }

    fn reconstruct_x(shares: &[RssShare], modulus: &BigUint) -> Fp {
        ReplicatedSharing::reconstruct_from_party_shares(shares, modulus)
    }

    #[test]
    fn honest_end_to_end_reconstructs_input() {
        let (family, modulus, pre_shared) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..3).map(|_| Fp::random(&modulus, &mut rng)).collect();

        let (verdict, shares, u, _comm) =
            client_input_share(&xs, &pre_shared, &family, &modulus, 90_000);
        assert_eq!(verdict, InputResult::Accept);
        assert_eq!(shares.len(), xs.len());
        assert_eq!(u.len(), xs.len());

        for (j, x) in xs.iter().enumerate() {
            assert_eq!(reconstruct_x(&shares[j], &modulus).value, x.value);
        }
    }

    #[test]
    fn honest_end_to_end_standalone_reconstructs_input() {
        let (family, modulus, pre_shared) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..2).map(|_| Fp::random(&modulus, &mut rng)).collect();

        let (verdict, shares, comm) =
            client_input_share_standalone(&xs, &pre_shared, &family, &modulus, 91_000);
        assert_eq!(verdict, InputResult::Accept);
        for (j, x) in xs.iter().enumerate() {
            assert_eq!(reconstruct_x(&shares[j], &modulus).value, x.value);
        }
        // Standalone cost: 2 (client-facing) + 1 (echo) = 3 rounds.
        assert_eq!(comm.rounds, 3);
    }

    #[test]
    fn empty_input_accepts_trivially() {
        let (family, modulus, pre_shared) = small_setup();
        let (verdict, shares, u, comm) =
            client_input_share(&[], &pre_shared, &family, &modulus, 92_000);
        assert_eq!(verdict, InputResult::Accept);
        assert!(shares.is_empty());
        assert!(u.is_empty());
        assert_eq!(comm.client_bytes, 0);
    }

    #[test]
    fn psi_check_rejects_tampered_component() {
        let (family, modulus, pre_shared) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..2).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let counter_base = 93_000;

        let mut assembled = assemble_components(xs.len(), &pre_shared, &family, &modulus, counter_base);
        // Tamper one component of one input — simulates a malicious
        // designated sender reporting a wrong value to the client.
        let subset = family.subsets[0];
        let tampered = &assembled[0][&subset] + &Fp::one(&modulus);
        assembled[0].insert(subset, tampered);

        let verdict = check_psi(&assembled, &pre_shared, &family, &modulus, counter_base);
        assert_eq!(verdict, InputResult::Abort);
    }

    #[test]
    fn psi_check_accepts_untampered_assembly() {
        let (family, modulus, pre_shared) = small_setup();
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..2).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let counter_base = 94_000;

        let assembled = assemble_components(xs.len(), &pre_shared, &family, &modulus, counter_base);
        let verdict = check_psi(&assembled, &pre_shared, &family, &modulus, counter_base);
        assert_eq!(verdict, InputResult::Accept);
    }

    #[test]
    fn echo_check_rejects_disagreeing_u() {
        let (family, modulus, _pre_shared) = small_setup();
        let mut rng = rand::thread_rng();
        let u: Vec<Fp> = (0..2).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let mut u_tampered = u.clone();
        u_tampered[0] = &u_tampered[0] + &Fp::one(&modulus);

        let mut per_party_u: Vec<Vec<Fp>> = (0..family.n).map(|_| u.clone()).collect();
        per_party_u[2] = u_tampered; // one server received a different u

        let (verdict, _comm) = verify_echo_standalone(&per_party_u, &family);
        assert_eq!(verdict, InputResult::Abort);
    }

    #[test]
    fn echo_check_accepts_unanimous_u() {
        let (family, modulus, _pre_shared) = small_setup();
        let mut rng = rand::thread_rng();
        let u: Vec<Fp> = (0..2).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let per_party_u: Vec<Vec<Fp>> = (0..family.n).map(|_| u.clone()).collect();

        let (verdict, comm) = verify_echo_standalone(&per_party_u, &family);
        assert_eq!(verdict, InputResult::Accept);
        assert_eq!(comm.rounds, 1);
    }

    /// Each `[r^(j)]` must be consumed by exactly one input — distinct `j`
    /// must give independent `r^(j)` values. Mirrors `vip.rs`'s
    /// `coin_toss` distinctness regression test.
    #[test]
    fn distinct_inputs_get_independent_masks() {
        let (family, modulus, pre_shared) = small_setup();
        let xs = vec![Fp::zero(&modulus), Fp::zero(&modulus)];
        let counter_base = 95_000;

        let assembled = assemble_components(xs.len(), &pre_shared, &family, &modulus, counter_base);
        let u = compute_u(&assembled, &xs, &modulus);
        // x = 0 for both, so u[j] = r^(j) directly.
        assert_ne!(
            u[0].value, u[1].value,
            "r^(0) and r^(1) must be independent — distinct counters must give distinct masks",
        );
    }
}
