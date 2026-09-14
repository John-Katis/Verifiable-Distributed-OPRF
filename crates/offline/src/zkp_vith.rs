//! Approach III-a: VOLEitH-based Dual-Share ZKP (Π_VitH)
//!
//! Proves that a dealer's public correction δ = Π_T m_T - Σ_T a_T
//! is consistent with the privately held (m_T, a_T) pairs.
//!
//! Uses GGM tree expansion for VOLE generation and QuickSilver gate checks.
//!
//! VOLE correlation: q_j = u_j·Δ + v_j
//! Commitment: Commit(w_j) = w_j·Δ + v_j
//! Gate check: B_ℓ = Commit(a^ℓ)·Commit(b^ℓ) - Commit(c^ℓ)·Δ
//! Batched: V = Σ χ^ℓ B_ℓ = A₀ + A₁·Δ
//! Final: V + q₀ = Ã₀ + Ã₁·Δ

use num_bigint::BigUint;
use rand::RngCore;
use std::collections::BTreeMap;
use vdoprf_crypto::ggm::GgmTree;
use vdoprf_crypto::hash::{hash_commitment, hash_field_elements};
use vdoprf_crypto::merkle::MerkleTree;
use vdoprf_crypto::prg::PairwisePrg;
use vdoprf_crypto::transcript::Transcript;
use vdoprf_field::Fp;

/// Parameters for VOLEitH.
#[derive(Clone, Debug)]
pub struct VitHParams {
    pub tau: usize,
    pub kappa: usize,
    pub repetitions: usize,
}

impl VitHParams {
    pub fn new(tau: usize, kappa: usize) -> Self {
        // One repetition has soundness error 2/τ: the QuickSilver check is a
        // degree-2 polynomial identity in Δ ∈ [τ], and a cheating prover can
        // place both of its roots in [τ]. Each repetition therefore gives
        // log2(τ) − 1 bits, and (2/τ)^R ≤ 2^{-κ} needs R ≥ κ / (log2(τ) − 1).
        assert!(tau >= 4, "VitH needs τ ≥ 4 (a repetition has soundness error 2/τ)");
        let bits_per_rep = (tau as f64).log2() - 1.0;
        let repetitions = (kappa as f64 / bits_per_rep).ceil() as usize;
        VitHParams { tau, kappa, repetitions }
    }
}

/// Extended witness for the dual-share circuit C_dual.
/// Committed wire layout: [m_0, a_0, m_1, a_1, ..., m_{N-1}, a_{N-1}, w_1, ..., w_{N-2}]
/// where w_0 = m_0 and w_j = w_{j-1} * m_j. Neither w_0 nor w_{N-1} is a wire:
/// gate 1 takes m_0 directly (a separate w_0 wire would be unconstrained), and
/// the last gate's output is the linear expression Σ_k a_k + δ, which binds δ.
/// Index 0 of the VOLE vectors is reserved as the check wire (not part of the circuit).
#[derive(Clone, Debug)]
pub struct ExtendedWitness {
    pub m_values: Vec<Fp>,
    pub a_values: Vec<Fp>,
    pub running_products: Vec<Fp>,
}

impl ExtendedWitness {
    pub fn new(m_values: Vec<Fp>, a_values: Vec<Fp>, _modulus: &BigUint) -> Self {
        let n = m_values.len();
        assert_eq!(n, a_values.len());
        assert!(n >= 2, "C_dual needs N >= 2: the output constraint lives in the last gate");

        let mut running_products = Vec::with_capacity(n);
        running_products.push(m_values[0].clone());
        for i in 1..n {
            running_products.push(&running_products[i - 1] * &m_values[i]);
        }

        ExtendedWitness { m_values, a_values, running_products }
    }

    /// Number of committed wires (excluding the check wire): 2N inputs plus
    /// the N−2 intermediate running products w_1..w_{N-2}, i.e. 3N − 2.
    pub fn num_wires(&self) -> usize {
        3 * self.m_values.len() - 2
    }

    /// Flatten witness into wire vector.
    /// Layout: [m_0, a_0, m_1, a_1, ..., w_1, ..., w_{N-2}]
    pub fn flatten(&self) -> Vec<Fp> {
        let n = self.m_values.len();
        let mut w = Vec::with_capacity(self.num_wires());
        for i in 0..n {
            w.push(self.m_values[i].clone());
            w.push(self.a_values[i].clone());
        }
        for rp in &self.running_products[1..n - 1] {
            w.push(rp.clone());
        }
        w
    }

    /// Multiplication gates as (left, right, output) wire indices; see `dual_gates`.
    fn gates(&self) -> Vec<(usize, usize, Option<usize>)> {
        dual_gates(self.m_values.len())
    }

    pub fn compute_delta(&self, modulus: &BigUint) -> Fp {
        let product = self.running_products.last().unwrap();
        let mut sum = Fp::zero(modulus);
        for a in &self.a_values {
            sum = &sum + a;
        }
        product - &sum
    }
}

/// The N−1 multiplication gates of C_dual over the committed wire layout
/// (m_k at 2k, a_k at 2k+1, w_j at 2N + j − 1 for j = 1..N−2).
/// Gate j (j = 1..N−1) checks w_{j-1} · m_j = w_j, with w_0 = m_0 read from
/// wire 0. The last gate's output is `None`: the linear expression
/// Σ_k a_k + δ, committed as Σ_k Commit(a_k) + δ·Δ with VOLE mask Σ_k v_{a_k}.
fn dual_gates(n: usize) -> Vec<(usize, usize, Option<usize>)> {
    (1..n)
        .map(|j| {
            let left = if j == 1 { 0 } else { 2 * n + j - 2 };
            let out = if j == n - 1 { None } else { Some(2 * n + j - 1) };
            (left, 2 * j, out)
        })
        .collect()
}

/// VOLE shares for one repetition.
struct VoleShares {
    /// u_j = Σ_i s_j^(i) for j = 0..L (wire 0 is check wire)
    u: Vec<Fp>,
    /// v_j = -Σ_i i·s_j^(i)
    v: Vec<Fp>,
}

/// One GGM leaf's share vector `(s_0^(i), ..., s_L^(i))` — index 0 is the
/// check wire. Used both for VOLE aggregation and as the leaves of that
/// leaf's own two-level-commitment sub-Merkle-tree.
fn leaf_shares(leaf_seed: [u8; 16], num_wires_with_check: usize, modulus: &BigUint) -> Vec<Fp> {
    let prg = PairwisePrg::new(leaf_seed);
    (0..num_wires_with_check)
        .map(|j| prg.generate(j as u64, modulus))
        .collect()
}

/// Domain-separated commitment to one field-element share at a fixed wire
/// position, used as a two-level-commitment sub-tree leaf. Without the
/// positional salt, `BigUint::to_bytes_be`'s variable-width encoding would
/// let the same numeric share value at two different wire positions hash
/// identically.
fn share_commitment(value: &Fp, position: usize) -> [u8; 32] {
    hash_commitment(&value.value.to_bytes_be(), &(position as u64).to_be_bytes())
}

/// Build the sub-Merkle-tree over one GGM leaf's share vector — this leaf's
/// root is `h_i` in the paper's two-level GGM commitment (Section 4.2.2):
/// "for each GGM leaf i, build a sub-Merkle-tree over the expanded share
/// vector s^(i), giving h_i = MT(s^(i))."
fn leaf_commitment_tree(shares: &[Fp]) -> MerkleTree {
    let leaves: Vec<[u8; 32]> = shares
        .iter()
        .enumerate()
        .map(|(j, s)| share_commitment(s, j))
        .collect();
    MerkleTree::new(leaves)
}

/// Aggregate VOLE shares from every leaf's already-computed share vector.
/// `u_j = Σ_i s_j^(i)`, `v_j = -Σ_i i·s_j^(i)`.
fn aggregate_vole_shares(
    all_leaf_shares: &[Vec<Fp>],
    num_wires_with_check: usize, // L + 1
    modulus: &BigUint,
) -> VoleShares {
    let mut u = vec![Fp::zero(modulus); num_wires_with_check];
    let mut v = vec![Fp::zero(modulus); num_wires_with_check];

    for (i, shares) in all_leaf_shares.iter().enumerate() {
        let i_fp = Fp::new(BigUint::from(i), modulus);
        for j in 0..num_wires_with_check {
            u[j] = &u[j] + &shares[j];
            v[j] = &v[j] - &(&i_fp * &shares[j]);
        }
    }

    VoleShares { u, v }
}

/// Reconstruct VOLE tags q_j for the verifier (who knows all leaves except Δ).
/// q_j = Σ_{i≠Δ} s_j^(i) · (Δ - i)
fn reconstruct_vole_tags(
    revealed_seeds: &[(usize, [u8; 16])],
    hidden_idx: usize,
    num_wires_with_check: usize,
    modulus: &BigUint,
) -> Vec<Fp> {
    let delta_fp = Fp::new(BigUint::from(hidden_idx), modulus);
    let mut q = vec![Fp::zero(modulus); num_wires_with_check];

    for &(i, seed) in revealed_seeds {
        let prg = PairwisePrg::new(seed);
        let i_fp = Fp::new(BigUint::from(i), modulus);
        let factor = &delta_fp - &i_fp; // (Δ - i)
        for j in 0..num_wires_with_check {
            let s_j_i = prg.generate(j as u64, modulus);
            q[j] = &q[j] + &(&s_j_i * &factor);
        }
    }

    q
}

/// A VOLEitH proof.
#[derive(Clone, Debug)]
pub struct VitHProof {
    pub commitment: [u8; 32],
    /// Masked witness: w̃_j = w_j + u_j for each circuit wire (per repetition).
    pub masked_witnesses: Vec<Vec<Fp>>,  // [rep][wire]
    /// Masked check values per repetition: (Ã₀, Ã₁)
    pub check_values: Vec<(Fp, Fp)>,     // [rep] = (a0_tilde, a1_tilde)
    /// GGM co-paths (sibling *seeds*) for each repetition — lets the
    /// verifier reconstruct every non-hidden leaf's seed.
    pub copaths: Vec<Vec<[u8; 16]>>,
    /// Hidden leaf indices Δ per repetition.
    pub hidden_indices: Vec<usize>,
    /// Two-level GGM commitment outer root `rt` per repetition — Merkle
    /// root over the τ per-leaf sub-tree roots `{h_i}`. Committed into the
    /// transcript *before* Δ is drawn, closing the gap where the GGM root
    /// seed used to be derived from (and hence fully predictable from)
    /// public transcript data.
    pub seed_roots: Vec<[u8; 32]>,
    /// The hidden leaf's own sub-tree root `h_Δ` per repetition — the
    /// verifier cannot derive this itself (it never learns `sd_Δ`), so the
    /// prover must reveal it directly.
    pub hidden_leaf_roots: Vec<[u8; 32]>,
    /// Per-repetition, per-wire-position authentication paths within the
    /// hidden leaf's own sub-tree (`hidden_sub_tree_paths[rep][wire_idx]`,
    /// where `wire_idx = position + 1` accounts for the check-wire offset).
    /// This is what lets each verifier check the committed shares at the
    /// witness positions it knows locally against `h_Δ` — the paper's
    /// "input binding via two-level GGM commitment" (Section 4.2.2).
    pub hidden_sub_tree_paths: Vec<Vec<Vec<[u8; 32]>>>,
}

impl VitHProof {
    /// Wire-byte size as placed on the network by the broadcast accounting in
    /// `approach_iii::gen_zkp` (VitH branch). Matches the exact byte stream
    /// dealers push out: commitment, then per-repetition masked witnesses,
    /// (Ã₀, Ã₁), GGM copath seeds, the two-level commitment roots, and the
    /// hidden leaf's sub-tree authentication paths. `hidden_indices` is
    /// re-derived from the Fiat-Shamir transcript and not sent on the wire.
    pub fn wire_bytes(&self, feb: usize) -> usize {
        let mut bytes = self.commitment.len();
        for rep in &self.masked_witnesses {
            bytes += rep.len() * feb;
        }
        bytes += self.check_values.len() * 2 * feb;
        for copath in &self.copaths {
            bytes += copath.len() * 16;
        }
        bytes += self.seed_roots.len() * 32;
        bytes += self.hidden_leaf_roots.len() * 32;
        for rep_paths in &self.hidden_sub_tree_paths {
            for path in rep_paths {
                bytes += path.len() * 32;
            }
        }
        bytes
    }
}

/// Generate a VOLEitH proof.
///
/// Challenge order, over one Fiat–Shamir transcript for all repetitions:
///   1. every repetition's GGM commitment `rt` and masked witness;
///   2. χ_ρ for every ρ, derived from (1);
///   3. every repetition's QuickSilver check values (Ã₀, Ã₁);
///   4. Δ_ρ for every ρ, derived from (1)+(3); only then are co-paths opened.
/// Deriving Δ after (Ã₀, Ã₁) is essential (given Δ, anyone can satisfy the
/// check), and deriving each family jointly prevents grinding one repetition
/// at a time.
pub fn vith_prove(
    witness: &ExtendedWitness,
    delta: &Fp,
    params: &VitHParams,
    modulus: &BigUint,
) -> VitHProof {
    let n = witness.m_values.len();
    let l = witness.num_wires(); // committed circuit wires
    let l_check = l + 1;        // +1 for check wire (index 0)
    let w_flat = witness.flatten();
    let gates = witness.gates();

    // Commit to witness
    let commitment = hash_field_elements(&w_flat);

    // Fiat-Shamir transcript
    let mut transcript = Transcript::new(b"VitH");
    transcript.append_commitment(&commitment);
    transcript.append_field_element(delta);

    // (1) Per repetition: GGM tree, two-level commitment, masked witness.
    struct Rep {
        tree: GgmTree,
        all_leaf_shares: Vec<Vec<Fp>>,
        leaf_roots: Vec<[u8; 32]>,
        rt: [u8; 32],
        vole: VoleShares,
        masked: Vec<Fp>,
    }
    let reps: Vec<Rep> = (0..params.repetitions)
        .map(|_| {
            // GGM root seed: the prover's own secret randomness (Protocol 16, line 1).
            let mut root_seed = [0u8; 16];
            rand::thread_rng().fill_bytes(&mut root_seed);
            let tree = GgmTree::expand(root_seed, params.tau);
            let all_leaf_shares: Vec<Vec<Fp>> = tree
                .leaf_seeds
                .iter()
                .map(|seed| leaf_shares(*seed, l_check, modulus))
                .collect();
            // Two-level GGM commitment: a sub-Merkle root h_i per leaf, outer root rt.
            let leaf_roots: Vec<[u8; 32]> = all_leaf_shares
                .iter()
                .map(|shares| leaf_commitment_tree(shares).root())
                .collect();
            let rt = MerkleTree::new(leaf_roots.clone()).root();
            let vole = aggregate_vole_shares(&all_leaf_shares, l_check, modulus);
            // Masked witness: w̃_j = w_j + u_{j+1} (shift by 1: index 0 is the check wire)
            let masked: Vec<Fp> = (0..l).map(|j| &w_flat[j] + &vole.u[j + 1]).collect();
            Rep { tree, all_leaf_shares, leaf_roots, rt, vole, masked }
        })
        .collect();
    for (rep, r) in reps.iter().enumerate() {
        transcript.append_bytes(&(rep as u64).to_be_bytes());
        transcript.append_commitment(&r.rt);
        for mw in &r.masked {
            transcript.append_field_element(mw);
        }
    }

    // (2) Batching challenges χ_ρ.
    transcript.append_bytes(b"chi");
    let chis = transcript.challenge_vec(params.repetitions, modulus);

    // (3) QuickSilver check values. With Commit(w) = w·Δ − v, gate ℓ gives
    //   B_ℓ = (a·b − c)·Δ² + (v_c − b·v_a − a·v_b)·Δ + v_a·v_b,
    // so A₀ = Σ χ^ℓ v_a v_b and A₁ = Σ χ^ℓ (v_c − b·v_a − a·v_b). The last
    // gate's output is Σ_k a_k + δ, whose VOLE mask is Σ_k v_{a_k} (δ is a
    // public constant with mask 0).
    let check_values: Vec<(Fp, Fp)> = reps
        .iter()
        .zip(&chis)
        .map(|(r, chi)| {
            let v_out = (0..n).fold(Fp::zero(modulus), |acc, k| &acc + &r.vole.v[2 * k + 2]);
            let mut a0 = Fp::zero(modulus);
            let mut a1 = Fp::zero(modulus);
            let mut chi_power = Fp::one(modulus);
            for &(a_idx, b_idx, c_idx) in &gates {
                let a_val = &w_flat[a_idx];
                let b_val = &w_flat[b_idx];
                let v_a = &r.vole.v[a_idx + 1];
                let v_b = &r.vole.v[b_idx + 1];
                let v_c = match c_idx {
                    Some(c) => &r.vole.v[c + 1],
                    None => &v_out,
                };
                a0 = &a0 + &(&chi_power * &(v_a * v_b));
                let a1_term = &(v_c - &(b_val * v_a)) - &(a_val * v_b);
                a1 = &a1 + &(&chi_power * &a1_term);
                chi_power = &chi_power * chi;
            }
            // Masked check values: Ã₀ = A₀ + v₀, Ã₁ = A₁ + u₀ (check wire at index 0)
            (&a0 + &r.vole.v[0], &a1 + &r.vole.u[0])
        })
        .collect();
    for (a0_tilde, a1_tilde) in &check_values {
        transcript.append_field_element(a0_tilde);
        transcript.append_field_element(a1_tilde);
    }

    // (4) Hidden leaves Δ_ρ, derived only now; then open.
    transcript.append_bytes(b"hidden_leaf");
    let hidden_indices = derive_hidden_indices(&transcript, params);

    let mut copaths = Vec::with_capacity(params.repetitions);
    let mut masked_witnesses = Vec::with_capacity(params.repetitions);
    let mut seed_roots = Vec::with_capacity(params.repetitions);
    let mut hidden_leaf_roots = Vec::with_capacity(params.repetitions);
    let mut hidden_sub_tree_paths = Vec::with_capacity(params.repetitions);
    for (r, &hidden) in reps.into_iter().zip(&hidden_indices) {
        // GGM PRG co-path: lets the verifier reconstruct every non-hidden leaf's seed.
        copaths.push(r.tree.copath(hidden));
        // The hidden leaf's sub-tree root and per-position authentication
        // paths, so each verifier can check the positions it holds against h_Δ.
        let hidden_tree = leaf_commitment_tree(&r.all_leaf_shares[hidden]);
        hidden_sub_tree_paths.push(
            (0..l_check)
                .map(|pos| hidden_tree.authentication_path(pos))
                .collect(),
        );
        hidden_leaf_roots.push(r.leaf_roots[hidden]);
        seed_roots.push(r.rt);
        masked_witnesses.push(r.masked);
    }

    VitHProof {
        commitment,
        masked_witnesses,
        check_values,
        copaths,
        hidden_indices,
        seed_roots,
        hidden_leaf_roots,
        hidden_sub_tree_paths,
    }
}

/// Δ_ρ for every repetition, from the transcript after all check values.
fn derive_hidden_indices(transcript: &Transcript, params: &VitHParams) -> Vec<usize> {
    (0..params.repetitions)
        .map(|rep| {
            let mut t = transcript.clone();
            t.append_bytes(&(rep as u64).to_be_bytes());
            t.challenge_index(params.tau)
        })
        .collect()
}

/// Verify a VOLEitH proof.
///
/// `local_shares` maps witness *position* (0-indexed into
/// `ExtendedWitness::flatten()`, i.e. unshifted — matching the caller's own
/// view of which `(m_T, a_T)` values it independently knows) to the value
/// this verifier holds locally for that position. For every position
/// present, the dual-share consistency check (Section 4.2.2) recovers the
/// hidden leaf's share at that position from the masked witness and checks
/// it against the two-level GGM commitment — catching a dealer whose
/// broadcast proof disagrees with what this verifier independently knows,
/// even when the QuickSilver check alone would still pass.
pub fn vith_verify(
    proof: &VitHProof,
    delta: &Fp,
    local_shares: &BTreeMap<usize, Fp>,
    params: &VitHParams,
    modulus: &BigUint,
) -> bool {
    let reps = params.repetitions;
    if proof.masked_witnesses.len() != reps
        || proof.check_values.len() != reps
        || proof.copaths.len() != reps
        || proof.hidden_indices.len() != reps
        || proof.seed_roots.len() != reps
        || proof.hidden_leaf_roots.len() != reps
        || proof.hidden_sub_tree_paths.len() != reps
    {
        return false;
    }

    // Replay the transcript in the prover's order (see `vith_prove`):
    // commitments and masked witnesses → χ_ρ → check values → Δ_ρ.
    let mut transcript = Transcript::new(b"VitH");
    transcript.append_commitment(&proof.commitment);
    transcript.append_field_element(delta);
    for rep in 0..reps {
        transcript.append_bytes(&(rep as u64).to_be_bytes());
        transcript.append_commitment(&proof.seed_roots[rep]);
        for mw in &proof.masked_witnesses[rep] {
            transcript.append_field_element(mw);
        }
    }
    transcript.append_bytes(b"chi");
    let chis = transcript.challenge_vec(reps, modulus);
    for (a0_tilde, a1_tilde) in &proof.check_values {
        transcript.append_field_element(a0_tilde);
        transcript.append_field_element(a1_tilde);
    }
    transcript.append_bytes(b"hidden_leaf");
    let expected_hidden = derive_hidden_indices(&transcript, params);
    if proof.hidden_indices != expected_hidden {
        return false;
    }

    for rep in 0..reps {
        let masked = &proof.masked_witnesses[rep];
        let l = masked.len();
        // Committed layout has 3N − 2 wires with N ≥ 2 (see `ExtendedWitness`).
        if l < 4 || (l + 2) % 3 != 0 {
            return false;
        }
        let n_subsets = (l + 2) / 3;
        let l_check = l + 1;
        let hidden = expected_hidden[rep];
        let chi = &chis[rep];

        // Reconstruct revealed leaf seeds from co-path
        let revealed = GgmTree::reconstruct_except(
            &proof.copaths[rep],
            hidden,
            params.tau,
        );
        if revealed.len() != params.tau - 1 {
            return false;
        }

        // Two-level GGM commitment check: the τ-1 revealed leaves are now
        // self-derivable (the verifier just recovered their seeds), so it
        // can recompute their sub-tree roots `h_i` itself; the hidden
        // leaf's root `h_Δ` is taken from the proof (the verifier cannot
        // derive it without `sd_Δ`). Rebuilding the outer tree from this
        // full set of τ roots and checking it equals the committed `rt`
        // binds `rt` to the actual leaf structure — without this, `rt`
        // (and the whole co-path reveal) is unauthenticated against
        // anything (findings Offline #1/#2).
        let mut leaf_roots = vec![[0u8; 32]; params.tau];
        for &(i, seed) in &revealed {
            let shares = leaf_shares(seed, l_check, modulus);
            leaf_roots[i] = leaf_commitment_tree(&shares).root();
        }
        leaf_roots[hidden] = proof.hidden_leaf_roots[rep];
        if MerkleTree::new(leaf_roots).root() != proof.seed_roots[rep] {
            return false;
        }

        // Dual-share consistency: for every witness position this verifier
        // holds locally, recover the hidden leaf's share at that position
        // from the masked witness (u_k = w̃_k - w_k^local, per Protocol 16
        // steps 4-9) and check it against `h_Δ` via the sub-tree
        // authentication path. `s_k^(Δ) = u_k - Σ_{i≠Δ} s_k^(i)` — plain
        // sum, matching u_j's own definition (no (Δ-i) weighting, unlike
        // the VOLE tags q_j computed below).
        for (&position, local_value) in local_shares.iter() {
            if position >= l {
                return false;
            }
            let wire_idx = position + 1; // +1 for the check-wire offset
            let mut s_k_hidden = &masked[position] - local_value;
            for &(i, seed) in &revealed {
                let prg = PairwisePrg::new(seed);
                let _ = i;
                s_k_hidden = &s_k_hidden - &prg.generate(wire_idx as u64, modulus);
            }
            let leaf = share_commitment(&s_k_hidden, wire_idx);
            let path = match proof.hidden_sub_tree_paths[rep].get(wire_idx) {
                Some(p) => p,
                None => return false,
            };
            if !MerkleTree::verify_path(
                &proof.hidden_leaf_roots[rep],
                &leaf,
                wire_idx,
                l_check,
                path,
            ) {
                return false;
            }
        }

        // Reconstruct VOLE tags: q_j = Σ_{i≠Δ} s_j^(i) · (Δ - i)
        let q = reconstruct_vole_tags(&revealed, hidden, l_check, modulus);
        let delta_fp = Fp::new(BigUint::from(hidden), modulus);
        // Commit(w_j) = w̃_j · Δ − q_{j+1} = w_j · Δ − v_j
        let commit = |j: usize| &(&masked[j] * &delta_fp) - &q[j + 1];
        // Output of the last gate: Commit(Σ_k a_k + δ) = Σ_k Commit(a_k) + δ·Δ
        let commit_out = (0..n_subsets).fold(delta * &delta_fp, |acc, k| &acc + &commit(2 * k + 1));

        // QuickSilver: V = Σ χ^ℓ B_ℓ with B_ℓ = Commit(a)·Commit(b) − Commit(c)·Δ
        let mut v_check = Fp::zero(modulus);
        let mut chi_power = Fp::one(modulus);
        for (a_idx, b_idx, c_idx) in dual_gates(n_subsets) {
            let commit_c = match c_idx {
                Some(c) => commit(c),
                None => commit_out.clone(),
            };
            let b_ell = &(&commit(a_idx) * &commit(b_idx)) - &(&commit_c * &delta_fp);
            v_check = &v_check + &(&chi_power * &b_ell);
            chi_power = &chi_power * chi;
        }

        // Check: V + q₀ = Ã₀ + Ã₁·Δ
        let (ref a0_tilde, ref a1_tilde) = proof.check_values[rep];
        let lhs = &v_check + &q[0];
        let rhs = a0_tilde + &(a1_tilde * &delta_fp);
        if lhs != rhs {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_witness() {
        let p = BigUint::from(113u32);
        let m = vec![
            Fp::new(BigUint::from(3u32), &p),
            Fp::new(BigUint::from(5u32), &p),
            Fp::new(BigUint::from(7u32), &p),
        ];
        let a = vec![
            Fp::new(BigUint::from(10u32), &p),
            Fp::new(BigUint::from(20u32), &p),
            Fp::new(BigUint::from(30u32), &p),
        ];
        let w = ExtendedWitness::new(m, a, &p);

        assert_eq!(w.running_products[0].value, BigUint::from(3u32));
        assert_eq!(w.running_products[1].value, BigUint::from(15u32));
        assert_eq!(w.running_products[2].value, BigUint::from(105u32) % &p);

        let delta = w.compute_delta(&p);
        assert_eq!(delta.value, BigUint::from(45u32));

        assert_eq!(w.num_wires(), 7); // 2N inputs + the N−2 = 1 intermediate product

        // Check gates
        let gates = w.gates();
        assert_eq!(gates.len(), 2); // N-1 = 2 gates
        // Gate 0: m[0] * m[1] = rp[1] (wire 6) → (0, 2, Some(6))
        assert_eq!(gates[0], (0, 2, Some(6)));
        // Gate 1: rp[1] * m[2] = Σ a + δ (linear output) → (6, 4, None)
        assert_eq!(gates[1], (6, 4, None));
    }

    #[test]
    fn test_vith_prove_verify_honest() {
        let p = BigUint::from(113u32);
        let m = vec![
            Fp::new(BigUint::from(3u32), &p),
            Fp::new(BigUint::from(5u32), &p),
            Fp::new(BigUint::from(7u32), &p),
        ];
        let a = vec![
            Fp::new(BigUint::from(10u32), &p),
            Fp::new(BigUint::from(20u32), &p),
            Fp::new(BigUint::from(30u32), &p),
        ];
        let w = ExtendedWitness::new(m, a, &p);
        let delta = w.compute_delta(&p);

        let params = VitHParams::new(4, 8);
        let proof = vith_prove(&w, &delta, &params, &p);

        let local_shares = BTreeMap::new();
        assert!(vith_verify(&proof, &delta, &local_shares, &params, &p));
    }

    /// Regression guard for the GGM-seed / two-level-commitment fix
    /// (findings Offline #1/#2): before the fix, `rt` (the GGM commitment
    /// root) was never checked against anything — a forged `h_Δ` unrelated
    /// to the real committed tree would still pass, since the old
    /// `vith_verify` didn't touch `rt`/`h_Δ` at all. `hidden_indices`/`χ`
    /// re-derivation (unchanged, pre-existing check) still passes here
    /// because `rt` and the masked witness — the only transcript inputs —
    /// are untouched; only the two-level commitment check (new) can catch
    /// this tamper.
    #[test]
    fn test_vith_rejects_forged_hidden_leaf_root() {
        let p = BigUint::from(113u32);
        let m = vec![
            Fp::new(BigUint::from(3u32), &p),
            Fp::new(BigUint::from(5u32), &p),
            Fp::new(BigUint::from(7u32), &p),
        ];
        let a = vec![
            Fp::new(BigUint::from(10u32), &p),
            Fp::new(BigUint::from(20u32), &p),
            Fp::new(BigUint::from(30u32), &p),
        ];
        let w = ExtendedWitness::new(m, a, &p);
        let delta = w.compute_delta(&p);

        let params = VitHParams::new(4, 8);
        let mut proof = vith_prove(&w, &delta, &params, &p);
        proof.hidden_leaf_roots[0] = [0xABu8; 32];

        let local_shares = BTreeMap::new();
        assert!(!vith_verify(&proof, &delta, &local_shares, &params, &p));
    }

    /// Dual-share consistency (Section 4.2.2, "our contribution"): a
    /// verifier whose own locally-known witness value at a position agrees
    /// with what the dealer actually used must still accept.
    #[test]
    fn test_vith_dual_share_consistency_accepts_correct_local_share() {
        let p = BigUint::from(113u32);
        let m = vec![
            Fp::new(BigUint::from(3u32), &p),
            Fp::new(BigUint::from(5u32), &p),
            Fp::new(BigUint::from(7u32), &p),
        ];
        let a = vec![
            Fp::new(BigUint::from(10u32), &p),
            Fp::new(BigUint::from(20u32), &p),
            Fp::new(BigUint::from(30u32), &p),
        ];
        let w = ExtendedWitness::new(m.clone(), a.clone(), &p);
        let delta = w.compute_delta(&p);

        let params = VitHParams::new(4, 8);
        let proof = vith_prove(&w, &delta, &params, &p);

        // Position 0 = m_values[0], position 1 = a_values[0] (flatten layout).
        let mut local_shares = BTreeMap::new();
        local_shares.insert(0, m[0].clone());
        local_shares.insert(1, a[0].clone());
        assert!(vith_verify(&proof, &delta, &local_shares, &params, &p));
    }

    /// Dual-share consistency must reject when the verifier's own
    /// locally-known witness value disagrees with what the dealer actually
    /// used — this is the exact gap finding Offline #2 flagged as entirely
    /// missing (`_local_shares` was unused dead code).
    #[test]
    fn test_vith_dual_share_consistency_rejects_disagreeing_local_share() {
        let p = BigUint::from(113u32);
        let m = vec![
            Fp::new(BigUint::from(3u32), &p),
            Fp::new(BigUint::from(5u32), &p),
            Fp::new(BigUint::from(7u32), &p),
        ];
        let a = vec![
            Fp::new(BigUint::from(10u32), &p),
            Fp::new(BigUint::from(20u32), &p),
            Fp::new(BigUint::from(30u32), &p),
        ];
        let w = ExtendedWitness::new(m.clone(), a.clone(), &p);
        let delta = w.compute_delta(&p);

        let params = VitHParams::new(4, 8);
        let proof = vith_prove(&w, &delta, &params, &p);

        // Verifier's own m_0 disagrees with the dealer's actual m_values[0].
        let mut local_shares = BTreeMap::new();
        local_shares.insert(0, Fp::new(BigUint::from(99u32), &p));
        local_shares.insert(1, a[0].clone());
        assert!(!vith_verify(&proof, &delta, &local_shares, &params, &p));
    }

    #[test]
    fn test_vith_rejects_wrong_witness() {
        let p = BigUint::from(113u32);
        let m = vec![
            Fp::new(BigUint::from(3u32), &p),
            Fp::new(BigUint::from(5u32), &p),
            Fp::new(BigUint::from(7u32), &p),
        ];
        let a = vec![
            Fp::new(BigUint::from(10u32), &p),
            Fp::new(BigUint::from(20u32), &p),
            Fp::new(BigUint::from(30u32), &p),
        ];
        // Honest witness
        let w_honest = ExtendedWitness::new(m.clone(), a.clone(), &p);
        let delta = w_honest.compute_delta(&p);

        // Tampered witness: wrong intermediate running product (rp[2] is no
        // longer a wire: the last gate outputs Σ a + δ instead)
        let mut w_bad = w_honest.clone();
        w_bad.running_products[1] = Fp::new(BigUint::from(99u32), &p); // should be 15

        let params = VitHParams::new(4, 8);
        let proof = vith_prove(&w_bad, &delta, &params, &p);

        let local_shares = BTreeMap::new();
        // Should fail because m[0]*m[1] != rp[1] and rp[1]*m[2] != Σ a + δ
        assert!(!vith_verify(&proof, &delta, &local_shares, &params, &p));
    }

    #[test]
    fn test_vith_larger_params() {
        let p = BigUint::from(113u32);
        let m = vec![
            Fp::new(BigUint::from(2u32), &p),
            Fp::new(BigUint::from(3u32), &p),
            Fp::new(BigUint::from(4u32), &p),
        ];
        let a = vec![
            Fp::new(BigUint::from(1u32), &p),
            Fp::new(BigUint::from(1u32), &p),
            Fp::new(BigUint::from(1u32), &p),
        ];
        let w = ExtendedWitness::new(m, a, &p);
        let delta = w.compute_delta(&p);
        // product = 2*3*4 = 24, sum = 3, delta = 21
        assert_eq!(delta.value, BigUint::from(21u32));

        // Test with realistic-ish params
        let params = VitHParams::new(16, 40);
        let proof = vith_prove(&w, &delta, &params, &p);

        let local_shares = BTreeMap::new();
        assert!(vith_verify(&proof, &delta, &local_shares, &params, &p));
    }

    fn random_witness(n: usize, p: &BigUint) -> ExtendedWitness {
        let mut rng = rand::thread_rng();
        let m = (0..n).map(|_| Fp::random(p, &mut rng)).collect();
        let a = (0..n).map(|_| Fp::random(p, &mut rng)).collect();
        ExtendedWitness::new(m, a, p)
    }

    /// Every input-wire position, as the union of honest verifiers holds them.
    fn input_shares(w: &ExtendedWitness) -> BTreeMap<usize, Fp> {
        w.flatten().into_iter().take(2 * w.m_values.len()).enumerate().collect()
    }

    /// Z1: δ is bound to the witness through the last gate.
    #[test]
    fn test_vith_rejects_wrong_delta() {
        let p = BigUint::from(2305843009213693951u64); // 2^61 − 1
        let w = random_witness(10, &p);
        let delta = w.compute_delta(&p);
        let params = VitHParams::new(16, 40);
        let local = input_shares(&w);
        assert!(vith_verify(&vith_prove(&w, &delta, &params, &p), &delta, &local, &params, &p));
        let wrong = &delta + &Fp::one(&p);
        let proof = vith_prove(&w, &wrong, &params, &p);
        assert!(!vith_verify(&proof, &wrong, &local, &params, &p));
    }

    /// Z1: gate 1 reads m_0 directly, so the product chain cannot start from a
    /// free value chosen to hit a false δ.
    #[test]
    fn test_vith_rejects_product_chain_not_anchored_at_m0() {
        let p = BigUint::from(2305843009213693951u64);
        let w = random_witness(3, &p);
        let wrong = &w.compute_delta(&p) + &Fp::one(&p);
        // Choose w_1 so that the last gate w_1 · m_2 = Σ a + wrong holds.
        let sum_a = w.a_values.iter().fold(Fp::zero(&p), |acc, a| &acc + a);
        let mut bad = w.clone();
        bad.running_products[1] = &(&sum_a + &wrong) * &w.m_values[2].inv().unwrap();
        let params = VitHParams::new(16, 40);
        let proof = vith_prove(&bad, &wrong, &params, &p);
        assert!(!vith_verify(&proof, &wrong, &input_shares(&w), &params, &p));
    }

    /// Z2: (Ã₀, Ã₁) enter the hash that yields Δ, so recomputing them from the
    /// published Δ, which forged a false witness before the fix, now fails.
    #[test]
    fn test_vith_rejects_check_values_forged_from_published_delta() {
        let p = BigUint::from(2305843009213693951u64);
        let w = random_witness(10, &p);
        let delta = w.compute_delta(&p);
        let mut bad = w.clone();
        bad.running_products[4] = &bad.running_products[4] + &Fp::one(&p);
        let params = VitHParams::new(16, 40);
        let local = input_shares(&w);
        let mut proof = vith_prove(&bad, &delta, &params, &p);
        assert!(!vith_verify(&proof, &delta, &local, &params, &p));

        // Recompute V + q₀ from public proof data under the published Δ_ρ and
        // set (Ã₀, Ã₁) = (V + q₀, 0).
        let mut tr = Transcript::new(b"VitH");
        tr.append_commitment(&proof.commitment);
        tr.append_field_element(&delta);
        for rep in 0..params.repetitions {
            tr.append_bytes(&(rep as u64).to_be_bytes());
            tr.append_commitment(&proof.seed_roots[rep]);
            for mw in &proof.masked_witnesses[rep] {
                tr.append_field_element(mw);
            }
        }
        tr.append_bytes(b"chi");
        let chis = tr.challenge_vec(params.repetitions, &p);
        for rep in 0..params.repetitions {
            let masked = proof.masked_witnesses[rep].clone();
            let n = (masked.len() + 2) / 3;
            let hidden = proof.hidden_indices[rep];
            let revealed = GgmTree::reconstruct_except(&proof.copaths[rep], hidden, params.tau);
            let q = reconstruct_vole_tags(&revealed, hidden, masked.len() + 1, &p);
            let d = Fp::new(BigUint::from(hidden), &p);
            let commit = |j: usize| &(&masked[j] * &d) - &q[j + 1];
            let out = (0..n).fold(&delta * &d, |acc, k| &acc + &commit(2 * k + 1));
            let (mut v, mut cp) = (Fp::zero(&p), Fp::one(&p));
            for (ia, ib, ic) in dual_gates(n) {
                let cc = ic.map_or(out.clone(), |c| commit(c));
                v = &v + &(&cp * &(&(&commit(ia) * &commit(ib)) - &(&cc * &d)));
                cp = &cp * &chis[rep];
            }
            proof.check_values[rep] = (&v + &q[0], Fp::zero(&p));
        }
        assert!(!vith_verify(&proof, &delta, &local, &params, &p));
    }

    /// Z3: a repetition has soundness error 2/τ, so R·(log2 τ − 1) ≥ κ.
    #[test]
    fn test_vith_repetitions_cover_two_roots() {
        assert_eq!(VitHParams::new(16, 40).repetitions, 14);
        assert_eq!(VitHParams::new(4, 8).repetitions, 8);
        for (tau, kappa) in [(4usize, 40usize), (8, 40), (16, 40), (32, 40), (256, 128)] {
            let r = VitHParams::new(tau, kappa).repetitions as f64;
            assert!(r * ((tau as f64).log2() - 1.0) >= kappa as f64);
        }
    }
}
