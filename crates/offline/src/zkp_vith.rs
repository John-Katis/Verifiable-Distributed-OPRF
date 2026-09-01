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
        let log_tau = (tau as f64).log2().ceil() as usize;
        let repetitions = (kappa + log_tau - 1) / log_tau;
        VitHParams { tau, kappa, repetitions }
    }
}

/// Extended witness for the dual-share circuit C_dual.
/// Wire layout: [m_0, a_0, m_1, a_1, ..., m_{N-1}, a_{N-1}, w_0, w_1, ..., w_{N-1}]
/// where w_0 = m_0, w_j = w_{j-1} * m_j for j >= 1.
/// Wire 0 is reserved as the check wire (not part of the circuit).
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

        let mut running_products = Vec::with_capacity(n);
        running_products.push(m_values[0].clone());
        for i in 1..n {
            running_products.push(&running_products[i - 1] * &m_values[i]);
        }

        ExtendedWitness { m_values, a_values, running_products }
    }

    /// Total number of circuit wires (excluding check wire): 2N + N = 3N.
    pub fn num_wires(&self) -> usize {
        2 * self.m_values.len() + self.running_products.len()
    }

    /// Flatten witness into wire vector.
    /// Layout: [m_0, a_0, m_1, a_1, ..., w_0, w_1, ...]
    pub fn flatten(&self) -> Vec<Fp> {
        let n = self.m_values.len();
        let mut w = Vec::with_capacity(self.num_wires());
        for i in 0..n {
            w.push(self.m_values[i].clone());
            w.push(self.a_values[i].clone());
        }
        for rp in &self.running_products {
            w.push(rp.clone());
        }
        w
    }

    /// Get the multiplication gates as (a_wire_idx, b_wire_idx, c_wire_idx).
    /// Gate j (j=1..N-1): running_products[j] = running_products[j-1] * m_values[j]
    /// Wire indices: m_j is at 2*j, running_products[j] is at 2*N + j
    fn gates(&self) -> Vec<(usize, usize, usize)> {
        let n = self.m_values.len();
        let mut gates = Vec::new();
        for j in 1..n {
            let a_idx = 2 * n + (j - 1); // running_products[j-1]
            let b_idx = 2 * j;            // m_values[j]
            let c_idx = 2 * n + j;        // running_products[j]
            gates.push((a_idx, b_idx, c_idx));
        }
        gates
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
pub fn vith_prove(
    witness: &ExtendedWitness,
    delta: &Fp,
    params: &VitHParams,
    modulus: &BigUint,
) -> VitHProof {
    let l = witness.num_wires(); // circuit wires
    let l_check = l + 1;        // +1 for check wire (index 0)
    let w_flat = witness.flatten();
    let gates = witness.gates();

    // Commit to witness
    let commitment = hash_field_elements(&w_flat);

    // Fiat-Shamir transcript
    let mut transcript = Transcript::new(b"VitH");
    transcript.append_commitment(&commitment);
    transcript.append_field_element(delta);

    let mut copaths = Vec::new();
    let mut hidden_indices = Vec::new();
    let mut masked_witnesses = Vec::new();
    let mut check_values = Vec::new();
    let mut seed_roots = Vec::new();
    let mut hidden_leaf_roots = Vec::new();
    let mut hidden_sub_tree_paths = Vec::new();

    for rep in 0..params.repetitions {
        // GGM root seed: the prover's own secret randomness, generated by a
        // real RNG — *not* derived from the transcript. Protocol 15 line 1:
        // "sd ← {0,1}^κ". Deriving it from public transcript data (the old
        // code's bug) makes the entire GGM tree — including the hidden
        // leaf's shares — publicly recomputable, breaking both
        // zero-knowledge (the masked witness can be unmasked by anyone) and
        // soundness (a forged witness can be solved for post hoc).
        let mut root_seed = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut root_seed);

        // Expand GGM tree and compute every leaf's share vector once.
        let tree = GgmTree::expand(root_seed, params.tau);
        let all_leaf_shares: Vec<Vec<Fp>> = tree
            .leaf_seeds
            .iter()
            .map(|seed| leaf_shares(*seed, l_check, modulus))
            .collect();

        // Two-level GGM commitment (Section 4.2.2, our contribution): a
        // sub-Merkle-tree over each leaf's share vector gives h_i; the
        // outer Merkle tree over {h_i}_{i∈[τ]} gives the root `rt`. This is
        // committed into the transcript *before* Δ is drawn (below), so
        // altering any leaf's shares changes `rt` and hence the challenge —
        // exactly the property the old FS-derived seed never had.
        let leaf_roots: Vec<[u8; 32]> = all_leaf_shares
            .iter()
            .map(|shares| leaf_commitment_tree(shares).root())
            .collect();
        let outer_tree = MerkleTree::new(leaf_roots.clone());
        let rt = outer_tree.root();

        let vole = aggregate_vole_shares(&all_leaf_shares, l_check, modulus);

        // Masked witness: w̃_j = w_j + u_{j+1} (shift by 1 because index 0 is check wire)
        let masked: Vec<Fp> = (0..l)
            .map(|j| &w_flat[j] + &vole.u[j + 1])
            .collect();

        // Sample Δ (hidden leaf) and χ (batching challenge) via Fiat-Shamir,
        // from H(δ, rt, w̃) per Protocol 15 step 6 — rt and the masked
        // witness are committed before Δ is drawn.
        transcript.append_bytes(&(rep as u64).to_be_bytes());
        transcript.append_commitment(&rt);
        for mw in &masked {
            transcript.append_field_element(mw);
        }
        transcript.append_bytes(b"hidden_leaf");
        let hidden = transcript.challenge_index(params.tau);
        hidden_indices.push(hidden);

        transcript.append_bytes(b"chi");
        let chi = transcript.challenge(modulus);

        // GGM PRG co-path: lets the verifier reconstruct every non-hidden
        // leaf's seed (unchanged mechanism).
        let copath = tree.copath(hidden);
        copaths.push(copath);

        // The hidden leaf's own sub-tree root and per-position
        // authentication paths — this is what lets a verifier check the
        // witness positions it holds locally against `h_Δ` (dual-share
        // consistency, Section 4.2.2 "our contribution"). This codebase
        // models a broadcast channel rather than a literal per-verifier P2P
        // link, so paths for every position are included in one proof
        // rather than targeted per recipient.
        let hidden_tree = leaf_commitment_tree(&all_leaf_shares[hidden]);
        seed_roots.push(rt);
        hidden_leaf_roots.push(leaf_roots[hidden]);
        hidden_sub_tree_paths.push(
            (0..l_check)
                .map(|pos| hidden_tree.authentication_path(pos))
                .collect(),
        );

        // QuickSilver batched gate check
        // For each multiplication gate ℓ: a^ℓ * b^ℓ = c^ℓ
        // B_ℓ = v_a · v_b + (a·v_b + b·v_a - v_c)·Δ_hidden
        // But we don't know Δ_hidden as a field element in the clear...
        // Actually, the prover knows ALL leaf seeds including the hidden one,
        // so the prover knows ALL (u_j, v_j). The prover computes A₀, A₁:
        //   A₀ = Σ χ^ℓ · v_a^ℓ · v_b^ℓ
        //   A₁ = Σ χ^ℓ · (v_c^ℓ - b^ℓ·v_a^ℓ - a^ℓ·v_b^ℓ)
        // Note: sign on A₁ follows the paper: A₁ contributes with +Δ, and
        // the gate check is A₀ + A₁·Δ = 0 when gates are satisfied.

        let mut a0 = Fp::zero(modulus);
        let mut a1 = Fp::zero(modulus);
        let mut chi_power = Fp::one(modulus);

        for &(a_idx, b_idx, c_idx) in &gates {
            // Wire values (from witness)
            let a_val = &w_flat[a_idx];
            let b_val = &w_flat[b_idx];
            // VOLE v values (shifted by 1 for check wire)
            let v_a = &vole.v[a_idx + 1];
            let v_b = &vole.v[b_idx + 1];
            let v_c = &vole.v[c_idx + 1];

            // A₀ += χ^ℓ · v_a · v_b
            a0 = &a0 + &(&chi_power * &(v_a * v_b));
            // A₁ += χ^ℓ · (v_c - b·v_a - a·v_b)
            let a1_term = &(v_c - &(b_val * v_a)) - &(a_val * v_b);
            a1 = &a1 + &(&chi_power * &a1_term);

            chi_power = &chi_power * &chi;
        }

        // Masked check values: Ã₀ = A₀ + v₀, Ã₁ = A₁ + u₀
        // where (u₀, v₀) are the check wire's VOLE shares
        let a0_tilde = &a0 + &vole.v[0]; // v₀ is at index 0 (check wire)
        let a1_tilde = &a1 + &vole.u[0]; // u₀ is at index 0

        masked_witnesses.push(masked);
        check_values.push((a0_tilde, a1_tilde));
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
    // Reconstruct transcript
    let mut transcript = Transcript::new(b"VitH");
    transcript.append_commitment(&proof.commitment);
    transcript.append_field_element(delta);

    for rep in 0..params.repetitions {
        if rep >= proof.seed_roots.len()
            || rep >= proof.hidden_leaf_roots.len()
            || rep >= proof.hidden_sub_tree_paths.len()
        {
            return false;
        }

        let masked = &proof.masked_witnesses[rep];
        let l = masked.len();
        let l_check = l + 1;

        // Re-derive Δ and χ from H(δ, rt, w̃) — rt must be appended before
        // the masked witness, matching `vith_prove`'s commit order.
        transcript.append_bytes(&(rep as u64).to_be_bytes());
        transcript.append_commitment(&proof.seed_roots[rep]);
        for mw in masked {
            transcript.append_field_element(mw);
        }
        transcript.append_bytes(b"hidden_leaf");
        let expected_hidden = transcript.challenge_index(params.tau);
        if proof.hidden_indices[rep] != expected_hidden {
            return false;
        }
        let hidden = expected_hidden;

        transcript.append_bytes(b"chi");
        let chi = transcript.challenge(modulus);

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

        // Compute wire commitments from masked witness:
        // Commit(w_j) = w̃_j · Δ_fp - q_{j+1}
        // where Δ_fp is the hidden leaf index as a field element
        let delta_fp = Fp::new(BigUint::from(hidden), modulus);

        // Reconstruct the witness structure to get gate indices
        // We need to know which wires are gate inputs/outputs.
        // For N subsets: 2N input wires + N running product wires = 3N total
        // Gates: for j=1..N-1, gate (2N+j-1, 2j, 2N+j)
        let n_subsets = l / 3; // l = 3N
        let mut gates = Vec::new();
        for j in 1..n_subsets {
            gates.push((2 * n_subsets + j - 1, 2 * j, 2 * n_subsets + j));
        }

        // QuickSilver verification:
        // V = Σ χ^ℓ B_ℓ where B_ℓ = Commit(a^ℓ)·Commit(b^ℓ) - Commit(c^ℓ)·Δ_fp
        let mut v_check = Fp::zero(modulus);
        let mut chi_power = Fp::one(modulus);

        for &(a_idx, b_idx, c_idx) in &gates {
            // Commit(w_j) = w̃_j · Δ_fp - q_{j+1}
            let commit_a = &(&masked[a_idx] * &delta_fp) - &q[a_idx + 1];
            let commit_b = &(&masked[b_idx] * &delta_fp) - &q[b_idx + 1];
            let commit_c = &(&masked[c_idx] * &delta_fp) - &q[c_idx + 1];

            // B_ℓ = Commit(a) · Commit(b) - Commit(c) · Δ_fp
            let b_ell = &(&commit_a * &commit_b) - &(&commit_c * &delta_fp);

            v_check = &v_check + &(&chi_power * &b_ell);
            chi_power = &chi_power * &chi;
        }

        // Check: V + q₀ = Ã₀ + Ã₁·Δ_fp
        let (ref a0_tilde, ref a1_tilde) = proof.check_values[rep];
        let lhs = &v_check + &q[0]; // V + q₀
        let rhs = a0_tilde + &(a1_tilde * &delta_fp); // Ã₀ + Ã₁·Δ

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

        assert_eq!(w.num_wires(), 9);

        // Check gates
        let gates = w.gates();
        assert_eq!(gates.len(), 2); // N-1 = 2 gates
        // Gate 0: rp[0] * m[1] = rp[1] → (6, 2, 7)
        assert_eq!(gates[0], (6, 2, 7));
        // Gate 1: rp[1] * m[2] = rp[2] → (7, 4, 8)
        assert_eq!(gates[1], (7, 4, 8));
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

        // Tampered witness: wrong running product
        let mut w_bad = w_honest.clone();
        w_bad.running_products[2] = Fp::new(BigUint::from(99u32), &p); // should be 105

        let params = VitHParams::new(4, 8);
        let proof = vith_prove(&w_bad, &delta, &params, &p);

        let local_shares = BTreeMap::new();
        // Should fail because gate check rp[1]*m[2] != rp[2]
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
}
