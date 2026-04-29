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
use std::collections::BTreeMap;
use vdoprf_crypto::ggm::GgmTree;
use vdoprf_crypto::hash::hash_field_elements;
use vdoprf_crypto::prg::PairwisePrg;
use vdoprf_crypto::transcript::Transcript;
use vdoprf_field::Fp;
use vdoprf_ss::SubsetT;

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

/// Generate VOLE shares from GGM leaf seeds.
/// L = number of circuit wires. Wire index 0 is the check wire.
/// Total wire count including check wire = L + 1.
fn generate_vole_shares(
    tree: &GgmTree,
    num_wires_with_check: usize, // L + 1
    modulus: &BigUint,
) -> VoleShares {
    let mut u = vec![Fp::zero(modulus); num_wires_with_check];
    let mut v = vec![Fp::zero(modulus); num_wires_with_check];

    for (i, leaf_seed) in tree.leaf_seeds.iter().enumerate() {
        let prg = PairwisePrg::new(*leaf_seed);
        let i_fp = Fp::new(BigUint::from(i), modulus);
        for j in 0..num_wires_with_check {
            let s_j_i = prg.generate(j as u64, modulus);
            // u_j += s_j^(i)
            u[j] = &u[j] + &s_j_i;
            // v_j += -i * s_j^(i)
            v[j] = &v[j] - &(&i_fp * &s_j_i);
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
    /// Co-paths for each repetition.
    pub copaths: Vec<Vec<[u8; 16]>>,
    /// Hidden leaf indices Δ per repetition.
    pub hidden_indices: Vec<usize>,
}

impl VitHProof {
    /// Wire-byte size as placed on the network by the broadcast accounting in
    /// `approach_iii::gen_zkp` (VitH branch). Matches the exact byte stream
    /// dealers push out: commitment, then per-repetition masked witnesses and
    /// (Ã₀, Ã₁), then all copath seeds. `hidden_indices` is re-derived from
    /// the Fiat-Shamir transcript and not sent on the wire.
    pub fn wire_bytes(&self, feb: usize) -> usize {
        let mut bytes = self.commitment.len();
        for rep in &self.masked_witnesses {
            bytes += rep.len() * feb;
        }
        bytes += self.check_values.len() * 2 * feb;
        for copath in &self.copaths {
            bytes += copath.len() * 16;
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

    for rep in 0..params.repetitions {
        // Derive GGM root seed from transcript
        transcript.append_bytes(&(rep as u64).to_be_bytes());
        transcript.append_bytes(b"ggm_seed");
        let seed_challenge = transcript.challenge(modulus);
        let seed_bytes = seed_challenge.value.to_bytes_be();
        let mut root_seed = [0u8; 16];
        for (i, &b) in seed_bytes.iter().rev().take(16).enumerate() {
            root_seed[i] = b;
        }

        // Expand GGM tree
        let tree = GgmTree::expand(root_seed, params.tau);

        // Generate VOLE shares
        let vole = generate_vole_shares(&tree, l_check, modulus);

        // Masked witness: w̃_j = w_j + u_{j+1} (shift by 1 because index 0 is check wire)
        let masked: Vec<Fp> = (0..l)
            .map(|j| &w_flat[j] + &vole.u[j + 1])
            .collect();

        // Sample Δ (hidden leaf) and χ (batching challenge) via Fiat-Shamir
        for mw in &masked {
            transcript.append_field_element(mw);
        }
        transcript.append_bytes(b"hidden_leaf");
        let hidden = transcript.challenge_index(params.tau);
        hidden_indices.push(hidden);

        transcript.append_bytes(b"chi");
        let chi = transcript.challenge(modulus);

        // Get co-path
        let copath = tree.copath(hidden);
        copaths.push(copath);

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
    }
}

/// Verify a VOLEitH proof.
pub fn vith_verify(
    proof: &VitHProof,
    delta: &Fp,
    _local_shares: &BTreeMap<SubsetT, (Fp, Fp)>,
    params: &VitHParams,
    modulus: &BigUint,
) -> bool {
    // Reconstruct transcript
    let mut transcript = Transcript::new(b"VitH");
    transcript.append_commitment(&proof.commitment);
    transcript.append_field_element(delta);

    for rep in 0..params.repetitions {
        // Re-derive GGM seed
        transcript.append_bytes(&(rep as u64).to_be_bytes());
        transcript.append_bytes(b"ggm_seed");
        let seed_challenge = transcript.challenge(modulus);
        let seed_bytes = seed_challenge.value.to_bytes_be();
        let mut root_seed = [0u8; 16];
        for (i, &b) in seed_bytes.iter().rev().take(16).enumerate() {
            root_seed[i] = b;
        }

        let masked = &proof.masked_witnesses[rep];
        let l = masked.len();
        let l_check = l + 1;

        // Re-derive Δ and χ
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
