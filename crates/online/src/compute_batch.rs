//! `compute_batch`: full Π_dVOPRF (paper §4-Online Protocol
//! `fig:doprf_protocol`). Zero sharings mask per-input cross-products;
//! a commit-then-hash round binds each server's `ṽ_i^{(k)}` values; ε_k
//! challenges batch the m relations into one; a single Π_VIP^Prl verifies
//! the aggregated relation.

use num_bigint::BigUint;
use vdoprf_crypto::hash::{hash_bytes, hash_field_elements};
use vdoprf_crypto::transcript::Transcript;
use vdoprf_field::Fp;
use vdoprf_network::SimulatedNetwork;
use vdoprf_offline::double_rand::generate_zero_additive_sharing;
use vdoprf_offline::PreSharedMaterial;
use vdoprf_ss::{SubsetFamily, SubsetT};

use crate::vip::vip_parallel;
use crate::{
    share_add_and_extract_pairs, OnlinePreprocessed, OnlineProof, OnlineResult, VipDvoprfOutput,
};

pub fn compute_batch(
    xs: &[Fp],
    pre: &OnlinePreprocessed,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> OnlineResult {
    let (per_input, mut comm) =
        share_add_and_extract_pairs(xs, pre, family, modulus);

    let n = family.n;
    let m = per_input.len();

    if m == 0 {
        return OnlineResult {
            client_outputs: Vec::new(),
            proof: OnlineProof::Batched(VipDvoprfOutput {
                vip: crate::vip::VipParallelOutput {
                    w_rho_shares: Vec::new(),
                    u_rho_shares: Vec::new(),
                    v_rho_shares: Vec::new(),
                    sigma_shares: Vec::new(),
                    c_shares: Vec::new(),
                },
                tilde_v: vec![Vec::new(); n],
            }),
            comm,
        };
    }

    // Step 1: zero sharings z_i^{(k)} with Σ_i z_i^{(k)} = 0 (F_Zero).
    // The counter MUST be shared across all parties for a given k so that
    // each party's pairwise-PRG terms cancel pairwise — otherwise the
    // shares don't sum to zero.
    let mut tilde_v: Vec<Vec<Fp>> = vec![Vec::with_capacity(m); n];
    for k in 0..m {
        let counter = 100_000 + k as u64;
        for i in 0..n {
            let z = generate_zero_additive_sharing(
                i,
                counter,
                &pre_shared[i],
                family,
                modulus,
            );
            let v_i = per_input[k].party_cp[i].clone();
            tilde_v[i].push(&v_i + &z);
        }
    }

    // Step 2: commit-then-hash round. Each server S_i broadcasts
    // h_i = H(ṽ_i^{(1)} ‖ … ‖ ṽ_i^{(m)}). Charge bytes and 1 round.
    let mut commit_net = SimulatedNetwork::new(n);
    let mut h_list: Vec<[u8; 32]> = Vec::with_capacity(n);
    for i in 0..n {
        let h = hash_field_elements(&tilde_v[i]);
        commit_net.broadcast(i, h.to_vec());
        h_list.push(h);
    }
    comm.merge(&commit_net.stats());

    // Step 3: derive ρ_ε = H(h_1 ‖ … ‖ h_n), ε_k = H(ρ_ε ‖ k).
    let mut rho_hasher_buf = Vec::with_capacity(32 * n);
    for h in &h_list {
        rho_hasher_buf.extend_from_slice(h);
    }
    let rho_eps_digest = hash_bytes(&rho_hasher_buf);
    let epsilons: Vec<Fp> = (0..m)
        .map(|k| {
            let mut buf = Vec::with_capacity(32 + 8);
            buf.extend_from_slice(&rho_eps_digest);
            buf.extend_from_slice(&(k as u64).to_be_bytes());
            let digest = hash_bytes(&buf);
            Fp::new(BigUint::from_bytes_be(&digest) % modulus, modulus)
        })
        .collect();

    // Step 4: aggregate — scale every a-value in the cross-product pairs by
    // ε_k and concatenate across k into per-prover lists for VIP. §5-Online
    // line 250: `⟨â_i^{(j,ℓ)}⟩ = ε_j · ⟨a_i^{(j,ℓ)}⟩`. Each pair carries its
    // originating subset `(T_u, T_v)` so `vip_single` can reconstruct the
    // residual a^{(1)}/b^{(1)} sharings from fold weights — no per-pair
    // RSS sharing materialised. ε_k applied to plaintext a_val is equivalent
    // to scaling the degenerate sharing (whose target subset is unchanged).
    let mut per_prover_pairs: Vec<Vec<(Fp, Fp)>> = vec![Vec::new(); n];
    let mut per_prover_targets: Vec<Vec<(SubsetT, SubsetT)>> = vec![Vec::new(); n];
    for k in 0..m {
        let eps = &epsilons[k];
        for i in 0..n {
            for (idx, (a_val, b_val)) in per_input[k].party_pairs[i].iter().enumerate() {
                per_prover_pairs[i].push((eps * a_val, b_val.clone()));
                per_prover_targets[i].push(per_input[k].party_targets[i][idx]);
            }
        }
    }

    // Step 5: single Π_VIP^Prl over the aggregated relation. Seed the
    // transcript with the commit-round root so the FS state binds the
    // proof to each server's {ṽ_i^{(k)}}.
    let mut transcript = Transcript::new(b"vdoprf.online.compute_batch");
    transcript.append_commitment(&rho_eps_digest);
    let mut rand_counter = 4_000u64;
    let (vip_out, vip_comm) = vip_parallel(
        &per_prover_pairs,
        &per_prover_targets,
        family,
        modulus,
        &mut transcript,
        pre_shared,
        &mut rand_counter,
    );
    comm.merge(&vip_comm);

    // Step 6: return `tilde_v` and the 5 Π_VIP^Prl proof shares — the bench
    // simulates server→client delivery and reconstruction outside this
    // function, so the protocol itself stays pure on server-side state.
    // `client_outputs` is still populated as a convenience for tests.
    let mut client_outputs = Vec::with_capacity(m);
    for k in 0..m {
        let mut v = Fp::zero(modulus);
        for i in 0..n {
            v = &v + &tilde_v[i][k];
        }
        client_outputs.push(v);
    }

    OnlineResult {
        client_outputs,
        proof: OnlineProof::Batched(VipDvoprfOutput {
            vip: vip_out,
            tilde_v,
        }),
        comm,
    }
}

#[cfg(test)]
mod tests {
    // ========================================================================
    // Property tests against §4-Online.tex `fig:doprf_protocol` (Π_dVOPRF).
    //
    // Six things the paper promises (lines 253-274):
    //   * Zero-sharing cancellation: Σ_i z_i^{(k)} = 0 ⇒ Σ_i ṽ_i^{(k)} = v^{(k)}.
    //   * Per-server cross-product: v_i^{(k)} = Σ_ℓ a^{(k,ℓ)}_i · b^{(k,ℓ)}_i.
    //   * Commit-then-hash: h_i = H(ṽ_i^{(1)} ‖ … ‖ ṽ_i^{(m)}).
    //   * Challenge derivation: ρ_ε = H(h_1 ‖ … ‖ h_n); ε_k = H(ρ_ε ‖ k).
    //   * Random-linear aggregation: Π_VIP^Prl input = {(ε_k·a, b)}_{k,ℓ}.
    //   * Client verdict: Accept iff W(ρ)=U(ρ)·V(ρ) ∧ Σ=0 ∧ c = Σ_k ε_k·v^{(k)}.
    // ========================================================================

    use super::*;
    use crate::setup_random_preprocessed_m;
    use crate::vip::{client_verify_dvoprf, client_verify_vip_parallel, VipResult};
    use vdoprf_offline::setup_pre_shared;
    use vdoprf_ss::{ReplicatedSharing, RssShare};

    fn small_setup(
        m: usize,
    ) -> (
        usize,
        BigUint,
        SubsetFamily,
        Vec<PreSharedMaterial>,
        OnlinePreprocessed,
    ) {
        let n = 3;
        let t = 1;
        let modulus = BigUint::from(65537u32);
        let family = SubsetFamily::new(n, t);
        let pre_shared = setup_pre_shared(n, t, &modulus);
        let pre = setup_random_preprocessed_m(m.max(1), &family, &modulus);
        (n, modulus, family, pre_shared, pre)
    }

    /// Paper §4-Online line 7 / 274: v^{(k)} = α_k^e · (x_k + k).
    fn expected_v(xs: &[Fp], pre: &OnlinePreprocessed, modulus: &BigUint) -> Vec<Fp> {
        let k = pre.k_sharing.reconstruct(modulus);
        xs.iter()
            .enumerate()
            .map(|(j, x)| {
                let alpha_e = pre.alpha_e_sharings[j % pre.alpha_e_sharings.len()]
                    .reconstruct(modulus);
                &(x + &k) * &alpha_e
            })
            .collect()
    }

    /// Client-side re-derivation per §4-Online line 263:
    ///   h_i = H(ṽ_i^{(1)} ‖ … ‖ ṽ_i^{(m)}),
    ///   ρ_ε = H(h_1 ‖ … ‖ h_n),
    ///   ε_k = H(ρ_ε ‖ k).
    /// Uses the *same* hash primitives (`hash_field_elements`, `hash_bytes`)
    /// that `compute_batch` calls internally, so this byte-matches the
    /// protocol rather than reimplementing the derivation.
    fn rederive_epsilons(tilde_v: &[Vec<Fp>], modulus: &BigUint) -> Vec<Fp> {
        let n = tilde_v.len();
        let m = if n == 0 { 0 } else { tilde_v[0].len() };
        let mut buf = Vec::with_capacity(32 * n);
        for row in tilde_v {
            buf.extend_from_slice(&hash_field_elements(row));
        }
        let rho_eps = hash_bytes(&buf);
        (0..m)
            .map(|k| {
                let mut b = Vec::with_capacity(40);
                b.extend_from_slice(&rho_eps);
                b.extend_from_slice(&(k as u64).to_be_bytes());
                let d = hash_bytes(&b);
                Fp::new(BigUint::from_bytes_be(&d) % modulus, modulus)
            })
            .collect()
    }

    /// Add 1 to one subset component of party-0's RSS share. RSS
    /// reconstruction uses first-write-wins, so a single-component bump
    /// shifts the reconstructed scalar by +1.
    fn bump_party0(shares: &mut [RssShare], modulus: &BigUint) {
        let one = Fp::one(modulus);
        let (_subset, fp) = shares[0]
            .shares
            .iter_mut()
            .next()
            .expect("RssShare has at least one subset component");
        *fp = &*fp + &one;
    }

    fn unwrap_batched(proof: &OnlineProof) -> &VipDvoprfOutput {
        match proof {
            OnlineProof::Batched(b) => b,
            _ => panic!("compute_batch must emit OnlineProof::Batched"),
        }
    }

    fn unwrap_batched_mut(proof: &mut OnlineProof) -> &mut VipDvoprfOutput {
        match proof {
            OnlineProof::Batched(b) => b,
            _ => panic!("compute_batch must emit OnlineProof::Batched"),
        }
    }

    fn run_batch(m: usize) -> (BigUint, usize, OnlinePreprocessed, Vec<Fp>, OnlineResult) {
        let (n, modulus, family, pre_shared, pre) = small_setup(m);
        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..m).map(|_| Fp::random(&modulus, &mut rng)).collect();
        let r = compute_batch(&xs, &pre, &pre_shared, &family, &modulus);
        (modulus, n, pre, xs, r)
    }

    // ---- A. Output shape and honest correctness ----

    /// §4-Online line 274: client_outputs[k] = v^{(k)} = α_k^e · (x_k + k).
    #[test]
    fn compute_batch_client_outputs_equal_v_k() {
        let (modulus, _n, pre, xs, r) = run_batch(4);
        let expected = expected_v(&xs, &pre, &modulus);
        for (a, e) in r.client_outputs.iter().zip(expected.iter()) {
            assert_eq!(a.value, e.value);
        }
    }

    /// The wrapper always returns `OnlineProof::Batched` and `m` client outputs.
    #[test]
    fn compute_batch_output_is_batched_with_m_entries() {
        let (_modulus, _n, _pre, xs, r) = run_batch(3);
        assert_eq!(r.client_outputs.len(), xs.len());
        assert!(matches!(r.proof, OnlineProof::Batched(_)));
    }

    /// §4-Online line 262: h_i is computed over `{ṽ_i^{(k)}}_{k∈[m]}`, so
    /// `tilde_v` must be shaped `n × m`.
    #[test]
    fn compute_batch_tilde_v_shape_is_n_by_m() {
        let (_modulus, n, _pre, xs, r) = run_batch(4);
        let b = unwrap_batched(&r.proof);
        assert_eq!(b.tilde_v.len(), n);
        for row in &b.tilde_v {
            assert_eq!(row.len(), xs.len());
        }
    }

    // ---- B. Per-protocol invariants ----

    /// §4-Online line 253: `Σ_i z_i^{(k)} = 0` ⇒ `Σ_i ṽ_i^{(k)} = v^{(k)}`.
    #[test]
    fn compute_batch_tilde_v_sums_to_v_k() {
        let (modulus, n, pre, xs, r) = run_batch(3);
        let b = unwrap_batched(&r.proof);
        let expected = expected_v(&xs, &pre, &modulus);
        for k in 0..xs.len() {
            let mut sum = Fp::zero(&modulus);
            for i in 0..n {
                sum = &sum + &b.tilde_v[i][k];
            }
            assert_eq!(sum.value, expected[k].value);
        }
    }

    /// §4-Online line 263: ε_k is a deterministic function of the received
    /// ṽ. Re-derivation is a pure function — two calls agree.
    #[test]
    fn compute_batch_epsilons_rederivable_from_tilde_v() {
        let (modulus, _n, _pre, xs, r) = run_batch(3);
        let b = unwrap_batched(&r.proof);
        let eps1 = rederive_epsilons(&b.tilde_v, &modulus);
        let eps2 = rederive_epsilons(&b.tilde_v, &modulus);
        assert_eq!(eps1.len(), xs.len());
        for (a, c) in eps1.iter().zip(eps2.iter()) {
            assert_eq!(a.value, c.value);
        }
    }

    /// §4-Online line 274 (third verifier check): c = Σ_k ε_k · v^{(k)}, with
    /// ε_k re-derived from the received ṽ.
    #[test]
    fn compute_batch_c_binds_to_tilde_v() {
        let (modulus, _n, pre, xs, r) = run_batch(4);
        let b = unwrap_batched(&r.proof);
        let v = expected_v(&xs, &pre, &modulus);
        let eps = rederive_epsilons(&b.tilde_v, &modulus);

        let mut expected_c = Fp::zero(&modulus);
        for k in 0..xs.len() {
            expected_c = &expected_c + &(&eps[k] * &v[k]);
        }
        let c = ReplicatedSharing::reconstruct_from_party_shares(&b.vip.c_shares, &modulus);
        assert_eq!(c.value, expected_c.value);
    }

    /// §4-Online line 274 (first verifier check): W(ρ) = U(ρ)·V(ρ).
    #[test]
    fn compute_batch_vip_w_rho_equals_u_rho_v_rho() {
        let (modulus, _n, _pre, _xs, r) = run_batch(3);
        let b = unwrap_batched(&r.proof);
        let w = ReplicatedSharing::reconstruct_from_party_shares(&b.vip.w_rho_shares, &modulus);
        let u = ReplicatedSharing::reconstruct_from_party_shares(&b.vip.u_rho_shares, &modulus);
        let v = ReplicatedSharing::reconstruct_from_party_shares(&b.vip.v_rho_shares, &modulus);
        assert_eq!(w.value, (&u * &v).value);
    }

    /// §4-Online line 274 (second verifier check): Σ = 0.
    #[test]
    fn compute_batch_vip_sigma_is_zero() {
        let (modulus, _n, _pre, _xs, r) = run_batch(3);
        let b = unwrap_batched(&r.proof);
        assert!(ReplicatedSharing::reconstruct_from_party_shares(&b.vip.sigma_shares, &modulus).is_zero());
    }

    /// §4-Online line 274: with all VIP checks passing, the client verdict
    /// (via `client_verify_vip_parallel`) is Accept. Asserted at the
    /// `compute_batch` layer for coverage independent of `lib.rs`.
    #[test]
    fn compute_batch_client_verify_accepts_honest_local() {
        let (modulus, _n, _pre, _xs, r) = run_batch(3);
        let b = unwrap_batched(&r.proof);
        assert_eq!(
            client_verify_vip_parallel(&b.vip, &modulus),
            VipResult::Accept,
        );
    }

    // ---- C. Edge cases on m ----

    /// xs = []: early-return branch — empty outputs, empty VIP shares, empty
    /// per-server `tilde_v` rows.
    #[test]
    fn compute_batch_empty_input_m_zero() {
        let (n, modulus, family, pre_shared, pre) = small_setup(1);
        let r = compute_batch(&[], &pre, &pre_shared, &family, &modulus);
        assert!(r.client_outputs.is_empty());
        let b = unwrap_batched(&r.proof);
        assert!(b.vip.w_rho_shares.is_empty());
        assert!(b.vip.u_rho_shares.is_empty());
        assert!(b.vip.v_rho_shares.is_empty());
        assert!(b.vip.sigma_shares.is_empty());
        assert!(b.vip.c_shares.is_empty());
        assert_eq!(b.tilde_v.len(), n);
        for row in &b.tilde_v {
            assert!(row.is_empty());
        }
    }

    /// m = 1: minimal batch preserves every §4-Online invariant.
    #[test]
    fn compute_batch_single_input_m_one() {
        let (modulus, _n, pre, xs, r) = run_batch(1);
        assert_eq!(r.client_outputs.len(), 1);

        let b = unwrap_batched(&r.proof);
        let v = expected_v(&xs, &pre, &modulus);
        assert_eq!(r.client_outputs[0].value, v[0].value);

        let w = ReplicatedSharing::reconstruct_from_party_shares(&b.vip.w_rho_shares, &modulus);
        let u = ReplicatedSharing::reconstruct_from_party_shares(&b.vip.u_rho_shares, &modulus);
        let v_sh = ReplicatedSharing::reconstruct_from_party_shares(&b.vip.v_rho_shares, &modulus);
        assert_eq!(w.value, (&u * &v_sh).value);
        assert!(ReplicatedSharing::reconstruct_from_party_shares(&b.vip.sigma_shares, &modulus).is_zero());

        let eps = rederive_epsilons(&b.tilde_v, &modulus);
        let c = ReplicatedSharing::reconstruct_from_party_shares(&b.vip.c_shares, &modulus);
        assert_eq!(c.value, (&eps[0] * &v[0]).value);
    }

    /// m = 4: larger batch checks every per-k identity + aggregated c-binding
    /// + full-verdict Accept.
    #[test]
    fn compute_batch_larger_batch() {
        let (modulus, n, pre, xs, r) = run_batch(4);
        let b = unwrap_batched(&r.proof);
        let v = expected_v(&xs, &pre, &modulus);

        for k in 0..xs.len() {
            let mut sum = Fp::zero(&modulus);
            for i in 0..n {
                sum = &sum + &b.tilde_v[i][k];
            }
            assert_eq!(sum.value, v[k].value);
            assert_eq!(r.client_outputs[k].value, v[k].value);
        }

        let eps = rederive_epsilons(&b.tilde_v, &modulus);
        let mut expected_c = Fp::zero(&modulus);
        for k in 0..xs.len() {
            expected_c = &expected_c + &(&eps[k] * &v[k]);
        }
        assert_eq!(
            ReplicatedSharing::reconstruct_from_party_shares(&b.vip.c_shares, &modulus).value,
            expected_c.value,
        );
        assert_eq!(
            client_verify_vip_parallel(&b.vip, &modulus),
            VipResult::Accept,
        );
    }

    // ---- D. Soundness (§4-Online lines 168, 274) ----

    /// Tampering W(ρ) breaks the first verifier check ⇒ Abort.
    #[test]
    fn compute_batch_tampered_w_rho_rejected() {
        let (modulus, _n, _pre, _xs, mut r) = run_batch(3);
        {
            let b = unwrap_batched_mut(&mut r.proof);
            bump_party0(&mut b.vip.w_rho_shares, &modulus);
        }
        let b = unwrap_batched(&r.proof);
        assert_eq!(
            client_verify_vip_parallel(&b.vip, &modulus),
            VipResult::Abort,
        );
    }

    /// Tampering Σ breaks the second verifier check ⇒ Abort.
    #[test]
    fn compute_batch_tampered_sigma_rejected() {
        let (modulus, _n, _pre, _xs, mut r) = run_batch(3);
        {
            let b = unwrap_batched_mut(&mut r.proof);
            bump_party0(&mut b.vip.sigma_shares, &modulus);
        }
        let b = unwrap_batched(&r.proof);
        assert_eq!(
            client_verify_vip_parallel(&b.vip, &modulus),
            VipResult::Abort,
        );
    }

    /// Regression check: after Fix A/B/C the per-server total S↔S + S↔C
    /// traffic at (5, 2), m=1 lands in the Chapter-5-consistent band for
    /// Π_dVOPRF. Guards against accidentally reverting to:
    ///
    ///   * additive-only `rss_to_additive` delivery of the five VIP proof
    ///     outputs (Fix B) — that would collapse `5·c_Open` → `5` field
    ///     elements per server,
    ///   * the `HashMap::insert` overwrite in `send_p2p` (Fix A) — that
    ///     would drop γ−1 of the γ VSS iterations per prover,
    ///   * full-RSS-share wire cost for every VSS (Fix C) — that would
    ///     scale each shared value by `C(n-1, t)` instead of `≈1.5`.
    ///
    /// Chapter 5 Appendix §A.2 (`appendix.tex:1105–1109`) predicts ~78
    /// field elements per server at (5, 2), m=1 using the amortised
    /// `c_Share ≈ 1.5` formula. That formula measures one prover's
    /// outgoing cost; the actual wire total across all n parallel VIP
    /// instances is `≈ n · 3 · c_Share · γ` (each server pays as dealer in
    /// one instance + non-dealer in n−1), so per-server comes out a factor
    /// of a few higher. We assert on a wide band that still catches the
    /// pre-fix regressions (which measured ~1000+ or ~16, each a 10×+
    /// outlier) without over-constraining the exact paper formula.
    #[test]
    fn compute_batch_per_server_bytes_match_chapter_5_at_5_2_m_1() {
        use vdoprf_network::SimulatedNetwork;
        use vdoprf_offline::rss_share::charge_client_rss_share;

        let n = 5;
        let t = 2;
        let m = 1;
        // 128-bit prime: feb = 16. Large enough that 32-byte hashes don't
        // dominate the per-share byte count, so the per-server Fp-count
        // comparison stays meaningful.
        let modulus =
            BigUint::parse_bytes(b"340282366920938463463374607431768211297", 10).unwrap();
        let family = SubsetFamily::new(n, t);
        let pre_shared = vdoprf_offline::setup_pre_shared(n, t, &modulus);
        let pre = setup_random_preprocessed_m(m, &family, &modulus);

        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..m).map(|_| Fp::random(&modulus, &mut rng)).collect();

        // Client input VSS (Π_RSS.Share, client as dealer).
        let mut input_net = SimulatedNetwork::new(n);
        charge_client_rss_share(m, &family, &modulus, &mut input_net);

        let r = compute_batch(&xs, &pre, &pre_shared, &family, &modulus);

        // Simulate server→client VIP.Open delivery.
        let mut client_net = SimulatedNetwork::new(n);
        let (tilde_v, vip) = match &r.proof {
            OnlineProof::Batched(b) => (&b.tilde_v, &b.vip),
            _ => panic!(),
        };
        let feb = ((modulus.bits() + 7) / 8) as usize;
        const HASH_BYTES: usize = 32;
        for i in 0..n {
            for k in 0..m {
                let payload = vec![0u8; feb];
                let _ = k;
                client_net.send_to_client(i, payload);
            }
        }
        // Full RSS opens of w, u, v, σ, c (5 values).
        for field in [
            &vip.w_rho_shares,
            &vip.u_rho_shares,
            &vip.v_rho_shares,
            &vip.sigma_shares,
            &vip.c_shares,
        ] {
            for (i, share) in field.iter().enumerate() {
                let bytes = share.shares.len() * feb + HASH_BYTES;
                client_net.send_to_client(i, vec![0u8; bytes]);
            }
        }

        let mut comm = r.comm;
        comm.merge(&input_net.stats());
        comm.merge(&client_net.stats());
        let total_bytes = comm.p2p_bytes + comm.broadcast_bytes + comm.client_bytes;
        let per_server_fp = total_bytes as f64 / (n as f64 * feb as f64);

        // Regression band: catches the two pre-fix extremes.
        // Pre-Fix-B would give ≈ input (1.5) + hashes (4) + VIP p2p (~105)
        // + triple-mul (~7) + additive-only opens (5+m=6): per-server total
        // ~125 — but WITHOUT the 5-proof RSS.Open cost. Hard to distinguish
        // purely numerically, so pair with the separate `client_verify`
        // tests above to guard Fix B specifically. Pre-Fix-A would have the
        // VIP p2p drop by a factor of ~γ=5 (only the end-of-loop VSS would
        // land in `p2p_buf`), giving per-server <30. Pre-Fix-C would have
        // every VSS scale by `C(n-1, t)/1.5 ≈ 4×`, pushing per-server >500.
        let (lower, upper) = (50.0_f64, 400.0_f64);
        assert!(
            per_server_fp >= lower && per_server_fp <= upper,
            "per-server bytes-as-Fp = {per_server_fp:.1} outside [{:.1}, {:.1}]",
            lower,
            upper,
        );
    }

    /// §4-Online line 274 (third verifier check): tampering ṽ post-hoc shifts
    /// the client-re-derived ε_k away from the ε_k used during Π_VIP^Prl, so
    /// `c = Σ_k ε_k·v^{(k)}` fails. This is the commit-then-hash binding
    /// mechanism the paper attributes to `h_i = H(ṽ_i^{(1)} ‖ … ‖ ṽ_i^{(m)})`.
    #[test]
    fn compute_batch_tampered_tilde_v_breaks_c_binding() {
        let (modulus, n, _pre, xs, mut r) = run_batch(3);
        {
            let b = unwrap_batched_mut(&mut r.proof);
            b.tilde_v[0][0] = &b.tilde_v[0][0] + &Fp::one(&modulus);
        }
        let b = unwrap_batched(&r.proof);

        // Client reconstruction against the tampered ṽ.
        let mut v_from_client = Vec::with_capacity(xs.len());
        for k in 0..xs.len() {
            let mut s = Fp::zero(&modulus);
            for i in 0..n {
                s = &s + &b.tilde_v[i][k];
            }
            v_from_client.push(s);
        }
        let eps = rederive_epsilons(&b.tilde_v, &modulus);
        let mut c_from_client = Fp::zero(&modulus);
        for k in 0..xs.len() {
            c_from_client = &c_from_client + &(&eps[k] * &v_from_client[k]);
        }
        let c = ReplicatedSharing::reconstruct_from_party_shares(&b.vip.c_shares, &modulus);
        assert_ne!(c.value, c_from_client.value);
    }

    /// §5-Online line 255 — full Π_dVOPRF^m verdict. Honest execution accepts.
    #[test]
    fn client_verify_dvoprf_accepts_honest() {
        let (modulus, _n, _pre, _xs, r) = run_batch(4);
        let b = unwrap_batched(&r.proof);
        assert_eq!(
            client_verify_dvoprf(&b.vip, &b.tilde_v, &modulus),
            VipResult::Accept,
        );
    }

    /// §5-Online line 255 — the third verifier arm. Tampering ṽ post-hoc
    /// shifts the rederived ε_k, so `c = Σ_k ε_k · v^{(k)}` mismatches the
    /// committed `c` ⇒ Abort. Critically, `client_verify_vip_parallel` would
    /// still Accept this case (W=UV and Σ=0 are unaffected) — the new arm
    /// is exactly what catches it.
    #[test]
    fn client_verify_dvoprf_rejects_tampered_tilde_v() {
        let (modulus, _n, _pre, _xs, mut r) = run_batch(3);
        {
            let b = unwrap_batched_mut(&mut r.proof);
            b.tilde_v[0][0] = &b.tilde_v[0][0] + &Fp::one(&modulus);
        }
        let b = unwrap_batched(&r.proof);

        // Old verifier would still Accept — the gap.
        assert_eq!(
            client_verify_vip_parallel(&b.vip, &modulus),
            VipResult::Accept,
        );
        // New verifier rejects.
        assert_eq!(
            client_verify_dvoprf(&b.vip, &b.tilde_v, &modulus),
            VipResult::Abort,
        );
    }

    /// `client_verify_dvoprf` still subsumes the W=UV arm.
    #[test]
    fn client_verify_dvoprf_rejects_tampered_w_rho() {
        let (modulus, _n, _pre, _xs, mut r) = run_batch(3);
        {
            let b = unwrap_batched_mut(&mut r.proof);
            bump_party0(&mut b.vip.w_rho_shares, &modulus);
        }
        let b = unwrap_batched(&r.proof);
        assert_eq!(
            client_verify_dvoprf(&b.vip, &b.tilde_v, &modulus),
            VipResult::Abort,
        );
    }

    /// `client_verify_dvoprf` still subsumes the Σ=0 arm.
    #[test]
    fn client_verify_dvoprf_rejects_tampered_sigma() {
        let (modulus, _n, _pre, _xs, mut r) = run_batch(3);
        {
            let b = unwrap_batched_mut(&mut r.proof);
            bump_party0(&mut b.vip.sigma_shares, &modulus);
        }
        let b = unwrap_batched(&r.proof);
        assert_eq!(
            client_verify_dvoprf(&b.vip, &b.tilde_v, &modulus),
            VipResult::Abort,
        );
    }
}
