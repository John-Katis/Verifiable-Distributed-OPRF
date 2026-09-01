//! `compute`: per-input sequential Π_VIP^Prl — paper §4-Online "Compute"
//! (processes queries one at a time). Each of the `m` client inputs drives
//! one full `Π_VIP^Prl` invocation; rounds stack sequentially.

use num_bigint::BigUint;
use vdoprf_field::Fp;
use vdoprf_network::CommStats;
use vdoprf_offline::PreSharedMaterial;
use vdoprf_ss::{ReplicatedSharing, SubsetFamily};

use crate::vip::vip_parallel;
use crate::{
    share_add_and_extract_pairs, OnlinePreprocessed, OnlineProof, OnlineResult,
};

pub fn compute(
    xs: &[Fp],
    pre: &OnlinePreprocessed,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> OnlineResult {
    let (per_input, mut comm) =
        share_add_and_extract_pairs(xs, pre, family, modulus);

    let mut rand_counter = 3_000u64;
    let mut bundles = Vec::with_capacity(per_input.len());
    let mut client_outputs: Vec<Fp> = Vec::with_capacity(per_input.len());

    for input in per_input.iter() {
        let (out, vip_comm) = vip_parallel(
            &input.party_pairs,
            &input.party_targets,
            family,
            modulus,
            pre_shared,
            &mut rand_counter,
        );
        comm.merge(&vip_comm);

        // Reconstruct c for the test surface; the bench simulates client
        // delivery of the 5 proof shares (w_ρ, u_ρ, v_ρ, σ, c) outside.
        let c = ReplicatedSharing::reconstruct_from_party_shares(&out.c_shares, modulus);
        client_outputs.push(c);
        bundles.push(out);
    }

    let _ = CommStats::default();
    OnlineResult {
        client_outputs,
        proof: OnlineProof::Components(bundles),
        comm,
    }
}
