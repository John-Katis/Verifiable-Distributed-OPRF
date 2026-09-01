//! `compute_parallel`: m parallel Π_VIP^Prl invocations (one per client
//! input), accounted with `CommStats::merge_parallel` — rounds take the
//! max, bytes sum. Matches paper §4-Online "ComputeParallel" (keeps the
//! per-query DZKP but parallel-dispatches the m local degree-2 evaluations).

use num_bigint::BigUint;
use vdoprf_field::Fp;
use vdoprf_network::CommStats;
use vdoprf_offline::PreSharedMaterial;
use vdoprf_ss::{ReplicatedSharing, SubsetFamily};

use crate::vip::vip_parallel;
use crate::{
    share_add_and_extract_pairs, OnlinePreprocessed, OnlineProof, OnlineResult,
};

pub fn compute_parallel(
    xs: &[Fp],
    pre: &OnlinePreprocessed,
    pre_shared: &[PreSharedMaterial],
    family: &SubsetFamily,
    modulus: &BigUint,
) -> OnlineResult {
    let (per_input, mut comm) =
        share_add_and_extract_pairs(xs, pre, family, modulus);

    let mut rand_counter = 2_000u64;
    let mut bundles = Vec::with_capacity(per_input.len());
    let mut client_outputs: Vec<Fp> = Vec::with_capacity(per_input.len());
    let mut parallel_comm = CommStats::default();

    for input in per_input.iter() {
        let (out, vip_comm) = vip_parallel(
            &input.party_pairs,
            &input.party_targets,
            family,
            modulus,
            pre_shared,
            &mut rand_counter,
        );
        // Parallel composition across inputs: rounds = max, bytes sum.
        parallel_comm.merge_parallel(&vip_comm);

        let c = ReplicatedSharing::reconstruct_from_party_shares(&out.c_shares, modulus);
        client_outputs.push(c);
        bundles.push(out);
    }
    comm.merge(&parallel_comm);

    OnlineResult {
        client_outputs,
        proof: OnlineProof::Components(bundles),
        comm,
    }
}
