//! Smoke tests for memory-budget-sensitive cells. The (9,4), m=35
//! configuration is the one that previously OOMed because the old
//! `vip_single` materialised an `n × C(n-1,t)` Fp grid per pair × m=35.
//! After the weighted-residual refactor (see `vip::vip_single`) the peak
//! transient state is `O(m·N²)` plaintext field elements, so this cell
//! must complete in well under a few hundred MB of working memory.
//!
//! The test is `#[ignore]`'d by default — run with
//! `cargo test -p vdoprf-online --test memory_smoke -- --ignored --nocapture`
//! to exercise it. CI would otherwise spend ~10s per build on it.

use num_bigint::BigUint;
use vdoprf_field::Fp;
use vdoprf_offline::setup_pre_shared;
use vdoprf_online::{compute_batch::compute_batch, setup_random_preprocessed_m, OnlineProof};
use vdoprf_online::vip::{client_verify_dvoprf, VipResult};
use vdoprf_ss::SubsetFamily;

#[test]
#[ignore]
fn vip_compute_batch_at_9_4_m35_does_not_oom() {
    // Gold prime: same one used in the bench harness. p = 2^384 - 573·2^128 + 1.
    let p_hex = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffdc300000000000000000000000000000001";
    let modulus = BigUint::parse_bytes(p_hex.as_bytes(), 16).unwrap();
    let (n, t, m) = (9usize, 4usize, 35usize);
    let family = SubsetFamily::new(n, t);
    let pre_shared = setup_pre_shared(n, t, &modulus);
    let pre = setup_random_preprocessed_m(m, &family, &modulus);

    let mut rng = rand::thread_rng();
    let xs: Vec<Fp> = (0..m).map(|_| Fp::random(&modulus, &mut rng)).collect();

    let r = compute_batch(&xs, &pre, &pre_shared, &family, &modulus);
    assert_eq!(r.client_outputs.len(), m);

    let batched = match &r.proof {
        OnlineProof::Batched(b) => b,
        _ => panic!("compute_batch must emit OnlineProof::Batched"),
    };
    assert_eq!(
        client_verify_dvoprf(&batched.vip, &batched.tilde_v, &modulus),
        VipResult::Accept,
    );
}
