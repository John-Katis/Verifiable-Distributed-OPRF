//! Cross-primitive smoke test: GGM → per-leaf hash → Merkle commit →
//! transcript challenge → authentication-path verify.
//!
//! This composition mirrors the shape of the `zkp_ligero` / `zkp_vith`
//! proving paths (leaves are derived from a GGM tree, columns are
//! Merkle-committed, the verifier samples an index via Fiat-Shamir and
//! opens the path). It's deliberately minimal — just enough to trip if
//! hash domain separation, GGM leaf ordering, or transcript framing
//! changes incompatibly.

use vdoprf_crypto::ggm::GgmTree;
use vdoprf_crypto::hash::hash_bytes;
use vdoprf_crypto::merkle::MerkleTree;
use vdoprf_crypto::transcript::Transcript;

#[test]
fn ggm_merkle_transcript_roundtrip() {
    let arity = 8usize;

    // 1. Expand a GGM tree; hash each leaf seed to get a 32-byte column digest.
    let root = [7u8; 16];
    let tree = GgmTree::expand(root, arity);
    assert_eq!(tree.leaf_seeds.len(), arity);
    let column_digests: Vec<[u8; 32]> = tree
        .leaf_seeds
        .iter()
        .map(|s| hash_bytes(s))
        .collect();

    // 2. Merkle-commit the column digests.
    let merkle = MerkleTree::new(column_digests.clone());
    let root_digest = merkle.root();

    // 3. Feed the root into a Fiat-Shamir transcript and derive an index in [0, arity).
    let mut fs = Transcript::new(b"ggm.merkle.integration");
    fs.append_commitment(&root_digest);
    let idx = fs.challenge_index(arity);
    assert!(idx < arity);

    // 4. Open the authentication path for that index and verify.
    let path = merkle.authentication_path(idx);
    // Balanced tree over 8 leaves → depth 3.
    assert_eq!(path.len(), 3);
    assert!(MerkleTree::verify_path(
        &root_digest,
        &column_digests[idx],
        idx,
        arity,
        &path,
    ));

    // 5. Negative check: a wrong leaf must not verify against the same path.
    let wrong_leaf = hash_bytes(b"not a real column");
    assert!(!MerkleTree::verify_path(
        &root_digest,
        &wrong_leaf,
        idx,
        arity,
        &path,
    ));
}
