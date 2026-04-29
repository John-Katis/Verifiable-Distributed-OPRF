use crate::hash::hash_pair;

/// A Merkle tree over 32-byte leaves.
/// Stored as a complete binary tree in an array.
/// For n leaves, the tree has 2*next_pow2(n) nodes.
#[derive(Clone, Debug)]
pub struct MerkleTree {
    /// All nodes: index 1 = root, leaves start at `leaf_offset`.
    nodes: Vec<[u8; 32]>,
    /// Number of actual leaves.
    num_leaves: usize,
    /// Starting index of leaves in the nodes array.
    leaf_offset: usize,
}

impl MerkleTree {
    /// Build a Merkle tree from a list of leaves.
    pub fn new(leaves: Vec<[u8; 32]>) -> Self {
        let num_leaves = leaves.len();
        assert!(num_leaves > 0, "need at least one leaf");

        let capacity = num_leaves.next_power_of_two();
        let leaf_offset = capacity;
        let total = 2 * capacity;
        let mut nodes = vec![[0u8; 32]; total];

        // Fill leaves
        for (i, leaf) in leaves.into_iter().enumerate() {
            nodes[leaf_offset + i] = leaf;
        }

        // Build internal nodes bottom-up
        for i in (1..leaf_offset).rev() {
            nodes[i] = hash_pair(&nodes[2 * i], &nodes[2 * i + 1]);
        }

        MerkleTree {
            nodes,
            num_leaves,
            leaf_offset,
        }
    }

    /// Get the root hash.
    pub fn root(&self) -> [u8; 32] {
        self.nodes[1]
    }

    /// Get the authentication path for a leaf at the given index.
    /// Returns sibling hashes from leaf to root.
    pub fn authentication_path(&self, leaf_index: usize) -> Vec<[u8; 32]> {
        assert!(leaf_index < self.num_leaves);
        let mut path = Vec::new();
        let mut idx = self.leaf_offset + leaf_index;
        while idx > 1 {
            let sibling = idx ^ 1;
            path.push(self.nodes[sibling]);
            idx /= 2;
        }
        path
    }

    /// Verify an authentication path.
    pub fn verify_path(
        root: &[u8; 32],
        leaf: &[u8; 32],
        leaf_index: usize,
        num_leaves: usize,
        path: &[[u8; 32]],
    ) -> bool {
        let capacity = num_leaves.next_power_of_two();
        let mut idx = capacity + leaf_index;
        let mut current = *leaf;

        for sibling in path {
            current = if idx % 2 == 0 {
                hash_pair(&current, sibling)
            } else {
                hash_pair(sibling, &current)
            };
            idx /= 2;
        }

        current == *root
    }

    /// Get a leaf value.
    pub fn get_leaf(&self, index: usize) -> [u8; 32] {
        self.nodes[self.leaf_offset + index]
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::hash_bytes;

    #[test]
    fn test_merkle_tree_basic() {
        let leaves: Vec<[u8; 32]> = (0..4u8)
            .map(|i| hash_bytes(&[i]))
            .collect();
        let tree = MerkleTree::new(leaves.clone());

        // Verify all paths
        for i in 0..4 {
            let path = tree.authentication_path(i);
            assert!(MerkleTree::verify_path(
                &tree.root(),
                &leaves[i],
                i,
                4,
                &path
            ));
        }
    }

    #[test]
    fn test_merkle_tree_tamper_detection() {
        let leaves: Vec<[u8; 32]> = (0..4u8)
            .map(|i| hash_bytes(&[i]))
            .collect();
        let tree = MerkleTree::new(leaves.clone());

        let path = tree.authentication_path(0);
        let fake_leaf = hash_bytes(&[99]);
        assert!(!MerkleTree::verify_path(
            &tree.root(),
            &fake_leaf,
            0,
            4,
            &path
        ));
    }

    #[test]
    fn test_merkle_tree_non_power_of_two() {
        let leaves: Vec<[u8; 32]> = (0..3u8)
            .map(|i| hash_bytes(&[i]))
            .collect();
        let tree = MerkleTree::new(leaves.clone());

        for i in 0..3 {
            let path = tree.authentication_path(i);
            assert!(MerkleTree::verify_path(
                &tree.root(),
                &leaves[i],
                i,
                3,
                &path
            ));
        }
    }
}
