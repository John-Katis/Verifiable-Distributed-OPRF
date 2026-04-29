use crate::prg::PairwisePrg;

/// GGM (Goldreich-Goldwasser-Micali) expansion tree.
/// Expands a root seed into `arity` leaf seeds using a binary tree of PRG expansions.
///
/// For arity τ, the tree has depth ⌈log₂ τ⌉.
/// Each internal node with seed s produces two children:
///   left  = PRG(s, 0)
///   right = PRG(s, 1)
#[derive(Clone, Debug)]
pub struct GgmTree {
    pub arity: usize,
    pub depth: usize,
    pub root_seed: [u8; 16],
    pub leaf_seeds: Vec<[u8; 16]>,
}

impl GgmTree {
    /// Expand a root seed into `arity` leaf seeds.
    pub fn expand(root_seed: [u8; 16], arity: usize) -> Self {
        assert!(arity >= 2, "arity must be at least 2");
        let depth = (arity as f64).log2().ceil() as usize;
        let capacity = 1 << depth; // next power of 2 >= arity

        // Level-by-level expansion
        let mut current_level = vec![root_seed];

        for _ in 0..depth {
            let mut next_level = Vec::with_capacity(current_level.len() * 2);
            for seed in &current_level {
                let prg = PairwisePrg::new(*seed);
                let left = prg.expand_seed(0);
                let right = prg.expand_seed(1);
                next_level.push(left);
                next_level.push(right);
            }
            current_level = next_level;
        }

        // Take only the first `arity` leaves (if arity is not a power of 2)
        let leaf_seeds: Vec<[u8; 16]> = current_level.into_iter().take(arity).collect();

        GgmTree {
            arity,
            depth,
            root_seed,
            leaf_seeds,
        }
    }

    /// Get a specific leaf seed.
    pub fn get_leaf_seed(&self, index: usize) -> [u8; 16] {
        self.leaf_seeds[index]
    }

    /// Get the co-path for hiding leaf `hidden_leaf`.
    /// Returns the sibling seeds along the path from root to the hidden leaf,
    /// which allow reconstructing all other leaf seeds.
    pub fn copath(&self, hidden_leaf: usize) -> Vec<[u8; 16]> {
        assert!(hidden_leaf < self.arity);
        let capacity = 1 << self.depth;

        // Build the full tree to extract siblings
        let mut tree = vec![[0u8; 16]; 2 * capacity];
        tree[1] = self.root_seed;

        // Expand all internal nodes
        for i in 1..capacity {
            let prg = PairwisePrg::new(tree[i]);
            tree[2 * i] = prg.expand_seed(0);
            tree[2 * i + 1] = prg.expand_seed(1);
        }

        // Walk from hidden leaf to root, collecting siblings
        let mut path = Vec::new();
        let mut idx = capacity + hidden_leaf;
        while idx > 1 {
            let sibling = idx ^ 1;
            path.push(tree[sibling]);
            idx /= 2;
        }

        path
    }

    /// Reconstruct all leaf seeds except the hidden one from a co-path.
    /// Returns (index, seed) pairs for all revealed leaves.
    pub fn reconstruct_except(
        copath: &[[u8; 16]],
        hidden_leaf: usize,
        arity: usize,
    ) -> Vec<(usize, [u8; 16])> {
        let depth = copath.len();
        let capacity = 1 << depth;

        // Reconstruct the tree from co-path siblings
        let mut tree = vec![None; 2 * capacity];

        // Place siblings
        let mut idx = capacity + hidden_leaf;
        for (level, sibling_seed) in copath.iter().enumerate() {
            let sibling = idx ^ 1;
            tree[sibling] = Some(*sibling_seed);
            idx /= 2;
            let _ = level;
        }

        // Expand from top down
        for i in 1..capacity {
            if let Some(seed) = tree[i] {
                let prg = PairwisePrg::new(seed);
                let left = prg.expand_seed(0);
                let right = prg.expand_seed(1);
                // Only set children if they aren't already set (don't overwrite siblings)
                if tree[2 * i].is_none() {
                    tree[2 * i] = Some(left);
                }
                if tree[2 * i + 1].is_none() {
                    tree[2 * i + 1] = Some(right);
                }
            }
        }

        // Collect leaf seeds (except the hidden one)
        let mut result = Vec::new();
        for i in 0..arity {
            if i != hidden_leaf {
                if let Some(seed) = tree[capacity + i] {
                    result.push((i, seed));
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggm_expand() {
        let root = [42u8; 16];
        let tree = GgmTree::expand(root, 4);
        assert_eq!(tree.leaf_seeds.len(), 4);
        assert_eq!(tree.depth, 2);
        // All leaves should be different
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(tree.leaf_seeds[i], tree.leaf_seeds[j]);
            }
        }
    }

    #[test]
    fn test_ggm_deterministic() {
        let root = [42u8; 16];
        let tree1 = GgmTree::expand(root, 4);
        let tree2 = GgmTree::expand(root, 4);
        assert_eq!(tree1.leaf_seeds, tree2.leaf_seeds);
    }

    #[test]
    fn test_ggm_copath_reconstruct() {
        let root = [42u8; 16];
        let tree = GgmTree::expand(root, 4);

        for hidden in 0..4 {
            let copath = tree.copath(hidden);
            let reconstructed = GgmTree::reconstruct_except(&copath, hidden, 4);

            // Should reconstruct all leaves except the hidden one
            assert_eq!(reconstructed.len(), 3);

            for (idx, seed) in &reconstructed {
                assert_ne!(*idx, hidden);
                assert_eq!(*seed, tree.leaf_seeds[*idx]);
            }

            // Hiding property: the hidden leaf's index is not in the output,
            // and no reconstructed seed equals the hidden leaf's seed.
            assert!(reconstructed.iter().all(|(i, _)| *i != hidden));
            assert!(reconstructed.iter().all(|(_, s)| *s != tree.leaf_seeds[hidden]));
        }
    }

    #[test]
    fn test_ggm_non_power_of_two() {
        let root = [1u8; 16];
        let tree = GgmTree::expand(root, 3);
        assert_eq!(tree.leaf_seeds.len(), 3);
        assert_eq!(tree.depth, 2); // ceil(log2(3)) = 2
    }

    #[test]
    fn test_ggm_large_arity() {
        let root = [7u8; 16];
        let tree = GgmTree::expand(root, 16);
        assert_eq!(tree.leaf_seeds.len(), 16);
        assert_eq!(tree.depth, 4);

        // Verify copath works for all positions
        for hidden in 0..16 {
            let copath = tree.copath(hidden);
            assert_eq!(copath.len(), 4); // depth = 4
            let reconstructed = GgmTree::reconstruct_except(&copath, hidden, 16);
            assert_eq!(reconstructed.len(), 15);
            for (idx, seed) in &reconstructed {
                assert_eq!(*seed, tree.leaf_seeds[*idx]);
            }
        }
    }
}
