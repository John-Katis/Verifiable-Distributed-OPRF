pub mod double_rand;
pub mod rss_mul;
pub mod rss_share;
pub mod pub_base_exp;
pub mod approach_i;
pub mod approach_ii;
pub mod dzkp;
pub mod approach_iii;
pub mod zkp_vith;
pub mod zkp_ligero;

use num_bigint::BigUint;
use std::collections::BTreeMap;
use vdoprf_crypto::prg::PairwisePrg;
use vdoprf_crypto::prg::ReplicatedPrf;
use vdoprf_ss::SubsetT;

/// Pre-shared material for a server party.
/// Includes pairwise PRG seeds and replicated PRF keys.
#[derive(Clone, Debug)]
pub struct PreSharedMaterial {
    pub party_id: usize,
    /// Pairwise PRG seeds: prg_seeds[(i,j)] for i < j.
    /// Both parties i and j hold the same seed.
    pub prg_seeds: BTreeMap<(usize, usize), PairwisePrg>,
    /// Replicated PRF keys: prf_keys[T] for T not containing this party.
    pub prf_keys: BTreeMap<SubsetT, ReplicatedPrf>,
}

/// Server state during protocol execution.
#[derive(Clone, Debug)]
pub struct ServerState {
    pub party_id: usize,
    pub n: usize,
    pub t: usize,
    pub modulus: BigUint,
    pub pre_shared: PreSharedMaterial,
}

/// Set up pre-shared material for all n parties.
/// In a real system this would use a trusted dealer or MPC setup.
pub fn setup_pre_shared(
    n: usize,
    t: usize,
    modulus: &BigUint,
) -> Vec<PreSharedMaterial> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let family = vdoprf_ss::SubsetFamily::new(n, t);

    // Generate pairwise PRG seeds
    let mut all_prg_seeds: BTreeMap<(usize, usize), PairwisePrg> = BTreeMap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut seed = [0u8; 16];
            rng.fill(&mut seed);
            all_prg_seeds.insert((i, j), PairwisePrg::new(seed));
        }
    }

    // Generate replicated PRF keys (one per subset T)
    let mut all_prf_keys: BTreeMap<SubsetT, ReplicatedPrf> = BTreeMap::new();
    for subset in &family.subsets {
        let mut key = [0u8; 16];
        rng.fill(&mut key);
        all_prf_keys.insert(subset.clone(), ReplicatedPrf::new(key));
    }

    // Distribute to parties
    let mut parties = Vec::new();
    for i in 0..n {
        // Party i gets PRG seeds for all pairs involving i
        let mut my_prg_seeds = BTreeMap::new();
        for j in 0..n {
            if j == i {
                continue;
            }
            let key = if i < j { (i, j) } else { (j, i) };
            my_prg_seeds.insert(key, all_prg_seeds[&key].clone());
        }

        // Party i gets PRF keys for all subsets T not containing i
        let mut my_prf_keys = BTreeMap::new();
        for subset in &family.subsets {
            if !subset.contains(&i) {
                my_prf_keys.insert(subset.clone(), all_prf_keys[subset].clone());
            }
        }

        parties.push(PreSharedMaterial {
            party_id: i,
            prg_seeds: my_prg_seeds,
            prf_keys: my_prf_keys,
        });
    }

    parties
}

/// Create server states from pre-shared material.
pub fn create_servers(
    n: usize,
    t: usize,
    modulus: &BigUint,
    pre_shared: Vec<PreSharedMaterial>,
) -> Vec<ServerState> {
    pre_shared
        .into_iter()
        .map(|ps| ServerState {
            party_id: ps.party_id,
            n,
            t,
            modulus: modulus.clone(),
            pre_shared: ps,
        })
        .collect()
}
