use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::Zero;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::BitOr;
use vdoprf_field::Fp;

/// A subset T ⊂ [n] represented as a `u64` bitmask (bit `p` set ⇔ `p ∈ T`).
/// Upper bound `n ≤ 64` is far beyond the honest-majority regime the protocol
/// targets (C(n,t) blows up long before bit-packing does), so this is safe.
///
/// Drop-in replacement for `BTreeSet<usize>` at `BTreeMap<SubsetT, Fp>` key
/// sites: Copy key, register-sized clone, lexicographic `Ord` matching
/// `BTreeSet<usize>::cmp` (iterate elements ascending, compare element-by-
/// element, shorter-prefix-is-less), same `contains(&usize)` surface.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Subset {
    bits: u64,
}

/// Alias kept for API compatibility with callers written against the older
/// `BTreeSet<usize>` representation. All call sites see `SubsetT` == [`Subset`].
pub type SubsetT = Subset;

impl Subset {
    /// Empty subset.
    pub const fn empty() -> Self {
        Subset { bits: 0 }
    }

    /// Raw bit representation. Useful for debugging; not part of the public
    /// set-algebra API.
    pub const fn bits(&self) -> u64 {
        self.bits
    }

    /// Membership test. Signature matches `BTreeSet::contains` so most call
    /// sites don't need to change.
    pub fn contains(&self, p: &usize) -> bool {
        debug_assert!(*p < 64, "party index must fit in 64-bit bitmask");
        (self.bits >> *p) & 1 == 1
    }

    /// Cardinality |T|.
    pub fn len(&self) -> usize {
        self.bits.count_ones() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Insert `p` into the subset. Returns `true` if the element was newly
    /// inserted (matches `BTreeSet::insert`).
    pub fn insert(&mut self, p: usize) -> bool {
        debug_assert!(p < 64, "party index must fit in 64-bit bitmask");
        let mask = 1u64 << p;
        let was_absent = (self.bits & mask) == 0;
        self.bits |= mask;
        was_absent
    }

    /// Remove `p` from the subset. Returns `true` if the element was present
    /// (matches `BTreeSet::remove`).
    pub fn remove(&mut self, p: &usize) -> bool {
        debug_assert!(*p < 64, "party index must fit in 64-bit bitmask");
        let mask = 1u64 << *p;
        let was_present = (self.bits & mask) != 0;
        self.bits &= !mask;
        was_present
    }

    /// Iterate elements in ascending order. Yields `usize` **by value**
    /// (unlike `BTreeSet::iter` which yields `&usize`) — since elements have
    /// no storage to borrow from, there is nothing to reference.
    pub fn iter(&self) -> SubsetIter {
        SubsetIter { bits: self.bits }
    }
}

impl BitOr for Subset {
    type Output = Subset;
    fn bitor(self, rhs: Self) -> Self::Output {
        Subset { bits: self.bits | rhs.bits }
    }
}

impl BitOr for &Subset {
    type Output = Subset;
    fn bitor(self, rhs: Self) -> Self::Output {
        Subset { bits: self.bits | rhs.bits }
    }
}

impl Ord for Subset {
    /// Lexicographic comparison matching `BTreeSet<usize>::cmp`: iterate
    /// elements in ascending order and compare element-by-element; the first
    /// differing element decides, and a shorter prefix compares less.
    ///
    /// NOT the same as raw `u64` comparison — for n=4 we have `{2}` (bits=4)
    /// `<` `{1,3}` (bits=10) under u64 ordering but `{2} > {1,3}` under the
    /// BTreeSet iterator rule (compare 2 > 1 at the first element). We
    /// preserve the BTreeSet ordering so iteration order over
    /// `BTreeMap<Subset, _>` matches what the codebase historically produced.
    fn cmp(&self, other: &Self) -> Ordering {
        let mut a = self.bits;
        let mut b = other.bits;
        loop {
            match (a == 0, b == 0) {
                (true, true) => return Ordering::Equal,
                (true, false) => return Ordering::Less,
                (false, true) => return Ordering::Greater,
                (false, false) => {
                    let xa = a.trailing_zeros();
                    let xb = b.trailing_zeros();
                    match xa.cmp(&xb) {
                        Ordering::Equal => {
                            a &= a - 1;
                            b &= b - 1;
                        }
                        neq => return neq,
                    }
                }
            }
        }
    }
}

impl PartialOrd for Subset {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl FromIterator<usize> for Subset {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut bits = 0u64;
        for p in iter {
            debug_assert!(p < 64, "party index must fit in 64-bit bitmask");
            bits |= 1u64 << p;
        }
        Subset { bits }
    }
}

impl<const N: usize> From<[usize; N]> for Subset {
    fn from(arr: [usize; N]) -> Self {
        arr.into_iter().collect()
    }
}

impl From<Vec<usize>> for Subset {
    fn from(v: Vec<usize>) -> Self {
        v.into_iter().collect()
    }
}

impl fmt::Debug for Subset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

/// Ascending iterator over the elements of a [`Subset`].
pub struct SubsetIter {
    bits: u64,
}

impl Iterator for SubsetIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.bits == 0 {
            None
        } else {
            let p = self.bits.trailing_zeros() as usize;
            self.bits &= self.bits - 1;
            Some(p)
        }
    }
}

/// The family T of all subsets of [n] of size t.
#[derive(Clone, Debug)]
pub struct SubsetFamily {
    pub n: usize,
    pub t: usize,
    /// All subsets of {0, ..., n-1} of size t, in a fixed canonical order.
    pub subsets: Vec<SubsetT>,
    /// Precomputed cross-product assignment per §4-Online "Share Assignments"
    /// (lines 41-64). For each party i, `assignment[i]` lists the pairs
    /// `(idx_t1, idx_t2)` (into `subsets`) that party i is assigned to compute
    /// locally, under the same round-robin rule
    /// `holders[pair_idx % holders.len()]` used historically by
    /// `cross_multiply`. Built once in `new()` so the cross-multiply hot path
    /// collapses from O(N²) (with per-pair eligibility check + BTreeSet union)
    /// to O(N²/n) (flat walk over precomputed pairs) per party.
    assignment: Vec<Vec<(usize, usize)>>,
}

impl SubsetFamily {
    pub fn new(n: usize, t: usize) -> Self {
        assert!(t < n, "t must be < n");
        assert!(2 * t < n, "need t < n/2 for honest majority");
        assert!(n <= 64, "SubsetT is a u64 bitmask; n must be ≤ 64");
        let subsets: Vec<SubsetT> = (0..n)
            .combinations(t)
            .map(|c| c.into_iter().collect::<SubsetT>())
            .collect();
        let assignment = Self::build_assignment(n, &subsets);
        SubsetFamily { n, t, subsets, assignment }
    }

    /// Build the per-party cross-product assignment table.
    ///
    /// Must match the historical assignment rule used in `cross_multiply` so
    /// that swapping to the precomputed table is a drop-in replacement: for
    /// every (idx1, idx2), compute `holders = [n] \ (T_{idx1} ∪ T_{idx2})`
    /// and assign the pair to `holders[pair_idx % holders.len()]` with
    /// `pair_idx = idx1 * N + idx2`. Paper §4-Online line 48 guarantees
    /// `|holders| ≥ 1`.
    fn build_assignment(n: usize, subsets: &[SubsetT]) -> Vec<Vec<(usize, usize)>> {
        let num_subsets = subsets.len();
        let mut assignment: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for (idx1, t1) in subsets.iter().enumerate() {
            for (idx2, t2) in subsets.iter().enumerate() {
                let union: SubsetT = t1 | t2;
                let holders: Vec<usize> = (0..n).filter(|p| !union.contains(p)).collect();
                if holders.is_empty() {
                    continue;
                }
                let pair_idx = idx1 * num_subsets + idx2;
                let assigned_to = holders[pair_idx % holders.len()];
                assignment[assigned_to].push((idx1, idx2));
            }
        }
        assignment
    }

    /// Precomputed cross-product assignment for party `i`. Each pair
    /// `(idx_t1, idx_t2)` indexes into `self.subsets`; the party is guaranteed
    /// to hold both components (i.e., `party ∉ T_{idx1} ∪ T_{idx2}`).
    pub fn assignment_for(&self, party: usize) -> &[(usize, usize)] {
        &self.assignment[party]
    }

    /// Number of subsets N = C(n, t).
    pub fn num_subsets(&self) -> usize {
        self.subsets.len()
    }

    /// Get the index of a subset in the canonical ordering.
    pub fn subset_index(&self, subset: &SubsetT) -> Option<usize> {
        self.subsets.iter().position(|s| s == subset)
    }

    /// Get subsets not containing party i (these are the shares party i holds).
    pub fn subsets_not_containing(&self, party: usize) -> Vec<&SubsetT> {
        self.subsets.iter().filter(|s| !s.contains(&party)).collect()
    }

    /// Get subsets containing party i.
    pub fn subsets_containing(&self, party: usize) -> Vec<&SubsetT> {
        self.subsets.iter().filter(|s| s.contains(&party)).collect()
    }
}

/// A full replicated secret sharing: the additive components indexed by subsets.
/// The secret is s = Σ_{T ∈ T} addss_T.
#[derive(Clone, Debug)]
pub struct ReplicatedSharing {
    pub components: BTreeMap<SubsetT, Fp>,
}

impl ReplicatedSharing {
    /// Reconstruct the secret by summing all additive components.
    pub fn reconstruct(&self, modulus: &BigUint) -> Fp {
        let mut sum = Fp::zero(modulus);
        for v in self.components.values() {
            sum = &sum + v;
        }
        sum
    }

    /// Reassemble a full sharing from per-party `RssShare`s.
    ///
    /// Each `RssShare` for party i carries `{addss_T : i ∉ T}`. The union of
    /// components across parties yields a complete `ReplicatedSharing`.
    /// First-write-wins: when multiple parties hold the same subset (as they
    /// must, under replicated sharing), the earliest party in iteration order
    /// provides the value. This matters for tamper-detection code paths where
    /// one party's share may disagree with the others — the caller sees the
    /// earliest-encountered value rather than silently averaging disagreement
    /// away.
    pub fn from_party_shares<'a, I: IntoIterator<Item = &'a RssShare>>(shares: I) -> Self {
        let mut components: BTreeMap<SubsetT, Fp> = BTreeMap::new();
        for rs in shares {
            for (t, v) in &rs.shares {
                components.entry(t.clone()).or_insert_with(|| v.clone());
            }
        }
        ReplicatedSharing { components }
    }

    /// Convenience: reassemble from per-party shares and reconstruct the secret.
    pub fn reconstruct_from_party_shares<'a, I: IntoIterator<Item = &'a RssShare>>(
        shares: I,
        modulus: &BigUint,
    ) -> Fp {
        Self::from_party_shares(shares).reconstruct(modulus)
    }

    /// Component-wise multiplication of every additive slot by a public scalar.
    /// Reconstructs to `scalar · self.reconstruct()`. Purely local.
    pub fn local_scalar_mul(&self, scalar: &Fp) -> ReplicatedSharing {
        ReplicatedSharing {
            components: self
                .components
                .iter()
                .map(|(t, v)| (t.clone(), v * scalar))
                .collect(),
        }
    }
}

/// RSS share for a single party i: contains addss_T for each T with i ∉ T.
#[derive(Clone, Debug)]
pub struct RssShare {
    pub party_id: usize,
    pub shares: BTreeMap<SubsetT, Fp>,
}

impl RssShare {
    /// Local addition of two RSS shares (component-wise).
    pub fn local_add(&self, other: &RssShare) -> RssShare {
        debug_assert_eq!(self.party_id, other.party_id);
        let shares: BTreeMap<SubsetT, Fp> = self
            .shares
            .iter()
            .map(|(t, v)| {
                let other_v = &other.shares[t];
                (t.clone(), v + other_v)
            })
            .collect();
        RssShare {
            party_id: self.party_id,
            shares,
        }
    }

    /// Local subtraction of two RSS shares.
    pub fn local_sub(&self, other: &RssShare) -> RssShare {
        debug_assert_eq!(self.party_id, other.party_id);
        let shares: BTreeMap<SubsetT, Fp> = self
            .shares
            .iter()
            .map(|(t, v)| {
                let other_v = &other.shares[t];
                (t.clone(), v - other_v)
            })
            .collect();
        RssShare {
            party_id: self.party_id,
            shares,
        }
    }

    /// Local scalar multiplication.
    pub fn local_scalar_mul(&self, scalar: &Fp) -> RssShare {
        let shares: BTreeMap<SubsetT, Fp> = self
            .shares
            .iter()
            .map(|(t, v)| (t.clone(), v * scalar))
            .collect();
        RssShare {
            party_id: self.party_id,
            shares,
        }
    }

    /// In-place `self += other`. Reuses `self.shares`' allocation; no new
    /// BTreeMap is built. The map shapes must match (same `party_id` ⇒ same
    /// set of subset keys).
    pub fn local_add_assign(&mut self, other: &RssShare) {
        debug_assert_eq!(self.party_id, other.party_id);
        for (t, v) in self.shares.iter_mut() {
            let ov = &other.shares[t];
            *v = &*v + ov;
        }
    }

    /// In-place `self -= other`.
    pub fn local_sub_assign(&mut self, other: &RssShare) {
        debug_assert_eq!(self.party_id, other.party_id);
        for (t, v) in self.shares.iter_mut() {
            let ov = &other.shares[t];
            *v = &*v - ov;
        }
    }

    /// In-place `self *= scalar`. No new BTreeMap; no key clones.
    pub fn local_scalar_mul_assign(&mut self, scalar: &Fp) {
        for v in self.shares.values_mut() {
            *v = &*v * scalar;
        }
    }

    /// Fused `self · self_scalar + other · other_scalar`, returning the new
    /// share. One BTreeMap allocation per call (vs three for the naive
    /// `self.local_scalar_mul(a).local_add(&other.local_scalar_mul(b))`).
    ///
    /// Matches the two-point Lagrange update in `vip.rs:277-294`:
    /// ⟨f_{2ℓ−1}(r_k)⟩ = (2 − r_k)·⟨a_lo⟩ + (r_k − 1)·⟨a_hi⟩.
    pub fn lagrange_combine(
        &self,
        self_scalar: &Fp,
        other: &RssShare,
        other_scalar: &Fp,
    ) -> RssShare {
        debug_assert_eq!(self.party_id, other.party_id);
        let shares: BTreeMap<SubsetT, Fp> = self
            .shares
            .iter()
            .map(|(t, v)| {
                let ov = &other.shares[t];
                let combined = &(v * self_scalar) + &(ov * other_scalar);
                (*t, combined)
            })
            .collect();
        RssShare {
            party_id: self.party_id,
            shares,
        }
    }
}

/// Generate a random RSS of a secret s.
pub fn share(
    secret: &Fp,
    family: &SubsetFamily,
    modulus: &BigUint,
    rng: &mut impl Rng,
) -> ReplicatedSharing {
    let n = family.num_subsets();
    let mut components = BTreeMap::new();

    // Generate n-1 random components, last one is determined by the secret.
    let mut sum = Fp::zero(modulus);
    for (i, subset) in family.subsets.iter().enumerate() {
        if i < n - 1 {
            let r = Fp::random(modulus, rng);
            sum = &sum + &r;
            components.insert(subset.clone(), r);
        } else {
            // Last component: s - sum of others
            let last = secret - &sum;
            components.insert(subset.clone(), last);
        }
    }

    ReplicatedSharing { components }
}

/// Extract the RSS share for party i from a full sharing.
pub fn get_party_share(sharing: &ReplicatedSharing, party: usize, family: &SubsetFamily) -> RssShare {
    let shares: BTreeMap<SubsetT, Fp> = family
        .subsets_not_containing(party)
        .into_iter()
        .map(|t| (t.clone(), sharing.components[t].clone()))
        .collect();
    RssShare {
        party_id: party,
        shares,
    }
}

/// Compute the cross-product of two RSS shares for party i,
/// producing party i's additive share of a*b.
///
/// The assignment rule distributes cross-products evenly:
/// For each pair (T1, T2), we assign it to the smallest-indexed party
/// that holds both components (i.e., not in T1 ∪ T2).
/// Party i then sums all cross-products assigned to it.
pub fn cross_multiply(
    a: &RssShare,
    b: &RssShare,
    family: &SubsetFamily,
) -> Fp {
    debug_assert_eq!(a.party_id, b.party_id);
    let party = a.party_id;
    let modulus = a.shares.values().next().unwrap().modulus();

    let mut sum = Fp::zero(modulus);

    // Walk the precomputed assignment table. By construction, every (idx1,
    // idx2) in `assignment_for(party)` satisfies `party ∉ T_{idx1} ∪ T_{idx2}`,
    // so `a.shares.get(t1).unwrap()` and `b.shares.get(t2).unwrap()` are
    // guaranteed Some — no per-pair eligibility check.
    for &(idx1, idx2) in family.assignment_for(party) {
        let t1 = &family.subsets[idx1];
        let t2 = &family.subsets[idx2];
        let a_val = &a.shares[t1];
        let b_val = &b.shares[t2];
        sum = &sum + &(a_val * b_val);
    }
    sum
}

/// Like cross_multiply, but also returns the individual (a_val, b_val) pairs
/// assigned to this party. Needed for DZKP where each party proves its
/// cross-product computation was honest.
pub fn cross_multiply_with_pairs(
    a: &RssShare,
    b: &RssShare,
    family: &SubsetFamily,
) -> (Fp, Vec<(Fp, Fp)>) {
    debug_assert_eq!(a.party_id, b.party_id);
    let party = a.party_id;
    let modulus = a.shares.values().next().unwrap().modulus();

    let assignment = family.assignment_for(party);
    let mut sum = Fp::zero(modulus);
    let mut pairs: Vec<(Fp, Fp)> = Vec::with_capacity(assignment.len());

    // Flat walk over precomputed pairs — no BTreeSet union, no per-pair
    // holders vec, no eligibility check. Boyle's RSS.Mul pays only the field
    // ops + Fp clones into the pair vector.
    for &(idx1, idx2) in assignment {
        let t1 = &family.subsets[idx1];
        let t2 = &family.subsets[idx2];
        let a_val = &a.shares[t1];
        let b_val = &b.shares[t2];
        sum = &sum + &(a_val * b_val);
        pairs.push((a_val.clone(), b_val.clone()));
    }
    (sum, pairs)
}

/// Variant of [`cross_multiply_with_pairs`] that additionally surfaces the
/// subset indices `(T_u, T_v)` of the additive-share components that each
/// emitted pair came from. This is what `Π_dVOPRF` §4-Online lines 235–236
/// and 245 need in order to perform the **local share conversion** of
/// Cramer–Damgård–Ishai (TCC'05) via [`DegenerateEncoding`]: given the pair
/// `(⟦a⟧_{T_u}, ⟦b⟧_{T_v})` the caller builds `⟨[a]_{T_u}⟩` and
/// `⟨[b]_{T_v}⟩` as degenerate sharings, which are then folded locally inside
/// Π_VIP alongside the plaintext pairs.
///
/// The emitted plaintext pairs match those of [`cross_multiply_with_pairs`]
/// in both order and multiplicity (both functions walk the same
/// `pair_idx % holders.len()` assignment).
pub fn cross_multiply_with_pairs_and_subsets(
    a: &RssShare,
    b: &RssShare,
    family: &SubsetFamily,
) -> (Fp, Vec<(Fp, Fp, SubsetT, SubsetT)>) {
    debug_assert_eq!(a.party_id, b.party_id);
    let party = a.party_id;
    let modulus = a.shares.values().next().unwrap().modulus();

    let assignment = family.assignment_for(party);
    let mut sum = Fp::zero(modulus);
    let mut pairs: Vec<(Fp, Fp, SubsetT, SubsetT)> = Vec::with_capacity(assignment.len());

    // Flat walk; VIP still pays the `t1.clone(), t2.clone()` because downstream
    // builds `DegenerateEncoding` sharings from them, but the per-pair
    // eligibility check and BTreeSet union are gone.
    for &(idx1, idx2) in assignment {
        let t1 = &family.subsets[idx1];
        let t2 = &family.subsets[idx2];
        let a_val = &a.shares[t1];
        let b_val = &b.shares[t2];
        sum = &sum + &(a_val * b_val);
        pairs.push((a_val.clone(), b_val.clone(), t1.clone(), t2.clone()));
    }
    (sum, pairs)
}

/// Covering policy P: T → [n] where P(T) ∉ T.
/// Returns the smallest index not in T.
pub fn covering_policy(subset: &SubsetT, n: usize) -> usize {
    for i in 0..n {
        if !subset.contains(&i) {
            return i;
        }
    }
    panic!("No party outside T, impossible with t < n");
}

/// Additive reconstruction: `s = Σ_i shares[i]`.
///
/// Paired with [`rss_to_additive`], this is the §4-Online `\tss` sharing —
/// one Fp per party, summed to recover the secret (paper `fig:doprf_protocol`
/// line 267: `C_i = Σ_k ε_k · v_i^{(k)}`, `c = Σ_i C_i`).
pub fn additive_reconstruct(shares: &[Fp], modulus: &BigUint) -> Fp {
    let mut sum = Fp::zero(modulus);
    for s in shares {
        sum = &sum + s;
    }
    sum
}

/// Convert per-party RSS shares into an additive sharing.
///
/// Each subset `T ∈ T` is credited to its canonical holder
/// `covering_policy(T, n)` (the smallest `i ∉ T`). Party `i` then sums all
/// subsets it owns canonically, yielding `Vec<Fp>` of length `n` with
/// `Σ_i result[i] = ReplicatedSharing::reconstruct_from_party_shares(&shares)`.
///
/// Assumes `shares` contains one entry per party — the caller's ordering
/// determines which party index each entry is bound to via `party_id`.
pub fn rss_to_additive(shares: &[RssShare], n: usize, modulus: &BigUint) -> Vec<Fp> {
    let mut out: Vec<Fp> = (0..n).map(|_| Fp::zero(modulus)).collect();
    for ps in shares {
        let i = ps.party_id;
        for (t, v) in &ps.shares {
            if covering_policy(t, n) == i {
                out[i] = &out[i] + v;
            }
        }
    }
    out
}

/// Lagrange basis coefficients `L_i(eval_at)` for interpolation abscissae
/// `xs`. Returns `Vec<Fp>` of length `xs.len()`. Depends only on the public
/// `xs` and `eval_at` — **independent of any RSS share content**.
///
/// Use this to hoist coefficient computation out of per-party loops:
/// coefficients are the same for every party evaluating the same polynomial
/// at the same point, so one computation suffices for all n parties.
pub fn lagrange_coeffs(xs: &[Fp], eval_at: &Fp, modulus: &BigUint) -> Vec<Fp> {
    let n = xs.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // L_i(x) = Π_{j≠i} (x − x_j) / (x_i − x_j)
        let mut numer = Fp::new(BigUint::from(1u32), modulus);
        let mut denom = Fp::new(BigUint::from(1u32), modulus);
        for j in 0..n {
            if j != i {
                numer = &numer * &(eval_at - &xs[j]);
                denom = &denom * &(&xs[i] - &xs[j]);
            }
        }
        out.push(&numer * &denom.inv().expect("Lagrange denominator must be non-zero"));
    }
    out
}

/// Evaluate a share-valued polynomial at a point using **precomputed**
/// Lagrange coefficients. `shares.len() == coeffs.len()` and all `shares`
/// must be for the same party.
///
/// Fused into a single pass over the BTreeMap components: one `BTreeMap`
/// allocation per call (vs one `local_scalar_mul` per point + one
/// `local_add` per join = k allocations for k points). This is what the
/// per-party Lagrange loops in `vip.rs` want.
pub fn lagrange_eval_with_coeffs(shares: &[&RssShare], coeffs: &[Fp]) -> RssShare {
    assert_eq!(shares.len(), coeffs.len(), "shares and coeffs must agree in length");
    assert!(!shares.is_empty(), "need at least one share");
    let party_id = shares[0].party_id;
    let shares_out: BTreeMap<SubsetT, Fp> = shares[0]
        .shares
        .iter()
        .map(|(t, v0)| {
            let mut acc = v0 * &coeffs[0];
            for (s, c) in shares[1..].iter().zip(coeffs[1..].iter()) {
                debug_assert_eq!(s.party_id, party_id);
                acc = &acc + &(&s.shares[t] * c);
            }
            (*t, acc)
        })
        .collect();
    RssShare { party_id, shares: shares_out }
}

/// Lagrange interpolation over RSS shares.
///
/// Given points `[(x_0, share_0), ..., (x_k, share_k)]` where x_i are public
/// field elements and share_i are RSS shares (all for the same party), compute
/// the RSS share of `y(eval_at)` where y is the unique polynomial through these points.
///
/// This works because Lagrange interpolation is a linear operation:
///   y(x) = Σ_i  L_i(x) · y_i
/// where L_i(x) are the Lagrange basis polynomials (public scalars).
/// Since RSS shares are linear, we compute L_i(eval_at) and combine via
/// `local_scalar_mul` and `local_add`.
///
/// For **per-party loops** that reuse the same xs + eval_at across parties,
/// prefer hoisting coefficients via [`lagrange_coeffs`] and looping over
/// [`lagrange_eval_with_coeffs`] — the basis is then computed once, not n
/// times.
pub fn lagrange_eval_on_shares(
    points: &[(Fp, RssShare)],
    eval_at: &Fp,
    modulus: &BigUint,
) -> RssShare {
    assert!(!points.is_empty(), "need at least one point");
    let xs: Vec<Fp> = points.iter().map(|(x, _)| x.clone()).collect();
    let coeffs = lagrange_coeffs(&xs, eval_at, modulus);
    let shares: Vec<&RssShare> = points.iter().map(|(_, s)| s).collect();
    lagrange_eval_with_coeffs(&shares, &coeffs)
}

/// Degenerate additive encoding `⟨v⟩_{T'}` (Definition 3.a of the paper):
/// a replicated sharing where only parties in the complement of `target_subset`
/// hold a non-zero share of the value `v`.
#[derive(Clone, Debug)]
pub struct DegenerateEncoding {
    pub target_subset: SubsetT,
    pub value: Fp,
}

impl DegenerateEncoding {
    /// Active set `A = [n] \ target_subset` — the parties that participate in
    /// `Π_DegMul`. Ordered by party index.
    pub fn active_set(&self, n: usize) -> Vec<usize> {
        (0..n).filter(|p| !self.target_subset.contains(p)).collect()
    }

    /// Materialise this degenerate encoding as a full `ReplicatedSharing`
    /// following Definition 3.a of the paper: `addss_{T'} = value` and
    /// `addss_T = 0` for all other `T`. Reconstructing the resulting sharing
    /// yields `value`.
    pub fn to_replicated_sharing(
        &self,
        family: &SubsetFamily,
        modulus: &BigUint,
    ) -> ReplicatedSharing {
        let mut components: BTreeMap<SubsetT, Fp> = BTreeMap::new();
        for subset in &family.subsets {
            let v = if subset == &self.target_subset {
                self.value.clone()
            } else {
                Fp::zero(modulus)
            };
            components.insert(subset.clone(), v);
        }
        ReplicatedSharing { components }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    fn modulus() -> BigUint {
        BigUint::from(113u32)
    }

    #[test]
    fn test_subset_family_3_1() {
        let family = SubsetFamily::new(3, 1);
        assert_eq!(family.num_subsets(), 3);
        // Subsets: {0}, {1}, {2}
        assert_eq!(family.subsets[0], Subset::from([0]));
        assert_eq!(family.subsets[1], Subset::from([1]));
        assert_eq!(family.subsets[2], Subset::from([2]));
    }

    #[test]
    fn test_subsets_not_containing() {
        let family = SubsetFamily::new(3, 1);
        // Party 0 doesn't hold T={0}, holds T={1} and T={2}
        let subs = family.subsets_not_containing(0);
        assert_eq!(subs.len(), 2);
        assert!(subs.contains(&&Subset::from([1])));
        assert!(subs.contains(&&Subset::from([2])));
    }

    #[test]
    fn test_share_reconstruct() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(42u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        let reconstructed = sharing.reconstruct(&p);
        assert_eq!(reconstructed.value, secret.value);
    }

    #[test]
    fn test_party_shares() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(42u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);

        // Each party holds N - t = 2 shares for n=3, t=1
        for i in 0..3 {
            let ps = get_party_share(&sharing, i, &family);
            assert_eq!(ps.shares.len(), 2);
            assert_eq!(ps.party_id, i);
        }
    }

    #[test]
    fn test_local_add() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let mut rng = rand::thread_rng();

        let a = Fp::new(BigUint::from(20u32), &p);
        let b = Fp::new(BigUint::from(30u32), &p);
        let expected = &a + &b;

        let sa = share(&a, &family, &p, &mut rng);
        let sb = share(&b, &family, &p, &mut rng);

        // Each party locally adds
        for i in 0..3 {
            let _ = get_party_share(&sa, i, &family)
                .local_add(&get_party_share(&sb, i, &family));
        }

        // Reconstruct the sum by manually adding components
        let mut sum_sharing = BTreeMap::new();
        for t in &family.subsets {
            let v = &sa.components[t] + &sb.components[t];
            sum_sharing.insert(t.clone(), v);
        }
        let rs = ReplicatedSharing { components: sum_sharing };
        assert_eq!(rs.reconstruct(&p).value, expected.value);
    }

    #[test]
    fn test_cross_multiply() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let mut rng = rand::thread_rng();

        let a = Fp::new(BigUint::from(7u32), &p);
        let b = Fp::new(BigUint::from(11u32), &p);
        let expected = &a * &b; // 77

        let sa = share(&a, &family, &p, &mut rng);
        let sb = share(&b, &family, &p, &mut rng);

        // Each party computes its additive share of a*b
        let mut sum = Fp::zero(&p);
        for i in 0..3 {
            let pa = get_party_share(&sa, i, &family);
            let pb = get_party_share(&sb, i, &family);
            let share_i = cross_multiply(&pa, &pb, &family);
            sum = &sum + &share_i;
        }

        assert_eq!(sum.value, expected.value);
    }

    #[test]
    fn test_covering_policy() {
        assert_eq!(covering_policy(&Subset::from([0]), 3), 1);
        assert_eq!(covering_policy(&Subset::from([1]), 3), 0);
        assert_eq!(covering_policy(&Subset::from([2]), 3), 0);
    }

    #[test]
    fn test_from_party_shares_round_trip() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(42u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        let party_shares: Vec<RssShare> =
            (0..3).map(|i| get_party_share(&sharing, i, &family)).collect();

        let reassembled = ReplicatedSharing::from_party_shares(&party_shares);
        assert_eq!(reassembled.reconstruct(&p).value, secret.value);
    }

    #[test]
    fn test_reconstruct_from_party_shares_matches_direct() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(99u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        let party_shares: Vec<RssShare> =
            (0..3).map(|i| get_party_share(&sharing, i, &family)).collect();

        let via_helper = ReplicatedSharing::reconstruct_from_party_shares(&party_shares, &p);
        let direct = sharing.reconstruct(&p);
        assert_eq!(via_helper.value, direct.value);
    }

    #[test]
    fn test_from_party_shares_single_party_incomplete() {
        // For (n=3, t=1) each party holds 2 of the 3 components.
        // Reassembling from just one party must leave 1 component missing.
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(7u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        let one_party = get_party_share(&sharing, 0, &family);

        let partial = ReplicatedSharing::from_party_shares(std::slice::from_ref(&one_party));
        assert_eq!(partial.components.len(), 2);
        assert_eq!(family.subsets.len(), 3);
    }

    #[test]
    fn test_from_party_shares_full_coverage() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(5u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        let party_shares: Vec<RssShare> =
            (0..3).map(|i| get_party_share(&sharing, i, &family)).collect();

        let reassembled = ReplicatedSharing::from_party_shares(&party_shares);
        assert_eq!(reassembled.components.len(), family.subsets.len());
        for subset in &family.subsets {
            assert!(reassembled.components.contains_key(subset));
        }
    }

    #[test]
    fn test_reconstruct_from_party_shares_linearity() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let mut rng = rand::thread_rng();

        let a = Fp::new(BigUint::from(20u32), &p);
        let b = Fp::new(BigUint::from(30u32), &p);
        let expected = &a + &b;

        let sa = share(&a, &family, &p, &mut rng);
        let sb = share(&b, &family, &p, &mut rng);
        let summed: Vec<RssShare> = (0..3)
            .map(|i| get_party_share(&sa, i, &family).local_add(&get_party_share(&sb, i, &family)))
            .collect();

        let reconstructed = ReplicatedSharing::reconstruct_from_party_shares(&summed, &p);
        assert_eq!(reconstructed.value, expected.value);
    }

    #[test]
    fn test_from_party_shares_first_write_wins() {
        // Replicated sharing: multiple parties hold each subset T. If one
        // party's share disagrees (tampered), the earliest party in
        // iteration order should provide the value — so a tampered party 0
        // is visible to callers that reconstruct, instead of being silently
        // overwritten by the honest majority.
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(50u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        let mut party_shares: Vec<RssShare> =
            (0..3).map(|i| get_party_share(&sharing, i, &family)).collect();

        // Tamper party 0's value for the first subset it holds.
        let first_key = party_shares[0].shares.keys().next().unwrap().clone();
        let orig = party_shares[0].shares.get(&first_key).unwrap().clone();
        let tampered = &orig + &Fp::new(BigUint::from(1u32), &p);
        party_shares[0].shares.insert(first_key.clone(), tampered.clone());

        let reassembled = ReplicatedSharing::from_party_shares(&party_shares);
        assert_eq!(reassembled.components[&first_key].value, tampered.value);
    }

    // --- RSS share consistency, assignment correctness, and tampering ---

    #[test]
    fn test_rss_share_consistency_across_parties() {
        // For replicated sharing, multiple parties hold every subset T (all
        // parties with i ∉ T). They must agree on the value for T.
        let p = modulus();
        let family = SubsetFamily::new(5, 2); // N=10 subsets, each held by 3 parties
        let secret = Fp::new(BigUint::from(88u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        let party_shares: Vec<RssShare> =
            (0..family.n).map(|i| get_party_share(&sharing, i, &family)).collect();

        for subset in &family.subsets {
            let mut reported: Option<&Fp> = None;
            for (i, ps) in party_shares.iter().enumerate() {
                if subset.contains(&i) {
                    continue; // party i doesn't hold this subset
                }
                let val = ps.shares.get(subset).expect("party should hold this subset");
                match reported {
                    None => reported = Some(val),
                    Some(prev) => assert_eq!(prev.value, val.value,
                        "parties disagreed on subset {:?}", subset),
                }
            }
            assert!(reported.is_some(), "no party held subset {:?}", subset);
        }
    }

    #[test]
    fn test_rss_share_assignment_correct() {
        // Party i holds `addss_T` iff i ∉ T. Test the iff rather than just
        // the count.
        let p = modulus();
        let family = SubsetFamily::new(5, 2);
        let secret = Fp::new(BigUint::from(33u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        for i in 0..family.n {
            let ps = get_party_share(&sharing, i, &family);
            for subset in &family.subsets {
                let should_hold = !subset.contains(&i);
                let does_hold = ps.shares.contains_key(subset);
                assert_eq!(does_hold, should_hold,
                    "party {} should_hold={} subset {:?} but does_hold={}",
                    i, should_hold, subset, does_hold);
            }
        }
    }

    #[test]
    fn test_rss_tampered_share_changes_reconstruction() {
        // Tampering one party's share value for some subset T must change the
        // reconstructed secret — otherwise tamper detection downstream (DZKP,
        // client_verify) would be meaningless.
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(42u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        let mut party_shares: Vec<RssShare> =
            (0..family.n).map(|i| get_party_share(&sharing, i, &family)).collect();

        // Tamper party 0's value for the first subset it holds.
        let first_key = party_shares[0].shares.keys().next().unwrap().clone();
        let orig = party_shares[0].shares.get(&first_key).unwrap().clone();
        let tampered = &orig + &Fp::new(BigUint::from(1u32), &p);
        party_shares[0].shares.insert(first_key, tampered);

        // With first-write-wins reassembly, party 0's tampered value wins.
        let reconstructed =
            ReplicatedSharing::reconstruct_from_party_shares(&party_shares, &p);
        assert_ne!(reconstructed.value, secret.value,
            "tampered share should produce a different reconstruction");
    }

    // --- DegenerateEncoding matches the paper's definition ---

    #[test]
    fn test_degenerate_encoding_reconstructs_to_value() {
        // Per Definition 3.a of the paper, ⟨v⟩_{T'} is a replicated sharing
        // whose secret is v.
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let v = Fp::new(BigUint::from(55u32), &p);
        let enc = DegenerateEncoding {
            target_subset: family.subsets[1].clone(),
            value: v.clone(),
        };
        let rss = enc.to_replicated_sharing(&family, &p);
        assert_eq!(rss.reconstruct(&p).value, v.value);
    }

    #[test]
    fn test_degenerate_encoding_only_target_component_nonzero() {
        // Per the paper: addss_{T'} = v, and addss_T = 0 for every other T.
        let p = modulus();
        let family = SubsetFamily::new(5, 2);
        let v = Fp::new(BigUint::from(17u32), &p);
        let target = family.subsets[3].clone();
        let enc = DegenerateEncoding { target_subset: target.clone(), value: v.clone() };
        let rss = enc.to_replicated_sharing(&family, &p);

        for subset in &family.subsets {
            let stored = rss.components.get(subset).expect("component should exist");
            if subset == &target {
                assert_eq!(stored.value, v.value, "target T' must hold value v");
            } else {
                assert!(stored.value.bits() == 0, "non-target T must hold 0, got {:?}", stored.value);
            }
        }
    }

    #[test]
    fn test_degenerate_encoding_party_view_matches_paper() {
        // The paper writes s_T = v if i ∉ T, 0 otherwise. In the RSS view this
        // means: party i holds the T'-component iff i ∉ T', and when held the
        // value is v. (Parties in T' do not hold the T'-component at all.)
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let target = family.subsets[1].clone(); // {1}
        let v = Fp::new(BigUint::from(9u32), &p);
        let enc = DegenerateEncoding { target_subset: target.clone(), value: v.clone() };
        let rss = enc.to_replicated_sharing(&family, &p);

        for i in 0..family.n {
            let ps = get_party_share(&rss, i, &family);
            if target.contains(&i) {
                assert!(!ps.shares.contains_key(&target),
                    "party {} is in target T' and must not hold the T'-component", i);
            } else {
                let held = ps.shares.get(&target).expect("party must hold T'-component");
                assert_eq!(held.value, v.value,
                    "party {} (i ∉ T') must hold the value v for T'", i);
            }
        }
    }

    #[test]
    fn test_degenerate_encoding_active_set_excludes_target() {
        // ⟨v⟩_{T'} with T' = {1} and n = 3 → active set = {0, 2}.
        let p = modulus();
        let enc = DegenerateEncoding {
            target_subset: Subset::from([1]),
            value: Fp::new(BigUint::from(7u32), &p),
        };
        assert_eq!(enc.active_set(3), vec![0, 2]);
    }

    #[test]
    fn test_degenerate_encoding_active_set_full_when_target_empty() {
        let p = modulus();
        let enc = DegenerateEncoding {
            target_subset: Subset::empty(),
            value: Fp::new(BigUint::from(1u32), &p),
        };
        assert_eq!(enc.active_set(4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_degenerate_encoding_active_set_ordered() {
        // active_set must be sorted ascending so `active_set[0]` is a stable
        // aggregator choice in Π_DegMul.
        let p = modulus();
        let enc = DegenerateEncoding {
            target_subset: Subset::from([0, 3]),
            value: Fp::new(BigUint::from(9u32), &p),
        };
        let active = enc.active_set(5);
        assert_eq!(active, vec![1, 2, 4]);
        assert!(active.windows(2).all(|w| w[0] < w[1]));
    }

    // --- additive_reconstruct / rss_to_additive (§4-Online `\tss` sharing) ---

    /// `additive_reconstruct` is just a sum over the input Fp slice.
    #[test]
    fn test_additive_reconstruct_sums_shares() {
        let p = modulus();
        let a = Fp::new(BigUint::from(7u32), &p);
        let b = Fp::new(BigUint::from(11u32), &p);
        let c = Fp::new(BigUint::from(13u32), &p);
        let expected = &(&a + &b) + &c;
        let shares = vec![a, b, c];
        assert_eq!(additive_reconstruct(&shares, &p).value, expected.value);
    }

    #[test]
    fn test_additive_reconstruct_empty_is_zero() {
        let p = modulus();
        assert!(additive_reconstruct(&[], &p).is_zero());
    }

    /// Round-trip invariant: `Σ_i rss_to_additive(shares)[i]` equals the secret
    /// reconstructed by `reconstruct_from_party_shares`. This is the core
    /// guarantee that lets `VipParallelOutput` ship `Vec<Fp>` instead of
    /// `Vec<RssShare>` without losing correctness.
    #[test]
    fn test_rss_to_additive_round_trip_3_1() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(42u32), &p);
        let mut rng = rand::thread_rng();
        let sharing = share(&secret, &family, &p, &mut rng);
        let party_shares: Vec<RssShare> =
            (0..family.n).map(|i| get_party_share(&sharing, i, &family)).collect();

        let add_shares = rss_to_additive(&party_shares, family.n, &p);
        assert_eq!(add_shares.len(), family.n);
        assert_eq!(additive_reconstruct(&add_shares, &p).value, secret.value);
    }

    #[test]
    fn test_rss_to_additive_round_trip_5_2() {
        let p = modulus();
        let family = SubsetFamily::new(5, 2);
        let secret = Fp::new(BigUint::from(88u32), &p);
        let mut rng = rand::thread_rng();
        let sharing = share(&secret, &family, &p, &mut rng);
        let party_shares: Vec<RssShare> =
            (0..family.n).map(|i| get_party_share(&sharing, i, &family)).collect();

        let add_shares = rss_to_additive(&party_shares, family.n, &p);
        assert_eq!(add_shares.len(), family.n);
        assert_eq!(additive_reconstruct(&add_shares, &p).value, secret.value);
    }

    /// Canonical holder covers every subset exactly once across all parties.
    /// Together with `test_rss_to_additive_round_trip_*`, this proves the
    /// conversion has no double-counting and no missing components.
    #[test]
    fn test_rss_to_additive_matches_first_write_wins() {
        let p = modulus();
        let family = SubsetFamily::new(5, 2);
        let secret = Fp::new(BigUint::from(77u32), &p);
        let mut rng = rand::thread_rng();
        let sharing = share(&secret, &family, &p, &mut rng);
        let party_shares: Vec<RssShare> =
            (0..family.n).map(|i| get_party_share(&sharing, i, &family)).collect();

        let via_additive =
            additive_reconstruct(&rss_to_additive(&party_shares, family.n, &p), &p);
        let via_rss =
            ReplicatedSharing::reconstruct_from_party_shares(&party_shares, &p);
        assert_eq!(via_additive.value, via_rss.value);
    }

    /// Linearity: the additive shares of `a + b` equal the pointwise sum of
    /// the additive shares of `a` and `b`. Follows from RSS linearity plus
    /// covering-policy stability.
    #[test]
    fn test_rss_to_additive_is_linear() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let mut rng = rand::thread_rng();
        let a_secret = Fp::new(BigUint::from(20u32), &p);
        let b_secret = Fp::new(BigUint::from(30u32), &p);

        let sa = share(&a_secret, &family, &p, &mut rng);
        let sb = share(&b_secret, &family, &p, &mut rng);
        let a_shares: Vec<RssShare> =
            (0..family.n).map(|i| get_party_share(&sa, i, &family)).collect();
        let b_shares: Vec<RssShare> =
            (0..family.n).map(|i| get_party_share(&sb, i, &family)).collect();
        let sum_shares: Vec<RssShare> = (0..family.n)
            .map(|i| a_shares[i].local_add(&b_shares[i]))
            .collect();

        let a_add = rss_to_additive(&a_shares, family.n, &p);
        let b_add = rss_to_additive(&b_shares, family.n, &p);
        let sum_add = rss_to_additive(&sum_shares, family.n, &p);

        for i in 0..family.n {
            assert_eq!(sum_add[i].value, (&a_add[i] + &b_add[i]).value);
        }
    }

    /// Tampering one RSS component changes the additive share held by exactly
    /// one party — and hence changes the reconstructed secret. This is the
    /// property the VIP client-verify tamper-detection tests rely on.
    #[test]
    fn test_rss_to_additive_tamper_propagates() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(42u32), &p);
        let mut rng = rand::thread_rng();
        let sharing = share(&secret, &family, &p, &mut rng);
        let mut party_shares: Vec<RssShare> =
            (0..family.n).map(|i| get_party_share(&sharing, i, &family)).collect();

        let key = party_shares[0].shares.keys().next().unwrap().clone();
        let orig = party_shares[0].shares.get(&key).unwrap().clone();
        party_shares[0]
            .shares
            .insert(key, &orig + &Fp::new(BigUint::from(1u32), &p));

        let add = rss_to_additive(&party_shares, family.n, &p);
        assert_ne!(additive_reconstruct(&add, &p).value, secret.value);
    }

    // --- cross_multiply_with_pairs_and_subsets (TCC'05 local share conversion) ---

    /// The subset-augmented variant emits the same plaintext pairs, in the
    /// same order, as the non-augmented one — essential for callers that
    /// previously relied on the index ℓ in `pairs[ℓ]` being stable.
    #[test]
    fn test_cross_multiply_with_pairs_and_subsets_matches_pairs_only() {
        let p = modulus();
        let family = SubsetFamily::new(5, 2);
        let mut rng = rand::thread_rng();
        let a = Fp::new(BigUint::from(41u32), &p);
        let b = Fp::new(BigUint::from(53u32), &p);
        let sa = share(&a, &family, &p, &mut rng);
        let sb = share(&b, &family, &p, &mut rng);

        for i in 0..family.n {
            let pa = get_party_share(&sa, i, &family);
            let pb = get_party_share(&sb, i, &family);
            let (sum_plain, pairs_plain) = cross_multiply_with_pairs(&pa, &pb, &family);
            let (sum_full, pairs_full) =
                cross_multiply_with_pairs_and_subsets(&pa, &pb, &family);
            assert_eq!(sum_plain.value, sum_full.value);
            assert_eq!(pairs_plain.len(), pairs_full.len());
            for (p0, p1) in pairs_plain.iter().zip(pairs_full.iter()) {
                assert_eq!(p0.0.value, p1.0.value);
                assert_eq!(p0.1.value, p1.1.value);
            }
        }
    }

    /// For each emitted `(a_val, b_val, T_u, T_v)` the degenerate encoding at
    /// subset `T_u` reconstructs to `a_val` — i.e. the subset indices name the
    /// correct additive-share slot, so callers can safely build
    /// `⟨[a]_{T_u}⟩` via `DegenerateEncoding::to_replicated_sharing`.
    #[test]
    fn test_cross_multiply_subsets_identify_addss_slots() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let mut rng = rand::thread_rng();
        let a = Fp::new(BigUint::from(7u32), &p);
        let b = Fp::new(BigUint::from(11u32), &p);
        let sa = share(&a, &family, &p, &mut rng);
        let sb = share(&b, &family, &p, &mut rng);

        for i in 0..family.n {
            let pa = get_party_share(&sa, i, &family);
            let pb = get_party_share(&sb, i, &family);
            let (_, pairs) = cross_multiply_with_pairs_and_subsets(&pa, &pb, &family);
            for (a_val, b_val, t_u, t_v) in pairs {
                let da = DegenerateEncoding { target_subset: t_u.clone(), value: a_val.clone() }
                    .to_replicated_sharing(&family, &p);
                let db = DegenerateEncoding { target_subset: t_v.clone(), value: b_val.clone() }
                    .to_replicated_sharing(&family, &p);
                assert_eq!(da.reconstruct(&p).value, a_val.value);
                assert_eq!(db.reconstruct(&p).value, b_val.value);
                assert_eq!(sa.components[&t_u].value, a_val.value);
                assert_eq!(sb.components[&t_v].value, b_val.value);
            }
        }
    }

    // --- ReplicatedSharing::local_scalar_mul ---

    /// `(scalar · sharing).reconstruct() = scalar · sharing.reconstruct()`.
    #[test]
    fn test_replicated_sharing_local_scalar_mul() {
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let mut rng = rand::thread_rng();
        let s = Fp::new(BigUint::from(17u32), &p);
        let k = Fp::new(BigUint::from(5u32), &p);
        let sharing = share(&s, &family, &p, &mut rng);
        let scaled = sharing.local_scalar_mul(&k);
        assert_eq!(scaled.reconstruct(&p).value, (&s * &k).value);
    }

    #[test]
    fn test_from_party_shares_accepts_ref_iterator() {
        // Exercises the IntoIterator<Item = &RssShare> bound via a map adapter
        // (the dzkp.rs call sites use this shape to extract a field from a
        // struct of shares).
        let p = modulus();
        let family = SubsetFamily::new(3, 1);
        let secret = Fp::new(BigUint::from(11u32), &p);
        let mut rng = rand::thread_rng();

        let sharing = share(&secret, &family, &p, &mut rng);
        let party_shares: Vec<RssShare> =
            (0..3).map(|i| get_party_share(&sharing, i, &family)).collect();

        let reconstructed = ReplicatedSharing::reconstruct_from_party_shares(
            party_shares.iter(),
            &p,
        );
        assert_eq!(reconstructed.value, secret.value);
    }
}
