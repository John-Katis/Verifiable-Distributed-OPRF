# Changelog

This log groups changes by the finding/task each cluster of edits addresses, rather than by commit or file, so the *why* behind each group of changes stays clear without re-reading every diff.

## Group A — Online: genuine F_coin realization for r_k / ε_sigma / ε'_batch / ρ

**Files:** `crates/online/src/vip.rs`, `crates/online/src/compute.rs`, `crates/online/src/compute_parallel.rs`, `crates/online/src/compute_batch.rs`, `crates/offline/src/rss_share.rs`

**What was wrong.** In `vip_parallel`, the γ-round Fiat-Shamir sequence `r_ks` was derived by appending the *same constant label* (`b"r_k"`) every loop iteration and calling `Transcript::challenge()` — which does not mutate the hasher — so every value in the sequence came out bit-identical, not merely unbound from committed prover data. Separately, the σ-update coefficients (`ε_2..ε_γ`, inside `vip_single`) and the Σ-batch coefficients (`ε'_i`, in `vip_parallel`) were sampled from `rand::thread_rng()` — private, per-call randomness that a corrupt party could effectively choose, and that wasn't even shared across the `n` parallel `Π_VIP` instances the way the protocol requires.

**The fix.** Added a `coin_toss` helper in `vip.rs` that realizes a genuine, un-biasable F_coin: every party independently derives its own PRF-keyed RSS share of the same secret value (no communication needed — deterministic given a shared counter and `pre_shared`), then the value is opened via `ReplicatedSharing::reconstruct_from_party_shares`. This exactly mirrors the `generate_double_sharing` pattern already used correctly in this file for the `(a₀, b₀)` output-randomization values — it just wasn't applied to `r_k`/`ε`/`ρ`. All of `vip_parallel`'s F_coin calls (`r_k` per round, `ε_2.. ε_γ`, `ε'_i`, `ρ`) now go through `coin_toss`, drawn up front (a genuine coin toss has no ordering dependency on committed data, unlike Fiat-Shamir, so there's no need to interleave rounds). `vip_single` gained one new parameter, `eps_sigma: &[Fp]`, for the caller-supplied shared σ-update coefficients. The now-unused `Transcript` parameter was removed from `vip_parallel`'s signature. Call sites in `compute.rs`, `compute_parallel.rs`, and `compute_batch.rs` were updated accordingly.

**Round-count follow-up.** The fold/batch coefficients `r_k`/`ε_sigma`/`ε'` are independent of each other (unlike a Fiat-Shamir sequence) and none is a challenge *point* for a committed polynomial, so they can be requested and opened together in one round. Added `charge_f_coin_batch` (`crates/offline/src/rss_share.rs`) so `vip_parallel` charges the `r_k`/`ε_sigma`/`ε'` draw as a single batched round, matching Appendix M's own round accounting ("`FCoin` to generate `ε'_1,...,ε'_n` can be simultaneously called with `FCoin` at the last step of the single VIP protocol"). Added `vip_parallel_round_count_independent_of_gamma` as a regression test: `comm.rounds` must not grow with γ = ⌈log L⌉, only the byte count should.

## Group B — Online: compute_boyle.rs must return Abort, not panic

**Files:** `crates/online/src/compute_boyle.rs`, `crates/bench/src/main.rs`

**What was wrong.** `compute_boyle_single`/`compute_boyle_batch` called `assert_eq!(verdict, DzkpResult::Accept, ...)` after running the DZKP — a real Fvrfy failure (`DzkpResult::Abort`) would panic the process instead of being reported as a controlled verdict.

**The fix.** Removed both `assert_eq!` panics. The functions now return `BoyleProof`/`BoyleBatchProof` with whatever verdict `dzkp_compute_batch` actually produced, mirroring the mutable-verdict-accumulator pattern already used correctly in `approach_iii.rs`. `client_deliver_boyle` in `bench/src/main.rs` was updated to check `proof.verdict` before delivering `open_shares` to the client — on `Abort` it now sends only the verdict-announcement bytes and returns no outputs, instead of unconditionally treating (possibly meaningless) shares as valid.

## Group C — Offline: VitH secret GGM seed + two-level Merkle input binding

**Files:** `crates/offline/src/zkp_vith.rs`, `crates/offline/src/approach_iii.rs`

**What was wrong (two findings, one mechanism).** The GGM tree's root seed was derived via `transcript.challenge(...)` — a pure function of public data (the witness commitment, δ, and the repetition index) — instead of the prover's own secret randomness. Anyone, including the verifier, could recompute the entire GGM tree, including the "hidden" leaf, directly inverting the masked witness to recover the real witness (breaking zero-knowledge) and forging proofs for false witnesses (breaking soundness), since the only check on the free values `(Ã₀, Ã₁)` was a single linear equation solvable given a public correlation. Separately, the paper's headline contribution — a two-level GGM/Merkle commitment binding the proof to each verifier's own locally-held `(m_T, a_T)` shares — was entirely absent: `VitHProof` had no commitment-root field, `vith_verify`'s `_local_shares` parameter was unused dead code, and `approach_iii.rs` fed it an always-empty map plus a zero-filled placeholder for the "authentication path" data.

**The fix**, following the paper's Protocols 15-16 (Section 4.2.2) exactly, reusing primitives already in the codebase (`MerkleTree`, `hash_commitment`, `Transcript`) rather than inventing new crypto machinery:
- The GGM root seed (`sd`) is now generated via `rand::thread_rng()` — the prover's own secret randomness, never derived from the transcript.
- For each of the τ GGM leaves, a sub-Merkle-tree is built over that leaf's expanded share vector (`leaf_commitment_tree`), giving a per-leaf root `h_i`. An outer Merkle tree over `{h_i}` gives the commitment root `rt`. `rt` is committed into the transcript *before* the masked witness and before the hidden index `Δ` is drawn (previously nothing meaningful was committed before `Δ`).
- `vith_verify` now checks that `rt` genuinely commits to the real leaf structure: it recomputes `h_i` for every non-hidden leaf (whose seed it can now reconstruct from the existing GGM co-path) and takes the prover-supplied `h_Δ` for the hidden leaf, then verifies the resulting Merkle root matches `rt`.
- `vith_verify`'s `local_shares` parameter is now used and retyped to `BTreeMap<usize, Fp>` (witness position → locally-known value). For every position present, it recovers the hidden leaf's committed share at that position from the masked witness and the verifier's own local value, and checks it against `h_Δ` via a sub-tree Merkle authentication path. This is the actual "dual-share consistency" check: a dealer whose broadcast proof disagrees with what a verifier already knows locally is now caught, independent of the QuickSilver check.
- `approach_iii.rs` no longer passes an empty map / zero-filled bytes: it builds the real `local_shares` map for one simulated verifier by re-deriving `(m_T, a_T)` from that verifier's own PRF-keyed view (reusing the dealer's own derivation logic), and sends the genuine sub-tree authentication path bytes for those positions instead of a placeholder.

## Group D — Shared: hash_field_elements length-framing

**Files:** `crates/crypto-primitives/src/hash.rs`

**What was wrong.** `hash_field_elements` concatenated each field element's unpadded, variable-length big-endian bytes with no length prefix or delimiter, so two structurally different element vectors could hash identically (e.g. `[Fp(1), Fp(2)]` and `[Fp(0x0102)]` both concatenate to `01 02`). `Transcript::append_field_element` already handled this correctly elsewhere in the codebase with a 4-byte length prefix per element — this function just hadn't been brought in line.

**The fix.** Added the same 4-byte big-endian length prefix per element. Confirmed safe across all 9 call sites (each calls the function identically on both the producing and verifying side within the same protocol. No test hardcodes raw digest bytes. The two cross-file byte-identity requirements — `compute_batch.rs` ↔ `vip.rs` — both go through this same function, so they stay mutually consistent automatically).

## Group E — Offline: Ligero verifier panic + query-position dedup

**Files:** `crates/offline/src/zkp_ligero.rs`

**What was wrong.** `ligero_verify` validated the lengths of four of its proof vectors before use, but not `opened_columns_j`/`opened_columns_next`/`merkle_paths_j`/`merkle_paths_next` — a malformed or adversarial proof with fewer entries than `query_positions` would panic with an out-of-bounds index instead of being cleanly rejected (an availability/DoS-class bug). Separately, the query-position sampling loop (in both `ligero_prove` and `ligero_verify`) silently dropped a duplicate sampled position instead of resampling, so `query_positions` could end up shorter than `params.num_queries` — slightly under-sampling below the nominal soundness parameter.

**The fix.** Added the missing length checks before the indexing loop, returning `false` on mismatch. Changed both sampling loops (kept structurally identical to each other, as required for prove/verify to derive the same positions) to resample on a duplicate instead of dropping it, so `query_positions.len()` always equals exactly `params.num_queries`.

## Group F — Online: implement Π_Input, verifiable client input sharing

**Files:** `crates/online/src/input.rs` (new), `crates/online/src/lib.rs`, `crates/online/src/compute_batch.rs`, `crates/secret-sharing/src/lib.rs`, `crates/offline/src/rss_share.rs`, `crates/bench/src/main.rs`

**What was wrong.** `share_add_and_extract_pairs` (used by `compute`, `compute_parallel`, and `compute_batch`) treats the client as an ad-hoc local dealer: it just calls `vdoprf_ss::share(x, ...)` directly, with no consistency check at all. The paper (Appendix E.1) calls this out explicitly: "a malicious client can send mismatched copies of a replicated component `[x]_ℓ` to two servers that both hold it, and nothing detects the inconsistency."

**The fix.** Implemented `Π_Input` in a new `crates/online/src/input.rs` module:
- `client_input_share` — steps 1-3 (mask-and-open with an abort-checked `ψ_i` hash) and step 5 (each server locally derives `[x^(j)]_i = u^(j) − [r^(j)]_i`). The `P:[N]→[n]` designated-sender assignment reuses the existing `covering_policy` (already used for this exact role by `generate_double_sharing`/`rss_to_additive`). The pre-computed `[r^(j)]` reuses `generate_rss_random` (the same PRF-derived, no-communication primitive this session's `vip.rs` `coin_toss` fix already relies on).
- `client_input_share_standalone` — the *full* protocol at the paper's stated standalone 3-round cost, composing the above with `verify_echo_standalone` (step 4's echo, checked as its own round).
- Added `RssShare::local_sub_public` (public-scalar subtraction on an RSS share) for step 5 to have *every* holder of the canonical secret share subset apply the identical adjustment — caught by `compute_batch_with_verified_input_matches_compute_batch`, which failed loudly (reconstructed value mismatch) the first time it ran.
- Added `charge_input_client_facing`/`charge_input_echo_standalone` to `rss_share.rs` for `Π_Input`'s real communication cost (the existing `charge_client_rss_share` approximation is left untouched — nothing here removes it).

**Rounds Bench:** The new `Π_Input` protocol adds +3 rounds in full or +2 when a broadcast message is folded into an existing communication round. In our benchmarks, we use only the 2 rounds version. For our benchmarks, this only adds +1 round in the end because the old code version already counted the client input as one round. So this new adjustement, replacing the old, unverified client input with `Π_Input`, replaces a one round protocol with a two round one, adding only +1 rounds.

## Group G — Bench: standalone/parameterized experiment interfaces + Legendre-dOPRF baseline wrapper

**Files:** `crates/bench/src/main.rs`, `crates/bench/src/avg.rs`, `run_bench.sh`, `run_legendre_baseline.sh` (new)

**What was missing.** The bench crate only supported "run everything, exactly as hardcoded", there was no way to run a single experiment, no CLI parameter handling at all. Separately, a fourth baseline — the external Legendre-dOPRF C/network implementation (Kaluđerović et al., ESORICS'25), vendored as the `d-OPRF` git submodule — was only ever *documented* and never actually runnable alongside the Rust harness.

**The fix.** Added a hand-rolled `std::env::args()` CLI to `crates/bench/src/main.rs` (no new dependency, matching the crate's existing style): an `Experiment` enum (`all` / `offline` / `online` / `e2e` / `our-protocol` / `our-protocol-verified-input` / `naive-boyle-aly`), `--n N --t T` (both-or-neither) to override the (n,t) sweep to a single pair, and `--m M1,M2,...` to override the m sweep — all defaulting to today's values, so `Experiment::All` with no flags reproduces the historical unconditional three-section run byte-for-byte (confirmed: rounds/bytes for `our-protocol` vs `our-protocol-verified-input` differ only in the expected small `Π_Input`-cost increment — see Group F — with `I: AlyGen + BoyleBatch` identical between them, as it must be).

For detailed usage instructions of the CLI, please visit see [README](README.md)


## Group H — Bench: verified-input Boyle+AlyGen, merged e2e table, dOPRF in `run_bench.sh`

**Files:** `crates/online/src/compute_boyle.rs`, `crates/bench/src/main.rs`, `run_bench.sh`

The naive Boyle+AlyGen baseline is now wired through the same Π_Input-verified client input as our own protocol, everywhere it's benchmarked (Section 2's `Boyle-Batch` row, the `naive-boyle-aly` command, and the e2e Aly-Boyle baseline row). All benchmark numbers reported in the final paper reflect this.

## Group I — Legendre-dOPRF baseline: genuine 384-bit field infrastructure

**Files:** `d-OPRF/Legendre-dOPRF-network/parameters.h`, d-OPRF/Legendre-dOPRF-network/Makefile`, d-OPRF/Legendre-dOPRF-network/p384/` (new), d-OPRF/Legendre-dOPRF-network/network-version/server.c`, `run_legendre_baseline.sh`

**What was wrong.** The vendored Legendre-dOPRF code (Kaluđerović et al., ESORICS'25) only ever shipped with 64/128/192/256/**512**-bit fields, i.e. `NBYTES_FIELD ∈ {8,16,24,32,64}`. There was no 384-bit field, which is the width the v-dOPRF paper's §Evaluation pins for *both* PRFs given their specific security requirements (`|p| = 3λ = 384`, so "all benchmarks share the same 𝔽_p"). The originally-submitted Legendre numbers were therefore produced by an **ad-hoc 384-bit configuration that ran on the 512-bit infrastructure**: the element *counts* and the Legendre-symbol batch (`LAMBDA = NBITS_FIELD/2 = 192`) were the true 384-bit values, but each 𝔽_p element was represented and billed at the 512-bit width — 64 bytes, no reduction to a tight 384-bit encoding ("supported, but no wrapping mod 384").

**The fix.** Added a real 384-bit field to the dOPRF implementation:
- `SEC_LEVEL=5` → `NBITS_FIELD=384`, `NBYTES_FIELD=48` (`parameters.h`), a new `p384/generic/arith_generic.c`, and `client384`/`server384` Makefile targets. The modulus is this repo's own Gold-PRF prime `p = 2^384 − 573·2^128 + 1`, so Legendre now runs over *exactly* the same 𝔽_p as the rest of the evaluation, at a matched λ = 128 post-quantum level **over the correct infrastructure**.
- `run_legendre_baseline.sh` builds and drives `client384`/`server384` (`SEC_LEVEL=5`). `FE = 48` bytes throughout, consistent with the `FE = 48` used for our own protocol's numbers.
- Offline-cost instrumentation in `server.c` (submodule `8966290`) uses `sizeof()` on the actual field structures, so it now reflects the 48-byte width automatically.

**Effect on the reported numbers.** Every 𝔽_p element in the Legendre transcript shrinks 64 → 48 bytes, so **all Legendre communication figures drop to exactly 3/4 (48/64) of the submitted values**. Computation and round counts are unchanged (identical element counts, identical `LAMBDA`, same arithmetic). Verified by reconstruction — `submitted_comm ≈ recode_comm × 4/3` at both `(t,n)` points, both `m`:

| point | originally submitted (KB) | newly recoded (KB) | ratio |
|---|---|---|---|
| (4,1) m=1   | 4 425.22      | 3 381.0       | 1.309 |
| (7,2) m=1   | 1 761 188.35  | 1 323 617.9   | 1.331 |
| (4,1) m=100 | 442 522.00    | 338 100.0     | 1.309 |
| (7,2) m=100 | 176 118 835   | 132 361 785.9 | 1.331 |

The `(7,2)` point (communication is almost entirely field elements) lands on 4/3 to within 0.2 %, `(4,1)` sits ~1.8 % below, the difference being fixed-size framing (party indices, subset representations, TCP/serialization headers) that does not scale with element width. As a cross-check, running the *actual* 512-bit binary instead gives `(1,4) m=1 = 5996 KB ≈ recode × (4/3)²` — because in the real code `NBYTES_FIELD` *and* `LAMBDA` both scale with bit-width.

**Not a regression.** The corrected figures are the faithful instantiation of the `|p| = 384` claim the paper already makes. The originally submitted communication was inflated by 4/3. Despite that, no comparison in §Evaluation changes — Legendre still loses by orders of magnitude on communication (≈ 1.3 GB at `t = 2`, ≈ 132 GB at `m = 100`).

## Group J — Offline: VitH soundness fix (δ binding, challenge ordering, repetition count) + shared per-verifier/verdict-echo fix

**Files:** `crates/offline/src/zkp_vith.rs`, `crates/offline/src/approach_iii.rs`

**What was wrong (five findings, two shared with Ligero).** `δ` was never checked by a gate — the multiplication chain only constrained the running-product wires, and `δ` entered solely through the Fiat-Shamir hash, so a prover could start the chain from any value that hits a false `δ`. `Δ` (the hidden GGM leaf) was derived from the transcript *before* the QuickSilver check values `(Ã₀, Ã₁)` were appended, and challenges were hashed per repetition rather than jointly — together letting a prover forge `(Ã₀, Ã₁)` post hoc and grind one repetition at a time. The repetition count used `R = ⌈κ / log₂τ⌉`, treating the per-repetition soundness error as `1/τ`, when the QuickSilver check is degree 2 in `Δ` and a cheating prover can plant both roots in `[τ]` (error `2/τ`): for `τ=16, κ=40` the code computed `R=10` against the paper's own stated `R=14` — the implementation had drifted from the paper. Even after that formula is fixed, `R` was still being sized off the statistical parameter `κ=40` rather than the computational `λ=128`, and since `Δ` is a hash-derived challenge over only `τ=16` values, a prover could re-randomize its commitment and re-hash to grind it offline; an unused `commitment = H(witness)` field was also transcript-appended and wire-cost-counted despite never being independently re-verifiable. Finally, `gen_zkp` (`approach_iii.rs`) only ever re-verified *one* representative non-dealer party per proof, even though the module's design intends every server to independently re-run the public verifier — most non-dealer parties' checks were silently never executed — and there was no verdict-echo round, so a corrupted dealer could send inconsistent per-verifier material and split honest servers into different verdicts; this last pair of issues is shared with Ligero (Group K).

**The fix.** The last gate's output is now the linear expression `Σ_k a_k + δ`, committed as `Σ_k Commit(a_k) + δ·Δ` with VOLE mask `Σ_k v_{a_k}` (committed wires: `3N → 3N − 2`). The Fiat-Shamir transcript is now one joint sequence over all repetitions — `rt, w̃ → χ_ρ → (Ã₀,Ã₁) → Δ_ρ` — with co-paths opened only after `Δ_ρ` is fixed. `VitHParams::new` now computes `R = ⌈λ / (log₂τ − 1)⌉` off a renamed `lambda` field (call sites switched from `KAPPA` to the pre-existing `LAMBDA=128` constant), so `τ=16, λ=128 → R=43`; the dead `commitment` field is removed. `approach_iii.rs` now loops over every non-dealer verifier (not just the first), giving each its own independently-derived local shares and its own genuine P2P payload (VitH's authentication path, or Ligero's `q_j` binding polynomial — Group K), and adds one verdict-echo round where servers broadcast accept/reject and abort on disagreement — shared by all dealers in a `gen_zkp` call, regardless of `(n,t)` or `m`.

**Effect on the reported numbers.** Re-measured every `Π_ZKPGen^VitH` cell (offline table) and `Π_dVOPRF^VitH` cell (e2e table) across all four `(n,t)` and both `m`, combining the circuit fix, the `λ`-based `R`, and the per-verifier/verdict-echo change into one final set of numbers. Round counts go up by exactly 1 everywhere — only the verdict echo touches round count; the circuit and `R` fixes don't. Total communication grows 2.4–4.2×, driven by `R: 10→43` and by every non-dealer party now genuinely receiving its own P2P payload instead of one shared stand-in. Computation grows far more, dominated by `vith_verify`'s GGM-leaf reconstruction now being re-run once per non-dealer party (`n−1` times) instead of once:

| point | old Total | old Rnds | final Total | final Rnds | Total ratio | final Comp (ms) |
|---|---|---|---|---|---|---|
| offline (3,1) m=1   | 27.92 KB       | 10 | 77.55 KB        | 11 | 2.777 | 29.2      |
| offline (9,4) m=100 | 766 822.57 KB  | 14 | 3 240 809.36 KB | 15 | 4.226 | 598 505.5 |
| e2e (3,1) m=1        | 35.56 KB       | 19 | 85.19 KB        | 20 | 2.396 | 30.1      |
| e2e (9,4) m=100      | 769 264.36 KB  | 23 | 3 243 251.21 KB | 24 | 4.216 | 616 425.4 |

Full per-`(n,t)`, per-`m` breakdown of the final numbers (all Total in KB, Comp in ms):

| (n,t) | m | offline Total | offline Rnds | offline Comp | e2e Total | e2e Rnds | e2e Comp |
|---|---|---|---|---|---|---|---|
| (3,1) | 1   | 77.55        | 11 | 29.2      | 85.19        | 20 | 30.1      |
| (3,1) | 100 | 6 763.92     | 11 | 1 896.2   | 6 827.11     | 20 | 1 934.5   |
| (5,2) | 1   | 766.17       | 13 | 198.8     | 820.11       | 22 | 208.5     |
| (5,2) | 100 | 61 216.18    | 13 | 16 500.2  | 61 414.09    | 22 | 16 768.2  |
| (7,3) | 1   | 5 916.80     | 13 | 1 121.1   | 6 181.37     | 22 | 1 183.2   |
| (7,3) | 100 | 439 915.21   | 13 | 102 036.1 | 440 564.05   | 22 | 103 534.4 |
| (9,4) | 1   | 43 534.42    | 15 | 6 375.1   | 44 784.29    | 24 | 6 738.1   |
| (9,4) | 100 | 3 240 809.36 | 15 | 598 505.5 | 3 243 251.21 | 24 | 616 425.4 |

AlyGen, DegGen, naive-dVOPRF and the Legendre baseline are untouched by these fixes and were not re-run. Ligero is affected by the same `approach_iii.rs` fix — see Group K.

## Group K — Offline: Ligero redesign and the shared per-verifier/verdict-echo fix

**Files:** `crates/offline/src/zkp_ligero.rs`, `crates/offline/src/approach_iii.rs`, `crates/bench/src/main.rs`

**What was wrong (four findings).** `Π_Lig` at the prior commit had no zero-knowledge padding: each data row was exactly the degree-`(N−1)` interpolant of its `N` witness values, so a single opened extension column was a fixed public linear combination of all `N` values — enough for a colluding coalition to recover the one share pair only the honest dealer knew. The soundness bound only accounted for the constraint test, omitted the proximity test entirely, and didn't match the degree bounds a fix would need. Query points were sampled from a small range without first committing to per-verifier binding-polynomial digests, and sized for the statistical `κ=40` rather than the computational `λ=128`, so a prover could re-randomize its commitment and re-hash to steer the sample. And the one binding check that did exist, `ligero_verify_partial_opening`, checked a claimed share value's Merkle membership at a witness position entirely independently of the codeword/constraint checks (which only ever sampled extension positions) — confirmed dead code, never actually called from `approach_iii.rs` (which sent a zero-filled placeholder instead) — so nothing tied a server's shares to the polynomial the dealer actually committed to; a proof for `delta+1` verified while every server's own check still passed.

**The fix.** A near-total redesign implementing the "balanced layout" protocol: `B` instances are grouped into segments of `c`, each segment holding four length-`L=cN` rows (`M`, `a`, running product, running sum) interpolated over a subgroup `H` with **fresh uniform random padding** on the unused positions of `H` (opened columns are now provably uniform, so zero knowledge no longer limits how many points can be sampled). One mask row `Z_j = Z_W·g_j` is added per verifier plus one shared blinding row, replacing the old two-level share/private Merkle split with a flat single-level leaf. The constraint system (`h1..h5`, routed through `Z_E`/`Z_{W\S}`/`Z_{W\E}`) and the corrected two-term soundness bound (proximity test + constraint/binding-identity test) drive a new layout-search (`try_new_layout`) over both the domain size `k'` and the query count `q`. Every party gets a private **binding polynomial** `q_j`, sent P2P only (never broadcast — broadcasting it would let any recipient evaluate it at a position it doesn't hold and learn a linear relation on the hidden witness); its digest `h_j` is committed and appended to the transcript **before** query points are sampled (sampling now depends on the binding-polynomial digests, not just the row/composition commitments). Verification splits into `ligero_verify` (the shared proximity/constraint/consistency checks every party runs identically) and a new `ligero_verify_as_party` (every party checks its own `q_j` against the shared opened columns *and* against its own locally-known shares at the positions it actually holds — the missing link that ties a server's input to the codewords the dealer committed to). `ligero_verify_partial_opening` is deleted. `approach_iii.rs` now builds `verifier_positions` for every party (not one fixed representative) and sends every party its own real `q_j` over P2P; the shared per-verifier and verdict-echo fix described in Group J applies here too — `ligero_verify_as_party` is now run for every non-dealer party, not one representative, plus the same one-round verdict echo.

**Effect on the reported numbers.** Re-measured every `Π_ZKPGen^Lig` cell (offline table) and `Π_dVOPRF^Lig` cell (e2e table) at `(7,3)` and `(9,4)`, both `m`. Round counts go up by 1 everywhere (verdict echo, shared with Group J). Total communication grows 1.2–2.3× (padding, mask rows, and per-party binding polynomials all add data); computation grows much more (12–30×), dominated by the new `O(n_c log n_c)` NTT-based row encoding at `n_c=16384` (up from a much smaller `n_c` before) and by `ligero_verify_as_party` now running per non-dealer party instead of once:

| (n,t) | m | old offline Total | new offline Total | ratio | old Rnds | new Rnds | new offline Comp (ms) |
|---|---|---|---|---|---|---|---|
| (7,3) | 1   | 1 625.65 KB   | 3 789.78 KB  | 2.331 | 12 | 13 | 4 894.6   |
| (7,3) | 100 | 3 814.65 KB   | 8 464.07 KB  | 2.219 | 12 | 13 | 87 954.0  |
| (9,4) | 1   | 11 426.13 KB  | 13 936.19 KB | 1.220 | 14 | 15 | 11 304.4  |
| (9,4) | 100 | 21 087.96 KB  | 35 931.50 KB | 1.704 | 14 | 15 | 461 238.3 |

| (n,t) | m | old e2e Total | new e2e Total | ratio | old Rnds | new Rnds | new e2e Comp (ms) |
|---|---|---|---|---|---|---|---|
| (7,3) | 1   | 1 890.23 KB  | 4 054.36 KB  | 2.145 | 21 | 22 | 4 964.2   |
| (7,3) | 100 | 4 463.45 KB  | 9 112.87 KB  | 2.042 | 21 | 22 | 88 994.0  |
| (9,4) | 1   | 12 676.01 KB | 15 186.07 KB | 1.198 | 23 | 24 | 11 618.3  |
| (9,4) | 100 | 23 529.77 KB | 38 373.28 KB | 1.631 | 23 | 24 | 472 811.0 |

`(3,1)` and `(5,2)` Ligero rows are omitted from these tables, matching the paper's own `N<20` policy (see above). AlyGen, DegGen, VitH (covered in Group J), naive-dVOPRF and the Legendre baseline are untouched by this redesign and were not re-run.
