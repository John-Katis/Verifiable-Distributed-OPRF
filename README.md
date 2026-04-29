# v-dOPRF

Rust prototype implementation of the verifiable distributed OPRF protocols described in `overleaf-protocols/`, plus the external Legendre-dOPRF baseline of Kaluđerović, Cheng & Mitrokotsa (ESORICS 2025) used in the evaluation.

## Layout

```
crates/                Rust workspace (this paper's protocols + bench harness)
overleaf-protocols/    LaTeX sources for the paper
d-OPRF/                External baseline: Legendre-dOPRF (Kaluđerović et al., ESORICS 2025)
```

## `crates/` — what each crate implements

| Crate | Implements | Paper reference |
|---|---|---|
| `field` | `Fp` arithmetic over the Gold-PRF prime (`p = 2^384 − 573·2^128 + 1`, `\|p\|=384`). | §2 Preliminaries |
| `secret-sharing` | Replicated secret sharing (RSS) and doubly-replicated secret sharing (DRSS) — sharing, reconstruction, subset-family combinatorics, party-share folds. | §2 Preliminaries (RSS, DRSS); appendix `rss_open`, `rss_share` |
| `crypto-primitives` | Hash (BLAKE3), PRG, Merkle tree, GGM puncturable PRF, NTT, Reed–Solomon encoder, Fiat–Shamir transcript. Building blocks for the two ZKP back-ends. | Appendix-zkp |
| `network` | `SimulatedNetwork` + `CommStats`: in-process simulation that counts P2P bytes (server↔server unicast), broadcast bytes (one-to-many, counted once), and server↔client bytes; estimates WAN wall-clock as `T_comp + Rounds·RTT + 8·Bytes/BW`. | §6 Evaluation harness |
| `offline` | The offline phase $\mathcal{F}_{\mathsf{Offline}}^{\mathsf{abort}}$ (§4) and all four α^e generators: | §4, §4a, §4b, appendix |
| ↳ `approach_i.rs` | $\Pi_{\mathsf{AlyGen}}$ baseline (log-depth degree-doubling + DZKP). | Cited from [FC:AlyAbiNik18, TIFS:Mennink23] |
| ↳ `approach_ii.rs` | $\Pi_{\mathsf{DegGen}}$ — degenerate-encoding generator. | §4a |
| ↳ `approach_iii.rs` | $\Pi_{\mathsf{ZKPGen}}$ dispatcher (VitH / Ligero variants). | §4b |
| ↳ `zkp_vith.rs` | $\Pi_{\mathsf{ZKPGen}}^{\mathsf{VitH}}$ — VOLE-in-the-Head ZKP. | §4b, appendix-zkp |
| ↳ `zkp_ligero.rs` | $\Pi_{\mathsf{ZKPGen}}^{\mathsf{Lig}}$ — Ligero (RS-encoded) ZKP. | §4b, appendix-zkp |
| ↳ `dzkp.rs` | Distributed ZKP server-side verification (Boyle/BGIN20 `RSS.Open`). | Approach I/II/III offline verification |
| ↳ `pub_base_exp.rs` | Public-base exponentiation subroutine (used by `approach_i`). | §4 Approach I |
| ↳ `rss_mul.rs` | Maliciously-secure RSS multiplication (one-round, DRSS-seeded). | §2 / appendix |
| ↳ `rss_share.rs` | $\Pi_{\mathsf{RSS.Share}}$ — client-as-dealer input distribution; byte accounting per appendix `c_Share`. | Appendix |
| ↳ `double_rand.rs` | DRSS random / zero-share generation. | Appendix A.1–A.2 |
| `online` | The online phase $\Pi_{\mathsf{dVOPRF}}^{m}$ (§5) and the naive baseline. | §5 |
| ↳ `compute.rs`, `compute_batch.rs`, `compute_parallel.rs` | $\Pi_{\mathsf{dVOPRF}}^{m}$ — batched designated-verifier online evaluation (single-input, batched, and parallel variants). | §5 |
| ↳ `vip.rs` | $\Pi_{\mathsf{VIP}}^{\mathsf{Prl}}$ — fused multiply-and-open with client-side verification (`W=UV ∧ Σ=0 ∧ c=…`). | §5 |
| ↳ `compute_boyle.rs` | $\Pi_{\mathsf{naive\text{-}dVOPRF}}^{m}$ baseline — RSS.Mul + batched server-verified DZKP. | Cited from [C:BBCGI19] |
| `bench` | End-to-end benchmark harness: offline-only, online-only (VIP vs. Boyle), and full e2e sweeps over $(n,t) \in \{(3,1),(5,2),(7,3),(9,4)\}$ and $m \in \{1,100\}$. Produces the numbers in Tab. `bench-offline` / `bench-online` / `bench-e2e` of §6. | §6 Evaluation |

## `d-OPRF/` — external baseline

Implementation of the Legendre-OPRF construction by Kaluđerović, Cheng & Mitrokotsa (ESORICS 2025) [[ESORICS:KalCheMit25]](https://eprint.iacr.org/2024/1834), used as the closest available distributed OPRF baseline in §6. Two trees:

- `Legendre-dOPRF/` — original single-machine implementation (https://github.com/nkKolja/Legendre-dOPRF).
- `Legendre-dOPRF-network/` — network-instrumented fork producing the WAN numbers reported in `bench-e2e`.

## Running the benchmarks

```
./run_bench.sh
```

Produces all three §6 tables. WAN model is fixed at 100 ms RTT / 100 Mbps; bytes and rounds are counted analytically by `SimulatedNetwork`, computation is the mean of `BENCH_ITERS` real runs. See `crates/bench/src/main.rs` for the section gating and parameter sweep.

The legendre d-OPRF results can be reproduced by entering d-OPRF/Legendre-dOPRF-network and running:

```
python3 benche2e.py
```
