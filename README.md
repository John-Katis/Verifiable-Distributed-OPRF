# Supplementary code artifact for CCS submission: High-throughput Verifiable Distributed OPRF from Gold PRF

Authors: Nan Cheng, Yugo Kasashima, Yohei Watanabe, Ioannis Katis, Aikaterini Mitrokotsa

Links to code artifact and full paper:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22770327.svg)](https://doi.org/10.5281/zenodo.22770327)
[![ePrint](https://img.shields.io/badge/IACR%20ePrint-2026%2F1953-blue)](https://eprint.iacr.org/2026/1953)

Rust prototype implementation of the verifiable distributed OPRF protocols described in the paper (v-dOPRF), plus the external Legendre-dOPRF baseline of Kaluđerović et al (ESORICS 2025) used in the evaluation.

## Which paper version does this reproduce?

This repo backs two related but distinct documents, and the benchmark CLI has one dedicated entry point for each — see "Which command do I run?" below:

- **Camera-ready.** The short paper that appears officially in the CCS '26 proceedings (DOI above). It reports a single end-to-end table (its Table 2) and omits the per-phase offline/online breakdowns for space. Reproduce it with `./run_bench.sh camera-ready`.
- **Full version.** The extended paper on IACR ePrint (linked above) — full protocol descriptions, deferred proofs, and three benchmark tables instead of one (offline, online, and end-to-end). Its end-to-end table (Table 5) is row-identical to the camera-ready paper's Table 2: same experiment, same numbers, just reported at different granularity. Reproduce it with `./run_bench.sh full-version`.

A note on "which full version": the paper has been revised since our original submission — see `CHANGELOG.md` for the substance of those fixes (e.g. the Legendre-baseline field-size correction, the VIP round-count fix). The numbers in an earlier submitted draft's "full version" are not guaranteed to match what this repo produces today; `full-version` reproduces the **current** full version linked above, i.e. the one this version of the repo was built against.

Both `full-version` and `camera-ready` also run the external Legendre-dOPRF baseline as a trailing section, since it's the only external comparison point either paper's end-to-end table reports against.

If you just want to poke at one protocol or one `(n,t,m)` cell instead of reproducing a whole paper table, jump straight to "Individual benchmarking runs" below.

## Layout

```
crates/                Rust workspace (this paper's protocols + bench harness)
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

Implementation of the Legendre-OPRF construction by Kaluđerović et al (ESORICS 2025) [[ESORICS:KalCheMit25]](https://eprint.iacr.org/2024/1834), used as the closest available distributed OPRF baseline in §6. Vendored directly into this repo (originally tracked as a `git submodule` of https://github.com/nann-cheng/d-OPRF.git during development; flattened into plain tracked files here so the artifact doesn't depend on a third-party remote at clone/archive time — some local patches below were never pushed upstream). Two trees:

- `Legendre-dOPRF/` — original single-machine implementation (https://github.com/nkKolja/Legendre-dOPRF).
- `Legendre-dOPRF-network/` — network-instrumented fork producing the WAN numbers reported in `bench-e2e`. Two local patches on top of the upstream fork, both described in full in `run_legendre_baseline.sh`'s header comment: (1) a 384-bit field (`SEC_LEVEL=5`, using this repo's own Gold-PRF modulus) matching the paper's own `|p|=384` setup — upstream only ships 64/128/192/256/512-bit fields; (2) `sizeof()`-exact offline-phase byte instrumentation in `network-version/server.c`.

## Requirements & setup

Everything here runs on a single machine — `SimulatedNetwork` counts bytes/rounds analytically in-process, so no multi-node setup or real network is used at runtime. Network access is only needed once, to fetch build dependencies (Rust crates via `cargo`, and — only if you're building the Legendre baseline — the BLAKE3 C library).

**1. Nothing to fetch separately.** `d-OPRF/` is vendored directly into this repo (not a git submodule) — a plain `git clone` gets everything, including the local patches to the upstream Legendre-dOPRF code (see below).

**2. Rust toolchain.** A `rust-toolchain.toml` pins `1.87.0` (what this artifact was built/tested with); `rustup` will fetch it automatically on first `cargo build`/`cargo run` in this directory. Needed for every experiment.

**3. C toolchain + BLAKE3 — needed for `full-version`, `camera-ready` (and their `all`/`e2e` aliases), and `legendre-dOPRF`** (anything that touches the external Legendre-dOPRF baseline). The individual Rust-only runs (`offline`, `online`, `our-protocol-verified-input`, `naive-boyle-aly`, or `cargo run` directly) don't need this. `run_legendre_baseline.sh` rebuilds the baseline's C client/server on every invocation (`make clean && make client384 server384`), which requires `gcc`, `make`, and the BLAKE3 C library installed system-wide:

```bash
# Debian/Ubuntu
sudo apt-get install -y gcc make cmake git
git clone https://github.com/BLAKE3-team/BLAKE3.git /tmp/BLAKE3
cmake -S /tmp/BLAKE3/c -B /tmp/BLAKE3/build -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/BLAKE3/build
sudo cmake --install /tmp/BLAKE3/build
# installs libblake3.a to /usr/local/lib and blake3.h to /usr/local/include,
# where the vendored Makefile expects them
```

(macOS: `brew install blake3`. See `d-OPRF/Legendre-dOPRF-network/README.md` for the upstream instructions this is adapted from.)

## Which command do I run?

There are three ways to invoke the benchmark CLI, from most to least coarse-grained. All three go through the same unified dispatcher, `./run_bench.sh`, which also knows how to fold in the external Legendre-dOPRF baseline; the Rust harness alone is also reachable directly via `cargo run` (see below). In every case, `--n N --t T` (both together) restricts a run to a single `(n,t)` pair instead of sweeping `{(3,1),(5,2),(7,3),(9,4)}`, and `--m M1,M2,...` restricts the `m` sweep instead of `{1,100}`.

| # | Command | Reproduces | Includes Legendre-dOPRF? |
|---|---|---|---|
| 1 | `./run_bench.sh full-version` | Every table the *current* full version (ePrint) reports: offline (Tab. 3), online (Tab. 4), e2e (Tab. 5). | Yes, trailing section |
| 2 | `./run_bench.sh camera-ready` | Just the end-to-end table the CCS camera-ready paper prints (its Table 2 — row-identical to the full version's Table 5). | Yes, trailing section |
| 3 | `./run_bench.sh <name>` (see table below) | One phase or one protocol, in isolation — for poking at a single row rather than reproducing a whole paper table. | Only `legendre-dOPRF` itself |

`./run_bench.sh` with no arguments is an alias for `full-version`. See "Which paper version does this reproduce?" above if you're unsure which of (1)/(2) you want.

**Byte and round counts are analytic** (counted by `SimulatedNetwork`/the Legendre baseline's own `sizeof()` instrumentation) and should match exactly across machines. **Wall-clock timings will not match exactly** across hardware — they're the mean of `BENCH_ITERS` real local runs (fixed at `10`, a compile-time constant in `crates/bench/src/avg.rs`, not a runtime flag), scaled through the WAN model. The WAN model itself (100 ms RTT / 100 Mbps) is likewise fixed at compile time (`WAN_RTT_MS`/`WAN_BW_MBPS` in `crates/bench/src/main.rs`) rather than a CLI/env option — edit those constants and rebuild to change it.

**Runtime:** the Rust-only sections (`offline`/`online`/`e2e`, and hence every individual run in (3)) run in well under a minute for the full sweep. The Legendre-dOPRF baseline that `full-version`/`camera-ready` append is the slow part: each `m=100` cell restarts its server set and re-runs the client 100 times sequentially, and the `(t,n)=(2,7)` cell alone takes on the order of ~8 minutes (7-server restart × 100; see `CHANGELOG.md`). A full `./run_bench.sh full-version` (both `(t,n)` pairs × both `m` values for Legendre, plus the Rust sweep) should be expected to take on the order of tens of minutes, dominated by the two `m=100` Legendre cells — it has not hung if it's still running past that.

## (1) Full paper version

```bash
./run_bench.sh full-version                        # full (n,t) x m∈{1,100} sweep — everything
./run_bench.sh full-version --n 5 --t 2 --m 1       # a single (n,t,m) cell across all three tables
```

`full-version` is an alias for the underlying `all` experiment name (both work identically — `all` is kept for backward compatibility). It runs the Rust harness's three sections back to back (offline, online, e2e — `crates/bench/src/main.rs`), then appends the Legendre-dOPRF baseline (both `(t,n)` pairs, `m ∈ {1,100}`).

## (2) Camera-ready CCS paper

```bash
./run_bench.sh camera-ready                         # full (n,t) x m∈{1,100} sweep — the paper's Table 2
./run_bench.sh camera-ready --m 1                   # m=1 rows only
```

`camera-ready` is an alias for the underlying `e2e` experiment name (both work identically — `e2e` is kept for backward compatibility). It runs only the end-to-end section, then appends the Legendre-dOPRF baseline, matching exactly what the short CCS paper reports (its Table 2 / Fig. 4).

## (3) Individual benchmarking runs

For everything short of a full paper table — one phase, one protocol, or one baseline in isolation. Six named Rust experiments (case-sensitive, exact strings; the last two are just `e2e` restricted to one row) plus the separate external-baseline endpoint:

| Name | What it runs |
|---|---|
| `offline` | Offline phase only — all four α^e-generation approaches (I: AlyGen, II: Degenerate Enc, III-a: VOLEitH, III-b: Ligero). |
| `online` | Online phase only, standalone — VIP-ComputeBatch vs. Boyle-Batch (both Π_Input-verified). |
| `e2e` | End-to-end — offline approaches II/III-a/III-b combined with our online protocol, plus the naive Boyle+AlyGen baseline, all in one table (4 rows per `m`). Same table as `camera-ready`, just without the trailing Legendre-dOPRF section. |
| `our-protocol-verified-input` | Currently produces the exact same table as `e2e` (there's only one benched "our protocol" variant right now) — kept as its own explicit name for discoverability. |
| `naive-boyle-aly` | Just the naive Boyle+AlyGen baseline in isolation (offline AlyGen + online Boyle), Π_Input-verified. |
| `all` | Underlying name for `full-version` — `offline` + `online` + `e2e`, run one after another. |
| `legendre-dOPRF` | The external Legendre-dOPRF baseline on its own (see its own subsection below) — a different program entirely, with its own `--tn` flag instead of `--n`/`--t`. |

`offline`/`online`/`our-protocol-verified-input`/`naive-boyle-aly` stay Rust-only (no Legendre section) — Legendre-dOPRF has no offline/online phase split of its own to compare against, only a full end-to-end query, so it's only ever paired with `e2e`/`camera-ready`/`all`/`full-version`.

**Sweep examples** (a whole experiment, one or all `(n,t)` pairs, one or both `m`):

```bash
./run_bench.sh offline                       # offline phase, full (n,t) x m∈{1,100} sweep
./run_bench.sh e2e --m 1                     # e2e (Rust only, no Legendre), full (n,t) sweep, m=1 only
./run_bench.sh online --n 5 --t 2            # online phase, single (n,t)=(5,2), m∈{1,100}
```

**Single protocol / single cell examples** (one `(n,t,m)` triple):

```bash
./run_bench.sh our-protocol-verified-input --n 5 --t 2 --m 1
./run_bench.sh naive-boyle-aly --n 7 --t 3 --m 100
```

### Rust harness directly (`cargo run`)

```
cargo run --release -p vdoprf-bench -- full-version|camera-ready|all|offline|online|e2e|our-protocol-verified-input|naive-boyle-aly [--n N --t T] [--m M1,M2,...]
cargo run --release -p vdoprf-bench -- --help
```

The Rust binary understands `full-version`/`camera-ready` as aliases for `all`/`e2e` too (same names throughout, so nothing to relearn between `run_bench.sh` and `cargo run`) — just without the trailing Legendre-dOPRF section that `run_bench.sh` adds, since that baseline lives outside the Rust workspace. Client input is verified via Π_Input (Protocol 9) in every benched row of `online`/`e2e`/`camera-ready`/`our-protocol-verified-input`/`naive-boyle-aly` — there is no unverified-input comparison row in any bench output.

Examples (Rust only, no Legendre — faster than the same command through `run_bench.sh`'s `camera-ready`/`full-version`, which also run the external baseline):

```bash
cargo run --release -p vdoprf-bench -- full-version                                    # offline + online + e2e, full sweep (Rust only)
cargo run --release -p vdoprf-bench -- our-protocol-verified-input --n 3 --t 1 --m 1   # single cell
```

### External Legendre-dOPRF baseline directly

```
./run_legendre_baseline.sh [--tn T,N] [--m 1|100|1,100]
./run_legendre_baseline.sh --help
```

This baseline has its own, unrelated parameters — not `--n`/`--t`. It runs at a 384-bit field (this repo's own Gold-PRF modulus, giving Legendre the same λ=128 post-quantum security level the paper's matched-security setup targets), at the two `(t,n)` pairs the paper's own end-to-end table reports:
- `--tn T,N`: restrict to one `(t,n)` pair — `(1,4)` or `(2,7)` — omit to run both.
- `--m 1|100|1,100`: since the C client/server only ever serve one query per process, `m=100` means "restart the servers and re-run the client 100 times in sequence, then sum" — a materially slower, different cost model from the Rust side's `m` (each `(t,n)` cell at `m=100` takes minutes, not milliseconds). Default: both `1` and `100`.

Examples:

```bash
./run_legendre_baseline.sh                       # both (t,n) pairs, both m values (slow)
./run_legendre_baseline.sh --tn 1,4 --m 1        # one cell, fast
./run_legendre_baseline.sh --tn 2,7 --m 100      # one cell, m=100 (slow — 7-server restart x 100)
```

Runs the vendored C client/server binaries safely from outside a shell session's own working directory and restores the vendored tree to its committed baseline on exit, even on interruption.
