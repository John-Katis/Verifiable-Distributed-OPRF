# v-dOPRF

Rust prototype implementation of the verifiable distributed OPRF protocols described in `overleaf-protocols/`, plus the external Legendre-dOPRF baseline of Kaluđerović et al (ESORICS 2025) used in the evaluation.

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

Implementation of the Legendre-OPRF construction by Kaluđerović et al (ESORICS 2025) [[ESORICS:KalCheMit25]](https://eprint.iacr.org/2024/1834), used as the closest available distributed OPRF baseline in §6. Two trees:

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

## Exposed experiment endpoints

Every experiment can be run standalone or all at once, either through the unified dispatcher (`./run_bench.sh`, which also knows how to fold in the external Legendre-dOPRF baseline) or directly via `cargo run` for the Rust side alone. `--n N --t T` (both together) restricts a run to a single `(n,t)` pair instead of sweeping `{(3,1),(5,2),(7,3),(9,4)}`; `--m M1,M2,...` restricts the `m` sweep instead of `{1,100}`.

There are six named Rust experiments (case-sensitive, exact strings), plus one more endpoint for the separate external baseline (see below the table):

| Name | What it runs |
|---|---|
| `offline` | Offline phase only — all four α^e-generation approaches (I: AlyGen, II: Degenerate Enc, III-a: VOLEitH, III-b: Ligero). |
| `online` | Online phase only, standalone — VIP-ComputeBatch vs. Boyle-Batch (both Π_Input-verified). |
| `e2e` | End-to-end — offline approaches II/III-a/III-b combined with our online protocol, plus the naive Boyle+AlyGen baseline, all in one table (4 rows per `m`). |
| `our-protocol-verified-input` | Currently produces the exact same table as `e2e` (there's only one benched "our protocol" variant right now) — kept as its own explicit name for discoverability. |
| `naive-boyle-aly` | Just the naive Boyle+AlyGen baseline in isolation (offline AlyGen + online Boyle) — no offline II/III-a/III-b rows. |
| `all` | **`offline` + `online` + `e2e`, run one after another** — every table above, in one invocation. |

**The seventh endpoint, `legendre-dOPRF`, is the external Legendre-dOPRF baseline** (Kaluđerović et al., ESORICS 2025, in the `d-OPRF/` submodule) — it's not one of the six Rust `Experiment` names above (it's a different program entirely, run via `./run_legendre_baseline.sh`, with its own `--tn` flag instead of `--n`/`--t` (see its own subsection below). `./run_bench.sh e2e` and `./run_bench.sh all` run it automatically as a trailing section since it's the closest external comparison point for an end-to-end query. It's not run alongside `offline`/`online` (which have no counterpart in Legendre-dOPRF to compare against) unless you call `./run_bench.sh legendre-dOPRF` directly.

**Omitting `--n`/`--t`/`--m` entirely already gives you the full sweep over every `(n,t)` pair and `m ∈ {1,100}`** — that's the default, not something you need to spell out:

```bash
./run_bench.sh                    # `all` (offline + online + e2e), full sweep, plus Legendre-dOPRF
./run_bench.sh online             # online phase only, full sweep
```

### `./run_bench.sh` — unified dispatcher

```
./run_bench.sh                                          # `all` + Legendre-dOPRF (both (t,n) pairs, m ∈ {1,100}) — the same as the next line
./run_bench.sh all [--n N --t T] [--m M1,M2,...]         # offline + online + e2e (Rust) + Legendre-dOPRF, explicit
./run_bench.sh e2e [--n N --t T] [--m M1,M2,...]         # e2e only (Rust) + Legendre-dOPRF trailing section
./run_bench.sh offline [--n N --t T] [--m M1,M2,...]     # Rust-only: offline phase, all four approaches
./run_bench.sh online [--n N --t T] [--m M1,M2,...]      # Rust-only: online phase, VIP-ComputeBatch vs Boyle-Batch
./run_bench.sh our-protocol-verified-input [--n N --t T] [--m M1,M2,...]   # Rust-only: our e2e protocol alone
./run_bench.sh naive-boyle-aly [--n N --t T] [--m M1,M2,...]               # Rust-only: the naive Boyle+AlyGen baseline alone
./run_bench.sh legendre-dOPRF [--tn T,N] [--m 1|100|1,100]   # external baseline only, on its own
./run_bench.sh --help
```

`offline`/`online`/`our-protocol-verified-input`/`naive-boyle-aly` stay Rust-only (no Legendre section) — Legendre-dOPRF has no offline/online phase split of its own to compare against, only a full end-to-end query, so it's only ever paired with `e2e`/`all`.

**Sweep examples** (a whole experiment, one or all `(n,t)` pairs, one or both `m`):

```bash
./run_bench.sh offline                       # offline phase, full (n,t) x m∈{1,100} sweep
./run_bench.sh e2e --m 1                     # e2e (Rust + Legendre), full (n,t) sweep, m=1 only
./run_bench.sh online --n 5 --t 2            # online phase, single (n,t)=(5,2), m∈{1,100}
./run_bench.sh all                           # offline + online + e2e (Rust) + Legendre-dOPRF, full sweep — everything
```

**Single protocol / single cell examples** (one `(n,t,m)` triple):

```bash
./run_bench.sh our-protocol-verified-input --n 5 --t 2 --m 1
./run_bench.sh naive-boyle-aly --n 7 --t 3 --m 100
```

### Rust harness directly (`cargo run`)

```
cargo run --release -p vdoprf-bench -- all|offline|online|e2e|our-protocol-verified-input|naive-boyle-aly [--n N --t T] [--m M1,M2,...]
cargo run --release -p vdoprf-bench -- --help
```

`all` here means the same three sections as above — offline + online + e2e — just without the trailing Legendre-dOPRF section that `run_bench.sh` adds. Client input is verified via Π_Input (Protocol 9) in every benched row of `online`/`e2e`/`our-protocol-verified-input`/`naive-boyle-aly` — there is no unverified-input comparison row in any bench output.

Examples (Rust only, no Legendre — faster than the same command through `run_bench.sh`'s `e2e`/`all`, which also runs the external baseline):

```bash
cargo run --release -p vdoprf-bench -- all                                    # offline + online + e2e, full sweep (Rust only)
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

Runs the vendored C client/server binaries safely from outside a shell session's own working directory and restores the submodule to its committed baseline on exit, even on interruption.
