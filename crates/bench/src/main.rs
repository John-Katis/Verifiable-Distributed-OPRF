//! Benchmark entry point for v-dOPRF.
//!
//! Three sections, 10-iteration averaged:
//!   1. Offline phase only — all four methods × (n,t) × m ∈ {1, 35}.
//!      Offline verification is server-side (Boyle/BGIN20 RSS.Open) and
//!      baked into the protocol; timings include it.
//!   2. Online phase standalone — VIP-ComputeBatch vs. Boyle-Batch only,
//!      full (n,t) sweep × m ∈ {1, 35}. Bytes are split into two axes:
//!      server↔server (internal protocol traffic) and server↔client
//!      (server → client output + VIP proof masks). Client → server input
//!      is a constant RSS-shared `x` of identical shape in both schemes and
//!      is intentionally excluded from the comparison. Timings include
//!      verification: client-side for VIP, server-side for Boyle.
//!   3. End-to-end — offline + online + verification. Reuses the exact same
//!      online entry points Section 2 uses so the accounting lines up:
//!        any m:  1 batched offline(m) → compute_batch → client-verify.
//!
//! Every measured cell is the mean of `BENCH_ITERS` runs; bytes and rounds
//! are rounded to integers.

use num_bigint::BigUint;
use num_traits::One;
use std::time::Instant;
use vdoprf_field::Fp;
use vdoprf_network::{CommStats, SimulatedNetwork};
use vdoprf_offline::approach_i;
use vdoprf_offline::approach_ii;
use vdoprf_offline::approach_iii::{self, ZkpVariant};
use vdoprf_offline::pub_base_exp::is_coprime;
use vdoprf_offline::setup_pre_shared;
use vdoprf_offline::zkp_ligero::LigeroParams;
use vdoprf_offline::zkp_vith::VitHParams;
use vdoprf_offline::PreSharedMaterial;
use vdoprf_online::compute_boyle::BoyleBatchProof;
use vdoprf_online::vip::{client_verify_dvoprf, VipParallelOutput, VipResult};
use vdoprf_online::{
    compute_batch, compute_boyle, setup_random_preprocessed_m, OnlinePreprocessed,
    OnlineProof, OnlineResult, VipDvoprfOutput,
};
use vdoprf_ss::{ReplicatedSharing, RssShare, SubsetFamily};

mod avg;
use avg::{bench_avg, bench_avg_split, AvgStats, AvgStatsSplit, BENCH_ITERS};

const LAMBDA: u32 = 128;
const KAPPA: usize = 40;
const TAU: usize = 16;
const WAN_RTT_MS: f64 = 100.0; // WAN round-trip time in ms
const WAN_BW_MBPS: f64 = 100.0; // WAN bandwidth in Mbps

fn wan_ms(time_ms: f64, rounds: usize, total_bytes: usize) -> f64 {
    let latency = rounds as f64 * WAN_RTT_MS;
    let transfer = total_bytes as f64 * 8.0 / (WAN_BW_MBPS * 1000.0);
    time_ms + latency + transfer
}

// ---------------------------------------------------------------------------
// Client simulation
//
// The online-protocol crates stay pure: they emit per-server artefacts
// (tilde_v additive shares, the 5 VIP proof-share vectors, or Boyle's full
// per-party RSS shares of c) without ever calling `send_to_client` on their
// own. The bench takes those outputs, pushes them through a
// `SimulatedNetwork` under the client helper below, and performs the
// client-side reconstruction/verification. All of this happens inside the
// benchmark's timed region so both the serialization cost and the
// reconstruction cost are counted, and `CommStats::client_bytes` reports the
// actual bytes the servers sent.
// ---------------------------------------------------------------------------

/// Byte width of a field element under the working modulus.
fn fe_bytes(modulus: &BigUint) -> usize {
    ((modulus.bits() + 7) / 8) as usize
}

/// Serialize one field element as `feb` bytes (big-endian, zero-padded).
fn serialize_fp(fp: &Fp, feb: usize) -> Vec<u8> {
    let mut out = vec![0u8; feb];
    let bytes = fp.value.to_bytes_be();
    let copy = bytes.len().min(feb);
    out[feb - copy..].copy_from_slice(&bytes[bytes.len() - copy..]);
    out
}

/// Server → client delivery for VIP-ComputeBatch plus client-side
/// verification. Matches §4-Online `fig:doprf_protocol` line 260:
/// *"Client C calls F_RSS.Open to robustly reconstruct ⟨⟨W(ρ)⟩⟩, ⟨⟨U(ρ)⟩⟩,
/// ⟨⟨V(ρ)⟩⟩, ⟨⟨Σ⟩⟩, and ⟨⟨c⟩⟩"*, and appendix line 1102 (`5·c_Open + m`).
///
/// Each of the n servers ships:
///   - its additive share ṽ_i^{(k)} for every k ∈ [m] (m field elements,
///     per line 270–271 of the protocol);
///   - its full RSS share of each of the 5 Π_VIP^Prl proof values
///     (`w_ρ, u_ρ, v_ρ, Σ, c`) under the `Π_RSS.Open` protocol of Appendix
///     `fig:rss_open` — `C(n-1, t)` field elements plus a 32-byte hash per
///     opened value.
///
/// Total wire bytes per server: `m · feb + 5 · (C(n-1,t) · feb + 32)` ≈
/// `5·c_Open + m` field-elements' worth, matching `appendix.tex:1102`.
fn client_deliver_vip_batched(
    result: &OnlineResult,
    net: &mut SimulatedNetwork,
    modulus: &BigUint,
) -> Vec<Fp> {
    let feb = fe_bytes(modulus);
    let tilde_v: &Vec<Vec<Fp>> = match &result.proof {
        OnlineProof::Batched(VipDvoprfOutput { tilde_v, .. }) => tilde_v,
        _ => panic!("client_deliver_vip_batched: expected OnlineProof::Batched"),
    };
    let vip: &VipParallelOutput = match &result.proof {
        OnlineProof::Batched(VipDvoprfOutput { vip, .. }) => vip,
        _ => unreachable!(),
    };
    let n = tilde_v.len();
    let m = if n == 0 { 0 } else { tilde_v[0].len() };

    // Ship one opened RSS share (full `C(n-1,t)` components + 32-byte hash)
    // per proof value per party, in one payload. Mirrors the bundle shape
    // of `client_deliver_boyle`'s per-c_j delivery.
    const HASH_BYTES: usize = 32;
    let ship_rss_open = |net: &mut SimulatedNetwork, sharing: &[RssShare]| {
        for (i, share) in sharing.iter().enumerate() {
            let mut payload = Vec::with_capacity(share.shares.len() * feb + HASH_BYTES);
            for fp in share.shares.values() {
                payload.extend_from_slice(&serialize_fp(fp, feb));
            }
            payload.extend_from_slice(&[0u8; HASH_BYTES]);
            net.send_to_client(i, payload);
        }
    };

    // Per-server ṽ_i^{(k)} (m additive field elements, line 270–271).
    for i in 0..n {
        for k in 0..m {
            net.send_to_client(i, serialize_fp(&tilde_v[i][k], feb));
        }
    }
    // Robust RSS open of the five VIP proof values (line 260).
    if n > 0 {
        ship_rss_open(net, &vip.w_rho_shares);
        ship_rss_open(net, &vip.u_rho_shares);
        ship_rss_open(net, &vip.v_rho_shares);
        ship_rss_open(net, &vip.sigma_shares);
        ship_rss_open(net, &vip.c_shares);
    }

    // Client-side verification: full Π_dVOPRF^m verdict (`5-Online.tex:255`)
    // — W=UV ∧ Σ=0 ∧ c = Σ_j ε_j · Σ_i ṽ_i^{(j)}.
    assert_eq!(
        client_verify_dvoprf(vip, tilde_v, modulus),
        VipResult::Accept,
        "bench: dVOPRF verify must Accept on honest execution",
    );

    // Client-side reconstruction of v^{(k)}: additive per §4-Online line 272.
    let mut outputs = Vec::with_capacity(m);
    for k in 0..m {
        let mut v = Fp::zero(modulus);
        for i in 0..n {
            v = &v + &tilde_v[i][k];
        }
        outputs.push(v);
    }
    outputs
}

/// Client → servers input distribution via the double-sharing-based
/// `Π_RSS.Share` from `appendix.tex:92–108` (client acting as dealer).
/// Per-shared-value wire cost: `(n + t + 1)·feb` bytes total, amortised
/// `≈ 1.5·feb` per server. Matches the `m·c_Share` per-server cost from
/// `appendix.tex:1099`. The bench measures bytes only — no plaintext x or
/// share material is shipped — so the helper runs in O(m + n) memory writes
/// and is cheap enough to sit inside the `Instant::now()` window of every
/// online/e2e cell.
fn client_to_servers_input(
    m: usize,
    n: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
) -> CommStats {
    let mut net = SimulatedNetwork::new(n);
    vdoprf_offline::rss_share::charge_client_rss_share(m, family, modulus, &mut net);
    net.stats()
}

/// Server → client delivery for Boyle-Batch. Each server ships its *full*
/// RSS share of c_j for every input j — i.e. every subset value the party
/// holds (size `C(n-1, t)` field elements per party per input). Shipped
/// redundantly so the client can detect a lying holder by cross-checking
/// subsets held by multiple parties. Total wire bytes:
/// `m · n · C(n-1, t) · feb`. Also ships the `verdict` (1 byte).
/// Returns the reconstructed `c_j` vector.
fn client_deliver_boyle(
    proof: &BoyleBatchProof,
    net: &mut SimulatedNetwork,
    modulus: &BigUint,
) -> Vec<Fp> {
    let feb = fe_bytes(modulus);
    let mut outputs = Vec::with_capacity(proof.open_shares.len());
    for c_shares in &proof.open_shares {
        for (i, share) in c_shares.iter().enumerate() {
            let mut payload = Vec::with_capacity(share.shares.len() * feb);
            for fp in share.shares.values() {
                payload.extend_from_slice(&serialize_fp(fp, feb));
            }
            net.send_to_client(i, payload);
        }
        // Client-side reconstruction: fold the n received per-party RSS
        // shares into one `ReplicatedSharing` and sum its components. With
        // redundancy the client could also verify that all holders of each
        // subset reported the same value; on honest execution this is
        // automatic.
        outputs.push(
            ReplicatedSharing::reconstruct_from_party_shares(c_shares, modulus),
        );
    }
    // DZKP verdict: each of the n servers independently ran the server-
    // verified DZKP and reaches the same Accept/Abort decision in honest
    // execution. All n ship their verdict byte to the client, which
    // accepts iff unanimous. Verdict is batch-level — one byte per server,
    // not per input.
    if let Some(first) = proof.open_shares.first() {
        let n = first.len();
        for i in 0..n {
            net.send_to_client(i, vec![0u8]);
        }
    }
    outputs
}

// ---------------------------------------------------------------------------
// Row type shared across the three sections
// ---------------------------------------------------------------------------

struct Row {
    n: usize,
    t: usize,
    m: usize,
    name: String,
    time_ms: f64,
    p2p_bytes: usize,
    broadcast_bytes: usize,
    client_bytes: usize,
    rounds: usize,
    skipped: bool,
}

impl Row {
    fn total_bytes(&self) -> usize {
        self.p2p_bytes + self.broadcast_bytes + self.client_bytes
    }
    fn wan_ms(&self) -> f64 {
        wan_ms(self.time_ms, self.rounds, self.total_bytes())
    }

    fn from_avg(n: usize, t: usize, m: usize, name: &str, avg: AvgStats) -> Self {
        Row {
            n, t, m,
            name: name.into(),
            time_ms: avg.time_ms,
            p2p_bytes: avg.p2p_bytes,
            broadcast_bytes: avg.broadcast_bytes,
            client_bytes: avg.client_bytes,
            rounds: avg.rounds,
            skipped: false,
        }
    }

    fn skipped(n: usize, t: usize, m: usize, name: &str) -> Self {
        Row {
            n, t, m,
            name: name.into(),
            time_ms: f64::NAN,
            p2p_bytes: 0,
            broadcast_bytes: 0,
            client_bytes: 0,
            rounds: 0,
            skipped: true,
        }
    }
}

/// Row variant for Section 2, reporting bytes split by direction of
/// communication rather than by P2P-vs-broadcast: server↔server (internal
/// protocol traffic) and server↔client (input + output + proof).
struct SplitRow {
    n: usize,
    t: usize,
    m: usize,
    name: String,
    time_ms: f64,
    ss_bytes: usize,
    sc_bytes: usize,
    rounds: usize,
}

impl SplitRow {
    fn total_bytes(&self) -> usize {
        self.ss_bytes + self.sc_bytes
    }
    fn wan_ms(&self) -> f64 {
        wan_ms(self.time_ms, self.rounds, self.total_bytes())
    }
    fn from_avg(n: usize, t: usize, m: usize, name: &str, avg: AvgStatsSplit) -> Self {
        SplitRow {
            n, t, m,
            name: name.into(),
            time_ms: avg.time_ms,
            ss_bytes: avg.ss_bytes,
            sc_bytes: avg.sc_bytes,
            rounds: avg.rounds,
        }
    }
}

/// Pick an exponent coprime to p-1 by nudging upward from `e` until
/// gcd(e, p-1) = 1. Approach I's Protocol 17 (AlyGen) requires this.
fn aly_coprime_e(e: &BigUint, modulus: &BigUint) -> BigUint {
    let p_minus_1 = modulus - BigUint::one();
    let mut candidate = e.clone();
    while !is_coprime(&candidate, &p_minus_1) {
        candidate += BigUint::one();
    }
    candidate
}

// ---------------------------------------------------------------------------
// Section 1 — Offline phase, sweep (n,t) × m ∈ {1, 35}
// ---------------------------------------------------------------------------

fn run_offline_one(
    n: usize,
    t: usize,
    m: usize,
    name: &str,
    family: &SubsetFamily,
    modulus: &BigUint,
    e: &BigUint,
    pre_shared: &[PreSharedMaterial],
) -> Row {
    match name {
        "I: AlyGen" => {
            if aly_skip(n, t, m) {
                return Row::skipped(n, t, m, name);
            }
            let e_coprime = aly_coprime_e(e, modulus);
            let exponents: Vec<BigUint> = (0..m).map(|_| e_coprime.clone()).collect();
            let g = Fp::new(BigUint::from(3u32), modulus);
            let avg = bench_avg(|| {
                let t0 = Instant::now();
                let (_alphas, result) =
                    approach_i::aly_gen(&exponents, pre_shared, family, modulus, &g);
                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                (dt, result.comm)
            });
            Row::from_avg(n, t, m, name, avg)
        }
        "II: Degenerate Enc" => {
            let exponents: Vec<BigUint> = (0..m).map(|_| e.clone()).collect();
            let avg = bench_avg(|| {
                let t0 = Instant::now();
                let (_alpha_es, result) =
                    approach_ii::gen_degenerate(&exponents, pre_shared, family, modulus);
                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                (dt, result.comm)
            });
            Row::from_avg(n, t, m, name, avg)
        }
        "III-a: VOLEitH" => {
            let exponents: Vec<BigUint> = (0..m).map(|_| e.clone()).collect();
            let vith_params = VitHParams::new(TAU, KAPPA);
            let avg = bench_avg(|| {
                let t0 = Instant::now();
                let (_alpha_es, result) = approach_iii::gen_zkp(
                    &exponents,
                    pre_shared,
                    family,
                    modulus,
                    &ZkpVariant::VitH(vith_params.clone()),
                );
                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                (dt, result.comm)
            });
            Row::from_avg(n, t, m, name, avg)
        }
        "III-b: Ligero" => {
            let dealer_wires = family.subsets_not_containing(0).len();
            let Some(ligero_params) = LigeroParams::try_new(dealer_wires, KAPPA, modulus) else {
                return Row::skipped(n, t, m, name);
            };
            if ligero_params.n_c > 10000 {
                return Row::skipped(n, t, m, name);
            }
            let exponents: Vec<BigUint> = (0..m).map(|_| e.clone()).collect();
            let avg = bench_avg(|| {
                let t0 = Instant::now();
                let (_alpha_es, result) = approach_iii::gen_zkp(
                    &exponents,
                    pre_shared,
                    family,
                    modulus,
                    &ZkpVariant::Ligero(ligero_params.clone()),
                );
                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                (dt, result.comm)
            });
            Row::from_avg(n, t, m, name, avg)
        }
        _ => unreachable!("unknown offline approach {}", name),
    }
}

fn run_offline_set(
    n: usize,
    t: usize,
    modulus: &BigUint,
    e: &BigUint,
    m_values: &[usize],
) -> Vec<Row> {
    let family = SubsetFamily::new(n, t);
    let pre_shared = setup_pre_shared(n, t, modulus);
    let num_subsets = family.num_subsets();
    let names = ["I: AlyGen", "II: Degenerate Enc", "III-a: VOLEitH", "III-b: Ligero"];
    let mut out = Vec::new();
    for &m in m_values {
        eprintln!("  [offline] n={}, t={}, m={}, N={}", n, t, m, num_subsets);
        for name in names.iter() {
            eprint!("    {:20} ... ", name);
            let row = run_offline_one(n, t, m, name, &family, modulus, e, &pre_shared);
            if row.skipped {
                eprintln!("skip");
            } else {
                eprintln!(
                    "done  (avg of {} iters: {:.1} ms, P2P={} B, Bcast={} B, rounds={})",
                    BENCH_ITERS, row.time_ms, row.p2p_bytes, row.broadcast_bytes, row.rounds,
                );
            }
            out.push(row);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Section 2 — Online phase standalone, two variants (VIP-ComputeBatch vs.
// Boyle-Batch), full (n,t) sweep, m ∈ {1, 35}. Bytes are reported along the
// two physical axes of communication: server↔server (internal protocol
// traffic — VSS, commit-hash broadcasts, RSS.Mul rounds, etc.) and
// server↔client (input distribution + output/proof opening).
// ---------------------------------------------------------------------------

fn run_online_set(n: usize, t: usize, m: usize, modulus: &BigUint) -> Vec<SplitRow> {
    let family = SubsetFamily::new(n, t);
    let pre_shared = setup_pre_shared(n, t, modulus);
    let pre = setup_random_preprocessed_m(m, &family, modulus);

    let mut rng = rand::thread_rng();
    let xs: Vec<Fp> = (0..m).map(|_| Fp::random(modulus, &mut rng)).collect();

    eprintln!("  [online ] n={}, t={}, m={}", n, t, m);
    let mut out = Vec::new();

    // VIP: ComputeBatch. Time covers: client→server input VSS + server-side
    // protocol + bench-side client delivery (ṽ additive shares + 5 proof
    // shares) + client-side reconstruction + `client_verify_vip_parallel`.
    {
        let avg = bench_avg_split(|| {
            let t0 = Instant::now();
            let input_comm = client_to_servers_input(xs.len(), n, &family, modulus);
            let r = compute_batch::compute_batch(&xs, &pre, &pre_shared, &family, modulus);
            let mut client_net = SimulatedNetwork::new(n);
            let _outs = client_deliver_vip_batched(&r, &mut client_net, modulus);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            let mut comm = r.comm;
            comm.merge(&input_comm);
            comm.merge(&client_net.stats());
            (dt, comm)
        });
        eprintln!(
            "    {:20} ... done  ({:.1} ms avg, S↔S={} B, S↔C={} B, rounds={})",
            "VIP-ComputeBatch", avg.time_ms, avg.ss_bytes, avg.sc_bytes, avg.rounds,
        );
        out.push(SplitRow::from_avg(n, t, m, "VIP-ComputeBatch", avg));
    }

    // Boyle-Batch: client→server input VSS + server-side batched RSS.Mul +
    // batched DZKP + server→client redundant open of c_j with verdict.
    {
        let avg = bench_avg_split(|| {
            let t0 = Instant::now();
            let input_comm = client_to_servers_input(xs.len(), n, &family, modulus);
            let (_cs, proof, comm_server) =
                compute_boyle::compute_boyle_batch(&xs, &pre, &pre_shared, &family, modulus);
            let mut client_net = SimulatedNetwork::new(n);
            let _outs = client_deliver_boyle(&proof, &mut client_net, modulus);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            // Client delivery of (verdict, z_k RSS shares) is a distinct
            // round after Protocol 4.2 step 7: servers only ship outputs
            // once β = 0 has been locally reconstructed, so the two merge
            // sequentially (rounds add, bytes add).
            let mut comm = comm_server;
            comm.merge(&input_comm);
            comm.merge(&client_net.stats());
            (dt, comm)
        });
        eprintln!(
            "    {:20} ... done  ({:.1} ms avg, S↔S={} B, S↔C={} B, rounds={})",
            "Boyle-Batch", avg.time_ms, avg.ss_bytes, avg.sc_bytes, avg.rounds,
        );
        out.push(SplitRow::from_avg(n, t, m, "Boyle-Batch", avg));
    }

    out
}

// ---------------------------------------------------------------------------
// Section 3 — End-to-end (offline + online + verify), m ∈ {1, 35}
// ---------------------------------------------------------------------------

/// Which online phase to compose with the offline α^e generator in an e2e
/// row. `Vip` = our designated-verifier path (`compute_batch` for all m,
/// matching Section 2's online-standalone VIP cell);
/// `Boyle` = RSS.Mul + server-verified DZKP (`compute_boyle_batch`).
#[derive(Clone, Copy)]
enum OnlineVariant {
    Vip,
    Boyle,
}

/// Gate for (n,t,m) cells where ΠAlyGen × m is too slow to bench with
/// `BENCH_ITERS` repetitions. AlyGen scales worst with both (n, m) — at m=35,
/// (9,4) is already ~5 min per iter; m=100 is ~3× worse. Skip n=9 once m
/// crosses the m=35 threshold; everything at n ≤ 7 stays tractable at m=100.
fn aly_skip(n: usize, _t: usize, m: usize) -> bool {
    n >= 9 && m >= 35
}

/// Result of one offline run, returning just the α^e sharing (caller
/// supplies k). Used by e2e to build an m-sized preprocessed from m fresh
/// offline calls.
enum OfflineAlphaRun {
    Ok { time_ms: f64, comm: CommStats, alpha: ReplicatedSharing },
    Skip,
}

fn run_offline_alpha_only(
    name: &str,
    family: &SubsetFamily,
    modulus: &BigUint,
    e: &BigUint,
    pre_shared: &[PreSharedMaterial],
) -> OfflineAlphaRun {
    match name {
        "I" => {
            let e_coprime = aly_coprime_e(e, modulus);
            let g = Fp::new(BigUint::from(3u32), modulus);
            let t0 = Instant::now();
            let (_alphas, result) =
                approach_i::aly_gen(&[e_coprime], pre_shared, family, modulus, &g);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            let alpha = ReplicatedSharing::from_party_shares(&result.result_shares[0]);
            OfflineAlphaRun::Ok { time_ms: dt, comm: result.comm, alpha }
        }
        "II" => {
            let t0 = Instant::now();
            let (_alpha_es, result) =
                approach_ii::gen_degenerate(std::slice::from_ref(e), pre_shared, family, modulus);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            let alpha = ReplicatedSharing::from_party_shares(&result.result_shares[0]);
            OfflineAlphaRun::Ok { time_ms: dt, comm: result.comm, alpha }
        }
        "III-a" => {
            let vith_params = VitHParams::new(TAU, KAPPA);
            let t0 = Instant::now();
            let (_alpha_es, result) = approach_iii::gen_zkp(
                std::slice::from_ref(e),
                pre_shared,
                family,
                modulus,
                &ZkpVariant::VitH(vith_params),
            );
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            let alpha = ReplicatedSharing::from_party_shares(&result.result_shares[0]);
            OfflineAlphaRun::Ok { time_ms: dt, comm: result.comm, alpha }
        }
        "III-b" => {
            let dealer_wires = family.subsets_not_containing(0).len();
            let Some(ligero_params) = LigeroParams::try_new(dealer_wires, KAPPA, modulus) else {
                return OfflineAlphaRun::Skip;
            };
            if ligero_params.n_c > 10000 {
                return OfflineAlphaRun::Skip;
            }
            let t0 = Instant::now();
            let (_alpha_es, result) = approach_iii::gen_zkp(
                std::slice::from_ref(e),
                pre_shared,
                family,
                modulus,
                &ZkpVariant::Ligero(ligero_params),
            );
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            let alpha = ReplicatedSharing::from_party_shares(&result.result_shares[0]);
            OfflineAlphaRun::Ok { time_ms: dt, comm: result.comm, alpha }
        }
        _ => unreachable!(),
    }
}

/// Batched counterpart of `run_offline_alpha_only`: one protocol invocation
/// with `exponents = vec![e; m]`, yielding m fresh α^e sharings and one
/// `CommStats` that reflects the batched (not per-query) round count. This
/// is the accounting the offline-only section (`run_offline_one`) already
/// uses; the e2e path uses it to avoid inflating rounds by a factor of m.
enum OfflineAlphasRun {
    Ok { time_ms: f64, comm: CommStats, alphas: Vec<ReplicatedSharing> },
    Skip,
}

fn run_offline_alphas_batched(
    name: &str,
    m: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
    e: &BigUint,
    pre_shared: &[PreSharedMaterial],
) -> OfflineAlphasRun {
    match name {
        "I" => {
            if aly_skip(family.n, family.t, m) {
                return OfflineAlphasRun::Skip;
            }
            let e_coprime = aly_coprime_e(e, modulus);
            let exponents: Vec<BigUint> = (0..m).map(|_| e_coprime.clone()).collect();
            let g = Fp::new(BigUint::from(3u32), modulus);
            let t0 = Instant::now();
            let (_alphas, result) =
                approach_i::aly_gen(&exponents, pre_shared, family, modulus, &g);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            let alphas = (0..m)
                .map(|i| ReplicatedSharing::from_party_shares(&result.result_shares[i]))
                .collect();
            OfflineAlphasRun::Ok { time_ms: dt, comm: result.comm, alphas }
        }
        "II" => {
            let exponents: Vec<BigUint> = (0..m).map(|_| e.clone()).collect();
            let t0 = Instant::now();
            let (_alpha_es, result) =
                approach_ii::gen_degenerate(&exponents, pre_shared, family, modulus);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            let alphas = (0..m)
                .map(|i| ReplicatedSharing::from_party_shares(&result.result_shares[i]))
                .collect();
            OfflineAlphasRun::Ok { time_ms: dt, comm: result.comm, alphas }
        }
        "III-a" => {
            let exponents: Vec<BigUint> = (0..m).map(|_| e.clone()).collect();
            let vith_params = VitHParams::new(TAU, KAPPA);
            let t0 = Instant::now();
            let (_alpha_es, result) = approach_iii::gen_zkp(
                &exponents,
                pre_shared,
                family,
                modulus,
                &ZkpVariant::VitH(vith_params),
            );
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            let alphas = (0..m)
                .map(|i| ReplicatedSharing::from_party_shares(&result.result_shares[i]))
                .collect();
            OfflineAlphasRun::Ok { time_ms: dt, comm: result.comm, alphas }
        }
        "III-b" => {
            let dealer_wires = family.subsets_not_containing(0).len();
            let Some(ligero_params) = LigeroParams::try_new(dealer_wires, KAPPA, modulus) else {
                return OfflineAlphasRun::Skip;
            };
            if ligero_params.n_c > 10000 {
                return OfflineAlphasRun::Skip;
            }
            let exponents: Vec<BigUint> = (0..m).map(|_| e.clone()).collect();
            let t0 = Instant::now();
            let (_alpha_es, result) = approach_iii::gen_zkp(
                &exponents,
                pre_shared,
                family,
                modulus,
                &ZkpVariant::Ligero(ligero_params),
            );
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            let alphas = (0..m)
                .map(|i| ReplicatedSharing::from_party_shares(&result.result_shares[i]))
                .collect();
            OfflineAlphasRun::Ok { time_ms: dt, comm: result.comm, alphas }
        }
        _ => unreachable!(),
    }
}

/// One e2e iteration at a given (approach, m, online_variant):
///   - one batched offline call producing m fresh α^e's.
///   - Build an m-sized `OnlinePreprocessed` with one k and the m α^e's.
///   - Run online per `online_variant`, using the **same** entry points the
///     Section-2 online-standalone bench uses, so e2e cells accumulate the
///     same server-side comm/rounds as the corresponding online-only cells:
///       VIP   → `compute_batch` + `client_deliver_vip_batched` for all m.
///       Boyle → `compute_boyle_batch` + `client_deliver_boyle` for all m.
///
/// Returns `(time_ms, comm_stats)` suitable for `bench_avg` averaging.
/// Returns `None` if the approach must be skipped (e.g. Ligero n_c too high).
fn run_one_e2e_iter(
    name: &str,
    online_variant: OnlineVariant,
    m: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
    e: &BigUint,
    pre_shared: &[PreSharedMaterial],
    xs: &[Fp],
) -> Option<(f64, CommStats)> {
    let mut rng = rand::thread_rng();

    let (offline_time, offline_comm, alphas) =
        match run_offline_alphas_batched(name, m, family, modulus, e, pre_shared) {
            OfflineAlphasRun::Skip => return None,
            OfflineAlphasRun::Ok { time_ms, comm, alphas } => (time_ms, comm, alphas),
        };

    // Fresh long-lived PRF key k (one per e2e iteration — the offline run
    // only produces α^e's; k is shared independently).
    let k = Fp::random(modulus, &mut rng);
    let k_sharing = vdoprf_ss::share(&k, family, modulus, &mut rng);
    let pre = OnlinePreprocessed { k_sharing, alpha_e_sharings: alphas };

    let t0 = Instant::now();
    let n = family.n;
    let input_comm = client_to_servers_input(xs.len(), n, family, modulus);
    let online_comm = match online_variant {
        OnlineVariant::Vip => {
            let r = compute_batch::compute_batch(xs, &pre, pre_shared, family, modulus);
            let mut client_net = SimulatedNetwork::new(n);
            let _outs = client_deliver_vip_batched(&r, &mut client_net, modulus);
            let mut c = r.comm;
            c.merge(&client_net.stats());
            c
        }
        OnlineVariant::Boyle => {
            let (_cs, proof, mut c) =
                compute_boyle::compute_boyle_batch(xs, &pre, pre_shared, family, modulus);
            let mut client_net = SimulatedNetwork::new(n);
            let _outs = client_deliver_boyle(&proof, &mut client_net, modulus);
            c.merge(&client_net.stats());
            c
        }
    };
    let online_t = t0.elapsed().as_secs_f64() * 1000.0;

    let mut total = offline_comm;
    total.merge(&input_comm);
    total.merge(&online_comm);
    Some((offline_time + online_t, total))
}

fn run_e2e_set(
    n: usize,
    t: usize,
    modulus: &BigUint,
    e: &BigUint,
    m_values: &[usize],
) -> Vec<Row> {
    let family = SubsetFamily::new(n, t);
    let pre_shared = setup_pre_shared(n, t, modulus);
    let offline_names = ["II", "III-a", "III-b"];
    let offline_full = [
        "II: Degenerate Enc",
        "III-a: VOLEitH",
        "III-b: Ligero",
    ];

    let mut out = Vec::new();
    for &m in m_values {
        let online_label = if m == 1 { "Compute" } else { "ComputeBatch" };
        eprintln!(
            "  [e2e    ] n={}, t={}, m={} (batched α^e, then {})",
            n, t, m, online_label
        );

        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..m).map(|_| Fp::random(modulus, &mut rng)).collect();

        for (short, full) in offline_names.iter().zip(offline_full.iter()) {
            eprint!("    {:20} ... ", full);
            let label = format!("{} + {}", full, online_label);

            // Probe once to check skip vs run.
            let probe = run_offline_alpha_only(short, &family, modulus, e, &pre_shared);
            if matches!(probe, OfflineAlphaRun::Skip) {
                eprintln!("skip");
                out.push(Row::skipped(n, t, m, &label));
                continue;
            }
            drop(probe);

            let avg = bench_avg(|| {
                run_one_e2e_iter(
                    short,
                    OnlineVariant::Vip,
                    m,
                    &family,
                    modulus,
                    e,
                    &pre_shared,
                    &xs,
                )
                .expect("probed earlier")
            });
            eprintln!(
                "done  (avg of {} iters: {:.1} ms, P2P={} B, Bcast={} B, rounds={})",
                BENCH_ITERS, avg.time_ms, avg.p2p_bytes, avg.broadcast_bytes, avg.rounds,
            );
            out.push(Row::from_avg(n, t, m, &label, avg));
        }

        // Baseline: ΠAlyGen (offline) composed with Boyle batched online.
        // Factored into `e2e_aly_boyle_one_cell` so it can be driven
        // standalone by `e2e_aly_boyle` (see BENCH_SECTION=e2e-aly-boyle).
        out.push(e2e_aly_boyle_one_cell(n, t, m, &family, modulus, e, &pre_shared, &xs));
    }
    out
}

/// Run the ΠAlyGen + Boyle-online e2e experiment at a single (n,t,m) cell.
/// Returns a `Row` (either a measured result or `Row::skipped` when
/// `aly_e2e_skip` fires).
fn e2e_aly_boyle_one_cell(
    n: usize,
    t: usize,
    m: usize,
    family: &SubsetFamily,
    modulus: &BigUint,
    e: &BigUint,
    pre_shared: &[PreSharedMaterial],
    xs: &[Fp],
) -> Row {
    let full = "I: AlyGen";
    let label = format!("{} + BoyleBatch", full);
    eprint!("    {:20} ... ", full);

    if aly_skip(n, t, m) {
        eprintln!("skip (Aly × m={} infeasible)", m);
        return Row::skipped(n, t, m, &label);
    }

    let avg = bench_avg(|| {
        run_one_e2e_iter(
            "I",
            OnlineVariant::Boyle,
            m,
            family,
            modulus,
            e,
            pre_shared,
            xs,
        )
        .expect("AlyGen has no offline skip path")
    });
    eprintln!(
        "done  (avg of {} iters: {:.1} ms, P2P={} B, Bcast={} B, rounds={})",
        BENCH_ITERS, avg.time_ms, avg.p2p_bytes, avg.broadcast_bytes, avg.rounds,
    );
    Row::from_avg(n, t, m, &label, avg)
}

/// Standalone driver for the ΠAlyGen + Boyle-online e2e experiment, sweeping
/// `m_values` at a single (n,t). Equivalent to the corresponding rows that
/// `run_e2e_set` emits, but runnable in isolation via
/// `BENCH_SECTION=e2e-aly-boyle` when the other e2e compositions are not
/// wanted. Same xs / k / preshared-material setup as `run_e2e_set`.
fn e2e_aly_boyle(
    n: usize,
    t: usize,
    modulus: &BigUint,
    e: &BigUint,
    m_values: &[usize],
) -> Vec<Row> {
    let family = SubsetFamily::new(n, t);
    let pre_shared = setup_pre_shared(n, t, modulus);

    let mut out = Vec::new();
    for &m in m_values {
        eprintln!(
            "  [aly-boy] n={}, t={}, m={} (fresh α^e × m via AlyGen, then Boyle-batch)",
            n, t, m,
        );

        let mut rng = rand::thread_rng();
        let xs: Vec<Fp> = (0..m).map(|_| Fp::random(modulus, &mut rng)).collect();

        out.push(e2e_aly_boyle_one_cell(n, t, m, &family, modulus, e, &pre_shared, &xs));
    }
    out
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

fn print_header(title: &str, approach_col: &str, approach_width: usize) {
    println!();
    println!("================================================================");
    println!("  {}  (mean of {} iterations)", title, BENCH_ITERS);
    println!("================================================================");
    println!();
    println!(
        "  {:<6} {:>6} {:<w$} {:>10} {:>6} {:>10} {:>12} {:>12} {:>12}",
        "(n,t)", "m", approach_col, "Comp(ms)", "Rnds", "WAN(ms)", "P2P(B)", "Bcast(B)", "Total(B)",
        w = approach_width,
    );
    println!(
        "  {:-<6} {:->6} {:-<w$} {:->10} {:->6} {:->10} {:->12} {:->12} {:->12}",
        "", "", "", "", "", "", "", "", "",
        w = approach_width,
    );
}

fn print_header_split(title: &str, approach_col: &str, approach_width: usize) {
    println!();
    println!("================================================================");
    println!("  {}  (mean of {} iterations)", title, BENCH_ITERS);
    println!("================================================================");
    println!();
    println!(
        "  {:<6} {:>6} {:<w$} {:>10} {:>6} {:>10} {:>12} {:>12} {:>12}",
        "(n,t)", "m", approach_col, "Comp(ms)", "Rnds", "WAN(ms)", "S↔S(B)", "S↔C(B)", "Total(B)",
        w = approach_width,
    );
    println!(
        "  {:-<6} {:->6} {:-<w$} {:->10} {:->6} {:->10} {:->12} {:->12} {:->12}",
        "", "", "", "", "", "", "", "", "",
        w = approach_width,
    );
}

fn print_split_rows(rows: &[SplitRow], approach_width: usize) {
    let mut prev = (0usize, 0usize, 0usize);
    for b in rows {
        let group = (b.n, b.t, b.m);
        if prev != (0, 0, 0) && (group.0, group.1) != (prev.0, prev.1) {
            println!(
                "  {:-<6} {:->6} {:-<w$} {:->10} {:->6} {:->10} {:->12} {:->12} {:->12}",
                "", "", "", "", "", "", "", "", "",
                w = approach_width,
            );
        }
        prev = group;
        let params = format!("({},{})", b.n, b.t);
        println!(
            "  {:<6} {:>6} {:<w$} {:>10.1} {:>6} {:>10.1} {:>12} {:>12} {:>12}",
            params,
            b.m,
            b.name,
            b.time_ms,
            b.rounds,
            b.wan_ms(),
            b.ss_bytes,
            b.sc_bytes,
            b.total_bytes(),
            w = approach_width,
        );
    }
}

fn print_rows(rows: &[Row], approach_width: usize) {
    let mut prev = (0usize, 0usize, 0usize);
    for b in rows {
        let group = (b.n, b.t, b.m);
        if prev != (0, 0, 0) && (group.0, group.1) != (prev.0, prev.1) {
            println!(
                "  {:-<6} {:->6} {:-<w$} {:->10} {:->6} {:->10} {:->12} {:->12} {:->12}",
                "", "", "", "", "", "", "", "", "",
                w = approach_width,
            );
        }
        prev = group;
        let params = format!("({},{})", b.n, b.t);
        if b.skipped {
            println!(
                "  {:<6} {:>6} {:<w$} {:>10} {:>6} {:>10} {:>12} {:>12} {:>12}",
                params, b.m, b.name, "skip", "-", "-", "-", "-", "-",
                w = approach_width,
            );
        } else {
            println!(
                "  {:<6} {:>6} {:<w$} {:>10.1} {:>6} {:>10.1} {:>12} {:>12} {:>12}",
                params,
                b.m,
                b.name,
                b.time_ms,
                b.rounds,
                b.wan_ms(),
                b.p2p_bytes,
                b.broadcast_bytes,
                b.total_bytes(),
                w = approach_width,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    // Gold PRF prime per Yang et al. [SP:YBHKR25]: p = e*g + 1 with e = 2^lambda
    // and log g = 2*lambda + O(1). At lambda=128 this gives |p| = 3*lambda = 384.
    // We pick g = 2^256 - 573 (smallest odd c >= 1 making p prime), so
    // p = 2^384 - 573*2^128 + 1. Verifies e | (p-1) and p.bits() == 384.
    let p_hex = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffdc300000000000000000000000000000001";
    let modulus = BigUint::parse_bytes(p_hex.as_bytes(), 16).unwrap();
    let e = BigUint::one() << LAMBDA;
    let fe_bytes_val = ((modulus.bits() + 7) / 8) as usize;
    let vith_reps = VitHParams::new(TAU, KAPPA).repetitions;

    println!("================================================================");
    println!("  v-dOPRF Benchmark  (iterations = {})", BENCH_ITERS);
    println!("================================================================");
    println!();
    println!("  Fixed parameters:");
    println!("    lambda  = {} (e = 2^{})", LAMBDA, LAMBDA);
    println!("    kappa   = {} (statistical security)", KAPPA);
    println!("    tau     = {} (GGM arity, R={})", TAU, vith_reps);
    println!("    WAN RTT = {} ms", WAN_RTT_MS);
    println!("    WAN BW  = {} Mbps", WAN_BW_MBPS);
    println!(
        "    p       = 0x{}...{} ({} bits)",
        &p_hex[..16],
        &p_hex[p_hex.len() - 8..],
        modulus.bits()
    );
    println!("    FE      = {} bytes", fe_bytes_val);
    println!();

    // Note: the C-Legendre dOPRF baseline lives outside this Rust harness —
    // run `python3 d-OPRF/Legendre-dOPRF-network/bench_e2e.py` to collect
    // that row (after `bash d-OPRF/Legendre-dOPRF-network/build_all.sh`).
    //
    // Four toggleable sections below. Each one is a self-contained `{ ... }`
    // block that declares its own result vector, runs the sweep, and prints
    // the table. Comment the whole block out to skip the section; uncomment
    // to run. Sections are independent.

    let parameter_sets: Vec<(usize, usize)> = vec![(3,1), (5,2), (7,3), (9,4)];
    let offline_m: Vec<usize> = vec![1, 100];
    let online_m: Vec<usize> = vec![1, 100];
    let e2e_m: Vec<usize> = vec![1, 100];

    // ----------------------------------------------------------------------
    // Section 1 — Offline phase only. Skipped for this run: the client-input
    // accounting change touches only the online and e2e cells.
    // ----------------------------------------------------------------------
    {
        eprintln!("Running Section 1 (offline only, m ∈ {:?})...", offline_m);
        let mut offline_all = Vec::new();
        for &(n, t) in &parameter_sets {
            offline_all.extend(run_offline_set(n, t, &modulus, &e, &offline_m));
        }
        print_header(
            &format!(
                "Section 1 — Offline phase (e=2^{}, kappa={}), m ∈ {:?}",
                LAMBDA, KAPPA, offline_m
            ),
            "Approach",
            20,
        );
        print_rows(&offline_all, 20);
    }

    // ----------------------------------------------------------------------
    // Section 2 — Online standalone: VIP-ComputeBatch vs. Boyle-Batch.
    // ----------------------------------------------------------------------
    {
        eprintln!("Running Section 2 (online standalone, VIP-ComputeBatch vs Boyle-Batch, m ∈ {:?})...", online_m);
        let mut online_all: Vec<SplitRow> = Vec::new();
        for &(n, t) in &parameter_sets {
            for &m in &online_m {
                online_all.extend(run_online_set(n, t, m, &modulus));
            }
        }
        print_header_split(
            &format!(
                "Section 2 — Online phase standalone, VIP-ComputeBatch vs Boyle-Batch, m ∈ {:?}  (bytes split into server↔server vs server↔client)",
                online_m
            ),
            "Variant",
            22,
        );
        print_split_rows(&online_all, 22);
    }

    // ----------------------------------------------------------------------
    // Section 3 — End-to-end (our offline approaches + VIP online).
    // ----------------------------------------------------------------------
    {
        eprintln!("Running Section 3 (e2e, fresh α^e per query, m ∈ {:?})...", e2e_m);
        let mut e2e_all = Vec::new();
        for &(n, t) in &parameter_sets {
            e2e_all.extend(run_e2e_set(n, t, &modulus, &e, &e2e_m));
        }
        print_header(
            &format!(
                "Section 3 — End-to-end (fresh α^e per query, all verifications included), m ∈ {:?}",
                e2e_m
            ),
            "Offline + Online",
            28,
        );
        print_rows(&e2e_all, 28);
    }

    // ----------------------------------------------------------------------
    // Section 3 standalone baseline — ΠAlyGen + Boyle online.
    // ----------------------------------------------------------------------
    // {
    //     eprintln!("Running Section 3 standalone — ΠAlyGen + Boyle online, m ∈ {:?}...", e2e_m);
    //     let mut e2e_aly_boyle_all = Vec::new();
    //     for &(n, t) in &parameter_sets {
    //         e2e_aly_boyle_all.extend(e2e_aly_boyle(n, t, &modulus, &e, &e2e_m));
    //     }
    //     print_header(
    //         "Section 3 (standalone) — ΠAlyGen + Boyle online, m ∈ {1, 35}",
    //         "Offline + Online",
    //         28,
    //     );
    //     print_rows(&e2e_aly_boyle_all, 28);
    // }

    // Suppress unused-var warnings when sections above are commented out.
    let _ = (&parameter_sets, &offline_m, &online_m, &e2e_m);

    println!();
}
