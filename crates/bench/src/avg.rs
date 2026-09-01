//! 10-iteration averaging for bench cells.
//!
//! Each bench measurement runs `BENCH_ITERS` times and reports the mean of
//! time (ms) and of each comm-stats component (rounded to integer for
//! bytes/rounds, per the user spec: "for bytes please allow no decimals").

use vdoprf_network::CommStats;

pub const BENCH_ITERS: usize = 10;

/// Mean time and comm stats over `BENCH_ITERS` iterations. `p2p_messages`
/// and `broadcast_messages` are kept for completeness even when the bench
/// printer doesn't currently show them.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct AvgStats {
    pub time_ms: f64,
    pub rounds: usize,
    pub p2p_bytes: usize,
    pub p2p_messages: usize,
    pub broadcast_bytes: usize,
    pub broadcast_messages: usize,
    pub client_bytes: usize,
    pub client_messages: usize,
}

impl AvgStats {
    /// P2P + broadcast + server→client mean bytes. Used by the avg-module
    /// tests and by `Row::total_bytes` so WAN(ms) includes the transfer
    /// term for client delivery.
    #[allow(dead_code)]
    pub fn total_bytes(&self) -> usize {
        self.p2p_bytes + self.broadcast_bytes + self.client_bytes
    }
}

/// Same shape as [`AvgStats`] but with the bytes split into the two
/// **physical directions of communication** that Section 2 of the bench
/// reports on:
///
///   - `ss_bytes` — server↔server (everything the protocol crate charges
///     via its internal `SimulatedNetwork`, i.e. the `CommStats` returned
///     from `compute_batch` / `compute_boyle_batch`). Covers VSS,
///     commit-then-hash broadcasts, RSS.Mul aggregator rounds, etc.
///   - `sc_bytes` — server↔client (input distribution from client to
///     servers plus output/proof from servers to client).
///
/// `rounds` is the sum of both halves (sequential composition), matching
/// how the non-split path accounts rounds via `CommStats::merge`.
#[derive(Clone, Debug, Default)]
pub struct AvgStatsSplit {
    pub time_ms: f64,
    pub rounds: usize,
    pub ss_bytes: usize,
    pub sc_bytes: usize,
}

impl AvgStatsSplit {
    /// Kept for symmetry with [`AvgStats::total_bytes`] and exercised by the
    /// avg-module tests; the bench printer derives the same value via
    /// `SplitRow::total_bytes` on its own struct.
    #[allow(dead_code)]
    pub fn total_bytes(&self) -> usize {
        self.ss_bytes + self.sc_bytes
    }
}

/// Run `f` `BENCH_ITERS` times, return the mean `(time_ms, CommStats)`
/// with bytes/rounds rounded to integers.
pub fn bench_avg<F>(mut f: F) -> AvgStats
where
    F: FnMut() -> (f64, CommStats),
{
    let n = BENCH_ITERS as f64;
    let mut time_sum = 0.0f64;
    let mut p2p_bytes = 0u128;
    let mut p2p_messages = 0u128;
    let mut broadcast_bytes = 0u128;
    let mut broadcast_messages = 0u128;
    let mut client_bytes = 0u128;
    let mut client_messages = 0u128;
    let mut rounds = 0u128;

    for _ in 0..BENCH_ITERS {
        let (t, c) = f();
        time_sum += t;
        p2p_bytes += c.p2p_bytes as u128;
        p2p_messages += c.p2p_messages as u128;
        broadcast_bytes += c.broadcast_bytes as u128;
        broadcast_messages += c.broadcast_messages as u128;
        client_bytes += c.client_bytes as u128;
        client_messages += c.client_messages as u128;
        rounds += c.rounds as u128;
    }

    // Round means to integers for byte/round counts. Accumulator → f64 → round.
    let round_u = |total: u128| -> usize {
        ((total as f64) / n).round() as usize
    };

    AvgStats {
        time_ms: time_sum / n,
        rounds: round_u(rounds),
        p2p_bytes: round_u(p2p_bytes),
        p2p_messages: round_u(p2p_messages),
        broadcast_bytes: round_u(broadcast_bytes),
        broadcast_messages: round_u(broadcast_messages),
        client_bytes: round_u(client_bytes),
        client_messages: round_u(client_messages),
    }
}

/// Run `f` `BENCH_ITERS` times collecting one [`CommStats`] per iteration
/// and split it into the two physical axes: server↔server (p2p+broadcast)
/// and server↔client (`client_bytes`, set by the protocol when it calls
/// `SimulatedNetwork::send_to_client`).
pub fn bench_avg_split<F>(mut f: F) -> AvgStatsSplit
where
    F: FnMut() -> (f64, CommStats),
{
    let n = BENCH_ITERS as f64;
    let mut time_sum = 0.0f64;
    let mut ss_bytes = 0u128;
    let mut sc_bytes = 0u128;
    let mut rounds = 0u128;

    for _ in 0..BENCH_ITERS {
        let (t, c) = f();
        time_sum += t;
        ss_bytes += (c.p2p_bytes + c.broadcast_bytes) as u128;
        sc_bytes += c.client_bytes as u128;
        rounds += c.rounds as u128;
    }

    let round_u = |total: u128| -> usize { ((total as f64) / n).round() as usize };

    AvgStatsSplit {
        time_ms: time_sum / n,
        rounds: round_u(rounds),
        ss_bytes: round_u(ss_bytes),
        sc_bytes: round_u(sc_bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_stats(p2p: usize, bcast: usize, rounds: usize) -> CommStats {
        CommStats {
            p2p_bytes: p2p,
            p2p_messages: 1,
            broadcast_bytes: bcast,
            broadcast_messages: 1,
            rounds,
            client_bytes: 0,
            client_messages: 0,
        }
    }

    /// Constant input across iterations: mean equals the constant exactly.
    #[test]
    fn bench_avg_constant_time_and_stats() {
        let c = constant_stats(1024, 2048, 3);
        let avg = bench_avg(|| (42.5, c.clone()));
        assert!((avg.time_ms - 42.5).abs() < 1e-9);
        assert_eq!(avg.rounds, 3);
        assert_eq!(avg.p2p_bytes, 1024);
        assert_eq!(avg.broadcast_bytes, 2048);
        assert_eq!(avg.total_bytes(), 3072);
    }

    /// Bytes are rounded to integers — half-up rounding via f64::round.
    #[test]
    fn bench_avg_byte_rounding_no_decimals() {
        // Five iters of 100 bytes, five of 101 bytes → mean 100.5 → rounds up to 101.
        let mut call = 0;
        let avg = bench_avg(|| {
            call += 1;
            let p = if call <= 5 { 100 } else { 101 };
            (0.0, constant_stats(p, 0, 0))
        });
        assert_eq!(avg.p2p_bytes, 101, "100.5 mean must round to 101 (no decimals)");
    }

    /// Linear inputs: mean(1..=10) = 5.5 ms, mean of byte sequence 10..=100 (step 10) = 55.
    #[test]
    fn bench_avg_linear_sequence() {
        let mut call = 0;
        let avg = bench_avg(|| {
            call += 1;
            let time = call as f64;
            let bytes = call * 10;
            (time, constant_stats(bytes, 0, 0))
        });
        assert!((avg.time_ms - 5.5).abs() < 1e-9);
        assert_eq!(avg.p2p_bytes, 55);
    }

    /// Empty-ish stats round-trip: zeros in → zeros out.
    #[test]
    fn bench_avg_zero_inputs_yields_zero() {
        let avg = bench_avg(|| (0.0, CommStats::default()));
        assert_eq!(avg.time_ms, 0.0);
        assert_eq!(avg.rounds, 0);
        assert_eq!(avg.total_bytes(), 0);
    }

    /// BENCH_ITERS default is 10 — documented in the module header and relied
    /// on by bench captions ("mean of N iterations").
    #[test]
    fn bench_iters_is_ten() {
        assert_eq!(BENCH_ITERS, 10);
    }

    /// Split averager reads ss_bytes from p2p+broadcast and sc_bytes from
    /// `client_bytes` of the SAME `CommStats` (`bench_avg_split` takes one
    /// merged `CommStats` per iteration, not two separate ones — matching
    /// how real call sites build it via `.merge()` before passing it in).
    /// Constant input across iterations round-trips.
    #[test]
    fn bench_avg_split_constant() {
        let merged = CommStats {
            p2p_bytes: 300,
            p2p_messages: 1,
            broadcast_bytes: 700, // ss total = 300 + 700 = 1000
            broadcast_messages: 1,
            rounds: 3, // as if two sub-networks (2 rounds + 1 round) were merged
            client_bytes: 100, // sc total = 100
            client_messages: 1,
        };
        let avg = bench_avg_split(|| (12.5, merged.clone()));
        assert!((avg.time_ms - 12.5).abs() < 1e-9);
        assert_eq!(avg.ss_bytes, 1000);
        assert_eq!(avg.sc_bytes, 100);
        assert_eq!(avg.rounds, 3);
        assert_eq!(avg.total_bytes(), 1100);
    }

    /// Closure is invoked exactly BENCH_ITERS times.
    #[test]
    fn bench_avg_invokes_closure_bench_iters_times() {
        let mut call = 0;
        let _ = bench_avg(|| {
            call += 1;
            (0.0, CommStats::default())
        });
        assert_eq!(call, BENCH_ITERS);
    }
}
