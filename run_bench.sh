#!/bin/bash
# Unified entry point for every benchmark this repo can run:
#   - the Rust harness (`crates/bench`, offline/online/e2e sections + the
#     `our-protocol-verified-input`/`naive-boyle-aly` named experiments —
#     see `cargo run --release -p vdoprf-bench -- --help`), and
#   - the external Legendre-dOPRF baseline (vendored in `d-OPRF/`), wrapped
#     safely by `./run_legendre_baseline.sh` (see that script's header for
#     why it exists rather than calling the vendored tree's own scripts).
#
# Three ways to use this script — see README.md "Which command do I run?"
# for the full explanation:
#
#   (1) ./run_bench.sh full-version [--n N --t T] [--m M1,M2,...]
#       Reproduces every table the *current* full version paper reports
#       (offline Tab. 3, online Tab. 4, e2e Tab. 5) — alias for `all`.
#
#   (2) ./run_bench.sh camera-ready [--n N --t T] [--m M1,M2,...]
#       Reproduces just the end-to-end table the CCS camera-ready paper
#       prints (its Table 2 — row-identical to the full version's Table 5;
#       the camera-ready paper omits the per-phase breakdowns for space) —
#       alias for `e2e`.
#
#   (3) ./run_bench.sh offline|online|our-protocol-verified-input|naive-boyle-aly|legendre-dOPRF [flags]
#       Individual benchmarking runs — one section/protocol at a time, for
#       poking at a single row instead of reproducing a whole paper table.
#
# `full-version`/`all` and `camera-ready`/`e2e` additionally run the
# external Legendre-dOPRF baseline (both (t,n) pairs, m ∈ {1,100}) as a
# trailing section, since it's the closest external comparison point and
# has no offline/online phase split of its own to feed into (1)/(3) alone.
#
# No arguments: alias for `full-version` — everything.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RUST_ONLY_EXPERIMENTS="offline online our-protocol-verified-input naive-boyle-aly"
# `full-version`/`camera-ready` are the paper-facing aliases; `all`/`e2e`
# are the original underlying names and keep working identically.
RUST_PLUS_LEGENDRE_EXPERIMENTS="full-version camera-ready all e2e"

usage() {
    cat <<EOF
Usage: ./run_bench.sh [target] [flags]

No arguments: alias for 'full-version' — everything, Rust and Legendre-dOPRF.

(1) Full paper version (offline + online + e2e tables; Rust + Legendre-dOPRF):
  ./run_bench.sh full-version [--n N --t T] [--m M1,M2,...]
  (alias: all)

(2) Camera-ready CCS paper (e2e table only; Rust + Legendre-dOPRF):
  ./run_bench.sh camera-ready [--n N --t T] [--m M1,M2,...]
  (alias: e2e)

(3) Individual benchmarking runs (Rust-only, forwarded to
    'cargo run --release -p vdoprf-bench --'):
  ./run_bench.sh offline|online|our-protocol-verified-input|naive-boyle-aly [--n N --t T] [--m M1,M2,...]
  ./run_bench.sh legendre-dOPRF [--tn T,N] [--m 1|100|1,100]
  (forwarded to ./run_legendre_baseline.sh)

  --help   print this message and exit
EOF
}

run_rust_plus_legendre() {
    local experiment="$1"
    shift
    echo "### Running the Rust harness ('$experiment') ###"
    cargo run --release -p vdoprf-bench -- "$experiment" "$@"
    local rust_status=$?
    echo
    echo "### Running the Legendre-dOPRF baseline (both (t,n) pairs, m ∈ {1,100}) ###"
    ./run_legendre_baseline.sh --m 1,100
    local legendre_status=$?
    [ $rust_status -eq 0 ] && [ $legendre_status -eq 0 ]
}

if [ $# -eq 0 ]; then
    run_rust_plus_legendre full-version
    exit $?
fi

case "$1" in
    --help|-h)
        usage
        exit 0
        ;;
    legendre-dOPRF)
        shift
        exec ./run_legendre_baseline.sh "$@"
        ;;
    *)
        for exp in $RUST_PLUS_LEGENDRE_EXPERIMENTS; do
            if [ "$1" = "$exp" ]; then
                run_rust_plus_legendre "$@"
                exit $?
            fi
        done
        for exp in $RUST_ONLY_EXPERIMENTS; do
            if [ "$1" = "$exp" ]; then
                exec cargo run --release -p vdoprf-bench -- "$@"
            fi
        done
        echo "error: unknown experiment '$1'" >&2
        usage >&2
        exit 1
        ;;
esac
