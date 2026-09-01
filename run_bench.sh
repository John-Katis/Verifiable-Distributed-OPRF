#!/bin/bash
# Unified entry point for every benchmark this repo can run:
#   - the Rust harness (`crates/bench`, offline/online/e2e sections + the
#     `our-protocol-verified-input`/`naive-boyle-aly` named experiments —
#     see `cargo run --release -p vdoprf-bench -- --help`), and
#   - the external Legendre-dOPRF baseline (`d-OPRF` submodule), wrapped
#     safely by `./run_legendre_baseline.sh` (see that script's header for
#     why it exists rather than calling the submodule's own scripts).
#
# Usage:
#   ./run_bench.sh
#       No arguments: alias for `all` (below) — everything.
#
#   ./run_bench.sh all|e2e [--n N --t T] [--m M1,M2,...]
#       Runs the Rust harness, THEN also runs the Legendre-dOPRF baseline
#       (both (t,n) pairs, m ∈ {1,100}) as a trailing section. `all`/`e2e`
#       are the only two Rust experiments compared against Legendre-dOPRF,
#       since it has no offline/online phase split of its own to compare
#       against — only a full end-to-end query.
#
#   ./run_bench.sh offline|online|our-protocol-verified-input|naive-boyle-aly [--n N --t T] [--m M1,M2,...]
#       Forwards to `cargo run --release -p vdoprf-bench -- <name> [flags]`.
#       Rust-only — no Legendre section.
#
#   ./run_bench.sh legendre-dOPRF [--tn T,N] [--m 1|100|1,100]
#       Forwards to `./run_legendre_baseline.sh [flags]`.
#
#   ./run_bench.sh --help
#       Print this message (and the Rust harness's own --help).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RUST_ONLY_EXPERIMENTS="offline online our-protocol-verified-input naive-boyle-aly"
RUST_PLUS_LEGENDRE_EXPERIMENTS="all e2e"

usage() {
    cat <<EOF
Usage: ./run_bench.sh [experiment] [flags]

No arguments: alias for 'all' — everything, Rust and Legendre-dOPRF.

Rust + Legendre-dOPRF (Rust harness, then run_legendre_baseline.sh --m 1,100):
  $RUST_PLUS_LEGENDRE_EXPERIMENTS
  flags: [--n N --t T] [--m M1,M2,...]  (apply to the Rust side only)

Rust-only (forwarded to 'cargo run --release -p vdoprf-bench --'):
  $RUST_ONLY_EXPERIMENTS
  flags: [--n N --t T] [--m M1,M2,...]

External baseline only:
  legendre-dOPRF [--tn T,N] [--m 1|100|1,100]
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
    run_rust_plus_legendre all
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
