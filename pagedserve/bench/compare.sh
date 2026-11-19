#!/usr/bin/env bash
# Runs the two A/B comparisons behind the project's headline numbers:
#   1. decode throughput: continuous batching vs static batching
#   2. TTFT on a shared-prefix multi-turn workload: prefix caching on vs off
#
# Usage: bench/compare.sh <model_dir> [extra pagedserve args...]
# Env:   BUILD=build PORT=8123 OUT=bench_results
#        THROUGHPUT_ARGS="--num-requests 300 --output-len 16:512"
#        MULTITURN_ARGS="--conversations 32 --turns 6"
set -euo pipefail
MODEL=${1:?usage: bench/compare.sh <model_dir> [server args...]}
shift
BUILD=${BUILD:-build}
PORT=${PORT:-8123}
OUT=${OUT:-bench_results}
THROUGHPUT_ARGS=${THROUGHPUT_ARGS:-"--num-requests 200 --prompt-len 32:256 --output-len 16:512"}
MULTITURN_ARGS=${MULTITURN_ARGS:-"--conversations 32 --turns 5"}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

PID=""
start_server() {  # $1 = tag, rest = extra args
  local tag=$1
  shift
  "$BUILD/pagedserve" --model "$MODEL" --host 127.0.0.1 --port "$PORT" "$@" >"$OUT/server_$tag.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then return 0; fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "server failed to start, see $OUT/server_$tag.log" >&2
      exit 1
    fi
    sleep 0.5
  done
  echo "server did not become healthy" >&2
  exit 1
}
stop_server() {
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
}
trap stop_server EXIT

bench() {  # $1 = tag, $2 = scenario, rest = bench args
  local tag=$1 scenario=$2
  shift 2
  echo ">>> $tag"
  python3 "$HERE/bench_serving.py" --base-url "http://127.0.0.1:$PORT" --scenario "$scenario" "$@" \
    --out "$OUT/$tag.json" >/dev/null
}

# shellcheck disable=SC2086
{
  start_server continuous "$@";             bench throughput_continuous throughput $THROUGHPUT_ARGS; stop_server
  start_server static --static-batching "$@"; bench throughput_static throughput $THROUGHPUT_ARGS;     stop_server
  start_server prefix_on "$@";              bench multiturn_prefix_on multiturn $MULTITURN_ARGS;      stop_server
  start_server prefix_off --no-prefix-caching "$@"; bench multiturn_prefix_off multiturn $MULTITURN_ARGS; stop_server
}

python3 - "$OUT" <<'EOF'
import json, sys, os
d = sys.argv[1]
L = lambda n: json.load(open(os.path.join(d, n + ".json")))
c, s = L("throughput_continuous"), L("throughput_static")
on, off = L("multiturn_prefix_on"), L("multiturn_prefix_off")
print("\n== Decode throughput (output tokens/s)")
fmt = lambda r: (f"{r['output_tokens_per_s']:10.1f}   {r.get('server_steps', 0):6d} steps, "
                 f"avg decode batch {r.get('avg_decode_batch', 0):5.1f}")
print(f"  static batching      {fmt(s)}")
print(f"  continuous batching  {fmt(c)}")
print(f"  speedup              {c['output_tokens_per_s'] / s['output_tokens_per_s']:10.2f}x   "
      f"({s.get('server_steps', 1) / max(1, c.get('server_steps', 1)):.2f}x fewer steps for the same work)")
print("\n== Multi-turn shared-prefix TTFT")
print(f"  prefix cache off     p50 {off['ttft_p50_ms']:8.1f} ms   p95 {off['ttft_p95_ms']:8.1f} ms")
print(f"  prefix cache on      p50 {on['ttft_p50_ms']:8.1f} ms   p95 {on['ttft_p95_ms']:8.1f} ms"
      f"   (cache hit rate {100 * on['cache_hit_rate']:.1f}%)")
print(f"  p95 reduction        {100 * (1 - on['ttft_p95_ms'] / off['ttft_p95_ms']):8.1f}%")
print(f"\nraw results in {d}/")
EOF
