#!/bin/sh
# Orange Pi 5 (RK3588S): llama-server in embedding mode with the NPU (matmul + flash attention).
# usage: MODEL=path/to/model.gguf tools/opi5/run-server.sh [extra llama-server args]
# test:  curl -s http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' -d '{"input": "hello"}'
set -e
: "${MODEL:?set MODEL=path/to/model.gguf}"
BIN=${BIN:-./build/bin}
CTX=${CTX:-2048}
ulimit -n 65536
RKNPU_FA=${RKNPU_FA:-1} exec taskset -c 4-7 "$BIN/llama-server" -m "$MODEL" --embedding --pooling last \
  -c "$CTX" -b "$CTX" -ub "$CTX" -np 1 -t 4 --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" "$@"
