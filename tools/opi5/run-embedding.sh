#!/bin/sh
# Orange Pi 5 (RK3588S): NPU embedding with the settings used for the measurements in docs/opi5/README.md.
# usage: MODEL=path/to/model.gguf tools/opi5/run-embedding.sh input.txt [extra llama-embedding args]
set -e
: "${MODEL:?set MODEL=path/to/model.gguf}"
BIN=${BIN:-./build/bin}
CTX=${CTX:-2048}
IN=${1:?input file}; shift
ulimit -n 65536
RKNPU_FA=${RKNPU_FA:-1} exec taskset -c 4-7 "$BIN/llama-embedding" -m "$MODEL" -f "$IN" \
  --pooling last --embd-normalize 2 --embd-output-format array --embd-separator '<#embd#>' \
  -c "$CTX" -b "$CTX" -ub "$CTX" -t 4 "$@"
