# Adaptive NPU FA tiles

The candidate implementation is in `ggml-rknpu2.cpp`. The clean device worktree is `/home/liyifan/src/rknpu-fixes/fa-adaptive-20261005`, branch `perf/fa-adaptive-tiles`, based on `ec04e7bc3`.

`RKNPU_FA_ADAPTIVE=1` selects smaller query and key buckets for short jobs. The default is on; `RKNPU_FA_ADAPTIVE=0` keeps fixed-size jobs for comparison. Query buckets are 64, 128 and the configured `RKNPU_FA_MT` (normally 256). Smaller configured tiles are respected. Key windows up to 64/128 keys use smaller buckets, while large windows retain the configured bucket.

Disjoint mask ranges can split a query tile when the preceding range has at least 64 rows and its query bucket can shrink by at least half. This avoids adding jobs at boundaries such as 129 rows, where query padding would erase the expected saving. The original mask and softmax semantics remain in place, including generic biases and fully masked rows.

The context cache keeps every shape needed by the current FA node. Unused contexts are evicted across slots before warm-up when required; workers never create a context while another worker can be running. The configured LRU limit may be extended to fit the node's active set plus its buffer allocator context.

Growing buffers retain their old SDK allocations for the process lifetime because cached contexts can still import those fds. Key capacities double as they grow; with a fixed head dimension, the retired byte total is below the active capacity for each growing buffer. Initialization is synced to the device before a buffer is published. A failed allocation or initial sync keeps the previous allocation intact. This uses additional resident memory and trades it for safe shape switching.

Legacy K/V conversion and mandatory B rebinding are preserved. Offset windows still copy into each worker's independent staging buffers. The removed dynamic native-B path is not restored.

Native A/C scratch buffers are also isolated by actual matmul row count within each worker. A single DMA buffer must not alternate between contexts with different native row layouts. Short buffers cover the largest key window used by their row-count group. The numeric regression rejects cross-row-count native buffer aliasing.

Key windows shorter than the configured key bucket use exact-size per-worker K/V staging, including offset zero. Staging allocations are keyed by head dimension and key-window length. This prevents a small matmul from binding the oversized global KV cache after maximum-shape warm-up.

Tests:

```sh
python3 tests/rknpu2/test-adaptive-fa.py \
    ggml/src/ggml-rknpu2/ggml-rknpu2.cpp \
    ggml/src/ggml-rknpu2/libs/include
```

The mock regression executes the actual production FA function on ARM64, compares it with an fp32 reference, checks padded work for short/multi-sequence cases and rejects any context creation after the run phase begins. It exercises shape switching with an intentionally small context-cache limit.

The hardware validation script temporarily stops the existing embedding/gateway services, runs comparisons sequentially, and restores their prior active state in `finally`. It inherits a 65536 file-descriptor limit to match production; the SSH default of 1024 is insufficient for loading this model. Run it only with exclusive NPU access. Its embedding test records full vectors alongside timing results for numerical comparison.

The standalone hardware fixture exposed numerical discrepancies during continuous shape switching in the fixed-tile baseline, even after correcting its descriptor limit. Those exploratory results must not be used as a successful performance or correctness claim. Real server comparisons and isolated-shape checks are tracked separately in the final validation report.
