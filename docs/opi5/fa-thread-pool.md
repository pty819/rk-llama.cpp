# Persistent FA workers

Set `RKNPU_FA_POOL=1` to use a process-wide NPU FA worker pool. It defaults to off because hardware A/B did not establish a reliable inference latency gain. `RKNPU_FA_THREADS` selects total concurrency once (default 6, maximum 32). The caller runs slot 0; the pool owns the remaining workers. `RKNPU_FA_POOL=0` retains the previous create/join-per-node path for comparison and rollback.

Workers retain stable slot IDs, RKNN context keys and their existing private staging buffers. Each worker reserves its first job to prevent wake-up order from starving NPU cores; remaining jobs use the existing dynamic atomic queue. Numerical kernels and legacy-B conversion/rebinding are unchanged. Workers inherit the first caller's CPU affinity and sleep on a condition variable when idle. Normal process/library teardown wakes and joins them.

Dispatch publishes a stack callback and waits for every worker before returning. Exceptions are collected and rethrown only after this barrier; FA then uses its existing error/fallback policy. Partial thread creation failure joins the workers already started. A compute mutex serializes the existing global FA buffers and caches before preparation as well as dispatch.

## Validation

```sh
cmake --build build -j 4 --target llama-server llama-embedding
python3 tests/rknpu2/test-fa-pool.py ggml/src/ggml-rknpu2/ggml-rknpu2.cpp
FA_POOL_TEST_THREADS=1,2,3,4,6 RKNPU_FA_POOL=1 python3 tests/rknpu2/test-adaptive-fa.py ggml/src/ggml-rknpu2/ggml-rknpu2.cpp ggml/src/ggml-rknpu2/libs/include
RKNPU_FA_POOL=0 python3 tests/rknpu2/test-adaptive-fa.py ggml/src/ggml-rknpu2/ggml-rknpu2.cpp ggml/src/ggml-rknpu2/libs/include
FA_POOL_BENCH=1 taskset -c 4-7 python3 tests/rknpu2/test-fa-pool.py ggml/src/ggml-rknpu2/ggml-rknpu2.cpp
```

The worker-pool test extracts the production class and checks stable worker identities, completion before callback destruction, exception recovery, concurrent dispatch callers, repeated idle teardown and partial thread creation failure. ASan/UBSan and ThreadSanitizer runs passed. The ARM64 FA fixture passed 480 checks across five worker counts and 96 checks with the pool disabled, including injected SDK failures followed by successful retries.

The real-model harness (`validate-fa-pool.py`) is specific to 21's 28-layer jina-v5-small Q8_0 fixture. It counterbalances off/on/on/off processes, compares 112 embeddings at 53/129/257/547/1023/2048 tokens and a final 53-token return, and restores the production embedding/gateway services in `finally`. It checks five stable `rknpu-fa` task IDs plus the caller across all nodes and shape changes, zero idle worker CPU ticks, and graceful server shutdown.

CPU governor policy must be held constant when interpreting timings. On 21, `ondemand` drops the big cores to 600 MHz when idle; removing thread creation load can change frequency ramp behavior. Initial attempts using an entirely shared first-job queue had regressions at some sizes, including a fixed-governor run. Do not infer universal latency gains from the reduced dispatch cost. A trial that pinned job queues to NPU cores regressed and was removed; first-job reservation followed by dynamic scheduling is used.

An ARM64 six-worker empty-dispatch microbenchmark measured approximately 25 us for the pool versus 335 us for create/join in one six-slot test. Timings depend on host load and CPU frequency. This is not the inference latency improvement.

No dynamic FA native-B path was restored. No CI/CD, driver, production binary or permanent CPU policy changes are included.

## Final hardware result (2026-10-05)

The final caller-participation implementation passed 112 embedding requests with minimum reference cosine 0.9999580672, five stable worker IDs, correct 4-7 affinity, zero idle CPU ticks and graceful teardown. Warm A/B medians (three samples per process, two processes per mode) were:

| Tokens | Create/join ms | Pool ms | Change |
|---:|---:|---:|---:|
| 53 | 150.12 | 127.14 | -15.3% |
| 129 | 321.53 | 375.41 | +16.8% |
| 257 | 490.47 | 513.76 | +4.8% |
| 547 | 1146.26 | 1210.35 | +5.6% |
| 1023 | 2388.37 | 2587.92 | +8.4% |
| 2048 | 5767.30 | 5867.35 | +1.7% |
| Return to 53 | 130.06 | 140.40 | +7.9% |

Consequently this is an opt-in execution option, not a demonstrated general performance optimization. The exact cause of the hardware regressions remains unresolved; frequency and affinity checks did not explain them fully. Both production services and the original `ondemand` governor were restored after tests. Raw final evidence is `/home/liyifan/src/rknpu-fixes/fa-pool-caller-final-evidence-20261005`.
