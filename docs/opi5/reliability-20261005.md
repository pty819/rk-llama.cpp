# Orange Pi reliability fixes

The 2026-10-05 repair resets the NPU FA maximum for every job, uses fp32 PV accumulation with fp16 storage, and propagates RKNN graph errors independently of NDEBUG. Graph exits drain overlap workers; void upload failures abort because that callback has no status return. QK arithmetic is unchanged.

The production baseline records the existing CPU FA, calibration and model-graph changes separately from these repairs. Regressions use CPU RKNN mocks and do not require an NPU job.

On an ARM64 host with a compiler and sanitizer runtime:

```sh
CXX=g++ python3 tests/rknpu2/test-fa-max-reset.py
CXX=g++ python3 tests/rknpu2/test-cpu-fa-pv-fp32.py
python3 tests/rknpu2/test-fault-propagation.py build
```

The fault test expects an existing Linux shared RKNPU backend build. It uses that build's compiler flags and supplies mock RKNN entry points. It checks graph failures before result collection, pending-worker cleanup, and unconditional Release-mode failure for void uploads.

The optional poll adapter is built separately:

```sh
cc -Wall -Wextra -Werror -O2 -fPIC -shared tools/opi5/rknn_poll_deadline.c -ldl -pthread -o rknn_poll_deadline.so
```

Only poll callers from librknnrt.so (including versioned filenames) receive deadline retries. Negative or excessive timeouts are capped by RKNPU_POLL_MAX_MS (default 30000 ms); RKNPU_POLL_DEADLINE_DISABLE=1 restores passthrough. Preserve any existing LD_PRELOAD entries when enabling it. This bounds intercepted poll waits, not all possible closed-source deadlocks.

Deployment should use a pinned source commit, preserve the previous binaries/configuration, stop all NPU consumers before a module switch, and verify finite normalized embeddings plus driver/IRQ health after restart. Hardware stress and throughput results are separate from the mocked regression results.
