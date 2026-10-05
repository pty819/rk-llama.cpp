# RKNPU single-copy projection fusion

Qwen3 models can enable static QKV and gate/up fusion with `RKNPU_FUSE_QKV=1` and `RKNPU_FUSE_GATE_UP=1`. Both flags default to off and are read during model loading. Changing them requires a model reload. Adaptive FA defaults to on; `RKNPU_FA_ADAPTIVE=0` disables it.

## Weight storage

The loader concatenates GGUF rows once, then uses the existing static RKNPU packing and Hadamard pipeline. QKV rows are ordered Q/K/V; FFN rows are ordered gate/up. No GGUF conversion or per-forward weight concatenation is needed.

Only fused base weights are allocated and uploaded. Original projections retain names, types and dimensions for adapter lookup and validation. A separate allocation context excludes their storage, and both metadata contexts live with the model. Allocation estimates also exclude the original storage.

Fusion requires compatible contiguous F16/Q8_0 matrices in the same RKNPU buffer. Mixed pipeline patterns, unsupported shapes/types, QKV bias/scale and other architectures retain their original path. This initial single-copy implementation is limited to Qwen3, whose graph consumes both fused projection groups.

LoRA updates are added to the individual output slices before the existing clamp, bias, scale and activation operations. Adapter weights for metadata-only base tensors use CPU storage. Enabling or disabling an already-loaded adapter does not restore a duplicate base allocation.

Dynamic FA K/V still use legacy B conversion and rebinding. The broken dynamic native-B path remains disabled.

## Hardware regression

The fixture is `/home/liyifan/models/jina-v5/v5-small-retrieval-Q8_0.gguf`: 28 Qwen3 layers, hidden size 1024, FFN size 3072, Q8_0 weights. Run serially on an otherwise idle NPU. The harness stops the user embedding/gateway services and restores their previous running state in `finally`.

```sh
cmake --build build -j 4 --target llama-server llama-embedding
g++ -std=c++17 -Iggml/include tests/rknpu2/make-zero-lora.cpp -Lbuild/bin -lggml-base -Wl,-rpath,"$PWD/build/bin" -o /tmp/make-projection-lora
/tmp/make-projection-lora /tmp/projection-lora.gguf nonzero
FUSION_DIR=/tmp/fusion-results FUSION_MODES=base,qkv,gate,both,nommap,lora-base,lora FUSION_LENGTHS=53,257,1023,2048 FUSION_LORA=/tmp/projection-lora.gguf FUSION_TOGGLE=1 python3 tests/rknpu2/validate-fusion.py
python3 tests/rknpu2/check-fusion-memory.py /tmp/fusion-results
```

The adapter fixture covers Q, K, V, gate and up with nonzero rank-4 weights. The harness checks embeddings against the unfused reference, verifies the adapter has a measurable effect and tests runtime scale 1 -> 0 -> 1. Global server adapter changes retain old prompt caches; the toggle test erases the slot cache after changing the adapter. A direct core API regression (`test-projection-lora.cpp`) also switches scales 1 -> 0 -> 1 with `llama_memory_clear` and requires a zero-error return within 1e-5. Both fused and unfused paths passed with max error 0 and a nonzero adapter effect.

The harness also checks real profiler projection counts:

| Configuration | NPU weight projections per forward |
|---|---:|
| Original | 196 |
| QKV fusion | 140 |
| Gate/up fusion | 168 |
| Both, including active LoRA | 112 |

The memory regression is specific to this fixture: every configuration must report 448 MiB of static RKNPU weights. The earlier duplicate-storage implementation added 112 MiB for QKV and 168 MiB for gate/up, totaling 280 MiB. The new fused allocation matches the unfused allocation. RSS and total DMA-buf usage are recorded separately; they include other buffers and are not asserted to drop by exactly 280 MiB.

Release server/embedding builds and all 93 adaptive FA numerical/layout/ownership checks passed on 2026-10-05. The seven-configuration hardware matrix has 112 embedding samples at 53, 257, 1023 and 2048 tokens, including mmap/non-mmap loading and nonzero LoRA. Minimum reference cosine was 0.9999637188. Both server toggle tests returned max error 0 after slot-cache erasure. Short-input speedups were observed in earlier unprofiled measurements, but long-input timing varied; fusion remains opt-in. These checks do not establish all-architecture support or a long-duration soak result.

Evidence on 21 is under `/home/liyifan/src/rknpu-fixes/projection-memory-evidence-20261005` and `projection-memory-final-lora-20261005`. Production deployment is separate from merging this code.
