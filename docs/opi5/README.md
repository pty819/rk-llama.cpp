# Orange Pi 5 (RK3588S) 上的 NPU 加速 embedding：ggml-rknpu2 优化分支

本分支基于 [javer/rk-llama.cpp](https://github.com/javer/rk-llama.cpp)（`b10297-rknpu2`），其 RKNPU2 后端来自
[invisiofficial/rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp)（per-channel 量化的 `ggml-rknpu2` backend，作者 invisiofficial）。
感谢两位原作者，这里的全部工作都建立在他们的代码之上。许可证与上游一致（MIT，见仓库根目录 `LICENSE`）。

目标场景：在 **Orange Pi 5（RK3588S，8 GB）** 上用 `llama-embedding` / `llama-server --embedding` 跑
**jina-embeddings-v5-small（Qwen3-0.6B 架构，Q8_0 GGUF）** 这类 decoder-only embedding 模型（`--pooling last`），
输入长度 100–2000 token。下面所有数字都是在这块板子上实测得到的，没有做外推。

## 1. 改了什么（按 commit 顺序）

| 模块 | 改动 | 开关 |
|---|---|---|
| `ggml-rknpu2` 同步 | 后端源码同步到 invisiofficial `rknpu2` 分支最新版（+ 本 ggml 版本需要的 `set/get_tensor_2d` 空实现） | – |
| `rknpu2-configuration` | Q4_K 默认映射到 `W4A4_HADAMARD`；N 超大的权重（tied lm_head / token_embd，N = n_vocab）留在 CPU | `RKNPU_MAX_N` |
| `ggml-rknpu2` profile 修复 | `supports_op` 只接受权重真正在 RKNPU buffer 里的 MUL_MAT（之前是正确性 bug）；去掉 M 向上取 2 的幂，改为 256 行 M tile（减少无效计算和 context 爆炸）；q/k/v、gate/up 共享同一份 A 量化结果（A reuse）；A 量化向量化、多线程；去掉 dst memset + `+=`；逐阶段计时 | `RKNPU_M_TILE`、`RKNPU_NO_A_REUSE`、`RKNPU_PROFILE` |
| `qwen3` | pooled embedding 模式下不构建 lm_head（Qwen3-0.6B 的 lm_head 151936×1024，每个 batch 白算，原来占 NPU forward 的 83%） | `LLAMA_EMBD_KEEP_LM_HEAD=1` 恢复 |
| `ggml-cpu` | 新的 NEON fp16 flash-attention kernel（K/V 直接读 f16 cache、fp16 FMA 短链 + fp32 累加、heavy-first 调度、mask tile 分类，单层 FA 快约 2.4 倍）；可选的逐 op profiler | `GGML_FA_OPT1`、`GGML_FA_OPT1_PACK`、`GGML_FA_OPT1_QKCHUNK`、`GGML_CPU_OPPROF` |
| `ggml-rknpu2` host compute | RKNPU buffer 报告 `is_host`，CPU↔RKNPU 之间不再做 scheduler 拷贝（每次 forward 省 1.2–2.2 GB memcpy） | `RKNPU_HOST_COMPUTE`（默认 1） |
| `ggml-rknpu2` overlap | A 量化 / C 反量化与 NPU matmul 流水线重叠（实验性，收益约 100–150 ms/forward，默认 **关闭**） | `RKNPU_OVERLAP=1` |
| `ggml-rknpu2` NPU attention | `FLASH_ATTN_EXT` 在 NPU 上跑：Q·Kᵀ 和 P·V 用 NPU fp16 matmul（native A/C layout），mask + softmax 在 CPU（NEON），256 行 query tile、GQA 两个 Q head 叠在 M 上、按真实 mask 跳过全遮蔽块，6 个驱动线程（每个 NPU core 2 个），context 缓存以 256 key 为粒度（任何长度的 context 集合都是最大长度集合的子集），有上限 | `RKNPU_FA=1`（默认关） |
| `ggml-rknpu2` tile 级 key 起点跳过 | 多序列打包的 batch 里，每个 query tile 从它第一条未被遮蔽的 key 开始算（对齐到 tile 粒度），不再从 key 0 开始。k0 > 0 的任务把 K/V 窗口拷进各线程自己的暂存缓冲（fd 偏移视图在两个驱动线程并发时结果会错，不能直接绑偏移）。输出与不跳过时逐字节一致 | `RKNPU_FA_KSKIP`（默认 1；0 = 关） |
| `ggml-rknpu2` B 原生布局绑定 | QK/PV matmul 的 B 直接用 NPU 原生布局（RK3588 fp16：`(N/16, K/32, 16, 32)`），`set_io_mem` 不再在 CPU 上转换 B（实测 4.9 µs vs 186 µs/次）。K 侧原生布局对 Nq 前缀兼容（每个 KV head 一个共享缓冲）；V 侧不兼容（步长依赖 Nq），每个 PV 任务自己交织暂存。原生 B 在 run 时直读内存（板上探针验证），绑定只认缓冲指针。旧路径完整保留 | `RKNPU_FA_NATIVE_B`（默认 1；0 = 旧路径） |
| `ggml-rknpu2` softmax profiler 细分 + 省内存 | `softmax+syncs` 一列拆成 S sync / softmax max / softmax exp+P / generic / P sync 五列（mixed-40：7% / 36% / 53% / 0.1% / 4%）。曾实现逐行范围裁剪的 softmax（值逐位一致、causal tile 少算约一半配对），实测 mixed-40 持平、1975 token 慢约 3%、`RKNPU_FA_THREADS=3` 慢约 12%——每核 2 个提交线程时 CPU softmax 与另一线程的 NPU matmul 重叠，流水线是 NPU 吞吐瓶颈，CPU 侧节省都变成同步等待，故回退（证据留在代码注释）。native 模式下不再分配 legacy vbuf | `RKNPU_PROFILE=1` |
| `llama-context` | 3 行补丁：`flash_attn = auto` 时，如果 FA 节点被分配到 ACCEL 设备（RKNPU），而该层在 CPU 上，不再把 FA 整体关掉（RKNPU 直接在 host 内存上计算） | – |

## 2. 环境要求

- 硬件：Orange Pi 5（RK3588S，8 GB RAM）。big core 是 cpu4-7（A76，最高 2.4 GHz），little core 是 cpu0-3（A55）。
- 内核 / 驱动：rknpu 驱动 **0.9.8**。测试板用的是打过补丁的 DKMS 包 `0.9.8-3~opi5fix1`（内核 7.1.8-edge-rockchip64）。驱动补丁见
  <https://github.com/lurenJBD/rk3588-rknn-core/issues/1>。
- 运行库：librknnrt **2.3.2**。仓库里 `ggml/src/ggml-rknpu2/libs/` 自带（`rknn_matmul_api.h` 标注 2.3.2）。
- 每个 NPU 进程启动前：`ulimit -n 65536`（后端会打开大量 dma-buf fd，上游 README 同样要求）。
- 散热：长时间满载时 SoC 会到 75–80 °C（第一个 trip point 是 85 °C），实测已经出现降频导致的波动。**建议加散热片 + 风扇。**

## 3. 编译

```sh
# NPU build（本分支实测用的配置：Release、GGML_NATIVE=ON、OpenMP=ON 都是默认值）
cmake -B build -DLLAMA_RKNPU2=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-embedding llama-server -j4

# 可选：纯 CPU build，用来对比
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target llama-embedding -j4
```

## 4. 推荐启动命令（和 benchmark 用的完全一致）

### llama-embedding

```sh
ulimit -n 65536
RKNPU_FA=1 taskset -c 4-7 ./build/bin/llama-embedding \
  -m v5-small-retrieval-Q8_0.gguf -f input.txt \
  --pooling last --embd-normalize 2 --embd-output-format array --embd-separator '<#embd#>' \
  -c 2048 -b 2048 -ub 2048 -t 4
```

### llama-server（embedding 模式）

```sh
ulimit -n 65536
RKNPU_FA=1 taskset -c 4-7 ./build/bin/llama-server \
  -m v5-small-retrieval-Q8_0.gguf --embedding --pooling last \
  -c 2048 -b 2048 -ub 2048 -np 1 -t 4 --host 127.0.0.1 --port 8080

curl -s http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": "你好，世界"}'
```

参数说明：
- `taskset -c 4-7` + `-t 4`：只用 4 个 A76 big core。实测 8 线程（含 A55）端到端更慢，显式逐线程 pin（`OMP_PLACES`）也没有收益。
- `-c` / `-b` / `-ub`：pooled embedding 要求整条序列在一个 ubatch 里，所以三者设成一样，并且 ≥ 最长输入的 token 数（取 256 的倍数，比如 512 / 1024 / 2048）。
  llama-embedding 会把多条短输入打包进同一个 batch。
- `-np 1`（server）：如果有多个 slot，`-c` 会被平分给各个 slot，长输入就放不下了。需要并发时，按 `slot 数 × 最长输入` 设置 `-c`。
- `--pooling last`、`--embd-normalize 2`：jina-v5 用 last-token pooling、L2 归一化。
- flash attention 用默认的 `auto` 就行（要靠上面的 3 行 `llama-context` 补丁才能和 `RKNPU_FA=1` 配合；不要加 `-fa off`）。
- **不要加 `--no-mmap`**：默认 `RKNPU_HOST_COMPUTE=1` 时，`--no-mmap` 的加载路径不会调用 `set_tensor`，NPU 的权重打包不会发生，会直接加载失败。
  如果一定要用 `--no-mmap`，请同时设 `RKNPU_HOST_COMPUTE=0`（会恢复旧的拷贝路径，速度更慢）。
  另外：内存紧张时 mmap 的模型页可能被换出，导致个别 forward 卡顿，这是实测到的噪声来源之一。
- 多 NPU 进程 / 省电：参考上游的 `RKNPU_CORES`、`RKNPU_DOMAINS`（见 `ggml/src/ggml-rknpu2/README.md`）。

## 5. 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `RKNPU_FA` | 0 | 1 = `FLASH_ATTN_EXT` 在 NPU 上跑（本分支推荐开启）；0 = 原来的 CPU FA，输出和关闭前逐字节一致 |
| `RKNPU_FA_KSKIP` | 1 | 1 = 多序列 batch 里每个 tile 从自己的第一条 key 开始算（对齐 tile 粒度）；0 = 从 key 0 开始。两种模式输出逐字节一致 |
| `RKNPU_FA_NATIVE_B` | 1 | 1 = K/V 以 NPU 原生布局绑定（set_io_mem 不做 CPU 转换）；0 = 旧路径（TP_NORM/普通布局，每次绑定转换）。启动时自动探测，失败也回旧路径；两种模式输出逐字节一致 |
| `RKNPU_FA_MAX_KV` | 4096 | key 长度超过它时 `supports_op` 拒绝，回退到 CPU FA（上限 8192）。这是最主要的内存上限开关 |
| `RKNPU_FA_MAX_CTX` | 192 | NPU attention context 数量硬上限（每个驱动线程各自做 LRU 淘汰）。设得太小会反复重建 context，明显变慢 |
| `RKNPU_FA_THREADS` | 6 | NPU attention 驱动线程数（每个 NPU core 2 个） |
| `RKNPU_FA_MT` | 256 | query tile 行数（也是 key 范围的分桶粒度） |
| `RKNPU_FA_CHECK` | 0 | 调试：每 k 行算一次 fp32 参考，打印每层 cosine / 最大误差（很慢） |
| `RKNPU_HOST_COMPUTE` | 1 | 1 = RKNPU 直接读写 CPU buffer（零拷贝）；0 = 旧的 scheduler 拷贝路径（`--no-mmap` 时需要） |
| `RKNPU_M_TILE` | 256 | 权重 matmul 的 M 方向 tile 行数 |
| `RKNPU_NO_A_REUSE` | 未设置 | 设置后禁用 q/k/v、gate/up 之间的 A 量化复用（调试用） |
| `RKNPU_MAX_N` | 65536 | N 大于它的权重留在 CPU（lm_head / token_embd）；0 = 不限制 |
| `RKNPU_OVERLAP` | 0 | 1 = 量化/反量化与 NPU 运行重叠（实验性；有 NPU FA 节点的图会自动走串行路径） |
| `RKNPU_OV_THREADS` / `RKNPU_SUBMIT_CPUS` / `RKNPU_SPIN_US` (2000) / `RKNPU_WAIT_SPIN_US` (5000) | – | 仅 overlap 模式：Q/D 的 OMP 线程数、NPU 提交线程的 CPU 亲和性、自旋时间 |
| `RKNPU_PROFILE` | 0 | 1 = 每次 forward 打印逐阶段耗时（matmul 和 NPU FA），不设置时没有开销 |
| `RKNPU_DEVICE` / `RKNPU_HYBRID` / `RKNPU_CORES` / `RKNPU_DOMAINS` | 上游默认 | 上游原有选项，见 `ggml/src/ggml-rknpu2/README.md` |
| `GGML_FA_OPT1` | 2 | CPU FA kernel：0 = 上游 kernel，1 = NEON fp32，2 = NEON fp16（默认） |
| `GGML_FA_OPT1_PACK` | 1 | CPU FA：K/V tile 打包到连续 scratch |
| `GGML_FA_OPT1_QKCHUNK` | 32 | CPU FA：fp16 QK 累加链长度（32/64/128） |
| `GGML_CPU_OPPROF` | 0 | 1 = 退出时打印 CPU 各 op 的总耗时 |
| `LLAMA_EMBD_KEEP_LM_HEAD` | 未设置 | 设置后 pooled embedding 仍然构建 lm_head（旧行为） |

## 6. 实测结果

条件：jina-v5-small-retrieval Q8_0，`llama-embedding`，`taskset -c 4-7 -t 4`，F16 KV cache，flash attention auto。
每个长度的输入喂两遍：第一遍 = cold（含 context 创建），第二遍 = warm。warm tok/s 取交错运行的中位数；177 token 那一列只有一遍 cold。
cosine 是和原始 CPU Q8_0 build 的输出比较（对照输入：10 条短文本 + 3 条长文本）。

### CPU vs NPU（NPU attention 之前，同一轮测试）

| build | 177 tok | 547 tok | 1023 tok | 1975 tok | peak RSS |
|---|---|---|---|---|---|
| CPU（含新的 NEON fp16 FA kernel） | 234 | 220 | 194 | 161 | 2759 MB |
| NPU（matmul 在 NPU，FA 在 CPU） | 217 | 566 | 449 | 318 | 2810 MB |

作为参考，改动之前（上游代码，同一模型，另一轮测试）：CPU 167 / 133 / 116 / 80，NPU 59 / 71 / 102 / 62 tok/s（NPU 当时比 CPU 还慢，主要原因是 lm_head 在 NPU 上白算），peak RSS CPU 3736 MB / NPU 5145 MB。

### NPU attention：`RKNPU_FA` 关 vs 开（另一轮测试，同一轮内交错）

| tokens | 关，warm | 开，warm | 变化 | 关，cold | 开，cold | peak RSS 关 → 开 |
|---|---|---|---|---|---|---|
| 177（只有 cold） | 274 | 255 | −7% | – | – | – |
| 547 | 587 | 589 | ≈ 0 | 322 | 312 | 1813 → 1822 MB |
| 1023 | 501 | 584 | **+17%** | 340 | 362 | 2078 → 2113 MB |
| 1975 | 346 | 471 | **+36%** | 264 | 320 | 2812 → 2885 MB |

### tile key 跳过 + B 原生布局合入后（2026-09-26 晚复测，板温 77–81 °C，同轮交错）

本轮板温比上一轮高约 5 °C，绝对值整体下移（CPU FA 关侧也低约 15%），比率才有可比性。

| tokens | 关，warm | 开，warm | 变化 | peak RSS 关 → 开 |
|---|---|---|---|---|
| 1023 | 392 | 449 | **+15%** | 2078 → 2104–2108 MB |
| 1975 | 293 | 366 | **+25%** | 2809 → 2859 MB（vbuf 不再分配，比上一轮少 +26 MB） |

混合 batch（40 条 100–2000 token 随机文本，19 个 batch，中位数，`RKNPU_FA_THREADS=6`）：

| 模式 | warm tok/s | 说明 |
|---|---|---|
| FA 关（CPU） | 346 | |
| FA 开 + KSKIP=0 | 362 | 与关比 +4.7% |
| FA 开（默认，KSKIP=1 + native B） | **381** | 与关比 **+10%**，与不跳 key 比 +5% |

长输入 `RKNPU_FA_CHECK`（最终二进制）：56/56 层 cosine = 1.000000；混合 batch：532/532 层 cosine = 1.000000。
与 CPU 参考的 cosine 和上一轮相同（长文本 0.9972–0.9975；混合 40 条 on vs off min 0.99115 / mean 0.998，与改动前逐位同签名）。

- 每层 attention：1023 token 时 21.9 ms（CPU FA 34.7 ms），1975 token 时 78 ms（CPU FA 约 125 ms）。
- cosine（和 CPU Q8_0 参考比）：长文本 FA 关 0.99734–0.99740，开 0.99708–0.99725；短文本最低 FA 关 0.99643，开 0.99624。全部 ≥ 0.996，但短文本最低值离门限很近。
- `RKNPU_FA_CHECK` 逐层和 fp32 参考比较：每层 cosine 1.000000，最大绝对误差 ≤ 0.024。
- 测试时板子上有其他常驻进程在做磁盘 I/O，温度 70–80 °C。个别 warm 运行被 I/O 卡住（两种模式都有，60–270 tok/s），这些点没有算进中位数。

## 7. 内存

- 每个 NPU attention context 约 0.3–0.5 MB（96 个 context：VmRSS +36.5 MB，其中 RssShmem +28.4 MB；不占 CMA）。
- attention 缓冲区按见过的最长 key 长度分配一次，之后复用：2048 key 时约 47 MB。native-B 模式下 legacy 的 vbuf 暂存不再分配（2048 key 时省 4 MB，`RKNPU_FA_MAX_KV=4096` 时省 8 MB）。
- 1975 token 时 NPU attention 总共多占约 75–85 MB（peak RSS 2812 → 2885–2897 MB）。
- 上限：context 数 ≤ 6 线程 × 2 × (max_kv / 256)，2048 时是 96，`RKNPU_FA_MAX_KV=4096` 时最多 192（约 80 MB），再加约 90 MB 缓冲区。和输入长度有多少种无关。
- 长时间运行：40 条 100–2000 token 的随机文本（打包成 19 个 batch，每个 batch 1–5 条序列）在同一个进程里跑完。context 数第一个 batch 之后就固定在 96，
  NPU attention 的结构之后不再增长；进程 RSS 在 2.9–3.0 GB 之间趋于平稳（和 FA 关时比多 87 MB）。

## 8. 已知限制

- **首个 batch 的 context 创建开销**：每次出现新的 key 长度时要创建 context，547 / 1023 / 1975 token 分别约 36 / 52 / 141 ms。177 token 这种单次短 batch 开 FA 反而稍慢。
- **CPU softmax 不是流水线瓶颈，别再从这里挤速度**：profiler 细分显示 softmax 计算占该段 89%，但逐行范围裁剪的优化（值逐位一致）实测 e2e 持平或更慢——每核 2 个提交线程时，一个线程的 CPU softmax 正好覆盖另一个线程的 NPU matmul，流水线是 NPU 吞吐瓶颈。想再快只能把 softmax（或整个 FA）搬到 NPU 上融合。
- **需要 3 行 `llama-context` 补丁**：没有它时，`flash_attn = auto` 遇到 RKNPU 上的 FA 节点会把 flash attention 整体关掉（CPU FA 也一起关，变得更慢）。
- **发热降频**：满载时板子到 75–80 °C，warm 吞吐会有明显波动（例如 1975 token、FA 关，从 346 掉到 278–301 tok/s）。建议主动散热。
- 随机截取的文本片段（从句子中间开始）和 CPU 参考的 cosine 更低：40 条里有 19 条低于 0.996，FA 关的时候也一样（最低 0.988）。开启 NPU attention 后平均值基本不变（0.9948 vs 0.9950）。
- 只测了 Qwen3-0.6B（jina-v5-small）这种形状：head_dim 128、GQA 16/8、causal。其他 head_dim（32 的倍数且 ≤ 256）会走 NPU 路径，但没有验证过；
  ALiBi、softcap、sinks、非 F16 KV cache 会自动回退到 CPU。
- `--no-mmap` 需要同时设 `RKNPU_HOST_COMPUTE=0`（见上文）。
- 开启 `RKNPU_FA` 时，第一个（cold）batch 的结果偶尔有很小的 run-to-run 差异（同一输入 cos 0.99697 vs 0.99709，都 ≥ 0.996）；warm batch 和多 batch 的长时间运行在重复测试中逐字节一致。磁盘 I/O 压力大的时间窗里这个差异更容易出现（曾实测同一天安静时段 6 连跑逐字节一致，重 I/O 时段同二进制两轮之间 min cos 0.9975）；逐层 `RKNPU_FA_CHECK` 始终是 1.000000。

## 9. 工具

- `tools/opi5/run-embedding.sh`、`tools/opi5/run-server.sh`：上面的启动命令（模型路径通过环境变量 `MODEL` 传入）。
- `tools/opi5/ctx_mem.cpp`：测量 NPU matmul context 内存占用的小程序（用法见文件头）。
