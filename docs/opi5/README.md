> Current status (2026-10-10, `9ac4a089e`): NPU FA 默认开启；native-B 在 mx4 修复基线上恢复为默认路径；S 为 fp16（type-4 QK matmul）且 softmax 单缓冲原地写回；FA_POOL 与 fast-exp 均为 opt-in。早期可靠性修复（CPU PV fp32 累加、RKNN 错误传播、matmul ctx LRU）见 [2026-10-05 reliability notes](reliability-20261005.md)；量化全链损伤的 fp16 真值实测见第 6 节末尾。

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
| `ggml-rknpu2` NPU attention | `FLASH_ATTN_EXT` 在 NPU 上跑：Q·Kᵀ 和 P·V 用 NPU fp16 matmul（native A/C layout），mask + softmax 在 CPU（NEON），256 行 query tile、GQA 两个 Q head 叠在 M 上、按真实 mask 跳过全遮蔽块，6 个驱动线程（每个 NPU core 2 个），context 缓存以 256 key 为粒度（任何长度的 context 集合都是最大长度集合的子集），有上限。**2026-10-10 起默认开启** | `RKNPU_FA`（默认 1；0 = CPU FA） |
| `ggml-rknpu2` tile 级 key 起点跳过 | 多序列打包的 batch 里，每个 query tile 从它第一条未被遮蔽的 key 开始算（对齐到 tile 粒度），不再从 key 0 开始。k0 > 0 的任务把 K/V 窗口拷进各线程自己的暂存缓冲（fd 偏移视图在两个驱动线程并发时结果会错，不能直接绑偏移）。输出与不跳过时逐字节一致 | `RKNPU_FA_KSKIP`（默认 1；0 = 关） |
| `ggml-rknpu2` B 原生布局绑定 | QK/PV matmul 的 B 直接用 NPU 原生布局（RK3588 fp16：`(N/16, K/32, 16, 32)`），`set_io_mem` 不再在 CPU 上转换 B（实测 4.9 µs vs 186 µs/次）。K 侧原生布局对 Nq 前缀兼容（每个 KV head 一个共享缓冲）；V 侧不兼容（步长依赖 Nq），每个 PV 任务自己交织暂存。原生 B 在 run 时直读内存（板上探针验证），绑定只认缓冲指针。旧路径完整保留 | `RKNPU_FA_NATIVE_B`（默认 1；0 = 旧路径） |
| `ggml-rknpu2` softmax profiler 细分 + 省内存 | `softmax+syncs` 一列拆成 S sync / softmax max / softmax exp+P / generic / P sync 五列（mixed-40：7% / 36% / 53% / 0.1% / 4%）。曾实现逐行范围裁剪的 softmax（值逐位一致、causal tile 少算约一半配对），实测 mixed-40 持平、1975 token 慢约 3%、`RKNPU_FA_THREADS=3` 慢约 12%——每核 2 个提交线程时 CPU softmax 与另一线程的 NPU matmul 重叠，流水线是 NPU 吞吐瓶颈，CPU 侧节省都变成同步等待，故回退（证据留在代码注释）。native 模式下不再分配 legacy vbuf | `RKNPU_PROFILE=1` |
| `llama-context` | 3 行补丁：`flash_attn = auto` 时，如果 FA 节点被分配到 ACCEL 设备（RKNPU），而该层在 CPU 上，不再把 FA 整体关掉（RKNPU 直接在 host 内存上计算） | – |
| `ggml-rknpu2` W8A8_HADAMARD 定型（生产默认） | 激活侧 Hadamard 旋转的三个修复：NEON 融合 FWHT（bit-exact，quantize A 644→224 ms/forward）；符号向量从堆地址种子改为按 K_op 的确定性种子（修复嵌入跨重启漂移）；符号向量按 K_op 共享（q/k/gate/up 旋转后激活一致，恢复 A-reuse）。精度 ≥0.99（fp16 真值仲裁）+ 1089 tok/s，NPU buffer 仅 W16A16 一半 | `RKNPU_HYBRID=W8A8_HADAMARD` |
| 可靠性（2026-09-28 / 10-05） | matmul ctx 缓存加 LRU 驱逐 + A/C 缓冲 shared_ptr 生命周期（根治 fd/dma-buf 无界泄漏的"跑几天必挂"）；NPU FA 每 job 重置 softmax max（`b.mx4`，长度桶切换静默腐蚀的真根因，也是 native-B 的平反证据）；RKNN 失败不再被吞（`RKNN_CHECK_RETURN` + 节点级 CPU fp32 兜底）；`rknn_poll_deadline.so` 30 s 轮询上限防 submit 永不返回 | – |
| 自适应 tile + 投影融合（10-05） | FA tile 按 (序列数, key 长度) 自适应分桶，短请求不再付 256 行大 tile；QKV / gate-up 投影融合实装（正确性通过，但中长请求变慢，默认不启用） | `RKNPU_FA_ADAPTIVE`、`RKNPU_FUSE_QKV`、`RKNPU_FUSE_GATE_UP` |
| S fp16 + 单缓冲 softmax（10-10，`b2dccd0a7`+`9c782a1f9`） | QK matmul 换 type-4（fp16×fp16→fp16，板上探针 1.72×），S 的 native C 布局 `(Nq/8,M,8)` 与 PV 的 A 布局逐元素相同 → softmax 两遍式原地覆写 S，删除独立 P 缓冲和 P 同步。长文本吞吐 +11~22% | – |
| FA worker pool（10-05 加入，10-10 回退） | 常驻线程池实测对 525-token 短请求抽 42% 墙钟税（2.05→3.52 s），长请求无益——per-node 临时线程对短请求更优 | `RKNPU_FA_POOL`（默认 0，opt-in） |
| softmax fp16 fast-exp（10-10，`9ac4a089e`） | 5 阶 Horner `v_exp8_f16`（f32 归约 + f16 多项式 + 2^n 位域构造），内核对标量 expf 11.4×、精度 2.5× 于参考标量版；但同负载 A/B 端到端持平（循环瓶颈在访存/逐行逻辑，非 exp 算术），故 opt-in。同时 CMake 加 `-march=armv8.2-a+fp16`（FMLA.8H 原生生成） | `RKNPU_FA_FAST_EXP`（默认 0，opt-in） |

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
RKNPU_HYBRID=W8A8_HADAMARD RKNPU_FA=1 taskset -c 4-7 ./build/bin/llama-server \
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
- **生产部署（systemd 用户服务，开机自启）**：板上 `~/.config/systemd/user/jina-embed.service`
  （`loginctl enable-linger liyifan` 已开），参数 `-c 8192 -b 4096 -ub 4096 -np 1 --kv-unified -t 4`，
  `CPUAffinity=4-7`（板测必须复刻，否则自旋/CPU 数据全盘失真），管线经 `~/.config/jina-embed.env`
  （当前 `RKNPU_HYBRID=W8A8_HADAMARD RKNPU_FA=1 RKNPU_FA_MAX_CTX=512`）。
  `systemctl --user restart jina-embed` 重启；日志在 `~/services/logs/jina-embed.log`。`LimitNOFILE=65536` 由 systemd 下发
  （手动 nohup 起测试实例**必须先 `ulimit -n 65536`**，否则 FA ctx 预热时 fd 耗尽退化到 CPU FA）。
  手动测试实例另需 `taskset -c 4-7` 复刻绑核；验收对比直连 8310（8311 网关会加输入前缀，与裸文本基线不可直接比）。
- 多 NPU 进程 / 省电：参考上游的 `RKNPU_CORES`、`RKNPU_DOMAINS`（见 `ggml/src/ggml-rknpu2/README.md`）。

## 5. 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `RKNPU_FA` | 1 | 1 = `FLASH_ATTN_EXT` 在 NPU 上跑（**默认开启**）；0 = 原来的 CPU FA。key 长度硬上限 8192（RKNN 物理边界，2026-10-10 起不再有降档旋钮） |
| `RKNPU_FA_KSKIP` | 1 | 1 = 多序列 batch 里每个 tile 从自己的第一条 key 开始算（对齐 tile 粒度）；0 = 从 key 0 开始。两种模式输出逐字节一致 |
| `RKNPU_FA_NATIVE_B` | 1 | 1 = K/V 以 NPU 原生布局绑定（set_io_mem 不做 CPU 转换）；0 = 旧路径（TP_NORM/普通布局，每次绑定转换）。启动时自动探测，失败也回旧路径。曾在 2026-10-03 误删（长度切换腐蚀被错误归因），mx4 重置修复后于 10-10 恢复默认开 |
| `RKNPU_FA_POOL` | 0 | 1 = FA 常驻 worker 线程池。实测短请求（~525 tok）墙钟 +42%、长请求无益，故默认关 |
| `RKNPU_FA_FAST_EXP` | 0 | 1 = softmax exp 用 fp16 5 阶 Horner 内核。A/B 端到端持平（循环非 exp 受限），opt-in 保留 |
| `RKNPU_FA_MAX_CTX` | 192 | NPU attention context 数量上限（LRU 淘汰）。**生产设 512**：形状笛卡尔积超 192 时每请求换血（created 458/evicted 278，白烧 ~1.2 s 创建），512 后 warm 请求创建趋零 |
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

### tile key 跳过 + B 原生布局合入后（2026-09-26 晚复测，同轮交错）

温度窗口对绝对值影响很大：下表两轮是同一二进制。**热窗口**（77–81 °C，含 hermes 磁盘压力）绝对值整体下移约 15–20%（CPU FA 关侧同样降），比率才有可比性；**凉窗口**（起始 54 °C，测试中升到 70–76 °C）则回到甚至超过上一轮正式测试的水平。

| tokens | 窗口 | 关，warm | 开，warm | 变化 |
|---|---|---|---|---|
| 1023 | 热 77–81 °C | 392 | 449 | +15% |
| 1023 | 凉 70–76 °C | 486 | **612** | **+26%** |
| 1975 | 热 77–81 °C | 293 | 366 | +25% |
| 1975 | 凉 70–76 °C | 318 | **425** | **+34%** |

凉窗口里 1023 的 612 比上一轮正式测试的 584 还高 5%；1975 的 425 接近上一轮的 471（那轮板温更低）。同窗口对照 `RKNPU_FA_NATIVE_B=0`（旧绑定路径）：1975 token 时 393–463（中位 449，轮间波动大）vs 全开 423–433（中位 425），差异在噪声内；bind 的收益看线程时间（−41%）更可靠。peak RSS：1975 关→开 2809 → 2859 MB（vbuf 不再分配，比上一轮少 +26 MB）。

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

### 量化管线精度仲裁：W8A8 短文本退化、W8A16 不存在，与 W8A8_HADAMARD 生产配置（2026-09-26 深夜）

以官方 fp16 MLX 模型为真值（Mac 本地直跑）的最终仲裁（cos，7 用例：中/英/俄/日 + 短/长/混合）：

| 管线 | min cos（vs fp16 真值） | 2046 tok prompt eval* | NPU buffer |
|---|---|---|---|
| W8A8_STANDARD（上游默认） | 0.44–0.92（短文本/多语种明显退化） | 1267 tok/s | ~420 MB |
| W16A16_STANDARD | ≥ 0.990 | 940 tok/s | ~840 MB |
| **W8A8_HADAMARD（生产默认）** | **≥ 0.9897** | **1089 tok/s** | ~420 MB |

\* 这是 `llama-embedding` CLI 的批处理记账数字（两条 1023-token 文本打包处理的总 token ÷ 总时长，冷+温混合），
**不是单条长请求的稳态吞吐，偏乐观约 2×**。端到端参考（llama-server HTTP，单并发、~1235 token/条、warm、54–57 °C，
同夜 IO 压力窗口内）：单并发 **~430–480 tok/s**；CLI 单条同长度纯前向（FA 墙钟 + matmul）
约 440–580 tok/s（context 冷热决定）——**server 与 CLI 在同等负载形状下吞吐相同**，
server 固定开销（HTTP+JSON+分词+排队）实测仅 ~38 ms/条（3-token 请求全程），JSON 序列化 ~1–2 ms。
绝对值受温度窗口（±15–20%）和板上后台 IO（hermes 等，IO 压力可再压 20%+）影响，只有同窗口相对值可比。
并发：长文本双 slot 聚合吞吐反而 −25%（单请求已把 CPU+NPU 流水线吃满）；~250 token 短文本双 slot +0–10%。

- **W8A8 的精度问题在激活不在权重**：per-row 单 scale 的 int8 激活量化 × 28 层 × last-token pooling，对短输入/多语种敏感。
- **字面意义的 W8A16 在本平台不存在**：板上探针遍历 `rknn_matmul_type` 1–15，混合精度的 5（FP16×INT8→FP32）、6（FP16×INT8→FP16）、7、8、11、12 全部返回 `-5 unsupported matmul dtype`；支持的只有 1/2/4（纯 fp16）、2/3/9（纯 int8）、10（纯 int4）。fp16 权重的 2× 字节流量也使 W16A16 的 `matmul_run` 固定为 int8 的 ~1.78×（914 vs 514 ms/forward，NPU 带宽决定，CPU 侧不可回收）。
- **W8A8_HADAMARD 是第三条路**：QuaRot 式旋转（加载时权重按随机符号 + Hadamard 旋转打包，运行时激活同样旋转后 int8 量化，epilogue 除以 K_op），int8 的权重流量 + 接近 fp16 的精度。定型时修了三件事：
  1. **NEON 融合 FWHT**：原实现是标量蝶形 + 每行 3 次堆分配 + 3 次 memcpy（quantize A 644 ms/forward）。融合成单 pass NEON（按元素 vadd/vsub/vmul 与标量舍入逐位一致；板上单元测试 12 种尺寸 × 20 组随机输入全部 bit-identical）+ thread_local scratch 指针直通 → 224 ms。
  2. **符号向量确定性**：原来 `mt19937(张量堆地址)` 做种子，ASLR 导致每次启动旋转矩阵都不同，嵌入跨重启漂移（cos ~0.999）。改为按 K_op 的 FNV 哈希种子。
  3. **符号向量按 K_op 共享**：原来 per-tensor 各一个 s，q/k/gate/up 旋转后的激活互不相同，A-reuse 失效（每 forward 784 次量化）。共享后复用恢复（560 次，与 W8A8_STANDARD 一致）。
- 生产切换：`~/services/opi5-switch-accurate.sh [W8A8_HADAMARD|W16A16_STANDARD|W8A8_STANDARD]`（默认第一个）。

### 总量化损伤实测：fp16 真值对照（2026-10-10）

用官方 `jinaai/jina-embeddings-v5-text-small-retrieval-GGUF` 的 **F16.gguf**（与生产 Q8_0 同仓库，唯一差异是量化）
在 Mac（fork llama-server，`--embedding --pooling last` 同参）建立真值基准；15 条文本覆盖英文 14–3952 token
长度梯度、zh/ru/ja/ko/el/ar 短句、查询/代码/混合，**token 数两侧逐条一致**。另拉第三方（Q8_0 也在 Mac 纯 CPU 跑，
llama.cpp 原生反量化、无激活量化、无 NPU）把损伤分解成两半：

| 对照 | mean cos | 最差 |
|---|---|---|
| 权重 Q8_0 单独（fp16 ↔ Mac CPU Q8_0） | 0.99964 | 0.99706 |
| 激活量化 + NPU 平台（Mac CPU Q8_0 ↔ 板生产） | 0.99916 | 0.99691 |
| **全链总量化损伤（fp16 ↔ 板生产）** | **0.99847** | **0.98921** |

- **损伤约 2/3 来自激活侧**，权重 Q8 只占 1/3；对长度几乎不敏感（14 tok 0.99914 vs 3952 tok 0.99927）。
- 唯一离群是俄语 20-token 短句（0.9892），且在两列**同时**劣化（权重列 0.9971 / 激活列 0.9969）——
  这是"短非拉丁 + last-token 池化"对任意扰动的固有放大，非 HADAMARD 缺陷。对比 W8A8_STANDARD 时代同一位置是
  **0.44**，重灾区已被旋转修平。
- 结论：Q8_0 权重 + W8A8 激活 + NPU FA 全链对 fp16 的损伤在 cosine 上是 1e-3 量级，单库使用检索不可感知。

### Hadamard 机制备忘（为什么这套量化"几乎无损"）

- **覆盖清单**：7 种线性层（q/k/v/o/gate/up/down）× 28 层 = 196 个权重矩阵，**A、B 两侧用同一个正交旋转 R**
  （乘积精确不变，这是数学契约，只转一侧结果是垃圾）。Q·Kᵀ / P·V 是激活×激活的动态矩阵，走 NPU fp16，不量化不旋转
  （理论上可用"共享旋转不变性 + KV cache 预旋转 + O 端反旋转"的 QuaRot 方案 int8 化，属高风险性能项，未做）；
  lm_head 被 N 上限留在 CPU 且 pooled embedding 不计算；embedding 表是查表。
- **旋转是精确的，不是近似**：整条链为 `Q8_0 反量化 → fp32 → sign × → FWHT → per-channel int8`。FWHT 只有加减、
  sign 是乘 ±1，全部在 fp32 域，信息损失是浮点舍入级（~1e-7）。有损环节只有三个：GGUF 文件里已付的 Q8_0、
  旋转后的 per-channel 重量化、每层每 token 的激活 int8。
- **per-group → per-channel 重量化为什么无害**：per-group 量化的存在理由是"行内极值绑架 per-channel 尺子"。
  Hadamard 不适应这个问题而是消除它——旋转把尖峰分布洗成近高斯（范数不变，塌的是 max/典型值比值），
  1024 长近高斯行的整行 max ≈ 3.5σ、任意 32 元素子块 max ≈ 2.5–3σ，per-channel 尺子只比 per-group 大 1.2–1.4×，
  重量化退化为同精度格子上的一次独立舍入（误差功率 ×√2）。反向印证：**不旋转的 per-channel 才会痛**（W8A8_STANDARD
  短文本崩 0.44 的机理之一）。
- **主要恩情在激活侧**：激活 int8 是每层每 token 现场量化、误差源远多于权重（8.4e-4 里的大头）；权重侧的 per-channel
  获益是"顺带修好"。per-group 是给病态分布打的石膏，Hadamard 把骨头接正之后石膏自然多余。
- **随机种子无需校准**：随机 ±1 旋转在高维下误差集中，抽到哪个种子统计上几乎等价（校准是在近似常数里找最大值）；
  种子间差异 ≤ 平台噪声（~1e-4），15 条文本根本测不出。种子唯一重要的性质——确定性——已由按 K_op 的 FNV 哈希保证。
  反向账：换种子 = 存量向量库整体偏移 + 需全量重建索引，换一个测不出的收益，纯负交易。

### 当前生产性能快照（2026-10-10，`9ac4a089e`，绑核 4-7 热窗 ~75 °C）

- 稳态单流 **430–480 tok/s**（vs 纯 CPU 155–210，约 2.5×）；525 tok 请求 2.07 s、4355 tok 请求 16.46 s。
- S fp16 + 单缓冲使长文本在 9c782a1f9 基础上再 +11–22%。
- 4355 tok 的时间账：FA O(n²) ≈ 8.4 s（大头是 NPU QK/PV 串行吞吐 + librknnrt 每次 submit/sync 固定开销 × ~3800 job，
  CPU softmax 已被每核双提交线程遮蔽——fast-exp A/B 换 11.4× 内核端到端零变化即为证）+ 权重 matmul ≈ 3 s（贴 int8
  带宽极限）+ CPU 杂项（FWHT 量化/反量化/gather/OMP/分词）≈ 5 s。热降频另收 ~25% 税（凉窗 612 vs 热窗 449 tok/s）。
- 已判死的优化方向（勿重复踩）：CPU softmax 侧一切"算得更快"类改动（范围裁剪/fast-exp/f16 max）、K/V staging 缓存
  （仅占 FA 7%）、mask 跨层缓存（0.3%）、FA_POOL 默认开（短请求 +42% 税）、Q4_K_M（见第 8 节）、W8A16（硬件不支持）。

## 7. 内存

- 每个 NPU attention context 约 0.3–0.5 MB（96 个 context：VmRSS +36.5 MB，其中 RssShmem +28.4 MB；不占 CMA）。
- attention 缓冲区按见过的最长 key 长度分配一次，之后复用：2048 key 时约 47 MB。native-B 模式下 legacy 的 vbuf 暂存不再分配（2048 key 时省 4 MB；下文的 `MAX_KV=4096` 数字是旋钮删除前的历史测量，现在只有 8192 硬上限一档）。
- 1975 token 时 NPU attention 总共多占约 75–85 MB（peak RSS 2812 → 2885–2897 MB）。
- 上限：context 数 ≤ 6 线程 × 2 × (max_kv / 256)，2048 档是 96、4096 档 192、8192 档 384（`RKNPU_FA_MAX_CTX` 可另设；生产用 512 防 LRU 换血）。和输入长度有多少种无关。
- 长时间运行：40 条 100–2000 token 的随机文本（打包成 19 个 batch，每个 batch 1–5 条序列）在同一个进程里跑完。context 数第一个 batch 之后就固定在 96，
  NPU attention 的结构之后不再增长；进程 RSS 在 2.9–3.0 GB 之间趋于平稳（和 FA 关时比多 87 MB）。

## 8. 已知限制

- **Q4_K_M 别用在这条分支上（当前是负优化）**：Q4_K 会被路由到 NPU 的 `W4A4_HADAMARD`（INT4×INT4→INT16）路径，但该路径在 librknnrt 2.3.2 上有两个独立的坑：INT4 matmul context 创建约 31.7 ms/个（fp16 路径的 48 倍，每 forward 588 个，占 matmul 总时间 65%），单次 run 平均 11.3 ms（fp16 路径的 15.6 倍）。实测 jina-v5 Q4_K_M @1023 token 只有 90–160 tok/s，比纯 CPU 跑 Q4（155–210 tok/s）还慢，远低于 Q8+NPU（450–610 tok/s）。本分支请用 Q8_0；想让 Q4 留在 CPU 需要清空 `default_patterns[GGML_TYPE_Q4_K]`（目前没有现成环境变量）。
- **首个 batch 的 context 创建开销**：每次出现新的 key 长度时要创建 context，547 / 1023 / 1975 token 分别约 36 / 52 / 141 ms。177 token 这种单次短 batch 开 FA 反而稍慢。
- **CPU softmax 不是流水线瓶颈，别再从这里挤速度**：profiler 细分显示 softmax 计算占该段 89%，但逐行范围裁剪（值逐位一致）实测 e2e 持平或更慢；2026-10-10 的 fast-exp A/B 再添一证——把 exp 换成 11.4× 的 fp16 内核（同绑核同负载），端到端与 exp+P 线程时间均零变化，说明该循环瓶颈在访存与逐行逻辑而非 exp 算术，且 CPU softmax 本就被每核双提交线程遮在 NPU matmul 之下。想再快只能减少 job 数 / 把 softmax（或整个 FA）融合上 NPU。
- **需要 3 行 `llama-context` 补丁**：没有它时，`flash_attn = auto` 遇到 RKNPU 上的 FA 节点会把 flash attention 整体关掉（CPU FA 也一起关，变得更慢）。
- **发热降频**：满载时板子到 75–80 °C，warm 吞吐会有明显波动（例如 1975 token、FA 关，从 346 掉到 278–301 tok/s）。建议主动散热。
- 随机截取的文本片段（从句子中间开始）和 CPU 参考的 cosine 更低：40 条里有 19 条低于 0.996，FA 关的时候也一样（最低 0.988）。开启 NPU attention 后平均值基本不变（0.9948 vs 0.9950）。
- 只测了 Qwen3-0.6B（jina-v5-small）这种形状：head_dim 128、GQA 16/8、causal。其他 head_dim（32 的倍数且 ≤ 256）会走 NPU 路径，但没有验证过；
  ALiBi、softcap、sinks、非 F16 KV cache 会自动回退到 CPU。
- `--no-mmap` 需要同时设 `RKNPU_HOST_COMPUTE=0`（见上文）。
- 开启 `RKNPU_FA` 时，第一个（cold）batch 的结果偶尔有很小的 run-to-run 差异（同一输入 cos 0.99697 vs 0.99709，都 ≥ 0.996）；warm batch 和多 batch 的长时间运行在重复测试中逐字节一致。磁盘 I/O 压力大的时间窗里这个差异更容易出现（曾实测同一天安静时段 6 连跑逐字节一致，重 I/O 时段同二进制两轮之间 min cos 0.9975）；逐层 `RKNPU_FA_CHECK` 始终是 1.000000。
- **int8 类管线（W8A8 / W8A8_HADAMARD）的跨启动噪声 ~1e-4（cos）**：平台级浮点累加顺序差异（W16A16 也有，~3e-7）被 int8 舍入边界放大。同输入跨进程启动 cos ~0.9999，与 Hadamard 无关。W8A8_HADAMARD 的旋转矩阵本身已确定性（按 K_op 种子），修复前用堆地址做种子时是 ~0.999。

## 9. 工具

- `tools/opi5/run-embedding.sh`、`tools/opi5/run-server.sh`：上面的启动命令（模型路径通过环境变量 `MODEL` 传入）。
- `tools/opi5/ctx_mem.cpp`：测量 NPU matmul context 内存占用的小程序（用法见文件头）。
