#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"
#include "rknpu2-configuration.h"

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <omp.h>
#include <arm_neon.h>
#include <cmath>

#include <cassert>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <random>
#include <limits>
#include <sys/mman.h>
#include <sstream>

#define UNUSED(x) (void)(x)

// --- Env-gated profiling (RKNPU_PROFILE=1) ---
#include <chrono>
#include <atomic>
#include <map>
namespace rknpu_prof {
enum Stage { ST_GRAPH, ST_MATMUL, ST_SETUP, ST_MEMSET, ST_CTX, ST_CTX_CREATE, ST_BIND_B, ST_GET_A, ST_QUANT_A, ST_SETIO_A, ST_SYNC_A,
             ST_SETIO_C, ST_RUN, ST_SYNC_C, ST_DEQUANT, ST_COPY_IN, ST_COPY_OUT, ST_GAP, ST_COUNT };
static const char * stage_names[ST_COUNT] = {"graph_compute(total)", "matmul(total)", "setup(segs,scale copy)", "memset dst",
    "ctx lookup(incl create)", "  ctx create", "bind B (create_mem_from_fd+set_io B)", "get A buf", "quantize A (fp32->int8, omp)", "set_io_mem A (x cores)",
    "mem_sync A TO_DEVICE", "get C buf + set_io_mem C", "matmul_run (omp, all cores)", "mem_sync C FROM_DEVICE", "dequant C -> dst (omp)",
    "[sched] copy CPU->RKNPU (set_tensor memcpy)", "[sched] copy RKNPU->CPU (get_tensor memcpy)", "[outside] gap between RKNPU splits (CPU splits+copies)"};
struct Stats {
    double t[ST_COUNT] = {0};
    long   n[ST_COUNT] = {0};
    long n_graph = 0, n_matmul = 0, n_ctx_create = 0, n_b_bind = 0, n_a_alloc = 0, n_c_alloc = 0, n_run = 0, n_a_reuse = 0;
    double run_core_sum = 0, run_core_max_sum = 0;   // per-core run times: sum over cores vs max over cores
    double macs_real = 0, macs_op = 0;
    double bytes_in = 0, bytes_out = 0;                // M*K*N vs M_op*K*N
    std::map<std::tuple<int,int,int>, std::pair<long,double>> run_by_shape; // (M_op,K,N) -> count, run ms
};
static bool enabled() { static int e = [](){ const char* v = getenv("RKNPU_PROFILE"); return v && atoi(v) > 0; }(); return e; }
static Stats g;
static int forward_idx = 0;
static double last_graph_end = 0;
static inline double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void dump(const char * why) {
    if (!enabled() || g.n_matmul == 0) { g = Stats(); return; }
    fprintf(stderr, "\n[RKNPU_PROFILE] ===== %s #%d: graph_compute calls=%ld matmuls=%ld run calls=%ld ctx_create=%ld B_bind=%ld A_alloc=%ld C_alloc=%ld A_reuse=%ld m_tile=%d\n",
        why, forward_idx, g.n_graph, g.n_matmul, g.n_run, g.n_ctx_create, g.n_b_bind, g.n_a_alloc, g.n_c_alloc, g.n_a_reuse, (int)(getenv("RKNPU_M_TILE") ? atoi(getenv("RKNPU_M_TILE")) : 256));
    fprintf(stderr, "[RKNPU_PROFILE] sched copies: in %.1f MB, out %.1f MB\n", g.bytes_in/1e6, g.bytes_out/1e6);
    fprintf(stderr, "[RKNPU_PROFILE] MACs real=%.3fG padded(M_op)=%.3fG (pad x%.2f)\n", g.macs_real/1e9, g.macs_op/1e9, g.macs_real>0? g.macs_op/g.macs_real : 0);
    for (int i = 0; i < ST_COUNT; ++i)
        fprintf(stderr, "[RKNPU_PROFILE] %-40s %10.2f ms  n=%-6ld avg=%.3f ms  (%.1f%% of matmul)\n", stage_names[i], g.t[i], g.n[i], g.n[i]? g.t[i]/g.n[i] : 0.0,
            g.t[ST_MATMUL] > 0 ? 100.0*g.t[i]/g.t[ST_MATMUL] : 0.0);
    fprintf(stderr, "[RKNPU_PROFILE] per-core run: sum over cores=%.2f ms, sum of per-call max core=%.2f ms, wall run=%.2f ms (concurrency=%.2f)\n",
        g.run_core_sum, g.run_core_max_sum, g.t[ST_RUN], g.t[ST_RUN] > 0 ? g.run_core_sum / g.t[ST_RUN] : 0.0);
    for (auto & kv : g.run_by_shape) {
        double gmac = (double)std::get<0>(kv.first) * std::get<1>(kv.first) * std::get<2>(kv.first) / 1e9;
        double avg = kv.second.second / kv.second.first;
        fprintf(stderr, "[RKNPU_PROFILE]   run shape M_op=%d K=%d N=%d: n=%ld total=%.2f ms avg=%.3f ms  => %.1f GMAC/s\n",
            std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first), kv.second.first, kv.second.second, avg, gmac / (avg/1000.0));
    }
    g = Stats();
}
struct AtExit { ~AtExit() { dump("final"); } };
static AtExit at_exit_dumper;
struct Scope {
    Stage st; double t0; bool on;
    Scope(Stage s) : st(s), t0(0), on(enabled()) { if (on) t0 = now_ms(); }
    ~Scope() { if (on) { g.t[st] += now_ms() - t0; g.n[st]++; } }
};
}
#define RKPROF(stage) rknpu_prof::Scope _rkprof_##stage(rknpu_prof::stage)
#define RKPROF_BEGIN(var) double var = rknpu_prof::enabled() ? rknpu_prof::now_ms() : 0.0
#define RKPROF_END(var, stage) do { if (rknpu_prof::enabled()) { rknpu_prof::g.t[rknpu_prof::stage] += rknpu_prof::now_ms() - var; rknpu_prof::g.n[rknpu_prof::stage]++; } } while (0)

// --- IOMMU Domain Manager ---

// Helper function for parsing complex integer lists
static std::vector<int32_t> parse_domain_list(const std::string& str) {
    std::vector<int32_t> result;
    if (str.empty()) return result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        auto dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            int start = std::strtol(token.substr(0, dash_pos).c_str(), nullptr, 10);
            int end = std::strtol(token.substr(dash_pos + 1).c_str(), nullptr, 10);
            for (int i = start; i <= end; ++i) result.push_back(i);
        } else {
            result.push_back(std::strtol(token.c_str(), nullptr, 10));
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

struct IOMMUDomainManager {
    std::mutex mutex;

    // Max domain size for assigning
    const size_t max_domain_size = ((size_t) std::numeric_limits<int32_t>::max() - 65536);

    // Storage for domains and their sizes
    std::unordered_map<int32_t, size_t> domain_sizes;
    std::unordered_map<int32_t, rknn_matmul_ctx> allocator_contexts;

    // Allowed domain IDs defined by the user
    std::vector<int32_t> allowed_domains;

    IOMMUDomainManager() {
        // Read restricted domains from ENV variable
        const char* env_domains = std::getenv("RKNPU_DOMAINS");
        if (env_domains != nullptr) {
            allowed_domains = parse_domain_list(env_domains);

            if (!allowed_domains.empty()) {
                fprintf(stderr, "\n"
                    "RKNPU WARNING: Custom IOMMU domains detected via RKNPU_DOMAINS.\n"
                    "Due to Rockchip library limitations, concurrent execution of\n"
                    "multiple processes accessing the NPU simultaneously WILL LEAD\n"
                    "to a SYSTEM KERNEL PANIC and WILL FREEZE YOUR OPERATING SYSTEM.\n"
                    "Execute models SEQUENTIALLY if using multiple independent processes.\n");
            }
        }
    }

    // Function for assigning the domain for the tensor of given size
    int32_t assign_domain_memory(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Allocate strictly within the allowed domains
        if (!allowed_domains.empty()) {
            for (int32_t d : allowed_domains) {
                if (domain_sizes[d] + size <= max_domain_size) {
                    domain_sizes[d] += size;
                    ensure_allocator_context(d);
                    return d;
                }
            }

            fprintf(stderr, "RKNPU ERROR: Out of memory in allowed IOMMU domains!\n");
            assert(false);
            return -1;
        // Allocate dynamically
        } else {
            for (int32_t i = 0; i <= 15; ++i) {
                if (domain_sizes[i] + size <= max_domain_size) {
                    domain_sizes[i] += size;
                    ensure_allocator_context(i);
                    return i;
                }
            }
            fprintf(stderr, "RKNPU ERROR: Out of memory in all IOMMU domains!\n");
            assert(false);
            return -1;
        }
    }

    // Function for releasing the given size of the domain memory
    void release_domain_memory(int32_t domain_id, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = domain_sizes.find(domain_id);
        if (it != domain_sizes.end()) {
            if (it->second >= size) {
                it->second -= size;
            } else {
                it->second = 0;
            }
        }
    }

    // Function for getting a new dummy context in the required domain
    rknn_matmul_ctx get_allocator_context(int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);
        ensure_allocator_context(domain_id);
        return allocator_contexts[domain_id];
    }

private:
    // Function for ensuring a dummy context existence in the required domain
    void ensure_allocator_context(int32_t domain_id) {
        if (allocator_contexts.find(domain_id) == allocator_contexts.end()) {
            rknn_matmul_info info;
            memset(&info, 0, sizeof(info));
            info.M = 32; info.K = 32; info.N = 32;
            info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
            info.iommu_domain_id = domain_id;

            rknn_matmul_io_attr io_attr;
            rknn_matmul_ctx ctx = 0;
            rknn_matmul_create(&ctx, &info, &io_attr);
            allocator_contexts[domain_id] = ctx;
        }
    }
};
static IOMMUDomainManager g_domain_manager;

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            fprintf(stderr,"RKNN error %d at %s:%d: %s\n", ret,         \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)

// --- Hashers ---

// Function for hash combinations
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hasher for std::pair
struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

// Hasher for std::tuple
struct TupleHasher {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
        std::size_t seed = 0;
        std::apply([&](const auto&... args) {
            (hash_combine(seed, args), ...);
        }, t);
        return seed;
    }
};

// --- Segmenters ---

// Matrix segment information for N dimension
struct MatrixSegmentN {
    int offset_n;
    int size_n;
    int core_id;
};

// Matrix segment information for K dimension
struct MatrixSegmentK {
    int offset_k;
    int size_k;
};

// Split B-matrix into N-segments for cores
static std::vector<MatrixSegmentN> compute_n_segments(int N, const std::vector<int>& active_cores, int alignment) {
    std::vector<MatrixSegmentN> segments;
    int num_cores = active_cores.size();

    if (num_cores == 0) return segments;

    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);

    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegmentN seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = active_cores[i];

        if (i < remaining / alignment) {
            seg.size_n += alignment;
        }

        offset += seg.size_n;
        segments.push_back(seg);
    }
    return segments;
}

// Split B-matrix into K-segments for hardware limit
static std::vector<MatrixSegmentK> compute_k_segments(int K_op, int k_limit, int alignment) {
    std::vector<MatrixSegmentK> segments;

    if (k_limit <= 0 || K_op <= k_limit) {
        segments.push_back({0, K_op});
        return segments;
    }

    int k_limit_aligned = (k_limit / alignment) * alignment;
    int offset = 0;
    while (offset < K_op) {
        int size = std::min(k_limit_aligned, K_op - offset);
        segments.push_back({offset, size});
        offset += size;
    }
    return segments;
}

// --- Structs ---

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    void* virtual_base;
    size_t total_size;
    std::string name;

    // RKNN buffers allocations for each tensor
    struct TensorAllocation {
        rknn_tensor_mem* mem = nullptr;
        size_t size = 0;
        int32_t iommu_domain_id = 0;
    };
    std::unordered_map<size_t, TensorAllocation> tensor_allocs;

    // Per-block scaling factors for quantized weights
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> quantized_tensor_scales;

    // Per-tensor random sign vector for Hadamard Transform
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> hadamard_s_vectors;

    std::mutex mutex;

    // Function for the allocation of a RKNN buffer for the individual tensor
    TensorAllocation get_tensor_allocation(size_t tensor_offset, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Trying to find an existing buffer
        auto it = tensor_allocs.find(tensor_offset);
        if (it != tensor_allocs.end()) {
            if (it->second.size < size) {
                rknn_matmul_ctx old_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                rknn_destroy_mem(old_ctx, it->second.mem);
                g_domain_manager.release_domain_memory(it->second.iommu_domain_id, it->second.size);

                it->second.iommu_domain_id = g_domain_manager.assign_domain_memory(size);
                rknn_matmul_ctx new_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                it->second.mem = rknn_create_mem(new_ctx, size);
                it->second.size = size;
            }
            return it->second;
        }

        // Acquiring a domain for allocation
        int32_t domain_id = g_domain_manager.assign_domain_memory(size);
        rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(domain_id);

        // Allocating a new buffer for the tensor
        TensorAllocation alloc;
        alloc.mem = rknn_create_mem(alloc_ctx, size);
        alloc.size = size;
        alloc.iommu_domain_id = domain_id;

        GGML_ASSERT(alloc.mem != nullptr && "Failed to allocate tensor memory via RKNN API");
        tensor_allocs[tensor_offset] = alloc;

        return alloc;
    }
};


// RKNN matmul operation context
struct rknpu_matmul_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;

    bool b_bound = false;
    std::shared_ptr<rknn_tensor_mem> mem_B;

    rknpu_matmul_context(int M, int K, int N, rknn_matmul_type type, int32_t domain_id) {
        memset(&info, 0, sizeof(info));
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = type;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        info.iommu_domain_id = domain_id;

        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) ctx = 0;
    }

    ~rknpu_matmul_context() {
        mem_B.reset();

        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};

// Backend main context
struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;

    // RKNN matmul contexts cache (tensor_fd, offset, M, K, N, core_id, type, domain_id)
    std::unordered_map<std::tuple<uintptr_t, size_t, int, int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // A-matrices cache (M_op, K_seg, npu_type_a, domain_id, tile m0, k offset)
    std::unordered_map<std::tuple<int, int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;

    // What each A buffer currently holds (for re-use across matmuls sharing src1 within one split)
    struct AState {
        uint64_t epoch = 0;
        const ggml_tensor * src1 = nullptr;
        const void * data = nullptr;
        int Mt = 0;
        int pipeline_a = -1;
        bool hadamard = false;
        const float * s_vec = nullptr;
        std::vector<float> scales;
    };
    std::unordered_map<const rknn_tensor_mem*, AState> a_state;
    uint64_t graph_epoch = 0;

    // C-matrices cache (M, N, core_id, npu_type_c, domain_id)
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(uintptr_t tensor_id, size_t offset, int M, int K, int N, int core_id, rknn_matmul_type type, int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto key = std::make_tuple(tensor_id, offset, M, K, N, core_id, (int)type, (int)domain_id);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            return it->second;
        }

        RKPROF(ST_CTX_CREATE);
        if (rknpu_prof::enabled()) rknpu_prof::g.n_ctx_create++;
        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type, domain_id);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        rknn_core_mask core_mask;
        switch(core_id) {
            case 0: core_mask = RKNN_NPU_CORE_0; break;
            case 1: core_mask = RKNN_NPU_CORE_1; break;
            case 2: core_mask = RKNN_NPU_CORE_2; break;
            default: core_mask = RKNN_NPU_CORE_AUTO; break;
        }

        int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret != RKNN_SUCC) {
            // Handle error
        }

        matmul_ctx_cache[key] = ctx;
        return ctx;
    }
};


//
// Backend
//

static const char * ggml_backend_rknpu_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "RKNPU";
}

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    ggml_backend_rknpu_context * ctx = (ggml_backend_rknpu_context *)backend->context;
    delete ctx;
    delete backend;
}

// Function for acquiring a pointer for tensor data
static void* get_tensor_real_ptr(const struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return nullptr;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        auto* ctx = (ggml_backend_rknpu_buffer_context*)tensor->buffer->context;
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

        std::lock_guard<std::mutex> lock(ctx->mutex);
        auto it = ctx->tensor_allocs.find(offset);
        if (it != ctx->tensor_allocs.end()) {
            return it->second.mem->virt_addr;
        }
    }

    return tensor->data;
}

// Function for getting buffer from cache or creating new one
template <typename CacheKeyType>
static std::shared_ptr<rknn_tensor_mem> get_tensor_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    rknn_matmul_ctx matmul_ctx,
    size_t size,
    const CacheKeyType& key,
    std::unordered_map<CacheKeyType, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second->size >= size) {
            return it->second;
        }
    }

    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx, size);
    if (!mem) { return nullptr; }
    if (rknpu_prof::enabled()) { if ((void*)&cache == (void*)&backend_ctx->a_buffer_cache) rknpu_prof::g.n_a_alloc++; else rknpu_prof::g.n_c_alloc++; }

    auto deleter = [matmul_ctx](rknn_tensor_mem* m) {
        if (m != 0) {
            rknn_destroy_mem(matmul_ctx, m);
        }
    };

    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    backend_ctx->a_state.erase(mem); // fresh buffer: holds no quantized activations yet
    return mem_shared;
}

// Env-tunable M tiling: large batches are split into tiles of at most RKNPU_M_TILE rows
// (the tail tile is padded to the next power of two). Bounds matmul-context count/creation
// cost and avoids up to ~2x NPU work from padding M to the next power of two.
static int rknpu_m_tile() {
    static int t = [](){ const char* v = std::getenv("RKNPU_M_TILE"); return v ? std::atoi(v) : 256; }();
    return t;
}

// Fast int8 row quantization (symmetric, per-row scale), NEON with round-to-nearest
static inline float rknpu_row_amax(const float* x, int n) {
    int k = 0;
    float32x4_t vmax = vdupq_n_f32(0.0f);
    for (; k + 4 <= n; k += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + k)));
    float amax = vmaxvq_f32(vmax);
    for (; k < n; ++k) amax = std::max(amax, std::abs(x[k]));
    return amax;
}
static inline void rknpu_quant_row_i8(const float* x, int8_t* y, int n, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    const float32x4_t vs = vdupq_n_f32(iscale);
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        int32x4_t a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k +  0), vs));
        int32x4_t b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k +  4), vs));
        int32x4_t c = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k +  8), vs));
        int32x4_t d = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k + 12), vs));
        int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
        int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
        vst1q_s8(y + k, vcombine_s8(vqmovn_s16(ab), vqmovn_s16(cd)));
    }
    for (; k < n; ++k) {
        int v = (int)lrintf(x[k] * iscale);
        y[k] = (int8_t)std::max(-127, std::min(127, v));
    }
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;

    // Getting the current device configuration once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    // forward boundary detection for profiling: first matmul on blk.0 attn_q weight
    if (rknpu_prof::enabled()) {
        for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
            const ggml_tensor * nd = cgraph->nodes[node_i];
            if (nd->op == GGML_OP_MUL_MAT && nd->src[0] && strcmp(nd->src[0]->name, "blk.0.attn_q.weight") == 0) {
                rknpu_prof::dump("forward");
                rknpu_prof::forward_idx++;
                fprintf(stderr, "[RKNPU_PROFILE] forward #%d starts: M=%d\n", rknpu_prof::forward_idx, (int)nd->src[1]->ne[1]);
                break;
            }
        }
        rknpu_prof::g.n_graph++;
        double tnow = rknpu_prof::now_ms();
        if (rknpu_prof::last_graph_end > 0 && rknpu_prof::g.n_graph > 1) { rknpu_prof::g.t[rknpu_prof::ST_GAP] += tnow - rknpu_prof::last_graph_end; rknpu_prof::g.n[rknpu_prof::ST_GAP]++; }
    }
    struct GapEnd { ~GapEnd() { if (rknpu_prof::enabled()) rknpu_prof::last_graph_end = rknpu_prof::now_ms(); } } _gap_end;
    RKPROF(ST_GRAPH);

    // A-buffer reuse is only valid inside one graph_compute call (one scheduler split):
    // the same src1 tensor (q/k/v, gate/up) is quantized once per (tile, K segment).
    backend_ctx->graph_epoch++;

    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        struct ggml_tensor* node = cgraph->nodes[node_i];
        if (node->op != GGML_OP_MUL_MAT) continue;
        RKPROF(ST_MATMUL);
        RKPROF_BEGIN(t_setup);

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];

        // Skipping zero-dimension matmuls
        if (M == 0 || K == 0 || N == 0) {
            continue;
        }

        const auto* pipeline = config.resolve_op_support(src0);
        if (!pipeline) continue;

        // Initializing Hadamard Transform Logic
        const bool is_hadamard = (pipeline->use_hadamard);
        const int K_op = is_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        const rknn_matmul_type matmul_type = pipeline->mm_type;
        const int alignment = pipeline->n_align;

        // Computing specific hardware segments
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }
        auto all_k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto all_n_segments = compute_n_segments(N, config.active_cores, alignment);

        std::vector<MatrixSegmentN> active_n_segments;
        for (const auto& seg : all_n_segments) {
            if (seg.size_n > 0) active_n_segments.push_back(seg);
        }

        if (active_n_segments.empty()) continue;

        // M tiles
        const int m_tile = rknpu_m_tile() > 0 ? rknpu_m_tile() : M;

        if (rknpu_prof::enabled()) {
            rknpu_prof::g.n_matmul++; rknpu_prof::g.macs_real += (double)M*K*N;
            for (int m0 = 0; m0 < M; m0 += m_tile) { int mt = std::min(m_tile, M - m0); int mo = mt > 1 ? rknpu2_calibration::next_power_of_two(mt) : 1; rknpu_prof::g.macs_op += (double)mo*K_op*N; }
        }

        // Initializing variables
        const size_t num_active_segments = active_n_segments.size();
        std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs(num_active_segments);
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(num_active_segments);

        // Acquiring the B-matrix buffer
        ggml_backend_buffer_t src0_buffer = src0->buffer;
        auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
        size_t tensor_offset_in_virtual = (uintptr_t)src0->data - (uintptr_t)src0_buf_ctx->virtual_base;

        int32_t b_domain_id = 0;
        int tensor_fd = -1;
        void* tensor_virt_addr = nullptr;
        const float* s_vec = nullptr;
        const float* scales_B_grid = nullptr;
        {
            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
            auto it = src0_buf_ctx->tensor_allocs.find(tensor_offset_in_virtual);
            GGML_ASSERT(it != src0_buf_ctx->tensor_allocs.end() && "B-matrix RKNN buffer not found");

            tensor_fd = it->second.mem->fd;
            tensor_virt_addr = it->second.mem->virt_addr;
            b_domain_id = it->second.iommu_domain_id;

            // Acquiring the Hadamard vector (pointer, no copy; map entries are stable after load)
            if (is_hadamard) {
                auto its = src0_buf_ctx->hadamard_s_vectors.find(src0);
                GGML_ASSERT(its != src0_buf_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
                s_vec = its->second.data();
            }
            // B-matrix per-channel scales (pointer, no copy)
            if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 || pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                auto itq = src0_buf_ctx->quantized_tensor_scales.find(src0);
                GGML_ASSERT(itq != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scales grid not found");
                scales_B_grid = itq->second.data();
            }
        }

        float* dst_data = (float*)get_tensor_real_ptr(dst);
        const float* x_all = (const float*)get_tensor_real_ptr(src1);
        const int row_stride = (int)(src1->nb[1] / sizeof(float));

        // Calculating tensor packed size
        size_t type_size_packed = 0;
        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) type_size_packed = 2;
        else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) type_size_packed = 1;
        RKPROF_END(t_setup, ST_SETUP);

        for (int m0 = 0; m0 < M; m0 += m_tile) {
        const int Mt = std::min(m_tile, M - m0);
        const int M_op = Mt > 1 ? rknpu2_calibration::next_power_of_two(Mt) : 1;
        const float* x = x_all + (size_t)m0 * row_stride;
        float* dst_tile = dst_data + (size_t)m0 * N;

        // Computing K dimensions segments
        size_t current_offset_in_tensor = 0;
        for (size_t k_idx = 0; k_idx < all_k_segments.size(); ++k_idx) {
            const auto& k_seg = all_k_segments[k_idx];
            const int K_seg_op = k_seg.size_k;
            const bool first_k = (k_idx == 0);

            // ===========================================
            // ========== 1. Preparing Contexts ==========
            // ===========================================
            for (const auto& n_seg : all_n_segments) {
                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                    if (active_n_segments[idx].offset_n == n_seg.offset_n) {
                        size_t offset_in_dma = current_offset_in_tensor;

                        // Getting matmul context from cache
                        {
                        RKPROF(ST_CTX);
                        matmul_ctxs[idx] = backend_ctx->get_matmul_ctx(
                            (uintptr_t)tensor_virt_addr, offset_in_dma, M_op, K_seg_op, n_seg.size_n,
                            n_seg.core_id, matmul_type, b_domain_id
                        );
                        }
                        if (!matmul_ctxs[idx] || matmul_ctxs[idx]->ctx == 0) return GGML_STATUS_FAILED;

                        auto& matmul_ctx = matmul_ctxs[idx];

                        // Assigning B-matrix only once to reduce computation overhead
                        if (!matmul_ctx->b_bound) {
                            RKPROF(ST_BIND_B);
                            if (rknpu_prof::enabled()) rknpu_prof::g.n_b_bind++;
                            size_t segment_size_bytes = matmul_ctx->io_attr.B.size;

                            rknn_tensor_mem* mem = rknn_create_mem_from_fd(
                                matmul_ctx->ctx,
                                tensor_fd,
                                tensor_virt_addr,
                                segment_size_bytes,
                                offset_in_dma
                            );
                            if (!mem) return GGML_STATUS_FAILED;

                            auto deleter = [ctx = matmul_ctx->ctx](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(ctx, m); };
                            matmul_ctx->mem_B = std::shared_ptr<rknn_tensor_mem>(mem, deleter);

                            RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, matmul_ctx->mem_B.get(), &matmul_ctx->io_attr.B), "set_io_mem B segment");

                            matmul_ctx->b_bound = true;
                        }
                        break;
                    }
                }

                if (n_seg.size_n > 0) {
                    current_offset_in_tensor += type_size_packed > 0 ? (size_t)n_seg.size_n * K_seg_op * type_size_packed : (size_t)n_seg.size_n * K_seg_op / 2;
                }
            }

            // ===========================================
            // ========== 2. Preparing A-matrix ==========
            // ===========================================
            std::shared_ptr<rknn_tensor_mem> mem_A_shared;
            const float* scales_A = nullptr;
            {
                // One A buffer per (M_op, K segment, type, domain, tile start, K offset)
                auto cache_key = std::make_tuple(M_op, K_seg_op, (int)pipeline->npu_type_a, b_domain_id, m0, k_seg.offset_k);
                auto& matmul_ctx_0 = matmul_ctxs[0];

                // Getting A-buffer from cache
                {
                RKPROF(ST_GET_A);
                mem_A_shared = get_tensor_buffer(backend_ctx, matmul_ctx_0->ctx, matmul_ctx_0->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache);
                }
                if (!mem_A_shared) return GGML_STATUS_FAILED;

                auto& astate = backend_ctx->a_state[mem_A_shared.get()];
                const bool reuse = astate.epoch == backend_ctx->graph_epoch && astate.src1 == src1 && astate.data == src1->data &&
                                   astate.Mt == Mt && astate.pipeline_a == (int)pipeline->npu_type_a && astate.hadamard == is_hadamard &&
                                   astate.s_vec == s_vec && std::getenv("RKNPU_NO_A_REUSE") == nullptr;
                if (reuse) {
                    if (rknpu_prof::enabled()) rknpu_prof::g.n_a_reuse++;
                } else {
                RKPROF_BEGIN(t_qa);
                astate.scales.assign(Mt, 1.0f);
                float* sA = astate.scales.data();
                void* dst_base = mem_A_shared->virt_addr;

                #pragma omp parallel for
                for (int m = 0; m < Mt; ++m) {
                    const float* src_row = x + (size_t)m * row_stride;
                    std::vector<float> ready_row_buf;
                    const float* ready_row = src_row + k_seg.offset_k;

                    // Applying Hadamard Transform
                    if (is_hadamard) {
                        ready_row_buf.resize(K_seg_op);
                        std::vector<float> signed_row(K);
                        std::vector<float> full_hadamard_row(K_op);
                        for(int k=0; k<K; ++k) signed_row[k] = src_row[k] * s_vec[k];
                        rknpu2_calibration::hadamard_transform(full_hadamard_row.data(), signed_row.data(), K, K_op);

                        memcpy(ready_row_buf.data(), full_hadamard_row.data() + k_seg.offset_k, K_seg_op * sizeof(float));
                        ready_row = ready_row_buf.data();
                    }

                    // Handling types and quantizations
                    if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
                        uint16_t* dst_ptr = (uint16_t*)dst_base;
                        uint16_t* dst_row = dst_ptr + (size_t)m * K_seg_op;
                        rknpu2_quantization::convert_fp32_to_fp16(ready_row, dst_row, K_seg_op);
                    }
                    else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
                        float amax_m = rknpu_row_amax(ready_row, K_seg_op);
                        sA[m] = amax_m / 127.0f;

                        int8_t* dst_ptr = (int8_t*)dst_base;
                        int8_t* dst_row = dst_ptr + (size_t)m * K_seg_op;
                        rknpu_quant_row_i8(ready_row, dst_row, K_seg_op, sA[m]);
                    }
                    else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                        float amax_m = rknpu_row_amax(ready_row, K_seg_op);
                        sA[m] = amax_m / 7.0f;

                        uint8_t* dst_ptr = (uint8_t*)dst_base;
                        uint8_t* dst_row = dst_ptr + (size_t)m * (K_seg_op / 2);
                        rknpu2_quantization::quantize_fp32_to_int4_packed(ready_row, dst_row, K_seg_op, sA[m]);
                    }
                }
                RKPROF_END(t_qa, ST_QUANT_A);
                {
                RKPROF(ST_SYNC_A);
                RKNN_CHECK(rknn_mem_sync(matmul_ctxs[0]->ctx, mem_A_shared.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");
                }
                astate.epoch = backend_ctx->graph_epoch; astate.src1 = src1; astate.data = src1->data; astate.Mt = Mt;
                astate.pipeline_a = (int)pipeline->npu_type_a; astate.hadamard = is_hadamard; astate.s_vec = s_vec;
                }
                scales_A = astate.scales.data();

                // Assigning A-matrix to all contexts for the parallel execution
                {
                RKPROF(ST_SETIO_A);
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[idx]->ctx, mem_A_shared.get(), &matmul_ctxs[idx]->io_attr.A), "set_io_mem A for core");
                }
                }
            }

            // ===========================================
            // ========== 3. Preparing C-matrix ==========
            // ===========================================
            {
                RKPROF(ST_SETIO_C);
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    auto& matmul_ctx = matmul_ctxs[idx];
                    auto cache_key = std::make_tuple(M_op, active_n_segments[idx].size_n, active_n_segments[idx].core_id, (int)pipeline->npu_type_c, b_domain_id);

                    // Getting C-buffer from cache
                    mem_C_segments[idx] = get_tensor_buffer(backend_ctx, matmul_ctx->ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache);
                    if (!mem_C_segments[idx]) return GGML_STATUS_FAILED;

                    // Assigning C-matrix to current context for the parallel execution
                    RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[idx].get(), &matmul_ctx->io_attr.C), "set_io_mem C");
                }
            }

            // ==========================================
            // ========== 4. Running operation ==========
            // ==========================================
            {
                RKPROF_BEGIN(t_run);
                double core_ms[8] = {0};
                #pragma omp parallel for num_threads(num_active_segments)
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    double t0 = rknpu_prof::enabled() ? rknpu_prof::now_ms() : 0.0;
                    int ret = rknn_matmul_run(matmul_ctxs[idx]->ctx);
                    if (ret != RKNN_SUCC) {
                        fprintf(stderr, "RKNPU: rknn_matmul_run failed ret=%d\n", ret);
                    }
                    if (rknpu_prof::enabled() && idx < 8) core_ms[idx] = rknpu_prof::now_ms() - t0;
                }
                RKPROF_END(t_run, ST_RUN);
                if (rknpu_prof::enabled()) {
                    double mx = 0, sm = 0;
                    for (size_t idx = 0; idx < num_active_segments && idx < 8; idx++) { sm += core_ms[idx]; mx = std::max(mx, core_ms[idx]); }
                    rknpu_prof::g.run_core_sum += sm; rknpu_prof::g.run_core_max_sum += mx;
                    rknpu_prof::g.n_run += num_active_segments;
                    auto & e = rknpu_prof::g.run_by_shape[std::make_tuple(M_op, K_seg_op, N)];
                    e.first++; e.second += rknpu_prof::now_ms() - t_run;
                }
            }

            // ===========================================
            // ========== 5. Collecting results ==========
            // ===========================================
            {
                {
                RKPROF(ST_SYNC_C);
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    RKNN_CHECK(rknn_mem_sync(matmul_ctxs[idx]->ctx, mem_C_segments[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
                }
                }
                RKPROF(ST_DEQUANT);

                const float hadamard_divisor = pipeline->use_hadamard ? (float)K_op : 1.0f;
                const int npu_type_c = pipeline->npu_type_c;

                // dst = C * (scale_A[m] * scale_B[n] / hadamard) ; first K segment assigns, later ones accumulate
                #pragma omp parallel for
                for (int m = 0; m < Mt; m++) {
                    const float sa = scales_A[m] / hadamard_divisor;
                    for (size_t idx = 0; idx < num_active_segments; idx++) {
                        const int N_offset = active_n_segments[idx].offset_n;
                        const int N_segment = active_n_segments[idx].size_n;
                        const float* wscale = scales_B_grid ? (scales_B_grid + k_idx * N + N_offset) : nullptr;
                        float* dst_ptr = dst_tile + (size_t)m * N + N_offset;
                        if (npu_type_c == rknpu2_configuration::NPU_TYPE_FP32) {
                            const float* src_ptr = (const float*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;
                            if (first_k) { for (int n = 0; n < N_segment; ++n) dst_ptr[n]  = src_ptr[n] * (sa * (wscale ? wscale[n] : 1.0f)); }
                            else         { for (int n = 0; n < N_segment; ++n) dst_ptr[n] += src_ptr[n] * (sa * (wscale ? wscale[n] : 1.0f)); }
                        } else if (npu_type_c == rknpu2_configuration::NPU_TYPE_INT32) {
                            const int32_t* src_ptr = (const int32_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;
                            if (wscale) {
                                if (first_k) { for (int n = 0; n < N_segment; ++n) dst_ptr[n]  = (float)src_ptr[n] * (sa * wscale[n]); }
                                else         { for (int n = 0; n < N_segment; ++n) dst_ptr[n] += (float)src_ptr[n] * (sa * wscale[n]); }
                            } else {
                                if (first_k) { for (int n = 0; n < N_segment; ++n) dst_ptr[n]  = (float)src_ptr[n] * sa; }
                                else         { for (int n = 0; n < N_segment; ++n) dst_ptr[n] += (float)src_ptr[n] * sa; }
                            }
                        } else if (npu_type_c == rknpu2_configuration::NPU_TYPE_INT16) {
                            const int16_t* src_ptr = (const int16_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;
                            if (first_k) { for (int n = 0; n < N_segment; ++n) dst_ptr[n]  = (float)src_ptr[n] * (sa * (wscale ? wscale[n] : 1.0f)); }
                            else         { for (int n = 0; n < N_segment; ++n) dst_ptr[n] += (float)src_ptr[n] * (sa * (wscale ? wscale[n] : 1.0f)); }
                        }
                    }
                }
            }
        }
        } // M tiles
    }

    return GGML_STATUS_SUCCESS;
}


//
// Buffer
//

// Function for calculating a real tensor size for the NPU
static size_t get_tensor_packed_size(const struct ggml_tensor * tensor) {
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        size_t total_size = 0;
        for (const auto& k_seg : k_segments) {
            for (const auto& seg : n_segments) {
                if (seg.size_n > 0) {
                    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                        total_size += (size_t)seg.size_n * k_seg.size_k / 2;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
                        total_size += (size_t)seg.size_n * k_seg.size_k;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
                        total_size += (size_t)seg.size_n * k_seg.size_k * 2;
                    }
                }
            }
        }
        return total_size;
    }
    return ggml_nbytes(tensor);
}

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    // Freeing an every individual RKNN buffer using the allocator context
    for (auto& pair : ctx->tensor_allocs) {
        if (pair.second.mem) {
            rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(pair.second.iommu_domain_id);
            rknn_destroy_mem(alloc_ctx, pair.second.mem);
            g_domain_manager.release_domain_memory(pair.second.iommu_domain_id, pair.second.size);
        }
    }

    // Freeing the virtual memory block
    munmap(ctx->virtual_base, ctx->total_size);

    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->virtual_base;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    // Initialize tensor only if it is supported by the pipeline
    if (pipeline) {
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;
        size_t size = get_tensor_packed_size(tensor);
        ctx->get_tensor_allocation(offset, size);
    }

    return GGML_STATUS_SUCCESS;
}

// Function for dequantizing a single row from GGUF format to FP32
static void dequantize_row(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int n, int K,
    float * row_out)
{
    if (tensor->type == GGML_TYPE_F32) {
        const float* src = (const float*)raw_data;
        memcpy(row_out, src + (size_t)n * K, K * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)raw_data;
        const ggml_fp16_t* src_row = src + (size_t)n * K;
        for (int k = 0; k < K; ++k) row_out[k] = ggml_fp16_to_fp32(src_row[k]);
    } else if (tensor->type == GGML_TYPE_Q8_0) {
        const block_q8_0* src = (const block_q8_0*)raw_data;
        dequantize_row_q8_0(src + (size_t)n * (K / QK8_0), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q6_K) {
        const block_q6_K* src = (const block_q6_K*)raw_data;
        dequantize_row_q6_K(src + (size_t)n * (K / QK_K), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q4_0) {
        const block_q4_0* src = (const block_q4_0*)raw_data;
        dequantize_row_q4_0(src + (size_t)n * (K / QK4_0), row_out, K);
    } else {
        // Generic fallback (bench 2026-09-25): any ggml type with a to_float
        // trait (Q4_K, Q5_K, Q5_0, IQ*, ...) is dequantized row by row.
        const struct ggml_type_traits * tt = ggml_get_type_traits(tensor->type);
        GGML_ASSERT(tt && tt->to_float && "Unsupported weight type for NPU pipeline");
        const size_t row_bytes = ggml_row_size(tensor->type, K);
        tt->to_float((const char *)raw_data + (size_t)n * row_bytes, row_out, K);
    }
}

// Function for extracting a specific tensor segment and converting it to FP32
static void dequantize_tensor_segment(
    std::vector<float>& out_segment,
    const struct ggml_tensor * tensor,
    ggml_backend_rknpu_buffer_context * ctx,
    const void * raw_data,
    int K, int N, int K_op,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    bool use_hadamard)
{
    size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;
    out_segment.resize(seg_elements);

    std::vector<float> s_vec;
    if (use_hadamard) {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        s_vec = ctx->hadamard_s_vectors[tensor];
    }

    #pragma omp parallel for
    for (int i = 0; i < n_seg.size_n; ++i) {
        int global_n = n_seg.offset_n + i;

        if (global_n < N) {
            std::vector<float> row_raw(K);
            std::vector<float> row_processed(K_op, 0.0f);

            dequantize_row(tensor, raw_data, global_n, K, row_raw.data());

            if (use_hadamard) {
                std::vector<float> signed_row(K);
                for (int k = 0; k < K; ++k) signed_row[k] = row_raw[k] * s_vec[k];
                rknpu2_calibration::hadamard_transform(row_processed.data(), signed_row.data(), K, K_op);
            } else {
                memcpy(row_processed.data(), row_raw.data(), K * sizeof(float));
            }

            memcpy(&out_segment[i * k_seg.size_k], &row_processed[k_seg.offset_k], k_seg.size_k * sizeof(float));
        } else {
            memset(&out_segment[i * k_seg.size_k], 0, k_seg.size_k * sizeof(float));
        }
    }
}

// Function for quantizing the FP32 segment to the target NPU format
static void quantize_tensor_segment(
    const std::vector<float>& fp32_segment,
    std::vector<uint8_t>& out_quantized,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const std::vector<float>& row_scales,
    rknpu2_configuration::Rknpu2NpuType npu_type)
{
    const int K_seg = k_seg.size_k;
    const int N_seg = n_seg.size_n;
    const size_t seg_elements = (size_t)N_seg * K_seg;

    if (npu_type == rknpu2_configuration::NPU_TYPE_FP16) {
        out_quantized.resize(seg_elements * 2);
        rknpu2_quantization::convert_fp32_to_fp16(
            fp32_segment.data(),
            (uint16_t*)out_quantized.data(),
            seg_elements);
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT8) {
        out_quantized.resize(seg_elements);
        int8_t* dst = (int8_t*)out_quantized.data();
        for (int i = 0; i < N_seg; ++i) {
            const float* src_row = fp32_segment.data() + (size_t)i * K_seg;
            int8_t* dst_row = dst + (size_t)i * K_seg;
            rknpu2_quantization::quantize_fp32_to_int8(src_row, dst_row, K_seg, row_scales[i]);
        }
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT4) {
        out_quantized.resize(seg_elements / 2);
        uint8_t* dst = out_quantized.data();
        for (int i = 0; i < N_seg; ++i) {
            const float* src_row = fp32_segment.data() + (size_t)i * K_seg;
            uint8_t* dst_row = dst + (size_t)i * (K_seg / 2);
            rknpu2_quantization::quantize_fp32_to_int4_packed(src_row, dst_row, K_seg, row_scales[i]);
        }
    }
}

// Function for packing
static void pack_native(
    uint8_t* dst, const uint8_t* src,
    int K_total, int k_offset, int k_segment, int k_align,
    int N_total, int n_offset, int n_segment, int n_align,
    int element_bits)
{
    UNUSED(N_total);

    GGML_ASSERT(k_segment % k_align == 0 && "k_segment must be aligned to k_align");
    GGML_ASSERT(n_segment % n_align == 0 && "n_segment must be aligned to n_align");

    const size_t k_sub_bytes     = (size_t)k_align * element_bits / 8;
    const size_t src_row_bytes  = (size_t)K_total * element_bits / 8;
    const size_t n_blocks       = n_segment / n_align;
    const size_t k_blocks       = k_segment / k_align;
    const size_t kblock_stride  = (size_t)n_align * k_sub_bytes;
    const size_t nblock_stride  = k_blocks * kblock_stride;

    for (size_t ni = 0; ni < n_blocks; ++ni) {
        for (size_t ki = 0; ki < k_blocks; ++ki) {
            uint8_t* dst_tile = dst + ni * nblock_stride + ki * kblock_stride;

            for (int nn = 0; nn < n_align; ++nn) {
                const size_t n_global = (size_t)n_offset + ni * n_align + nn;
                const size_t k_start  = (size_t)k_offset + ki * k_align;

                const uint8_t* src_ptr = src + n_global * src_row_bytes
                                             + k_start * element_bits / 8;
                uint8_t* dst_ptr = dst_tile + nn * k_sub_bytes;

                size_t off = 0;
                for (; off + 16 <= k_sub_bytes; off += 16) {
                    vst1q_u8(dst_ptr + off, vld1q_u8(src_ptr + off));
                }
                for (; off < k_sub_bytes; ++off) {
                    dst_ptr[off] = src_ptr[off];
                }
            }
        }
    }
}

// Function for packing the quantized segment into the native NPU layout and writing to DMA
static size_t pack_tensor_segment(
    const std::vector<uint8_t>& quantized_segment,
    uint8_t * dst_dma_ptr,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline)
{
    int element_bits = 0;
    size_t segment_packed_size = 0;

    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
        element_bits = 16;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k * 2;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
        element_bits = 8;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
        element_bits = 4;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k / 2;
    }

    pack_native(dst_dma_ptr, quantized_segment.data(),
                k_seg.size_k, 0, k_seg.size_k, pipeline->k_align,
                n_seg.size_n, 0, n_seg.size_n, pipeline->n_align,
                element_bits);

    return segment_packed_size;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    size_t tensor_offset_in_virtual = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

    if (pipeline) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];
        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        // Initializing Hadamard Transform Logic
        if (pipeline->use_hadamard) {
            std::vector<float> s_vec(K_op, 1.0f);
            std::mt19937 gen(reinterpret_cast<uintptr_t>(tensor));
            std::uniform_int_distribution<int> distrib(0, 1);

            for(int k = 0; k < K_op; ++k) {
                s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
            }

            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->hadamard_s_vectors[tensor] = s_vec;
        }

        // Computing global scale
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        // Allocating a new buffer for a tensor
        size_t required_size = get_tensor_packed_size(tensor);
        auto alloc = ctx->get_tensor_allocation(tensor_offset_in_virtual, required_size);
        uint8_t* tensor_dma_ptr = (uint8_t*)alloc.mem->virt_addr;

        // Computing specific hardware segments
        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        std::vector<float> seg_fp32;
        std::vector<uint8_t> seg_npu;
        uint8_t* current_write_ptr = tensor_dma_ptr + offset;

        // Per-channel weight scales storage
        std::vector<float> per_channel_scales;
        if (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
            per_channel_scales.resize(k_segments.size() * N, 1.0f);
        }

        std::vector<float> row_scales;

        // Processing individual segments block-by-block
        for (size_t k_idx = 0; k_idx < k_segments.size(); ++k_idx) {
            const auto& k_seg = k_segments[k_idx];
            for (const auto& n_seg : n_segments) {
                if (n_seg.size_n == 0) continue;

                // Dequantizing the block
                dequantize_tensor_segment(seg_fp32, tensor, ctx, data, K, N, K_op, k_seg, n_seg, pipeline->use_hadamard);

                // Calculating per-channel scales of the segment
                if (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
                    const float quant_divisor = (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) ? 7.0f : 127.0f;
                    row_scales.resize(n_seg.size_n);

                    #pragma omp parallel for
                    for (int i = 0; i < n_seg.size_n; ++i) {
                        const float* row_fp32 = seg_fp32.data() + (size_t)i * k_seg.size_k;
                        float amax = 0.0f;
                        for (int j = 0; j < k_seg.size_k; ++j) {
                            amax = std::max(amax, std::abs(row_fp32[j]));
                        }
                        float sw = (amax == 0.0f) ? 1.0f : amax / quant_divisor;
                        row_scales[i] = sw;

                        int global_n = n_seg.offset_n + i;
                        per_channel_scales[k_idx * N + global_n] = sw;
                    }
                } else {
                    row_scales.assign(n_seg.size_n, 1.0f);
                }

                // Quantizing
                quantize_tensor_segment(seg_fp32, seg_npu, k_seg, n_seg, row_scales, pipeline->npu_type_b);

                // Packing into chip native layout
                size_t bytes_written = pack_tensor_segment(seg_npu, current_write_ptr, k_seg, n_seg, pipeline);

                current_write_ptr += bytes_written;
            }
        }

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->quantized_tensor_scales[tensor] = std::move(per_channel_scales);
        }

        rknn_matmul_ctx sync_ctx = g_domain_manager.get_allocator_context(alloc.iommu_domain_id);
        RKNN_CHECK(rknn_mem_sync(sync_ctx, alloc.mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B TO_DEVICE");
    } else {
        RKPROF(ST_COPY_IN);
        if (rknpu_prof::enabled()) rknpu_prof::g.bytes_in += size;
        memcpy((uint8_t*)tensor->data + offset, data, size);
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context*)buffer->context;
    size_t tensor_offset_in_virtual = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

    std::lock_guard<std::mutex> lock(ctx->mutex);
    auto it = ctx->tensor_allocs.find(tensor_offset_in_virtual);
    if (it != ctx->tensor_allocs.end()) {
        memcpy(data, (uint8_t*)it->second.mem->virt_addr + offset, size);
    } else {
        RKPROF(ST_COPY_OUT);
        if (rknpu_prof::enabled()) rknpu_prof::g.bytes_out += size;
        memcpy(data, (uint8_t*)tensor->data + offset, size);
    }
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);

    for (auto& pair : ctx->tensor_allocs) {
        memset((uint8_t*)pair.second.mem->virt_addr, value, pair.second.size);
    }
}


//
// Buffer Type
//

static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "RKNPU";
}

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    UNUSED(buft);

    // Reserving virtual memory block
    void* virtual_base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (virtual_base == MAP_FAILED) {
        return NULL;
    }

    // Initializing buffer context
    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context();
    ctx->virtual_base = virtual_base;
    ctx->total_size = size;
    ctx->name = "rknpu_virtual_buffer";

    static const ggml_backend_buffer_i rknpu_buffer_interface = {
        /* .free_buffer   = */ ggml_backend_rknpu_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rknpu_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rknpu_buffer_init_tensor,
        /* .memset_tensor = */ NULL,
        /* .set_tensor    = */ ggml_backend_rknpu_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rknpu_buffer_get_tensor,
        /* .set_tensor_2d = */ NULL,
        /* .get_tensor_2d = */ NULL,
        /* .cpy_tensor    = */ NULL,
        /* .clear         = */ ggml_backend_rknpu_buffer_clear,
        /* .reset         = */ NULL,
    };

    return ggml_backend_buffer_init(buft, rknpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    return get_tensor_packed_size(tensor);
}


//
// Device
//

static const char * ggml_backend_rknpu_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "RKNPU";
}

static const char * ggml_backend_rknpu_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU";
}

static void ggml_backend_rknpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_rknpu_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rknpu_device_get_name(dev);
    props->description = ggml_backend_rknpu_device_get_description(dev);
    props->type = ggml_backend_rknpu_device_get_type(dev);
    ggml_backend_rknpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    switch (op->op) {
        case GGML_OP_NONE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations

            // src0 must be a weight stored (pre-packed) in an RKNPU buffer. Activations / KV-cache
            // views (e.g. KQV with -fa off) have no packed NPU copy -> computing them here is garbage.
            if (src0->buffer == nullptr || src0->buffer->buft == nullptr ||
                src0->buffer->buft->iface.get_name != ggml_backend_rknpu_buffer_type_get_name) {
                return false;
            }

            // Searching for available hardware pipeline for this tensor
            const auto* pipeline = config.resolve_op_support(src0);
            if (!pipeline) {
                return false;
            }

            // Rejecting zero-dimension ops
            if (src0->ne[0] == 0 || src0->ne[1] == 0 ||
                src1->ne[0] == 0 || src1->ne[1] == 0) {
                return false;
            }

            // Checking if activation type matches the supported operation
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }

            // Checking for K alignment
            if (src0->ne[0] % pipeline->k_align != 0) {
                return false;
            }

            // Checking for N alignment
            if (src0->ne[1] % pipeline->n_align != 0) {
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                 return false;
            }

            // Checking contiguous memory
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }

            return true;
        }
        default:
            return false;
    }
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    // Fetch device from environment variable, default to RK3588 if not set
    const char* env_device = std::getenv("RKNPU_DEVICE");
    std::string target_device = env_device ? env_device : "RK3588";
    if (!rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device(target_device)) return NULL;

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context();

    static const struct ggml_backend_i rknpu_backend_interface = {
        /* .get_name           = */ ggml_backend_rknpu_name,
        /* .free               = */ ggml_backend_rknpu_free,
        /* .set_tensor_async   = */ NULL,
        /* .get_tensor_async   = */ NULL,
        /* .set_tensor_2d_async = */ NULL,
        /* .get_tensor_2d_async = */ NULL,
        /* .cpy_tensor_async   = */ NULL,
        /* .synchronize        = */ NULL,
        /* .graph_plan_create  = */ NULL,
        /* .graph_plan_free    = */ NULL,
        /* .graph_plan_update  = */ NULL,
        /* .graph_plan_compute = */ NULL,
        /* .graph_compute      = */ ggml_backend_rknpu_graph_compute,
        /* .event_record       = */ NULL,
        /* .event_wait         = */ NULL,
        /* .graph_optimize     = */ NULL,
    };

    return new ggml_backend{
        /* .guid    = */ {0},
        /* .iface   = */ rknpu_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}


//
// Registry
//

static const char * ggml_backend_rknpu_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "RKNPU";
}

static size_t ggml_backend_rknpu_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_rknpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return NULL;
    }

    static const struct ggml_backend_buffer_type_i rknpu_buffer_type_interface = {
        /* .get_name       = */ ggml_backend_rknpu_buffer_type_get_name,
        /* .alloc_buffer   = */ ggml_backend_rknpu_buffer_type_alloc_buffer,
        /* .get_alignment  = */ ggml_backend_rknpu_buffer_type_get_alignment,
        /* .get_max_size   = */ NULL,
        /* .get_alloc_size = */ ggml_backend_rknpu_buffer_type_get_alloc_size,
        /* .is_host        = */ NULL,
    };

    static struct ggml_backend_buffer_type rknpu_buffer_type = {
        /* .iface   = */ rknpu_buffer_type_interface,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    static const struct ggml_backend_device_i rknpu_device_interface = {
        /* .get_name             = */ ggml_backend_rknpu_device_get_name,
        /* .get_description      = */ ggml_backend_rknpu_device_get_description,
        /* .get_memory           = */ ggml_backend_rknpu_device_get_memory,
        /* .get_type             = */ ggml_backend_rknpu_device_get_type,
        /* .get_props            = */ ggml_backend_rknpu_device_get_props,
        /* .init_backend         = */ ggml_backend_rknpu_device_init_backend,
        /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return &rknpu_buffer_type; },
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ ggml_backend_rknpu_device_supports_op,
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == &rknpu_buffer_type; },
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
    };

    static struct ggml_backend_device rknpu_device = {
        /* .iface   = */ rknpu_device_interface,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };

    if (rknpu_buffer_type.device == NULL) {
        rknpu_buffer_type.device = &rknpu_device;
    }

    return &rknpu_device;
}


//
// Public API
//

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static const struct ggml_backend_reg_i rknpu_reg_interface = {
        /* .get_name         = */ ggml_backend_rknpu_reg_get_name,
        /* .get_device_count = */ ggml_backend_rknpu_reg_get_device_count,
        /* .get_device       = */ ggml_backend_rknpu_reg_get_device,
        /* .get_proc_address = */ NULL,
    };

    static struct ggml_backend_reg rknpu_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu_reg_interface,
        /* .context     = */ NULL,
    };

    return &rknpu_backend_reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
#endif