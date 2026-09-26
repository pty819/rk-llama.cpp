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
#include <array>
#include <thread>
#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

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
    double ov_busy = 0, ov_start_lat = 0, ov_notice_lat = 0; long ov_jobs = 0, ov_stall_a = 0, ov_stall_dep = 0; // opt3 overlap: NPU submit->done, pipeline stalls
};
static bool enabled() { static int e = [](){ const char* v = getenv("RKNPU_PROFILE"); return v && atoi(v) > 0; }(); return e; }
static Stats g;
static int forward_idx = 0;
static double last_graph_end = 0;
static inline double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void (*extra_dump)() = nullptr;   // npufa: NPU flash-attention stats
static void dump(const char * why) {
    if (enabled() && extra_dump) extra_dump();
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
    if (g.ov_jobs > 0)
        fprintf(stderr, "[RKNPU_PROFILE] overlap: jobs=%ld NPU busy (submit->last core done)=%.2f ms, exposed wait (matmul_run row)=%.2f ms, hidden=%.2f ms, stalls: A-conflict=%ld dep=%ld\n",
            g.ov_jobs, g.ov_busy, g.t[ST_RUN], g.ov_busy - g.t[ST_RUN], g.ov_stall_a, g.ov_stall_dep);
    if (g.ov_jobs > 0)
        fprintf(stderr, "[RKNPU_PROFILE] overlap latency: submit->last core start=%.2f ms, last core done->main sees it=%.2f ms (sum over jobs)\n", g.ov_start_lat, g.ov_notice_lat);
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

    // Per-K_op random sign vector for the Hadamard Transform. Shared across all
    // weight tensors with the same padded K so activations rotated for q/k/gate/up
    // (same src1) stay identical and A-quantization results can be reused.
    std::unordered_map<int, std::vector<float>> hadamard_s_vectors;

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
        std::shared_ptr<std::vector<float>> sp; // opt3 overlap path: per-quantization scale vector
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

// opt2 (2026-09-25): host-visible compute buffers. The RKNPU buffer memory behind activations/dst is plain
// anonymous host memory (only packed weights live in separate DMA buffers), and graph_compute reads src1 / writes dst
// with the CPU (quantize / dequant) anyway. With RKNPU_HOST_COMPUTE=1 (default):
//   - the buffer type reports is_host = true, so the CPU backend reads RKNPU split outputs in place
//     (no RKNPU->CPU get_tensor memcpy),
//   - supports_buft also accepts host buffer types, so matmul activations are read straight from CPU buffers
//     (no CPU->RKNPU set_tensor memcpy).
// Weights are still uploaded via set_tensor (mmap load path); the NPU A/C DMA buffers keep their explicit mem_sync.
// Note: is_host also affects llama's --no-mmap load path (raw read into tensor->data); that path is not supported
// with RKNPU_HOST_COMPUTE=1 (packed weights would never be built; the int8 scale lookup asserts).
static bool rknpu_host_compute() {
    static const bool v = [](){ const char* e = std::getenv("RKNPU_HOST_COMPUTE"); return e ? std::atoi(e) != 0 : true; }();
    return v;
}
static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft);
static inline bool rknpu_is_rknpu_buffer(const ggml_backend_buffer_t buf) {
    return buf && buf->buft && buf->buft->iface.get_name == ggml_backend_rknpu_buffer_type_get_name;
}

// Function for acquiring a pointer for tensor data
static void* get_tensor_real_ptr(const struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return nullptr;
    // tensors in foreign (CPU host) buffers: data pointer is directly usable
    if (!rknpu_is_rknpu_buffer(tensor->view_src ? tensor->view_src->buffer : tensor->buffer)) return tensor->data;

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

// ---------------------------------------------------------------------------------------------------------------
// opt3 (2026-09-25): overlap A quantization / C dequantization with the NPU matmul run (RKNPU_OVERLAP=1 enables; default OFF).
// Every (matmul, M tile, K segment) of one graph_compute call becomes a job. Contexts, B binding, A/C buffers and the
// A-reuse decisions are resolved serially up front (in job order, same decisions as the serial path). Execution is a
// 3-stage software pipeline on the main thread + persistent NPU-submit threads (one per core segment):
//     Q(0); for j: set_io(j); submit run(j) -> NPU threads;  D(j-1); Q(j+1);  wait run(j);   D(last)
// Q = quantize (omp) + mem_sync A TO_DEVICE, D = mem_sync C FROM_DEVICE + dequant (omp). C buffers are double-buffered
// (slot = job index & 1), each A buffer is per (tile, K offset); per-job A scale vectors are reference-counted so a later
// re-quantization of the same A buffer never clobbers scales an earlier job still needs. Hazards handled:
//   - Q(j+1) targets the same A buffer run(j) reads            -> wait run(j) first
//   - src1 of job j+1 overlaps dst of job j (not yet dequant'd) -> wait run(j), D(j), then Q(j+1)
// Q and D stay in job order, so K-segment accumulation order and all arithmetic are unchanged (bit-identical output).
// Default (RKNPU_OVERLAP unset/0) is the original serial path. Interim result: ~100-150 ms/forward gain at 1023 tok, but
// NPU runs slow down under concurrent CPU memory traffic; not yet enabled by default.
// ---------------------------------------------------------------------------------------------------------------
static bool rknpu_overlap() {
    static const bool v = [](){ const char* e = std::getenv("RKNPU_OVERLAP"); return e ? std::atoi(e) != 0 : false; }();
    return v;
}

// CPU threads used for quantize/dequant in the overlap path (0 = OpenMP default). Leaving one core of the
// taskset free lets the NPU-submit threads wake up / return from the run ioctl without waiting for a CFS preemption.
static int rknpu_ov_threads() {
    static const int v = [](){ const char* e = std::getenv("RKNPU_OV_THREADS"); return e ? std::atoi(e) : 0; }();
    return v > 0 ? v : omp_get_max_threads();
}
// Optional CPU list for the NPU-submit threads (e.g. "0-3" = little cores); unset = inherit process affinity.
static void rknpu_ov_set_submit_affinity() {
    const char* e = std::getenv("RKNPU_SUBMIT_CPUS");
    if (!e || !*e) return;
    auto cpus = parse_domain_list(e);
    cpu_set_t set; CPU_ZERO(&set);
    for (int c : cpus) if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    if (CPU_COUNT(&set) > 0) pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static inline void rk_futex_wait(std::atomic<int>* a, int val) {
    syscall(SYS_futex, reinterpret_cast<int*>(a), FUTEX_WAIT_PRIVATE, val, nullptr, nullptr, 0);
}
static inline void rk_futex_wake(std::atomic<int>* a) {
    syscall(SYS_futex, reinterpret_cast<int*>(a), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}
static inline void rk_cpu_relax() { __asm__ __volatile__("yield" ::: "memory"); }
static inline uint64_t rk_cntvct() { uint64_t v; __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(v) :: "memory"); return v; }
static inline uint64_t rk_cntfrq() { uint64_t v; __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(v)); return v; }
// Spin budget (microseconds, sched_yield between polls) before a waiter sleeps on the futex.
// NPU-submit threads: RKNPU_SPIN_US (default 2000; a job cycle is ~0.5-1.5 ms, so they rarely sleep inside a split);
// main thread waiting for the NPU: RKNPU_WAIT_SPIN_US (default 5000).
static uint64_t rk_spin_ticks(const char * env, int def_us) {
    const char* e = std::getenv(env);
    const int us = e ? std::atoi(e) : def_us;
    return (uint64_t)((double)rk_cntfrq() * (us > 0 ? us : 0) / 1e6);
}

#define RK_MAX_SEG 8
struct RkJob {
    const ggml_tensor * node = nullptr;
    const ggml_tensor * src1 = nullptr;
    int M_op = 0, Mt = 0, K = 0, K_op = 0, N = 0, K_seg_op = 0, offset_k = 0, k_idx = 0;
    bool first_k = true, hadamard = false, need_quant = false, deq_done = false;
    int npu_type_a = 0, npu_type_c = 0;
    const float * x = nullptr;
    float * dst_tile = nullptr;
    int row_stride = 0;
    const float * s_vec = nullptr;
    const float * scales_B_grid = nullptr;
    int nseg = 0;
    std::shared_ptr<rknpu_matmul_context> ctx[RK_MAX_SEG];
    std::shared_ptr<rknn_tensor_mem> C[RK_MAX_SEG];
    int n_off[RK_MAX_SEG] = {0}, n_size[RK_MAX_SEG] = {0};
    std::shared_ptr<rknn_tensor_mem> A;
    std::shared_ptr<std::vector<float>> scales;
    double t_submit = 0;
    double core_ms[RK_MAX_SEG] = {0};
    double core_end[RK_MAX_SEG] = {0};
    double core_start[RK_MAX_SEG] = {0};
};

// Persistent NPU-submit threads: thread i runs rknn_matmul_run for segment i of the submitted job (blocking ioctl).
struct RkNpuPool {
    int nthreads;
    std::vector<std::thread> th;
    std::atomic<int> gen{0};
    std::atomic<int> pending{0};
    std::atomic<RkJob*> jobp{nullptr};
    uint64_t worker_spin = rk_spin_ticks("RKNPU_SPIN_US", 2000);
    uint64_t wait_spin = rk_spin_ticks("RKNPU_WAIT_SPIN_US", 5000);
    explicit RkNpuPool(int n) : nthreads(n) {
        for (int i = 0; i < n; ++i) th.emplace_back([this, i]{ worker(i); });
    }
    void worker(int i) {
        rknpu_ov_set_submit_affinity();
        int last = 0;
        for (;;) {
            int g;
            uint64_t t_spin0 = 0;
            while ((g = gen.load(std::memory_order_acquire)) == last) {
                const uint64_t now = rk_cntvct();
                if (t_spin0 == 0) t_spin0 = now;
                if (now - t_spin0 < worker_spin) { sched_yield(); continue; }
                rk_futex_wait(&gen, last);
                t_spin0 = 0;
            }
            last = g;
            RkJob * jb = jobp.load(std::memory_order_acquire);
            if (jb && i < jb->nseg) {
                const bool prof = rknpu_prof::enabled();
                double t0 = prof ? rknpu_prof::now_ms() : 0.0;
                if (prof) jb->core_start[i] = t0;
                int ret = rknn_matmul_run(jb->ctx[i]->ctx);
                if (ret != RKNN_SUCC) fprintf(stderr, "RKNPU: rknn_matmul_run failed ret=%d\n", ret);
                if (prof) { jb->core_end[i] = rknpu_prof::now_ms(); jb->core_ms[i] = jb->core_end[i] - t0; }
            }
            if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) rk_futex_wake(&pending);
        }
    }
    void submit(RkJob * jb) {
        pending.store(nthreads, std::memory_order_relaxed);
        jobp.store(jb, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_release);
        rk_futex_wake(&gen);
    }
    void wait() {
        int p;
        uint64_t t_spin0 = 0;
        while ((p = pending.load(std::memory_order_acquire)) != 0) {
            const uint64_t now = rk_cntvct();
            if (t_spin0 == 0) t_spin0 = now;
            if (now - t_spin0 < wait_spin) { sched_yield(); continue; }
            rk_futex_wait(&pending, p);
            t_spin0 = 0;
        }
    }
};
static RkNpuPool * rknpu_get_pool(int nseg) {
    // threads are intentionally leaked (idle in futex wait at process exit)
    static RkNpuPool * pool = nullptr;
    if (!pool || pool->nthreads < nseg) {
        if (pool) { std::fprintf(stderr, "RKNPU: overlap pool grows to %d threads\n", nseg); }
        pool = new RkNpuPool(std::max(nseg, 3));
    }
    return pool;
}

static inline bool rk_tensor_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *)a->data; const char * a1 = a0 + ggml_nbytes(a);
    const char * b0 = (const char *)b->data; const char * b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

// NEON C-matrix epilogue: dst = src * (sa*wscale), optionally accumulated.
// Bit-identical to the scalar loops it replaces: int->float conversion is exact
// for |x| < 2^24 (per-segment int8 dot products stay far below), the combined
// scale sa*wscale[i] is rounded once before the product, and accumulation uses
// explicit vmulq/vaddq — never fmla, which fuses into a single rounding on A76.
static inline void rknpu_epi_f32(float * dst, const float * src, int n, float sa, const float * wscale, bool accum) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    if (wscale) {
        for (; i + 4 <= n; i += 4) {
            float32x4_t w = vmulq_f32(vld1q_f32(wscale + i), vdupq_n_f32(sa));
            float32x4_t p = vmulq_f32(vld1q_f32(src + i), w);
            if (accum) p = vaddq_f32(vld1q_f32(dst + i), p);
            vst1q_f32(dst + i, p);
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            float32x4_t p = vmulq_f32(vld1q_f32(src + i), vdupq_n_f32(sa));
            if (accum) p = vaddq_f32(vld1q_f32(dst + i), p);
            vst1q_f32(dst + i, p);
        }
    }
    for (; i < n; ++i) {
        float p = src[i] * (wscale ? wscale[i] * sa : sa);
        dst[i] = accum ? dst[i] + p : p;
    }
#else
    for (int i = 0; i < n; ++i) {
        float p = src[i] * (wscale ? wscale[i] * sa : sa);
        dst[i] = accum ? dst[i] + p : p;
    }
#endif
}

static inline void rknpu_epi_i32(float * dst, const int32_t * src, int n, float sa, const float * wscale, bool accum) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    if (wscale) {
        for (; i + 4 <= n; i += 4) {
            float32x4_t w = vmulq_f32(vld1q_f32(wscale + i), vdupq_n_f32(sa));
            float32x4_t p = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + i)), w);
            if (accum) p = vaddq_f32(vld1q_f32(dst + i), p);
            vst1q_f32(dst + i, p);
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            float32x4_t p = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + i)), vdupq_n_f32(sa));
            if (accum) p = vaddq_f32(vld1q_f32(dst + i), p);
            vst1q_f32(dst + i, p);
        }
    }
    for (; i < n; ++i) {
        float p = (float)src[i] * (wscale ? wscale[i] * sa : sa);
        dst[i] = accum ? dst[i] + p : p;
    }
#else
    for (int i = 0; i < n; ++i) {
        float p = (float)src[i] * (wscale ? wscale[i] * sa : sa);
        dst[i] = accum ? dst[i] + p : p;
    }
#endif
}

static void rk_job_quant(ggml_backend_rknpu_context * backend_ctx, RkJob & jb) {
    UNUSED(backend_ctx);
    if (!jb.need_quant) return;
    RKPROF_BEGIN(t_qa);
    float * sA = jb.scales->data();
    void * dst_base = jb.A->virt_addr;
    const int Mt = jb.Mt, K = jb.K, K_op = jb.K_op, K_seg_op = jb.K_seg_op, off_k = jb.offset_k, row_stride = jb.row_stride;
    const bool is_hadamard = jb.hadamard;
    const float * x = jb.x; const float * s_vec = jb.s_vec;
    const int ta = jb.npu_type_a;
    #pragma omp parallel for num_threads(rknpu_ov_threads())
    for (int m = 0; m < Mt; ++m) {
        const float* src_row = x + (size_t)m * row_stride;
        const float* ready_row = src_row + off_k;
        if (is_hadamard) {
            thread_local static std::vector<float> had_scratch;
            if ((int)had_scratch.size() < K_op) had_scratch.resize(K_op);
            rknpu2_calibration::hadamard_signed_fwht(had_scratch.data(), src_row, s_vec, K, K_op);
            ready_row = had_scratch.data() + off_k;
        }
        if (ta == rknpu2_configuration::NPU_TYPE_FP16) {
            uint16_t* dst_row = (uint16_t*)dst_base + (size_t)m * K_seg_op;
            rknpu2_quantization::convert_fp32_to_fp16(ready_row, dst_row, K_seg_op);
        } else if (ta == rknpu2_configuration::NPU_TYPE_INT8) {
            float amax_m = rknpu_row_amax(ready_row, K_seg_op);
            sA[m] = amax_m / 127.0f;
            int8_t* dst_row = (int8_t*)dst_base + (size_t)m * K_seg_op;
            rknpu_quant_row_i8(ready_row, dst_row, K_seg_op, sA[m]);
        } else if (ta == rknpu2_configuration::NPU_TYPE_INT4) {
            float amax_m = rknpu_row_amax(ready_row, K_seg_op);
            sA[m] = amax_m / 7.0f;
            uint8_t* dst_row = (uint8_t*)dst_base + (size_t)m * (K_seg_op / 2);
            rknpu2_quantization::quantize_fp32_to_int4_packed(ready_row, dst_row, K_seg_op, sA[m]);
        }
    }
    RKPROF_END(t_qa, ST_QUANT_A);
    RKPROF(ST_SYNC_A);
    RKNN_CHECK(rknn_mem_sync(jb.ctx[0]->ctx, jb.A.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");
}

static void rk_job_dequant(RkJob & jb) {
    if (jb.deq_done) return;
    {
        RKPROF(ST_SYNC_C);
        for (int idx = 0; idx < jb.nseg; idx++) {
            RKNN_CHECK(rknn_mem_sync(jb.ctx[idx]->ctx, jb.C[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
        }
    }
    RKPROF(ST_DEQUANT);
    const float hadamard_divisor = jb.hadamard ? (float)jb.K_op : 1.0f;
    const int npu_type_c = jb.npu_type_c;
    const int Mt = jb.Mt, N = jb.N, nseg = jb.nseg, k_idx = jb.k_idx;
    const bool first_k = jb.first_k;
    const float * scales_A = jb.scales->data();
    const float * scales_B_grid = jb.scales_B_grid;
    float * dst_tile = jb.dst_tile;
    #pragma omp parallel for num_threads(rknpu_ov_threads())
    for (int m = 0; m < Mt; m++) {
        const float sa = scales_A[m] / hadamard_divisor;
        for (int idx = 0; idx < nseg; idx++) {
            const int N_offset = jb.n_off[idx];
            const int N_segment = jb.n_size[idx];
            const float* wscale = scales_B_grid ? (scales_B_grid + (size_t)k_idx * N + N_offset) : nullptr;
            float* dst_ptr = dst_tile + (size_t)m * N + N_offset;
            if (npu_type_c == rknpu2_configuration::NPU_TYPE_FP32) {
                const float* src_ptr = (const float*)jb.C[idx]->virt_addr + (size_t)m * N_segment;
                rknpu_epi_f32(dst_ptr, src_ptr, N_segment, sa, wscale, !first_k);
            } else if (npu_type_c == rknpu2_configuration::NPU_TYPE_INT32) {
                const int32_t* src_ptr = (const int32_t*)jb.C[idx]->virt_addr + (size_t)m * N_segment;
                rknpu_epi_i32(dst_ptr, src_ptr, N_segment, sa, wscale, !first_k);
            } else if (npu_type_c == rknpu2_configuration::NPU_TYPE_INT16) {
                const int16_t* src_ptr = (const int16_t*)jb.C[idx]->virt_addr + (size_t)m * N_segment;
                if (first_k) { for (int n = 0; n < N_segment; ++n) dst_ptr[n]  = (float)src_ptr[n] * (sa * (wscale ? wscale[n] : 1.0f)); }
                else         { for (int n = 0; n < N_segment; ++n) dst_ptr[n] += (float)src_ptr[n] * (sa * (wscale ? wscale[n] : 1.0f)); }
            }
        }
    }
    jb.deq_done = true;
}

template <typename Cfg>
static enum ggml_status rknpu_graph_compute_overlap(ggml_backend_rknpu_context * backend_ctx, struct ggml_cgraph * cgraph, const Cfg & config) {
    RKPROF(ST_MATMUL);
    std::vector<RkJob> jobs;
    jobs.reserve(64);
    static const bool no_a_reuse = std::getenv("RKNPU_NO_A_REUSE") != nullptr;

    // ---------------- job list (serial prep, same resource/reuse decisions as the serial path) ----------------
    {
    RKPROF(ST_SETUP);
    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        struct ggml_tensor* node = cgraph->nodes[node_i];
        if (node->op != GGML_OP_MUL_MAT) continue;
        const struct ggml_tensor* src0 = node->src[0];
        const struct ggml_tensor* src1 = node->src[1];
        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];
        if (M == 0 || K == 0 || N == 0) continue;
        const auto* pipeline = config.resolve_op_support(src0);
        if (!pipeline) continue;
        const bool is_hadamard = (pipeline->use_hadamard);
        const int K_op = is_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;
        const rknn_matmul_type matmul_type = pipeline->mm_type;
        const int alignment = pipeline->n_align;
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }
        auto all_k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto all_n_segments = compute_n_segments(N, config.active_cores, alignment);
        std::vector<MatrixSegmentN> active_n_segments;
        for (const auto& seg : all_n_segments) if (seg.size_n > 0) active_n_segments.push_back(seg);
        if (active_n_segments.empty()) continue;
        GGML_ASSERT(active_n_segments.size() <= RK_MAX_SEG);
        const int m_tile = rknpu_m_tile() > 0 ? rknpu_m_tile() : M;
        if (rknpu_prof::enabled()) {
            rknpu_prof::g.n_matmul++; rknpu_prof::g.macs_real += (double)M*K*N;
            for (int m0 = 0; m0 < M; m0 += m_tile) { int mt = std::min(m_tile, M - m0); int mo = mt > 1 ? rknpu2_calibration::next_power_of_two(mt) : 1; rknpu_prof::g.macs_op += (double)mo*K_op*N; }
        }
        const size_t num_active_segments = active_n_segments.size();

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
            if (is_hadamard) {
                auto its = src0_buf_ctx->hadamard_s_vectors.find(K_op);
                GGML_ASSERT(its != src0_buf_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
                s_vec = its->second.data();
            }
            if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 || pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                auto itq = src0_buf_ctx->quantized_tensor_scales.find(src0);
                GGML_ASSERT(itq != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scales grid not found");
                scales_B_grid = itq->second.data();
            }
        }
        float* dst_data = (float*)get_tensor_real_ptr(node);
        const float* x_all = (const float*)get_tensor_real_ptr(src1);
        const int row_stride = (int)(src1->nb[1] / sizeof(float));
        size_t type_size_packed = 0;
        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) type_size_packed = 2;
        else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) type_size_packed = 1;

        for (int m0 = 0; m0 < M; m0 += m_tile) {
            const int Mt = std::min(m_tile, M - m0);
            const int M_op = Mt > 1 ? rknpu2_calibration::next_power_of_two(Mt) : 1;
            size_t current_offset_in_tensor = 0;
            for (size_t k_idx = 0; k_idx < all_k_segments.size(); ++k_idx) {
                const auto& k_seg = all_k_segments[k_idx];
                const int K_seg_op = k_seg.size_k;
                RkJob jb;
                jb.node = node; jb.src1 = src1;
                jb.M_op = M_op; jb.Mt = Mt; jb.K = K; jb.K_op = K_op; jb.N = N; jb.K_seg_op = K_seg_op;
                jb.offset_k = k_seg.offset_k; jb.k_idx = (int)k_idx; jb.first_k = (k_idx == 0); jb.hadamard = is_hadamard;
                jb.npu_type_a = (int)pipeline->npu_type_a; jb.npu_type_c = (int)pipeline->npu_type_c;
                jb.x = x_all + (size_t)m0 * row_stride; jb.dst_tile = dst_data + (size_t)m0 * N; jb.row_stride = row_stride;
                jb.s_vec = s_vec; jb.scales_B_grid = scales_B_grid;
                jb.nseg = (int)num_active_segments;

                // 1. contexts + B binding
                for (const auto& n_seg : all_n_segments) {
                    for (size_t idx = 0; idx < num_active_segments; ++idx) {
                        if (active_n_segments[idx].offset_n == n_seg.offset_n) {
                            size_t offset_in_dma = current_offset_in_tensor;
                            {
                            RKPROF(ST_CTX);
                            jb.ctx[idx] = backend_ctx->get_matmul_ctx((uintptr_t)tensor_virt_addr, offset_in_dma, M_op, K_seg_op, n_seg.size_n,
                                                                      n_seg.core_id, matmul_type, b_domain_id);
                            }
                            if (!jb.ctx[idx] || jb.ctx[idx]->ctx == 0) return GGML_STATUS_FAILED;
                            auto& matmul_ctx = jb.ctx[idx];
                            if (!matmul_ctx->b_bound) {
                                RKPROF(ST_BIND_B);
                                if (rknpu_prof::enabled()) rknpu_prof::g.n_b_bind++;
                                size_t segment_size_bytes = matmul_ctx->io_attr.B.size;
                                rknn_tensor_mem* mem = rknn_create_mem_from_fd(matmul_ctx->ctx, tensor_fd, tensor_virt_addr, segment_size_bytes, offset_in_dma);
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
                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                    jb.n_off[idx] = active_n_segments[idx].offset_n; jb.n_size[idx] = active_n_segments[idx].size_n;
                }

                // 2. A buffer + reuse decision (quantization itself happens later, in job order)
                {
                    auto cache_key = std::make_tuple(M_op, K_seg_op, (int)pipeline->npu_type_a, b_domain_id, m0, k_seg.offset_k);
                    {
                    RKPROF(ST_GET_A);
                    jb.A = get_tensor_buffer(backend_ctx, jb.ctx[0]->ctx, jb.ctx[0]->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache);
                    }
                    if (!jb.A) return GGML_STATUS_FAILED;
                    auto& astate = backend_ctx->a_state[jb.A.get()];
                    const bool reuse = astate.epoch == backend_ctx->graph_epoch && astate.src1 == src1 && astate.data == src1->data &&
                                       astate.Mt == Mt && astate.pipeline_a == (int)pipeline->npu_type_a && astate.hadamard == is_hadamard &&
                                       astate.s_vec == s_vec && astate.sp && !no_a_reuse;
                    if (reuse) {
                        if (rknpu_prof::enabled()) rknpu_prof::g.n_a_reuse++;
                        jb.need_quant = false;
                    } else {
                        jb.need_quant = true;
                        astate.sp = std::make_shared<std::vector<float>>(Mt, 1.0f);
                        astate.epoch = backend_ctx->graph_epoch; astate.src1 = src1; astate.data = src1->data; astate.Mt = Mt;
                        astate.pipeline_a = (int)pipeline->npu_type_a; astate.hadamard = is_hadamard; astate.s_vec = s_vec;
                    }
                    jb.scales = astate.sp;
                }

                // 3. C buffers: double-buffered by job parity
                {
                    const int slot = (int)(jobs.size() & 1);
                    for (size_t idx = 0; idx < num_active_segments; idx++) {
                        auto cache_key = std::make_tuple(M_op, active_n_segments[idx].size_n, active_n_segments[idx].core_id,
                                                         (int)pipeline->npu_type_c + 1000 * (slot + 1), b_domain_id);
                        jb.C[idx] = get_tensor_buffer(backend_ctx, jb.ctx[idx]->ctx, jb.ctx[idx]->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache);
                        if (!jb.C[idx]) return GGML_STATUS_FAILED;
                    }
                }
                jobs.push_back(std::move(jb));
            }
        }
    }
    }
    const int J = (int)jobs.size();
    if (J == 0) return GGML_STATUS_SUCCESS;
    int max_seg = 0;
    for (auto & jb : jobs) max_seg = std::max(max_seg, jb.nseg);
    RkNpuPool * pool = rknpu_get_pool(max_seg);
    const bool prof = rknpu_prof::enabled();

    auto setio_submit = [&](RkJob & jb) {
        {
        RKPROF(ST_SETIO_A);
        for (int idx = 0; idx < jb.nseg; idx++)
            RKNN_CHECK(rknn_matmul_set_io_mem(jb.ctx[idx]->ctx, jb.A.get(), &jb.ctx[idx]->io_attr.A), "set_io_mem A for core");
        }
        {
        RKPROF(ST_SETIO_C);
        for (int idx = 0; idx < jb.nseg; idx++)
            RKNN_CHECK(rknn_matmul_set_io_mem(jb.ctx[idx]->ctx, jb.C[idx].get(), &jb.ctx[idx]->io_attr.C), "set_io_mem C");
        }
        if (prof) jb.t_submit = rknpu_prof::now_ms();
        pool->submit(&jb);
    };
    auto wait_run = [&](RkJob & jb) {
        {
        RKPROF(ST_RUN);   // overlap mode: exposed (main-thread blocked) NPU wait
        pool->wait();
        }
        if (prof) {
            double mx = 0, sm = 0, end = jb.t_submit, st = 0;
            for (int idx = 0; idx < jb.nseg; idx++) { sm += jb.core_ms[idx]; mx = std::max(mx, jb.core_ms[idx]); end = std::max(end, jb.core_end[idx]); st = std::max(st, jb.core_start[idx] - jb.t_submit); }
            rknpu_prof::g.ov_start_lat += st; rknpu_prof::g.ov_notice_lat += rknpu_prof::now_ms() - end;
            rknpu_prof::g.run_core_sum += sm; rknpu_prof::g.run_core_max_sum += mx;
            rknpu_prof::g.n_run += jb.nseg;
            rknpu_prof::g.ov_busy += end - jb.t_submit; rknpu_prof::g.ov_jobs++;
            auto & e = rknpu_prof::g.run_by_shape[std::make_tuple(jb.M_op, jb.K_seg_op, jb.N)];
            e.first++; e.second += end - jb.t_submit;
        }
    };

    rk_job_quant(backend_ctx, jobs[0]);
    for (int j = 0; j < J; ++j) {
        RkJob & cur = jobs[j];
        setio_submit(cur);
        bool waited = false;
        if (j > 0) rk_job_dequant(jobs[j - 1]);
        if (j + 1 < J) {
            RkJob & nx = jobs[j + 1];
            if (nx.need_quant) {
                const bool conflict_a = nx.A.get() == cur.A.get();
                const bool dep = rk_tensor_overlap(nx.src1, cur.node);
                if (conflict_a || dep) {
                    wait_run(cur); waited = true;
                    if (prof) { if (dep) rknpu_prof::g.ov_stall_dep++; else rknpu_prof::g.ov_stall_a++; }
                    if (dep) rk_job_dequant(cur);
                }
                rk_job_quant(backend_ctx, nx);
            }
        }
        if (!waited) wait_run(cur);
    }
    rk_job_dequant(jobs[J - 1]);
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------
// npufa (2026-09-26): FLASH_ATTN_EXT on the NPU (RKNPU_FA=1; default off = previous behaviour).
// Non-fused: per (query tile of RKNPU_FA_MT rows, KV head): S = Q.K^T on the NPU (fp16 x fp16 -> fp32; the GQA Q heads
// sharing the KV head are stacked in M; B = K in the NPU native layout by default — see rknpu_fa_native_b; legacy
// fallback binds K rows as TP_NORM), exact mask + scale + softmax on the CPU (fp32, P written as
// unnormalized fp16 straight into the PV A buffer), O = P.V on the NPU (B = V rows, normal layout), O *= 1/rowsum.
// A/C use the NPU native layouts (normal layouts make rknn_matmul_run convert on the CPU). The key range per tile is
// taken from the actual ggml mask tensor (per-row first/last unmasked key; rows whose in-range mask is not all-zero take an
// exact generic path), rounded up to a multiple of the query tile (256) and the partial last tile is zero-padded to a full
// tile, so the context set for any length is a subset of the set for the longest length (bounded by RKNPU_FA_MAX_KV, not by
// the variety of lengths). RKNPU_FA_THREADS driver threads (default 6 = 2 per NPU core) each own their contexts.
// Memory bounds: RKNPU_FA_MAX_CTX (default 192) = LRU context eviction (per driver thread); RKNPU_FA_MAX_KV (default 4096)
// = key length above which supports_op declines (CPU FA fallback).
// ---------------------------------------------------------------------------------------------------------------
static bool rknpu_fa_enabled() {
    static const bool v = [](){ const char* e = std::getenv("RKNPU_FA"); return e && std::atoi(e) != 0; }();
    return v;
}
static int rknpu_fa_env(const char * name, int def) { const char* e = std::getenv(name); int v = e ? std::atoi(e) : def; return v > 0 ? v : def; }

namespace rkfa {
static inline int rup(int x, int a) { return (x + a - 1) / a * a; }
struct Ctx { rknn_matmul_ctx ctx = 0; rknn_matmul_info info; rknn_matmul_io_attr io; rknn_tensor_mem *bA = nullptr, *bB = nullptr, *bC = nullptr; uint64_t last = 0; };
struct Mem { rknn_tensor_mem * m = nullptr; size_t size = 0; rknn_matmul_ctx owner = 0; };
struct TB { Mem q, s, p, o, ks, vs; std::vector<float> inv; std::vector<float32x4_t> mx4, sm4; };   // ks/vs: K/V rows [k0, k0+Nq) staged for jobs with k0 > 0
static std::mutex mu;
static std::map<std::tuple<int,int,int,int,int>, Ctx*> ctxs;   // (M, K, N, B layout, slot)
static Ctx * anyctx = nullptr;
static double create_ms = 0; static long n_create = 0, n_evict = 0;
static uint64_t tick = 0;          // LRU clock (under mu)
static size_t g_max_ctx = 192;
static std::vector<Mem> kbuf, vbuf;
static std::vector<TB> tbs;
static std::vector<int> row_lo, row_hi; static std::vector<uint8_t> row_clean;

// Native-B mode (RKNPU_FA_NATIVE_B=1, default): the QK/PV matmul contexts take B already in the NPU native layout
// (RK3588 fp16: (N/16, K/32, 16, 32)), so set_io_mem stops converting B on the CPU (4.9 us vs 186 us per bind,
// measured). Native B is read at run time (verified on-device: new data behind an unchanged binding is picked up),
// so B is bound once per buffer and per-layer refreshes only need the gather + mem_sync. QK's native K layout is
// N-prefix compatible (one shared buffer per KV head; shorter Nq binds the same prefix), but PV's native V layout is
// not (its stride depends on Nq), so each PV job interleaves its own window into the per-thread staging buffer.
static bool rknpu_fa_native_b() {
    static const bool ok = [](){
        const char * e = std::getenv("RKNPU_FA_NATIVE_B");
        if (e && std::atoi(e) == 0) return false;
        rknn_matmul_ctx c; rknn_matmul_info info; rknn_matmul_io_attr io;
        memset(&info, 0, sizeof(info)); memset(&io, 0, sizeof(io));
        info.M = 64; info.K = 128; info.N = 256; info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE; info.AC_layout = RKNN_MM_LAYOUT_NATIVE;
        const int ret = rknn_matmul_create(&c, &info, &io);
        if (ret < 0) return false;
        const bool good = io.B.n_dims == 4 && io.B.dims[0] == 16 && io.B.dims[1] == 4 && io.B.dims[2] == 16 && io.B.dims[3] == 32;
        rknn_matmul_destroy(c);
        return good;
    }();
    return ok;
}
// K rows [j][d] -> QK native B (nkv/16, D/32, 16, 32): per key one D*2-byte row becomes D/32 chunks of 64 B.
// Byte-identical to the runtime's own conversion (verified against rknn_B_normal_layout_to_native_layout).
static void knat_fill(const ggml_tensor * k, int g, int nkv, int D, __fp16 * out) {
    for (int j = 0; j < nkv; j++) {
        const __fp16 * src = (const __fp16 *)((const char *)k->data + (size_t)j * k->nb[1] + (size_t)g * k->nb[2]);
        __fp16 * dst = out + (size_t)(j / 16) * (D / 32) * 512 + (size_t)(j % 16) * 32;
        for (int db = 0; db < D / 32; db++)
            for (int c = 0; c < 32; c += 8)
                vst1q_f16(dst + (size_t)db * 512 + c, vld1q_f16(src + (size_t)db * 32 + c));
    }
}
static inline void trn8x8_f16(const __fp16 * const in[8], __fp16 * const out[8]) {
    uint16x8_t x0 = vld1q_u16((const uint16_t *)in[0]), x1 = vld1q_u16((const uint16_t *)in[1]);
    uint16x8_t x2 = vld1q_u16((const uint16_t *)in[2]), x3 = vld1q_u16((const uint16_t *)in[3]);
    uint16x8_t x4 = vld1q_u16((const uint16_t *)in[4]), x5 = vld1q_u16((const uint16_t *)in[5]);
    uint16x8_t x6 = vld1q_u16((const uint16_t *)in[6]), x7 = vld1q_u16((const uint16_t *)in[7]);
    uint16x8x2_t t01 = vtrnq_u16(x0, x1), t23 = vtrnq_u16(x2, x3), t45 = vtrnq_u16(x4, x5), t67 = vtrnq_u16(x6, x7);
    uint32x4x2_t u0 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[0]), vreinterpretq_u32_u16(t23.val[0]));
    uint32x4x2_t u1 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[1]), vreinterpretq_u32_u16(t23.val[1]));
    uint32x4x2_t u2 = vtrnq_u32(vreinterpretq_u32_u16(t45.val[0]), vreinterpretq_u32_u16(t67.val[0]));
    uint32x4x2_t u3 = vtrnq_u32(vreinterpretq_u32_u16(t45.val[1]), vreinterpretq_u32_u16(t67.val[1]));
    uint64x2_t v0lo = vtrn1q_u64(vreinterpretq_u64_u32(u0.val[0]), vreinterpretq_u64_u32(u2.val[0]));
    uint64x2_t v0hi = vtrn2q_u64(vreinterpretq_u64_u32(u0.val[0]), vreinterpretq_u64_u32(u2.val[0]));
    uint64x2_t v1lo = vtrn1q_u64(vreinterpretq_u64_u32(u1.val[0]), vreinterpretq_u64_u32(u3.val[0]));
    uint64x2_t v1hi = vtrn2q_u64(vreinterpretq_u64_u32(u1.val[0]), vreinterpretq_u64_u32(u3.val[0]));
    uint64x2_t v2lo = vtrn1q_u64(vreinterpretq_u64_u32(u0.val[1]), vreinterpretq_u64_u32(u2.val[1]));
    uint64x2_t v2hi = vtrn2q_u64(vreinterpretq_u64_u32(u0.val[1]), vreinterpretq_u64_u32(u2.val[1]));
    uint64x2_t v3lo = vtrn1q_u64(vreinterpretq_u64_u32(u1.val[1]), vreinterpretq_u64_u32(u3.val[1]));
    uint64x2_t v3hi = vtrn2q_u64(vreinterpretq_u64_u32(u1.val[1]), vreinterpretq_u64_u32(u3.val[1]));
    vst1q_u16((uint16_t *)out[0], vreinterpretq_u16_u64(v0lo));
    vst1q_u16((uint16_t *)out[1], vreinterpretq_u16_u64(v1lo));
    vst1q_u16((uint16_t *)out[2], vreinterpretq_u16_u64(v2lo));
    vst1q_u16((uint16_t *)out[3], vreinterpretq_u16_u64(v3lo));
    vst1q_u16((uint16_t *)out[4], vreinterpretq_u16_u64(v0hi));
    vst1q_u16((uint16_t *)out[5], vreinterpretq_u16_u64(v1hi));
    vst1q_u16((uint16_t *)out[6], vreinterpretq_u16_u64(v2hi));
    vst1q_u16((uint16_t *)out[7], vreinterpretq_u16_u64(v3hi));
}
// V rows [j][d] (window [k0, k0+Nq) of one KV head) -> PV native B (D/16, Nq/32, 16, 32) via 8x8 transposes;
// Nq and D are multiples of 32 (tile + head-dim constraints). Same byte layout as the runtime's conversion.
static void vnat_fill(const ggml_tensor * v, int g, int k0, int Nq, int D, __fp16 * out) {
    const int nk32 = Nq / 32;
    const __fp16 * base = (const __fp16 *)((const char *)v->data + (size_t)g * v->nb[2]);
    for (int j8 = 0; j8 + 8 <= Nq; j8 += 8)
        for (int d8 = 0; d8 + 8 <= D; d8 += 8) {
            const __fp16 * in[8]; __fp16 * o[8];
            for (int r = 0; r < 8; r++) in[r] = (const __fp16 *)((const char *)base + (size_t)(k0 + j8 + r) * v->nb[1]) + d8;
            for (int r = 0; r < 8; r++) {
                const int d = d8 + r;
                o[r] = out + (size_t)(d / 16) * nk32 * 512 + (size_t)(j8 / 32) * 512 + (size_t)(d % 16) * 32 + (j8 % 32);
            }
            trn8x8_f16(in, o);
        }
    for (int j = Nq - Nq % 8; j < Nq; j++)   // unreachable with tile-aligned Nq; kept for safety
        for (int d = 0; d < D; d++)
            out[(size_t)(d / 16) * nk32 * 512 + (size_t)(j / 32) * 512 + (size_t)(d % 16) * 32 + (j % 32)] =
                *(const __fp16 *)((const char *)base + (size_t)(k0 + j) * v->nb[1] + (size_t)d * 2);
}

// Contexts of a slot are only used by that slot's driver thread, so a thread may evict its own LRU contexts inside a
// call (except `keep`, the context it currently holds, and anyctx which owns the buffers). Hard cap = max(RKNPU_FA_MAX_CTX,
// what is needed for one job per thread).
static Ctx * get_ctx(int M, int K, int N, int layout, int slot, Ctx * keep = nullptr) {
    std::lock_guard<std::mutex> lk(mu);
    auto key = std::make_tuple(M, K, N, layout, slot);
    auto it = ctxs.find(key);
    if (it != ctxs.end()) { it->second->last = ++tick; return it->second; }
    while (ctxs.size() >= g_max_ctx) {
        auto victim = ctxs.end();
        for (auto jt = ctxs.begin(); jt != ctxs.end(); ++jt) {
            if (std::get<4>(jt->first) != slot || jt->second == anyctx || jt->second == keep) continue;
            if (victim == ctxs.end() || jt->second->last < victim->second->last) victim = jt;
        }
        if (victim == ctxs.end()) break;
        rknn_matmul_destroy(victim->second->ctx); delete victim->second; ctxs.erase(victim); n_evict++;
    }
    double t0 = rknpu_prof::now_ms();
    Ctx * c = new Ctx();
    c->last = ++tick;
    memset(&c->info, 0, sizeof(c->info)); memset(&c->io, 0, sizeof(c->io));
    c->info.M = M; c->info.K = K; c->info.N = N; c->info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    c->info.B_layout = layout; c->info.AC_layout = 1;
    int ret = rknn_matmul_create(&c->ctx, &c->info, &c->io);
    if (ret < 0) { fprintf(stderr, "RKNPU FA: rknn_matmul_create failed %d (M=%d K=%d N=%d layout=%d)\n", ret, M, K, N, layout); delete c; return nullptr; }
    if (c->io.A.dims[c->io.A.n_dims - 1] != 8 || c->io.C.dims[c->io.C.n_dims - 1] != 4) {
        fprintf(stderr, "RKNPU FA: unexpected native layout (A group %u, C group %u)\n", c->io.A.dims[c->io.A.n_dims - 1], c->io.C.dims[c->io.C.n_dims - 1]);
        rknn_matmul_destroy(c->ctx); delete c; return nullptr;
    }
    const int core = slot % 3;
    rknn_matmul_set_core_mask(c->ctx, core == 0 ? RKNN_NPU_CORE_0 : core == 1 ? RKNN_NPU_CORE_1 : RKNN_NPU_CORE_2);
    if (!anyctx) anyctx = c;
    create_ms += rknpu_prof::now_ms() - t0; n_create++;
    ctxs[key] = c;
    return c;
}
static void clear_binds() { for (auto & kv : ctxs) { kv.second->bA = kv.second->bB = kv.second->bC = nullptr; } }
static bool ensure(Mem & mm, size_t size) {   // call only while no driver thread runs
    if (mm.m && mm.size >= size) return true;
    if (mm.m) rknn_destroy_mem(mm.owner, mm.m);
    mm.m = rknn_create_mem(anyctx->ctx, size); mm.owner = anyctx->ctx; mm.size = size;
    if (!mm.m) { mm.size = 0; return false; }
    memset(mm.m->virt_addr, 0, size);   // V padding rows must be finite (P=0 there)
    clear_binds();
    return true;
}
static inline bool bind(Ctx * c, rknn_tensor_mem * a, rknn_tensor_mem * b, rknn_tensor_mem * cc, bool native_b = false) {
    if (c->bA != a) { if (rknn_matmul_set_io_mem(c->ctx, a, &c->io.A) < 0) return false; c->bA = a; }
    if (native_b) {
        // native B is read at run time: the binding only tracks the buffer, data refresh + mem_sync is enough
        if (c->bB != b) { if (rknn_matmul_set_io_mem(c->ctx, b, &c->io.B) < 0) return false; c->bB = b; }
    } else if (rknn_matmul_set_io_mem(c->ctx, b, &c->io.B) < 0) {
        // legacy: B is always re-bound: for non-native B layouts the runtime converts/caches B at set_io_mem time,
        // so new K/V data behind the same buffer is not picked up otherwise (stale K/V from the previous layer).
        return false;
    } else c->bB = b;
    if (c->bC != cc) { if (rknn_matmul_set_io_mem(c->ctx, cc, &c->io.C) < 0) return false; c->bC = cc; }
    return true;
}
inline static float32x4_t v_expf(float32x4_t x) {   // same as ggml_v_expf
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}
enum { F_MASK, F_GATHER, F_QFILL, F_BIND, F_RUNQK, F_SSYNC, F_SMAX, F_SEXP, F_GEN, F_PSYNC, F_RUNPV, F_OUT, F_N };
static double ft[F_N]; static long f_n = 0; static double f_wall = 0; static double f_cols = 0, f_cols_noskip = 0;
static const char * fname[F_N] = {"mask scan (wall)", "K/V gather+sync (wall)", "Q fill+sync (thr)", "bind+B fill (thr)", "run QK (thr)",
                                  "S sync (thr)", "softmax max (thr)", "softmax exp+P (thr)", "softmax generic (thr)", "P sync (thr)",
                                  "run PV (thr)", "O sync+scale (thr)"};
static void dump_prof() {
    if (f_n == 0) return;
    fprintf(stderr, "[RKNPU_PROFILE] NPU FA: nodes=%ld wall=%.2f ms (%.3f ms/node) contexts live=%zu created=%ld evicted=%ld (create %.1f ms total)\n", f_n, f_wall, f_wall / f_n, ctxs.size(), n_create, n_evict, create_ms);
    for (int i = 0; i < F_N; i++) fprintf(stderr, "[RKNPU_PROFILE]   FA %-26s %9.2f ms\n", fname[i], ft[i]);
    fprintf(stderr, "[RKNPU_PROFILE]   FA key columns computed %.3g (without key-start skip %.3g, %.1f%%)\n", f_cols, f_cols_noskip, f_cols_noskip > 0 ? 100.0 * f_cols / f_cols_noskip : 0.0);
    memset(ft, 0, sizeof(ft)); f_n = 0; f_wall = 0; f_cols = f_cols_noskip = 0;
}
} // namespace rkfa

static bool rknpu_fa_supported(const ggml_tensor * op) {
    if (!rknpu_fa_enabled()) return false;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * mask = op->src[3];
    if (!q || !k || !v) return false;
    if (op->src[4] != nullptr) return false;                         // attention sinks
    float max_bias = 0.f, softcap = 0.f;
    memcpy(&max_bias, (const float *)op->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *)op->op_params + 2, sizeof(float));
    if (max_bias != 0.f || softcap != 0.f) return false;            // ALiBi / logit softcap
    if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) return false;
    if (mask && mask->type != GGML_TYPE_F16) return false;
    const int64_t D = q->ne[0];
    if (D % 32 != 0 || D > 256 || k->ne[0] != D || v->ne[0] != D) return false;
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1) return false;
    if (k->ne[2] <= 0 || q->ne[2] % k->ne[2] != 0 || v->ne[2] != k->ne[2] || v->ne[1] != k->ne[1]) return false;
    if (q->ne[2] / k->ne[2] > 4) return false;
    static const int max_kv = std::min(8192, rknpu_fa_env("RKNPU_FA_MAX_KV", 4096));
    if (q->ne[1] < 16 || k->ne[1] > max_kv) return false;            // tiny batches stay on the CPU; key-length cap (memory / PV K limit)
    if (q->nb[0] != 4 || k->nb[0] != 2 || v->nb[0] != 2 || op->nb[0] != 4) return false;
    if (op->ne[0] != D || op->ne[1] != q->ne[2] || op->ne[2] != q->ne[1]) return false;
    if (mask && (mask->nb[0] != 2 || mask->ne[2] != 1 || mask->ne[3] != 1 || mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1])) return false;
    return true;
}

static enum ggml_status rknpu_fa_compute(const ggml_tensor * dst) {
    using namespace rkfa;
    const bool prof = rknpu_prof::enabled();
    if (prof) rknpu_prof::extra_dump = rkfa::dump_prof;
    const double T0 = rknpu_prof::now_ms();
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    float scale; memcpy(&scale, (const float *)dst->op_params + 0, sizeof(float));
    const int D = (int)q->ne[0], n = (int)q->ne[1], H = (int)q->ne[2], nkv = (int)k->ne[1], Hkv = (int)k->ne[2], ratio = H / Hkv;
    static const int mt = rkfa::rup(rknpu_fa_env("RKNPU_FA_MT", 256), 32);
    const int NB = mt;   // key-range buckets = multiples of the query tile
    g_max_ctx = (size_t)rknpu_fa_env("RKNPU_FA_MAX_CTX", 192);
    static const int nthr = std::min(32, rknpu_fa_env("RKNPU_FA_THREADS", 6));
    const int kvcap = rup(nkv, NB);

    // ---- 1. mask scan: per query row first/last unmasked key and whether the in-range mask is all zero ----
    row_lo.resize(n); row_hi.resize(n); row_clean.resize(n);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n; t++) {
        if (!mask) { row_lo[t] = 0; row_hi[t] = nkv - 1; row_clean[t] = 1; continue; }
        const uint16_t * mr = (const uint16_t *)((const char *)mask->data + (size_t)t * mask->nb[1]);
        int lo = 0; while (lo < nkv && mr[lo] == 0xFC00) lo++;
        int hi = nkv - 1; while (hi >= lo && mr[hi] == 0xFC00) hi--;
        bool clean = true;
        int j = lo;
        const uint16x8_t m7 = vdupq_n_u16(0x7FFF);
        for (; j + 8 <= hi + 1; j += 8) { if (vmaxvq_u16(vandq_u16(vld1q_u16(mr + j), m7)) != 0) { clean = false; break; } }
        if (clean) for (; j <= hi; j++) if ((mr[j] & 0x7FFF) != 0) { clean = false; break; }
        if (lo > hi) { lo = 0; hi = -1; clean = true; }   // fully masked row -> output 0
        row_lo[t] = lo; row_hi[t] = hi; row_clean[t] = clean;
    }
    const double T1 = rknpu_prof::now_ms();

    // ---- 2. jobs: (tile, kv head); tile key range from the mask, bucketed ----
    // key range [k0, k0+Nq): k0 = first unmasked key of the tile rounded down to the tile size (bundled multi-sequence
    // batches: keys of earlier sequences are skipped), Nq a multiple of the tile size -> contexts stay a subset of the
    // max-length set. K/V are bound at offset k0 through buffer views.
    static const bool kskip = [](){ const char * e = std::getenv("RKNPU_FA_KSKIP"); return !e || std::atoi(e) != 0; }();
    struct Job { int r0, mtc, mtp, g, Nq, k0; };
    std::vector<Job> jobs;
    for (int r0 = 0; r0 < n; r0 += mt) {
        const int mtc = std::min(mt, n - r0);
        int kend = 0, kbeg = INT_MAX;
        for (int i = 0; i < mtc; i++) if (row_hi[r0 + i] >= row_lo[r0 + i]) { kend = std::max(kend, row_hi[r0 + i] + 1); kbeg = std::min(kbeg, row_lo[r0 + i]); }
        if (kend <= 0) {   // fully masked tile: zeros
            for (int i = 0; i < mtc; i++) for (int h = 0; h < H; h++)
                memset((char *)dst->data + (size_t)(r0 + i) * dst->nb[2] + (size_t)h * dst->nb[1], 0, (size_t)D * sizeof(float));
            continue;
        }
        const int k1 = std::min(rup(kend, NB), kvcap);
        const int k0 = kskip ? kbeg / NB * NB : 0;
        if (prof) { f_cols += (double)Hkv * (k1 - k0); f_cols_noskip += (double)Hkv * k1; }
        for (int g = 0; g < Hkv; g++) jobs.push_back({r0, mtc, mt, g, k1 - k0, k0});   // partial tile zero-padded to a full tile (fixed M)
    }
    std::stable_sort(jobs.begin(), jobs.end(), [](const Job & a, const Job & b){ return (long)a.Nq * a.mtp > (long)b.Nq * b.mtp; });
    if (jobs.empty()) return GGML_STATUS_SUCCESS;

    // ---- 3. buffers ----
    const bool nat = rknpu_fa_native_b();
    if (!anyctx && !get_ctx(ratio * jobs[0].mtp, D, jobs[0].Nq, nat ? 1 : 2, 0)) return GGML_STATUS_FAILED;
    if ((int)kbuf.size() < Hkv) { kbuf.resize(Hkv); vbuf.resize(Hkv); }
    if ((int)tbs.size() < nthr) tbs.resize(nthr);
    const size_t Mmax = (size_t)ratio * mt;
    for (int g = 0; g < Hkv; g++) {
        if (!ensure(kbuf[g], (size_t)kvcap * D * 2)) return GGML_STATUS_FAILED;   // native K or legacy K rows
        if (!nat && !ensure(vbuf[g], (size_t)kvcap * D * 2)) return GGML_STATUS_FAILED;   // legacy-only
    }
    for (int t = 0; t < nthr; t++) {
        TB & b = tbs[t];
        if (!ensure(b.q, Mmax * D * 2) || !ensure(b.s, Mmax * kvcap * 4) || !ensure(b.p, Mmax * kvcap * 2) || !ensure(b.o, Mmax * D * 4)) return GGML_STATUS_FAILED;
        if ((kskip || nat) && (!ensure(b.ks, (size_t)kvcap * D * 2) || !ensure(b.vs, (size_t)kvcap * D * 2))) return GGML_STATUS_FAILED;
        b.inv.resize(Mmax); b.mx4.resize(Mmax); b.sm4.resize(Mmax);
    }
    // ---- 4. gather per-KV-head K/V from the (strided) F16 cache: native K layout, or legacy plain rows ----
    if (nat) {
        #pragma omp parallel for schedule(static)
        for (int g = 0; g < Hkv; g++) knat_fill(k, g, nkv, D, (__fp16 *)kbuf[g].m->virt_addr);
        for (int g = 0; g < Hkv; g++) if (rknn_mem_sync(anyctx->ctx, kbuf[g].m, RKNN_MEMORY_SYNC_TO_DEVICE) < 0) return GGML_STATUS_FAILED;
    } else {
        #pragma omp parallel for schedule(static)
        for (int x = 0; x < 2 * Hkv; x++) {
            const int g = x >> 1; const ggml_tensor * src = (x & 1) ? v : k;
            __fp16 * d = (__fp16 *)((x & 1) ? vbuf[g].m->virt_addr : kbuf[g].m->virt_addr);
            for (int j = 0; j < nkv; j++) memcpy(d + (size_t)j * D, (const char *)src->data + (size_t)j * src->nb[1] + (size_t)g * src->nb[2], (size_t)D * 2);
        }
        for (int g = 0; g < Hkv; g++) {
            if (rknn_mem_sync(anyctx->ctx, kbuf[g].m, RKNN_MEMORY_SYNC_TO_DEVICE) < 0 || rknn_mem_sync(anyctx->ctx, vbuf[g].m, RKNN_MEMORY_SYNC_TO_DEVICE) < 0) return GGML_STATUS_FAILED;
        }
    }
    const double T2 = rknpu_prof::now_ms();

    // ---- 5. driver threads ----
    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    std::vector<std::array<double, F_N>> tacc(nthr);
    auto worker = [&](int tid) {
        TB & b = tbs[tid];
        std::array<double, F_N> & ta = tacc[tid]; ta.fill(0.0);
        const float32x4_t ninf = vdupq_n_f32(-INFINITY), zero = vdupq_n_f32(0.f), vs = vdupq_n_f32(scale);
        const int32x4_t lane = {0, 1, 2, 3};
        for (;;) {
            const int ji = next.fetch_add(1);
            if (ji >= (int)jobs.size() || failed.load(std::memory_order_relaxed)) break;
            const Job & jb = jobs[ji];
            const int M = ratio * jb.mtp, Nq = jb.Nq;
            Ctx * cq = get_ctx(M, D, Nq, nat ? 1 : 2, tid);
            Ctx * cv = get_ctx(M, Nq, D, nat ? 1 : 0, tid, cq);
            if (!cq || !cv) { failed = true; break; }
            double t0 = prof ? rknpu_prof::now_ms() : 0;
            // Q -> native A (D/8, M, 8) fp16; padded rows zero
            __fp16 * qa = (__fp16 *)b.q.m->virt_addr;
            for (int hh = 0; hh < ratio; hh++) for (int i = 0; i < jb.mtp; i++) {
                const int r = hh * jb.mtp + i;
                if (i >= jb.mtc) { for (int d = 0; d < D; d += 8) vst1q_f16(qa + ((size_t)(d / 8) * M + r) * 8, vdupq_n_f16(0)); continue; }
                const float * src = (const float *)((const char *)q->data + (size_t)(jb.r0 + i) * q->nb[1] + (size_t)(jb.g * ratio + hh) * q->nb[2]);
                for (int d = 0; d < D; d += 8)
                    vst1q_f16(qa + ((size_t)(d / 8) * M + r) * 8, vcombine_f16(vcvt_f16_f32(vld1q_f32(src + d)), vcvt_f16_f32(vld1q_f32(src + d + 4))));
            }
            if (rknn_mem_sync(cq->ctx, b.q.m, RKNN_MEMORY_SYNC_TO_DEVICE) < 0) { failed = true; break; }
            double t1 = prof ? rknpu_prof::now_ms() : 0;
            // K/V from key k0: jobs with k0 > 0 copy their window into this thread's staging buffers. (Binding
            // create_mem_from_fd views of the shared K/V buffer at an offset gives wrong results when the two driver threads
            // of one NPU core use different offsets of the same fd concurrently.)
            const int k0 = jb.k0;   // S/P column j <-> key k0 + j
            rknn_tensor_mem * kb = kbuf[jb.g].m, * vb = vbuf[jb.g].m;
            if (nat) {
                if (k0 > 0) {   // K native layout is N-prefix compatible: stage whole (Nq/16) key blocks
                    const size_t blk = (size_t)(D / 32) * 512 * 2;
                    memcpy(b.ks.m->virt_addr, (const char *)kb->virt_addr + (size_t)(k0 / 16) * blk, (size_t)(Nq / 16) * blk);
                    if (rknn_mem_sync(cq->ctx, b.ks.m, RKNN_MEMORY_SYNC_TO_DEVICE) < 0) { failed = true; break; }
                    kb = b.ks.m;
                }
                // PV's native layout depends on Nq, so every PV job interleaves its own V window
                vnat_fill(v, jb.g, k0, Nq, D, (__fp16 *)b.vs.m->virt_addr);
                if (rknn_mem_sync(cv->ctx, b.vs.m, RKNN_MEMORY_SYNC_TO_DEVICE) < 0) { failed = true; break; }
                vb = b.vs.m;
            } else if (k0 > 0) {
                memcpy(b.ks.m->virt_addr, (const char *)kb->virt_addr + (size_t)k0 * D * 2, (size_t)Nq * D * 2);
                memcpy(b.vs.m->virt_addr, (const char *)vb->virt_addr + (size_t)k0 * D * 2, (size_t)Nq * D * 2);
                if (rknn_mem_sync(cq->ctx, b.ks.m, RKNN_MEMORY_SYNC_TO_DEVICE) < 0 || rknn_mem_sync(cq->ctx, b.vs.m, RKNN_MEMORY_SYNC_TO_DEVICE) < 0) { failed = true; break; }
                kb = b.ks.m; vb = b.vs.m;
            }
            if (!bind(cq, b.q.m, kb, b.s.m, nat) || !bind(cv, b.p.m, vb, b.o.m, nat)) { failed = true; break; }
            double t2 = prof ? rknpu_prof::now_ms() : 0;
            if (rknn_matmul_run(cq->ctx) < 0) { failed = true; break; }
            double t3 = prof ? rknpu_prof::now_ms() : 0;
            if (rknn_mem_sync(cq->ctx, b.s.m, RKNN_MEMORY_SYNC_FROM_DEVICE) < 0) { failed = true; break; }
            double t3a = prof ? rknpu_prof::now_ms() : 0;
            // ---- softmax over native S (Nq/4, M, 4) -> native P (Nq/8, M, 8) ----
            // (Range-restricted row iteration was tried here - bit-identical, fewer pairs - but measured neutral on
            // mixed batches and ~3% slower end to end on long inputs: with two submitter threads per NPU core the CPU
            // softmax overlaps the other thread's matmul, so the pipeline is NPU-bound and CPU savings turn into sync
            // wait. Kept the original full scan; see the split profiler columns below for the breakdown.)
            const float * S = (const float *)b.s.m->virt_addr; __fp16 * P = (__fp16 *)b.p.m->virt_addr;
            auto row_tok = [&](int r, int & t) -> bool { const int i = r % jb.mtp; t = jb.r0 + i; return i < jb.mtc; };
            for (int n4 = 0; n4 < Nq / 4; n4++) {
                const int j0 = n4 * 4;
                const float * Sb = S + (size_t)n4 * M * 4;
                for (int r = 0; r < M; r++) {
                    int t; if (!row_tok(r, t) || !row_clean[t]) continue;
                    const int lo = row_lo[t] - k0, hi = row_hi[t] - k0;
                    if (j0 > hi || j0 + 3 < lo) continue;
                    float32x4_t x = vld1q_f32(Sb + (size_t)r * 4);
                    if (j0 < lo || j0 + 3 > hi) {
                        const int32x4_t jj = vaddq_s32(lane, vdupq_n_s32(j0));
                        x = vbslq_f32(vandq_u32(vcgeq_s32(jj, vdupq_n_s32(lo)), vcleq_s32(jj, vdupq_n_s32(hi))), x, ninf);
                    }
                    b.mx4[r] = vmaxq_f32(b.mx4[r], x);
                }
            }
            for (int r = 0; r < M; r++) { const float m = vmaxvq_f32(b.mx4[r]); b.mx4[r] = vdupq_n_f32(std::isinf(m) ? 0.f : -m * scale); b.sm4[r] = zero; }
            double t3b = prof ? rknpu_prof::now_ms() : 0;
            for (int n8 = 0; n8 < Nq / 8; n8++) {
                const int j0 = n8 * 8;
                const float * Sb0 = S + (size_t)(2 * n8) * M * 4, * Sb1 = S + (size_t)(2 * n8 + 1) * M * 4;
                __fp16 * Pb = P + (size_t)n8 * M * 8;
                for (int r = 0; r < M; r++) {
                    int t; const bool valid = row_tok(r, t);
                    if (!valid || !row_clean[t] || j0 > row_hi[t] - k0 || j0 + 7 < row_lo[t] - k0) { vst1q_f16(Pb + (size_t)r * 8, vdupq_n_f16(0)); continue; }
                    const int lo = row_lo[t] - k0, hi = row_hi[t] - k0;
                    float32x4_t e0 = v_expf(vfmaq_f32(b.mx4[r], vld1q_f32(Sb0 + (size_t)r * 4), vs));
                    float32x4_t e1 = v_expf(vfmaq_f32(b.mx4[r], vld1q_f32(Sb1 + (size_t)r * 4), vs));
                    if (j0 < lo || j0 + 7 > hi) {
                        const int32x4_t j0v = vaddq_s32(lane, vdupq_n_s32(j0)), j1v = vaddq_s32(lane, vdupq_n_s32(j0 + 4));
                        const int32x4_t lov = vdupq_n_s32(lo), hiv = vdupq_n_s32(hi);
                        e0 = vbslq_f32(vandq_u32(vcgeq_s32(j0v, lov), vcleq_s32(j0v, hiv)), e0, zero);
                        e1 = vbslq_f32(vandq_u32(vcgeq_s32(j1v, lov), vcleq_s32(j1v, hiv)), e1, zero);
                    }
                    b.sm4[r] = vaddq_f32(b.sm4[r], vaddq_f32(e0, e1));
                    vst1q_f16(Pb + (size_t)r * 8, vcombine_f16(vcvt_f16_f32(e0), vcvt_f16_f32(e1)));
                }
            }
            for (int r = 0; r < M; r++) { const float s = vaddvq_f32(b.sm4[r]); b.inv[r] = s > 0.f ? 1.0f / s : 0.f; }
            double t3c = prof ? rknpu_prof::now_ms() : 0;
            // exact generic path for rows with non-trivial mask values inside their range
            for (int r = 0; r < M; r++) {
                int t; if (!row_tok(r, t) || row_clean[t]) continue;
                const uint16_t * mr = (const uint16_t *)((const char *)mask->data + (size_t)t * mask->nb[1]);
                const __fp16 * mh = (const __fp16 *)mr;
                auto logit = [&](int j) -> float {
                    const int kj = k0 + j;
                    if (kj >= nkv || mr[kj] == 0xFC00) return -INFINITY;
                    return S[((size_t)(j / 4) * M + r) * 4 + (j & 3)] * scale + (float)mh[kj];
                };
                float mx = -INFINITY;
                for (int j = 0; j < Nq; j++) mx = std::max(mx, logit(j));
                double sum = 0;
                for (int j = 0; j < Nq; j++) {
                    const float l = logit(j);
                    const float e = std::isinf(mx) || std::isinf(l) ? 0.f : expf(l - mx);
                    sum += e; P[((size_t)(j / 8) * M + r) * 8 + (j & 7)] = (__fp16)e;
                }
                b.inv[r] = sum > 0 ? (float)(1.0 / sum) : 0.f;
            }
            double t3d = prof ? rknpu_prof::now_ms() : 0;
            if (rknn_mem_sync(cv->ctx, b.p.m, RKNN_MEMORY_SYNC_TO_DEVICE) < 0) { failed = true; break; }
            double t4 = prof ? rknpu_prof::now_ms() : 0;
            if (rknn_matmul_run(cv->ctx) < 0) { failed = true; break; }
            double t5 = prof ? rknpu_prof::now_ms() : 0;
            if (rknn_mem_sync(cv->ctx, b.o.m, RKNN_MEMORY_SYNC_FROM_DEVICE) < 0) { failed = true; break; }
            // native O (D/4, M, 4) -> dst[token][head][d] * 1/rowsum
            const float * O = (const float *)b.o.m->virt_addr;
            for (int d4 = 0; d4 < D / 4; d4++) for (int r = 0; r < M; r++) {
                int t; if (!row_tok(r, t)) continue;
                const int h = jb.g * ratio + r / jb.mtp;
                float * o = (float *)((char *)dst->data + (size_t)t * dst->nb[2] + (size_t)h * dst->nb[1]) + d4 * 4;
                vst1q_f32(o, vmulq_f32(vld1q_f32(O + ((size_t)d4 * M + r) * 4), vdupq_n_f32(b.inv[r])));
            }
            if (prof) {
                double t6 = rknpu_prof::now_ms();
                ta[F_QFILL] += t1 - t0; ta[F_BIND] += t2 - t1; ta[F_RUNQK] += t3 - t2; ta[F_SSYNC] += t3a - t3; ta[F_SMAX] += t3b - t3a;
                ta[F_SEXP] += t3c - t3b; ta[F_GEN] += t3d - t3c; ta[F_PSYNC] += t4 - t3d; ta[F_RUNPV] += t5 - t4; ta[F_OUT] += t6 - t5;
            }
        }
    };
    {
        std::vector<std::thread> th;
        th.reserve(nthr);
        for (int t = 0; t < nthr; t++) th.emplace_back(worker, t);
        for (auto & t : th) t.join();
    }
    if (failed) { fprintf(stderr, "RKNPU FA: NPU attention failed\n"); return GGML_STATUS_FAILED; }
    static const int fa_check = rknpu_fa_env("RKNPU_FA_CHECK", 0);
    if (fa_check) {   // debug: fp32 reference for sampled rows
        double dot = 0, na = 0, nb = 0, maxe = 0; int worst_t = -1, worst_h = -1;
        std::vector<float> lg(nkv), ref(D);
        for (int t = 0; t < n; t += fa_check) for (int h = 0; h < H; h++) {
            const int g = h / ratio;
            const float * qr = (const float *)((const char *)q->data + (size_t)t * q->nb[1] + (size_t)h * q->nb[2]);
            float mx = -INFINITY;
            for (int j = 0; j < nkv; j++) {
                const __fp16 * kr = (const __fp16 *)((const char *)k->data + (size_t)j * k->nb[1] + (size_t)g * k->nb[2]);
                float d = 0; for (int x = 0; x < D; x++) d += qr[x] * (float)kr[x];
                const float mv = mask ? (float)((const __fp16 *)((const char *)mask->data + (size_t)t * mask->nb[1]))[j] : 0.f;
                lg[j] = d * scale + mv; mx = std::max(mx, lg[j]);
            }
            double sum = 0; std::fill(ref.begin(), ref.end(), 0.f);
            for (int j = 0; j < nkv; j++) {
                const float e = std::isinf(lg[j]) ? 0.f : expf(lg[j] - mx); sum += e;
                if (e == 0.f) continue;
                const __fp16 * vr = (const __fp16 *)((const char *)v->data + (size_t)j * v->nb[1] + (size_t)g * v->nb[2]);
                for (int x = 0; x < D; x++) ref[x] += e * (float)vr[x];
            }
            const float * o = (const float *)((const char *)dst->data + (size_t)t * dst->nb[2] + (size_t)h * dst->nb[1]);
            double e1 = 0;
            for (int x = 0; x < D; x++) { const float r = sum > 0 ? ref[x] / sum : 0.f; dot += r * o[x]; na += r * r; nb += o[x] * o[x]; e1 = std::max(e1, (double)fabsf(r - o[x])); }
            if (e1 > maxe) { maxe = e1; worst_t = t; worst_h = h; }
        }
        fprintf(stderr, "RKNPU FA CHECK: n=%d nkv=%d H=%d Hkv=%d scale=%g cos=%.6f maxerr=%.4g (t=%d h=%d lo=%d hi=%d clean=%d) mask=%s ne=[%d,%d] nb1=%zu\n", n, nkv, H, Hkv, scale,
                dot / sqrt(na * nb + 1e-30), maxe, worst_t, worst_h, worst_t >= 0 ? row_lo[worst_t] : -1, worst_t >= 0 ? row_hi[worst_t] : -1, worst_t >= 0 ? row_clean[worst_t] : -1,
                mask ? "yes" : "no", mask ? (int)mask->ne[0] : 0, mask ? (int)mask->ne[1] : 0, mask ? mask->nb[1] : 0);
    }
    if (prof) {
        ft[F_MASK] += T1 - T0; ft[F_GATHER] += T2 - T1;
        for (auto & a : tacc) for (int i = F_QFILL; i < F_N; i++) ft[i] += a[i];
        f_n++; f_wall += rknpu_prof::now_ms() - T0;
    }
    return GGML_STATUS_SUCCESS;
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

    bool has_fa = false;
    for (int node_i = 0; node_i < cgraph->n_nodes && !has_fa; node_i++) has_fa = cgraph->nodes[node_i]->op == GGML_OP_FLASH_ATTN_EXT;
    if (rknpu_overlap() && !has_fa) return rknpu_graph_compute_overlap(backend_ctx, cgraph, config);

    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        struct ggml_tensor* node = cgraph->nodes[node_i];
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            const enum ggml_status st = rknpu_fa_compute(node);
            if (st != GGML_STATUS_SUCCESS) return st;
            continue;
        }
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
                auto its = src0_buf_ctx->hadamard_s_vectors.find(K_op);
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
                    const float* ready_row = src_row + k_seg.offset_k;

                    // Applying Hadamard Transform
                    if (is_hadamard) {
                        thread_local static std::vector<float> had_scratch;
                        if ((int)had_scratch.size() < K_op) had_scratch.resize(K_op);
                        rknpu2_calibration::hadamard_signed_fwht(had_scratch.data(), src_row, s_vec, K, K_op);
                        ready_row = had_scratch.data() + k_seg.offset_k;
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
                            rknpu_epi_f32(dst_ptr, src_ptr, N_segment, sa, wscale, !first_k);
                        } else if (npu_type_c == rknpu2_configuration::NPU_TYPE_INT32) {
                            const int32_t* src_ptr = (const int32_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;
                            rknpu_epi_i32(dst_ptr, src_ptr, N_segment, sa, wscale, !first_k);
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
        s_vec = ctx->hadamard_s_vectors[K_op];
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
            std::lock_guard<std::mutex> lock(ctx->mutex);
            // One sign vector per K_op, shared by every tensor with that padded K. The seed
            // depends only on K_op: deterministic across launches (an earlier version seeded
            // from the tensor's heap address, so embeddings changed on every restart), and
            // identical for q/k/gate/up so their rotated activations — and therefore the
            // A-quantization cache — can be shared.
            if (ctx->hadamard_s_vectors.find(K_op) == ctx->hadamard_s_vectors.end()) {
                std::vector<float> s_vec(K_op, 1.0f);
                uint64_t h = 1469598103934665603ULL;
                h ^= (uint64_t)K_op; h *= 1099511628211ULL;
                std::mt19937 gen((uint32_t)(h ^ (h >> 32)));
                std::uniform_int_distribution<int> distrib(0, 1);
                for (int k = 0; k < K_op; ++k) {
                    s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
                }
                ctx->hadamard_s_vectors.emplace(K_op, std::move(s_vec));
            }
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

        case GGML_OP_FLASH_ATTN_EXT:
            return rknpu_fa_supported(op);

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
        /* .is_host        = */ [](ggml_backend_buffer_type_t buft) { UNUSED(buft); return rknpu_host_compute(); },
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
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
            UNUSED(dev);
            return buft == &rknpu_buffer_type || (rknpu_host_compute() && ggml_backend_buft_is_host(buft));
        },
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