// usage: g++ -O2 -std=c++17 ctx_mem.cpp -I<rknn include dir> -L<dir with librknnrt.so> -lrknnrt -o ctx_mem && ./ctx_mem [max_kv=2048] [threads=6]
// memory cost of NPU matmul contexts (same shapes/settings as the ggml-rknpu2 FA contexts) + FA buffers.
#include "rknn_matmul_api.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
static long kv(const char* file, const char* key) {
    std::ifstream f(file); std::string l;
    while (std::getline(f, l)) if (l.rfind(key, 0) == 0) { std::istringstream is(l.substr(strlen(key))); long v; is >> v; return v; }
    return -1;
}
static void snap(const char* tag, int n) {
    printf("%-28s n=%4d VmRSS=%7ld RssAnon=%7ld RssFile=%6ld RssShmem=%6ld | MemAvailable=%8ld MemFree=%8ld CmaFree=%7ld Shmem=%7ld kB\n", tag, n,
        kv("/proc/self/status", "VmRSS:"), kv("/proc/self/status", "RssAnon:"), kv("/proc/self/status", "RssFile:"), kv("/proc/self/status", "RssShmem:"),
        kv("/proc/meminfo", "MemAvailable:"), kv("/proc/meminfo", "MemFree:"), kv("/proc/meminfo", "CmaFree:"), kv("/proc/meminfo", "Shmem:"));
    fflush(stdout);
}
int main(int argc, char** argv) {
    const int maxkv = argc > 1 ? atoi(argv[1]) : 2048, nthr = argc > 2 ? atoi(argv[2]) : 6, M = 512, D = 128;
    std::vector<rknn_matmul_ctx> v;
    snap("start", 0);
    for (int N = 256; N <= maxkv; N += 256) {
        for (int t = 0; t < nthr; t++) for (int lay = 0; lay < 2; lay++) {
            rknn_matmul_info info; rknn_matmul_io_attr io; memset(&info, 0, sizeof(info)); memset(&io, 0, sizeof(io));
            info.M = M; info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32; info.AC_layout = 1;
            if (lay == 0) { info.K = D; info.N = N; info.B_layout = 2; } else { info.K = N; info.N = D; info.B_layout = 0; }
            rknn_matmul_ctx c; if (rknn_matmul_create(&c, &info, &io) < 0) { printf("create failed N=%d\n", N); return 1; }
            rknn_matmul_set_core_mask(c, t % 3 == 0 ? RKNN_NPU_CORE_0 : t % 3 == 1 ? RKNN_NPU_CORE_1 : RKNN_NPU_CORE_2);
            v.push_back(c);
        }
        char tag[64]; snprintf(tag, sizeof tag, "ctx up to N=%d", N); snap(tag, (int)v.size());
    }
    // FA buffers as allocated by the backend at maxkv: per KV head K,V (kvcap*D*2 each), per thread q/s/p/o
    std::vector<rknn_tensor_mem*> mems;
    for (int g = 0; g < 8; g++) { mems.push_back(rknn_create_mem(v[0], (size_t)maxkv * D * 2)); mems.push_back(rknn_create_mem(v[0], (size_t)maxkv * D * 2)); }
    for (int t = 0; t < nthr; t++) { mems.push_back(rknn_create_mem(v[0], (size_t)M * D * 2)); mems.push_back(rknn_create_mem(v[0], (size_t)M * maxkv * 4));
                                     mems.push_back(rknn_create_mem(v[0], (size_t)M * maxkv * 2)); mems.push_back(rknn_create_mem(v[0], (size_t)M * D * 4)); }
    for (auto m : mems) if (m) memset(m->virt_addr, 0, m->size);
    snap("+ FA buffers (touched)", (int)v.size());
    for (auto m : mems) rknn_destroy_mem(v[0], m);
    snap("buffers freed", (int)v.size());
    for (size_t i = 1; i < v.size(); i++) rknn_matmul_destroy(v[i]);
    rknn_matmul_destroy(v[0]);
    snap("all destroyed", 0);
    return 0;
}
