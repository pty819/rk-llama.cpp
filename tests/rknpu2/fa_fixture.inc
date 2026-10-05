
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <cassert>
#include <condition_variable>
#include <exception>
#include <thread>
#include <pthread.h>
#include <vector>
#include "rknn_matmul_api.h"
enum { GGML_TYPE_F32, GGML_TYPE_F16 };
enum ggml_status { GGML_STATUS_SUCCESS, GGML_STATUS_FAILED };
struct ggml_tensor { int type; int64_t ne[4]; size_t nb[4]; void *data; ggml_tensor *src[5]{}; float op_params[4]{}; };
namespace rknpu_prof { bool enabled() {return false;} double now_ms() {return 0;} void (*extra_dump)() = nullptr; }
struct Fake { rknn_matmul_info info; rknn_tensor_mem *a{}, *b{}, *c{}; };
int rknn_matmul_create(rknn_matmul_ctx *ctx, rknn_matmul_info *info, rknn_matmul_io_attr *io) {
    auto *f = new Fake; f->info = *info; *ctx = (rknn_matmul_ctx)f;
    memset(io, 0, sizeof(*io));
    strcpy(io->A.name, "A"); strcpy(io->B.name, "B"); strcpy(io->C.name, "C");
    io->A.n_dims = io->C.n_dims = 1; io->A.dims[0] = 8; io->C.dims[0] = 4;
    io->A.size = info->M * info->K * 2; io->B.size = info->K * info->N * 2; io->C.size = info->M * info->N * 4;
    return 0;
}
int rknn_matmul_destroy(rknn_matmul_ctx ctx) { delete (Fake*)ctx; return 0; }
int rknn_matmul_set_core_mask(rknn_matmul_ctx, rknn_core_mask) { return 0; }
rknn_tensor_mem *rknn_create_mem(rknn_context, uint32_t size) { auto *m = new rknn_tensor_mem{}; m->virt_addr = calloc(1,size); m->size=size; return m; }
int rknn_destroy_mem(rknn_context, rknn_tensor_mem *m) { free(m->virt_addr); delete m; return 0; }
int rknn_mem_sync(rknn_context, rknn_tensor_mem*, rknn_mem_sync_mode) {return 0;}
int rknn_matmul_set_io_mem(rknn_matmul_ctx ctx, rknn_tensor_mem *m, rknn_matmul_tensor_attr *attr) {
    auto *f=(Fake*)ctx; if(attr->name[0]=='A') f->a=m; if(attr->name[0]=='B') f->b=m; if(attr->name[0]=='C') f->c=m; return 0;
}
int rknn_matmul_run(rknn_matmul_ctx ctx) {
    auto *f=(Fake*)ctx; int M=f->info.M,K=f->info.K,N=f->info.N;
    auto *a=(__fp16*)f->a->virt_addr; auto *b=(__fp16*)f->b->virt_addr; auto *c=(float*)f->c->virt_addr;
    for(int m=0;m<M;m++) for(int n=0;n<N;n++) {
        float s=0; for(int k=0;k<K;k++) s+=(float)a[((k/8)*M+m)*8+k%8]*(float)b[f->info.B_layout==2?n*K+k:k*N+n];
        c[((n/4)*M+m)*4+n%4]=s;
    } return 0;
}
