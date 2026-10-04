
#include <arm_neon.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#define GGML_RESTRICT __restrict
#define GGML_ASSERT assert
#define MIN(a,b) ((a)<(b)?(a):(b))
#define CACHE_LINE_SIZE_F32 16
#define GGML_CPU_FP16_TO_FP32(x) ((float)(x))
#define GGML_TENSOR_LOCALS(T,P,X,F) const T P##0=X->F[0],P##1=X->F[1],P##2=X->F[2],P##3=X->F[3];
typedef __fp16 ggml_fp16_t;
enum ggml_type {GGML_TYPE_F32, GGML_TYPE_F16};
struct ggml_tensor { ggml_type type; int64_t ne[4]; size_t nb[4]; void *data; ggml_tensor *src[5]{}; float op_params[4]{}; };
struct ggml_compute_params {int ith,nth; void *wdata;size_t wsize; bool use_ref; int *threadpool;};
struct ggml_fa_tile_config {static constexpr int Q=64,KV=64;};
void ggml_barrier(int*) {}
void ggml_threadpool_chunk_set(int *p,int v) {*p=v;}
int ggml_threadpool_chunk_add(int *p,int v) {int old=*p;*p+=v;return old;}
void ggml_vec_scale_f32(int n,float *p,float s) {for(int i=0;i<n;i++)p[i]*=s;}
void ggml_vec_tanh_f32(int n,float *o,const float *p) {for(int i=0;i<n;i++)o[i]=tanhf(p[i]);}
void ggml_cpu_fp16_to_fp32(const __fp16 *p,float *o,int n) {for(int i=0;i<n;i++)o[i]=p[i];}
float32x4_t ggml_v_expf(float32x4_t x) {float a[4];vst1q_f32(a,x);for(float &v:a)v=expf(v);return vld1q_f32(a);}
