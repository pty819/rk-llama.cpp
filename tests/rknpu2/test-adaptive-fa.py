#!/usr/bin/env python3
"""Run the production FA code with ARM64 RKNN mocks and layout checks."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
source = Path(sys.argv[1]).read_text()
include = Path(sys.argv[2])
prefix = (here / 'fa_fixture.inc').read_text()
prefix = prefix.replace('struct Fake {', '''
static std::mutex test_mu;
static std::atomic<bool> test_running{false};
static int test_late_create = 0;
static long long test_pairs = 0;
static std::map<rknn_tensor_mem*, int> test_native_rows;
static int test_native_aliases = 0;
static int test_short_b_oversized = 0;
static int test_fail_sync = -1;
static bool test_fail_alloc = false;
struct Fake {''')
prefix = prefix.replace('auto *f = new Fake;', '''
    if (test_running) test_late_create++;
    auto *f = new Fake;''')
prefix = prefix.replace('auto *f=(Fake*)ctx; int M=', '''
    test_running = true;
    auto *f=(Fake*)ctx; int M=''')
prefix = prefix.replace('auto *a=(__fp16*)f->a->virt_addr;', '''
    if (f->info.B_layout == 2) {
        std::lock_guard<std::mutex> lk(test_mu);
        test_pairs += (long long)M * N;
    }
    auto *a=(__fp16*)f->a->virt_addr;''')
prefix = prefix.replace('auto *f=(Fake*)ctx; if(attr->name', '''
    auto *f=(Fake*)ctx;
    if(attr->name[0]=='A' || attr->name[0]=='C') {
        std::lock_guard<std::mutex> lk(test_mu);
        auto it=test_native_rows.emplace(m,f->info.M);
        if(!it.second && it.first->second!=f->info.M) test_native_aliases++;
    }
    if(attr->name[0]=='B' && ((f->info.B_layout==2 && f->info.N<256) || (f->info.B_layout==0 && f->info.K<256))) {
        std::lock_guard<std::mutex> lk(test_mu);
        if(m->size!=(uint32_t)(f->info.K*f->info.N*2)) test_short_b_oversized++;
    }
    if(attr->name''')
prefix = prefix.replace('free(m->virt_addr); delete m;', '''
    {std::lock_guard<std::mutex> lk(test_mu);test_native_rows.erase(m);}
    free(m->virt_addr); delete m;''')
prefix = prefix.replace('auto *m = new rknn_tensor_mem{};', 'if(test_fail_alloc) return nullptr; auto *m = new rknn_tensor_mem{};')
prefix = prefix.replace('{return 0;}\nint rknn_matmul_set_io_mem', '{if(test_fail_sync==0) {test_fail_sync=-1; return -1;} if(test_fail_sync>0) test_fail_sync--; return 0;}\nint rknn_matmul_set_io_mem')
fa = source[source.index('static bool rknpu_fa_enabled()'):source.index('static enum ggml_status ggml_backend_rknpu_graph_compute')]
main = r'''
int main(int argc, char ** argv) {
    const bool adaptive = argc < 2 || strcmp(argv[1], "0") != 0;
    setenv("RKNPU_FA", "1", 1);
    setenv("RKNPU_FA_CPU_FALLBACK", "0", 1);
    setenv("RKNPU_FA_THREADS", "3", 1);
    setenv("RKNPU_FA_MT", "256", 1);
    setenv("RKNPU_FA_MAX_CTX", "6", 1);
    if(argc > 1 && strcmp(argv[1], "unset") == 0) unsetenv("RKNPU_FA_ADAPTIVE");
    else setenv("RKNPU_FA_ADAPTIVE", adaptive ? "1" : "0", 1);
    const int D = 32, Hkv = 2, ratio = 2, H = Hkv * ratio;
    int failures = 0;
    const int lengths[] = {16, 32, 53, 64, 65, 127, 128, 129, 255, 256, 257};
    for (int mode = 0; mode < 4; mode++) for (int n0 : lengths) {
        if (mode >= 2 && n0 != 53 && n0 != 64 && n0 != 129 && n0 != 257) continue;
        const int n = mode == 2 ? 3*n0 : n0;
        const int nkv = mode == 3 ? n + 41 : n;
        const int qs = D * H + 8, ks = D * Hkv + 8, ms = nkv + 8;
        std::vector<float> qv(n * qs), ov(n * D * H), ref(ov.size());
        std::vector<__fp16> kv(nkv * ks), vv(kv.size()), mv(n * ms);
        for (int t = 0; t < n; t++) for (int h = 0; h < H; h++) for (int d = 0; d < D; d++)
            qv[t * qs + h * D + d] = sinf((t + 3*h + d) * .13f) * .7f;
        for (int j = 0; j < nkv; j++) for (int g = 0; g < Hkv; g++) for (int d = 0; d < D; d++) {
            kv[j * ks + g * D + d] = (__fp16)cosf((2*j + g + d) * .17f);
            vv[j * ks + g * D + d] = (__fp16)(sinf((j + 5*g + d) * .23f) * 2.f);
        }
        for (int t = 0; t < n; t++) for (int j = 0; j < nkv; j++) {
            bool visible = mode == 0 || (mode == 1 && j <= t);
            if (mode == 2) visible = j / n0 == t / n0 && j <= t;
            if (mode == 3) visible = t % 9 != 0 && j >= 17 && j <= t + 24 && j % 11 != 0;
            mv[t * ms + j] = visible ? (__fp16)(mode == 3 && j % 7 == 0 ? -.5f : 0.f) : (__fp16)-INFINITY;
        }
        ggml_tensor q{GGML_TYPE_F32,{D,n,H,1},{4,(size_t)qs*4,D*4,(size_t)n*qs*4},qv.data()};
        ggml_tensor k{GGML_TYPE_F16,{D,nkv,Hkv,1},{2,(size_t)ks*2,D*2,(size_t)nkv*ks*2},kv.data()};
        ggml_tensor v = k; v.data = vv.data();
        ggml_tensor mask{GGML_TYPE_F16,{nkv,n,1,1},{2,(size_t)ms*2,(size_t)n*ms*2,(size_t)n*ms*2},mv.data()};
        ggml_tensor dst{GGML_TYPE_F32,{D,H,n,1},{4,D*4,(size_t)D*H*4,(size_t)D*H*n*4},ov.data()};
        dst.src[0]=&q; dst.src[1]=&k; dst.src[2]=&v; dst.src[3]=mode ? &mask : nullptr; dst.op_params[0]=.125f;
        ggml_tensor expected=dst; expected.data=ref.data();
        rknpu_fa_reference(&expected,&q,&k,&v,dst.src[3],.125f);
        test_running=false; test_late_create=0; test_pairs=0; test_native_aliases=0; test_short_b_oversized=0;
        int status=rknpu_fa_compute(&dst);
        float err=0; bool finite=true;
        for(size_t i=0;i<ov.size();i++) {finite &= std::isfinite(ov[i]); err=std::max(err,fabsf(ov[i]-ref[i]));}
        bool ok=status==GGML_STATUS_SUCCESS && finite && err<.004f && test_late_create==0 && test_native_aliases==0;
        if(adaptive) ok &= test_short_b_oversized==0;
        if(adaptive && mode==0 && n==53) ok &= test_pairs == (long long)H * 64 * 64;
        if(adaptive && mode==2) {
            long long fixed_pairs=0;
            for(int r=0;r<n;r+=256) {
                const int k0=(r/n0*n0)/256*256;
                const int k1=(std::min(r+256,n)+255)/256*256;
                fixed_pairs += (long long)H*256*(k1-k0);
            }
            ok &= test_pairs <= fixed_pairs;
            if(n0==64) ok &= test_pairs == (long long)H * 3 * 64 * 64;
        }
        printf("adaptive=%d mode=%d n=%d nkv=%d pairs=%lld error=%g late_create=%d native_aliases=%d short_b_oversized=%d %s\n",adaptive,mode,n,nkv,test_pairs,err,test_late_create,test_native_aliases,test_short_b_oversized,ok?"PASS":"FAIL");
        failures += !ok;
        test_running=false;
    }
    rkfa::Mem probe;
    bool ownership = rkfa::ensure(probe, 128);
    auto * old = probe.m;
    test_fail_sync=0;
    ownership &= !rkfa::ensure(probe,256) && probe.m==old && probe.size==128 && probe.retired.empty();
    test_fail_alloc=true;
    ownership &= !rkfa::ensure(probe,256) && probe.m==old && probe.retired.empty();
    test_fail_alloc=false;
    ownership &= rkfa::ensure(probe,256) && rkfa::ensure(probe,512) && rkfa::ensure(probe,1024);
    size_t retired=0; for(auto * mem:probe.retired) retired+=mem->size;
    ownership &= retired<probe.size && probe.retired.size()==3;
    printf("buffer growth ownership %s retired=%zu active=%zu\n",ownership?"PASS":"FAIL",retired,probe.size);
    failures += !ownership;
    printf("failures=%d\n",failures);
    return failures ? 1 : 0;
}
'''
with tempfile.TemporaryDirectory(prefix='rknpu-fa-adaptive-') as tmp:
    cpp = Path(tmp) / 'test.cpp'
    cpp.write_text(prefix + fa + main)
    exe = Path(tmp) / 'test'
    subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O2', '-pthread', '-Wno-unknown-pragmas', '-I', str(include), str(cpp), '-o', str(exe)], check=True)
    for adaptive in ('1', '0', 'unset'):
        subprocess.run([str(exe), adaptive], check=True)
