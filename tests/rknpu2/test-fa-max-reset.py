#!/usr/bin/env python3
# Standalone ARM64 numeric regression with CPU RKNN mocks.
from pathlib import Path
import ast
import subprocess
import sys
import os

repo=Path(__file__).resolve().parents[2]
import tempfile
_tmp=tempfile.TemporaryDirectory(prefix="rknpu2-regression-")
out=Path(_tmp.name)
prefix=(Path(__file__).resolve().parent/"fa_fixture.inc").read_text()
src = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / 'ggml/src/ggml-rknpu2/ggml-rknpu2.cpp'
source = src.read_text()
fa = source[source.index('static bool rknpu_fa_enabled()'):source.index('static enum ggml_status ggml_backend_rknpu_graph_compute')]
main = r'''
int main() {
    setenv("RKNPU_FA", "1", 1); setenv("RKNPU_FA_MT", "32", 1); setenv("RKNPU_FA_THREADS", "1", 1);
    const int D=32;
    int failures=0, cases=0;
    for (int mode=0; mode<4; mode++) for(int repeat=0;repeat<3;repeat++) {
        int n=mode==0?16:65, nkv=mode==0?32:96;
        int Hkv=mode==0?1:2, ratio=mode==0?1:2,H=Hkv*ratio;
        std::vector<float> qv(D*n*H,1),ov(D*n*H),ref(D*n*H);
        std::vector<__fp16> kv(D*nkv*Hkv,(__fp16)-5),vv(D*nkv*Hkv),mv(n*nkv,(__fp16)0);
        for(int g=0;g<Hkv;g++) for(int j=0;j<nkv;j++) for(int d=0;d<D;d++) vv[(g*nkv+j)*D+d]=(__fp16)(1+g+(j%3)*0.25f);
        if(mode==1) std::fill(qv.begin(),qv.end(),repeat%2?0.5f:1.f);
        if(mode==2 || mode==3) for(int t=0;t<n;t++) for(int j=0;j<nkv;j++) {
            if((mode==2 && t<32) || j<(t<32?0:32) || j>t+20) mv[t*nkv+j]=(__fp16)-INFINITY;
            else if(mode==3 && j%7==0) mv[t*nkv+j]=(__fp16)-0.5f;
        }
        ggml_tensor q{GGML_TYPE_F32,{D,n,H,1},{4,D*4,(size_t)D*n*4,(size_t)D*n*H*4},qv.data()};
        ggml_tensor k{GGML_TYPE_F16,{D,nkv,Hkv,1},{2,D*2,(size_t)D*(size_t)nkv*2,(size_t)D*nkv*Hkv*2},kv.data()};
        ggml_tensor v=k;v.data=vv.data();
        ggml_tensor mask{GGML_TYPE_F16,{nkv,n,1,1},{2,(size_t)nkv*2,(size_t)nkv*n*2,(size_t)nkv*n*2},mv.data()};
        ggml_tensor dst{GGML_TYPE_F32,{D,H,n,1},{4,D*4,(size_t)D*H*4,(size_t)D*n*H*4},ov.data()};
        dst.src[0]=&q;dst.src[1]=&k;dst.src[2]=&v;dst.src[3]=mode>=2?&mask:nullptr;dst.op_params[0]=0.125f;
        ggml_tensor expected=dst;expected.data=ref.data();
        rknpu_fa_reference(&expected,&q,&k,&v,dst.src[3],0.125f);
        // Exercise a reused worker buffer containing an unrelated previous offset.
        for(auto &tb:rkfa::tbs) for(auto &mx:tb.mx4) mx=vdupq_n_f32(100.f);
        bool supported=rknpu_fa_supported(&dst);
        int status=rknpu_fa_compute(&dst);
        float maxerr=0;bool finite=true;
        for(size_t i=0;i<ov.size();i++) {finite &= std::isfinite(ov[i]);maxerr=std::max(maxerr,fabsf(ov[i]-ref[i]));}
        bool ok=supported && status==GGML_STATUS_SUCCESS && finite && maxerr<0.005f;
        printf("mode=%d repeat=%d n=%d H=%d maxerr=%g %s\n",mode,repeat,n,H,maxerr,ok?"PASS":"FAIL");
        cases++;failures+=!ok;
    }
    printf("cases=%d failures=%d\n",cases,failures);
    return failures?1:0;
}
'''
cpp = out / 'regression.cpp'
cpp.write_text(prefix+fa+main)
include = repo / 'ggml/src/ggml-rknpu2/libs/include'
subprocess.run([os.getenv('CXX','c++'),'-std=c++17','-O2','-Wno-unknown-pragmas','-I',str(include),str(cpp),'-o',str(out/'regression')],check=True)
result = subprocess.run([str(out/'regression')],text=True,capture_output=True)
print(result.stdout,end='')
print(result.stderr,end='',file=sys.stderr)
(out / ('baseline-results.txt' if len(sys.argv)>1 else 'fixed-results.txt')).write_text(result.stdout+result.stderr)
sys.exit(result.returncode)
