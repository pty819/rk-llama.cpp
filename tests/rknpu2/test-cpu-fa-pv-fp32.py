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
prefix=(Path(__file__).resolve().parent/"cpu_fa_fixture.inc").read_text()
source = (Path(sys.argv[1]) if len(sys.argv)>1 else repo/'ggml/src/ggml-cpu/ops.cpp').read_text()
start = source.index('// ---- opt1 (2026-09-25): NEON tiled flash attention')
kernel = source[start:source.index('// ---- end opt1 ----',start)]
main = r'''
int main() {
    setenv("GGML_FA_OPT1","2",1);
    int cases=0, failures=0;
    for(int magnitude: {0,1}) for(int K: {1,17,63,64}) for(int M: {8,16}) for(int N: {16,32}) {
        int ldat=M+8,ldb=N+16,ldc=N+8;
        std::vector<__fp16> a(K*ldat),b(K*ldb);
        std::vector<float> c(M*ldc,0.5f);
        for(int k=0;k<K;k++) {
            for(int i=0;i<M;i++) a[k*ldat+i]=(__fp16)((i%3+1)*0.25f);
            for(int j=0;j<N;j++) b[k*ldb+j]=(__fp16)((k%3==0?-1.f:1.f)*(magnitude?(j%2?65504.f:1024.f):(j%2?0.75f:0.125f)));
        }
        fa1h_gemm_AT(c.data(),ldc,a.data(),ldat,b.data(),ldb,M,N,K);
        bool ok=true;double maxerr=0;
        for(int i=0;i<M;i++) for(int j=0;j<ldc;j++) {
            double expected=0.5;
            if(j<N) for(int k=0;k<K;k++) expected+=(double)a[k*ldat+i]*(double)b[k*ldb+j];
            double err=fabs(c[i*ldc+j]-expected);maxerr=std::max(maxerr,err);
            ok &= std::isfinite(c[i*ldc+j]) && err<=1e-5*std::max(1.0,fabs(expected));
        }
        printf("PV large=%d K=%d M=%d N=%d err=%g %s\n",magnitude,K,M,N,maxerr,ok?"PASS":"FAIL");
        cases++;failures+=!ok;
    }
    for(int nkv: {64,65,128}) for(float value: {1024.f,65504.f,-65504.f}) {
        const int D=32,n=65;
        std::vector<float> qv(D*n,0),ov(D*n,-9),scratch(50000);
        std::vector<__fp16> kv(D*nkv,(__fp16)0),vv(D*nkv,(__fp16)value);
        ggml_tensor q{GGML_TYPE_F32,{D,n,1,1},{4,D*4,D*n*4,D*n*4},qv.data()};
        ggml_tensor k{GGML_TYPE_F16,{D,nkv,1,1},{2,D*2,(size_t)D*nkv*2,(size_t)D*nkv*2},kv.data()};
        ggml_tensor v=k;v.data=vv.data();
        ggml_tensor dst{GGML_TYPE_F32,{D,1,n,1},{4,D*4,D*4,D*n*4},ov.data()};
        dst.src[0]=&q;dst.src[1]=&k;dst.src[2]=&v;dst.op_params[0]=1;
        int counter=0;ggml_compute_params p{0,1,scratch.data(),scratch.size()*4,false,&counter};
        bool ok=ggml_fa_opt1_try_compute(&p,&dst);double maxerr=0;
        for(float x:ov) {double err=fabs(x-value);maxerr=std::max(maxerr,err);ok &= std::isfinite(x) && err<=1e-5*fabs(value);}
        printf("FA nkv=%d V=%g err=%g %s\n",nkv,value,maxerr,ok?"PASS":"FAIL");
        cases++;failures+=!ok;
    }
    printf("cases=%d failures=%d\n",cases,failures);
    return failures?1:0;
}
'''
cpp=out/'regression.cpp'
cpp.write_text(prefix+kernel+main)
subprocess.run([os.getenv('CXX','c++'),'-std=c++17','-O2','-march=armv8.2-a+fp16','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(cpp),'-o',str(out/'regression')],check=True)
r=subprocess.run([str(out/'regression')],capture_output=True,text=True)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
(out/('baseline-results.txt' if len(sys.argv)>1 else 'fixed-results.txt')).write_text(r.stdout+r.stderr)
sys.exit(r.returncode)
