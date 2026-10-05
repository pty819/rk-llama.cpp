#!/usr/bin/env python3
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
source = Path(sys.argv[1]).read_text()
prefix = (here / 'fa_fixture.inc').read_text().split('struct Fake {')[0]
fa = source[source.index('static bool rknpu_fa_enabled()'):source.index('static enum ggml_status ggml_backend_rknpu_graph_compute')]
main = r'''
int main(int argc, char ** argv) {
    setenv("RKNPU_FA", "1", 1); setenv("RKNPU_FA_CPU_FALLBACK", "0", 1);
    setenv("RKNPU_FA_MAX_KV", "2048", 1);
    setenv("RKNPU_FA_ADAPTIVE", argc > 1 ? argv[1] : "1", 1);
    const int D=128, Hkv=8, ratio=2, H=Hkv*ratio;
    const int lengths[] = {53,64,65,127,128,129,255,256,257,547,1023,2048};
    for(int mode=0;mode<3;mode++) for(int n0:lengths) {
        if(argc>2 && (mode!=(argc>3?atoi(argv[3]):0) || n0!=atoi(argv[2])))continue;
        if(mode==1 && n0!=64 && n0!=129) continue;
        if(mode==2 && n0!=53 && n0!=257) continue;
        const int n=mode==1 ? n0*3 : n0, nkv=n;
        std::vector<float> qv(n*D*H),ov(qv.size());
        std::vector<__fp16> kv(nkv*D*Hkv),vv(kv.size()),mv(n*nkv);
        for(int t=0;t<n;t++) for(int h=0;h<H;h++) for(int d=0;d<D;d++)
            qv[(t*H+h)*D+d]=sinf((t+3*h+d)*.13f)*.7f;
        for(int j=0;j<nkv;j++) for(int g=0;g<Hkv;g++) for(int d=0;d<D;d++) {
            kv[(j*Hkv+g)*D+d]=(__fp16)cosf((2*j+g+d)*.17f);
            vv[(j*Hkv+g)*D+d]=(__fp16)(sinf((j+5*g+d)*.23f)*2.f);
        }
        for(int t=0;t<n;t++) for(int j=0;j<nkv;j++) {
            bool visible=j<=t;
            if(mode==1)visible &= j/n0==t/n0;
            if(mode==2)visible &= t%9!=0 && j%11!=0;
            mv[t*nkv+j]=visible ? (__fp16)(mode==2&&j%7==0?-.5f:0.f) : (__fp16)-INFINITY;
        }
        ggml_tensor q{GGML_TYPE_F32,{D,n,H,1},{4,D*H*4,D*4,(size_t)n*D*H*4},qv.data()};
        ggml_tensor k{GGML_TYPE_F16,{D,nkv,Hkv,1},{2,D*Hkv*2,D*2,(size_t)nkv*D*Hkv*2},kv.data()};
        ggml_tensor v=k;v.data=vv.data();
        ggml_tensor mask{GGML_TYPE_F16,{nkv,n,1,1},{2,(size_t)nkv*2,(size_t)n*nkv*2,(size_t)n*nkv*2},mv.data()};
        ggml_tensor dst{GGML_TYPE_F32,{D,H,n,1},{4,D*4,D*H*4,(size_t)n*D*H*4},ov.data()};
        dst.src[0]=&q;dst.src[1]=&k;dst.src[2]=&v;dst.src[3]=&mask;dst.op_params[0]=.125f;
        std::vector<double> times;
        for(int it=0;it<7;it++) {
            auto start=std::chrono::steady_clock::now();
            if(rknpu_fa_compute(&dst)!=GGML_STATUS_SUCCESS) return 2;
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(it>=2)times.push_back(ms);
        }
        float maxerr=0; bool finite=true;
        for(float x:ov)finite &= std::isfinite(x);
        for(int t : {0,n/2,n-1}) {
            ggml_tensor qr=q;qr.ne[1]=1;qr.data=qv.data()+(size_t)t*D*H;
            ggml_tensor mr=mask;mr.ne[1]=1;mr.data=mv.data()+(size_t)t*nkv;
            std::vector<float> ref(D*H);
            ggml_tensor dr=dst;dr.ne[2]=1;dr.data=ref.data();
            rknpu_fa_reference(&dr,&qr,&k,&v,&mr,.125f);
            float rowerr=0; int worst=0;
            for(int i=0;i<D*H;i++) if(fabsf(ov[(size_t)t*D*H+i]-ref[i])>rowerr) {rowerr=fabsf(ov[(size_t)t*D*H+i]-ref[i]);worst=i;}
            maxerr=std::max(maxerr,rowerr);
            if(rowerr>.006f)fprintf(stderr,"row=%d h=%d d=%d actual=%g expected=%g error=%g\n",t,worst/D,worst%D,ov[(size_t)t*D*H+worst],ref[worst],rowerr);
        }
        std::sort(times.begin(),times.end());
        printf("{\"adaptive\":%s,\"mode\":%d,\"n\":%d,\"median_ms\":%.4f,\"min_ms\":%.4f,\"maxerr\":%.7f,\"finite\":%s,\"contexts\":%zu,\"created\":%ld,\"evicted\":%ld}\n",argc>1?argv[1]:"1",mode,n,times[2],times[0],maxerr,finite?"true":"false",rkfa::ctxs.size(),rkfa::n_create,rkfa::n_evict);
        fflush(stdout);
        if(!finite || maxerr>.006f)return 3;
    }
}
'''
Path(sys.argv[2]).write_text(prefix + fa + main)
