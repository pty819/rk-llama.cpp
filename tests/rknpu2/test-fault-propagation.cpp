#include "../../ggml/src/ggml-rknpu2/ggml-rknpu2.cpp"
#include <atomic>
#include <chrono>

struct TestMatmul { rknn_matmul_info info; rknn_tensor_mem *c=nullptr; };
static int fault;
static std::atomic<int> runs{0}, from_sync{0}, active{0}, to_sync{0};

int rknn_matmul_create(rknn_matmul_ctx *ctx,rknn_matmul_info *info,rknn_matmul_io_attr *io) {
    auto *m=new TestMatmul; m->info=*info;*ctx=(rknn_matmul_ctx)m;
    memset(io,0,sizeof(*io));
    strcpy(io->A.name,"A");strcpy(io->B.name,"B");strcpy(io->C.name,"C");
    io->A.size=info->M*info->K*2;io->B.size=info->K*info->N*2;io->C.size=info->M*info->N*4;
    return 0;
}
int rknn_matmul_destroy(rknn_matmul_ctx ctx) {delete (TestMatmul*)ctx;return 0;}
int rknn_matmul_set_core_mask(rknn_matmul_ctx,rknn_core_mask) {return 0;}
rknn_tensor_mem *rknn_create_mem(rknn_context,uint32_t size) {
    auto *m=new rknn_tensor_mem{};m->virt_addr=calloc(1,size);m->size=size;m->fd=-1;m->flags=1;return m;
}
rknn_tensor_mem *rknn_create_mem_from_fd(rknn_context,int32_t fd,void *addr,uint32_t size,int32_t offset) {
    auto *m=new rknn_tensor_mem{};m->virt_addr=(char*)addr+offset;m->size=size;m->fd=fd;m->flags=0;return m;
}
int rknn_destroy_mem(rknn_context,rknn_tensor_mem *mem) {if(mem->flags==1)free(mem->virt_addr);delete mem;return 0;}
int rknn_mem_sync(rknn_context,rknn_tensor_mem*,rknn_mem_sync_mode mode) {
    if(mode==RKNN_MEMORY_SYNC_FROM_DEVICE) {from_sync++;return (fault==3 || (fault==8 && from_sync==2))?-5:0;}
    to_sync++;return (fault==2 || (fault==7 && to_sync>=2))?-5:0;
}
int rknn_matmul_set_io_mem(rknn_matmul_ctx ctx,rknn_tensor_mem *mem,rknn_matmul_tensor_attr *attr) {
    if(fault==4 || (fault==5 && attr->name[0]=='A') || (fault==6 && attr->name[0]=='C'))return -5;
    if(attr->name[0]=='C')((TestMatmul*)ctx)->c=mem;
    return 0;
}
int rknn_matmul_run(rknn_matmul_ctx ctx) {
    runs++;active++;std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto *m=(TestMatmul*)ctx;
    if(fault!=1)std::fill_n((float*)m->c->virt_addr,m->info.M*m->info.N,1.f);
    active--;return fault==1?-5:0;
}
int main(int argc,char **argv) {
    fault=argc>1?atoi(argv[1]):0;
    const bool overlap=argc>2&&atoi(argv[2]);
    if (fault==9) { RKNN_CHECK(-5,"void upload failure"); return 3; }
    setenv("RKNPU_HYBRID","W16A16_STANDARD",1);setenv("RKNPU_OVERLAP",overlap?"1":"0",1);
    rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device("RK3588");
    ggml_init_params ip{1<<20,nullptr,false};auto *gc=ggml_init(ip);
    auto *w=ggml_new_tensor_2d(gc,GGML_TYPE_F16,32,96);ggml_set_name(w,"test.weight");
    auto *x=ggml_new_tensor_2d(gc,GGML_TYPE_F32,32,fault>=7?65:16);auto *y=ggml_mul_mat(gc,w,x);
    int M=fault>=7?65:16;
    setenv("RKNPU_M_TILE","32",1);
    std::fill_n((float*)x->data,32*M,1.f);std::fill_n((float*)y->data,96*M,-9.f);
    ggml_backend_rknpu_buffer_context bc{};bc.virtual_base=w->data;bc.total_size=ggml_nbytes(w);
    rknn_tensor_mem weight{};weight.virt_addr=w->data;weight.fd=333;weight.size=ggml_nbytes(w);
    bc.tensor_allocs[0]={&weight,weight.size,0};
    ggml_backend_buffer_type bt{};bt.iface.get_name=ggml_backend_rknpu_buffer_type_get_name;
    ggml_backend_buffer wb{};wb.buft=&bt;wb.context=&bc;w->buffer=&wb;
    ggml_backend_rknpu_context ctx{};ggml_backend backend{};backend.context=&ctx;
    auto *graph=ggml_new_graph(gc);ggml_build_forward_expand(graph,y);
    auto status=ggml_backend_rknpu_graph_compute(&backend,graph);
    bool ok=active==0 && status==(fault?GGML_STATUS_FAILED:GGML_STATUS_SUCCESS);
    if(fault==1)ok &= from_sync==0;
    if(fault && fault<7)for(int i=0;i<96*M;i++)ok &= ((float*)y->data)[i]==-9.f;
    else if(!fault)for(int i=0;i<96*M;i++)ok &= ((float*)y->data)[i]==1.f;
    printf("fault=%d overlap=%d status=%d runs=%d from_sync=%d active=%d %s\n",fault,overlap,status,(int)runs,(int)from_sync,(int)active,ok?"PASS":"FAIL");
    w->buffer=nullptr;ggml_free(gc);return !ok;
}
