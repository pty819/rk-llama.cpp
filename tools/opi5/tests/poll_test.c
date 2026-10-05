#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static void handler(int signal) {(void)signal;}
static pthread_t waiter;
static void *signals(void *unused)
{
    (void)unused;
    for (int i=0;i<15;i++) {usleep(20000);pthread_kill(waiter,SIGUSR1);}
    return NULL;
}
static double ms(void)
{
    struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000.0+ts.tv_nsec/1e6;
}
int main(int argc,char **argv)
{
    int which=argc>1?atoi(argv[1]):0;
    void *lib=dlopen(argv[2],RTLD_NOW);
    if(!lib){fprintf(stderr,"%s\n",dlerror());return 2;}
    int (*runtime_poll)(struct pollfd*,nfds_t,int);
    *(void **)(&runtime_poll)=dlsym(lib,"test_runtime_poll");
    if(!runtime_poll)return 2;
    struct sigaction sa={0};sa.sa_handler=handler;sigaction(SIGUSR1,&sa,NULL);
    int fds[2];if(pipe(fds))return 2;
    struct pollfd p={fds[0],POLLIN,0};
    waiter=pthread_self();pthread_t sender;
    if(which<2 || which==6)pthread_create(&sender,NULL,signals,NULL);
    if(which==3)write(fds[1],"x",1);
    double start=ms();
    int timeout=which==2?-1:which==4?0:which==7?500:100;
    int ret=which==1?poll(&p,1,timeout):runtime_poll(&p,1,timeout);
    int err=errno;double elapsed=ms()-start;
    if(which<2 || which==6)pthread_join(sender,NULL);
    int ok;
    if(which==1 || which==6)ok=ret==-1&&err==EINTR&&elapsed<80;
    else if(which==3)ok=ret==1&&(p.revents&POLLIN)&&elapsed<20;
    else if(which==4)ok=ret==0&&elapsed<20;
    else ok=ret==0&&elapsed>=90&&elapsed<220;
    printf("case=%d ret=%d errno=%d elapsed_ms=%.2f %s\n",which,ret,err,elapsed,ok?"PASS":"FAIL");
    close(fds[0]);close(fds[1]);dlclose(lib);
    return !ok;
}
