#include <poll.h>

int test_runtime_poll(struct pollfd *fds, nfds_t nfds, int timeout)
{
    /* Keep the call in this shared object for the caller-identity test. */
    int ret = poll(fds, nfds, timeout);
    __asm__ __volatile__("" : "+r"(ret));
    return ret;
}
