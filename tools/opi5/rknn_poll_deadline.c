/* LD_PRELOAD compatibility shim for librknnrt poll deadlines. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static int (*next_poll)(struct pollfd *, nfds_t, int);
static int max_ms = 30000;
static int disabled;

static void init_shim(void)
{
    *(void **)(&next_poll) = dlsym(RTLD_NEXT, "poll");
    const char *e = getenv("RKNPU_POLL_MAX_MS");
    if (e) {
        char *end;
        long value = strtol(e, &end, 10);
        if (*e && !*end && value > 0 && value <= INT_MAX)
            max_ms = (int)value;
    }
    e = getenv("RKNPU_POLL_DEADLINE_DISABLE");
    disabled = e && atoi(e) != 0;
}

static int is_runtime_call(void *caller)
{
    Dl_info info;
    if (!dladdr(caller, &info) || !info.dli_fname)
        return 0;
    const char *base = strrchr(info.dli_fname, '/');
    base = base ? base + 1 : info.dli_fname;
    const char *name = "librknnrt.so";
    const size_t n = strlen(name);
    return strncmp(base, name, n) == 0 && (base[n] == '\0' || base[n] == '.');
}

static int64_t monotonic_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts))
        return -1;
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static int wait_until(struct pollfd *fds, nfds_t nfds, int timeout)
{
    int64_t now = monotonic_ns();
    if (now < 0)
        return -1;
    const int64_t deadline = now + (int64_t)timeout * 1000000;
    for (;;) {
        now = monotonic_ns();
        if (now < 0)
            return -1;
        const int64_t left = deadline - now;
        if (left <= 0)
            return 0;
        const int remaining = (int)((left + 999999) / 1000000);
        int ret = next_poll(fds, nfds, remaining);
        if (ret >= 0 || (errno != EINTR && errno != EAGAIN))
            return ret;
    }
}

int poll(struct pollfd *fds, nfds_t nfds, int timeout)
{
    const int saved_errno = errno;
    void *caller = __builtin_return_address(0);
    pthread_once(&init_once, init_shim);
    if (!next_poll) {
        errno = ENOSYS;
        return -1;
    }
    if (disabled || timeout == 0 || !is_runtime_call(caller)) {
        errno = saved_errno;
        return next_poll(fds, nfds, timeout);
    }
    if (timeout < 0 || timeout > max_ms)
        timeout = max_ms;
    errno = saved_errno;
    return wait_until(fds, nfds, timeout);
}
