#!/usr/bin/env python3
"""Exercise the production FA worker pool without RKNN hardware."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]).read_text()
pool = source[source.index('class WorkerPool {'):source.index('struct Ctx { rknn_matmul_ctx')]
headers = r'''
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <pthread.h>
#include <algorithm>
'''
main = r'''
int main() {
    for (int n : {1, 3, 6, 32}) {
        WorkerPool pool(n);
        std::vector<std::thread::id> ids(n);
        std::atomic<int> calls{0};
        for (int round = 0; round < 200; ++round) {
            auto fn = [&](int tid) {
                if (round == 0) ids[tid] = std::this_thread::get_id();
                if (ids[tid] != std::this_thread::get_id()) throw std::runtime_error("worker identity changed");
                calls++;
            };
            pool.run(fn);
            if (calls != n * (round + 1)) return 1;
        }
        std::atomic<int> completed{0};
        auto fail = [&](int tid) {
            if (tid == 0) throw std::runtime_error("injected worker failure");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            completed++;
        };
        bool caught = false;
        try { pool.run(fail); } catch (const std::runtime_error &) { caught = true; }
        if (!caught || completed != n - 1) return 2;
        auto recover = [&](int) { calls++; };
        pool.run(recover);
        if (calls != n * 201) return 3;
        printf("workers=%d identity, barrier, exception recovery PASS\n", n);
    }
    {
        WorkerPool pool(6);
        std::atomic<int> calls{0};
        auto caller = [&] {
            for (int round = 0; round < 200; ++round) {
                std::atomic<int> local{0};
                auto fn = [&](int) { local++; calls++; };
                pool.run(fn);
                if (local != 6) std::terminate();
            }
        };
        std::thread a(caller), b(caller);
        a.join(); b.join();
        if (calls != 2400) return 4;
        puts("concurrent dispatch callback lifetime PASS");
    }
    for (int i = 0; i < 50; ++i) {
        WorkerPool pool(3);
        auto fn = [](int) {};
        if (i % 2 == 0) pool.run(fn);
    }
    puts("repeated construction and idle destruction PASS");
    if (std::getenv("FA_POOL_BENCH")) {
        WorkerPool pool(6);
        auto noop = [](int) {};
        for (int i = 0; i < 100; ++i) pool.run(noop);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 500; ++i) pool.run(noop);
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < 500; ++i) {
            std::vector<std::thread> threads;
            for (int tid = 0; tid < 6; ++tid) threads.emplace_back(noop, tid);
            for (auto & thread : threads) thread.join();
        }
        auto t2 = std::chrono::steady_clock::now();
        printf("dispatch benchmark pool_us=%g spawn_us=%g\n", std::chrono::duration<double, std::micro>(t1-t0).count()/500, std::chrono::duration<double, std::micro>(t2-t1).count()/500);
    }
}
'''
# Fail after two threads have started; constructor cleanup must join them.
fault = r'''
static int started = 0, joined = 0;
struct TestThread {
    std::thread thread;
    template<typename Fn> explicit TestThread(Fn fn) {
        if (started == 2) throw std::runtime_error("injected thread creation failure");
        thread = std::thread(fn); started++;
    }
    TestThread(TestThread &&) = default;
    void join() { thread.join(); joined++; }
};
'''
fault_main = r'''
int main() {
    bool caught = false;
    try { WorkerPool pool(6); } catch (const std::runtime_error &) { caught = true; }
    if (!caught || started != 2 || joined != 2) return 1;
    puts("partial thread creation cleanup PASS");
}
'''
with tempfile.TemporaryDirectory(prefix='rknpu-fa-pool-') as tmp:
    for name, body in [('pool', headers + pool + main), ('creation', headers + fault + pool.replace('std::thread', 'TestThread') + fault_main)]:
        cpp = Path(tmp) / f'{name}.cpp'
        cpp.write_text(body)
        exe = Path(tmp) / name
        flags = ['-std=c++17', '-O2', '-pthread']
        if os.getenv('SANITIZE'): flags += ['-g', '-fno-omit-frame-pointer', '-fsanitize=' + os.environ['SANITIZE']]
        subprocess.run([os.getenv('CXX', 'c++'), *flags, str(cpp), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True, timeout=60)
