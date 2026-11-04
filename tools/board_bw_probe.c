// E0a: board-side memory bandwidth probe.
//
// Measures achievable LPDDR4X bandwidth on the Atlas 200I A2 (3 CPU cores):
//   copy (1 thread), copy (3 threads), read (1/3 threads), write (1 thread).
// The NPU shares the same LPDDR4X; this gives an upper-bound reference for the
// 9.73 GB/s effective weight bandwidth measured in T04/T07.
//
// Build: gcc -O2 -std=c11 -pthread board_bw_probe.c -o board_bw_probe
// Run:   ./board_bw_probe [MiB=256] [iters=5] [threads=3]

#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint8_t *src;
    uint8_t *dst;
    size_t bytes;
    int mode; // 0=copy, 1=read, 2=write
    volatile uint64_t sink;
} Task;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *worker(void *arg) {
    Task *t = (Task *)arg;
    if (t->mode == 0) {
        memcpy(t->dst, t->src, t->bytes);
        t->sink = ((uint64_t *)t->dst)[0] + ((uint64_t *)t->dst)[t->bytes / 8 - 1];
    } else if (t->mode == 1) {
        const uint64_t *p = (const uint64_t *)t->src;
        uint64_t s = 0;
        size_t n = t->bytes / 8;
        for (size_t i = 0; i < n; ++i) s += p[i];
        t->sink = s;
    } else {
        memset(t->dst, 0xA5, t->bytes);
        t->sink = ((uint64_t *)t->dst)[t->bytes / 8 - 1];
    }
    return NULL;
}

static double median(double *v, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (v[j] < v[i]) { double x = v[i]; v[i] = v[j]; v[j] = x; }
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double run_case(const char *tag, uint8_t *src, uint8_t *dst, size_t total,
                       int threads, int mode, int iters, int logical_factor) {
    double gbps[16];
    if (iters > 16) iters = 16;
    for (int it = 0; it < iters; ++it) {
        pthread_t th[8];
        Task t[8];
        size_t chunk = total / (size_t)threads;
        double t0 = now_sec();
        for (int k = 0; k < threads; ++k) {
            t[k].src = src + (size_t)k * chunk;
            t[k].dst = dst + (size_t)k * chunk;
            t[k].bytes = (k + 1 == threads) ? (total - (size_t)k * chunk) : chunk;
            t[k].mode = mode;
            t[k].sink = 0;
            if (pthread_create(&th[k], NULL, worker, &t[k]) != 0) {
                fprintf(stderr, "pthread_create failed\n");
                exit(3);
            }
        }
        for (int k = 0; k < threads; ++k) pthread_join(th[k], NULL);
        double dt = now_sec() - t0;
        double bytes = (double)total * (double)logical_factor;
        gbps[it] = bytes / dt / 1e9;
        uint64_t sink = 0;
        for (int k = 0; k < threads; ++k) sink += t[k].sink;
        printf("RUN tag=%s it=%d threads=%d sec=%.6f GBps=%.3f sink=%llu\n",
               tag, it, threads, dt, gbps[it], (unsigned long long)sink);
    }
    double m = median(gbps, iters);
    printf("MED tag=%s threads=%d mode=%d GBps_median=%.3f\n", tag, threads, mode, m);
    return m;
}

int main(int argc, char **argv) {
    size_t mib = (argc > 1) ? (size_t)atol(argv[1]) : 256;
    int iters = (argc > 2) ? atoi(argv[2]) : 5;
    int threads = (argc > 3) ? atoi(argv[3]) : 3;
    if (mib < 16) mib = 16;
    if (iters < 1) iters = 1;
    if (threads < 1) threads = 1;
    if (threads > 8) threads = 8;
    size_t total = mib * 1024 * 1024;

    uint8_t *src = NULL, *dst = NULL;
    if (posix_memalign((void **)&src, 64, total) != 0 ||
        posix_memalign((void **)&dst, 64, total) != 0) {
        fprintf(stderr, "FATAL: posix_memalign(%zu MiB x2) failed\n", mib);
        return 2;
    }
    memset(src, 0x3C, total);
    memset(dst, 0, total);
    printf("PROBE mib=%zu iters=%d threads=%d src=%p dst=%p\n",
           mib, iters, threads, (void *)src, (void *)dst);
    fflush(stdout);

    run_case("copy1", src, dst, total, 1, 0, iters, 2);
    run_case("copyN", src, dst, total, threads, 0, iters, 2);
    run_case("read1", src, dst, total, 1, 1, iters, 1);
    run_case("readN", src, dst, total, threads, 1, iters, 1);
    run_case("write1", src, dst, total, 1, 2, iters, 1);
    run_case("writeN", src, dst, total, threads, 2, iters, 1);

    free(src);
    free(dst);
    printf("DONE\n");
    return 0;
}
