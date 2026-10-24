//
// Created by Jan Thieme on 24.10.2026.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#include <stdio.h>
#include <stdlib.h>
#include "../src/linear_algebra/lamp_matrix.h"
#include "../src/neural_network/lamp_nn.h"
#include "lamp_version.h"

#include <time.h>

#if defined(_WIN32)
  #include <windows.h>

  static inline double benchmark_now(void) {
      LARGE_INTEGER t;
      QueryPerformanceCounter(&t);
      static LARGE_INTEGER freq;
      if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
      return (double)t.QuadPart / (double)freq.QuadPart;
  }

#else
#include <time.h>

static inline double benchmark_now(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return ts.tv_sec + ts.tv_nsec * 1e-9;
  }
#endif

// Training "bigger" models on my old, underpowered machine is no fun.
// Things are implemented very naively to just see if a can make it work.
// Well - it works. But I don´t like to use it because its so tedious.
// Lets find out what can be optimised before looking for other options.

typedef struct {
    const char *desc;
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    double fwd_pass_per_sec_cnt;   // forward passes per second
    double fwd_pass_ms;    // ms per single forward pass
} BenchmarkFwdResult;

typedef struct {
    const char *desc;
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    int batch;
    double iter_per_sec;
    double iter_ms;
} BenchmarkBpResult;

typedef struct {
    size_t input_size;
    size_t hidden_size;
    int batch;
    double fwd_ms;
    double fwd_bp_ms;
    double bp_ms;
    double ratio;
} BenchmarkCachingResult;

typedef struct {
    double *mat_mul_gflops;
    int   mat_mul_gflops_count;

    // forward pass
    BenchmarkFwdResult *fwd_results;
    int fwd_count;

    // backprop
    BenchmarkBpResult *bp_results;
    int bp_count;

    // caching comparison
    BenchmarkCachingResult caching;
    int has_caching;
} BenchmarkResults;

static BenchmarkResults bench_results = {0};

static void record_matmul_gflops(double gflops) {
    bench_results.mat_mul_gflops_count++;
    double *tmp = realloc(bench_results.mat_mul_gflops,
                          bench_results.mat_mul_gflops_count * sizeof(double));
    if (tmp) {
        bench_results.mat_mul_gflops = tmp;
        bench_results.mat_mul_gflops[bench_results.mat_mul_gflops_count - 1] = gflops;
    }
}

static void record_fwd(const char *desc, size_t in, size_t hidden, size_t out,
                       double fwd_per_sec, double ms_per_fwd) {
    bench_results.fwd_count++;
    BenchmarkFwdResult *tmp = realloc(bench_results.fwd_results,
                          bench_results.fwd_count * sizeof(bench_results.fwd_results[0]));
    if (tmp) {
        bench_results.fwd_results = tmp;
        tmp[bench_results.fwd_count - 1].desc        = desc;
        tmp[bench_results.fwd_count - 1].input_size          = in;
        tmp[bench_results.fwd_count - 1].hidden_size      = hidden;
        tmp[bench_results.fwd_count - 1].output_size         = out;
        tmp[bench_results.fwd_count - 1].fwd_pass_per_sec_cnt = fwd_per_sec;
        tmp[bench_results.fwd_count - 1].fwd_pass_ms  = ms_per_fwd;
    }
}

static void record_bp(const char *desc, size_t in, size_t hidden, size_t out,
                      int batch, double iter_per_sec, double ms_per_iter) {
    bench_results.bp_count++;
    BenchmarkBpResult *tmp = realloc(bench_results.bp_results,
                          bench_results.bp_count * sizeof(bench_results.bp_results[0]));
    if (tmp) {
        bench_results.bp_results = tmp;
        tmp[bench_results.bp_count - 1].desc         = desc;
        tmp[bench_results.bp_count - 1].input_size           = in;
        tmp[bench_results.bp_count - 1].hidden_size       = hidden;
        tmp[bench_results.bp_count - 1].output_size          = out;
        tmp[bench_results.bp_count - 1].batch        = batch;
        tmp[bench_results.bp_count - 1].iter_per_sec = iter_per_sec;
        tmp[bench_results.bp_count - 1].iter_ms  = ms_per_iter;
    }
}

static void record_caching(size_t in, size_t hidden, int batch,
                           double fwd_ms, double fwd_bp_ms, double bp_ms,
                           double ratio) {
    bench_results.caching.input_size       = in;
    bench_results.caching.hidden_size   = hidden;
    bench_results.caching.batch    = batch;
    bench_results.caching.fwd_ms   = fwd_ms;
    bench_results.caching.fwd_bp_ms= fwd_bp_ms;
    bench_results.caching.bp_ms    = bp_ms;
    bench_results.caching.ratio    = ratio;
    bench_results.has_caching      = 1;
}

void benchmark_matmul(void) {
    printf("=== Matrix Multiplication Benchmark ===\n");
    printf("  %12s  %12s  %12s  %12s\n", "Size (N)", "GFLOP/s", "Time (ms)", "Reps");
    printf("  ----------------------------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < n_sizes; i++) {
        int n = sizes[i];
        LampMatrix *A = lamp_mat_alloc(n, n);
        LampMatrix *B = lamp_mat_alloc(n, n);
        LampMatrix *C = lamp_mat_alloc(n, n);
        lamp_mat_rand(A); // NOLINT(cert-msc30-c, cert-msc50-cpp)
        lamp_mat_rand(B); // NOLINT(cert-msc30-c, cert-msc50-cpp)
        lamp_mat_fill_with(C, 0);

        // Warmup
        lamp_mat_multiply_into(C, A, B);
        lamp_mat_fill_with(C, 0);

        // Fixed reps based on size to keep total time small
        int reps = (n <= 32) ? 10000 : (n <= 64) ? 1000 : (n <= 128) ? 100 : 10;
        double start = benchmark_now();
        for (int r = 0; r < reps; r++) {
            lamp_mat_fill_with(C, 0);
            lamp_mat_multiply_into(C, A, B);
        }
        double elapsed = benchmark_now() - start;

        double gflop_s = (2.0 * n * n * n * reps) / elapsed / 1e9;
        printf("  %12d  %12.3f  %12.2f  %10d\n",
               n, gflop_s, elapsed * 1000.0 / reps, reps);

        record_matmul_gflops(gflop_s);

        lamp_mat_free(A);
        lamp_mat_free(B);
        lamp_mat_free(C);
    }
    printf("\n");
}

void benchmark_forward_pass(void) {
    printf("=== Forward Pass Benchmark ===\n");
    printf("  (measures a single forward pass through a network)\n\n");

    struct { const char *desc; int in; int hidden; int out; } nets[] = {
        { "Logic gate (tiny)", 2, 8, 1 },
        { "Small MLP", 64, 128, 10 },
        { "Medium MLP (MNIST-ish)", 784, 256, 10 },
    };
    int n_nets = sizeof(nets) / sizeof(nets[0]);

    for (int idx = 0; idx < n_nets; idx++) {
        size_t in = nets[idx].in, hid = nets[idx].hidden, out = nets[idx].out;
        LampNN *nn = lamp_nn_alloc_with(in, 1, &hid, out);
        lamp_nn_set_activation(nn, &LAMP_ACTIVATION_RELU);

        LampMatrix *input = lamp_mat_alloc(in, 1);
        lamp_mat_rand(input); // NOLINT(cert-msc30-c, cert-msc50-cpp)

        // Warmup
        lamp_nn_forward_single(nn, input);

        int reps = (in >= 256) ? 10000 : 1000000;
        double start = benchmark_now();
        for (int r = 0; r < reps; r++) {
            lamp_nn_forward_single(nn, input);
        }
        double elapsed = benchmark_now() - start;

        double fwd_per_s = reps / elapsed;
        double ms_per_fwd = elapsed * 1000.0 / reps;

        printf("  %-30s [%zu,%zu,%zu]  %.0f fwd/s\n",
               nets[idx].desc, in, hid, out, fwd_per_s);

        record_fwd(nets[idx].desc, in, hid, out, fwd_per_s, ms_per_fwd);

        lamp_nn_free(nn);
        lamp_mat_free(input);
    }
    printf("\n");
}

void benchmark_backprop(void) {
    printf("=== Backpropagation Benchmark ===\n");
    printf("  (measures one backprop step)\n\n");

    struct { const char *desc; int in; int hidden; int out; int batch; } nets[] = {
        { "Small MLP", 64, 128, 10, 4 },
        { "Medium MLP (MNIST-ish)", 784, 256, 10, 4 },
    };
    int n_nets = sizeof(nets) / sizeof(nets[0]);

    for (int idx = 0; idx < n_nets; idx++) {
        size_t in = nets[idx].in, hid = nets[idx].hidden, out = nets[idx].out, batch = nets[idx].batch;
        LampNN *nn = lamp_nn_alloc_with(in, 1, &hid, out);
        lamp_nn_set_activation(nn, &LAMP_ACTIVATION_RELU);

        LampMatrix *input = lamp_mat_alloc(batch, in);
        LampMatrix *target = lamp_mat_alloc(batch, out);
        lamp_mat_rand(input);
        lamp_mat_rand(target);

        // Warmup
        lamp_nn_backprop(nn, input, target, 0.01);

        int reps = 100;
        double start = benchmark_now();
        for (int r = 0; r < reps; r++) {
            lamp_nn_backprop(nn, input, target, 0.01);
        }
        double elapsed = benchmark_now() - start;

        double bp_per_s = reps / elapsed;
        double ms_per_iter = elapsed * 1000.0 / reps;

        printf("  %-20s [%d,%d,%d] batch=%d  %.1f iter/s  (%.1f ms/iter)\n",
               nets[idx].desc, in, hid, out, batch, bp_per_s, ms_per_iter);

        record_bp(nets[idx].desc, in, hid, out, batch, bp_per_s, ms_per_iter);

        lamp_nn_free(nn);
        lamp_mat_free(input);
        lamp_mat_free(target);
    }
    printf("\n");
}

void benchmark_caching_comparison(void) {
    printf("=== Caching Test: Forward Pass Reuse ===\n");
    printf("  (compares forward-only vs forward+backprop)\n\n");
    fflush(stdout);

    size_t in = 64, hid = 64;
    int batch = 16;
    printf("  Allocating network [%zu,%zu,10]...\n", in, hid);
    fflush(stdout);
    LampNN *nn = lamp_nn_alloc_with(in, 1, &hid, 10);
    printf("  Setting activation...\n");
    fflush(stdout);
    lamp_nn_set_activation(nn, &LAMP_ACTIVATION_RELU);
    printf("  Allocating matrices (batch=%d, in=%zu)...\n", batch, in);
    fflush(stdout);

    LampMatrix *input = lamp_mat_alloc(batch, in);
    LampMatrix *target = lamp_mat_alloc(batch, 10);
    LampMatrix *sample_input = lamp_mat_alloc(in, 1);
    lamp_mat_rand(input);
    lamp_mat_rand(target);

    int reps = 100;
    double start = benchmark_now();
    // Forward only
    for (int r = 0; r < reps; r++) {
        for (size_t j = 0; j < in; j++)
            sample_input->elements[j] = input->elements[j];
        lamp_nn_forward_single(nn, sample_input);
    }
    double fwd_time = benchmark_now() - start;

    start = benchmark_now();
    // Forward + backprop
    for (int r = 0; r < reps; r++) {
        lamp_nn_backprop(nn, input, target, 0.01);
    }
    double fwd_bprop_time = benchmark_now() - start;

    double fwd_ms   = fwd_time * 1000.0 / reps;
    double fwd_bp_ms = fwd_bprop_time * 1000.0 / reps;
    double bp_ms    = (fwd_bprop_time - fwd_time) * 1000.0 / reps;
    double ratio    = bp_ms > 0 ? fwd_time / (fwd_bprop_time - fwd_time) : 0;

    printf("  Forward only:     %.3f ms/iter\n", fwd_ms);
    printf("  Forward+Backprop: %.3f ms/iter\n", fwd_bp_ms);
    printf("  Backprop only:    %.3f ms/iter (forward cost subtracted)\n", bp_ms);
    printf("  Ratio (fwd:bp):   %.1f:1\n", ratio);

    record_caching(in, hid, batch, fwd_ms, fwd_bp_ms, bp_ms, ratio);

    lamp_mat_free(sample_input);
    lamp_nn_free(nn);
    lamp_mat_free(input);
    lamp_mat_free(target);
}

static void print_summary(void) {
    printf("\n=== Measured Results ===\n");

    // Average GFLOP/s across matmul sizes
    if (bench_results.mat_mul_gflops_count > 0) {
        double sum = 0;
        for (int i = 0; i < bench_results.mat_mul_gflops_count; i++)
            sum += bench_results.mat_mul_gflops[i];
        double avg = sum / bench_results.mat_mul_gflops_count;
        printf("Matrix multiply: %.3f GFLOP/s (average across %d sizes, 16-256)\n",
               avg, bench_results.mat_mul_gflops_count);
    }

    // Forward pass
    for (int i = 0; i < bench_results.fwd_count; i++) {
        printf("Forward pass %-30s [%zu,%zu,%zu]  %.3f ms per sample (%.0f fwd/s)\n",
               bench_results.fwd_results[i].desc, bench_results.fwd_results[i].input_size,
               bench_results.fwd_results[i].hidden_size, bench_results.fwd_results[i].output_size,
               bench_results.fwd_results[i].fwd_pass_ms, bench_results.fwd_results[i].fwd_pass_per_sec_cnt);
    }

    // Backprop
    for (int i = 0; i < bench_results.bp_count; i++) {
        printf("Backprop %-30s [%zu,%zu,%zu] batch=%d  %.1f ms/iter (%.0f iter/s)\n",
               bench_results.bp_results[i].desc, bench_results.bp_results[i].input_size,
               bench_results.bp_results[i].hidden_size, bench_results.bp_results[i].output_size,
               bench_results.bp_results[i].batch,
               bench_results.bp_results[i].iter_ms, bench_results.bp_results[i].iter_per_sec);
    }

    // Caching comparison
    if (bench_results.has_caching) {
        printf("Caching comparison [%zu,%zu,10] batch=%d:  fwd=%.3f ms, bp=%.3f ms, ratio %.1f:1\n",
               bench_results.caching.input_size, bench_results.caching.hidden_size,
               bench_results.caching.batch,
               bench_results.caching.fwd_ms, bench_results.caching.bp_ms, bench_results.caching.ratio);
    }
}

int main(void) {
    printf("Lamp Framework Performance Benchmark %s\n", LAMP_VERSION_STRING);
    printf("=====================================\n\n");
    fflush(stdout);

    benchmark_matmul();
    fflush(stdout);

    benchmark_forward_pass();
    fflush(stdout);

    benchmark_backprop();
    fflush(stdout);

    benchmark_caching_comparison();
    fflush(stdout);

    print_summary();
    printf("Benchmark settings: single-threaded, no optimizations\n");
    return 0;
}
