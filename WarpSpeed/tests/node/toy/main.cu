// WarpSpeed node-acceptance toy kernel.
// Implements the WarpSpeed benchmark-binary contract (see harness-template/README.md):
//   kernel_bench --reps N --warmup W [--shape small|default] [--check]
// prints one "LAT_US=<float>" line per measured rep (cudaEvent timing).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  exit(2); } } while (0)

__global__ void vecmul(const float* a, const float* b, float* c, long n, int inner) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (int k = 0; k < inner; ++k) {
    for (long j = i; j < n; j += stride) {
      c[j] = a[j] * b[j];
    }
  }
}

int main(int argc, char** argv) {
  int reps = 30, warmup = 5, check = 0;
  const char* shape = "default";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--shape") && i + 1 < argc) shape = argv[++i];
    else if (!strcmp(argv[i], "--check")) check = 1;
  }
  long n = !strcmp(shape, "small") ? (1L << 20) : (1L << 24);
  int inner = 8;

  float *a, *b, *c;
  CK(cudaMalloc(&a, n * sizeof(float)));
  CK(cudaMalloc(&b, n * sizeof(float)));
  CK(cudaMalloc(&c, n * sizeof(float)));
  float* h = (float*)malloc(n * sizeof(float));
  for (long i = 0; i < n; ++i) h[i] = 1.0f + (float)(i % 7);
  CK(cudaMemcpy(a, h, n * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b, h, n * sizeof(float), cudaMemcpyHostToDevice));

  int block = 256;
  int grid = 1024;
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));

  if (check) {
    vecmul<<<grid, block>>>(a, b, c, n, 1);
    CK(cudaDeviceSynchronize());
    float* out = (float*)malloc(n * sizeof(float));
    CK(cudaMemcpy(out, c, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (long i = 0; i < n; i += n / 1024 + 1) {
      float want = h[i] * h[i];
      if (out[i] < want - 1e-3f || out[i] > want + 1e-3f) {
        printf("{\"correct\": false, \"shape\": \"%s\"}\n", shape);
        return 1;
      }
    }
    printf("{\"correct\": true, \"shape\": \"%s\"}\n", shape);
    return 0;
  }

  for (int r = 0; r < warmup; ++r) vecmul<<<grid, block>>>(a, b, c, n, inner);
  CK(cudaDeviceSynchronize());
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(t0));
    vecmul<<<grid, block>>>(a, b, c, n, inner);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    printf("LAT_US=%.3f\n", ms * 1000.0f);
  }
  return 0;
}
