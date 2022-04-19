#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void vec_inn(double *c, const double* a, const double* b, long N){
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (long i = 0; i < N; i++) {
    sum = sum + (a[i] * b[i]);
  }
  c[0] = sum;
}

__global__
void vec_inn_kernel(double *c, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < N) {
    double temp = a[idx] * b[idx];
    atomicAdd(c, temp);
    idx = idx + gridDim.x * blockDim.x;
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  long N = (1UL<<25);

  double *x, *y, *z;
  cudaMallocManaged(&x, N * sizeof(double));
  cudaMallocManaged(&y, N * sizeof(double));
  cudaMallocManaged(&z, N * sizeof(double));
  double* z_ref = (double*) malloc(N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = i+2;
    y[i] = 1.0/(i+1);
  }
  z[0] = 0;
  z_ref[0] = 0;

  double tt = omp_get_wtime();
  vec_inn(z_ref, x, y, N);
  printf("CPU Speed = %f s\n", omp_get_wtime()-tt);
  printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  tt = omp_get_wtime();
  vec_inn_kernel<<<N/1024,1024>>>(z, x, y, N);
  cudaDeviceSynchronize();
  printf("GPU Speed = %f s\n", omp_get_wtime()-tt);
  printf("GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double err = fabs(z[0]-z_ref[0]);
  printf("Error = %f\n", err);

  return 0;
}
