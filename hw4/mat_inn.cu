#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void mat_inn(double *c, const double* a, const double* b, long M, long N){
  #pragma omp parallel for
  for (long j = 0; j < M; j++) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) {
      sum = sum + (a[M*i+j] * b[i]);
    }
    c[j] = sum;
  }
}

__global__
void mat_inn_kernel(double *c, const double* a, const double* b, long M, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0;
  if (idx < M) {
    int i = 0;
    while (i < N) {
      sum = sum + a[M*i+idx] * b[i];
      i++;
    }  
    c[idx] = sum;
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
  long N = (1UL<<12);
  long M = (1UL<<12);
  //long N = 1024;
  //long M = 1024;

  double *x, *y, *z;
  cudaMallocManaged(&x, M*N * sizeof(double));
  cudaMallocManaged(&y, N * sizeof(double));
  cudaMallocManaged(&z, M * sizeof(double));
  double* z_ref = (double*) malloc(M * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    y[i] = 1.0/(i+1);  
    //y[i] = 1.0;
  }

  for (long i = 0; i < M; i++) {
    z[i] = 0;
    z_ref[i] = 0;  
  }

  for (int j = 0; j < M; j++)
      for (int i = 0; i < N; i++)
        x[i*M + j] = i*M + j;
        //x[i*M + j] = 1.0;

  double tt = omp_get_wtime();
  mat_inn(z_ref, x, y, M, N);
  printf("CPU Speed = %f s\n", omp_get_wtime()-tt);
  printf("CPU Bandwidth = %f GB/s\n", (M+2*M*N)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  tt = omp_get_wtime();
  mat_inn_kernel<<<M/1024,1024>>>(z, x, y, M, N);
  cudaDeviceSynchronize();
  printf("GPU Speed = %f s\n", omp_get_wtime()-tt);
  printf("GPU Bandwidth = %f GB/s\n", (M+2*M*N)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double err = 0;
  for (long i = 0; i < M; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", err);
  //printf("Error = %f\n", z[M-1]);
  //printf("Error = %f\n", z_ref[M-1]);

  return 0;
}
