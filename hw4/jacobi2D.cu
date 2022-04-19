#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>


void Jacobi(int k, double* u, double* un, int NT, long N, double h){
  #pragma omp parallel for collapse(2) num_threads(NT)
  for(int j = 1; j < N+1; j++) {
    for(int i = 1; i < N+1; i++) {
      un[i*(N+2) + j] = (h*h+u[(i-1)*(N+2) + j]+u[i*(N+2) + (j-1)]+u[(i+1)*(N+2) + j]+u[i*(N+2) + (j+1)])/4;
    }
  }
  #pragma omp barrier	
}

__global__
void Jacobi_kernel(int k, double* u, double* un, long N, double h){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int i = 1;
    while (i < N+1) {
      un[i*(N+2) + idx+1] = (h*h+u[(i-1)*(N+2) + idx+1]+u[i*(N+2) + idx]+u[(i+1)*(N+2) + idx+1]+u[i*(N+2) + (idx+2)])/4;
      i++;
    }  
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
 // long N = 1024;
  long size = (N+2)*(N+2);

  double h = 1.0/(N+1.0);
  int NT = 16;
  double *u_c, *un_c;
  cudaMallocManaged(&u_c, size * sizeof(double));
  cudaMallocManaged(&un_c, size * sizeof(double));
  double* u = (double*) malloc(size * sizeof(double));
  double* un = (double*) malloc(size * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < size; i++) {
    u[i] = 0.0;  
    un[i] = 0.0;
    u_c[i] = 0.0;  
    un_c[i] = 0.0;
  }
  #pragma omp barrier	

  double tt = omp_get_wtime();
  for(int k = 0; k < 100; k++) {
    Jacobi(k, u, un, NT, N, h);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < size; i++) {
      u[i] = un[i];
    }
    #pragma omp barrier  
  }
  printf("CPU Speed = %f s\n", omp_get_wtime()-tt);

  tt = omp_get_wtime();
  for(int k = 0; k < 100; k++) {
    Jacobi_kernel<<<N/1024,1024>>>(k, u_c, un_c, N, h);
    cudaDeviceSynchronize();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < size; i++) {
      u_c[i] = un_c[i];
    }
    #pragma omp barrier  
  }
  printf("GPU Speed = %f s\n", omp_get_wtime()-tt);

  double err = 0;
  for (long i = 0; i < size; i++) err += fabs(u[i]-u_c[i]);
  printf("Error = %f\n", err);

  free(u);
  free(un);
  cudaFree(u_c);
  cudaFree(un_c);

  return 0;
}
