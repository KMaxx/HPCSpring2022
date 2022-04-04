#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;
  int p = 16;
  long *s;

  #pragma omp parallel num_threads(p)
  {
    int tid = omp_get_thread_num();

    #pragma omp single
    s = (long*) malloc(p * sizeof(long)); 

    long sum = 0;
    #pragma omp for nowait
    for (long i = 1; i < n; i++) {
      sum = sum + A[i-1];
      prefix_sum[i] = sum;
    }
    s[tid] = sum;
    #pragma omp barrier

    long offby = 0;
    for(long i = 0; i < tid; i++) {
      offby = offby + s[i];
    }

    #pragma omp for
    for (long i = 1; i < n; i++) {
      prefix_sum[i] = prefix_sum[i] + offby;
    }
  }
  free(s);
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
