#include <stdio.h>
#include <math.h>
#include <stdlib.h>  
#include "utils.h"
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

void GS(int N) {
    double h = 1.0/(N+1.0);
    int NT = 16;
    double* u = (double*) calloc(sizeof(double), (N+2)*(N+2)); // (N+2) x (N+2)
    std::fill_n(u, (N+2)*(N+2), 0.0);
    Timer t;
    t.tic();
    for(int i = 0; i < 5000; i++) {
        // red points
	#ifdef _OPENMP
	#pragma omp parallel for collapse(2) num_threads(NT)
        #endif
        for(int j = 1; j < N+1; j=j+2) {
            for(int i = 1; i < N+1; i=i+2) {
               u[i*(N+2) + j] = (pow(h,2)+u[(i-1)*(N+2) + j]+u[i*(N+2) + (j-1)]+u[(i+1)*(N+2) + j]+u[i*(N+2) + (j+1)])/4;
            }
        }
	
	#ifdef _OPENMP
	#pragma omp parallel for collapse(2) num_threads(NT)
        #endif
        for(int j = 2; j < N+1; j=j+2) {
            for(int i = 2; i < N+1; i=i+2) {
               u[i*(N+2) + j] = (pow(h,2)+u[(i-1)*(N+2) + j]+u[i*(N+2) + (j-1)]+u[(i+1)*(N+2) + j]+u[i*(N+2) + (j+1)])/4;
            }
        }
        // black points
	#ifdef _OPENMP
	#pragma omp parallel for collapse(2) num_threads(NT)
        #endif
        for(int j = 1; j < N+1; j=j+2) {
            for(int i = 2; i < N+1; i=i+2) {
               u[i*(N+2) + j] = (pow(h,2)+u[(i-1)*(N+2) + j]+u[i*(N+2) + (j-1)]+u[(i+1)*(N+2) + j]+u[i*(N+2) + (j+1)])/4;
            }
        }
	
	#ifdef _OPENMP
	#pragma omp parallel for collapse(2) num_threads(NT)
        #endif
        for(int j = 2; j < N+1; j=j+2) {
            for(int i = 1; i < N+1; i=i+2) {
               u[i*(N+2) + j] = (pow(h,2)+u[(i-1)*(N+2) + j]+u[i*(N+2) + (j-1)]+u[(i+1)*(N+2) + j]+u[i*(N+2) + (j+1)])/4;
            }
        }

        if(i == 100) {
            double time = t.toc();
            printf("Run time for N = %d and for %d number of threads for 100 iterations: %f \n", N, NT, time);
            break;
        }
    }
    free(u);
}

int main() {
    //GS(100);
    GS(1000);
    return 0;
}