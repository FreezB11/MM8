///@file:main.cu 
///@note this file will not have much updates
///      we will keep the kernels in the .cuh file
///      so we just edit and improve that

#include "kernel.cuh"
#include <stdio.h>

void initMat(f8* A, int N){float tmp;for(int i = 0; i < N; i++){tmp = (float)rand() / (float)RAND_MAX;A[i] = f32tof8(tmp);}}
float timeKernel(cudaEvent_t s, cudaEvent_t e){ float ms; cudaEventElapsedTime(&ms,s,e); return ms; }

int main(){
    srand((unsigned)time(NULL));

    f8 *A, *B, *C;
    int N = 1024*4;

    cudaMallocManaged(&A, N*N*sizeof(f8));
    cudaMallocManaged(&B, N*N*sizeof(f8));
    cudaMallocManaged(&C, N*N*sizeof(f8));

    initMat(A, N*N);
    initMat(B, N*N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // mm8<<<(N+255)/256, 256>>>(A,B,C,N);
    {
        dim3 block(16,16);
        dim3 grid((N+15)/16, (N+15)/16);
        cudaEventRecord(start);
        mm8<<<grid, block>>>(A, B, C, N);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        printf("mm8:           %.3f ms\n", timeKernel(start, stop));
    }

    cudaDeviceSynchronize();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
}