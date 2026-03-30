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

    dim3 block(TILE/WPT, TILE/WPT);
    dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

    // --- warmup: run both once, discard timing ---
    mm85<<<grid, block>>>(A, B, C, N); cudaDeviceSynchronize();
    // --- real benchmark: run each 5x, average ---
    float t1 = 0;
    int RUNS = 15;

    for(int i = 0; i < RUNS; i++){
        cudaEventRecord(start);
        mm85<<<grid, block>>>(A, B, C, N);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        t1 += timeKernel(start, stop);
    }

    printf("mm85  avg: %.3f ms\n", t1/RUNS);

    // swap order and run again to confirm
    t1 = 0;
    for(int i = 0; i < RUNS; i++){
        cudaEventRecord(start);
        mm85<<<grid, block>>>(A, B, C, N);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        t1 += timeKernel(start, stop);
    }
    printf("--- order swapped ---\n");
    printf("mm85  avg: %.3f ms\n", t1/RUNS);

    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, mm85, 256, 0);
    printf("mm85 blocks/SM: %d\n", blocks_per_sm);
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, mm85);
    printf("mm85 registers/thread: %d\n", attr.numRegs);

    cudaDeviceSynchronize();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
}