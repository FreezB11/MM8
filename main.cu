// ///@file:main.cu 
// ///@note this file will not have much updates
// ///      we will keep the kernels in the .cuh file
// ///      so we just edit and improve that

// #include "kernel.cuh"
// #include <stdio.h>

// void initMat(f8* A, int N){float tmp;for(int i = 0; i < N; i++){tmp = (float)rand() / (float)RAND_MAX;A[i] = f32tof8(tmp);}}
// float timeKernel(cudaEvent_t s, cudaEvent_t e){ float ms; cudaEventElapsedTime(&ms,s,e); return ms; }

// int main(){
//     srand((unsigned)time(NULL));

//     f8 *A, *B, *C;
//     int N = 1024*4;

//     cudaMallocManaged(&A, N*N*sizeof(f8));
//     cudaMallocManaged(&B, N*N*sizeof(f8));
//     cudaMallocManaged(&C, N*N*sizeof(f8));

//     initMat(A, N*N);
//     initMat(B, N*N);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     dim3 block(TILE/WPT, TILE/WPT);
//     dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

//     // --- warmup: run both once, discard timing ---
//     mm84<<<grid, block>>>(A, B, C, N); cudaDeviceSynchronize();
//     // --- real benchmark: run each 5x, average ---
//     float t1 = 0;
//     int RUNS = 15;

//     for(int i = 0; i < RUNS; i++){
//         cudaEventRecord(start);
//         mm84<<<grid, block>>>(A, B, C, N);
//         cudaEventRecord(stop);
//         cudaDeviceSynchronize();
//         t1 += timeKernel(start, stop);
//     }

//     printf("mm84  avg: %.3f ms\n", t1/RUNS);

//     // swap order and run again to confirm
//     t1 = 0;
//     for(int i = 0; i < RUNS; i++){
//         cudaEventRecord(start);
//         mm84<<<grid, block>>>(A, B, C, N);
//         cudaEventRecord(stop);
//         cudaDeviceSynchronize();
//         t1 += timeKernel(start, stop);
//     }
//     printf("--- order swapped ---\n");
//     printf("mm84  avg: %.3f ms\n", t1/RUNS);

//     int blocks_per_sm;
//     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, mm84, 256, 0);
//     printf("mm84 blocks/SM: %d\n", blocks_per_sm);
//     cudaFuncAttributes attr;
//     cudaFuncGetAttributes(&attr, mm84);
//     printf("mm84 registers/thread: %d\n", attr.numRegs);

//     cudaDeviceSynchronize();
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//     cudaFree(A); cudaFree(B); cudaFree(C);
// }

#include "kernel.cuh"
#include <stdio.h>

void initMat(f8* A, int N){float tmp;for(int i = 0; i < N; i++){tmp = (float)rand() / (float)RAND_MAX;A[i] = f32tof8(tmp);}}
float timeKernel(cudaEvent_t s, cudaEvent_t e){ float ms; cudaEventElapsedTime(&ms,s,e); return ms; }

void benchmark(const char* name, auto kernel, dim3 grid, dim3 block,
               f8* A, f8* B, f8* C, int N, int RUNS,
               cudaEvent_t start, cudaEvent_t stop)
{
    // warmup
    kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();

    float t = 0;
    for(int i = 0; i < RUNS; i++){
        cudaEventRecord(start);
        kernel<<<grid, block>>>(A, B, C, N);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        t += timeKernel(start, stop);
    }
    printf("%-6s avg: %.3f ms\n", name, t/RUNS);

    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, 256, 0);
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    printf("%-6s blocks/SM: %d  regs/thread: %d\n\n", name, blocks_per_sm, attr.numRegs);
}

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

    int RUNS = 15;

    // mm87 needs more shared mem than the default 48KB limit
    // tell the driver before any launch (only needs to be called once)
    cudaFuncSetAttribute(mm87,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        49152);   // 48KB — max on Ampere without special opt-in

    benchmark("mm84", mm84, grid, block, A, B, C, N, RUNS, start, stop);
    benchmark("mm85", mm85, grid, block, A, B, C, N, RUNS, start, stop);
    benchmark("mm87", mm87, grid, block, A, B, C, N, RUNS, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
}