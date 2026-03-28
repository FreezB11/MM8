///@file:bench.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "kernel.cuh"

#define RUNS 5
#define WARMUP 2

int main(){
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("# GPU memory: %.2f MB free / %.2f MB total\n",
           free_mem/1e6, total_mem/1e6);
    printf("# TILE=%d WPT=%d\n\n", TILE, WPT);

    // csv header
    printf("N,Time_ms,GFLOPS,Memory_MB\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // sweep: every TILE step from TILE up to OOM
    for(int N = TILE; ; N += TILE*4){
        size_t mem_needed = 3ULL * N * N * sizeof(f8);

        // check if we have enough memory
        cudaMemGetInfo(&free_mem, &total_mem);
        if(mem_needed > (size_t)(free_mem * 0.90)){
            printf("# N=%d would need %.1f MB, only %.1f MB free — stopping\n",
                   N, mem_needed/1e6, free_mem/1e6);
            break;
        }

        f8 *A, *B, *C;
        cudaError_t e1 = cudaMalloc(&A, (size_t)N*N*sizeof(f8));
        cudaError_t e2 = cudaMalloc(&B, (size_t)N*N*sizeof(f8));
        cudaError_t e3 = cudaMalloc(&C, (size_t)N*N*sizeof(f8));

        if(e1!=cudaSuccess || e2!=cudaSuccess || e3!=cudaSuccess){
            fprintf(stderr, "# alloc failed at N=%d (%.1f MB)\n",
                    N, mem_needed/1e6);
            if(e1==cudaSuccess) cudaFree(A);
            if(e2==cudaSuccess) cudaFree(B);
            cudaGetLastError();
            break;
        }

        // init with random fp8
        f8 *hA = (f8*)malloc((size_t)N*N);
        f8 *hB = (f8*)malloc((size_t)N*N);
        for(int i = 0; i < N*N; i++){
            hA[i] = f32tof8((float)rand()/RAND_MAX);
            hB[i] = f32tof8((float)rand()/RAND_MAX);
        }
        cudaMemcpy(A, hA, (size_t)N*N, cudaMemcpyHostToDevice);
        cudaMemcpy(B, hB, (size_t)N*N, cudaMemcpyHostToDevice);
        free(hA); free(hB);

        dim3 block(TILE/WPT, TILE/WPT);
        dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

        // warmup
        for(int i = 0; i < WARMUP; i++)
            mm83<<<grid, block>>>(A, B, C, N);
        cudaDeviceSynchronize();

        if(cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "# kernel failed at N=%d\n", N);
            cudaFree(A); cudaFree(B); cudaFree(C);
            cudaGetLastError();
            break;
        }

        // benchmark
        float total_ms = 0;
        for(int i = 0; i < RUNS; i++){
            float ms;
            cudaEventRecord(start);
            mm83<<<grid, block>>>(A, B, C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }

        float avg_ms = total_ms / RUNS;
        double gflops = (2.0 * N * N * N) / (avg_ms / 1000.0) / 1e9;

        printf("%d,%.4f,%.4f,%.2f\n",
               N, avg_ms, gflops, mem_needed/1e6);
        fflush(stdout);   // flush each line so you can tail -f while running

        cudaFree(A); cudaFree(B); cudaFree(C);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}