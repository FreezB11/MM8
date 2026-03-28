///@file:baseline_final.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "kernel.cuh"

#define RUNS 10

#define CHECK_CUDA(x) if((x)!=cudaSuccess){printf("CUDA error at %d\n",__LINE__); return -1;}
#define CHECK_CUBLAS(x) if((x)!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS error at %d\n",__LINE__); return -1;}

int SIZES[] = {16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384};
int NUM_SIZES = sizeof(SIZES)/sizeof(SIZES[0]);

double flops(int N, double ms){
    return (2.0 * N * N * N) / (ms / 1000.0);
}

int main(){
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha   = 1.0f, beta  = 0.0f;
    int   alpha_i = 1,    beta_i = 0;

    printf("N,TYPE,Time(ms),GFLOPS\n");

    for(int s = 0; s < NUM_SIZES; s++){
        int N = SIZES[s];
        printf("\n===== N = %d =====\n", N);

        float ms;

        // ================= FP32 =================
        {
            float *A, *B, *C;
            CHECK_CUDA(cudaMalloc(&A, N*N*sizeof(float)));
            CHECK_CUDA(cudaMalloc(&B, N*N*sizeof(float)));
            CHECK_CUDA(cudaMalloc(&C, N*N*sizeof(float)));

            for(int i = 0; i < 3; i++)
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, N, N, &alpha, A, N, B, N, &beta, C, N);

            cudaEventRecord(start);
            for(int i = 0; i < RUNS; i++)
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, N, N, &alpha, A, N, B, N, &beta, C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            ms /= RUNS;
            printf("%d,FP32,%.4f,%.2f\n", N, ms, flops(N, ms)/1e9);

            cudaFree(A); cudaFree(B); cudaFree(C);
        }

        // ================= FP16 =================
        {
            __half *A, *B, *C;
            CHECK_CUDA(cudaMalloc(&A, N*N*sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&B, N*N*sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&C, N*N*sizeof(__half)));

            for(int i = 0; i < 3; i++)
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha,
                             A, CUDA_R_16F, N,
                             B, CUDA_R_16F, N,
                             &beta,
                             C, CUDA_R_16F, N,
                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            cudaEventRecord(start);
            for(int i = 0; i < RUNS; i++)
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha,
                             A, CUDA_R_16F, N,
                             B, CUDA_R_16F, N,
                             &beta,
                             C, CUDA_R_16F, N,
                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            ms /= RUNS;
            printf("%d,FP16,%.4f,%.2f\n", N, ms, flops(N, ms)/1e9);

            cudaFree(A); cudaFree(B); cudaFree(C);
        }

        // ================= INT8 =================
        {
            int8_t  *A, *B;
            int32_t *C;
            CHECK_CUDA(cudaMalloc(&A, N*N*sizeof(int8_t)));
            CHECK_CUDA(cudaMalloc(&B, N*N*sizeof(int8_t)));
            CHECK_CUDA(cudaMalloc(&C, N*N*sizeof(int32_t)));

            for(int i = 0; i < 3; i++)
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha_i,
                             A, CUDA_R_8I, N,
                             B, CUDA_R_8I, N,
                             &beta_i,
                             C, CUDA_R_32I, N,
                             CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            cudaEventRecord(start);
            for(int i = 0; i < RUNS; i++)
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha_i,
                             A, CUDA_R_8I, N,
                             B, CUDA_R_8I, N,
                             &beta_i,
                             C, CUDA_R_32I, N,
                             CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            ms /= RUNS;
            printf("%d,INT8,%.4f,%.2f\n", N, ms, flops(N, ms)/1e9);

            cudaFree(A); cudaFree(B); cudaFree(C);
        }

        // ================= MM83 (software fp8) =================
        {
            // skip sizes that aren't multiples of TILE — mm83 assumes alignment
            if(N % TILE != 0){
                printf("%d,MM83,skip (N not multiple of %d)\n", N, TILE);
                continue;
            }

            f8 *A, *B, *C;
            CHECK_CUDA(cudaMalloc(&A, N*N*sizeof(f8)));
            CHECK_CUDA(cudaMalloc(&B, N*N*sizeof(f8)));
            CHECK_CUDA(cudaMalloc(&C, N*N*sizeof(f8)));

            // init with random fp8 values on host then copy
            f8 *hA = (f8*)malloc(N*N), *hB = (f8*)malloc(N*N);
            for(int i = 0; i < N*N; i++){
                hA[i] = f32tof8((float)rand()/RAND_MAX);
                hB[i] = f32tof8((float)rand()/RAND_MAX);
            }
            cudaMemcpy(A, hA, N*N, cudaMemcpyHostToDevice);
            cudaMemcpy(B, hB, N*N, cudaMemcpyHostToDevice);
            free(hA); free(hB);

            dim3 block(TILE/WPT, TILE/WPT);
            dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

            // warmup
            for(int i = 0; i < 3; i++)
                mm83<<<grid, block>>>(A, B, C, N);
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for(int i = 0; i < RUNS; i++)
                mm83<<<grid, block>>>(A, B, C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            ms /= RUNS;
            printf("%d,MM83,%.4f,%.2f\n", N, ms, flops(N, ms)/1e9);

            cudaFree(A); cudaFree(B); cudaFree(C);
        }
    }

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}