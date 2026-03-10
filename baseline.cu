///@file:baseline.cu
// this will be my comparission like the file says
// baseline. and i will try to make fp8 fastr

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math.h>

#define N 4096
#define RUNS 10

#define CHECK_CUDA(x) if((x)!=cudaSuccess){printf("CUDA error\n"); return -1;}
#define CHECK_CUBLAS(x) if((x)!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS error\n"); return -1;}

double flops(double ms)
{
    double ops = 2.0 * N * N * N;
    return ops / (ms / 1000.0);
}

int main()
{
    printf("Matrix size: %d x %d\n", N, N);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *A32,*B32,*C32;
    __half *A16,*B16,*C16;

    size_t size32 = N*N*sizeof(float);
    size_t size16 = N*N*sizeof(__half);

    CHECK_CUDA(cudaMalloc(&A32,size32));
    CHECK_CUDA(cudaMalloc(&B32,size32));
    CHECK_CUDA(cudaMalloc(&C32,size32));

    CHECK_CUDA(cudaMalloc(&A16,size16));
    CHECK_CUDA(cudaMalloc(&B16,size16));
    CHECK_CUDA(cudaMalloc(&C16,size16));

    float alpha=1.0f;
    float beta=0.0f;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\n===== FP32 GEMM =====\n");

    for(int i=0;i<3;i++)
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                    N,N,N,
                    &alpha,
                    A32,N,
                    B32,N,
                    &beta,
                    C32,N);

    cudaEventRecord(start);

    for(int i=0;i<RUNS;i++)
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                    N,N,N,
                    &alpha,
                    A32,N,
                    B32,N,
                    &beta,
                    C32,N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    ms/=RUNS;

    double gflops = flops(ms)/1e9;

    printf("Time: %f ms\n",ms);
    printf("GFLOPS: %.2f\n",gflops);


    printf("\n===== FP16 GEMM (Tensor Core) =====\n");

    for(int i=0;i<3;i++)
        cublasGemmEx(handle,
                     CUBLAS_OP_N,CUBLAS_OP_N,
                     N,N,N,
                     &alpha,
                     A16,CUDA_R_16F,N,
                     B16,CUDA_R_16F,N,
                     &beta,
                     C16,CUDA_R_16F,N,
                     CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaEventRecord(start);

    for(int i=0;i<RUNS;i++)
        cublasGemmEx(handle,
                     CUBLAS_OP_N,CUBLAS_OP_N,
                     N,N,N,
                     &alpha,
                     A16,CUDA_R_16F,N,
                     B16,CUDA_R_16F,N,
                     &beta,
                     C16,CUDA_R_16F,N,
                     CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms,start,stop);

    ms/=RUNS;

    gflops = flops(ms)/1e9;

    printf("Time: %f ms\n",ms);
    printf("GFLOPS: %.2f\n",gflops);


    cudaFree(A32);
    cudaFree(B32);
    cudaFree(C32);

    cudaFree(A16);
    cudaFree(B16);
    cudaFree(C16);

    cublasDestroy(handle);

    return 0;
}

/*
==============================
    RESULTS on RTX3050
==============================
❯ nvcc baseline.cu -lcublas -O3 -o bench
❯ ./bench
Matrix size: 4096 x 4096

===== FP32 GEMM =====
Time: 25.242521 ms
GFLOPS: 5444.74

===== FP16 GEMM (Tensor Core) =====
Time: 7.185613 ms
GFLOPS: 19126.96
*/