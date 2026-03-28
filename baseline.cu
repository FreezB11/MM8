///@file:baseline_final.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define RUNS 10

#define CHECK_CUDA(x) if((x)!=cudaSuccess){printf("CUDA error at %d\n",__LINE__); return -1;}
#define CHECK_CUBLAS(x) if((x)!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS error at %d\n",__LINE__); return -1;}

int SIZES[] = {16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384};
int NUM_SIZES = sizeof(SIZES)/sizeof(SIZES[0]);

double flops(int N, double ms)
{
    double ops = 2.0 * N * N * N;
    return ops / (ms / 1000.0);
}

int main()
{
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha = 1.0f;
    float beta  = 0.0f;

    int alpha_i = 1;
    int beta_i  = 0;

    printf("N,TYPE,Time(ms),GFLOPS\n");

    for(int s = 0; s < NUM_SIZES; s++)
    {
        int N = SIZES[s];
        printf("\n===== N = %d =====\n", N);

        float ms;

        // ================= FP32 =================
        {
            float *A,*B,*C;
            size_t size = N*N*sizeof(float);

            CHECK_CUDA(cudaMalloc(&A,size));
            CHECK_CUDA(cudaMalloc(&B,size));
            CHECK_CUDA(cudaMalloc(&C,size));

            for(int i=0;i<3;i++)
                cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                            N,N,N,&alpha,A,N,B,N,&beta,C,N);

            cudaEventRecord(start);
            for(int i=0;i<RUNS;i++)
                cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                            N,N,N,&alpha,A,N,B,N,&beta,C,N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&ms,start,stop);
            ms/=RUNS;

            printf("%d,FP32,%.4f,%.2f\n", N, ms, flops(N,ms)/1e9);

            cudaFree(A); cudaFree(B); cudaFree(C);
        }

        // ================= FP16 =================
        {
            __half *A,*B,*C;
            size_t size = N*N*sizeof(__half);

            CHECK_CUDA(cudaMalloc(&A,size));
            CHECK_CUDA(cudaMalloc(&B,size));
            CHECK_CUDA(cudaMalloc(&C,size));

            for(int i=0;i<3;i++)
                cublasGemmEx(handle,
                             CUBLAS_OP_N,CUBLAS_OP_N,
                             N,N,N,
                             &alpha,
                             A,CUDA_R_16F,N,
                             B,CUDA_R_16F,N,
                             &beta,
                             C,CUDA_R_16F,N,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            cudaEventRecord(start);
            for(int i=0;i<RUNS;i++)
                cublasGemmEx(handle,
                             CUBLAS_OP_N,CUBLAS_OP_N,
                             N,N,N,
                             &alpha,
                             A,CUDA_R_16F,N,
                             B,CUDA_R_16F,N,
                             &beta,
                             C,CUDA_R_16F,N,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&ms,start,stop);
            ms/=RUNS;

            printf("%d,FP16,%.4f,%.2f\n", N, ms, flops(N,ms)/1e9);

            cudaFree(A); cudaFree(B); cudaFree(C);
        }

        // ================= INT8 =================
        {
            int8_t *A,*B;
            int32_t *C;

            size_t sizeA = N*N*sizeof(int8_t);
            size_t sizeC = N*N*sizeof(int32_t);

            CHECK_CUDA(cudaMalloc(&A,sizeA));
            CHECK_CUDA(cudaMalloc(&B,sizeA));
            CHECK_CUDA(cudaMalloc(&C,sizeC));

            for(int i=0;i<3;i++)
                cublasGemmEx(handle,
                             CUBLAS_OP_N,CUBLAS_OP_N,
                             N,N,N,
                             &alpha_i,
                             A,CUDA_R_8I,N,
                             B,CUDA_R_8I,N,
                             &beta_i,
                             C,CUDA_R_32I,N,
                             CUDA_R_32I,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            cudaEventRecord(start);
            for(int i=0;i<RUNS;i++)
                cublasGemmEx(handle,
                             CUBLAS_OP_N,CUBLAS_OP_N,
                             N,N,N,
                             &alpha_i,
                             A,CUDA_R_8I,N,
                             B,CUDA_R_8I,N,
                             &beta_i,
                             C,CUDA_R_32I,N,
                             CUDA_R_32I,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&ms,start,stop);
            ms/=RUNS;

            printf("%d,INT8,%.4f,%.2f\n", N, ms, flops(N,ms)/1e9);

            cudaFree(A); cudaFree(B); cudaFree(C);
        }
    }

    cublasDestroy(handle);
    return 0;
}