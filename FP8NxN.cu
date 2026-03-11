///@file: FP8NxN.cu
// i named this to let you&myself know this is for square matrix
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

//global defined
#define M 512/4
#define TILE 32*4 // since this is int8 not float
#define WPT 2// work per thread: each thread outputs wpt*wpt = 2x2 = 4 values

typedef uint8_t f8;
// we will opt the branchless version which should lead to better result
__device__ __host__ __forceinline__ f8 f8mul(f8 a, f8 b){
    int S = (a ^ b) & 0x80; //0x80 in binary is 10000000.
    int E = ((a >> 3) & 0xF) + ((b >> 3) & 0xF) - 7;
    int m = ((a & 0x7) | 8) * ((b & 0x7) | 8); // 8 == 1<<3

    int shift = m >> 7;   // 1 if M >= 128
    m >>= shift;           
    E += shift;

    int mant = (m + 4) >> 3;
    // Handle overflow mantissa
    int overflow = mant >> 4;   // 1 if mant >= 16
    mant &= 0xF;                // clamp to 0..15
    E += overflow;
    E = (E <= 0) ? 0 : ((E >= 15) ? 15 : E);
    return S | (E << 3) | (mant & 0x7);
}

//we need a f32 to f8
///@todo: we have to make this branchless
__device__ __host__ __forceinline__ f8 f32tof8(float x){
    if(x==0.0f) return 0;

    int sign = x < 0;
    if(sign) x = -x;

    int exp;
    float frac = frexpf(x, &exp);
    // x = frac * 2 ^ exp, frac in [0.5, 1)

    frac *= 2;
    exp--;
    int exp8 = exp + 7;

    int mant = (int)roundf((frac - 1.0f) * 8.0f);

    if(mant == 8){
        mant = 0;
        exp8++;
    }
    if(exp8 <= 0) return 0;
    if(exp8 >= 15) exp8 = 15;

    return (sign << 7) | (exp8 << 3) | (mant & 0x7);
}

__device__ __host__ __forceinline__ float f8tof32(f8 x){
    int sign = (x >> 7) & 1;
    int exp  = (x >> 3) & 0xF;
    int mant = x & 0x7;
    if (exp == 0 && mant == 0) return 0.0f;
    float val = ldexpf(1.0f + mant * 0.125f, exp - 7);
    return sign ? -val : val;
}

// first i will write a naive kernel
__global__ void MMul_kernel(f8* A, f8* B, f8* C, int N){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(col >= N || row >= N) return;
    float sum = 0.0f;
    for(int k = 0; k < N; k++){
        sum += f8tof32(f8mul(A[row*N+k], B[k*N + col]));
        // C[row*N + col] = f32tof8(sum);
    }
    C[row*N + col] = f32tof8(sum);
}

// this is the 2nd version of the kernel hoping to optimize this
__global__ void mmul_k2(f8* A, f8* B, f8* C, int N){
    __shared__ f8 As[TILE][TILE];
    __shared__ f8 Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y*TILE + ty;
    int col = blockIdx.x*TILE + tx;

    if(col >= N || row >= N) return;

    float sum = 0.0f;

    for(int t = 0; t < (N + TILE - 1)/TILE; t++){
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;
        As[ty][tx] = (row < N && a_col < N)
                     ? (A[row*N + a_col]) : 0.0f;
        Bs[ty][tx] = (b_row < N && col < N)
                     ? (B[b_row*N + col]) : 0.0f;

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < TILE; k++)
            sum += f8tof32(f8mul(As[ty][k], Bs[k][tx]));

        __syncthreads();
    }
    if(row < N && col < N)
        C[row*N + col] = f32tof8(sum);
}

// 3rd edition/opti
__global__ void mmul_k3(f8* A, f8* B, f8* C, int N){
    __shared__ f8 As[TILE][TILE+1];
    __shared__ f8 Bs[TILE][TILE+1];

    int tx = threadIdx.x, ty = threadIdx.y;

    int row0 = blockIdx.y * TILE + ty * WPT;
    int col0 = blockIdx.x * TILE + tx * WPT;

    float sum[WPT][WPT] = {};

    int num_tiles = (N + TILE - 1)/TILE;

    for(int t = 0; t < num_tiles; t++){

    }
}

void initMat(f8* A, int N){
    float tmp;
    for(int i = 0; i < N; i++){
        tmp = (float)rand() / (float)RAND_MAX;
        A[i] = f32tof8(tmp);
    }
}

// =========================================================================
// CPU matrix multiplication
void cpuMatMul(f8* A, f8* B, f8* C, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            float sum = 0.0f;
            for(int k = 0; k < N; k++){
                sum += f8tof32(f8mul(A[i*N+k], B[k*N + j]));
            }
            C[i*N + j] = f32tof8(sum);  // store back as FP8
        }
    }
}

// Compare GPU vs CPU result
void checkResult(f8* C_gpu, f8* C_cpu, int N){
    int errors = 0;
    float maxError = 0.0f;
    for(int i = 0; i < N*N; i++){
        float gpuVal = f8tof32(C_gpu[i]);
        float cpuVal = f8tof32(C_cpu[i]);
        float err = fabsf(gpuVal - cpuVal);
        if(err > 1e-2f){  // tolerance due to FP8 quantization
            if(errors < 10) // print first few mismatches
                printf("Mismatch at index %d: GPU=%.4f CPU=%.4f\n", i, gpuVal, cpuVal);
            errors++;
        }
        if(err > maxError) maxError = err;
    }
    if(errors == 0)
        printf("GPU and CPU results match!\n");
    else
        printf("Total mismatches: %d, max error: %.4f\n", errors, maxError);
}
// =========================================================================

int main(){
    uint8_t* A = nullptr, *B = nullptr, *C = nullptr;
    f8* C_cpu = nullptr;

    cudaMallocManaged(&A, M*M);
    cudaMallocManaged(&B, M*M);
    cudaMallocManaged(&C, M*M);
    cudaMallocManaged(&C_cpu, M*M);  // CPU reference matrix

    initMat(A, M*M);
    initMat(B, M*M);

    // dim3 grid(16,16);
    // dim3 bloc(8,8);
    dim3 block(16,16);
    dim3 grid( (M + block.x - 1)/block.x, (M + block.y - 1)/block.y );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // MMul_kernel<<<grid,block>>>(A, B, C, M);
    mmul_k2<<<grid,block>>>(A,B,C,M);
    cudaDeviceSynchronize(); // wait for gpu to finish;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    // printf("kernel version1 time: %.3f ms\n", ms);
    printf("kernel version2 time: %.3f ms\n", ms);

    // Run CPU matmul
    cpuMatMul(A, B, C_cpu, M);
    // // Compare
    checkResult(C, C_cpu, M);

    // free CPU reference
    cudaFree(C_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    // should i do a cpu check here
}