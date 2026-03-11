///@file: FP8NxN.cu
// i named this to let you&myself know this is for square matrix
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

//global defined
#define M 512
#define TILE 32 // since this is int8 not float
#define TILE3 64
#define WPT 4// work per thread: each thread outputs wpt*wpt = 2x2 = 4 values

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
        // sum += f8tof32(f8mul(A[row*N+k], B[k*N + col]));
        sum += f8tof32(A[row*N+k]) * f8tof32(B[k*N + col]);
        // C[row*N + col] = f32tof8(sum);
    }
    C[row*N + col] = f32tof8(sum);
}

// this is the 2nd version of the kernel hoping to optimize this
__global__ void mmul_k2(f8* A, f8* B, f8* C, int N){
    __shared__ f8 As[TILE][TILE+4];
    __shared__ f8 Bs[TILE][TILE+4];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y*TILE + ty;
    int col = blockIdx.x*TILE + tx;

    // if(col >= N || row >= N) return;

    float sum = 0.0f;

    for(int t = 0; t < (N + TILE - 1)/TILE; t++){
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;
        As[ty][tx] = (row  < N && a_col < N) ? A[row*N  + a_col] : 0;
        Bs[ty][tx] = (b_row < N && col  < N) ? B[b_row*N + col ] : 0;
        __syncthreads();

        #pragma unroll
        for(int k = 0; k < TILE; k++)
            sum += f8tof32(As[ty][k]) * f8tof32(Bs[k][tx]);

        __syncthreads();
    }
    if(row < N && col < N)
        C[row*N + col] = f32tof8(sum);
}


// 3rd edition/opti
__global__ void mmul_k3(f8* A, f8* B, f8* C, int N){
    __shared__ f8 As[TILE3][TILE3+4];
    __shared__ f8 Bs[TILE3][TILE3+4];

    int tx = threadIdx.x, ty = threadIdx.y;

    int row0 = blockIdx.y * TILE3 + ty * WPT;
    int col0 = blockIdx.x * TILE3 + tx * WPT;

    float sum[WPT][WPT] = {};

    int num_tiles = (N + TILE3 - 1)/TILE3;

    for(int t = 0; t < num_tiles; t++){
        #pragma unroll
        for(int i = 0; i < WPT; i++){
            #pragma unroll
            for(int j = 0; j < WPT; j++){
                int ar = blockIdx.y * TILE3 + ty * WPT + i;
                int ac = t * TILE3           + tx * WPT + j;
                As[ty*WPT + i][tx*WPT + j] =
                    (ar < N && ac < N) ? A[ar*N + ac] : 0;

                int br = t * TILE3           + ty * WPT + i;
                int bc = blockIdx.x * TILE3 + tx * WPT + j;
                Bs[ty*WPT + i][tx*WPT + j] =
                    (br < N && bc < N) ? B[br*N + bc] : 0;
            }
        }
        __syncthreads();

        // ── Compute: stream through k, reuse WPT rows of A and WPT cols of B ─
        // The key: a[] and b[] stay in registers — no repeated shared mem reads
        #pragma unroll
        for(int k = 0; k < TILE3; k++){
            float a[WPT], b[WPT];

            #pragma unroll
            for(int i = 0; i < WPT; i++)
                a[i] = f8tof32(As[ty*WPT + i][k]);  // WPT rows of A for this thread

            #pragma unroll
            for(int j = 0; j < WPT; j++)
                b[j] = f8tof32(Bs[k][tx*WPT + j]);  // WPT cols of B for this thread

            // Outer product: WPT×WPT = 16 FMAs per k iteration, all in registers
            #pragma unroll
            for(int i = 0; i < WPT; i++)
                #pragma unroll
                for(int j = 0; j < WPT; j++)
                    sum[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    // ── Store 16 results back ─────────────────────────────────────────────────
    #pragma unroll
    for(int i = 0; i < WPT; i++){
        #pragma unroll
        for(int j = 0; j < WPT; j++){
            int r = row0 + i, c = col0 + j;
            if(r < N && c < N)
                C[r*N + c] = f32tof8(sum[i][j]);
        }
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
                sum += f8tof32(A[i*N+k]) * f8tof32(B[k*N + j]);
            }
            C[i*N + j] = f32tof8(sum);  // store back as FP8
        }
    }
}

void checkResult(f8* C_gpu, f8* C_cpu, int N, const char* label){
    int errors = 0;
    float maxErr = 0.0f;
    for(int i = 0; i < N*N; i++){
        float g = f8tof32(C_gpu[i]), c = f8tof32(C_cpu[i]);
        float e = fabsf(g - c);
        if(e > 1e-2f){
            if(errors < 5) printf("  [%s] idx %d: GPU=%.4f CPU=%.4f\n", label, i, g, c);
            errors++;
        }
        if(e > maxErr) maxErr = e;
    }
    if(errors == 0) printf("[%s] PASS  maxErr=%.4f\n", label, maxErr);
    else            printf("[%s] FAIL  mismatches=%d  maxErr=%.4f\n", label, errors, maxErr);
}

float timeKernel(cudaEvent_t s, cudaEvent_t e){ float ms; cudaEventElapsedTime(&ms,s,e); return ms; }

// =========================================================================

int main(){
    srand((unsigned)time(NULL));

    f8 *A, *B, *C1, *C2, *C3, *C_cpu;
    size_t bytes = (size_t)M * M;
    cudaMallocManaged(&A,     bytes);
    cudaMallocManaged(&B,     bytes);
    cudaMallocManaged(&C1,    bytes);
    cudaMallocManaged(&C2,    bytes);
    cudaMallocManaged(&C3,    bytes);
    cudaMallocManaged(&C_cpu, bytes);

    initMat(A, M*M);
    initMat(B, M*M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ── k1 naive ──
    {
        dim3 block(16, 16);
        dim3 grid((M + 15)/16, (M + 15)/16);
        cudaEventRecord(start);
        MMul_kernel<<<grid, block>>>(A, B, C1, M);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        printf("k1 naive:           %.3f ms\n", timeKernel(start, stop));
    }

    // ── k2 tiled (TILE=32) ──
    // Block MUST be (TILE, TILE) — one thread per shared-mem cell
    {
        dim3 block(TILE, TILE);                        // 32×32 = 1024 threads
        dim3 grid((M + TILE-1)/TILE, (M + TILE-1)/TILE);
        cudaEventRecord(start);
        mmul_k2<<<grid, block>>>(A, B, C2, M);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        printf("k2 tiled TILE=%-3d:  %.3f ms\n", TILE, timeKernel(start, stop));
    }

    // ── k3 register-tiled (TILE3=64, WPT=4) ──
    {
        dim3 block(TILE3/WPT, TILE3/WPT);             // 16×16 = 256 threads
        dim3 grid((M + TILE3-1)/TILE3, (M + TILE3-1)/TILE3);
        cudaEventRecord(start);
        mmul_k3<<<grid, block>>>(A, B, C3, M);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        printf("k3 reg-tile T=%-2d W=%d: %.3f ms\n", TILE3, WPT, timeKernel(start, stop));
    }

    // ── CPU reference + correctness ──
    printf("\nRunning CPU reference...\n");
    cpuMatMul(A, B, C_cpu, M);
    checkResult(C1, C_cpu, M, "k1");
    checkResult(C2, C_cpu, M, "k2");
    checkResult(C3, C_cpu, M, "k3");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B);
    cudaFree(C1); cudaFree(C2); cudaFree(C3); cudaFree(C_cpu);
}