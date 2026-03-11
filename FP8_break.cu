///@file: FP8_break.cu
// Pipeline: generate real float32 → f32 matmul (truth)
//           then quantise inputs to FP8 → fp8 matmul → compare
//
// Input range [1,4] pushes outputs to ~3000 — FP8 max is ~480 → saturates hard
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

// ── Tune these ──────────────────────────────────────────────────────────────
#define N     512         // matrix size — big enough to saturate FP8
#define TILE3  64
#define WPT    4
#define BLOCK_PRINT 8     // print an 8×8 corner block

typedef uint8_t f8;

// ── FP8 codec ───────────────────────────────────────────────────────────────

__device__ __host__ __forceinline__ f8 f32tof8(float x){
    if(x == 0.0f) return 0;
    int sign = x < 0; if(sign) x = -x;
    int exp; float frac = frexpf(x, &exp);
    frac *= 2; exp--;
    int exp8 = exp + 7;
    int mant = (int)roundf((frac - 1.0f) * 8.0f);
    if(mant == 8){ mant = 0; exp8++; }
    if(exp8 <= 0) return 0;
    if(exp8 >= 15) exp8 = 15;           // saturate — no inf/nan in this format
    return (sign << 7) | (exp8 << 3) | (mant & 0x7);
}

__device__ __host__ __forceinline__ float f8tof32(f8 x){
    int sign = (x >> 7) & 1;
    int exp  = (x >> 3) & 0xF;
    int mant = x & 0x7;
    if(exp == 0 && mant == 0) return 0.0f;
    float val = ldexpf(1.0f + mant * 0.125f, exp - 7);
    return sign ? -val : val;
}

// ── Best GPU kernel (k3, register-tiled) ────────────────────────────────────

__global__ void mmul_k3(f8* A, f8* B, f8* C, int n){
    __shared__ f8 As[TILE3][TILE3+4];
    __shared__ f8 Bs[TILE3][TILE3+4];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y*TILE3 + ty*WPT;
    int col0 = blockIdx.x*TILE3 + tx*WPT;
    float sum[WPT][WPT] = {};

    for(int t = 0; t < (n+TILE3-1)/TILE3; t++){
        #pragma unroll
        for(int i = 0; i < WPT; i++) for(int j = 0; j < WPT; j++){
            int ar = blockIdx.y*TILE3+ty*WPT+i, ac = t*TILE3+tx*WPT+j;
            As[ty*WPT+i][tx*WPT+j] = (ar<n&&ac<n)?A[ar*n+ac]:0;
            int br = t*TILE3+ty*WPT+i, bc = blockIdx.x*TILE3+tx*WPT+j;
            Bs[ty*WPT+i][tx*WPT+j] = (br<n&&bc<n)?B[br*n+bc]:0;
        }
        __syncthreads();
        #pragma unroll
        for(int k = 0; k < TILE3; k++){
            float a[WPT], b[WPT];
            #pragma unroll
            for(int i = 0; i < WPT; i++) a[i] = f8tof32(As[ty*WPT+i][k]);
            #pragma unroll
            for(int j = 0; j < WPT; j++) b[j] = f8tof32(Bs[k][tx*WPT+j]);
            #pragma unroll
            for(int i = 0; i < WPT; i++)
                #pragma unroll
                for(int j = 0; j < WPT; j++)
                    sum[i][j] += a[i]*b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i = 0; i < WPT; i++) for(int j = 0; j < WPT; j++){
        int r = row0+i, c = col0+j;
        if(r < n && c < n) C[r*n+c] = f32tof8(sum[i][j]);
    }
}

// ── CPU helpers ─────────────────────────────────────────────────────────────

// Step 1: true float32 matmul — no quantisation anywhere
void matmul_f32(float* A, float* B, float* C, int n){
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++){
            float s = 0.0f;
            for(int k = 0; k < n; k++) s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
}

// Step 2: quantise float32 matrices to FP8
void quantise(float* src, f8* dst, int n){
    for(int i = 0; i < n*n; i++) dst[i] = f32tof8(src[i]);
}

// ── Pretty block printer ─────────────────────────────────────────────────────

void printBlock(float* C_f32, f8* C_f8, int n, int blk){
    printf("\n  ┌─────────────────────────────────────────────────────────────"
           "─────────────────────────────────────────────────────────────────┐\n");
    printf("  │  %3dx%-3d top-left block comparison"
           "                                                                  │\n", blk, blk);
    printf("  ├──────────┬──────────────┬──────────────┬──────────────┬──────┤\n");
    printf("  │  cell    │  F32 (true)  │  FP8 decoded │  abs error   │  rel %%│\n");
    printf("  ├──────────┼──────────────┼──────────────┼──────────────┼──────┤\n");

    for(int i = 0; i < blk; i++){
        for(int j = 0; j < blk; j++){
            float gt  = C_f32[i*n+j];
            float got = f8tof32(C_f8[i*n+j]);
            float ae  = fabsf(gt - got);
            float re  = (fabsf(gt) > 1e-6f) ? ae/fabsf(gt)*100.0f : 0.0f;
            // flag saturated cells
            int sat = (C_f8[i*n+j] & 0x7F) == 0x78;  // exp8=15, mant=0 = max positive
            printf("  │ [%3d,%3d] │ %12.3f │ %12.3f │ %12.3f │%5.1f%%%s│\n",
                   i, j, gt, got, ae, re, sat?" SAT":"    ");
        }
        if(i < blk-1)
            printf("  │           │              │              │              │      │\n");
    }
    printf("  └──────────┴──────────────┴──────────────┴──────────────┴──────┘\n");
}

// ── Input block printer ──────────────────────────────────────────────────────
// Shows both A and B: original f32 value, quantised f8 decoded, and the loss

void printInputBlock(float* A_f32, f8* A_f8,
                     float* B_f32, f8* B_f8, int n, int blk){
    printf("\n  ┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  INPUT MATRIX A  —  %dx%d top-left block                                        │\n", blk, blk);
    printf("  ├──────────┬──────────────┬──────────────┬────────────┐                           │\n");
    printf("  │  cell    │  F32 (true)  │  FP8 decoded │  quant err │                           │\n");
    printf("  ├──────────┼──────────────┼──────────────┼────────────┘                           │\n");
    for(int i = 0; i < blk; i++){
        for(int j = 0; j < blk; j++){
            float orig = A_f32[i*n+j];
            float quant = f8tof32(A_f8[i*n+j]);
            printf("  │ [%3d,%3d] │ %12.6f │ %12.6f │ %+.6f\n",
                   i, j, orig, quant, quant - orig);
        }
        if(i < blk-1)
        printf("  │           │              │              │            \n");
    }
    printf("  └──────────┴──────────────┴──────────────┴────────────\n");

    printf("\n  ┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  INPUT MATRIX B  —  %dx%d top-left block                                        │\n", blk, blk);
    printf("  ├──────────┬──────────────┬──────────────┬────────────┐                           │\n");
    printf("  │  cell    │  F32 (true)  │  FP8 decoded │  quant err │                           │\n");
    printf("  ├──────────┼──────────────┼──────────────┼────────────┘                           │\n");
    for(int i = 0; i < blk; i++){
        for(int j = 0; j < blk; j++){
            float orig = B_f32[i*n+j];
            float quant = f8tof32(B_f8[i*n+j]);
            printf("  │ [%3d,%3d] │ %12.6f │ %12.6f │ %+.6f\n",
                   i, j, orig, quant, quant - orig);
        }
        if(i < blk-1)
        printf("  │           │              │              │            \n");
    }
    printf("  └──────────┴──────────────┴──────────────┴────────────\n");
}

// ── Accuracy summary ─────────────────────────────────────────────────────────

void accuracySummary(float* C_f32, f8* C_f8, int n){
    double sumAE = 0, sumRE = 0, sumSQ = 0;
    float  maxAE = 0, maxRE = 0;
    long   sat   = 0;
    long   total = (long)n*n;

    long histo[8] = {};
    const float bounds[] = {0.001f,0.01f,0.1f,1.f,10.f,100.f,1000.f};

    for(int i = 0; i < total; i++){
        float gt  = C_f32[i];
        float got = f8tof32(C_f8[i]);
        float ae  = fabsf(gt - got);
        float re  = (fabsf(gt) > 1e-6f) ? ae/fabsf(gt) : ae;

        sumAE += ae; sumRE += re; sumSQ += (double)ae*ae;
        if(ae > maxAE) maxAE = ae;
        if(re > maxRE) maxRE = re;
        if((C_f8[i] & 0x7F) == 0x78) sat++;  // saturated to FP8 max

        int b = 7;
        for(int k = 0; k < 7; k++) if(ae < bounds[k]){ b=k; break; }
        histo[b]++;
    }

    printf("\n════════════════════════════════════════════════════════\n");
    printf("  ACCURACY  F32 inputs → F32 matmul  vs  F8 inputs → F8 matmul\n");
    printf("  Matrix: %dx%d   Input range: [1, 4]   Expected output: ~%.0f\n",
           n, n, (float)n * 2.5f * 2.5f);
    printf("  FP8 max representable value: %.3f  (exp8=15)\n",
           f8tof32(0x78));   // 0x78 = 0 11110 000 = max positive
    printf("════════════════════════════════════════════════════════\n");
    printf("  Max absolute error  : %12.3f\n", maxAE);
    printf("  Mean absolute error : %12.3f\n", (float)(sumAE/total));
    printf("  RMSE                : %12.3f\n", (float)sqrt(sumSQ/total));
    printf("  Mean relative error :   %.4f%%\n", (float)(sumRE/total)*100.0f);
    printf("  Max  relative error :   %.4f%%\n", maxRE*100.0f);
    printf("  Saturated cells     : %ld / %ld  (%.2f%%) ← clipped to FP8 max\n",
           sat, total, 100.0f*sat/total);
    printf("\n  Absolute-error distribution:\n");
    const char* labels[] = {"<0.001","<0.01","<0.1","<1","<10","<100","<1000",">=1000"};
    for(int b = 0; b < 8; b++)
        printf("    %-8s : %8ld  (%5.2f%%)\n",
               labels[b], histo[b], 100.0f*histo[b]/total);
}

float timeKernel(cudaEvent_t s, cudaEvent_t e){ float ms; cudaEventElapsedTime(&ms,s,e); return ms; }

// ── Main ─────────────────────────────────────────────────────────────────────

int main(){
    srand((unsigned)time(NULL));

    size_t fbytes = (size_t)N*N*sizeof(float);
    size_t i8bytes = (size_t)N*N;

    // ── 1. Allocate ──────────────────────────────────────────────────────────
    float *A_f32 = (float*)malloc(fbytes);   // original float32 inputs
    float *B_f32 = (float*)malloc(fbytes);
    float *C_f32 = (float*)malloc(fbytes);   // ground truth output

    f8 *A_f8, *B_f8, *C_f8;                  // quantised versions
    cudaMallocManaged(&A_f8, i8bytes);
    cudaMallocManaged(&B_f8, i8bytes);
    cudaMallocManaged(&C_f8, i8bytes);

    // ── 2. Generate float32 inputs in [1, 4] ─────────────────────────────────
    //       outputs ≈ N * mean(a)*mean(b) = 512 * 2.5 * 2.5 = 3200
    //       FP8 max ≈ 480  →  most cells will saturate
    printf("Generating %dx%d float32 inputs in range [1, 4]...\n", N, N);
    printf("Expected output magnitude: ~%.0f\n", (float)N * 2.5f * 2.5f);
    printf("FP8 max value: %.3f  (exp8=15, mant=0)\n\n", f8tof32(0x78));

    for(int i = 0; i < N*N; i++){
        A_f32[i] = 1.0f + 3.0f * ((float)rand()/(float)RAND_MAX);  // [1,4]
        B_f32[i] = 1.0f + 3.0f * ((float)rand()/(float)RAND_MAX);
    }

    // ── 3. Ground truth: float32 matmul ─────────────────────────────────────
    printf("Step 1: float32 matmul (ground truth)...\n");
    matmul_f32(A_f32, B_f32, C_f32, N);
    printf("        C_f32 sample [0,0] = %.3f\n\n", C_f32[0]);

    // ── 4. Quantise inputs to FP8 ────────────────────────────────────────────
    printf("Step 2: quantise A,B to FP8...\n");
    quantise(A_f32, A_f8, N);
    quantise(B_f32, B_f8, N);
    // show quantisation loss on a sample value
    printf("        A[0,0]: f32=%.6f  →  f8=%.6f  (lost %.6f)\n\n",
           A_f32[0], f8tof32(A_f8[0]), fabsf(A_f32[0]-f8tof32(A_f8[0])));

    // ── 5. FP8 matmul on GPU ─────────────────────────────────────────────────
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    dim3 block(TILE3/WPT, TILE3/WPT);
    dim3 grid((N+TILE3-1)/TILE3, (N+TILE3-1)/TILE3);

    cudaEventRecord(start);
    mmul_k3<<<grid, block>>>(A_f8, B_f8, C_f8, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    printf("Step 3: FP8 matmul on GPU... %.3f ms\n", timeKernel(start,stop));
    printf("        C_f8[0,0] decoded = %.3f  (true = %.3f)\n\n",
           f8tof32(C_f8[0]), C_f32[0]);

    // ── 6. Block printout ────────────────────────────────────────────────────
    printf("Step 4: Input quantisation — what FP8 does to A and B (%dx%d corner):\n", BLOCK_PRINT, BLOCK_PRINT);
    printInputBlock(A_f32, A_f8, B_f32, B_f8, N, BLOCK_PRINT);

    printf("\nStep 5: Output comparison — F32 truth vs FP8 result (%dx%d corner):\n", BLOCK_PRINT, BLOCK_PRINT);
    printBlock(C_f32, C_f8, N, BLOCK_PRINT);

    // ── 7. Full accuracy report ──────────────────────────────────────────────
    printf("\nStep 6: Full accuracy report:\n");
    accuracySummary(C_f32, C_f8, N);

    // ── 8. Bonus: show what range would NOT break FP8 ────────────────────────
    printf("\n────────────────────────────────────────────────────────\n");
    printf("  For reference — what inputs keep outputs inside FP8 range?\n");
    float fp8_max = f8tof32(0x78);
    float safe_mean = sqrtf(fp8_max / (float)N);
    printf("  FP8 max = %.3f,  N = %d\n", fp8_max, N);
    printf("  Need: mean(a)*mean(b) < %.3f/N = %.5f\n", fp8_max, fp8_max/N);
    printf("  Safe uniform input range ≈ [0, %.3f]\n", safe_mean * 2.0f);
    printf("────────────────────────────────────────────────────────\n");

    free(A_f32); free(B_f32); free(C_f32);
    cudaFree(A_f8); cudaFree(B_f8); cudaFree(C_f8);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
