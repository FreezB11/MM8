///@file: FP8_break.cu
// Shows two paths with the same inputs:
//   BROKEN : raw quantise → FP8 matmul → outputs saturate at 480, step=32
//   FIXED  : scale inputs into FP8 sweet-spot → matmul → scale output back
//
// Pipeline for both:
//   float32 A,B  →  f32 matmul  (ground truth)
//                →  [raw|scaled] quantise  →  GPU FP8 matmul  →  compare
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define N           512
#define TILE3        64
#define WPT           4
#define BLOCK_PRINT   6     // NxN corner to print

typedef uint8_t f8;

// ── FP8 codec ────────────────────────────────────────────────────────────────

__device__ __host__ __forceinline__ f8 f32tof8(float x){
    if(x == 0.0f) return 0;
    int sign = x < 0; if(sign) x = -x;
    int exp; float frac = frexpf(x, &exp);
    frac *= 2; exp--;
    int exp8 = exp + 7;
    int mant = (int)roundf((frac - 1.0f) * 8.0f);
    if(mant == 8){ mant = 0; exp8++; }
    if(exp8 <= 0) return 0;
    if(exp8 >= 15) exp8 = 15;
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

// ── GPU kernel (k3, register-tiled) ──────────────────────────────────────────

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

// ── CPU helpers ───────────────────────────────────────────────────────────────

void matmul_f32(float* A, float* B, float* C, int n){
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++){
            float s = 0.0f;
            for(int k = 0; k < n; k++) s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
}

// BROKEN: raw quantise, no scaling
void quantise_raw(float* src, f8* dst, int n){
    for(int i = 0; i < n*n; i++) dst[i] = f32tof8(src[i]);
}

// FIXED: scale values into [0, fp8_max] before quantising, return scale factor
float quantise_scaled(float* src, f8* dst, int n){
    float fp8_max = f8tof32(0x78);   // 480.0 — the largest FP8 value
    float maxval  = 0.0f;
    for(int i = 0; i < n*n; i++) if(fabsf(src[i]) > maxval) maxval = fabsf(src[i]);
    float scale = maxval / fp8_max;
    for(int i = 0; i < n*n; i++) dst[i] = f32tof8(src[i] / scale);
    return scale;
}

// ── Printers ─────────────────────────────────────────────────────────────────

void printInputBlock(float* A_f32, f8* A_raw, f8* A_sc, float scA,
                     float* B_f32, f8* B_raw, f8* B_sc, float scB,
                     int n, int blk){
    float*  f32s[] = {A_f32, B_f32};
    f8*     raws[] = {A_raw, B_raw};
    f8*     scs[]  = {A_sc,  B_sc};
    float   scv[]  = {scA,   scB};
    const char* nm[] = {"A", "B"};

    for(int m = 0; m < 2; m++){
        printf("\n  ┌── INPUT %s  %dx%d corner ─────────────────────────────────────────────────────────────────────┐\n",
               nm[m], blk, blk);
        printf("  │ cell    │ F32 original  │ raw FP8 dec   │ raw err    │ scaled FP8 dec│ scaled err  │\n");
        printf("  ├─────────┼───────────────┼───────────────┼────────────┼───────────────┼─────────────┤\n");
        for(int i = 0; i < blk; i++){
            for(int j = 0; j < blk; j++){
                float orig  = f32s[m][i*n+j];
                float raw_d = f8tof32(raws[m][i*n+j]);
                float sc_d  = f8tof32(scs[m][i*n+j]) * scv[m];
                printf("  │[%2d,%2d]  │ %13.6f │ %13.6f │ %+10.6f │ %13.6f │ %+11.6f│\n",
                       i, j, orig, raw_d, raw_d-orig, sc_d, sc_d-orig);
            }
            if(i < blk-1)
            printf("  │         │               │               │            │               │             │\n");
        }
        printf("  └─────────┴───────────────┴───────────────┴────────────┴───────────────┴─────────────┘\n");
        printf("  scale_%s = %.6f   (divide input by this → fits FP8, multiply back after)\n", nm[m], scv[m]);
    }
}

void printOutputBlock(float* C_f32, f8* C_raw, f8* C_sc,
                      float scA, float scB, int n, int blk){
    printf("\n  ┌── OUTPUT C  %dx%d corner ───────────────────────────────────────────────────────────────────────┐\n",
           blk, blk);
    printf("  │ cell    │ F32 truth     │ BROKEN dec    │ broken err%% │ FIXED dec     │ fixed err%%  │\n");
    printf("  ├─────────┼───────────────┼───────────────┼─────────────┼───────────────┼─────────────┤\n");
    for(int i = 0; i < blk; i++){
        for(int j = 0; j < blk; j++){
            float truth  = C_f32[i*n+j];
            float broken = f8tof32(C_raw[i*n+j]);
            float fixed_ = f8tof32(C_sc[i*n+j]) * scA * scB;
            float berr   = (fabsf(truth)>1e-6f)?fabsf(broken-truth)/fabsf(truth)*100.f:0.f;
            float ferr   = (fabsf(truth)>1e-6f)?fabsf(fixed_-truth)/fabsf(truth)*100.f:0.f;
            int sat = (C_raw[i*n+j] & 0x7F) == 0x78;
            printf("  │[%2d,%2d]  │ %13.4f │ %11.4f%s  │ %10.3f%%  │ %13.4f │ %10.3f%%  │\n",
                   i, j, truth, broken, sat?"*":" ", berr, fixed_, ferr);
        }
        if(i < blk-1)
        printf("  │         │               │               │             │               │             │\n");
    }
    printf("  └─────────┴───────────────┴───────────────┴─────────────┴───────────────┴─────────────┘\n");
    printf("  * = saturated (clipped to FP8 max 480, true value was ~3x higher)\n");
}

void accuracySummary(float* C_f32, f8* C_f8, float rescale, int n, const char* label){
    double sumAE=0, sumRE=0, sumSQ=0;
    float  maxAE=0, maxRE=0;
    long   sat=0, total=(long)n*n;
    long   histo[8]={};
    const float bounds[]={0.001f,0.01f,0.1f,1.f,10.f,100.f,1000.f};
    for(int i=0;i<total;i++){
        float gt  = C_f32[i];
        float got = f8tof32(C_f8[i]) * rescale;
        float ae  = fabsf(gt-got);
        float re  = (fabsf(gt)>1e-6f)?ae/fabsf(gt):ae;
        sumAE+=ae; sumRE+=re; sumSQ+=(double)ae*ae;
        if(ae>maxAE) maxAE=ae;
        if(re>maxRE) maxRE=re;
        if(rescale==1.f && (C_f8[i]&0x7F)==0x78) sat++;
        int b=7; for(int k=0;k<7;k++) if(ae<bounds[k]){b=k;break;}
        histo[b]++;
    }
    printf("  %-8s │ MAE=%10.4f  RMSE=%10.4f  MRE=%7.3f%%  maxRE=%7.3f%%",
           label,
           (float)(sumAE/total), (float)sqrt(sumSQ/total),
           (float)(sumRE/total)*100.f, maxRE*100.f);
    if(rescale==1.f) printf("  SAT=%ld(%.1f%%)", sat, 100.f*sat/total);
    printf("\n           │ dist: ");
    const char* bk[]={"<0.001","<0.01","<0.1","<1","<10","<100","<1000",">=1000"};
    for(int b=0;b<8;b++) if(histo[b])
        printf("%s:%.1f%%  ", bk[b], 100.f*histo[b]/total);
    printf("\n");
}

float timeKernel(cudaEvent_t s, cudaEvent_t e){ float ms; cudaEventElapsedTime(&ms,s,e); return ms; }

// ── Main ──────────────────────────────────────────────────────────────────────

int main(){
    srand((unsigned)time(NULL));
    size_t fbytes  = (size_t)N*N*sizeof(float);
    size_t i8bytes = (size_t)N*N;

    float *A_f32=(float*)malloc(fbytes), *B_f32=(float*)malloc(fbytes);
    float *C_f32=(float*)malloc(fbytes);

    f8 *A_raw, *B_raw, *C_raw;   // broken path
    f8 *A_sc,  *B_sc,  *C_sc;   // fixed path
    cudaMallocManaged(&A_raw,i8bytes); cudaMallocManaged(&B_raw,i8bytes); cudaMallocManaged(&C_raw,i8bytes);
    cudaMallocManaged(&A_sc, i8bytes); cudaMallocManaged(&B_sc, i8bytes); cudaMallocManaged(&C_sc, i8bytes);

    printf("════════════════════════════════════════════════════════════════\n");
    printf("  %dx%d matrix  |  inputs [1,4]  |  expected output ~%.0f\n",
           N, N, (float)N*2.5f*2.5f);
    printf("  FP8 max = %.1f   step size at that magnitude = 32\n", f8tof32(0x78));
    printf("════════════════════════════════════════════════════════════════\n\n");

    for(int i=0;i<N*N;i++){
        A_f32[i] = 1.f + 3.f*((float)rand()/(float)RAND_MAX);
        B_f32[i] = 1.f + 3.f*((float)rand()/(float)RAND_MAX);
    }

    // ── 1. F32 ground truth ───────────────────────────────────────────────────
    printf("Step 1: float32 matmul (ground truth)\n");
    matmul_f32(A_f32, B_f32, C_f32, N);
    printf("        C[0,0] = %.4f\n\n", C_f32[0]);

    // ── 2a. Raw quantise (broken) ─────────────────────────────────────────────
    printf("Step 2a: raw quantise — no scaling\n");
    quantise_raw(A_f32, A_raw, N);
    quantise_raw(B_f32, B_raw, N);
    printf("         A[0,0]: f32=%.6f → fp8 decoded=%.6f  err=%+.6f\n",
           A_f32[0], f8tof32(A_raw[0]), f8tof32(A_raw[0])-A_f32[0]);

    // ── 2b. Scaled quantise (fixed) ───────────────────────────────────────────
    printf("\nStep 2b: scaled quantise — compress into FP8 sweet-spot\n");
    float scA = quantise_scaled(A_f32, A_sc, N);
    float scB = quantise_scaled(B_f32, B_sc, N);
    printf("         scale_A=%.4f  scale_B=%.4f  output_rescale=%.4f\n", scA, scB, scA*scB);
    printf("         A[0,0]: f32=%.6f → store as FP8(%.6f) → recover %.6f  err=%+.6f\n",
           A_f32[0], A_f32[0]/scA,
           f8tof32(A_sc[0])*scA, f8tof32(A_sc[0])*scA - A_f32[0]);

    // ── 3. GPU runs ───────────────────────────────────────────────────────────
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    dim3 block(TILE3/WPT, TILE3/WPT);
    dim3 grid((N+TILE3-1)/TILE3, (N+TILE3-1)/TILE3);

    cudaEventRecord(start);
    mmul_k3<<<grid,block>>>(A_raw, B_raw, C_raw, N);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    float t_raw = timeKernel(start,stop);

    cudaEventRecord(start);
    mmul_k3<<<grid,block>>>(A_sc, B_sc, C_sc, N);
    cudaEventRecord(stop); cudaDeviceSynchronize();
    float t_sc = timeKernel(start,stop);

    printf("\nStep 3: GPU matmul\n");
    printf("        BROKEN %.3f ms  C[0,0]=%.3f  (truth=%.3f  err=%.1f%%)\n",
           t_raw, f8tof32(C_raw[0]), C_f32[0],
           fabsf(f8tof32(C_raw[0])-C_f32[0])/C_f32[0]*100.f);
    printf("        FIXED  %.3f ms  C[0,0]=%.3f  (truth=%.3f  err=%.1f%%)\n\n",
           t_sc, f8tof32(C_sc[0])*scA*scB, C_f32[0],
           fabsf(f8tof32(C_sc[0])*scA*scB-C_f32[0])/C_f32[0]*100.f);

    // ── 4. Input block ────────────────────────────────────────────────────────
    printf("Step 4: Input quantisation  raw vs scaled  (%dx%d corner)\n", BLOCK_PRINT, BLOCK_PRINT);
    printInputBlock(A_f32, A_raw, A_sc, scA,
                    B_f32, B_raw, B_sc, scB, N, BLOCK_PRINT);

    // ── 5. Output block ───────────────────────────────────────────────────────
    printf("\nStep 5: Output  truth vs BROKEN vs FIXED  (%dx%d corner)\n", BLOCK_PRINT, BLOCK_PRINT);
    printOutputBlock(C_f32, C_raw, C_sc, scA, scB, N, BLOCK_PRINT);

    // ── 6. Accuracy ───────────────────────────────────────────────────────────
    printf("\nStep 6: Full accuracy comparison\n");
    printf("  ─────────┼─────────────────────────────────────────────────────────────────\n");
    accuracySummary(C_f32, C_raw, 1.f,     N, "BROKEN");
    printf("  ─────────┼─────────────────────────────────────────────────────────────────\n");
    accuracySummary(C_f32, C_sc,  scA*scB, N, "FIXED");
    printf("  ─────────┴─────────────────────────────────────────────────────────────────\n");
    printf("\n  Same kernel, same hardware — only difference is the scaling.\n");

    free(A_f32); free(B_f32); free(C_f32);
    cudaFree(A_raw); cudaFree(B_raw); cudaFree(C_raw);
    cudaFree(A_sc);  cudaFree(B_sc);  cudaFree(C_sc);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
