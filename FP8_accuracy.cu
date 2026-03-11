///@file: FP8_accuracy.cu
// Same kernels, but now measures accuracy vs true float32 reference
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define M 512*2
#define TILE  32*4
#define TILE3 64
#define WPT   4

typedef uint8_t f8;

__device__ __host__ __forceinline__ f8 f8mul(f8 a, f8 b){
    int S = (a ^ b) & 0x80;
    int E = ((a >> 3) & 0xF) + ((b >> 3) & 0xF) - 7;
    int m = ((a & 0x7) | 8) * ((b & 0x7) | 8);
    int shift = m >> 7;
    m >>= shift;
    E += shift;
    int mant = (m + 4) >> 3;
    int overflow = mant >> 4;
    mant &= 0xF;
    E += overflow;
    E = (E <= 0) ? 0 : ((E >= 15) ? 15 : E);
    return S | (E << 3) | (mant & 0x7);
}

__device__ __host__ __forceinline__ f8 f32tof8(float x){
    if(x == 0.0f) return 0;
    int sign = x < 0;
    if(sign) x = -x;
    int exp;
    float frac = frexpf(x, &exp);
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

// ── Kernels (unchanged) ────────────────────────────────────────────────────

__global__ void MMul_kernel(f8* A, f8* B, f8* C, int N){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(col >= N || row >= N) return;
    float sum = 0.0f;
    for(int k = 0; k < N; k++)
        sum += f8tof32(A[row*N+k]) * f8tof32(B[k*N+col]);
    C[row*N + col] = f32tof8(sum);
}

__global__ void mmul_k2(f8* A, f8* B, f8* C, int N){
    __shared__ f8 As[TILE][TILE+4];
    __shared__ f8 Bs[TILE][TILE+4];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y*TILE + ty;
    int col = blockIdx.x*TILE + tx;
    float sum = 0.0f;
    for(int t = 0; t < (N+TILE-1)/TILE; t++){
        int a_col = t*TILE+tx, b_row = t*TILE+ty;
        As[ty][tx] = (row  < N && a_col < N) ? A[row*N+a_col] : 0;
        Bs[ty][tx] = (b_row < N && col  < N) ? B[b_row*N+col] : 0;
        __syncthreads();
        #pragma unroll
        for(int k = 0; k < TILE; k++)
            sum += f8tof32(As[ty][k]) * f8tof32(Bs[k][tx]);
        __syncthreads();
    }
    if(row < N && col < N) C[row*N+col] = f32tof8(sum);
}

__global__ void mmul_k3(f8* A, f8* B, f8* C, int N){
    __shared__ f8 As[TILE3][TILE3+4];
    __shared__ f8 Bs[TILE3][TILE3+4];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y*TILE3 + ty*WPT;
    int col0 = blockIdx.x*TILE3 + tx*WPT;
    float sum[WPT][WPT] = {};
    for(int t = 0; t < (N+TILE3-1)/TILE3; t++){
        #pragma unroll
        for(int i = 0; i < WPT; i++) for(int j = 0; j < WPT; j++){
            int ar = blockIdx.y*TILE3+ty*WPT+i, ac = t*TILE3+tx*WPT+j;
            As[ty*WPT+i][tx*WPT+j] = (ar<N&&ac<N)?A[ar*N+ac]:0;
            int br = t*TILE3+ty*WPT+i, bc = blockIdx.x*TILE3+tx*WPT+j;
            Bs[ty*WPT+i][tx*WPT+j] = (br<N&&bc<N)?B[br*N+bc]:0;
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
        if(r < N && c < N) C[r*N+c] = f32tof8(sum[i][j]);
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

void initMat(f8* A, int N){
    for(int i = 0; i < N; i++) A[i] = f32tof8((float)rand()/(float)RAND_MAX);
}

// TRUE ground truth: float32 inputs, float32 accumulation, float32 output
// (inputs are the f32-decoded versions of the FP8 input matrices)
void cpuMatMulF32(f8* A, f8* B, float* C, int N){
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++){
            float sum = 0.0f;
            for(int k = 0; k < N; k++)
                sum += f8tof32(A[i*N+k]) * f8tof32(B[k*N+j]);
            C[i*N+j] = sum;   // exact float32 — no quantisation here
        }
}

// ── Accuracy report vs the float32 reference ──────────────────────────────
void accuracyReport(f8* C_fp8, float* C_f32, int N, const char* label){
    double sumAbsErr  = 0.0;
    double sumRelErr  = 0.0;
    double sumSqErr   = 0.0;
    float  maxAbsErr  = 0.0f;
    float  maxRelErr  = 0.0f;
    long   total      = (long)N*N;
    int    exact      = 0;       // cells where decoded value matches exactly

    // histogram of absolute errors (bucket by order of magnitude)
    long histo[8] = {};  // <0.001, <0.01, <0.1, <1, <10, <100, <1000, >=1000

    for(int i = 0; i < total; i++){
        float g  = f8tof32(C_fp8[i]);
        float gt = C_f32[i];
        float ae = fabsf(g - gt);
        float re = (fabsf(gt) > 1e-9f) ? ae / fabsf(gt) : ae;

        sumAbsErr += ae;
        sumRelErr += re;
        sumSqErr  += (double)ae * ae;
        if(ae > maxAbsErr) maxAbsErr = ae;
        if(re > maxRelErr) maxRelErr = re;
        if(ae == 0.0f) exact++;

        // bucket
        if     (ae < 0.001f) histo[0]++;
        else if(ae < 0.01f)  histo[1]++;
        else if(ae < 0.1f)   histo[2]++;
        else if(ae < 1.0f)   histo[3]++;
        else if(ae < 10.0f)  histo[4]++;
        else if(ae < 100.0f) histo[5]++;
        else if(ae < 1000.f) histo[6]++;
        else                 histo[7]++;
    }

    float mae  = (float)(sumAbsErr / total);
    float mre  = (float)(sumRelErr / total);
    float rmse = (float)sqrt(sumSqErr / total);

    printf("\n══════════════════════════════════════════\n");
    printf("  Accuracy report: %-10s  (%dx%d)\n", label, N, N);
    printf("══════════════════════════════════════════\n");
    printf("  vs TRUE float32 reference\n");
    printf("  (inputs already quantised to FP8 before mul)\n\n");
    printf("  Max absolute error : %.6f\n", maxAbsErr);
    printf("  Mean absolute error: %.6f  (MAE)\n", mae);
    printf("  RMSE               : %.6f\n", rmse);
    printf("  Mean relative error: %.4f%%\n", mre * 100.0f);
    printf("  Max  relative error: %.4f%%\n", maxRelErr * 100.0f);
    printf("  Exact matches      : %d / %ld  (%.2f%%)\n",
           exact, total, 100.0f * exact / total);
    printf("\n  Absolute-error distribution:\n");
    const char* buckets[] = {"<0.001","<0.01","<0.1","<1","<10","<100","<1000",">=1000"};
    for(int b = 0; b < 8; b++)
        printf("    %-8s : %7ld  (%5.2f%%)\n",
               buckets[b], histo[b], 100.0f*histo[b]/total);
}

// ── Also show a few raw values so you can see what's happening ────────────
void showSampleValues(f8* C_fp8, float* C_f32, int N, const char* label){
    printf("\n  Sample values [%s] (first 8 diagonal cells):\n", label);
    printf("  %-6s  %-12s  %-12s  %-12s\n",
           "idx", "FP8-decoded", "F32-exact", "abs-err");
    for(int i = 0; i < 8 && i < N; i++){
        int idx = i*N + i;   // diagonal
        float g  = f8tof32(C_fp8[idx]);
        float gt = C_f32[idx];
        printf("  [%3d,%3d]  %11.5f  %11.5f  %11.6f\n",
               i, i, g, gt, fabsf(g-gt));
    }
}

float timeKernel(cudaEvent_t s, cudaEvent_t e){ float ms; cudaEventElapsedTime(&ms,s,e); return ms; }

// ── Main ──────────────────────────────────────────────────────────────────

int main(){
    srand((unsigned)time(NULL));

    f8    *A, *B, *C1, *C2, *C3;
    float *C_f32;                          // ground truth
    size_t bytes    = (size_t)M * M;
    size_t fbytes   = (size_t)M * M * sizeof(float);

    cudaMallocManaged(&A,     bytes);
    cudaMallocManaged(&B,     bytes);
    cudaMallocManaged(&C1,    bytes);
    cudaMallocManaged(&C2,    bytes);
    cudaMallocManaged(&C3,    bytes);
    C_f32 = (float*)malloc(fbytes);

    initMat(A, M*M);
    initMat(B, M*M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // ── k1 ──
    { dim3 b(16,16), g((M+15)/16,(M+15)/16);
      cudaEventRecord(start);
      MMul_kernel<<<g,b>>>(A,B,C1,M);
      cudaEventRecord(stop); cudaDeviceSynchronize();
      printf("k1 naive:              %.3f ms\n", timeKernel(start,stop)); }

    // ── k2 ──
    { dim3 b(TILE,TILE), g((M+TILE-1)/TILE,(M+TILE-1)/TILE);
      cudaEventRecord(start);
      mmul_k2<<<g,b>>>(A,B,C2,M);
      cudaEventRecord(stop); cudaDeviceSynchronize();
      printf("k2 tiled TILE=%-3d:     %.3f ms\n", TILE, timeKernel(start,stop)); }

    // ── k3 ──
    { dim3 b(TILE3/WPT,TILE3/WPT), g((M+TILE3-1)/TILE3,(M+TILE3-1)/TILE3);
      cudaEventRecord(start);
      mmul_k3<<<g,b>>>(A,B,C3,M);
      cudaEventRecord(stop); cudaDeviceSynchronize();
      printf("k3 reg-tile T=%-2d W=%d:  %.3f ms\n", TILE3, WPT, timeKernel(start,stop)); }

    // ── Ground truth ──
    printf("\nBuilding float32 ground-truth reference...\n");
    cpuMatMulF32(A, B, C_f32, M);

    // ── Accuracy reports ──
    accuracyReport(C1, C_f32, M, "k1-naive");
    showSampleValues(C1, C_f32, M, "k1-naive");

    accuracyReport(C2, C_f32, M, "k2-tiled");
    accuracyReport(C3, C_f32, M, "k3-regtile");

    // ── FP8 quantisation error on the INPUT matrices alone ──
    printf("\n══════════════════════════════════════════\n");
    printf("  FP8 quantisation noise (inputs only)\n");
    printf("══════════════════════════════════════════\n");
    printf("  (how much info was lost just by storing A,B as FP8)\n");
    double qErr = 0; float qMax = 0;
    for(int i = 0; i < M*M; i++){
        float orig = (float)i / (M*M);  // NOTE: initMat already quantised,
        // so instead measure round-trip error on a fresh uniform sweep
        float v = (float)rand()/(float)RAND_MAX;
        float rt = f8tof32(f32tof8(v));
        float e  = fabsf(v - rt);
        qErr += e; if(e > qMax) qMax = e;
    }
    printf("  FP8 round-trip MAE (scalar): %.6f\n", (float)(qErr/(M*M)));
    printf("  FP8 round-trip max err     : %.6f\n", qMax);
    printf("  FP8 has 3 mantissa bits → ~12.5%% spacing per exponent band\n");
    printf("  Accumulated over N=%d dot-product terms, errors add ~√N × quant_step\n", M);

    free(C_f32);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C1); cudaFree(C2); cudaFree(C3);
    return 0;
}
