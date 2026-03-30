///@file:kernel.cuh
///@note this is the kernel file where i will keep the kernel
///      we will keep a pure fp8 mult and that in the matrix mult
///      but we can also do convert the matrix then get value and
///      then convert back to fp8
///         [opt1] fp8 -> fp32 -> fp8     <-- loss can be lower due to fp32 accum
///         [opt2] fp8 -> fp8             <-- loss will be high
/**
Format	Max	    Min (normal)	Precision
FP32	~1e38	~1e-38	        Very high
E4M3	448	    ~1e-2	        Better precision
E5M2	57k	    ~1e-5	        Better range
*/
#include <stdint.h>
typedef uint8_t f8;
#define TILE 64
#define WPT 4

// __device__ __host__ __forceinline__ float f8tof32(f8 x){
//     int sign = (x >> 7) & 1;
//     int exp  = (x >> 3) & 0xF;
//     int mant = x & 0x7;
//     if (exp == 0 && mant == 0) return 0.0f;
//     float val = ldexpf(1.0f + mant * 0.125f, exp - 7);
//     return sign ? -val : val;
// }
__device__ __host__ __forceinline__ float f8tof32(f8 x){
    int sign = (x >> 7) & 1;
    int exp  = (x >> 3) & 0xF;
    int mant = x & 0x7;
    if (exp == 0 && mant == 0) return 0.0f;

    // build IEEE 754 float bits directly — zero math functions
    // float: 1 sign | 8 exp | 23 mant
    // f8 exp bias=7, float bias=127, so float_exp = exp - 7 + 127 = exp + 120
    unsigned int fbits = (sign << 31)
                       | ((exp + 120) << 23)
                       | (mant << 20);   // 3-bit mant → top 3 of 23 mantissa bits
    float val;
    memcpy(&val, &fbits, sizeof(float));
    return val;
}

// __device__ __host__ __forceinline__ f8 f32tof8(float x){
//     if(x==0.0f) return 0;

//     int sign = x < 0;
//     if(sign) x = -x;

//     int exp;
//     float frac = frexpf(x, &exp);
//     // x = frac * 2 ^ exp, frac in [0.5, 1)

//     frac *= 2;
//     exp--;
//     int exp8 = exp + 7;

//     int mant = (int)roundf((frac - 1.0f) * 8.0f);

//     if(mant == 8){
//         mant = 0;
//         exp8++;
//     }
//     if(exp8 <= 0) return 0;
//     if(exp8 >= 15) exp8 = 15;

//     return (sign << 7) | (exp8 << 3) | (mant & 0x7);
// }
__device__ __host__ __forceinline__ f8 f32tof8(float x){
    unsigned int bits;
    memcpy(&bits, &x, sizeof(float));

    int sign  = (bits >> 31) & 1;
    int fexp  = (bits >> 23) & 0xFF;
    // int fmant = (bits >> 23) & 0x7FFFFF;  // not needed, just for clamping

    if(fexp == 0) return (sign << 7);          // zero / denormal → zero
    if(x != x)   return 0;                     // nan → zero

    int exp8  = fexp - 120;                    // 127 - 7 = 120
    int mant8 = (bits >> 20) & 0x7;           // top 3 mantissa bits

    if(exp8 <= 0)  return (sign << 7);
    if(exp8 >= 15) exp8 = 15;

    return (sign << 7) | (exp8 << 3) | mant8;
}

/// @brief this is the version where i cud do all that i know just tiling vector loading and that
/// @param A input mat
/// @param B input mat
/// @param C ouput 
/// @param N assuming it is a square matrix
/// @return 
__global__ void mm83(f8* __restrict__ A, f8* __restrict__ B, f8* __restrict__ C, int N){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y*TILE + ty*WPT;
    int col = blockIdx.x*TILE + tx*WPT;

    float sum[WPT][WPT] = {};

    for(int t = 0; t < (N + TILE - 1)/TILE; t++){

        #pragma unroll
        for(int i = 0; i < WPT; i++){
            // --- A tile: 4 bytes in one load ---
            int ar = blockIdx.y*TILE + ty*WPT + i;
            int ac = t*TILE + tx*WPT;           // base col, WPT=4 cols at once

            uint32_t a4 = (ar < N && ac+3 < N)
                        ? *(uint32_t*)(&A[ar*N + ac])
                        : 0u;

            As[ty*WPT+i][tx*WPT+0] = f8tof32((a4 >>  0) & 0xFF);
            As[ty*WPT+i][tx*WPT+1] = f8tof32((a4 >>  8) & 0xFF);
            As[ty*WPT+i][tx*WPT+2] = f8tof32((a4 >> 16) & 0xFF);
            As[ty*WPT+i][tx*WPT+3] = f8tof32((a4 >> 24) & 0xFF);

            // --- B tile: 4 bytes in one load ---
            int br = t*TILE + ty*WPT + i;
            int bc = blockIdx.x*TILE + tx*WPT;

            uint32_t b4 = (br < N && bc+3 < N)
                        ? *(uint32_t*)(&B[br*N + bc])
                        : 0u;

            Bs[ty*WPT+i][tx*WPT+0] = f8tof32((b4 >>  0) & 0xFF);
            Bs[ty*WPT+i][tx*WPT+1] = f8tof32((b4 >>  8) & 0xFF);
            Bs[ty*WPT+i][tx*WPT+2] = f8tof32((b4 >> 16) & 0xFF);
            Bs[ty*WPT+i][tx*WPT+3] = f8tof32((b4 >> 24) & 0xFF);
        }
        __syncthreads();

        #pragma unroll
        for(int k = 0; k < TILE; k++){
            float a[WPT], b[WPT];
            // #pragma unroll
            // for(int i = 0; i < WPT; i++)
            //     a[i] = As[ty*WPT+i][k];
            // #pragma unroll
            // for(int j = 0; j < WPT; j++)
            //     b[j] = Bs[k][tx*WPT+j];
            #pragma unroll
            for(int i = 0; i < WPT; i++){
                a[i] = As[ty*WPT + i][k];
                b[i] = Bs[k][tx*WPT + i];
            }

            #pragma unroll
            for(int i = 0; i < WPT; i++)
                #pragma unroll
                for(int j = 0; j < WPT; j++)
                    sum[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i = 0; i < WPT; i++)
        #pragma unroll
        for(int j = 0; j < WPT; j++){
            int r = row+i, c = col+j;
            if(r < N && c < N)
                C[r*N+c] = f32tof8(sum[i][j]);
        }
}

__global__ void mm84(f8* __restrict__ A,
                       f8* __restrict__ B,
                       f8* __restrict__ C,
                       int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty * WPT;
    int col = blockIdx.x * TILE + tx * WPT;

    float sum[WPT][WPT] = {0};

    for (int t = 0; t < (N + TILE - 1)/TILE; t++) {

        #pragma unroll
        for (int i = 0; i < WPT; i++) {

            // ================= A LOAD =================
            int ar = row + i;
            int ac = t*TILE + tx*WPT;

            uint32_t a4 = (ar < N && ac+3 < N)
                        ? *(uint32_t*)(&A[ar*N + ac])
                        : 0u;

            #pragma unroll
            for(int k=0;k<4;k++){
                uint32_t x = (a4 >> (k*8)) & 0xFF;

                uint32_t sign = (x & 0x80) << 24;
                uint32_t exp  = ((x >> 3) & 0xF) + 120;
                uint32_t mant = (x & 0x7) << 20;

                uint32_t fbits = sign | (exp << 23) | mant;

                As[ty*WPT + i][tx*WPT + k] = __uint_as_float(fbits);
            }

            // ================= B LOAD =================
            int br = t*TILE + ty*WPT + i;
            int bc = col;

            uint32_t b4 = (br < N && bc+3 < N)
                        ? *(uint32_t*)(&B[br*N + bc])
                        : 0u;

            #pragma unroll
            for(int k=0;k<4;k++){
                uint32_t x = (b4 >> (k*8)) & 0xFF;

                uint32_t sign = (x & 0x80) << 24;
                uint32_t exp  = ((x >> 3) & 0xF) + 120;
                uint32_t mant = (x & 0x7) << 20;

                uint32_t fbits = sign | (exp << 23) | mant;

                Bs[ty*WPT + i][tx*WPT + k] = __uint_as_float(fbits);
            }
        }

        __syncthreads();

        // ================= COMPUTE =================
        #pragma unroll
        for (int k = 0; k < TILE; k++) {

            float a[WPT], b[WPT];

            #pragma unroll
            for (int i = 0; i < WPT; i++) {
                a[i] = As[ty*WPT + i][k];
                b[i] = Bs[k][tx*WPT + i];
            }

            #pragma unroll
            for (int i = 0; i < WPT; i++)
                #pragma unroll
                for (int j = 0; j < WPT; j++)
                    sum[i][j] += a[i] * b[j];
        }

        __syncthreads();
    }

    // ================= STORE (PACKED) =================
    #pragma unroll
    for (int i = 0; i < WPT; i++) {

        int r = row + i;
        int c = col;

        if (r < N && c+3 < N) {

            uint32_t packed = 0;

            #pragma unroll
            for(int j=0;j<4;j++){
                float v = sum[i][j];

                uint32_t bits = __float_as_uint(v);

                uint32_t sign = (bits >> 31) & 1;
                uint32_t exp  = (bits >> 23) & 0xFF;
                uint32_t mant = (bits >> 20) & 0x7;

                int exp8 = exp - 120;

                uint32_t out =
                    (sign << 7) |
                    ((exp8 > 0 ? (exp8 < 15 ? exp8 : 15) : 0) << 3) |
                    mant;

                packed |= (out << (j*8));
            }

            *(uint32_t*)(&C[r*N + c]) = packed;
        }
    }
}

__global__ void mm85(f8* __restrict__ A,
                     f8* __restrict__ B,
                     f8* __restrict__ C,
                     int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty * WPT;
    int col = blockIdx.x * TILE + tx * WPT;

    // ================= 16 scalar accumulators =================
    float s00=0, s01=0, s02=0, s03=0;
    float s10=0, s11=0, s12=0, s13=0;
    float s20=0, s21=0, s22=0, s23=0;
    float s30=0, s31=0, s32=0, s33=0;

    for (int t = 0; t < (N + TILE - 1)/TILE; t++) {

        #pragma unroll
        for (int i = 0; i < WPT; i++) {

            // ================= LOAD A =================
            int ar = row + i;
            int ac = t*TILE + tx*WPT;

            uint32_t a4 = (ar < N && ac+3 < N)
                        ? *(uint32_t*)(&A[ar*N + ac])
                        : 0u;

            #pragma unroll
            for(int k=0;k<4;k++){
                uint32_t x = (a4 >> (k*8)) & 0xFF;

                uint32_t sign = (x & 0x80) << 24;
                uint32_t exp  = ((x >> 3) & 0xF) + 120;
                uint32_t mant = (x & 0x7) << 20;

                As[ty*WPT + i][tx*WPT + k] =
                    __uint_as_float(sign | (exp << 23) | mant);
            }

            // ================= LOAD B =================
            int br = t*TILE + ty*WPT + i;
            int bc = col;

            uint32_t b4 = (br < N && bc+3 < N)
                        ? *(uint32_t*)(&B[br*N + bc])
                        : 0u;

            #pragma unroll
            for(int k=0;k<4;k++){
                uint32_t x = (b4 >> (k*8)) & 0xFF;

                uint32_t sign = (x & 0x80) << 24;
                uint32_t exp  = ((x >> 3) & 0xF) + 120;
                uint32_t mant = (x & 0x7) << 20;

                Bs[ty*WPT + i][tx*WPT + k] =
                    __uint_as_float(sign | (exp << 23) | mant);
            }
        }

        __syncthreads();

        // ================= COMPUTE =================
        #pragma unroll
        for (int k = 0; k < TILE; k++) {

            float a0 = As[ty*WPT + 0][k];
            float a1 = As[ty*WPT + 1][k];
            float a2 = As[ty*WPT + 2][k];
            float a3 = As[ty*WPT + 3][k];

            float b0 = Bs[k][tx*WPT + 0];
            float b1 = Bs[k][tx*WPT + 1];
            float b2 = Bs[k][tx*WPT + 2];
            float b3 = Bs[k][tx*WPT + 3];

            // FMA expansion (fully unrolled)
            s00 += a0*b0; s01 += a0*b1; s02 += a0*b2; s03 += a0*b3;
            s10 += a1*b0; s11 += a1*b1; s12 += a1*b2; s13 += a1*b3;
            s20 += a2*b0; s21 += a2*b1; s22 += a2*b2; s23 += a2*b3;
            s30 += a3*b0; s31 += a3*b1; s32 += a3*b2; s33 += a3*b3;
        }

        __syncthreads();
    }

    // ================= STORE (PACKED) =================
    #pragma unroll
    for (int i = 0; i < WPT; i++) {

        int r = row + i;
        int c = col;

        if (r < N && c+3 < N) {

            float v0 = (i==0)?s00:(i==1)?s10:(i==2)?s20:s30;
            float v1 = (i==0)?s01:(i==1)?s11:(i==2)?s21:s31;
            float v2 = (i==0)?s02:(i==1)?s12:(i==2)?s22:s32;
            float v3 = (i==0)?s03:(i==1)?s13:(i==2)?s23:s33;

            uint32_t p = 0;

            #pragma unroll
            for(int j=0;j<4;j++){
                float v = (j==0)?v0:(j==1)?v1:(j==2)?v2:v3;

                uint32_t bits = __float_as_uint(v);

                uint32_t sign = (bits >> 31) & 1;
                uint32_t exp  = (bits >> 23) & 0xFF;
                uint32_t mant = (bits >> 20) & 0x7;

                int exp8 = exp - 120;

                uint32_t out =
                    (sign << 7) |
                    ((exp8 > 0 ? (exp8 < 15 ? exp8 : 15) : 0) << 3) |
                    mant;

                p |= (out << (j*8));
            }

            *(uint32_t*)(&C[r*N + c]) = p;
        }
    }
}