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

#define TILE 64
#define WPT  4

// cp.async intrinsics — these are PTX wrappers
__device__ __forceinline__
void cp_async4(void* dst, const void* src) {
    // copies 16 bytes (float4) asynchronously global → shared
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
           "l"((uint64_t)src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait1() {
    // wait until at most 1 group is still in flight
    // (so the group we issued 2 iterations ago is guaranteed done)
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

__global__ void mm87(f8* __restrict__ A,
                     f8* __restrict__ B,
                     f8* __restrict__ C,
                     int N)
{
    // Double-buffered Bs — buf 0 and buf 1 live back-to-back
    // As stays small from mm86 (WPT rows × CHUNK cols)
    __shared__ float Bs[2][TILE][TILE];   // two full Bs tiles
    __shared__ float As[WPT][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty * WPT;
    int col = blockIdx.x * TILE + tx * WPT;

    int num_tiles = (N + TILE - 1) / TILE;

    float s00=0,s01=0,s02=0,s03=0;
    float s10=0,s11=0,s12=0,s13=0;
    float s20=0,s21=0,s22=0,s23=0;
    float s30=0,s31=0,s32=0,s33=0;

    // ── Prologue: issue load for tile 0 into buf 0 ───────────────────────
    // Each thread is responsible for loading its share of Bs
    // With float4 (16 bytes = 4 floats), but our Bs is float after decode...
    // We load raw f8 bytes and decode — so each thread loads 4 bytes (uint32_t)
    // then decodes into 4 floats in shared mem.
    //
    // cp.async needs the destination to be in shared mem and copies
    // from global directly — BUT it copies raw bytes, so we can only
    // use it cleanly for f8→f8 (raw copy) then decode separately.
    // The cleanest approach: cp.async the raw f8 bytes into a staging
    // area, then decode to float in a separate pass.

    // Let's use a raw staging buffer for the async copy:
    __shared__ uint32_t Bs_raw[2][TILE * TILE / 4];  // f8 bytes packed

    // Thread linearization: blockDim = (TILE/WPT, TILE/WPT) = (16,16)
    // Total threads = 256, total uint32_t to load per Bs = TILE*TILE/4 = 1024
    // So each thread loads 1024/256 = 4 uint32_t values per tile
    int tid = ty * (TILE/WPT) + tx;   // 0..255

    auto issue_Bs_load = [&](int t, int buf) {
        if (t >= num_tiles) return;
        // Each thread loads 4 consecutive uint32_t (16 bytes total → one cp.async.ca 16B)
        int base = t * TILE * N;   // row-major base for tile t of B
        // Thread tid covers 4 packed words = 16 consecutive f8 bytes
        // Map tid to (row, col) in the TILE×TILE Bs region:
        // Each uint32_t = 4 f8 bytes = 4 consecutive columns
        // So thread tid → rows [tid/16 .. ] and cols [tid%16 * 4 ..]
        // Simpler: treat Bs as flat TILE*TILE f8 bytes, thread i loads bytes [i*16..(i+1)*16)
        // but B is row-major in global so we need 2D indexing.
        
        // Let's do it row by row: each thread loads one uint32_t per row it owns
        // (same as mm85's B load pattern), just issued as cp.async
        #pragma unroll
        for (int i = 0; i < WPT; i++) {
            int br = t * TILE + ty * WPT + i;
            int bc = col;   // 4 consecutive cols
            if (br < N && bc + 3 < N) {
                // dst: Bs_raw[buf][ (ty*WPT+i)*(TILE/4) + tx ]
                uint32_t* dst = &Bs_raw[buf][(ty*WPT+i) * (TILE/WPT) + tx];
                uint32_t* src = (uint32_t*)&B[br * N + bc];
                // 4-byte async copy
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4;\n"
                    :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
                       "l"((uint64_t)src)
                );
            }
        }
        cp_async_commit();
    };

    auto decode_Bs = [&](int buf) {
        // Decode raw f8 bytes into float Bs[buf]
        #pragma unroll
        for (int i = 0; i < WPT; i++) {
            uint32_t packed = Bs_raw[buf][(ty*WPT+i) * (TILE/WPT) + tx];
            #pragma unroll
            for (int k = 0; k < WPT; k++) {
                uint32_t x = (packed >> (k*8)) & 0xFF;
                Bs[buf][ty*WPT+i][tx*WPT+k] = __uint_as_float(
                    ((x&0x80)<<24) | ((((x>>3)&0xF)+120)<<23) | ((x&0x7)<<20));
            }
        }
    };

    // Issue tile 0 load
    issue_Bs_load(0, 0);

    // ── Main loop ────────────────────────────────────────────────────────
    for (int t = 0; t < num_tiles; t++) {
        int cur = t & 1;       // buffer holding current tile's Bs
        int nxt = cur ^ 1;     // buffer we'll load next tile into

        // Issue next tile's load BEFORE waiting for current
        // (so it runs in the background during compute)
        issue_Bs_load(t + 1, nxt);

        // Wait: allow at most 1 group in flight (so group t-1 = cur is done)
        cp_async_wait1();
        __syncthreads();

        // Decode raw f8 → float for current buffer
        decode_Bs(cur);
        __syncthreads();

        // ── Load As and compute (same sliding chunk pattern as mm86) ──────
        #pragma unroll
        for (int i = 0; i < WPT; i++) {
            int ar = row + i;
            int ac = t * TILE + tx * WPT;
            uint32_t a4 = (ar < N && ac + 3 < N)
                        ? *(uint32_t*)(&A[ar * N + ac]) : 0u;
            #pragma unroll
            for (int k = 0; k < WPT; k++) {
                uint32_t x = (a4 >> (k*8)) & 0xFF;
                As[i][tx*WPT+k] = __uint_as_float(
                    ((x&0x80)<<24) | ((((x>>3)&0xF)+120)<<23) | ((x&0x7)<<20));
            }
        }
        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            float a0 = As[0][k], a1 = As[1][k],
                  a2 = As[2][k], a3 = As[3][k];

            float b0 = Bs[cur][k][tx*WPT+0], b1 = Bs[cur][k][tx*WPT+1],
                  b2 = Bs[cur][k][tx*WPT+2], b3 = Bs[cur][k][tx*WPT+3];

            s00+=a0*b0; s01+=a0*b1; s02+=a0*b2; s03+=a0*b3;
            s10+=a1*b0; s11+=a1*b1; s12+=a1*b2; s13+=a1*b3;
            s20+=a2*b0; s21+=a2*b1; s22+=a2*b2; s23+=a2*b3;
            s30+=a3*b0; s31+=a3*b1; s32+=a3*b2; s33+=a3*b3;
        }

        __syncthreads();
    }

    // Epilogue: drain any remaining async ops
    cp_async_wait_all();

    // ── Store ─────────────────────────────────────────────────────────────
    #pragma unroll
    for (int i = 0; i < WPT; i++) {
        int r = row + i, c = col;
        if (r < N && c + 3 < N) {
            float v[4] = {
                (i==0)?s00:(i==1)?s10:(i==2)?s20:s30,
                (i==0)?s01:(i==1)?s11:(i==2)?s21:s31,
                (i==0)?s02:(i==1)?s12:(i==2)?s22:s32,
                (i==0)?s03:(i==1)?s13:(i==2)?s23:s33
            };
            uint32_t p = 0;
            #pragma unroll
            for (int j = 0; j < WPT; j++) {
                uint32_t bits = __float_as_uint(v[j]);
                int exp8 = (int)((bits>>23)&0xFF) - 120;
                p |= ((((bits>>31)<<7)|((exp8>0?(exp8<15?exp8:15):0)<<3)|((bits>>20)&0x7)) << (j*8));
            }
            *(uint32_t*)(&C[r*N+c]) = p;
        }
    }
}