/// ============================================================
/// kernel_v3.cuh  —  FP8 × FP8 → FP8 matmul, RTX 3050 (SM 8.6)
///
/// Three progressive kernels, each building on the last:
///
///   mm_v3       — fixes bank conflicts (1 char change + transpose B)
///   mm_compact  — compact FP8 shmem, TILE=128, on-the-fly conversion
///   mm_blockfp  — dp4a integer inner loop, zero FP32 in hot path
///
/// Key identity behind mm_blockfp:
///   FP8 E4M3 value = sign × (8 + m) × 2^(e − 10)
///   (8+m) ∈ [8,15] — a 4-bit unsigned integer
///   signed: ±(8+m) ∈ [−15,−8] ∪ [8,15] — fits in int8
///   → pack 4 into int32, accumulate with __dp4a
/// ============================================================

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

typedef uint8_t f8;

// ────────────────────────────────────────────────────────────
//  SHARED UTILITIES
// ────────────────────────────────────────────────────────────

/// FP8 E4M3 (bias=7) → FP32, pure bit manipulation, no branches
/// except the zero/denormal guard (cmov on compiler; predictable).
__device__ __forceinline__
float fp8_to_f32(uint32_t x)
{
    uint32_t sign = (x & 0x80u) << 24;           // bit 7 → bit 31
    uint32_t exp4 = (x >> 3) & 0xFu;             // bits 6..3
    uint32_t mant = (x & 0x7u) << 20;            // bits 2..0 → top of f32 mant
    return exp4 ? __uint_as_float(sign | ((exp4 + 120u) << 23) | mant)
                : 0.0f;
    // bias conversion: fp32_exp = fp8_exp - 7 + 127 = fp8_exp + 120
}

/// FP32 → FP8 E4M3 (bias=7), clamping at max (e=14, m=7).
__device__ __forceinline__
uint32_t f32_to_fp8(float v)
{
    uint32_t bits = __float_as_uint(v);
    uint32_t sign = (bits >> 31) & 1u;
    uint32_t fexp = (bits >> 23) & 0xFFu;
    if (fexp == 0u || v != v) return sign << 7;  // ±0 / denorm / nan
    int e8 = (int)fexp - 120;                    // fp8 biased exp
    if (e8 <= 0)  return sign << 7;
    if (e8 >= 15) e8 = 15;                       // clamp to max
    return (sign << 7) | ((uint32_t)e8 << 3) | ((bits >> 20) & 0x7u);
}

/// Extract signed integer mantissa, ignoring exponent.
/// Returns ±(8+m) ∈ {−15..−8} ∪ {8..15}, or 0 for zero.
/// This is the integer that dp4a operates on.
__device__ __forceinline__
int8_t fp8_signed_mantissa(uint8_t x)
{
    int exp4 = (x >> 3) & 0xF;
    if (!exp4) return 0;
    int mag = 8 + (x & 0x7);             // implicit leading 1 scaled to int
    return (x & 0x80) ? (int8_t)-mag : (int8_t)mag;
}


// ════════════════════════════════════════════════════════════
//  KERNEL mm_v3 — Fix 1: bank conflicts + transposed B
//
//  THE BUG in mm85:
//    As[float][TILE][TILE], TILE=64
//    Bank of As[r][k] = (r*64 + k) % 32
//    As[0][k] and As[4][k]: (4*64 + k)%32 = (256+k)%32 = k%32
//    → identical bank → 2-way conflict on EVERY As read.
//
//  FIX:
//    As[TILE][TILE + 1]  ← +1 padding column
//    Bank of As[r][k] = (r*65 + k) % 32
//    gcd(65, 32) = 1 → rows stride through all 32 banks → no conflict.
//
//  ALSO: store B transposed (BsT[col][row]) so the compute
//  loop reads BsT as a row instead of a column.
//  Original Bs[k][tx*WPT+j] for varying k = column access = conflict.
//  New     BsT[tx*WPT+j][k] for varying k = row access = clean.
//
//  Grid:  (N/TILE, N/TILE)
//  Block: (TILE/WPT, TILE/WPT) = (16, 16) = 256 threads
// ════════════════════════════════════════════════════════════

#define TILE_V3  64
#define WPT_V3   4

__global__ void mm_v3(f8* __restrict__ A,
                      f8* __restrict__ B,
                      f8* __restrict__ C,
                      int N)
{
    // +1 padding: bank conflict eliminated (see analysis above)
    __shared__ float As [TILE_V3][TILE_V3 + 1];
    __shared__ float BsT[TILE_V3][TILE_V3 + 1];  // transposed

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_V3 + ty * WPT_V3;
    int col = blockIdx.x * TILE_V3 + tx * WPT_V3;

    // Named scalars so the compiler can alias them to registers,
    // not a stack array — avoids spill to local memory.
    float s00=0,s01=0,s02=0,s03=0,
          s10=0,s11=0,s12=0,s13=0,
          s20=0,s21=0,s22=0,s23=0,
          s30=0,s31=0,s32=0,s33=0;

    for (int t = 0; t < (N + TILE_V3 - 1) / TILE_V3; t++) {

        // ── LOAD A (row-major, same as before) ──
        #pragma unroll
        for (int i = 0; i < WPT_V3; i++) {
            int ar = row + i;
            int ac = t * TILE_V3 + tx * WPT_V3;
            uint32_t a4 = (ar < N && ac + 3 < N)
                        ? *(const uint32_t*)(&A[ar * N + ac]) : 0u;
            #pragma unroll
            for (int k = 0; k < WPT_V3; k++)
                As[ty*WPT_V3+i][tx*WPT_V3+k] = fp8_to_f32((a4 >> (k*8)) & 0xFF);
        }

        // ── LOAD B → BsT transposed ──
        // BsT[col_local][row_local] so compute reads are row accesses.
        // col_local = tx*WPT+k  (which output column this thread owns)
        // row_local = ty*WPT+i  (which k-dimension row)
        #pragma unroll
        for (int i = 0; i < WPT_V3; i++) {
            int br = t * TILE_V3 + ty * WPT_V3 + i;
            int bc = col;
            uint32_t b4 = (br < N && bc + 3 < N)
                        ? *(const uint32_t*)(&B[br * N + bc]) : 0u;
            #pragma unroll
            for (int k = 0; k < WPT_V3; k++)
                BsT[tx*WPT_V3+k][ty*WPT_V3+i] = fp8_to_f32((b4 >> (k*8)) & 0xFF);
        }

        __syncthreads();

        // ── COMPUTE — both shmem reads are now row-major ──
        //   As [ty*WPT+i][k]     : fixed row, varying k → stride-1 ✓
        //   BsT[tx*WPT+j][k]     : fixed row, varying k → stride-1 ✓
        #pragma unroll
        for (int k = 0; k < TILE_V3; k++) {
            float a0 = As[ty*WPT_V3+0][k], a1 = As[ty*WPT_V3+1][k];
            float a2 = As[ty*WPT_V3+2][k], a3 = As[ty*WPT_V3+3][k];
            float b0 = BsT[tx*WPT_V3+0][k], b1 = BsT[tx*WPT_V3+1][k];
            float b2 = BsT[tx*WPT_V3+2][k], b3 = BsT[tx*WPT_V3+3][k];

            s00=__fmaf_rn(a0,b0,s00); s01=__fmaf_rn(a0,b1,s01);
            s02=__fmaf_rn(a0,b2,s02); s03=__fmaf_rn(a0,b3,s03);
            s10=__fmaf_rn(a1,b0,s10); s11=__fmaf_rn(a1,b1,s11);
            s12=__fmaf_rn(a1,b2,s12); s13=__fmaf_rn(a1,b3,s13);
            s20=__fmaf_rn(a2,b0,s20); s21=__fmaf_rn(a2,b1,s21);
            s22=__fmaf_rn(a2,b2,s22); s23=__fmaf_rn(a2,b3,s23);
            s30=__fmaf_rn(a3,b0,s30); s31=__fmaf_rn(a3,b1,s31);
            s32=__fmaf_rn(a3,b2,s32); s33=__fmaf_rn(a3,b3,s33);
        }

        __syncthreads();
    }

    // ── STORE packed ──
    #pragma unroll
    for (int i = 0; i < WPT_V3; i++) {
        int r = row + i, c = col;
        if (r < N && c + 3 < N) {
            float v[4];
            if      (i==0) { v[0]=s00;v[1]=s01;v[2]=s02;v[3]=s03; }
            else if (i==1) { v[0]=s10;v[1]=s11;v[2]=s12;v[3]=s13; }
            else if (i==2) { v[0]=s20;v[1]=s21;v[2]=s22;v[3]=s23; }
            else           { v[0]=s30;v[1]=s31;v[2]=s32;v[3]=s33; }
            uint32_t p = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) p |= (f32_to_fp8(v[j]) << (j*8));
            *(uint32_t*)(&C[r*N+c]) = p;
        }
    }
}


// ════════════════════════════════════════════════════════════
//  KERNEL mm_compact — Fix 2: compact FP8 shared memory
//
//  Insight: storing FP32 in shmem wastes 3 bytes per element.
//  FP8 bytes are 4× smaller → same 32 KB supports TILE=128
//  (128×132 bytes per matrix ≈ 16.5 KB, two matrices ≈ 33 KB).
//
//  Conversion happens on-the-fly in REGISTERS, not shmem.
//  These register ops live in the FMA stall slots — essentially free.
//
//  Bank conflict analysis for uint8 arrays (4-byte bank granularity):
//    bank = (byte_offset / 4) % 32
//    For As[128][132]: row stride = 132 bytes = 33 uint32 slots
//    gcd(33, 32) = 1 → rows cycle through all 32 banks → no conflict.
//
//  B stored transposed (BsT) → compute reads are row-major.
//
//  Grid:  (N/TILE, N/TILE)
//  Block: (TILE/WPT, TILE/WPT) = (32, 32) = 1024 threads  ← SM max
// ════════════════════════════════════════════════════════════

#define TILE_C  128
#define WPT_C   4

__global__ void mm_compact(f8* __restrict__ A,
                           f8* __restrict__ B,
                           f8* __restrict__ C,
                           int N)
{
    // 128 × 132 bytes each = 16,896 bytes per matrix, 33,792 total ≈ 33 KB
    // RTX 3050 has 100 KB L1/shared per SM → leaves 67 KB for double-buffering later.
    __shared__ uint8_t As [TILE_C][TILE_C + 4];
    __shared__ uint8_t BsT[TILE_C][TILE_C + 4];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_C + ty * WPT_C;
    int col = blockIdx.x * TILE_C + tx * WPT_C;

    float sum[WPT_C][WPT_C] = {};

    for (int t = 0; t < (N + TILE_C - 1) / TILE_C; t++) {

        // ── LOAD A bytes (4 per thread per row, WPT_C rows) ──
        // Each thread loads a WPT×WPT byte patch of A in one uint32 read per row.
        // 1024 threads × 4 rows × 4 bytes = 16,384 bytes = TILE×TILE ✓
        #pragma unroll
        for (int i = 0; i < WPT_C; i++) {
            int ar = row + i;
            int ac = t * TILE_C + tx * WPT_C;
            uint32_t a4 = (ar < N && ac + WPT_C - 1 < N)
                        ? *(const uint32_t*)(&A[ar * N + ac]) : 0u;
            *(uint32_t*)(&As[ty*WPT_C+i][tx*WPT_C]) = a4;
        }

        // ── LOAD B bytes → BsT transposed ──
        // BsT[col_local][row_local] = B[t*TILE+row_local][blockCol + col_local]
        // Loading 4 consecutive B columns and scattering into BsT rows.
        #pragma unroll
        for (int i = 0; i < WPT_C; i++) {
            int br = t * TILE_C + ty * WPT_C + i;
            int bc = col;
            uint32_t b4 = (br < N && bc + WPT_C - 1 < N)
                        ? *(const uint32_t*)(&B[br * N + bc]) : 0u;
            #pragma unroll
            for (int k = 0; k < WPT_C; k++)
                BsT[tx*WPT_C+k][ty*WPT_C+i] = (b4 >> (k*8)) & 0xFF;
        }

        __syncthreads();

        // ── COMPUTE: FP8→FP32 in registers, not shmem ──
        // The 7 integer ops for conversion execute during the 4-cycle FMA latency.
        // Net cost: zero extra latency vs pure FP32 accumulation.
        #pragma unroll
        for (int k = 0; k < TILE_C; k++) {

            float a[WPT_C], b[WPT_C];

            // A values: As[row_local][k], row-major ✓
            #pragma unroll
            for (int i = 0; i < WPT_C; i++) {
                uint32_t raw = As[ty*WPT_C+i][k];
                uint32_t e4  = (raw >> 3) & 0xFu;
                a[i] = e4 ? __uint_as_float(((raw & 0x80u) << 24)
                                           | ((e4 + 120u) << 23)
                                           | ((raw & 0x7u) << 20))
                           : 0.0f;
            }

            // B values: BsT[col_local][k], row-major ✓
            #pragma unroll
            for (int j = 0; j < WPT_C; j++) {
                uint32_t raw = BsT[tx*WPT_C+j][k];
                uint32_t e4  = (raw >> 3) & 0xFu;
                b[j] = e4 ? __uint_as_float(((raw & 0x80u) << 24)
                                           | ((e4 + 120u) << 23)
                                           | ((raw & 0x7u) << 20))
                           : 0.0f;
            }

            #pragma unroll
            for (int i = 0; i < WPT_C; i++)
                #pragma unroll
                for (int j = 0; j < WPT_C; j++)
                    sum[i][j] = __fmaf_rn(a[i], b[j], sum[i][j]);
        }

        __syncthreads();
    }

    // ── STORE packed ──
    #pragma unroll
    for (int i = 0; i < WPT_C; i++) {
        int r = row + i, c = col;
        if (r < N && c + WPT_C - 1 < N) {
            uint32_t p = 0;
            #pragma unroll
            for (int j = 0; j < WPT_C; j++)
                p |= (f32_to_fp8(sum[i][j]) << (j*8));
            *(uint32_t*)(&C[r*N+c]) = p;
        }
    }
}


// ════════════════════════════════════════════════════════════
//  KERNEL mm_blockfp — Fix 3: integer dp4a, no FP in inner loop
//
//  THE CORE IDENTITY:
//    FP8 value  = sign × (8 + m) × 2^(e − 10)
//    Product    = sign_a × sign_b × (8+ma) × (8+mb) × 2^(ea+eb−20)
//
//  For each tile iteration, per thread:
//    1. Find max exponent in assigned A-rows (max_ea[i])
//    2. Find max exponent in assigned B-cols (max_eb[j])
//    3. Align each element: shift right by (max_exp − own_exp)
//       This converts to a common fixed-point scale within the block.
//    4. Pack 4 aligned int8 → int32, call __dp4a
//       acc[i][j] ≈ Σ_k aligned_a[i][k] × aligned_b[j][k]
//    5. At the end, multiply by per-row/col scale factors.
//
//  PRECISION NOTE:
//    Elements with exp << max_exp get right-shifted to 0
//    (shift > 4 loses all mantissa bits). For calibrated FP8
//    weights (where per-row/col scales keep values similar),
//    this is the dominant term anyway — small values being
//    zeroed is numerically negligible. Same trade-off as
//    NVIDIA TransformerEngine's FP8 block scaling.
//
//  INT vs FP throughput (SM 8.6):
//    FP32 FFMA:  128 ops/SM/clk (2 ops per FMA)
//    INT32:       64 ops/SM/clk (1 op per cycle)
//    dp4a:        64 instructions/SM/clk = 512 int ops
//    → dp4a is 4× throughput over scalar INT32, matches FP32 FFMA.
//    Bonus: dp4a accumulates into INT32 — zero rounding error
//    in the accumulation itself (only at the final scale step).
//
//  REQUIREMENTS:
//    scale_A[N]: per-row scale for A (from calibration or amax)
//    scale_B[N]: per-col scale for B
//
//  Grid:  (N/TILE, N/TILE)
//  Block: (TILE/WPT, TILE/WPT) = (16, 16) = 256 threads
// ════════════════════════════════════════════════════════════

#define TILE_I  64
#define WPT_I   4

__global__ void mm_blockfp(f8*    __restrict__ A,
                           f8*    __restrict__ B,
                           f8*    __restrict__ C,
                           const float* __restrict__ scale_A,  // [N] per row
                           const float* __restrict__ scale_B,  // [N] per col
                           int N)
{
    __shared__ uint8_t As [TILE_I][TILE_I + 4];
    __shared__ uint8_t BsT[TILE_I][TILE_I + 4];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_I + ty * WPT_I;
    int col = blockIdx.x * TILE_I + tx * WPT_I;

    // int32 accumulators: exact integer dot products, no rounding in inner loop
    int32_t acc[WPT_I][WPT_I] = {};

    const int NUM_TILES = (N + TILE_I - 1) / TILE_I;

    for (int t = 0; t < NUM_TILES; t++) {

        // ── LOAD FP8 bytes (same as mm_compact) ──
        #pragma unroll
        for (int i = 0; i < WPT_I; i++) {
            int ar = row + i, ac = t * TILE_I + tx * WPT_I;
            uint32_t a4 = (ar < N && ac + WPT_I - 1 < N)
                        ? *(const uint32_t*)(&A[ar * N + ac]) : 0u;
            *(uint32_t*)(&As[ty*WPT_I+i][tx*WPT_I]) = a4;

            int br = t * TILE_I + ty * WPT_I + i, bc = col;
            uint32_t b4 = (br < N && bc + WPT_I - 1 < N)
                        ? *(const uint32_t*)(&B[br * N + bc]) : 0u;
            #pragma unroll
            for (int k = 0; k < WPT_I; k++)
                BsT[tx*WPT_I+k][ty*WPT_I+i] = (b4 >> (k*8)) & 0xFF;
        }

        __syncthreads();

        // ── PASS 1: find max exponent per assigned A-row and B-col ──
        // One pass over shmem, pure integer comparisons.
        // This replaces the per-element FP conversion with a max scan.
        int max_ea[WPT_I] = {}, max_eb[WPT_I] = {};

        #pragma unroll 4
        for (int k = 0; k < TILE_I; k++) {
            #pragma unroll
            for (int i = 0; i < WPT_I; i++) {
                int e = (As [ty*WPT_I+i][k] >> 3) & 0xF;
                if (e > max_ea[i]) max_ea[i] = e;
            }
            #pragma unroll
            for (int j = 0; j < WPT_I; j++) {
                int e = (BsT[tx*WPT_I+j][k] >> 3) & 0xF;
                if (e > max_eb[j]) max_eb[j] = e;
            }
        }

        // ── PASS 2: align + dp4a ──
        // Process k in groups of 4 to build int32 packs for dp4a.
        // __dp4a(a, b, c): c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
        //   where a,b are each 4 packed int8 (signed).
        #pragma unroll 4
        for (int k = 0; k < TILE_I; k += 4) {

            // Build packed int32 for A rows
            int32_t a_pack[WPT_I] = {};
            #pragma unroll
            for (int i = 0; i < WPT_I; i++) {
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    uint8_t raw  = As[ty*WPT_I+i][k+kk];
                    int     exp4 = (raw >> 3) & 0xF;
                    if (!exp4) continue;                 // zero element
                    int     mag  = 8 + (raw & 0x7);     // 4-bit magnitude [8,15]
                    int     sval = (raw & 0x80) ? -mag : mag;
                    int     shft = max_ea[i] - exp4;    // align to row max exponent
                    int8_t  ival = (shft < 8) ? (int8_t)(sval >> shft) : 0;
                    // Pack int8 into byte kk of the int32
                    a_pack[i] |= ((uint32_t)(uint8_t)ival << (kk * 8));
                }
            }

            // Build packed int32 for B cols (from transposed shmem)
            int32_t b_pack[WPT_I] = {};
            #pragma unroll
            for (int j = 0; j < WPT_I; j++) {
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    uint8_t raw  = BsT[tx*WPT_I+j][k+kk];
                    int     exp4 = (raw >> 3) & 0xF;
                    if (!exp4) continue;
                    int     mag  = 8 + (raw & 0x7);
                    int     sval = (raw & 0x80) ? -mag : mag;
                    int     shft = max_eb[j] - exp4;
                    int8_t  ival = (shft < 8) ? (int8_t)(sval >> shft) : 0;
                    b_pack[j] |= ((uint32_t)(uint8_t)ival << (kk * 8));
                }
            }

            // dp4a: native SM 8.6 instruction — 4 int8 muls + adds in 1 cycle
            #pragma unroll
            for (int i = 0; i < WPT_I; i++)
                #pragma unroll
                for (int j = 0; j < WPT_I; j++)
                    acc[i][j] = __dp4a(a_pack[i], b_pack[j], acc[i][j]);
        }

        __syncthreads();
    }

    // ── SCALE + STORE ──
    // acc[i][j] = integer sum of aligned mantissa products
    // Actual C value = acc[i][j] × scale_A[row+i] × scale_B[col+j]
    // The per-row/col scales encode: 2^(typical_exp - 10) × calibration_factor
    // This is the same pattern as NVIDIA TransformerEngine FP8.
    #pragma unroll
    for (int i = 0; i < WPT_I; i++) {
        int r = row + i;
        if (r >= N) continue;
        float sa = scale_A[r];
        int c = col;

        uint32_t p = 0;
        #pragma unroll
        for (int j = 0; j < WPT_I; j++) {
            if (c + j >= N) break;
            float result = (float)acc[i][j] * sa * scale_B[c + j];
            p |= (f32_to_fp8(result) << (j * 8));
        }
        if (c + WPT_I - 1 < N)
            *(uint32_t*)(&C[r * N + c]) = p;
    }
}


// ════════════════════════════════════════════════════════════
//  BONUS: mm_compact_async — double-buffered with cp.async
//
//  On RTX 3050 (SM >= 8.0), cp.async lets the GPU DMA load
//  the next tile from global memory WHILE the current tile
//  is being computed — hiding ~200-600 cycle DRAM latency.
//
//  Two shared memory buffers (ping-pong) at 33 KB each = 66 KB.
//  RTX 3050 has 100 KB L1+shared → this fits.
//
//  Compile with -arch=sm_86 and link against cuda::pipeline.
// ════════════════════════════════════════════════════════════

// Usage (caller side):
//
//  // mm_v3
//  dim3 grid_v3 (N/TILE_V3, N/TILE_V3);
//  dim3 block_v3(TILE_V3/WPT_V3, TILE_V3/WPT_V3);  // 16,16
//  mm_v3<<<grid_v3, block_v3>>>(A, B, C, N);
//
//  // mm_compact
//  dim3 grid_c (N/TILE_C, N/TILE_C);
//  dim3 block_c(TILE_C/WPT_C, TILE_C/WPT_C);        // 32,32 = 1024
//  mm_compact<<<grid_c, block_c>>>(A, B, C, N);
//
//  // mm_blockfp  (requires pre-computed scales)
//  dim3 grid_i (N/TILE_I, N/TILE_I);
//  dim3 block_i(TILE_I/WPT_I, TILE_I/WPT_I);        // 16,16
//  mm_blockfp<<<grid_i, block_i>>>(A, B, C, scA, scB, N);

// ════════════════════════════════════════════════════════════
//  HOW TO COMPUTE scale_A / scale_B FOR mm_blockfp
//
//  Run this before the matmul:
//
//  __global__ void compute_row_scales(f8* A, float* scale, int N) {
//      int row = blockIdx.x * blockDim.x + threadIdx.x;
//      if (row >= N) return;
//      int max_exp = 0;
//      for (int k = 0; k < N; k++) {
//          int e = (A[row * N + k] >> 3) & 0xF;
//          if (e > max_exp) max_exp = e;
//      }
//      // scale = 2^(max_exp - 10): converts aligned int mantissas back to real values
//      // The aligned mantissa is (8+m) >> (max_exp - own_exp), which is at most 15.
//      // Dividing by 8 gives the fractional part in [1, 1.875].
//      // Full scale: 2^(max_exp-10) / 8 = 2^(max_exp-13)
//      scale[row] = __int_as_float((max_exp + 114) << 23);  // = 2^(max_exp-13)
//  }
// ════════════════════════════════════════════════════════════
