///@file: fp8_e5m2.c
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
@note:
    E5M2 — the real second fp8 format (used for gradients in ML training)
    -> |1bit sign|5bit exponent|2bit mantissa|
        sign = (fp8 >> 7) & 1
        exp  = (fp8 >> 2) & 0x1F   // 0x1F = 0b11111
        mant = fp8 & 0x3            // 0x3  = 0b11

    With 5 exponent bits:
        bias = 15  (= 2^(5-1) - 1)
        exp range stored: 0–31
        actual exponent:  -15 to +16
        representable range: ~[-57344, 57344]

    With only 2 mantissa bits:
        precision step = 1/4 = 0.25
        coarser than E4M3's 1/8 = 0.125

    Tradeoff vs E4M3:
        E4M3 → range ±240,    precision 0.125  ← activations
        E5M2 → range ±57344,  precision 0.250  ← gradients (need huge range)
        E2M5 → range ±7.875,  precision 0.031  ← useless in practice
*/

#define N 4
typedef uint8_t fp8;
typedef fp8 float8_e5m2;

int bias = 15; // for 5-bit exponent: bias = 2^(5-1) - 1 = 15

/// @brief fp8 E5M2 multiplication
float8_e5m2 mul(float8_e5m2 a, float8_e5m2 b) {
    // --- unpack ---
    int signA = (a >> 7) & 1;
    int signB = (b >> 7) & 1;

    int expA  = (a >> 2) & 0x1F;   // 5 exponent bits
    int expB  = (b >> 2) & 0x1F;

    int mantA = a & 0x3;            // 2 mantissa bits
    int mantB = b & 0x3;

    // --- sign: XOR ---
    int S = signA ^ signB;

    // --- exponent: add and debias ---
    int E = expA + expB - bias;

    // --- restore implicit 1: 1.mm ---
    // implicit 1 at bit 2, so ma/mb in [4, 7]
    int ma = (1 << 2) | mantA;
    int mb = (1 << 2) | mantB;

    // --- multiply ---
    // ma, mb in [4, 7]
    // M in [4*4, 7*7] = [16, 49]  →  4–6 bits wide
    // implicit 1 at bit 2+2 = bit 4
    int M = ma * mb;

    // --- carry check ---
    // no carry: M in [16, 31]  →  bit 5 = 0  →  result is 1.xx
    // carry:    M in [32, 49]  →  bit 5 = 1  →  result is 1x.xx → normalize
    if (M >= 32) {
        M >>= 1;
        E++;
    }
    // after this, M is always in [16, 31], implicit 1 at bit 4

    // --- round and extract 2 mantissa bits ---
    // want bits 3-2, rounding with bit 1
    // add 1<<1 = 2 for round-to-nearest, then shift down 2
    int mant = (M + 2) >> 2;
    // mant is now in [4, 8]

    if (mant >= 8) {
        // rounding overflowed: 1.11 + round → 10.00
        mant = 0;
        E++;
    } else {
        mant &= 0x3;  // strip implicit leading 1 (at bit 2)
    }

    // --- clamp ---
    if (E <= 0)  return 0;
    if (E >= 31) return (S << 7) | (31 << 2);  // max value

    return (S << 7) | (E << 2) | mant;
}

/// @brief fp8 E5M2 addition
float8_e5m2 add(float8_e5m2 a, float8_e5m2 b) {
    // --- unpack ---
    int signA = (a >> 7) & 1;
    int signB = (b >> 7) & 1;

    int expA  = (a >> 2) & 0x1F;
    int expB  = (b >> 2) & 0x1F;

    int mantA = a & 0x3;
    int mantB = b & 0x3;

    // restore implicit 1 — ma/mb in [4, 7]
    int ma = (1 << 2) | mantA;
    int mb = (1 << 2) | mantB;

    int exp;

    // --- align: shift smaller exponent to match larger ---
    if (expA > expB) {
        int shift = expA - expB;
        mb >>= shift;
        exp = expA;
    } else {
        int shift = expB - expA;
        ma >>= shift;
        exp = expB;
    }

    // --- add or subtract mantissas ---
    int M, sign;

    if (signA == signB) {
        M    = ma + mb;
        sign = signA;
    } else {
        if (ma >= mb) {
            M    = ma - mb;
            sign = signA;
        } else {
            M    = mb - ma;
            sign = signB;
        }
    }

    // --- normalize ---
    // addition carry: M >= 8 (bit 3 set), shift right
    if (M >= 8) {
        M >>= 1;
        exp++;
    }

    // subtraction shrink: M fell below implicit-1 position (bit 2)
    while (M > 0 && M < 4) {
        M <<= 1;
        exp--;
    }

    // --- clamp ---
    if (M == 0)   return 0;
    if (exp <= 0) return 0;
    if (exp >= 31) return (sign << 7) | (31 << 2);

    int mant = M & 0x3;  // strip implicit 1

    return (sign << 7) | (exp << 2) | mant;
}

/// @brief fp8 E5M2 → fp32
float fp8_to_float(fp8 x) {
    int sign = (x >> 7) & 1;
    int exp  = (x >> 2) & 0x1F;
    int mant = x & 0x3;

    if (exp == 0 && mant == 0)
        return 0.0f;

    // 2 mantissa bits: precision = 1/4
    float m = 1.0f + (mant / 4.0f);
    int e   = exp - bias;

    float val = ldexpf(m, e);  // m * 2^e

    return sign ? -val : val;
}

/// @brief fp32 → fp8 E5M2
fp8 float_to_fp8(float x) {
    if (x == 0.0f)
        return 0;

    int sign = x < 0;
    if (sign) x = -x;

    int exp;
    float frac = frexpf(x, &exp);
    // x = frac * 2^exp, frac in [0.5, 1)

    frac *= 2;
    exp--;
    // now frac in [1.0, 2.0)

    int exp8 = exp + bias;

    // 2 mantissa bits: extract fractional part as integer in [0, 4)
    int mant = (int)roundf((frac - 1.0f) * 4.0f);

    // rounding can push mant to 4 (overflow)
    if (mant == 4) {
        mant = 0;
        exp8++;
    }

    if (exp8 <= 0) return 0;
    if (exp8 >= 31) exp8 = 31;

    return (sign << 7) | (exp8 << 2) | (mant & 0x3);
}

void print_bits(fp8 x) {
    // S | EEEEE | MM
    printf("%d|%d%d%d%d%d|%d%d",
        (x>>7)&1,
        (x>>6)&1,(x>>5)&1,(x>>4)&1,(x>>3)&1,(x>>2)&1,
        (x>>1)&1, x&1);
}

void matmul_f32(float A[N][N], float B[N][N], float C[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

/// @brief matmul with fp32 accumulation
void matmul_fp8(fp8 A[N][N], fp8 B[N][N], fp8 C[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {

            float sum = 0.0f;

            for (int k = 0; k < N; k++) {
                fp8 p = mul(A[i][k], B[k][j]);
                sum  += fp8_to_float(p);
            }

            C[i][j] = float_to_fp8(sum);
        }
}

void convert_to_fp8(float A[N][N], fp8 B[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = float_to_fp8(A[i][j]);
}

void compute_error(float ref[N][N], fp8 test[N][N], float err[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float v   = fp8_to_float(test[i][j]);
            err[i][j] = ref[i][j] - v;
        }
}

/// @brief random matrix — E5M2 has huge range so [-4, 4] is fine
/// matmul outputs worst case: 4 * 4.0 * 4.0 = 64  <<  57344 max
void random_matrix(float A[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = ((float)rand() / RAND_MAX) * 8 - 4;  // [-4, 4]
}

// --- quick sanity check ---
void test_single(float a_f, float b_f) {
    fp8 a = float_to_fp8(a_f);
    fp8 b = float_to_fp8(b_f);
    fp8 r = mul(a, b);

    printf("A: "); print_bits(a); printf("  decoded=%.6f  (input=%.6f)\n", fp8_to_float(a), a_f);
    printf("B: "); print_bits(b); printf("  decoded=%.6f  (input=%.6f)\n", fp8_to_float(b), b_f);
    printf("R: "); print_bits(r); printf("  decoded=%.6f  (expected=%.6f)\n", fp8_to_float(r), a_f * b_f);
    printf("\n");
}

int main() {
    printf("=== E5M2 format info ===\n");
    printf("bits:      S | EEEEE | MM\n");
    printf("bias:      %d\n", bias);
    printf("exp range: stored [0,31] -> actual [%d, %d]\n", 0 - bias, 31 - bias);
    printf("precision: 1/4 = %.5f\n", 1.0f / 4.0f);
    printf("max value: %.2f\n",   fp8_to_float(0b01111111));  // S=0, EEEEE=11111, MM=11
    printf("min value: %.2f\n\n", fp8_to_float(0b11111111));  // S=1, EEEEE=11111, MM=11

    printf("=== single multiply tests ===\n");
    test_single(1.5f,  2.0f);
    test_single(1.25f, 1.5f);
    test_single(-1.0f, 2.0f);
    test_single(3.0f,  4.0f);

    printf("=== matmul error (fp32 ref vs e5m2) ===\n");
    float A[N][N], B[N][N];
    float C_ref[N][N];
    fp8   A8[N][N], B8[N][N], C8[N][N];
    float error[N][N];

    srand(42);
    random_matrix(A);
    random_matrix(B);

    matmul_f32(A, B, C_ref);
    convert_to_fp8(A, A8);
    convert_to_fp8(B, B8);
    matmul_fp8(A8, B8, C8);
    compute_error(C_ref, C8, error);

    printf("\nError matrix:\n");
    float max_err = 0, sum_err = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%9.6f ", error[i][j]);
            float e = fabsf(error[i][j]);
            if (e > max_err) max_err = e;
            sum_err += e;
        }
        printf("\n");
    }

    printf("\nMax absolute error : %.6f\n", max_err);
    printf("Mean absolute error: %.6f\n", sum_err / (N * N));
    printf("\n--- Format comparison ---\n");
    printf("E5M2  range ±57344,  precision 0.250  -> wider errors, never saturates\n");
    printf("E4M3  range ±240,    precision 0.125  -> tighter errors, can saturate\n");
    printf("E2M5  range ±7.875,  precision 0.031  -> best precision, saturates fast\n");

    return 0;
}