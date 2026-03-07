///@file: fp8_e2m5.c
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
@note:
    E2M5 is a hypothetical format — opposite tradeoff to E5M2:
    -> |1bit sign|2bit exponent|5bit mantissa|
        sign = (fp8 >> 7) & 1
        exp  = (fp8 >> 5) & 0x3   // 0x3 = 0b11
        mant = fp8 & 0x1F         // 0x1F = 0b11111

    With only 2 exponent bits:
        bias = 1  (= 2^(2-1) - 1)
        exp range stored: 0–3
        actual exponent:  -1 to +2
        representable range: ~[0.5, 8.0]

    With 5 mantissa bits:
        precision step = 1/32 = 0.03125
        much finer grained than E4M3's 1/8 = 0.125

    Tradeoff:
        E4M3 → range [-240, 240],  precision 0.125
        E2M5 → range [-8.0, 8.0], precision 0.03125
*/

#define N 4
typedef uint8_t fp8;
typedef fp8 float8_e2m5;

int bias = 1; // for 2-bit exponent: bias = 2^(2-1) - 1 = 1

/// @brief fp8 E2M5 multiplication
float8_e2m5 mul(float8_e2m5 a, float8_e2m5 b) {
    // --- unpack ---
    int signA = (a >> 7) & 1;
    int signB = (b >> 7) & 1;

    int expA  = (a >> 5) & 0x3;   // 2 exponent bits
    int expB  = (b >> 5) & 0x3;

    int mantA = a & 0x1F;         // 5 mantissa bits
    int mantB = b & 0x1F;

    // --- sign: XOR ---
    int S = signA ^ signB;

    // --- exponent: add and debias ---
    int E = expA + expB - bias;

    // --- restore implicit 1: 1.mmmmm ---
    // implicit 1 is at bit 5, so ma/mb in [32, 63]
    int ma = (1 << 5) | mantA;
    int mb = (1 << 5) | mantB;

    // --- multiply ---
    // ma, mb in [32, 63]
    // M in [32*32, 63*63] = [1024, 3969]  →  10–12 bits wide
    // implicit 1 is at bit 5+5 = bit 10
    int M = ma * mb;

    // --- carry check ---
    // no carry: M in [1024, 2047]  →  bit 11 = 0  →  result is 1.xxxxx
    // carry:    M in [2048, 3969]  →  bit 11 = 1  →  result is 1x.xxxxx → normalize
    if (M >= 2048) {
        M >>= 1;
        E++;
    }
    // after this, M is always in [1024, 2047], implicit 1 at bit 10

    // --- round and extract 5 mantissa bits ---
    // want bits 9-5, rounding with bit 4
    // add 1<<4 = 16 for round-to-nearest, then shift down 5
    int mant = (M + 16) >> 5;
    // mant is now in [32, 64]

    if (mant >= 64) {
        // rounding overflowed: 1.11111 + round → 10.00000
        mant = 0;
        E++;
    } else {
        mant &= 0x1F; // strip implicit leading 1 (at bit 5)
    }

    // --- clamp ---
    if (E <= 0)  return 0;
    if (E >= 3)  return (S << 7) | (3 << 5); // max value

    return (S << 7) | (E << 5) | mant;
}

/// @brief fp8 E2M5 addition
float8_e2m5 add(float8_e2m5 a, float8_e2m5 b) {
    // --- unpack ---
    int signA = (a >> 7) & 1;
    int signB = (b >> 7) & 1;

    int expA  = (a >> 5) & 0x3;
    int expB  = (b >> 5) & 0x3;

    int mantA = a & 0x1F;
    int mantB = b & 0x1F;

    // restore implicit 1
    int ma = (1 << 5) | mantA;   // [32, 63]
    int mb = (1 << 5) | mantB;

    int exp;

    // --- align: shift the smaller exponent to match the larger ---
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
    // addition can carry into bit 6 (M >= 64), shift right
    if (M >= 64) {
        M >>= 1;
        exp++;
    }

    // subtraction can shrink M below the implicit-1 position (bit 5)
    // shift left until implicit 1 is back at bit 5
    while (M > 0 && M < 32) {
        M <<= 1;
        exp--;
    }

    // --- clamp ---
    if (M == 0)  return 0;
    if (exp <= 0) return 0;
    if (exp >= 3) return (sign << 7) | (3 << 5);

    int mant = M & 0x1F; // strip implicit 1

    return (sign << 7) | (exp << 5) | mant;
}

/// @brief fp8 E2M5 → fp32
float fp8_to_float(fp8 x) {
    int sign = (x >> 7) & 1;
    int exp  = (x >> 5) & 0x3;
    int mant = x & 0x1F;

    if (exp == 0 && mant == 0)
        return 0.0f;

    // 5 mantissa bits: precision = 1/32
    float m = 1.0f + (mant / 32.0f);
    int e   = exp - bias;

    float val = ldexpf(m, e);  // m * 2^e

    return sign ? -val : val;
}

/// @brief fp32 → fp8 E2M5
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
    // now x = frac * 2^exp, frac in [1.0, 2.0)

    int exp8 = exp + bias;

    // 5 mantissa bits: extract fractional part as integer in [0, 32)
    int mant = (int)roundf((frac - 1.0f) * 32.0f);

    // rounding can push mant to 32 (overflow)
    if (mant == 32) {
        mant = 0;
        exp8++;
    }

    if (exp8 <= 0) return 0;
    if (exp8 >= 3) exp8 = 3;  // clamp to max exponent

    return (sign << 7) | (exp8 << 5) | (mant & 0x1F);
}

void print_bits(fp8 x) {
    // S | EE | MMMMM
    printf("%d|%d%d|%d%d%d%d%d",
        (x>>7)&1,
        (x>>6)&1, (x>>5)&1,
        (x>>4)&1, (x>>3)&1, (x>>2)&1, (x>>1)&1, x&1);
}

void matmul_f32(float A[N][N], float B[N][N], float C[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

/// @brief matmul with fp32 accumulation (same style as fp8 E4M3 version)
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
            float v  = fp8_to_float(test[i][j]);
            err[i][j] = ref[i][j] - v;
        }
}

/// @brief random matrix with values in [-1, 1]
/// NOTE: kept in [-1,1] so matmul sums (max ~4.0) stay inside E2M5's [-7.875, 7.875] range
void random_matrix(float A[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;  // [-1, 1]
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
    printf("=== E2M5 format info ===\n");
    printf("bits:      S | EE | MMMMM\n");
    printf("bias:      %d\n", bias);
    printf("exp range: stored [0,3] -> actual [%d, %d]\n", 0 - bias, 3 - bias);
    printf("precision: 1/32 = %.5f\n", 1.0f / 32.0f);
    printf("max value: %.5f\n", fp8_to_float(0b01111111));  // S=0, EE=11, MMMMM=11111
    printf("min value: %.5f\n\n", fp8_to_float(0b11111111)); // S=1, EE=11, MMMMM=11111

    printf("=== single multiply tests ===\n");
    test_single(1.5f, 2.0f);
    test_single(1.25f, 1.5f);
    test_single(-1.0f, 2.0f);

    printf("=== matmul error (fp32 ref vs e2m5) ===\n");
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
    printf("\n--- Expected error ranges ---\n");
    printf("E4M3 precision: 1/8  = 0.12500  -> matmul errors ~0.1-0.3\n");
    printf("E2M5 precision: 1/32 = 0.03125  -> matmul errors ~0.01-0.08\n");
    printf("(but E2M5 saturates fast if values exceed range [-8, 8]!)\n");

    return 0;
}