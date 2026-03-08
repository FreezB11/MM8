///@file: fp8.c
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
/**
@note:
    E4M3 has small range and good precision, so we will use this for activation
    -> |1bit sign|4bit exponent|3bit mantisa|
        sign = (fp8 >> 7) & 1
        exp  = (fp8 >> 3) & 0xF
        mant = fp8 & 0x7

    E5M2 has large range but low precision, this can be used in weights
    -> |1bit sign|5bit exponent|2bit mantisa|

*/
#define N 4
typedef uint8_t fp8;
typedef fp8 float8_e4m3fn; //<<-- we will work with this now
typedef fp8 float8_e5m2;
int bias  = 7; // this is for fp8

float8_e4m3fn mul(float8_e4m3fn a, float8_e4m3fn b){
    int signA = (a >> 7) & 1;
    int signB = (b >> 7) & 1;

    int expA = (a >> 3) & 0xF;//0xf == 1111
    int expB = (b >> 3) & 0xF;

    int mantA = a & 0x7;//0x7 == 111
    int mantB = b & 0x7;

    int S = signA ^ signB;
    int E = expA + expB - bias;
    int ma = (1 << 3) | mantA;//1.mmm
    int mb = (1 << 3) | mantB;
    int M = ma*mb; // the range will always be [8,15]

    // the carrying
    if (M >= 128)
    {
        M >>= 1;
        E++;
    }

    // int mant = (M >> 3) & 0x7;
    int mant = (M+4) >> 3;
    if(mant >=16){
        mant = 0;
        E++;
    }else{
        mant &= 0x7;
    }
    if (E <= 0) return 0;
    if (E >= 15) return (S<<7) | (15<<3);

    return (S<<7) | (E<<3) | mant;
}

/// @brief fp8 addition
/// @param a 
/// @param b 
/// @return float8_e4m3fn
float8_e4m3fn add(float8_e4m3fn a, float8_e4m3fn b){

    int signA = (a >> 7) & 1;
    int signB = (b >> 7) & 1;

    int expA = (a >> 3) & 0xF;
    int expB = (b >> 3) & 0xF;

    int mantA = a & 0x7;
    int mantB = b & 0x7;

    int ma = (1 << 3) | mantA;
    int mb = (1 << 3) | mantB;

    int exp;

    if (expA > expB) {
        int shift = expA - expB;
        mb >>= shift;
        exp = expA;
    } else {
        int shift = expB - expA;
        ma >>= shift;
        exp = expB;
    }

    int M;
    int sign;

    if (signA == signB) {
        M = ma + mb;
        sign = signA;
    } else {
        if (ma >= mb) {
            M = ma - mb;
            sign = signA;
        } else {
            M = mb - ma;
            sign = signB;
        }
    }

    if (M >= 16) {
        M >>= 1;
        exp++;
    }

    while (M > 0 && M < 8) {
        M <<= 1;
        exp--;
    }

    if (exp <= 0) return 0;
    if (exp >= 15) return (sign<<7) | (15<<3);

    int mant = M & 0x7;

    return (sign<<7) | (exp<<3) | mant;
}

/// @brief fp8 to fp32
/// @param x 
/// @return fp32 
float fp8_to_float(fp8 x)
{
    int sign = (x >> 7) & 1;
    int exp  = (x >> 3) & 0xF;
    int mant = x & 0x7;

    if (exp == 0 && mant == 0)
        return 0.0f;

    float m = 1.0f + (mant / 8.0f);
    int e = exp - 7;

    float val = ldexpf(m, e); // m * 2^e

    return sign ? -val : val;
}

/// @brief fp32 to fp8
/// @param x 
/// @return fp8
fp8 float_to_fp8(float x)
{
    if (x == 0.0f)
        return 0;

    int sign = x < 0;
    if (sign) x = -x;

    int exp;
    float frac = frexpf(x, &exp); 
    // x = frac * 2^exp , frac in [0.5,1)

    frac *= 2;
    exp--;

    int exp8 = exp + 7;

    // int mant = (int)((frac - 1.0f) * 8.0f);
    int mant = (int)roundf((frac - 1.0f) * 8.0f);

    if (mant == 8) {
    mant = 0;
    exp8++;
}

    if (exp8 <= 0) return 0;
    if (exp8 >= 15) exp8 = 15;

    return (sign << 7) | (exp8 << 3) | (mant & 0x7);
}

void print_bits(fp8 x)
{
    for(int i=7;i>=0;i--)
        printf("%d", (x>>i)&1);
}

void matmul_f32(float A[N][N], float B[N][N], float C[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            C[i][j] = 0;
            for(int k=0;k<N;k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

/// @brief version with fp8 accumulation
/// @param A 
/// @param B 
/// @param C 
// void matmul_fp8(fp8 A[N][N], fp8 B[N][N], fp8 C[N][N])
// {
//     for(int i=0;i<N;i++)
//         for(int j=0;j<N;j++){

//             fp8 sum = 0;

//             for(int k=0;k<N;k++){

//                 fp8 p = mul(A[i][k], B[k][j]);
//                 sum = add(sum, p);
//             }

//             C[i][j] = sum;
//         }
// }

/// @brief version 2 with fp16 acumulation
/// @param A 
/// @param B 
/// @param C 
void matmul_fp8(fp8 A[N][N], fp8 B[N][N], fp8 C[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){

            float sum = 0.0f;

            for(int k=0;k<N;k++){

                fp8 p = mul(A[i][k], B[k][j]);

                sum += fp8_to_float(p);
            }

            C[i][j] = float_to_fp8(sum);
        }
}

void convert_to_fp8(float A[N][N], fp8 B[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            B[i][j] = float_to_fp8(A[i][j]);
}

void compute_error(float ref[N][N], fp8 test[N][N], float err[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){

            float v = fp8_to_float(test[i][j]);
            err[i][j] = ref[i][j] - v;
        }
}

void random_matrix(float A[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            A[i][j] = ((float)rand()/RAND_MAX)*4 - 2;
}

// int main()
// {
//     float a_f = 1.5f;
//     float b_f = 2.25f;

//     fp8 a = float_to_fp8(a_f);
//     fp8 b = float_to_fp8(b_f);

//     fp8 r = mul(a,b);

//     printf("A bits: ");
//     print_bits(a);
//     printf("  value=%f\n", fp8_to_float(a));

//     printf("B bits: ");
//     print_bits(b);
//     printf("  value=%f\n", fp8_to_float(b));

//     printf("R bits: ");
//     print_bits(r);
//     printf("  value=%f\n", fp8_to_float(r));

//     printf("Expected float result: %f\n", a_f * b_f);

//     return 0;
// }

int main()
{
    float A[N][N], B[N][N];
    float C_ref[N][N];

    fp8 A8[N][N], B8[N][N], C8[N][N];

    float error[N][N];

    random_matrix(A);
    random_matrix(B);

    matmul_f32(A,B,C_ref);

    convert_to_fp8(A,A8);
    convert_to_fp8(B,B8);

    matmul_fp8(A8,B8,C8);

    compute_error(C_ref,C8,error);

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++)
            printf("%f ", error[i][j]);
        printf("\n");
    }
}