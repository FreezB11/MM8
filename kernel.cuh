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

__device__ __host__ __forceinline__ float f8tof32(f8 x){
    int sign = (x >> 7) & 1;
    int exp  = (x >> 3) & 0xF;
    int mant = x & 0x7;
    if (exp == 0 && mant == 0) return 0.0f;
    float val = ldexpf(1.0f + mant * 0.125f, exp - 7);
    return sign ? -val : val;
}

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

/// @brief this kernel is taking the assumptions that the matrix is square
/// @param A input mat_A
/// @param B input mat_B
/// @param C output mat_C
/// @param N matrix dimension
/// @return we will return the matrix C with the multiplied values
__global__ void mm8(f8* A, f8* B, f8* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= N || row >= N) return;
    /*
        sum = 0
        for k -> [0->N] :
            sum += A[row*N + k] * B[k*N + col]
        C[row*N + col] = sum
    */
    float sum = 0;
    for(int k = 0; k < N; k++){
        sum += f8tof32(A[row*N+k]) * f8tof32(B[k*N + col]);
    }
    C[row*N + col] = f32tof8(sum);
}