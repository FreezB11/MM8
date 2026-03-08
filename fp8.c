///@file: fp8.c
#include <stdio.h>
#include <stdint.h>
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

int main() {
    float a = 24.256f;

    uint32_t bits = *(uint32_t*)&a;

    printf("FP32 bits:\n");

    for(int i = 31; i >= 0; i--) {
        printf("%d", (bits >> i) & 1);

        if(i == 31 || i == 23) printf(" "); // separate sign/exponent/mantissa
    }

    printf("\n");
}