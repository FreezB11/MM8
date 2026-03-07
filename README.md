# MMUL
This will be a fp8 MATMUL implementation for cuda kernal for non supportated device\
we will try to do benchmark for fp32 and fp8 time comparission for naive approach then\
some optimization.

![fp8](https://developer-blogs.nvidia.com/wp-content/uploads/2025/06/floating-point-structure.png)

FP8 splits into two variants:

- E4M3, a format with 4 exponent 3 mantissa bits, prioritizes precision for forward passes, where weights and activations benefit from finer-grained values. Its approximate range of ±448, along with the ability to represent NaN, accommodates most layer outputs without overflow.
- E5M2, short for 5 exponent bits and 2 mantissa bits, trades mantissa bits for a wider dynamic range (±57,344, ±inf, nan). This broader range is crucial for backward passes, where gradients can vary significantly in magnitude.

BF16’s 8 exponent and 7 mantissa bits offer a vast dynamic range (from 1e-38 to 1e38), which enables it to represent the distributions of weights, activations, and gradients without scaling factors. FP8’s double datatype (E4M3: with a range up to approximately ±448, and E5M2:±57344) coupled with scaling factors, enables more efficient hardware utilization compared to BF16 without sacrificing convergence.

## GPU Data Type Support Table

Legend  
- ✅ = Hardware accelerated  
- ⚠️ = Supported but not tensor-core accelerated  
- ❌ = Not supported  

| GPU Architecture | Example GPUs | INT4 | INT8 | INT16 | INT32 | INT64 | FP4 | FP8 | FP16 | FP32 | FP64 |
|------------------|-------------|------|------|-------|-------|-------|-----|-----|------|------|------|
| Turing | T4(kaggle) | ✅ | ✅ | ⚠️ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ⚠️ |
| Ampere | RTX3050(personal) | ✅ | ✅ | ⚠️ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |

---

# Data Type Usage

| Data Type | Bits | Typical Usage |
|-----------|------|--------------|
| FP64 | 64 | Scientific computing, HPC |
| FP32 | 32 | Standard ML training |
| FP16 | 16 | Mixed precision training |
| BF16 | 16 | Stable training on large models |
| FP8 | 8 | Next-gen LLM training |
| FP4 | 4 | Experimental ultra-low precision |
| INT8 | 8 | Quantized inference |
| INT4 | 4 | LLM inference compression |
| INT32 | 32 | Accumulators / indexing |
| INT64 | 64 | CPU-style integer operations |

## GPU FLOPS / TOPS by Data Type

| GPU | Architecture | FP32 (TFLOPS) | FP16 (TFLOPS) | FP8 (TFLOPS) | INT8 (TOPS) | INT4 (TOPS) |
|-----|--------------|---------------|---------------|--------------|-------------|-------------|
| NVIDIA Tesla T4 | Turing | 8.1 | 65 | N/A | 130 | 260 |
| NVIDIA RTX 3050 | Ampere | ~9 | ~36 | N/A | ~72 | ~144 |


# PLAN
now that we know that my laptop has fp16 and fp32\
but i have less ram but i want larger model so i have to use fp8 so yea\
here we go i will use the int8 that is supported by cuda also and use that for fp8\
there will be issue but lets try it out

lets understand the iee format\
*number* = (-1)<sup>sign</sup> × mantissa× 2<sup>exponent</sup>

for my own sakes [E4M3 format docs](e4m3.md)

llm precision flow
Operation          | Format  | Why
-------------------|---------|---------------------------
forward matmuls    | E4M3    | activations need precision
backward matmuls   | E5M2    | gradients need range
optimizer state    | FP32    | momentum/variance need both
weight master copy | BF16    | range matches FP32
layer norm / softmax | FP32  | numerically sensitive