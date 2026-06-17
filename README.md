# Bare-Metal Transformer Engine

A zero-dependency, bare-metal C++ implementation of the core Transformer encoder architecture from the paper **"Attention Is All You Need"** *(Vaswani et al., 2017)*.

---

## Motivation

This repository bridges the gap between academic theory and low-level inference architecture. Instead of relying on high-level wrappers like PyTorch or optimized linear algebra libraries like Eigen or BLAS, this project implements the mathematical primitives of Large Language Models from scratch using the C++ standard library.

This is an exploratory engine designed to understand the exact matrix routing, masking constraints, hardware acceleration, and memory behavior that power highly optimized inference engines under the hood.

---

## Features

- **Zero Dependencies** — Pure C++ standard library + `<immintrin.h>` for Intel SIMD intrinsics. No external libraries.
- **Hardware Acceleration (SIMD)** — Dense matrix multiplication uses Intel AVX2 intrinsics and Fused Multiply-Add (FMA) to process 8 floats per clock cycle. A scalar tail loop handles remaining columns when the width is not a multiple of 8.
- **4-bit Weight Quantization** — Implements a `QuantizedMatrix` struct with per-row scaling and bitwise packing of two 4-bit integers per byte (~87.5% memory reduction vs float32). The quantized matmul unpacks weights on the fly during inference.
- **Numerically Stable Softmax** — Subtracts the row maximum before exponentiation (`exp(x - max)`) to prevent NaN overflow.
- **Optional Attention Masking** — `scaled_dot_product_attention` accepts an optional mask matrix; positions marked `0` are set to `-1e9` before softmax, effectively zeroing their attention weight. The mask shape and values are caller-defined — no built-in causal mask is generated.
- **Cache-Friendly Matrix Multiply** — Uses `i-k-j` loop ordering so the inner loop strides sequentially through memory, maximizing cache utilization.
- **Manual Memory Routing** — Column slicing (`slice_cols`) and horizontal concatenation (`concat_cols`) implement Multi-Head Attention head splitting and merging without any tensor library.
- **Layer Normalization** — Per-row mean/variance normalization with an epsilon floor (`1e-5`) for numerical stability. No learnable affine parameters (γ/β).
- **ReLU Activation** — Applied between the two FFN linear layers.

---

## Architecture

The `main()` function runs a single forward pass through a complete encoder block on a 2-token, 4-dimensional input sequence.

### Block 1 — Multi-Head Self-Attention + Add & Norm

1. **Multi-Head Attention** — The input `X` serves directly as Q, K, and V (no W_Q/W_K/W_V projection matrices). The embedding dimension is split into `num_heads = 2` heads (2 dims each). Each head runs independent scaled dot-product attention; outputs are concatenated column-wise.
2. **Scaled Dot-Product Attention** — Computes `softmax((Q × Kᵀ) / √d_k) × V` per head.
3. **Residual Connection** — Original input `X` is added to the MHA output element-wise.
4. **LayerNorm** — Normalizes the residual sum row-by-row.

### Block 2 — Feed-Forward Network + Add & Norm

1. **FFN** — Two linear layers with a ReLU in between. `W1` expands from dim 4 → 8; `W2` compresses back 8 → 4. Both weight matrices are quantized to 4-bit before the forward pass.
2. **Quantized Matmul** — Uses the scalar unpack-and-multiply path (no SIMD) against the `QuantizedMatrix` structs.
3. **Residual Connection** — LayerNorm 1 output is added to the FFN output.
4. **LayerNorm** — Final normalization produces the encoder output.

---

## Build and Run

Requires a CPU with AVX2 and FMA support (Intel Haswell / AMD Ryzen or newer).

```bash
# Clone the repository
git clone https://github.com/yourusername/baremetal-transformer.git
cd baremetal-transformer

# Compile with AVX2 + FMA intrinsics and full optimization
g++ main.cpp -o main -mavx2 -mfma -O3

# Run the forward pass
./main
```

---

## Example Output

```
--- Initializing Bare-Metal Transformer Engine ---

Input Embeddings [2x4]:
1 0.5 -0.2 0.8
0.2 -1 0.5 0.1
-------------------------------
After Multi-Head Attention & LayerNorm 1 [2x4]:
1.22023 -0.384652 -1.42126 0.58568
0.592674 -1.72852 0.656935 0.478909
-------------------------------
Quantizing W1 and W2 to 4-bit...
Final Encoder Output Matrix [2x4]:
1.30094 -0.236201 -1.45022 0.385477
0.630434 -1.72749 0.639503 0.457548
-------------------------------
--- Forward Pass Complete ---
```

The slight numerical drift in the final output relative to Block 1 is quantization noise introduced by the 4-bit FFN weights.

---

## Known Simplifications

- No learned projection matrices for Q, K, V, or the post-attention output (W_O).
- No learnable γ/β parameters in LayerNorm.
- SIMD acceleration only covers dense float matmul; quantized matmul is scalar.
- No batching — processes a single sequence per forward pass.

---

## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS 2017*.