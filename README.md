# Bare-Metal Attention

A zero-dependency, bare-metal C++ implementation of the core Scaled Dot-Product Attention mechanism from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## Motivation
This repository bridges the gap between academic theory and low-level inference architecture. Instead of relying on high-level wrappers like PyTorch or optimized linear algebra libraries like Eigen/BLAS, this project implements the mathematical primitives of Large Language Models from scratch using the C++ standard library. 

This is an exploratory engine designed to understand the exact matrix routing, masking constraints, and memory behavior that power inference engines like `llama.cpp` under the hood.

## Features
* **Zero Dependencies:** Pure C++ standard library implementation.
* **Numerically Stable Softmax:** Implements max-value subtraction (`exp(x - max)`) to prevent `NaN` overflow during raw dot-product exponentiation.
* **Autoregressive Masking:** Includes lower-triangular masking logic for decoder-only architectures.
* **Custom Matrix Math:** Hand-rolled $O(N^3)$ matrix multiplication, transposition, and scalar scaling primitives.

## The Architecture
The core pipeline processes Query ($Q$), Key ($K$), and Value ($V$) matrices through the exact formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

1. **Dot Product ($QK^T$):** Computes raw alignment scores between token embeddings.
2. **Scale ($\sqrt{d_k}$):** Normalizes variance to prevent gradient vanishing in the softmax layer.
3. **Mask (Optional):** Applies $-\infty$ to future tokens for causal language modeling.
4. **Softmax:** Converts unbounded scores into a clean probability distribution.
5. **Contextualization:** Computes the weighted average of the Value ($V$) matrix.

## Build and Run

Since this relies purely on standard C++, you can compile it directly from your terminal using `g++` or `clang++`.

```bash
# Clone the repository
git clone [https://github.com/yourusername/bare-metal-attention.git](https://github.com/yourusername/baremetal-attention.git)
cd baremetal-attention

# Compile the engine
g++ main.cpp -o main

# Execute the forward pass
./main