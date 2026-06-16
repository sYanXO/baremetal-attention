# Bare-Metal Transformer Engine

A zero-dependency, bare-metal C++ implementation of the core Transformer encoder architecture from the paper "Attention Is All You Need" (Vaswani et al., 2017).

## Motivation

This repository bridges the gap between academic theory and low-level inference architecture. Instead of relying on high-level wrappers like PyTorch or optimized linear algebra libraries like Eigen or BLAS, this project implements the mathematical primitives of Large Language Models from scratch using the C++ standard library. 

This is an exploratory engine designed to understand the exact matrix routing, masking constraints, and memory behavior that power hardware-optimized inference engines under the hood.

## Features

* **Zero Dependencies:** Pure C++ standard library implementation.
* **Numerically Stable Softmax:** Implements max-value subtraction, `exp(x - max)`, to prevent `NaN` overflow during raw dot-product exponentiation.
* **Autoregressive Masking:** Includes lower-triangular masking logic for causal language modeling constraints.
* **Custom Matrix Math:** Hand-rolled O(N^3) matrix multiplication, transposition, scalar scaling, and element-wise addition primitives.
* **Manual Memory Routing:** Custom slicing and concatenation functions to manage Multi-Head Attention without high-level tensor manipulation APIs.
* **Non-Linear Activations:** Custom ReLU implementation to handle feature extraction in the Feed-Forward block.
* **Variance Stabilization:** Bare-metal Layer Normalization to bound output distributions and prevent gradient/activation explosions.

## The Architecture

The pipeline processes input embeddings through a complete standard Encoder block:

1. **Scaled Dot-Product Attention:** The core mathematical engine that computes alignment scores between Query, Key, and Value matrices, scaling by the square root of the dimension to maintain gradient flow.
2. **Multi-Head Attention:** Slices the matrices into independent attention heads, computes the scaled dot-product for each, and stitches the memory blocks back together.
3. **Residual Connections & LayerNorm:** Adds the original input to the output of the attention mechanism and normalizes the variance to stabilize deep network flow.
4. **Feed-Forward Network (FFN):** A two-layer perceptron with a ReLU activation that expands the embedding dimension to process contextual relationships before compressing it back.
5. **Final Add & Norm:** A second residual connection and normalization step to complete the encoder block.

## Build and Run

Since this relies purely on standard C++, you can compile it directly from your terminal using `g++` or `clang++`.

```bash
# Clone the repository
git clone [https://github.com/yourusername/baremetal-transformer.git](https://github.com/yourusername/baremetal-transformer.git)
cd baremetal-transformer

# Compile the engine
g++ main.cpp -o main

# Execute the forward pass
./main