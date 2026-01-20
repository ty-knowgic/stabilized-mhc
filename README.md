# Stabilized Piecewise-Rational Charts (SPRC) for mHC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/ty-knowgic/stabilized-mhc/blob/main/experiments/reproduction_demo.ipynb)

> **A mathematically exact, iteration-free, and highly efficient replacement for Sinkhorn-based Manifold-Constrained Hyper-Connections (mHC).**

DeepSeek-V3 introduced mHC to preserve the identity mapping property in residual streams. However, the standard implementation relies on the iterative Sinkhorn-Knopp algorithm, which causes memory bandwidth saturation and high kernel latency.

**SPRC** solves this algebraically via **Smooth Parametrization**, achieving theoretical near-optimal performance.

## 🚀 Key Performance Results

Benchmarks performed on NVIDIA Tesla T4 GPU.

| Batch Size | Method | Kernel Time | Speedup | End-to-End Training |
| :--- | :--- | :--- | :--- | :--- |
| **65,536** | Sinkhorn ($t=20$) | 15.43 ms | 1.0x | 1.0x |
| **65,536** | **SPRC (Ours)** | **1.16 ms** | **13.3x** ⚡ | **2.53x** 🚀 |

### Exact Convergence
SPRC matches the learning dynamics of Sinkhorn almost perfectly, ensuring no loss in model expressivity.

![Loss Convergence](assets/loss_plot.png)

## 💡 Why Use SPRC?

1.  **13x Faster Kernel:** Replaces $O(T \cdot N^2)$ iterative loops with an $O(1)$ closed-form calculation.
2.  **Memory Efficient:** Eliminates intermediate memory reads/writes required by Sinkhorn iterations.
3.  **Mathematically Exact:** Guarantees **Doubly Stochastic** (row/col sum = 1) and **Non-Negative** constraints by construction ($10^{-16}$ precision).
4.  **Differentiable:** Uses smooth `LogSumExp` and `tanh` saturation to ensure stable gradients.

## 🛠️ Usage

SPRC is designed as a drop-in replacement for mHC layers.

```python
import torch
from src.sprc import stabilized_rational_chart

# Input parameters (Batch, 9) for n=4 mHC
u = torch.randn(64, 9, device='cuda', requires_grad=True)

# Generate Doubly Stochastic Matrix H (Batch, 4, 4)
# No loop, No approximation error.
H = stabilized_rational_chart(u)

print(H.shape) # torch.Size([64, 4, 4])