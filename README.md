# Stabilized-mHC (SPRC)

**An iteration-free alternative to Sinkhorn for mHC-style routing.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**SPRC (Stabilized Piecewise-Rational Chart)** provides an iteration-free mapping from
$\mathbb{R}^{(N-1)^2}$ to the Birkhoff Polytope $\mathcal{B}_N$ (doubly stochastic matrices).
It eliminates the iterative overhead and non-deterministic latency of Sinkhorn-Knopp,
with a fixed cost with respect to the number of Sinkhorn iterations.

<p align="center">
  <img src="assets/loss_convergence_n8.png" width="600" alt="Training Convergence N=8">
  <br>
  <em>Figure: Example training dynamics comparison against Sinkhorn (N=8).</em>
</p>

## 🚀 Key Features

* **Iteration-free:** No iterative Sinkhorn loop and no iteration jitter.
* **N-dimensional:** Supports arbitrary $N$ with input dimension $(N-1)^2$.
* **Deterministic mapping:** The computation is algebraic and deterministic.
* **Constraints in practice:** Row/column sums and non-negativity are satisfied within
  floating-point tolerances.

## ✅ Guaranteed Properties

* Outputs are (approximately) doubly stochastic within floating-point tolerances.
* Runtime is independent of the number of Sinkhorn iterations (because none are used).
* Gradients are supported by PyTorch autograd.

## 🚫 Non-Claims

* Not a claim of bijectivity between $\mathbb{R}^{(N-1)^2}$ and $\mathcal{B}_N$.
* Not a claim of $O(1)$ complexity in $N$.
* Not a claim of matching Sinkhorn for every downstream training run or task.

## 📦 Installation

```bash
git clone https://github.com/ty-knowgic/stabilized-mhc.git
cd stabilized-mhc
pip install -e .
```

## 💻 Usage

```python
import torch
from stabilized_mhc import stabilized_rational_chart

u = torch.randn(32, 9).cuda()  # (N-1)^2 for N=4
H = stabilized_rational_chart(u)
print(H.shape)  # torch.Size([32, 4, 4])
```

### Supporting General N

```python
u_large = torch.randn(32, 49).cuda()  # (8-1)^2
H_large = stabilized_rational_chart(u_large)
print(H_large.shape)  # torch.Size([32, 8, 8])
```

## 📊 Reproduction & Experiments

### 1. Mathematical Verification
```bash
python experiments/verify.py
python experiments/verify.py --stress
```

### 2. Benchmarks
Benchmarks include optimized baselines and optional `torch.compile` paths.
```bash
python benchmarks/benchmark_sinkhorn.py
```

### 3. Learning Dynamics Check
```bash
python experiments/visualize.py
```

## 📜 Algorithm Details

Unlike Sinkhorn which solves constraints iteratively:
$$ \min_{P \in \mathcal{U}(r, c)} \langle P, -U \rangle - \epsilon H(P) $$

SPRC constructs the matrix algebraically using a **Smooth Tropical Norm** and
**Piecewise-Rational Chart**:

1. **Tangent Projection:** Map $\mathbb{R}^{(N-1)^2}$ to a zero-sum matrix $V$.
2. **Tropical Stabilization:** Calculate distance to the polytope boundary with `logsumexp`.
3. **Rational Map:** $H = J_N + \frac{V}{m(V) + \epsilon}$

See `src/stabilized_mhc/sprc.py` for implementation details.

## 🤝 Contributing

Open to PRs! We are especially interested in:
* FPGA / Verilog implementations.
* Integration tests with full LLM training pipelines.

## License

MIT License.
