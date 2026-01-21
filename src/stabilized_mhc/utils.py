"""Utility helpers for diagnostics and stress testing."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import torch


def infer_n_from_u(u_dim: int) -> int:
    """Infer N from the flattened input dimension where (N-1)^2 == u_dim."""
    if u_dim <= 0:
        raise ValueError(f"u_dim must be positive, got {u_dim}.")
    m = int(math.isqrt(u_dim))
    if m * m != u_dim:
        raise ValueError(
            f"Input dimension {u_dim} must be a perfect square (N-1)^2."
        )
    return m + 1


def check_doubly_stochastic(
    H: torch.Tensor, *, atol_sum: float, atol_nonneg: float
) -> dict:
    """Return diagnostic metrics for doubly stochastic constraints."""
    row_sums = H.sum(dim=-1)
    col_sums = H.sum(dim=-2)

    max_row_err = (row_sums - 1.0).abs().max().item()
    max_col_err = (col_sums - 1.0).abs().max().item()
    min_entry = H.min().item()

    return {
        "max_row_err": max_row_err,
        "max_col_err": max_col_err,
        "min_entry": min_entry,
        "row_ok": max_row_err <= atol_sum,
        "col_ok": max_col_err <= atol_sum,
        "nonneg_ok": min_entry >= -atol_nonneg,
    }


def stress_inputs(
    shape: Tuple[int, int],
    scales: Iterable[float],
    device: torch.device | str,
    dtype: torch.dtype,
    seed: int,
) -> List[Tuple[float, torch.Tensor]]:
    """Generate scaled random inputs for stress testing."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    inputs: List[Tuple[float, torch.Tensor]] = []
    for scale in scales:
        base = torch.randn(*shape, generator=gen, dtype=torch.float32)
        u = (base * float(scale)).to(device=device, dtype=dtype)
        inputs.append((float(scale), u))
    return inputs
