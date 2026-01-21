import argparse
from typing import Iterable

import torch

from stabilized_mhc import stabilized_rational_chart
from stabilized_mhc.utils import check_doubly_stochastic, infer_n_from_u, stress_inputs


def _dtype_from_arg(name: str) -> torch.dtype:
    mapping = {
        "float64": torch.float64,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return mapping[name]


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float64:
        return 1e-14, 1e-12
    if dtype == torch.float32:
        return 1e-5, 1e-7
    if dtype in (torch.bfloat16, torch.float16):
        return 5e-4, 1e-6
    raise ValueError(f"Unsupported dtype {dtype}.")


def _run_checks(u: torch.Tensor, *, atol_sum: float, atol_nonneg: float) -> None:
    n = infer_n_from_u(u.shape[-1])
    B = u.shape[0]

    H = stabilized_rational_chart(u)

    if H.shape != (B, n, n):
        raise AssertionError(f"Shape mismatch: expected ({B}, {n}, {n}), got {H.shape}.")

    diag = check_doubly_stochastic(H, atol_sum=atol_sum, atol_nonneg=atol_nonneg)
    if not (diag["row_ok"] and diag["col_ok"] and diag["nonneg_ok"]):
        raise AssertionError(
            "Constraint check failed: "
            f"row_err={diag['max_row_err']:.2e}, "
            f"col_err={diag['max_col_err']:.2e}, "
            f"min_entry={diag['min_entry']:.2e}"
        )


def _gradcheck(u_dim: int, device: torch.device) -> None:
    u = torch.randn(2, u_dim, device=device, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(
        lambda x: stabilized_rational_chart(x).sum(),
        (u,),
        eps=1e-6,
        atol=1e-4,
    )


def verify_standard(device: torch.device, dtype: torch.dtype) -> None:
    print("=== SPRC Mathematical Verification ===")
    n_values = [4, 8, 16]
    atol_sum, atol_nonneg = _tolerances(dtype)

    for n in n_values:
        input_dim = (n - 1) ** 2
        u = torch.randn(128, input_dim, device=device, dtype=dtype)
        _run_checks(u, atol_sum=atol_sum, atol_nonneg=atol_nonneg)
        print(f"[PASS] N={n} constraints ok (dtype={dtype}).")

    _gradcheck((n_values[0] - 1) ** 2, device=device)
    print("[PASS] Gradcheck.")


def verify_stress(device: torch.device) -> None:
    print("=== SPRC Stress Verification ===")
    scales = [1, 10, 1e2, 1e3]
    dtypes = [torch.float64, torch.float32, torch.bfloat16, torch.float16]
    n_values = [8, 16]

    for dtype in dtypes:
        atol_sum, atol_nonneg = _tolerances(dtype)
        for n in n_values:
            input_dim = (n - 1) ** 2
            inputs = stress_inputs((64, input_dim), scales, device, dtype, seed=1234)
            for scale, u in inputs:
                _run_checks(u, atol_sum=atol_sum, atol_nonneg=atol_nonneg)
                print(f"[PASS] N={n} scale={scale:g} dtype={dtype}.")

    _gradcheck((n_values[0] - 1) ** 2, device=device)
    print("[PASS] Gradcheck (float64).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify SPRC constraints and gradients.")
    parser.add_argument("--stress", action="store_true", help="Run stress mode.")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda/mps).")
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float64", "float32", "bfloat16", "float16"],
        help="dtype for standard verification.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.stress:
        verify_stress(device)
    else:
        dtype = _dtype_from_arg(args.dtype)
        verify_standard(device, dtype)
