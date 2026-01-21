import torch

from stabilized_mhc import stabilized_rational_chart
from stabilized_mhc.utils import check_doubly_stochastic, infer_n_from_u, stress_inputs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SPRC Stress Verify (device={device}) ===")

    n_values = [8, 16, 32, 64]
    dtypes = [torch.float32, torch.bfloat16]
    scales = [1, 10, 1e2, 1e3]

    for dtype in dtypes:
        atol_sum = 1e-5 if dtype == torch.float32 else 5e-4
        atol_nonneg = 1e-6
        for n in n_values:
            input_dim = (n - 1) ** 2
            inputs = stress_inputs((32, input_dim), scales, device, dtype, seed=2024)
            for scale, u in inputs:
                H = stabilized_rational_chart(u)
                inferred = infer_n_from_u(u.shape[-1])
                diag = check_doubly_stochastic(
                    H, atol_sum=atol_sum, atol_nonneg=atol_nonneg
                )
                print(
                    "N={n} inferred={inf} scale={scale:g} dtype={dtype} "
                    "row_err={row:.2e} col_err={col:.2e} min={minv:.2e}".format(
                        n=n,
                        inf=inferred,
                        scale=scale,
                        dtype=str(dtype).replace("torch.", ""),
                        row=diag["max_row_err"],
                        col=diag["max_col_err"],
                        minv=diag["min_entry"],
                    )
                )


if __name__ == "__main__":
    main()
