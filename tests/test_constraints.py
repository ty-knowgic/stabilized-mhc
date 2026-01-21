import torch

from stabilized_mhc import stabilized_rational_chart
from stabilized_mhc.utils import check_doubly_stochastic


def test_constraints_float32():
    n = 8
    batch = 2
    u = torch.randn(batch, (n - 1) ** 2, dtype=torch.float32)
    H = stabilized_rational_chart(u)
    diag = check_doubly_stochastic(H, atol_sum=1e-5, atol_nonneg=1e-7)
    assert diag["row_ok"], diag
    assert diag["col_ok"], diag
    assert diag["nonneg_ok"], diag


def test_constraints_bfloat16_cpu():
    n = 8
    batch = 2
    u = torch.randn(batch, (n - 1) ** 2, dtype=torch.bfloat16)
    H = stabilized_rational_chart(u)
    diag = check_doubly_stochastic(H, atol_sum=5e-3, atol_nonneg=5e-4)
    assert diag["row_ok"], diag
    assert diag["col_ok"], diag
    assert diag["nonneg_ok"], diag
