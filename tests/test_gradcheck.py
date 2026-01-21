import torch

from stabilized_mhc import stabilized_rational_chart


def test_gradcheck():
    n = 6
    u = torch.randn(1, (n - 1) ** 2, dtype=torch.float64, requires_grad=True)

    torch.autograd.gradcheck(
        lambda x: stabilized_rational_chart(x).sum(),
        (u,),
        eps=1e-6,
        atol=1e-4,
    )
