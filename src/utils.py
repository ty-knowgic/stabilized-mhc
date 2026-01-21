import torch
import torch.nn as nn
from .sprc import stabilized_rational_chart

def sinkhorn_knopp(u_raw, n_iters=20):
    """
    Standard Sinkhorn-Knopp algorithm for comparison.
    u_raw: (Batch, N, N)
    """
    matrix = torch.exp(u_raw)
    for _ in range(n_iters):
        matrix = matrix / (matrix.sum(dim=2, keepdim=True) + 1e-6)
        matrix = matrix / (matrix.sum(dim=1, keepdim=True) + 1e-6)
    return matrix

class ToyMHCLayer(nn.Module):
    """
    A toy layer to simulate Manifold-Constrained Hyper-Connections (mHC).
    Used for checking learning dynamics.
    """
    def __init__(self, dim, n=4, method='proposal'):
        super().__init__()
        self.dim = dim
        self.n = n
        self.method = method
        
        # Dimension check
        if dim % n != 0:
            raise ValueError(f"Dimension {dim} must be divisible by n={n}")

        # Parameter initialization
        if method == 'proposal':
            # SPRC takes (N-1)^2 params
            input_dim = (n - 1) ** 2
            self.param = nn.Parameter(torch.randn(1, input_dim) * 0.1)
        else:
            # Sinkhorn takes NxN params
            self.param = nn.Parameter(torch.randn(1, n, n) * 0.1)

        # Learnable linear layer
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, Seq, Dim)
        B_size = x.size(0)

        # Generate connection matrix H
        if self.method == 'proposal':
            u_exp = self.param.expand(B_size, -1)
            H = stabilized_rational_chart(u_exp) # (B, n, n)
        else:
            u_exp = self.param.expand(B_size, -1, -1)
            H = sinkhorn_knopp(u_exp) # (B, n, n)

        # mHC-like mixing operation
        chunk_dim = self.dim // self.n
        # Split: (B, S, n, D/n)
        x_split = x.view(B_size, -1, self.n, chunk_dim) 

        H = H.unsqueeze(1) # (B, 1, n, n)

        # Einstein Summation: batch(b), seq(s), out_mix(i), in_mix(j), dim(d)
        x_mixed = torch.einsum('b s j d, b s i j -> b s i d', x_split, H)
        x_mixed = x_mixed.reshape(B_size, -1, self.dim)

        return self.norm(x_mixed + self.linear(x))