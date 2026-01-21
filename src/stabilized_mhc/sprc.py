import torch
import torch.nn.functional as F
import math

def stabilized_rational_chart(u, epsilon=1e-6, lambd=1.0, smoothness=1.0):
    """
    Stabilized Piecewise-Rational Chart (SPRC)

    A deterministic, iteration-free mapping from R^{(N-1)^2} to the Birkhoff Polytope B_N.
    
    Args:
        u (torch.Tensor): Input parameters of shape (Batch, (N-1)^2).
                          For N=4, input dim is 9. For N=8, input dim is 49.
        epsilon (float): Small constant for numerical stability.
        lambd (float): Controls the saturation rate of tanh.
        smoothness (float): Temperature for LogSumExp approximation.
        
    Returns:
        torch.Tensor: Doubly stochastic matrices of shape (Batch, N, N).
    """
    B = u.shape[0]
    dim_u = u.shape[-1]
    
    # 1. Determine dimension N from input vector size
    # u has size (N-1)^2.  m = N-1
    m = int(math.sqrt(dim_u))
    n = m + 1
    
    if m * m != dim_u:
        raise ValueError(f"Input dimension {dim_u} must be a perfect square ((N-1)^2). E.g., 9 for N=4.")

    # 2. Tangent Space Projection (Generalized for N)
    # Map R^{(N-1)^2} -> R^(NxN) with zero row/col sums
    
    # Reshape u to core matrix (Batch, N-1, N-1)
    u_core = u.view(B, m, m)
    
    # Calculate sums for the boundaries
    row_sum = u_core.sum(dim=2) # (Batch, N-1)
    col_sum = u_core.sum(dim=1) # (Batch, N-1)
    total_sum = row_sum.sum(dim=1, keepdim=True) # (Batch, 1) -> value for corner
    
    # Construct V using padding
    # Start with a zero matrix of (Batch, N, N)
    V = torch.zeros(B, n, n, device=u.device, dtype=u.dtype)
    
    # Fill the core (N-1)x(N-1)
    V[:, :m, :m] = u_core
    
    # Fill the last column (excluding bottom-right)
    V[:, :m, m] = -row_sum
    
    # Fill the last row (excluding bottom-right)
    V[:, m, :m] = -col_sum
    
    # Fill the bottom-right corner
    V[:, m, m] = total_sum.squeeze(-1)

    # 3. Smooth Tropical Norm (Distance to boundary)
    # Use LogSumExp(-V) for smoothness
    neg_V = -V
    # Flatten strictly for the LogSumExp calculation over all elements
    m_V_smooth = smoothness * torch.logsumexp(neg_V.view(B, -1) / smoothness, dim=1).view(B, 1, 1)
    
    # 4. Progressive Saturation
    # Scale: 0 (at center) -> 1 (at boundary) based on input magnitude
    V_norm = torch.norm(V.view(B, -1), p=2, dim=1).view(B, 1, 1)
    saturation = torch.tanh(lambd * V_norm)
    
    # 5. Construct H
    # Center point J_N is matrix with all elements 1/N
    J_val = 1.0 / n
    
    # Calculate dynamic scaling factor
    scale = (J_val * saturation) / (m_V_smooth + epsilon)
    
    H = J_val + scale * V
    
    return H
