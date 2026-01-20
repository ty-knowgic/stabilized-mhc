import torch
import torch.nn.functional as F

def stabilized_rational_chart(u, epsilon=1e-6, lambd=1.0, smoothness=1.0):
    """
    Stabilized Piecewise-Rational Chart (SPRC) for Manifold-Constrained Hyper-Connections (mHC).
    
    Args:
        u (torch.Tensor): Input parameters (Batch, 9).
        epsilon (float): Small constant for numerical stability.
        lambd (float): Controls the saturation rate of tanh.
        smoothness (float): Temperature for LogSumExp approximation (eta).
        
    Returns:
        torch.Tensor: Doubly stochastic matrices (Batch, 4, 4).
    """
    B = u.shape[0]
    
    # 1. Tangent Space Projection (O(1) linear map)
    # Map R^9 -> R^(4x4) with zero row/col sums
    u_3x3 = u.view(B, 3, 3)
    row_sum = u_3x3.sum(dim=2, keepdim=True)
    col_sum = u_3x3.sum(dim=1, keepdim=True)
    total_sum = row_sum.sum(dim=1, keepdim=True)
    
    top = torch.cat([u_3x3, -row_sum], dim=2)
    bot = torch.cat([-col_sum, total_sum], dim=2)
    V = torch.cat([top, bot], dim=1) # (B, 4, 4) zero-sum

    # 2. Smooth Tropical Norm (Distance to boundary)
    # Use LogSumExp(-V) for smoothness instead of max(-V)
    neg_V = -V
    m_V_smooth = smoothness * torch.logsumexp(neg_V.view(B, 16) / smoothness, dim=1).view(B, 1, 1)
    
    # 3. Progressive Saturation
    # Scale: 0 (at center) -> 1 (at boundary) based on input magnitude
    V_norm = torch.norm(V.view(B, 16), p=2, dim=1).view(B, 1, 1)
    saturation = torch.tanh(lambd * V_norm)
    
    # 4. Construct H
    # H = J + saturation * (Distance_to_Boundary * V_direction)
    # This guarantees non-negativity and sum=1 constraints by construction.
    scale = (0.25 * saturation) / (m_V_smooth + epsilon)
    
    H = 0.25 + scale * V
    
    return H