import torch
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sprc import stabilized_rational_chart

def verify_n_dimensional(n, device='cpu'):
    print(f"Testing Dimension N={n}...", end=" ")
    
    # Setup inputs
    B = 128
    input_dim = (n - 1) ** 2
    # Use float64 for strict mathematical verification
    u = torch.randn(B, input_dim, device=device, dtype=torch.float64, requires_grad=True)

    # 1. Forward Pass Check
    H = stabilized_rational_chart(u)
    
    # Check shape
    if H.shape != (B, n, n):
        print(f"[FAIL] Shape mismatch: expected ({B},{n},{n}), got {H.shape}")
        return

    # Check Row/Col Sums
    row_sums = H.sum(dim=2)
    col_sums = H.sum(dim=1)
    
    max_err_row = (row_sums - 1.0).abs().max().item()
    max_err_col = (col_sums - 1.0).abs().max().item()
    
    # Check Non-negativity
    min_val = H.min().item()
    is_non_negative = min_val >= -1e-12

    if max_err_row < 1e-14 and max_err_col < 1e-14 and is_non_negative:
        print(f"[PASS] Constraints OK (Err < 1e-14).", end=" ")
    else:
        print(f"[FAIL] Constraints violated. RowErr: {max_err_row:.1e}, ColErr: {max_err_col:.1e}")
        return

    # 2. Gradient Check
    try:
        # Smaller input for gradcheck
        u_grad = torch.randn(2, input_dim, device=device, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            stabilized_rational_chart, 
            (u_grad,), 
            eps=1e-6, 
            atol=1e-4
        )
        print(f"[PASS] Differentiable.")
    except Exception as e:
        print(f"[FAIL] Gradient check failed: {e}")

if __name__ == "__main__":
    print("=== SPRC Mathematical Verification ===")
    device = 'cpu' # CPU is preferred for double precision check
    
    verify_n_dimensional(4, device)
    verify_n_dimensional(8, device)
    verify_n_dimensional(16, device)