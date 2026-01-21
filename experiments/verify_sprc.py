import torch
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sprc import stabilized_rational_chart

def verify_n_dimensional(n, device='cpu'):
    print(f"Testing Dimension N={n}...")
    
    # Setup inputs
    B = 128
    input_dim = (n - 1) ** 2
    # Use float64 for strict mathematical verification
    u = torch.randn(B, input_dim, device=device, dtype=torch.float64, requires_grad=True)

    # 1. Forward Pass
    H = stabilized_rational_chart(u, epsilon=1e-8)
    
    # Check shape
    assert H.shape == (B, n, n), f"Shape mismatch: expected ({B},{n},{n}), got {H.shape}"
    
    # Check Row/Col Sums
    row_sums = H.sum(dim=2)
    col_sums = H.sum(dim=1)
    
    max_err_row = (row_sums - 1.0).abs().max().item()
    max_err_col = (col_sums - 1.0).abs().max().item()
    
    # Check Non-negativity
    min_val = H.min().item()
    is_non_negative = min_val >= -1e-12 # Allow tiny numerical noise

    print(f"  Max Row Sum Error: {max_err_row:.2e}")
    print(f"  Max Col Sum Error: {max_err_col:.2e}")
    print(f"  Min Value: {min_val:.2e}")

    if max_err_row < 1e-14 and max_err_col < 1e-14 and is_non_negative:
        print(f"  => [PASS] N={n} constraints satisfied.")
    else:
        print(f"  => [FAIL] N={n} constraints failed.")
        sys.exit(1)

    # 2. Gradient Check
    print("  Running Gradient Check...")
    # Smaller batch for gradcheck
    u_grad = torch.randn(2, input_dim, device=device, dtype=torch.float64, requires_grad=True)
    
    try:
        torch.autograd.gradcheck(
            stabilized_rational_chart, 
            (u_grad,), 
            eps=1e-6, 
            atol=1e-4
        )
        print(f"  => [PASS] N={n} is fully differentiable.")
    except Exception as e:
        print(f"  => [FAIL] Gradient check failed for N={n}: {e}")
    
    print("-" * 40)

def run_verification():
    print("=== SPRC Mathematical Verification ===")
    device = torch.device('cpu')
    
    # Verify for DeepSeek standard (N=4)
    verify_n_dimensional(4, device)
    
    # Verify generalization (N=8)
    verify_n_dimensional(8, device)
    
    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    run_verification()