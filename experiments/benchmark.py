import torch
import time
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sprc import stabilized_rational_chart

def sinkhorn_knopp(u_raw, n_iters=20):
    # Baseline implementation
    matrix = torch.exp(u_raw)
    for _ in range(n_iters):
        matrix = matrix / (matrix.sum(dim=2, keepdim=True) + 1e-6)
        matrix = matrix / (matrix.sum(dim=1, keepdim=True) + 1e-6)
    return matrix

def run_benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    device = 'cuda'
    batches = [1024, 4096, 16384, 65536]
    
    print(f"{'Batch Size':<12} | {'Sinkhorn (ms)':<15} | {'SPRC (ms)':<15} | {'Speedup':<10}")
    print("-" * 60)
    
    for B in batches:
        # Prepare inputs
        u_sprc = torch.randn(B, 9, device=device)
        u_sink = torch.randn(B, 4, 4, device=device)
        
        # Warmup
        for _ in range(10):
            _ = stabilized_rational_chart(u_sprc)
            _ = sinkhorn_knopp(u_sink)
            
        torch.cuda.synchronize()
        
        # Benchmark Sinkhorn
        start = time.time()
        for _ in range(100):
            _ = sinkhorn_knopp(u_sink)
        torch.cuda.synchronize()
        t_sink = (time.time() - start) / 100 * 1000
        
        # Benchmark SPRC
        start = time.time()
        for _ in range(100):
            _ = stabilized_rational_chart(u_sprc)
        torch.cuda.synchronize()
        t_sprc = (time.time() - start) / 100 * 1000
        
        print(f"{B:<12} | {t_sink:.3f} {'ms':<12} | {t_sprc:.3f} {'ms':<12} | {t_sink/t_sprc:.1f}x")

if __name__ == "__main__":
    run_benchmark()