import torch
import time
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sprc import stabilized_rational_chart
from src.utils import sinkhorn_knopp

def run_benchmark():
    # Automatic device detection
    if torch.cuda.is_available():
        device = 'cuda'
        batch_size = 16384
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = 'mps'
        batch_size = 4096
        device_name = "Apple Silicon (MPS)"
    else:
        device = 'cpu'
        batch_size = 256
        device_name = "CPU"

    print(f"=== Running Benchmark on {device_name} ===")
    print(f"Batch Size: {batch_size}")
    print("-" * 75)
    print(f"{'N':<4} | {'Method':<10} | {'Kernel Time (ms)':<18} | {'Speedup':<10}")
    print("-" * 75)
    
    dimensions = [4, 8, 16] # Test generalization capability
    
    for n in dimensions:
        input_dim = (n - 1) ** 2
        
        # Prepare inputs
        u_sprc = torch.randn(batch_size, input_dim, device=device)
        u_sink = torch.randn(batch_size, n, n, device=device)
        
        # Sync helper
        def sync():
            if device == 'cuda': torch.cuda.synchronize()
            elif device == 'mps': torch.mps.synchronize()

        # Warmup
        for _ in range(5):
            _ = stabilized_rational_chart(u_sprc)
            _ = sinkhorn_knopp(u_sink)
        sync()
        
        # Benchmark Sinkhorn
        start = time.time()
        for _ in range(100):
            _ = sinkhorn_knopp(u_sink)
        sync()
        t_sink = (time.time() - start) / 100 * 1000
        
        # Benchmark SPRC
        start = time.time()
        for _ in range(100):
            _ = stabilized_rational_chart(u_sprc)
        sync()
        t_sprc = (time.time() - start) / 100 * 1000
        
        print(f"{n:<4} | Sinkhorn   | {t_sink:.3f} {'ms':<15} | 1.0x")
        print(f"{'':<4} | SPRC (Ours)| {t_sprc:.3f} {'ms':<15} | {t_sink/t_sprc:.1f}x")
        print("-" * 75)

if __name__ == "__main__":
    run_benchmark()