import argparse
import time

import torch

from stabilized_mhc import stabilized_rational_chart


def sinkhorn_knopp(u_raw, n_iters=20):
    matrix = torch.exp(u_raw)
    for _ in range(n_iters):
        matrix = matrix / (matrix.sum(dim=2, keepdim=True) + 1e-6)
        matrix = matrix / (matrix.sum(dim=1, keepdim=True) + 1e-6)
    return matrix


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_it(fn, iters: int, device: torch.device) -> float:
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SPRC vs Sinkhorn")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default=None, help="cpu/cuda/mps")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    batch_size = 16384 if device.type == "cuda" else 4096 if device.type == "mps" else 512

    print(f"=== Benchmark on {device.type.upper()} ===")
    print(f"Batch Size: {batch_size}")
    print(f"Iters: {args.iters}, Sinkhorn iters: {args.sinkhorn_iters}")
    print("-" * 80)
    print(f"{'N':<6} | {'Method':<16} | {'Kernel Time (ms)':<18} | {'Speedup':<10}")
    print("-" * 80)

    dimensions = [4, 8, 16]

    sinkhorn = lambda x: sinkhorn_knopp(x, n_iters=args.sinkhorn_iters)
    sprc = stabilized_rational_chart

    if args.compile and hasattr(torch, "compile"):
        sinkhorn = torch.compile(sinkhorn)
        sprc = torch.compile(sprc)

    for n in dimensions:
        input_dim = (n - 1) ** 2
        u_sprc = torch.randn(batch_size, input_dim, device=device)
        u_sink = torch.randn(batch_size, n, n, device=device)

        for _ in range(5):
            _ = sprc(u_sprc)
            _ = sinkhorn(u_sink)
        _sync(device)

        t_sink = _time_it(lambda: sinkhorn(u_sink), args.iters, device)
        t_sprc = _time_it(lambda: sprc(u_sprc), args.iters, device)

        print(f"{n:<6} | {'Sinkhorn':<16} | {t_sink:>8.3f} ms{'':<6} | 1.0x")
        print(
            f"{n:<6} | {'SPRC':<16} | {t_sprc:>8.3f} ms{'':<6} | {t_sink / t_sprc:>4.1f}x"
        )
        print("-" * 80)


if __name__ == "__main__":
    main()
