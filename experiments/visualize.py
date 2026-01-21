import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from stabilized_mhc import stabilized_rational_chart


def sinkhorn_knopp(u_raw, n_iters=20):
    matrix = torch.exp(u_raw)
    for _ in range(n_iters):
        matrix = matrix / (matrix.sum(dim=2, keepdim=True) + 1e-6)
        matrix = matrix / (matrix.sum(dim=1, keepdim=True) + 1e-6)
    return matrix


class ToyMHCLayer(nn.Module):
    def __init__(self, dim, n=4, method="proposal"):
        super().__init__()
        self.dim = dim
        self.n = n
        self.method = method

        if dim % n != 0:
            raise ValueError(f"Dimension {dim} must be divisible by n={n}")

        if method == "proposal":
            input_dim = (n - 1) ** 2
            self.param = nn.Parameter(torch.randn(1, input_dim) * 0.1)
        else:
            self.param = nn.Parameter(torch.randn(1, n, n) * 0.1)

        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B_size = x.size(0)

        if self.method == "proposal":
            u_exp = self.param.expand(B_size, -1)
            H = stabilized_rational_chart(u_exp)
        else:
            u_exp = self.param.expand(B_size, -1, -1)
            H = sinkhorn_knopp(u_exp)

        chunk_dim = self.dim // self.n
        x_split = x.view(B_size, -1, self.n, chunk_dim)
        H = H.unsqueeze(1)
        x_mixed = torch.einsum("b s j d, b s i j -> b s i d", x_split, H)
        x_mixed = x_mixed.reshape(B_size, -1, self.dim)

        return self.norm(x_mixed + self.linear(x))


def run_visualization():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"=== Learning Dynamics Check on {device.upper()} ===")

    N_TEST = 8
    DIM = 256
    STEPS = 500
    BATCH = 64
    LR = 1e-3

    print(f"Config: N={N_TEST}, Steps={STEPS}, Batch={BATCH}")

    models = {
        "Sinkhorn": ToyMHCLayer(DIM, n=N_TEST, method="sinkhorn").to(device),
        "Proposal": ToyMHCLayer(DIM, n=N_TEST, method="proposal").to(device),
    }

    optimizers = {name: optim.AdamW(model.parameters(), lr=LR) for name, model in models.items()}

    history = {"Sinkhorn": [], "Proposal": []}

    print("Training...")
    for step in range(STEPS):
        x = torch.randn(BATCH, 16, DIM, device=device)
        target = x.clone()

        for name, model in models.items():
            optimizers[name].zero_grad()
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizers[name].step()
            history[name].append(loss.item())

        if step % 100 == 0:
            print(
                f"Step {step}: Sinkhorn={history['Sinkhorn'][-1]:.4f}, "
                f"Proposal={history['Proposal'][-1]:.4f}"
            )

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    save_path = os.path.join(assets_dir, f"loss_convergence_n{N_TEST}.png")

    print(f"Plotting results to {save_path}...")
    plt.figure(figsize=(10, 5))
    plt.plot(history["Sinkhorn"], label="Sinkhorn (Baseline)", alpha=0.7)
    plt.plot(
        history["Proposal"],
        label=f"Proposal (SPRC, N={N_TEST})",
        linewidth=2,
        linestyle="--",
    )
    plt.title(f"Training Loss Convergence (N={N_TEST})")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    print("Done.")


if __name__ == "__main__":
    run_visualization()
