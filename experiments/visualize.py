import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import ToyMHCLayer

def run_visualization():
    # Device detection
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"=== Learning Dynamics Check on {device.upper()} ===")
    
    # Experiment Config
    N_TEST = 8 
    DIM = 256
    STEPS = 500
    BATCH = 64
    LR = 1e-3

    print(f"Config: N={N_TEST}, Steps={STEPS}, Batch={BATCH}")

    # Initialize Models
    models = {
        'Sinkhorn': ToyMHCLayer(DIM, n=N_TEST, method='sinkhorn').to(device),
        'Proposal': ToyMHCLayer(DIM, n=N_TEST, method='proposal').to(device)
    }

    optimizers = {
        name: optim.AdamW(model.parameters(), lr=LR) for name, model in models.items()
    }

    history = {'Sinkhorn': [], 'Proposal': []}

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
            print(f"Step {step}: Sinkhorn={history['Sinkhorn'][-1]:.4f}, Proposal={history['Proposal'][-1]:.4f}")

    # Plotting
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    save_path = os.path.join(assets_dir, f'loss_convergence_n{N_TEST}.png')

    print(f"Plotting results to {save_path}...")
    plt.figure(figsize=(10, 5))
    plt.plot(history['Sinkhorn'], label='Sinkhorn (Baseline)', alpha=0.7)
    plt.plot(history['Proposal'], label=f'Proposal (SPRC, N={N_TEST})', linewidth=2, linestyle='--')
    plt.title(f'Training Loss Convergence (N={N_TEST})')
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    print("Done.")

if __name__ == "__main__":
    run_visualization()