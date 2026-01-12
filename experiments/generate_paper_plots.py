
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def generate_paper_plots():
    results_dir = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Plot the Parity Law (V8 data)
    # k values from V8: [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128]
    # Parity ratios from V8 output: [3.87, 3.16, 2.66, 2.32, 2.05, 1.83, 1.66, 1.51, 1.31, 1.18, 1.10, 1.11]
    ks = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128]
    parities = [3.87, 3.16, 2.66, 2.32, 2.05, 1.83, 1.66, 1.51, 1.31, 1.18, 1.10, 1.11]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, parities, 'o-', color='#4A90E2', linewidth=2.5, markersize=8)
    plt.axhline(1.0, color='#D0021B', linestyle='--', alpha=0.7, label='SAE Baseline (1.0x)')
    plt.fill_between(ks, 1.0, parities, color='#4A90E2', alpha=0.1)
    plt.xlabel("Active Polysemanticity ($k_{sq}$)", fontsize=12)
    plt.ylabel("Parity Ratio ($MSE_{Codec} / MSE_{SAE}$)", fontsize=12)
    plt.title("The Superposition Parity Law ($M=128$)", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "parity_law.png", dpi=300)
    plt.close()

    # 2. Plot Spatial Capacity (V7 data)
    # M values: [32, 64, 128, 256, 512, 1024]
    # Parity values: [5.92, 4.79, 3.69, 2.20, 0.27, 0.10]
    ms = [32, 64, 128, 256, 512, 1024]
    m_parities = [5.92, 4.79, 3.69, 2.20, 0.27, 0.10]

    plt.figure(figsize=(8, 5))
    plt.plot(ms, m_parities, 's-', color='#7ED321', linewidth=2.5, markersize=8)
    plt.axvline(128, color='#9B9B9B', linestyle='--', label='Model Dimension ($D=128$)')
    plt.axhline(1.0, color='#D0021B', linestyle='--')
    plt.xscale('log', base=2)
    plt.xlabel("Bottleneck Size ($M$)", fontsize=12)
    plt.ylabel("Parity Ratio", fontsize=12)
    plt.title("Spatial Capacity Threshold (Fixed $k=16$)", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "spatial_capacity.png", dpi=300)
    plt.close()

    print(f"✅ Paper plots generated in {plots_dir}")

if __name__ == "__main__":
    generate_paper_plots()
