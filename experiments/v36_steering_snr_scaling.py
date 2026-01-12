import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Zoology Cluster
ZOOLOGY_ATOMS = [224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

def get_logit_delta(model, sae, theme_indices, strength=10.0):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    prompt = "The biologist looked through the microscope and saw"
    
    # Target Tokens: cell, life, organism, small
    target_tokens = [2210, 1172, 34567, 1402] 
    # Interference Tokens (Unrelated): military, bank, space, music
    noise_tokens = [5114, 2325, 2124, 2525]

    def steering_hook(resid, hook):
        acts = sae.encode(resid)
        for idx in theme_indices:
            acts[:, :, idx] = (acts[:, :, idx] + 0.1) * strength
        return sae.decode(acts)

    # Baseline
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits_b = model(tokens)[0, -1]
        
    # Steered
    model.reset_hooks()
    model.add_hook("blocks.6.hook_resid_pre", steering_hook)
    with torch.no_grad():
        logits_s = model(tokens)[0, -1]
    model.reset_hooks()
    
    # Delta (Change in Logit)
    delta = logits_s - logits_b
    
    signal = delta[target_tokens].mean().item()
    noise = delta[noise_tokens].abs().mean().item()
    
    return signal, noise

def run_scaling_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)

    n_atoms = range(1, len(ZOOLOGY_ATOMS) + 1)
    snrs = []
    
    print("Testing SNR Scaling with Population Size...")
    for n in n_atoms:
        indices = ZOOLOGY_ATOMS[:n]
        # We keep TOTAL energy constant to see the 'population' effect
        # Strength = 20 / sqrt(n)
        sig, noise = get_logit_delta(model, sae, indices, strength=20.0 / np.sqrt(n))
        snrs.append(sig / (noise + 1e-6))

    plt.figure(figsize=(10, 6))
    plt.plot(n_atoms, snrs, marker='o', color='#2ecc71', linewidth=3, markersize=10)
    
    # Add a theoretical sqrt(n) line for comparison
    x = np.array(n_atoms)
    theory = snrs[0] * np.sqrt(x)
    plt.plot(x, theory, linestyle='--', color='#95a5a6', label="Theoretical $\sqrt{N}$ Scaling")
    
    plt.title("Thematic SNR vs. Population Size (Atoms in Steer)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Atoms in steering cluster", fontsize=12)
    plt.ylabel("Thematic Logit SNR (Signal / Interference)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/figures/steering_snr_scaling.png")
    print("✅ SNR Scaling Plot saved.")
    
    # Reporting
    print(f"SNR for 1 Atom:  {snrs[0]:.2f}")
    print(f"SNR for {len(n_atoms)} Atoms: {snrs[-1]:.2f}")
    print(f"Real SNR gain: {snrs[-1]/snrs[0]:.2f}x (Theoretical: {np.sqrt(len(n_atoms)):.2f}x)")

if __name__ == "__main__":
    run_scaling_test()
