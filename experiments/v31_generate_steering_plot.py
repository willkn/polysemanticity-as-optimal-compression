import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Targeted indices for GPT-2 Small Layer 6
WEAPONRY_INDICES = [512, 2048, 100, 200] # Representative 'Military' indices

def get_theme_support(model, sae, prompt, n_tokens=20, steer_indices=None, strength=10.0):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    supports = []
    
    def support_hook(resid, hook):
        acts = sae.encode(resid)
        # Calculate mean activation of the theme indices
        support = acts[:, :, steer_indices].mean().item()
        supports.append(support)
        
        if steer_indices and strength > 1.0:
            for idx in steer_indices:
                acts[:, :, idx] = (acts[:, :, idx] + 0.1) * strength
            return sae.decode(acts)
        return resid

    model.reset_hooks()
    model.add_hook("blocks.6.hook_resid_pre", support_hook)
    
    model.generate(prompt, max_new_tokens=n_tokens, verbose=False)
    model.reset_hooks()
    
    return supports

def build_steering_visualization():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    prompt = "The research into the"
    
    print("Collecting Baseline...")
    baseline_support = get_theme_support(model, sae, prompt, steer_indices=WEAPONRY_INDICES, strength=1.0)
    
    print("Collecting Steered...")
    steered_support = get_theme_support(model, sae, prompt, steer_indices=WEAPONRY_INDICES, strength=5.0)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_support, label='Baseline (Neutral)', color='#34495e', linewidth=2, linestyle='--')
    plt.plot(steered_support, label='Steered (Military Theme Boosted)', color='#e74c3c', linewidth=3)
    
    plt.fill_between(range(len(steered_support)), baseline_support, steered_support, color='#e74c3c', alpha=0.1)
    
    plt.title("Thematic Steering: Shifting Semantic Trajectory", fontsize=14, fontweight='bold')
    plt.xlabel("Generation Step (Tokens)", fontsize=12)
    plt.ylabel("Thematic Support (Aggregate Activation)", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    save_path = "/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/figures/steering_plot.png"
    plt.savefig(save_path)
    print(f"✅ Steering Plot saved to {save_path}")

if __name__ == "__main__":
    build_steering_visualization()
