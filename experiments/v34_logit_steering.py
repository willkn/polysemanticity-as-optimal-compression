import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Constants
THEME_ZOOLOGY = [224, 225, 226, 227, 228, 229, 230, 231]
SINGLE_ATOM = [224]

def measure_logit_shift(model, sae, theme_indices, strength):
    """
    Measures the boost in probability for a set of 'animal-related' tokens.
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    prompt = "The biologist looked through the microscope and saw"
    
    # Representative tokens for Zoology theme
    # cat, dog, lion, tiger, animal, cell, DNA, protein
    animal_tokens = [5480, 3968, 12696, 29014, 5740, 2210, 4124, 7626] 
    
    def steering_hook(resid, hook):
        acts = sae.encode(resid)
        for idx in theme_indices:
            acts[:, :, idx] = (acts[:, :, idx] + 0.1) * strength
        return sae.decode(acts)

    # Baseline Prob
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
        probs = torch.softmax(logits[0, -1], dim=-1)
        baseline_theme_prob = probs[animal_tokens].sum().item()

    # Steered Prob
    model.reset_hooks()
    model.add_hook("blocks.6.hook_resid_pre", steering_hook)
    with torch.no_grad():
        logits_s = model(tokens)
        probs_s = torch.softmax(logits_s[0, -1], dim=-1)
        steered_theme_prob = probs_s[animal_tokens].sum().item()
    model.reset_hooks()
    
    # Calculate the Factor Increase
    # Avoid div by zero. Start at 1.0 (no change)
    shift = steered_theme_prob / (baseline_theme_prob + 1e-9)
    return shift

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)

    strengths = np.linspace(0, 50, 10)
    
    theme_shifts = []
    single_shifts = []
    
    print("Evaluating Logit Shifts (Theme vs Single Atom)...")
    for s in strengths:
        # THEME: Boost all indices with strength s
        t_shift = measure_logit_shift(model, sae, THEME_ZOOLOGY, s)
        theme_shifts.append(t_shift)
        
        # SINGLE: Boost ONE index with strength s*sqrt(N) 
        # (This is a fairer energy scaling based on our SNR theory)
        s_shift = measure_logit_shift(model, sae, SINGLE_ATOM, s * np.sqrt(len(THEME_ZOOLOGY))) 
        single_shifts.append(s_shift)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(strengths, theme_shifts, label="Thematic Steering (8 Atoms)", marker='o', color='#2ecc71', linewidth=3)
    plt.plot(strengths, single_shifts, label="Single Atom Steering", marker='s', color='#e74c3c', linewidth=2)
    
    plt.yscale('log') # Use log scale because prob shifts are exponential
    plt.title("Steering Efficacy: Probability Shift of Thematic Tokens", fontsize=14, fontweight='bold')
    plt.xlabel("Steering Alpha (Strength)")
    plt.ylabel("Probability Multiplier (Steered / Baseline)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    plt.savefig("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/figures/steering_logit_shift.png")
    print("✅ Probability Shift Plot saved.")
    
    # Empirical Summary
    print(f"Final Theme Multiplier: {theme_shifts[-1]:.2f}x")
    print(f"Final Single Multiplier: {single_shifts[-1]:.2f}x")

if __name__ == "__main__":
    run_test()
