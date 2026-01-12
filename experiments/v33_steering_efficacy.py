import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Constants
THEME_ZOOLOGY = [224, 225, 226, 227, 228, 229, 230, 231] # A cohesive cluster from our Atlas
SINGLE_ATOM = [224] # Just one of them

def measure_generation(model, sae, theme_indices, strength):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    prompt = "The field was full of"
    
    def steering_hook(resid, hook):
        acts = sae.encode(resid)
        # Apply Steering
        for idx in theme_indices:
            acts[:, :, idx] = (acts[:, :, idx] + 0.1) * strength
        return sae.decode(acts)

    model.reset_hooks()
    model.add_hook("blocks.6.hook_resid_pre", steering_hook)
    
    output = model.generate(prompt, max_new_tokens=20, verbose=False)
    model.reset_hooks()
    
    # Analyze output for theme keywords (Zoology)
    keywords = ["animal", "cat", "dog", "insect", "bear", "lion", "predator", "creature", "beast", "zoology", "wildlife"]
    # We'll use a simple count of unique keywords present
    count = sum(1 for k in keywords if k in output.lower())
    
    return count, output

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)

    strengths = [5, 10, 20, 40]
    
    theme_counts = []
    single_counts = []
    
    print("Evaluating Steering Efficacy (Keywords per Prompt)...")
    for s in strengths:
        t_count, _ = measure_generation(model, sae, THEME_ZOOLOGY, s)
        theme_counts.append(t_count)
        
        # We tune single atom strength to match the TOTAL activation energy of the theme
        s_count, _ = measure_generation(model, sae, SINGLE_ATOM, s * 4.0) 
        single_counts.append(s_count)

    # Plotting Efficacy
    plt.figure(figsize=(10, 6))
    plt.plot(strengths, theme_counts, label="Theme Steering", marker='o', color='#2ecc71', linewidth=3)
    plt.plot(strengths, single_counts, label="Single Atom Steering", marker='s', color='#e74c3c', linewidth=2)
    
    plt.title("Steering Efficacy: Theme vs. Single Atom", fontsize=14, fontweight='bold')
    plt.xlabel("Total Steering Energy (Relative Strength)")
    plt.ylabel("Number of Thematic Keywords in 20 Tokens")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/figures/steering_efficacy.png")
    print("✅ Efficacy Plot saved.")

if __name__ == "__main__":
    run_test()
