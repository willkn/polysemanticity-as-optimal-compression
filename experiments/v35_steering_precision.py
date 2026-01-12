import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Zoology Theme atoms
THEME_ZOOLOGY = [224, 225, 226, 227, 228, 229, 230, 231]
# Military Theme atoms (The 'Distractor')
THEME_MILITARY = [496, 497, 498, 499, 500, 501, 502, 503]

def get_theme_precisions(model, sae, theme_indices, strength):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    prompt = "The research team was focus on the"
    
    # Representative tokens for both themes
    animal_tokens = [5480, 12696, 5740] # cat, lion, animal
    military_tokens = [5114, 2048, 50, 150] # military, weapon, etc.

    def steering_hook(resid, hook):
        acts = sae.encode(resid)
        for idx in theme_indices:
            acts[:, :, idx] = (acts[:, :, idx] + 0.1) * strength
        return sae.decode(acts)

    model.reset_hooks()
    model.add_hook("blocks.6.hook_resid_pre", steering_hook)
    
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
        probs = torch.softmax(logits[0, -1], dim=-1)
        
        target_p = probs[animal_tokens].mean().item()
        distractor_p = probs[military_tokens].mean().item()
        
    model.reset_hooks()
    return target_p / (distractor_p + 1e-9)

def run_precision_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)

    strengths = np.linspace(0, 50, 8)
    
    theme_precisions = []
    single_precisions = []
    
    for s in strengths:
        # Theme: Boost the 8 zoology atoms
        tp = get_theme_precisions(model, sae, THEME_ZOOLOGY, s)
        theme_precisions.append(tp)
        
        # Single: Boost just the FIRST zoology atom
        sp = get_theme_precisions(model, sae, [224], s * 2.8) # Adjusted energy
        single_precisions.append(sp)

    plt.figure(figsize=(10, 6))
    plt.plot(strengths, theme_precisions, label="Thematic Steering (Population)", marker='o', color='#2ecc71', linewidth=3)
    plt.plot(strengths, single_precisions, label="Single Atom Steering (Individual)", marker='s', color='#e74c3c', linewidth=2)
    
    plt.title("Steering Precision: Target Theme vs. Cross-Talk Interference", fontsize=14, fontweight='bold')
    plt.xlabel("Steering Alpha (Scaled Energy)")
    plt.ylabel("Precision Ratio (Target Prob / Distractor Prob)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    plt.savefig("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/figures/steering_precision.png")
    print("✅ Precision Plot saved.")
    
    print(f"Final Theme Precision: {theme_precisions[-1]:.2f}")
    print(f"Final Single Precision: {single_precisions[-1]:.2f}")

if __name__ == "__main__":
    run_precision_test()
