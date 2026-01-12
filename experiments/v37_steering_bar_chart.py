import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Zoology: 224-231 (Population)
THEme_indices = list(range(224, 232))

def run_gen_test(model, sae, indices, strength):
    prompt = "I looked outside and saw a"
    # Target: animal, creature, dog, cat, bear, lion
    targets = [" animal", " creature", " dog", " cat", " bear", " lion"]
    target_ids = model.to_tokens(targets, prepend_bos=False).flatten()

    def hook(resid, hook):
        acts = sae.encode(resid)
        for idx in indices:
            acts[:, :, idx] = (acts[:, :, idx] + 1.0) * strength
        return sae.decode(acts)

    model.reset_hooks()
    model.add_hook("blocks.6.hook_resid_pre", hook)
    
    with torch.no_grad():
        logits = model(model.to_tokens(prompt))[0, -1]
        probs = torch.softmax(logits, dim=-1)
        theme_prob = probs[target_ids].sum().item()
        
        # Interference: Measure a totally unrelated word ' financial'
        noise_prob = probs[model.to_single_token(" financial")].item()
        
    model.reset_hooks()
    return theme_prob, noise_prob

def build_bar_chart():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)

    # 1. Baseline
    b_target, b_noise = run_gen_test(model, sae, [], 0)
    
    # 2. Single Atom (idx 224) at high strength
    s_target, s_noise = run_gen_test(model, sae, [224], 50.0)
    
    # 3. Theme Population (8 atoms) at moderate strength
    t_target, t_noise = run_gen_test(model, sae, THEme_indices, 6.25) # 50 / 8

    # PLOTTING
    categories = ['Baseline', 'Single Atom', 'Thematic (Codec)']
    target_probs = [b_target * 100, s_target * 100, t_target * 100]
    noise_probs = [b_noise * 1000, s_noise * 1000, t_noise * 1000] # Scaled for visibility

    x = np.arange(len(categories))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    rects1 = ax1.bar(x - width/2, target_probs, width, label='Target Theme Prob (%)', color='#2ecc71')
    ax1.set_ylabel('Target Probability (%)', fontsize=12, color='#2ecc71')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, noise_probs, width, label='Interference Noise (x10)', color='#e74c3c', alpha=0.7)
    ax2.set_ylabel('Interference Noise (Arbitrary Units)', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    plt.title("Thematic Steering: Signal vs. Side-Effect Noise", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/figures/steering_bar_chart.png")
    print("✅ Bar Chart saved.")
    
    # Output the data for the paper
    print(f"Theme Prob: Baseline={b_target:.4f}, Single={s_target:.4f}, Theme={t_target:.4f}")
    print(f"Noise Prob:  Baseline={b_noise:.4f}, Single={s_noise:.4f}, Theme={t_noise:.4f}")

if __name__ == "__main__":
    build_bar_chart()
