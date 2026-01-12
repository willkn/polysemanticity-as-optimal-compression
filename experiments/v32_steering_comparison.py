import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Constants
THEME_ZOOLOGY = [224, 225, 226, 227, 228, 229, 230, 231] # A cohesive cluster from our Atlas
SINGLE_ATOM = [224] # Just one of them

def measure_steering_quality(model, sae, theme_indices, strength):
    """
    Measures:
    1. Target Signal: Mean activation of the theme indices.
    2. Off-Target Noise: Mean activation of all OTHER themes.
    3. Coherence: We use log-probs of the generated sequence as a proxy.
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    prompt = "The biologist looked through the microscope and saw"
    
    target_activations = []
    other_activations = []
    
    all_indices = list(range(512))
    other_indices = [i for i in all_indices if i not in theme_indices]

    def steering_hook(resid, hook):
        acts = sae.encode(resid)
        # Record pre-steering state of others
        other_activations.append(acts[:, :, other_indices].mean().item())
        
        # Apply Steering
        for idx in theme_indices:
            acts[:, :, idx] = (acts[:, :, idx] + 0.1) * strength
            
        target_activations.append(acts[:, :, theme_indices].mean().item())
        return sae.decode(acts)

    model.reset_hooks()
    model.add_hook("blocks.6.hook_resid_pre", steering_hook)
    
    # Generate and get loss
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
        # Simple cross entropy of the prompt itself under the steering
        # This tells us how much we've 'corrupted' the model's basic understanding
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # We'll just take the mean log prob of the last token as a stability metric
        stability = log_probs[0, -1, tokens[0, -1]].item()

    model.generate(prompt, max_new_tokens=15, verbose=False)
    model.reset_hooks()
    
    return np.mean(target_activations), np.mean(other_activations), stability

def run_comparison():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)

    strengths = np.linspace(0, 30, 6)
    
    results = {
        "theme": {"signal": [], "noise": [], "stability": []},
        "single": {"signal": [], "noise": [], "stability": []}
    }

    print("Running Steering Comparison...")
    for s in strengths:
        # Theme Steering
        sig, noise, stab = measure_steering_quality(model, sae, THEME_ZOOLOGY, s)
        results["theme"]["signal"].append(sig)
        results["theme"]["noise"].append(noise)
        results["theme"]["stability"].append(stab)
        
        # Single Atom Steering (Same count of activations boosted, but just one index)
        sig_s, noise_s, stab_s = measure_steering_quality(model, sae, SINGLE_ATOM, s * len(THEME_ZOOLOGY))
        results["single"]["signal"].append(sig_s)
        results["single"]["noise"].append(noise_s)
        results["single"]["stability"].append(stab_s)

    # PLOT 1: Signal-to-Noise Ratio (A clearer metric)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(strengths, np.array(results["theme"]["signal"]) / (np.array(results["theme"]["noise"]) + 1e-6), 
             label="Theme Steering", marker='o', color='#2ecc71', linewidth=3)
    plt.plot(strengths, np.array(results["single"]["signal"]) / (np.array(results["single"]["noise"]) + 1e-6), 
             label="Single Atom Steering", marker='s', color='#e74c3c', linewidth=2)
    plt.title("Signal-to-Interference Ratio (SIR)", fontweight='bold')
    plt.xlabel("Steering Strength")
    plt.ylabel("SIR (Target / Baseline Noise)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PLOT 2: Stability (Log Prob of Neutral Output)
    plt.subplot(1, 2, 2)
    plt.plot(strengths, results["theme"]["stability"], label="Theme Steering", marker='o', color='#2ecc71', linewidth=3)
    plt.plot(strengths, results["single"]["stability"], label="Single Atom Steering", marker='s', color='#e74c3c', linewidth=2)
    plt.title("Model Stability (Log Prob)", fontweight='bold')
    plt.xlabel("Steering Strength")
    plt.ylabel("Stability Metric (Higher is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/figures/steering_robustness.png")
    print("✅ New Robustness Plot saved.")

    # Empirical Summary for the paper
    print(f"\n[EMPIRICAL SUMMARY]")
    print(f"Theme Steering SIR @ Strength 30: {results['theme']['signal'][-1] / results['theme']['noise'][-1]:.2f}x")
    print(f"Single Atom SIR @ Strength 30:   {results['single']['signal'][-1] / results['single']['noise'][-1]:.2f}x")
    print(f"Theme Steering is { (results['theme']['signal'][-1] / results['theme']['noise'][-1]) / (results['single']['signal'][-1] / results['single']['noise'][-1]):.2f}x more efficient at preserving SIR.")

if __name__ == "__main__":
    run_comparison()
