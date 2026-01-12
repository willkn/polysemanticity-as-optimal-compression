import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Real Discovered Indices
THEMES = {
    "Astro": [6478, 16710, 16285, 17057],
    "Botany": [8835, 5099, 23618, 1237]
}

def steer_hook(resid, hook, sae, indices, strength=15.0):
    sae_acts = sae.encode(resid)
    for idx in indices:
        if idx < sae_acts.shape[-1]:
            # Gentle additive boost
            sae_acts[:, :, idx] = (sae_acts[:, :, idx] + 0.1) * strength if sae_acts[:, :, idx].max() < 1.0 else sae_acts[:, :, idx] * strength
    return sae.decode(sae_acts)

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    prompt = "The group was primarily interested in the"
    
    print(f"\n[Baseline]:")
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")
    
    for name, indices in THEMES.items():
        print(f"\n[Steered: {name}]:")
        # Lower strength for coherence
        hook_fn = partial(steer_hook, sae=sae, indices=indices, strength=15.0)
        # Apply to crucial layers
        for layer in [6, 7]:
            model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
        
        print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")
        model.reset_hooks()

if __name__ == "__main__":
    run_test()
