import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Theme Indices
THEMES = {
    "Astro": list(range(16, 32)),
    "Botany": list(range(208, 224))
}

# We use a lower strength and a different additive offset
def steer_hook(resid, hook, sae, indices, strength=5.0):
    sae_acts = sae.encode(resid)
    for idx in indices:
        if idx < sae_acts.shape[-1]:
            # Just amplify existing signals slightly rather than forcing a 1.0 floor
            sae_acts[:, :, idx] = (sae_acts[:, :, idx] + 0.1) * strength
    return sae.decode(sae_acts)

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    prompt = "A simple observation of the"
    
    print(f"\n[Baseline]:")
    print(f"'{model.generate(prompt, max_new_tokens=30, verbose=False)}'")
    
    for name, indices in THEMES.items():
        print(f"\n[Steered: {name} (Strength: 8.0)]: ")
        hook_fn = partial(steer_hook, sae=sae, indices=indices, strength=8.0)
        # Apply only to layer 6
        model.add_hook(f"blocks.6.hook_resid_pre", hook_fn)
        
        print(f"'{model.generate(prompt, max_new_tokens=30, verbose=False)}'")
        model.reset_hooks()

if __name__ == "__main__":
    run_test()
