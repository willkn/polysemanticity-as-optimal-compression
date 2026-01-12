import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Unbiased Steering Hook (from v49)
def steer_hook(resid, hook, sae, indices, strength=1.0):
    steering_acts = torch.zeros((resid.shape[0], resid.shape[1], sae.cfg.d_sae), device=resid.device)
    for idx in indices:
        steering_acts[:, :, idx] = strength
    
    # Pure direction without bias
    direction = steering_acts @ sae.W_dec 
    return resid + direction

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    # Themes
    THEMES = {
        "Astro": [8870, 6478, 403, 23389, 22931],
        "Botany": [17013, 6642, 8240, 20777, 12859]
    }
    
    prompt = "The research team published a paper on the"
    strengths = [0.5, 1.0, 1.4, 2.0, 5.0]

    print(f"Base Prompt: '{prompt}'")
    
    for theme, indices in THEMES.items():
        print(f"\n=== Theme: {theme} ===")
        for s in strengths:
            model.reset_hooks()
            hook_fn = partial(steer_hook, sae=sae, indices=indices, strength=s)
            for layer in [6, 7]:
                model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
            
            output = model.generate(prompt, max_new_tokens=20, verbose=False)
            print(f"[Strength {s}]: '{output}'")

if __name__ == "__main__":
    run_test()
