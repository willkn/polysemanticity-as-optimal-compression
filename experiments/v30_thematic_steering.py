import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Targeted indices for GPT-2 Small Layer 6
WEAPONRY_DX = 512 

def steer_hook(resid, hook, sae, indices, strength=500.0):
    sae_acts = sae.encode(resid)
    # Aggressively force these features to fire
    for idx in indices:
        if idx < sae_acts.shape[-1]:
            # Set to a very high constant value to 'overpower' the residual stream
            sae_acts[:, :, idx] = strength 
            
    return sae.decode(sae_acts)

def run_targeted_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    prompt = "I looked at the object and saw that it was a"
    
    # 1. Baseline
    print(f"\n[Baseline]:")
    print(f"'{model.generate(prompt, max_new_tokens=15, verbose=False)}'")
    
    # 2. Force Weaponry
    # We'll apply this to EVERY layer from 6 up to 9 to make it stick
    print(f"\n[Steered: FORCE WEAPONRY (Index 512) Strength 500x]:")
    hook_fn = partial(steer_hook, sae=sae, indices=[512], strength=500.0)
    
    # Add hooks to multiple layers to ensure the signal propagates
    for layer in range(6, 10):
        model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
        
    print(f"'{model.generate(prompt, max_new_tokens=15, verbose=False)}'")
    model.reset_hooks()

if __name__ == "__main__":
    run_targeted_test()
