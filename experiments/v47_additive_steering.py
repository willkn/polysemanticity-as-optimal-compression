import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Cleaner Additive Steering Logic
# Instead of replacing the stream with the SAE reconstruction,
# we only add the *delta* of the target features.
# This preserves all the subtle info the SAE misses.

def additive_steer_hook(resid, hook, sae, indices, strength=10.0):
    # 1. Encode to find direction
    sae_acts = sae.encode(resid)
    
    # 2. Create an empty "Steering Vector" in Atom Space
    steering_acts = torch.zeros_like(sae_acts)
    
    # 3. Only activate the target atoms
    for idx in indices:
        if idx < sae_acts.shape[-1]:
             # We can either boost existing activation or just inject constant energy
             # Let's simple inject constant energy for consistent steering
             steering_acts[:, :, idx] = strength
             
    # 4. Decode ONLY the steering vector
    steering_vec = sae.decode(steering_acts)
    
    # 5. Add to the ORIGINAL residual stream
    # Note: We do NOT replace resid with sae.decode(sae.encode(resid))
    return resid + steering_vec

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    # Themes (Astro from V42)
    ASTRO_INDICES = [8870, 6478, 403, 23389, 22931]
    prompt = "The quick brown fox jumps over the lazy dog."
    
    print(f"\n[Baseline]:")
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")
    
    # Test Additive Steering
    print(f"\n[Steered: Additive (Strength 2.0)]: ")
    # Note: Energies are different now. 10.0 might be huge if adding directly.
    # Let's try conservative strength first.
    hook_fn = partial(additive_steer_hook, sae=sae, indices=ASTRO_INDICES, strength=2.0)
    for layer in [6, 7]:
        model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")

if __name__ == "__main__":
    run_test()
