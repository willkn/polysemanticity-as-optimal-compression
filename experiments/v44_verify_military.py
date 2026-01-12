import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# New Refined Indices
NEW_MILITARY = [2781, 20777, 3320, 9097]

def steer_hook(resid, hook, sae, indices, strength=10.0):
    sae_acts = sae.encode(resid)
    for idx in indices:
        if idx < sae_acts.shape[-1]:
            # Additive Boost
            sae_acts[:, :, idx] = (sae_acts[:, :, idx] + 0.1) * strength
    return sae.decode(sae_acts)

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    prompt = "The group was primarily interested in the"
    
    print(f"\n[Baseline]:")
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")
    
    # Test New Military
    print(f"\n[Steered: Military (New Indices)]: ")
    hook_fn = partial(steer_hook, sae=sae, indices=NEW_MILITARY, strength=20.0)
    for layer in [6, 7]:
        model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")

if __name__ == "__main__":
    run_test()
