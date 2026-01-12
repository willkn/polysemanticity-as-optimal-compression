import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Gentle Additive Steering V2
# Adding raw vectors often pushes effective layers into saturation.
# The "safest" way is to add the vector but then RE-NORMALIZE the residual stream.

def normalized_steer_hook(resid, hook, sae, indices, strength=2.0):
    # 1. Create Steering Vector
    steering_acts = torch.zeros((resid.shape[0], resid.shape[1], sae.cfg.d_sae), device=resid.device)
    for idx in indices:
        steering_acts[:, :, idx] = strength
    steering_vec = sae.decode(steering_acts) # [Batch, Seq, d_model]
    
    # 2. Add to residual
    new_resid = resid + steering_vec
    
    # 3. SAFETY: Re-Normalize to original energy level
    # This prevents the "Normalization Shock" that causes gibberish.
    # We want the semantic direction of 'new_resid', but the magnitude of 'resid'.
    
    # Calculate norms per token
    old_norm = resid.norm(dim=-1, keepdim=True)
    new_norm = new_resid.norm(dim=-1, keepdim=True)
    
    # Rescale
    final_resid = new_resid * (old_norm / (new_norm + 1e-6))
    return final_resid

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    ASTRO_INDICES = [8870, 6478, 403, 23389, 22931]
    prompt = "The quick brown fox jumps over the lazy dog."
    
    print(f"\n[Baseline]:")
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")
    
    # Test Normalized Steering
    print(f"\n[Steered: Normalized Additive (Strength 50.0)]: ")
    # We can use huge strength because we normalize it back down!
    hook_fn = partial(normalized_steer_hook, sae=sae, indices=ASTRO_INDICES, strength=50.0)
    for layer in [6, 7]:
        model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")

if __name__ == "__main__":
    run_test()
