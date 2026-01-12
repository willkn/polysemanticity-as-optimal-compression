import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Bias-Free Steering
# The SAE decoder is x = W_dec @ f + b_dec.
# When we used sae.decode(steering_acts), we were adding W_dec @ f + b_dec to the residual stream.
# But the residual stream ALREADY contains the bias (implicitly).
# Adding it again shifts the mean of the distribution by +53.0 units.
# We must apply ONLY the direction: W_dec @ f.

def unbiased_steer_hook(resid, hook, sae, indices, strength=10.0):
    # 1. Create Sparse Acts
    steering_acts = torch.zeros((resid.shape[0], resid.shape[1], sae.cfg.d_sae), device=resid.device)
    for idx in indices:
        steering_acts[:, :, idx] = strength

    # 2. Decode manually WITHOUT bias
    # W_dec is [d_sae, d_model]
    direction = steering_acts @ sae.W_dec 
    
    # 3. Add to residual
    return resid + direction

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    ASTRO_INDICES = [8870, 6478, 403, 23389, 22931]
    prompt = "The quick brown fox jumps over the lazy dog."
    
    print(f"\n[Baseline]:")
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")
    
    # Test Unbiased Steering
    print(f"\n[Steered: Unbiased (Strength 10.0)]: ")
    hook_fn = partial(unbiased_steer_hook, sae=sae, indices=ASTRO_INDICES, strength=10.0)
    for layer in [6, 7]:
        model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")

if __name__ == "__main__":
    run_test()
