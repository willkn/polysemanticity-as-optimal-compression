import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

def intervention_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    prompt = "The quick brown fox jumps over the lazy dog."
    
    # 1. Clean Baseline (No Hooks)
    print("\n[1. Clean Baseline]:")
    clean_out = model.generate(prompt, max_new_tokens=20, verbose=False)
    print(f"'{clean_out}'")
    
    # 2. Reconstruction Hook (Encode -> Decode -> Replace)
    # This tests the "SAE Tax" - simply passing through the SAE without modifying features.
    def reconstruction_hook(resid, hook, sae):
        acts = sae.encode(resid)
        recon = sae.decode(acts)
        return recon

    model.add_hook("blocks.6.hook_resid_pre", partial(reconstruction_hook, sae=sae))
    print("\n[2. SAE Reconstruction Only (No Steering)]: ")
    recon_out = model.generate(prompt, max_new_tokens=20, verbose=False)
    print(f"'{recon_out}'")
    model.reset_hooks()
    
    # 3. Error Analysis
    # Let's measure the actual MSE of the reconstruction
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
        original = cache["blocks.6.hook_resid_pre"]
        
        acts = sae.encode(original)
        reconstruction = sae.decode(acts)
        
        mse = (original - reconstruction).pow(2).mean()
        norm_ratio = reconstruction.norm() / original.norm()
        
        print(f"\n[Metrics]:")
        print(f"Reconstruction MSE: {mse:.4f}")
        print(f"Norm Ratio (Recon/Original): {norm_ratio:.4f}")
        
    # 4. Bias Check
    # Does the decoder have a massive bias term?
    if hasattr(sae, 'b_dec'):
        print(f"Decoder Bias Norm: {sae.b_dec.norm():.4f}")
        
if __name__ == "__main__":
    intervention_test()
