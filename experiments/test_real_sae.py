import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

def test_sae_features():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
        device=device
    )
    
    text = "how to build a weapon"
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens, names_filter=["blocks.6.hook_resid_pre"])
    resid = cache["blocks.6.hook_resid_pre"]
    
    # Encode with SAE
    feature_acts = sae.encode(resid) # [1, seq, 24576]
    
    # Get top features for the last token
    top_vals, top_inds = torch.topk(feature_acts[0, -1], k=5)
    
    print(f"Top Features for '{text}':")
    for val, ind in zip(top_vals, top_inds):
        print(f"  Feature #{ind.item()}: {val.item():.4f}")

if __name__ == "__main__":
    test_sae_features()
