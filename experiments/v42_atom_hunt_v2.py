import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

def get_mean_acts(model, sae, text):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
        resid = cache["blocks.6.hook_resid_pre"]
        acts = sae.encode(resid)
        return acts[0, :, :].mean(dim=0)

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    # 1. Baseline (Common English)
    base_text = "The quick brown fox jumps over the lazy dog. The man walked down the street to the store."
    base_acts = get_mean_acts(model, sae, base_text)
    
    # 2. Astro
    astro_text = "galaxy star planet universe telescope astronomer solar system orbit light year nebula cosmos"
    astro_acts = get_mean_acts(model, sae, astro_text)
    # Subtract baseline to find specific features
    astro_diff = torch.relu(astro_acts - base_acts) # Only keep positive delta
    _, astro_indices = torch.topk(astro_diff, 20)
    
    # 3. Botany
    botany_text = "plant flower leaf tree garden root stem petal chlorophyll photosynthesi vegetation bloom"
    botany_acts = get_mean_acts(model, sae, botany_text)
    botany_diff = torch.relu(botany_acts - base_acts)
    _, botany_indices = torch.topk(botany_diff, 20)
    
    print(f"Distinct Astro Atoms: {astro_indices.tolist()}")
    print(f"Distinct Botany Atoms: {botany_indices.tolist()}")

if __name__ == "__main__":
    main()
