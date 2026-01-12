import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

def find_atoms(model, sae, text, top_k=5):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tokens = model.to_tokens(text)
    
    with torch.no_grad():
        # Get activations at layer 6
        _, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
        resid = cache["blocks.6.hook_resid_pre"]
        
        # Encode into atoms
        acts = sae.encode(resid)
        
        # Find top activating atoms for the last token
        last_token_acts = acts[0, -1, :]
        values, indices = torch.topk(last_token_acts, top_k)
        
    return indices.tolist()

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    print("Finding Astro atoms (text: 'galaxy star telescope')...")
    astro_atoms = find_atoms(model, sae, "The stars and galaxies observed through the telescope")
    print(f"Astro Indices: {astro_atoms}")
    
    print("\nFinding Botany atoms (text: 'plant flower leaf chloroplast')...")
    botany_atoms = find_atoms(model, sae, "The green leaves and flowers of the plants")
    print(f"Botany Indices: {botany_atoms}")

if __name__ == "__main__":
    main()
