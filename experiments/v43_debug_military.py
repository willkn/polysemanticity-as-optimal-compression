import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Current "Military" indices causing issues
CURRENT_INDICES = [496, 1024, 2048, 5114]

def get_activations(model, sae, text, indices):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
        resid = cache["blocks.6.hook_resid_pre"]
        acts = sae.encode(resid) # [Batch, Seq, Feat]
        
        # Mean activation of these specific indices
        # We look at the max activation across the sequence to see if they fire at all
        mean_act = acts[0, :, indices].max(dim=0).values
        return mean_act

def find_better_atoms(model, sae):
    # 1. Define Positive and Negative Anchors
    military_text = "soldier weapon army battle tank gun war enemy tactical missile troops general command fired shot violence kill"
    history_text = "museum fossil ancient archaelogy stone century ruins old past discovered excavation site"
    
    # 2. Get Mean Vectors
    mil_vec = get_vector(model, sae, military_text)
    hist_vec = get_vector(model, sae, history_text)
    
    # 3. Calculate "Pure Military" Vector
    # We want features high in Military but LOW in History
    pure_mil = mil_vec - (hist_vec * 1.5) # Aggressive subtraction
    
    # 4. Find Top K
    values, indices = torch.topk(pure_mil, 10)
    return indices.tolist(), values.tolist()

def get_vector(model, sae, text):
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
    
    print("diagnostic: Checking current indices on 'History' prompt...")
    # Why did we get fossils?
    hist_acts = get_activations(model, sae, "The archaeologists found a fossil in the museum.", CURRENT_INDICES)
    print(f"Current Indices Activation on HISTORY: {hist_acts}")
    
    mil_acts = get_activations(model, sae, "The army soldier fired his gun.", CURRENT_INDICES)
    print(f"Current Indices Activation on MILITARY: {mil_acts}")
    
    print("\nHunting for BETTER atoms (Military - History)...")
    new_indices, new_vals = find_better_atoms(model, sae)
    print(f"New Candidates: {new_indices}")

if __name__ == "__main__":
    main()
