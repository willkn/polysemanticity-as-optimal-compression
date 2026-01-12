import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Trying another subset from the candidate list
# Candidates: [2781, 20777, 3320, 9097, 14422, 21526, 4192, 1682, 6667, 7478]
# 20777 was also in botany? That might be polysemantic.
# Let's try [3320, 9097, 14422]
# And let's try finding atoms by "War" specifically
CANDIDATE_SET_2 = [3320, 9097, 14422]

def find_war_atoms(model, sae):
    tokens = model.to_tokens("War battle soldier gun army weapon kill violence combat")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
        resid = cache["blocks.6.hook_resid_pre"]
        acts = sae.encode(resid)
        # Mean across sequence
        mean_act = acts[0].mean(dim=0)
        # Subtract "Peace/Civilian"
        peace_tokens = model.to_tokens("Peace civilian garden home sleep love happy safe")
        _, pc = model.run_with_cache(peace_tokens, names_filter="blocks.6.hook_resid_pre")
        p_resid = pc["blocks.6.hook_resid_pre"]
        p_acts = sae.encode(p_resid).mean(dim=1).squeeze()
        
        diff = mean_act - p_acts
        values, indices = torch.topk(diff, 10)
        return indices.tolist()

def steer_hook(resid, hook, sae, indices, strength=10.0):
    sae_acts = sae.encode(resid)
    for idx in indices:
        if idx < sae_acts.shape[-1]:
            sae_acts[:, :, idx] = (sae_acts[:, :, idx] + 0.1) * strength
    return sae.decode(sae_acts)

def run_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    print("Hunting for WAR atoms...")
    war_indices = find_war_atoms(model, sae)
    print(f"War Indices: {war_indices}")
    
    prompt = "The group was primarily interested in the"
    
    # Test War Indices
    print(f"\n[Steered: War (Generated Indices)]: ")
    hook_fn = partial(steer_hook, sae=sae, indices=war_indices[:4], strength=20.0)
    for layer in [6, 7]:
        model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
    print(f"'{model.generate(prompt, max_new_tokens=25, verbose=False)}'")

if __name__ == "__main__":
    run_test()
