import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial
import json
import numpy as np

# Config
ATLAS_PATH = "/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/automated_atlas.json"
RESULTS_PATH = "/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/pareto_sweep.json"

def get_coherence_loss(model, text):
    # Measure perplexity/loss of the generated text
    # We just run one forward pass on the text to get loss
    tokens = model.to_tokens(text)
    with torch.no_grad():
        loss = model(tokens, return_type="loss")
    return loss.item()

def measure_thematic_alignment(model, sae, text, theme_indices):
    # Measure average activation of theme atoms in the generated text
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
        resid = cache["blocks.6.hook_resid_pre"]
        acts = sae.encode(resid) # [Batch, Seq, d_sae]
        
        # Select theme atoms
        theme_acts = acts[0, :, theme_indices] # [Seq, k]
        return theme_acts.mean().item()

def steer_hook(resid, hook, sae, indices, strength=1.0):
    steering_acts = torch.zeros((resid.shape[0], resid.shape[1], sae.cfg.d_sae), device=resid.device)
    for idx in indices:
        steering_acts[:, :, idx] = strength
    direction = steering_acts @ sae.W_dec 
    return resid + direction

def sweep_pareto():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    # Load Atlas
    with open(ATLAS_PATH, 'r') as f:
        atlas = json.load(f)
    
    # We'll test the top 3 densest themes
    themes = sorted(atlas.items(), key=lambda x: x[1]['coherence'], reverse=True)[:3]
    
    strengths = [0.0, 0.5, 1.0, 1.4, 2.0, 3.0, 5.0, 10.0]
    prompt = "The report detailed the new"
    
    results = []
    
    print("Starting Pareto Sweep...")
    
    for theme_name, theme_data in themes:
        indices = theme_data['indices']
        print(f"Testing {theme_name} ({len(indices)} atoms)...")
        
        for s in strengths:
            # Generate
            model.reset_hooks()
            if s > 0:
                hook_fn = partial(steer_hook, sae=sae, indices=indices, strength=s)
                for layer in [6, 7]:
                    model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
            
            # Generate 3 samples per setting to average
            batch_loss = []
            batch_align = []
            
            for _ in range(3):
                output = model.generate(prompt, max_new_tokens=30, verbose=False)
                
                loss = get_coherence_loss(model, output)
                align = measure_thematic_alignment(model, sae, output, indices)
                
                batch_loss.append(loss)
                batch_align.append(align)
            
            avg_loss = np.mean(batch_loss)
            avg_align = np.mean(batch_align)
            
            print(f"  S={s}: Loss={avg_loss:.2f}, Align={avg_align:.2f}")
            
            results.append({
                "theme": theme_name,
                "strength": s,
                "loss": avg_loss,
                "alignment": avg_align,
                "sample": output # save last sample
            })
            
    # Save
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Sweep saved to {RESULTS_PATH}")

if __name__ == "__main__":
    sweep_pareto()
