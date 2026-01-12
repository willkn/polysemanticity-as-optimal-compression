import torch
import time
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

# Constants
THEME_ZOOLOGY = list(range(224, 240))
THEME_MILITARY = list(range(496, 512))
THEME_BIOTECH = list(range(0, 16))

def steer_hook(resid, hook, sae, indices, strength=10.0):
    # This is the core overhead: 
    # 1. Linear Projection (Encoder)
    # 2. ReLU / TopK
    # 3. Modification
    # 4. Linear Projection (Decoder)
    sae_acts = sae.encode(resid)
    for idx in indices:
        if idx < sae_acts.shape[-1]:
            sae_acts[:, :, idx] = (sae_acts[:, :, idx] + 0.5) * strength
    return sae.decode(sae_acts)

def run_performance_test():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    
    prompt = "The primary focus is"
    tokens = model.to_tokens(prompt)
    
    print("Measuring Latency Overhead...")
    # Baseline Latency
    times_b = []
    for _ in range(20):
        start = time.time()
        with torch.no_grad():
            model(tokens)
        times_b.append(time.time() - start)
    
    avg_b = np.mean(times_b)
    
    # Steered Latency
    times_s = []
    hook_fn = partial(steer_hook, sae=sae, indices=THEME_ZOOLOGY, strength=10.0)
    model.add_hook("blocks.6.hook_resid_pre", hook_fn)
    
    for _ in range(20):
        start = time.time()
        with torch.no_grad():
            model(tokens)
        times_s.append(time.time() - start)
    
    avg_s = np.mean(times_s)
    model.reset_hooks()
    
    print(f"Baseline Latency: {avg_b*1000:.2f}ms")
    print(f"Steered Latency:  {avg_s*1000:.2f}ms")
    print(f"Overhead:         {(avg_s/avg_b - 1)*100:.1f}%")

    # CASE STUDIES
    print("\nExtracting Case Studies...")
    prompts = [
        "In this report, we examine",
        "The team spent years working on",
        "Recent breakthroughs have shown that",
        "The historical context suggests that"
    ]
    
    case_studies = []
    
    for p in prompts:
        # Baseline
        base_out = model.generate(p, max_new_tokens=15, verbose=False)
        
        # Zoology
        model.add_hook("blocks.6.hook_resid_pre", partial(steer_hook, sae=sae, indices=THEME_ZOOLOGY, strength=25.0))
        zoo_out = model.generate(p, max_new_tokens=15, verbose=False)
        model.reset_hooks()
        
        # Military
        model.add_hook("blocks.6.hook_resid_pre", partial(steer_hook, sae=sae, indices=THEME_MILITARY, strength=25.0))
        mil_out = model.generate(p, max_new_tokens=15, verbose=False)
        model.reset_hooks()
        
        case_studies.append({
            "Prompt": p,
            "Baseline": base_out.replace(p, "").strip(),
            "Zoology": zoo_out.replace(p, "").strip(),
            "Military": mil_out.replace(p, "").strip()
        })
        
    df = pd.DataFrame(case_studies)
    print(df.to_string())
    
    # Save results for paper
    df.to_csv("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/steering_case_studies.csv", index=False)
    
if __name__ == "__main__":
    run_performance_test()
