import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import os

# --- PRODUCTION CONFIG ---
# for robust 'Toplogy of Meaning' analysis
BATCH_SIZE = 8
NUM_BATCHES = 500  # ~500k tokens. Much better for 24k feature coverage.
SEQ_LEN = 128
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.6.hook_resid_pre"
SAVE_PATH = "/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/cofiring_matrix.npy"

def generate_cofiring_matrix():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model & SAE
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
    d_sae = sae.cfg.d_sae 
    
    # 2. Init Matrix (Float16 for speed/memory)
    # Note: We use CPU memory for the big matrix to save GPU VRAM for the model
    print(f"Initializing {d_sae}x{d_sae} matrix (approx {d_sae**2 * 2 / 1e9:.2f} GB)...")
    co_occurrence = torch.zeros((d_sae, d_sae), dtype=torch.float16, device='cpu') 
    total_firings = torch.zeros(d_sae, dtype=torch.float32, device='cpu')

    # 3. Data
    print("Loading Dataset...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    iterator = iter(dataset)

    print(f"Starting Collection ({NUM_BATCHES} batches of {BATCH_SIZE})...")
    
    for i in tqdm(range(NUM_BATCHES)):
        try:
            batch_text = [next(iterator)['text'] for _ in range(BATCH_SIZE)]
            tokens = model.to_tokens(batch_text, truncate=True)[:, :SEQ_LEN] 
            
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=SAE_ID)
                resid = cache[SAE_ID] 
                acts = sae.encode(resid) 
                
                # Binarize: Did the atom fire?
                fired_mask = (acts > 0).float() 
                # Did it fire ANYWHERE in the context?
                fired_in_context = (fired_mask.max(dim=1).values) # [Batch, d_sae]
                
                # Move to CPU for accumulation
                batch_fired = fired_in_context.cpu()
                
                total_firings += batch_fired.sum(dim=0)
                
                # Outer product addition
                co_occurrence += (batch_fired.T @ batch_fired).half()
        except Exception as e:
            print(f"Skipping batch {i} due to error: {e}")
            continue

    print("Saving Matrix...")
    np.save(SAVE_PATH, co_occurrence.numpy())
    np.save(SAVE_PATH.replace("cofiring_matrix.npy", "total_firings.npy"), total_firings.numpy())
    
    # Stats
    density = (co_occurrence > 0).float().mean().item()
    print(f"Saved to {SAVE_PATH}")
    print(f"Matrix Density: {density:.4f}")

if __name__ == "__main__":
    generate_cofiring_matrix()
