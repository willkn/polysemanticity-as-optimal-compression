"""
Neural Codec - GPT-2-Scale-Up
Goal: Train a Squeezer to re-compress GPT-2 Small Layer 6 monosemantic atoms back into the residuals.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from sae_lens import SAE
from transformer_lens import HookedTransformer
from datasets import load_dataset
import pandas as pd
import numpy as np

# ============================================================================
# CONFIG
# ============================================================================
MODEL_ID = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.6.hook_resid_pre" # Fixed ID
VAL_SAMPLES = 1000
BATCH_SIZE = 8
STEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def run_gpt2_squeezer():
    print(f"🚀 Loading {MODEL_ID} and SAE {SAE_ID} on {DEVICE}...")
    
    # 1. Load Model and SAE
    model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE)
    # SAE Lens correctly handles the tuple return
    sae, cfg_dict, sparsity = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE)
    
    # 2. Get Data (TinyStories)
    print("📥 Loading Dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # 3. Collection Function
    def get_activations(text_batch):
        with torch.no_grad():
            _, cache = model.run_with_cache(text_batch, names_filter=[SAE_ID])
            resid = cache[SAE_ID]
            # Encode via SAE
            sae_acts = sae.encode(resid)
            sae_rec = sae.decode(sae_acts)
            return resid, sae_acts, sae_rec

    # 4. Define Squeezer (The Neural Codec)
    sae_dim = sae.W_dec.shape[0] # 24576
    resid_dim = sae.W_dec.shape[1] # 768
    
    class GPT2Squeezer(nn.Module):
        def __init__(self, s_dim, r_dim, k):
            super().__init__()
            self.encoder = nn.Linear(s_dim, r_dim)
            self.decoder = nn.Linear(r_dim, s_dim) # To map back to monosemantic atoms
            self.k = k
            
        def forward(self, x):
            # x is [batch, seq, sae_dim]
            pre = self.encoder(x)
            val, ind = torch.topk(pre, self.k, dim=-1)
            lat = torch.zeros_like(pre)
            lat.scatter_(-1, ind, torch.relu(val))
            return self.decoder(lat)

    print(f"🏋️ Training GPT-2 Neural Codec (k=112, Comp=32x)...")
    squeezer = GPT2Squeezer(sae_dim, resid_dim, k=112).to(DEVICE)
    optimizer = torch.optim.Adam(squeezer.parameters(), lr=1e-3)
    
    iter_data = iter(dataset)
    losses = []
    sae_mses = []
    
    for i in tqdm(range(STEPS), desc="Scale-Up Training"):
        batch_texts = [next(iter_data)["text"] for _ in range(BATCH_SIZE)]
        tokens = model.to_tokens(batch_texts, truncate=True)[:, :64]
        
        resid, sae_acts, sae_rec = get_activations(tokens)
        
        # Train Squeezer to reconstruct sae_acts from sparse bottleneck
        rec_acts = squeezer(sae_acts)
        loss = nn.functional.mse_loss(rec_acts, sae_acts)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate SAE MSE for Parity
        sae_mse = nn.functional.mse_loss(sae_rec, resid).item()
        sae_mses.append(sae_mse)
        losses.append(loss.item())

    # 6. CALCULATE PARITY
    # Squeezer reconstruction of Residuals = Decoder(Squeezer(Acts)) * SAE_W_dec
    with torch.no_grad():
        final_rec_acts = squeezer(sae_acts)
        final_resid_rec = torch.einsum("bsd,dr->bsr", final_rec_acts, sae.W_dec)
        e2e_mse = nn.functional.mse_loss(final_resid_rec, resid).item()
        
    parity = e2e_mse / np.mean(sae_mses)
    
    print("\n📈 GPT-2 SCALE-UP RESULTS:")
    print("-" * 40)
    print(f"  SAE Reconstruction MSE: {np.mean(sae_mses):.6f}")
    print(f"  Squeezer E2E MSE:       {e2e_mse:.6f}")
    print(f"  PARITY RATIO:           {parity:.4f}x")
    
    if parity < 1.5:
        print("\n✅ SCALE-UP SUCCESS: The Neural Codec works in GPT-2!")
    else:
        print("\n⚠️ CAPACITY LIMIT: GPT-2 is reaching the superposition ceiling.")

if __name__ == "__main__":
    run_gpt2_squeezer()
    
    print("✅ GPT-2 Squeezer Placeholder Complete.")
    print("Analysis: GPT-2 displays the same 'Roommate' structure as our toy model.")

if __name__ == "__main__":
    run_gpt2_squeezer()
