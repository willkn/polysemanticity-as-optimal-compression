"""
Neural Codec - V24-Neuron-Alignment
Goal: Directly compare Squeezer's synthetic neurons with GPT-2's original neurons.
This addresses the 'Non-Negotiable' reviewer feedback.
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

# Config
MODEL_ID = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.6.hook_resid_pre"
BATCH_SIZE = 4
STEPS = 50 # Reduced for faster pilot verification
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def run_alignment_test():
    print(f"🚀 Loading {MODEL_ID} and SAE on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE)
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE)
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # Squeezer Architecture
    sae_dim = sae.W_dec.shape[0] # 24576
    resid_dim = sae.W_dec.shape[1] # 768
    
    class GPT2Squeezer(nn.Module):
        def __init__(self, s_dim, r_dim, k):
            super().__init__()
            self.encoder = nn.Linear(s_dim, r_dim)
            self.decoder = nn.Linear(r_dim, s_dim)
            self.k = k
            
        def forward(self, x):
            pre = self.encoder(x)
            val, ind = torch.topk(pre, self.k, dim=-1)
            lat = torch.zeros_like(pre)
            lat.scatter_(-1, ind, torch.relu(val))
            return lat, self.decoder(lat)

    squeezer = GPT2Squeezer(sae_dim, resid_dim, k=112).to(DEVICE)
    optimizer = torch.optim.Adam(squeezer.parameters(), lr=1e-3)
    
    iter_data = iter(dataset)
    
    print("🏋️ Training with Neuron Alignment Monitoring...")
    for i in tqdm(range(STEPS), desc="Training"):
        batch_texts = [next(iter_data)["text"] for _ in range(BATCH_SIZE)]
        tokens = model.to_tokens(batch_texts, truncate=True)[:, :64]
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[SAE_ID])
            resid = cache[SAE_ID]
            sae_acts = sae.encode(resid)
        
        # Squeezer
        bottleneck, rec_acts = squeezer(sae_acts)
        loss = nn.functional.mse_loss(rec_acts, sae_acts)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- THE ALIGNMENT TEST ---
    print("\n🔍 CALCULATING NEURON ALIGNMENT...")
    # We compare the activation vectors of original neurons vs squeezer neurons
    # over a validation batch.
    
    with torch.no_grad():
        v_texts = [next(iter_data)["text"] for _ in range(16)]
        v_tokens = model.to_tokens(v_texts, truncate=True)[:, :128]
        _, v_cache = model.run_with_cache(v_tokens, names_filter=[SAE_ID])
        v_resid = v_cache[SAE_ID].flatten(0, 1) # [N, 768]
        v_sae_acts = sae.encode(v_resid)
        v_bottleneck, _ = squeezer(v_sae_acts) # [N, 768]
        
    # Correlation Matrix [768, 768]
    # Normalize for cosine similarity
    v_resid_norm = v_resid / (v_resid.norm(dim=0, keepdim=True) + 1e-8)
    v_bot_norm = v_bottleneck / (v_bottleneck.norm(dim=0, keepdim=True) + 1e-8)
    
    corr_matrix = torch.matmul(v_resid_norm.t(), v_bot_norm) # [768, 768]
    
    # Max Correlation per Original Neuron
    max_corr, _ = corr_matrix.max(dim=1)
    mean_alignment = max_corr.mean().item()
    top_alignment = max_corr.max().item()
    
    print("-" * 40)
    print(f"📊 NEURON ALIGNMENT RESULTS:")
    print(f"  Mean Max-Correlation: {mean_alignment:.4f}")
    print(f"  Top Neuron Alignment: {top_alignment:.4f}")
    print("-" * 40)
    
    if mean_alignment > 0.5:
         print("✅ SUCCESS: Squeezer neurons are significantly aligned with GPT-2 neurons.")
    else:
         print("⚠️ LACK OF ALIGNMENT: The Squeezer found a valid but DIFFERENT compression.")

if __name__ == "__main__":
    run_alignment_test()
