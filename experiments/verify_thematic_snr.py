"""
Neural Codec - V21-Thematic-SNR-Verification
Goal: Quantify the Signal-to-Noise Ratio (SNR) for thematic inputs.
Does a 'Cat' input clearly activate Cat concepts over unrelated roommates?
"""

import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path

def calculate_thematic_snr():
    # 1. Load Atlas
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    with open(atlas_path, "r") as f:
        atlas = json.load(f)
        
    num_neurons = atlas["stats"]["total_neurons"]
    
    # Define Target Theme (Zoology / Biotech - The 'Cat' world)
    target_concepts = [c for c in atlas["concept_names"] if "Zoology" in c or "Biotech" in c]
    # Pick a few key 'Cat' atoms
    cat_atoms = ["Zoology_Whiskers", "Zoology_Paws", "Zoology_Fur"]
    
    # 2. Simulate Inference: Activate Cat Atoms
    print(f"🐈 SIMULATING INPUT: {cat_atoms}")
    
    # Calculate global activation votes
    votes = {}
    total_neuron_pool = torch.zeros(num_neurons)
    
    for atom in cat_atoms:
        if atom not in atlas["forward"]: continue
        for mapping in atlas["forward"][atom]:
            n_id = mapping["neuron"]
            strength = mapping["strength"]
            total_neuron_pool[n_id] += strength

    # Decode from neuron pool
    for n_id in range(num_neurons):
        mag = total_neuron_pool[n_id].item()
        if mag == 0: continue
        for roommate in atlas["inverse"][f"Neuron_{n_id}"]:
            c_name = roommate["concept"]
            weight = roommate["weight"]
            votes[c_name] = votes.get(c_name, 0) + (mag * weight)
            
    # 3. Analyze SNR
    results = pd.DataFrame(list(votes.items()), columns=["Concept", "Score"]).sort_values("Score", ascending=False)
    
    # Signal: Mean score of target cat atoms
    signal_score = results[results["Concept"].isin(cat_atoms)]["Score"].mean()
    
    # Noise: Mean score of completely unrelated themes (e.g., Quantum or Finance)
    noise_themes = ["Quantum", "Finance", "Astro"]
    noise_df = results[results["Concept"].str.contains('|'.join(noise_themes))]
    noise_score = noise_df["Score"].mean()
    
    print("\n📊 THEMATIC DECODING RESULTS:")
    print("-" * 50)
    print("Top 10 Active Concepts in Codec Output:")
    print(results.head(10))
    
    print("\n🔍 SNR ANALYSIS:")
    print(f"  Signal (Cat Theme) Score: {signal_score:.4f}")
    print(f"  Noise (Quantum/Finance) Score: {noise_score:.4f}")
    print(f"  Signal-to-Noise Ratio: {signal_score/noise_score:.2f}:1")
    
    if signal_score/noise_score > 1.5:
        print("\n✅ IDEAL RESULT: Target concepts significantly outshine roommates.")
    else:
        print("\n⚠️ CROSS-TALK DETECTED: Roommates are leaking into the signal.")

if __name__ == "__main__":
    calculate_thematic_snr()
