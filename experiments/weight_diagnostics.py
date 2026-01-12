"""
Neural Codec - V18-Weight-Diagnostics
Goal: Investigate the 'Overshadowing' problem. 
Check for high-weight concepts that create false positives.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import pandas as pd

def run_diagnostics():
    # 1. Load the Atlas (and simulated weights)
    results_dir = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results")
    atlas_path = results_dir / "neural_atlas.json"
    
    with open(atlas_path, "r") as f:
        atlas = json.load(f)
        
    num_neurons = atlas["stats"]["total_neurons"]
    num_atoms = atlas["stats"]["total_concepts"]
    
    # We need to look at the 'loudness' of each concept
    # Loudness = Sum of weights for this concept across all neurons
    concept_loudness = {}
    
    # Loudness = L2 Norm of weights for this concept across all neurons
    concept_loudness = {}
    
    for concept, mappings in atlas["forward"].items():
        # L2 Norm: sqrt(sum of weights^2)
        l2_sum = sum([m["strength"]**2 for m in mappings])
        concept_loudness[concept] = np.sqrt(l2_sum)
    
    # Create a DataFrame for analysis
    df = pd.DataFrame(list(concept_loudness.items()), columns=["Concept", "Loudness"])
    df = df.sort_values("Loudness", ascending=False)
    
    print("\n🔊 CONCEPT LOUDNESS ANALYSIS (Top 10)")
    print("-" * 50)
    print(df.head(10))
    
    print("\n🔉 CONCEPT LOUDNESS ANALYSIS (Bottom 10)")
    print("-" * 50)
    print(df.tail(10))
    
    # Check for 'Philosophy' vs 'DNA' specifically if they exist
    dna_row = df[df["Concept"].str.contains("DNA", case=False)]
    phil_row = df[df["Concept"].str.contains("Philosophy", case=False)]
    
    if not dna_row.empty and not phil_row.empty:
        print("\n⚖️ SPECIFIC COMPARISON:")
        print(f"  DNA Loudness: {dna_row.iloc[0]['Loudness']:.4f}")
        print(f"  Philosophy Loudness: {phil_row.iloc[0]['Loudness']:.4f}")
        ratio = phil_row.iloc[0]['Loudness'] / dna_row.iloc[0]['Loudness']
        print(f"  Ratio: {ratio:.2f}x")
    
    # Summary stats
    print("\n📊 STATISTICAL SUMMARY:")
    print(f"  Mean Loudness: {df['Loudness'].mean():.4f}")
    print(f"  Std Dev: {df['Loudness'].std():.4f}")
    print(f"  Max/Min Ratio: {df['Loudness'].max() / df['Loudness'].min():.2f}x")

    # Save diagnostics
    df.to_csv(results_dir / "concept_diagnostics.csv", index=False)

if __name__ == "__main__":
    run_diagnostics()
