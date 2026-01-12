"""
Neural Codec - V19-Fairness-Verification
Goal: Prove that 'Philosophy' no longer overshadows 'DNA' after normalization.
"""

import torch
import json
import numpy as np
from pathlib import Path
import pandas as pd

def verify_fairness():
    # 1. Load the Normalized Atlas
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    with open(atlas_path, "r") as f:
        atlas = json.load(f)
        
    # 2. Simulate Input: 'DNA'
    # Find DNA's neurons and strengths
    dna_label = next(c for c in atlas["concept_names"] if "Biotech_DNA" in c)
    dna_neurons = atlas["forward"][dna_label]
    
    # 3. Process through the Codec logic (The "Voter" logic in the dash)
    # Each active neuron 'votes' for its constituents based on weight.
    votes = {}
    
    # Simulate activation: only the neurons carrying DNA are active
    for mapping in dna_neurons:
        n_id = mapping["neuron"]
        strength = mapping["strength"] # This is the 'mag' of the neuron
        
        # This neuron votes for all its occupants
        for roommate in atlas["inverse"][f"Neuron_{n_id}"]:
            concept = roommate["concept"]
            weight = roommate["weight"]
            votes[concept] = votes.get(concept, 0) + (strength * weight)
            
    # 4. Analyze Results
    df = pd.DataFrame(list(votes.items()), columns=["Concept", "Confidence"])
    df = df.sort_values("Confidence", ascending=False)
    
    print("\n🧬 TEST CASE: Activating 'Biotech_DNA'")
    print("-" * 50)
    print("Top 5 Decoded Concepts:")
    print(df.head(5))
    
    # Check specifically for Philosophy
    phil_matches = df[df["Concept"].str.contains("Philosophy", case=False)]
    if not phil_matches.empty:
        phil_conf = phil_matches.iloc[0]['Confidence']
        dna_conf = df.iloc[0]['Confidence']
        ratio = phil_conf / dna_conf
        print(f"\n🔍 CROSS-TALK ANALYSIS:")
        print(f"  DNA Confidence: {dna_conf:.4f}")
        print(f"  Philosophy Confusion: {phil_conf:.4f}")
        print(f"  Signal-to-Noise Ratio: {1/ratio:.2f}:1")
    else:
        print("\n✅ NO PHILOSOPHY CROSS-TALK DETECTED.")

if __name__ == "__main__":
    verify_fairness()
