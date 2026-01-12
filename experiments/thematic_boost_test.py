"""
Neural Codec - V20-Thematic-Boost
Goal: Show how 'Thematic Multi-Feature Inputs' massively boost the SNR 
      compared to single-concept inputs.
"""

import torch
import json
import numpy as np
from pathlib import Path
import pandas as pd

def test_thematic_boost():
    # 1. Load Atlas
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    with open(atlas_path, "r") as f:
        atlas = json.load(f)
        
    def get_confidence(selected_labels):
        votes = {}
        for label in selected_labels:
            mappings = atlas["forward"][label]
            for mapping in mappings:
                n_id = mapping["neuron"]
                strength = mapping["strength"]
                for roommate in atlas["inverse"][f"Neuron_{n_id}"]:
                    concept = roommate["concept"]
                    weight = roommate["weight"]
                    votes[concept] = votes.get(concept, 0) + (strength * weight)
        return votes

    # --- TEST 1: Single DNA ---
    print("\n🧪 TEST 1: Single Feature ('Biotech_DNA')")
    dna_votes = get_confidence(["Biotech_DNA"])
    df_dna = pd.DataFrame(list(dna_votes.items()), columns=["Concept", "Conf"]).sort_values("Conf", ascending=False)
    
    # Calculate SNR: Target / Mean(Noise)
    target = df_dna.iloc[0]['Conf']
    noise_floor = df_dna.iloc[1:20]['Conf'].mean()
    print(f"  Target Confidence: {target:.4f}")
    print(f"  Avg Noise (Top 20): {noise_floor:.4f}")
    print(f"  SNR: {target/noise_floor:.2f}:1")

    # --- TEST 2: Thematic Package (Kitten) ---
    print("\n🚀 TEST 2: Thematic Package ('Whiskers' + 'Paws' + 'Fur' + 'Tail')")
    kitten_features = [c for c in atlas["concept_names"] if "Zoology" in c][:4]
    print(f"  Features: {kitten_features}")
    kitten_votes = get_confidence(kitten_features)
    df_kit = pd.DataFrame(list(kitten_votes.items()), columns=["Concept", "Conf"]).sort_values("Conf", ascending=False)
    
    target_kit = df_kit.iloc[0:4]['Conf'].mean()
    noise_kit = df_kit.iloc[4:20]['Conf'].mean()
    print(f"  Avg Target Confidence: {target_kit:.4f}")
    print(f"  Avg Noise (Top 20): {noise_kit:.4f}")
    print(f"  SNR: {target_kit/noise_kit:.2f}:1")

if __name__ == "__main__":
    test_thematic_boost()
