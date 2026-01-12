"""
Neural Codec - V27-Jigsaw-Redundancy-Proof
Goal: Prove that Theme-Level Sums provide a robust 'Error-Correcting' signal.
Hypothesis: Total(Theme) / Max(Unrelated_Theme) grows as the cluster completes.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def run_jigsaw_proof():
    print("🧩 THE JIGSAW REDUNDANCY PROOF")
    print("="*60)
    
    # 1. Load Atlas
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    with open(atlas_path, "r") as f:
        atlas = json.load(f)

    cluster_theme = "Zoology"
    cluster_features = [c for c in atlas["concept_names"] if cluster_theme in c][:10]
    
    proof_results = []
    
    for i in range(1, len(cluster_features) + 1):
        active_subset = cluster_features[:i]
        
        # Decode the signal at the Domain Level
        theme_scores = {}
        for concept in active_subset:
            mappings = atlas["forward"].get(concept, [])
            for m in mappings:
                n_id = m["neuron"]
                strength = m["strength"]
                for rm in atlas["inverse"][f"Neuron_{n_id}"]:
                    rm_concept = rm["concept"]
                    rm_theme = rm_concept.split("_")[0]
                    weight = rm["weight"]
                    score = strength * weight
                    theme_scores[rm_theme] = theme_scores.get(rm_theme, 0) + score
        
        # Signal: Target Theme Support
        signal = theme_scores.get(cluster_theme, 0)
        
        # Noise: Highest support for an unrelated theme (excluding Zoology)
        unrelated_themes = {t: v for t, v in theme_scores.items() if t != cluster_theme}
        if unrelated_themes:
            # Noise is the 'Peak interference' theme
            noise_theme, noise_val = max(unrelated_themes.items(), key=lambda x: x[1])
        else:
            noise_theme, noise_val = "None", 0.0001
            
        snr = signal / noise_val
        
        proof_results.append({
            "pieces": i,
            "signal": signal,
            "peak_noise_theme": noise_theme,
            "peak_noise_val": noise_val,
            "snr": snr
        })
        
        print(f"🧩 Pieces: {i:2d} | Theme Sig: {signal:5.2f} | Peak Noise ({noise_theme}): {noise_val:5.2f} | SNR: {snr:5.2f}:1")

    # Final Validation
    df = pd.DataFrame(proof_results)
    snr_trend = df['snr'].iloc[-1] / df['snr'].iloc[0]
    
    print("\n📈 THEMATIC REDUNDANCY ANALYSIS:")
    print(f"   Initial SNR (1 piece):  {df['snr'].iloc[0]:.2f}:1")
    print(f"   Final SNR (10 pieces):  {df['snr'].iloc[-1]:.2f}:1")
    print(f"   Redundancy Gain:        {snr_trend:.2f}x")
    
    if snr_trend > 1.0:
        print("\n✅ REDUNDANCY PROVEN: The thematic signal accumulates faster than the peak interfering theme.")
        print("   This is why the model can be 'noisy' at the neuron level but 'correct' at the circuit level.")
    else:
        print("\n⚠️ NOISE COLLISION: Unrelated themes are accumulating as fast as the signal.")

if __name__ == "__main__":
    run_jigsaw_proof()
