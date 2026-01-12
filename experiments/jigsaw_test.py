"""
Neural Codec - V25-Jigsaw-Test
Goal: Prove that 'Thematic Clusters' act as a coherent signal that cancels out 
      random polysemantic noise.
Hypothesis: SNR improves linearly as the 'number of pieces' in the jigsaw increases.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run_jigsaw_test():
    print("🧩 RUNNING THE JIGSAW TEST (Redundancy Proof)")
    print("="*60)
    
    # 1. Load Atlas
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    with open(atlas_path, "r") as f:
        atlas = json.load(f)

    # 2. Define a Cluster (e.g., Zoology for 'Kitten')
    # Let's take 10 features from the Zoology theme
    cluster_theme = "Zoology"
    cluster_features = [c for c in atlas["concept_names"] if cluster_theme in c][:10]
    
    # 3. Define 'Noise' themes for comparison
    noise_themes = ["Quantum", "Finance", "Legal", "Music"]
    
    snr_results = []
    
    # 4. Iteratively add 'Jigsaw Pieces'
    for i in range(1, len(cluster_features) + 1):
        active_subset = cluster_features[:i]
        
        # Simulate Codec Activation (Summing votes)
        votes = {}
        for concept in active_subset:
            mappings = atlas["forward"].get(concept, [])
            for m in mappings:
                n_id = m["neuron"]
                strength = m["strength"]
                # Each neuron votes for its roommates
                for rm in atlas["inverse"][f"Neuron_{n_id}"]:
                    votes[rm["concept"]] = votes.get(rm["concept"], 0) + (strength * rm["weight"])
        
        # Calculate Signal Score (Avg confidence of the active subset)
        signal_score = np.mean([votes.get(c, 0) for c in active_subset])
        
        # Calculate Noise Score (Avg confidence of unrelated roommates)
        all_noise_concepts = [c for c in votes.keys() if not any(t in c for t in [cluster_theme])]
        noise_score = np.mean([votes[c] for c in all_noise_concepts]) if all_noise_concepts else 0.0001
        
        snr = signal_score / noise_score
        snr_results.append({
            "num_pieces": i,
            "signal": signal_score,
            "noise": noise_score,
            "snr": snr
        })
        
        print(f"🧩 Pieces: {i:2d} | Signal: {signal_score:.4f} | Noise: {noise_score:.4f} | SNR: {snr:.2f}:1")

    # 5. Save Results and Plot
    df = pd.DataFrame(snr_results)
    df.to_csv("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/jigsaw_results.csv", index=False)
    
    print("\n📈 JIGSAW ANALYSIS:")
    snr_gain = df.iloc[-1]['snr'] / df.iloc[0]['snr']
    print(f"   SNR Gain (1 -> 10 pieces): {snr_gain:.2f}x")
    
    if snr_gain > 1.2:
        print("\n✅ HYPOTHESIS CONFIRMED: Thematic clusters provide redundancy that cancels noise.")
        print("   The 'Kitten' signal sums up while the random polysemantic noise stays low.")
    else:
        print("\n⚠️ HYPOTHESIS WEAK: Minimal noise cancellation observed.")

if __name__ == "__main__":
    run_jigsaw_test()
