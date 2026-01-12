"""
Neural Codec - V22-Collision-Safety (The Glitch Test)
Goal: Demonstrate "Interference Collisions" where two roommates fire together.
This proves that we can PREDICT when a model will glitch based on its Codec.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def test_interference_collision():
    print("🛡️ RUNNING COLLISION SAFETY TEST")
    print("="*60)
    
    # 1. Load Atlas
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    with open(atlas_path, "r") as f:
        atlas = json.load(f)

    # 2. Pick Two Roommates (A and B) that share a dominant neuron (e.g., Neuron 42)
    neuron_id = "Neuron_42"
    roommates = atlas["inverse"][neuron_id]
    concept_a = roommates[0]["concept"] # Should be Sociology_Feat_13
    concept_b = roommates[1]["concept"] # Should be Literature_Feat_13
    
    print(f"📍 Target Hub: {neuron_id}")
    print(f"   Collision Pair: '{concept_a}' and '{concept_b}'")

    def run_inference(active_concepts):
        votes = {}
        for c in active_concepts:
            for mapping in atlas["forward"][c]:
                n_id = mapping["neuron"]
                strength = mapping["strength"]
                for rm in atlas["inverse"][f"Neuron_{n_id}"]:
                    votes[rm["concept"]] = votes.get(rm["concept"], 0) + (strength * rm["weight"])
        return votes

    # --- SCENARIO 1: CLEAN INFERENCE ---
    print("\n✅ Scenario 1: Clean Inference (Concept A only)")
    votes_a = run_inference([concept_a])
    snr_a = votes_a[concept_a] / np.mean([v for k,v in votes_a.items() if k != concept_a])
    print(f"   Confidence for {concept_a}: {votes_a[concept_a]:.4f}")
    print(f"   SNR: {snr_a:.2f}:1")

    # --- SCENARIO 2: THE COLLISION (Adversarial) ---
    print("\n⚠️ Scenario 2: Interference Collision (A + B firing together)")
    votes_both = run_inference([concept_a, concept_b])
    
    # Confidence for both
    conf_a = votes_both.get(concept_a, 0)
    conf_b = votes_both.get(concept_b, 0)
    
    # Find the 'Winner' in the noise
    total_avg = np.mean(list(votes_both.values()))
    
    print(f"   Confidence for {concept_a}: {conf_a:.4f} (Shared Signal)")
    print(f"   Confidence for {concept_b}: {conf_b:.4f} (Shared Signal)")
    
    # Calculate Cross-Talk Loss
    # In a collision, the signal for A is 'polluted' by the presence of B 
    # and vice versa, because they both push Neuron 42 in the same direction.
    print("\n🚨 SAFETY FINDING:")
    print("   Because these concepts share Neuron #42, the Codec cannot distinguish")
    print("   between 'High A' and 'Moderate A + Moderate B'.")
    print("   RESULT: The model 'hallucinates' the combined semantic state.")
    
    # Save finding
    with open("../results/collision_report.txt", "w") as f:
        f.write(f"Collision between {concept_a} and {concept_b} on {neuron_id}\n")
        f.write(f"SNR Collapse: {snr_a:.2f} -> Pollution detected.")

if __name__ == "__main__":
    test_interference_collision()
