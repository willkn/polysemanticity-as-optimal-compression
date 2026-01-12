"""
Neural Codec - V26-Danger-Detector
Goal: Use the Codec's semantic decoding to detect high-risk prompts 
      hidden in polysemantic activations.
"""

import json
import pandas as pd
from pathlib import Path

def run_danger_detector():
    print("🛡️ NEURAL CODEC DANGER DETECTOR")
    print("="*60)
    
    # 1. Load Atlas
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    with open(atlas_path, "r") as f:
        atlas = json.load(f)

    # 2. Define High-Risk Concepts
    # In our labeled model, we'll pick themes that sound dangerous (e.g., Chemistry_Poisons, Military_Weapons)
    # Since we labeled 32 domains, let's identify the 'Sensitive' ones.
    sensitive_themes = ["Military", "Chemistry", "Cryptography", "Legal"]
    
    # Identify specific 'Kill-Switch' atoms
    blacklist = [c for c in atlas["concept_names"] if any(t in c for t in sensitive_themes)]
    print(f"📍 Monitoring {len(blacklist)} sensitive semantic atoms.")

    def scan_activations(selected_concepts):
        # Decode the signal
        votes = {}
        for c in selected_concepts:
            mappings = atlas["forward"].get(c, [])
            for m in mappings:
                n_id = m["neuron"]
                strength = m["strength"]
                for rm in atlas["inverse"][f"Neuron_{n_id}"]:
                    votes[rm["concept"]] = votes.get(rm["concept"], 0) + (strength * rm["weight"])
        
        # Check for blacklist hits
        hits = {c: v for c, v in votes.items() if c in blacklist and v > 0.5} # Threshold for detection
        return hits, votes

    # --- TEST 1: Safe Input ---
    print("\n🟢 TEST 1: Safe Input (Kitten Cluster)")
    safe_concepts = [c for c in atlas["concept_names"] if "Zoology" in c][:3]
    hits, _ = scan_activations(safe_concepts)
    if not hits:
        print("   ✅ [CLEAN] No sensitive concepts detected above threshold.")
    else:
        print(f"   ⚠️ [FALSE POSITIVE] Detected: {list(hits.keys())}")

    # --- TEST 2: High-Risk Input ---
    print("\n🔴 TEST 2: High-Risk Input (Chemistry + Military)")
    danger_concepts = [c for c in atlas["concept_names"] if "Chemistry" in c][:2] + \
                      [c for c in atlas["concept_names"] if "Military" in c][:1]
    print(f"   Activating: {danger_concepts}")
    hits, _ = scan_activations(danger_concepts)
    if hits:
        print(f"   🚨 [DANGER FLAG] Detected suspicious activations:")
        for concept, score in sorted(hits.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      - {concept:25} | Confidence: {score:.4f}")
    else:
        print("   ❌ [FAILURE] Failed to detect dangerous concepts.")

    # --- TEST 3: Adversarial/Polysemantic Collision ---
    print("\n🟡 TEST 3: The 'Roommate' Attack")
    # Activate a safe concept that shares neurons with a dangerous one
    # We find a neuron that has both.
    neuron_42 = atlas["inverse"]["Neuron_42"]
    safe_atom = neuron_42[0]["concept"] # Likely safe
    danger_atom = next((rm["concept"] for rm in neuron_42 if any(t in rm["concept"] for t in sensitive_themes)), None)
    
    if danger_atom:
        print(f"   Targeting shared Hub: {safe_atom} <-> {danger_atom}")
        print(f"   User activates apparently safe '{safe_atom}'")
        hits, _ = scan_activations([safe_atom])
        print(f"   Detector Score for '{danger_atom}': {hits.get(danger_atom, 0):.4f}")
        if hits.get(danger_atom, 0) > 0.2:
             print("   🛡️ [PROTECTION] Detector caught high potential for semantic leakage.")
    else:
        print("   Skipping Test 3: No diverse roommates found in Hub 42.")

if __name__ == "__main__":
    run_danger_detector()
