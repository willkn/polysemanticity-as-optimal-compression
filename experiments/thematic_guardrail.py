"""
Neural Codec - V28-Thematic-Guardrail
Goal: Use 'Thematic Consensus' to detect high-risk prompts with 
      drastically fewer false positives than individual atom monitoring.
"""

import json
import pandas as pd
from pathlib import Path

def run_thematic_guardrail():
    print("🛡️ THEMATIC GUARDRAIL (Safety Layer)")
    print("="*60)
    
    # 1. Load Atlas
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    with open(atlas_path, "r") as f:
        atlas = json.load(f)

    sensitive_themes = ["Military", "Chemistry", "Cryptography"]

    def analyze_risk(selected_concepts):
        # Tally theme-level support
        theme_votes = {}
        for c in selected_concepts:
            for m in atlas["forward"].get(c, []):
                n_id = m["neuron"]
                strength = m["strength"]
                for rm in atlas["inverse"][f"Neuron_{n_id}"]:
                    theme = rm["concept"].split("_")[0]
                    theme_votes[theme] = theme_votes.get(theme, 0) + (strength * rm["weight"])
        
        # Calculate Risk Score
        # Risk = Sum(Sensitive Themes) / Avg(All other Themes)
        sensitive_score = sum([theme_votes.get(t, 0) for t in sensitive_themes])
        other_scores = [v for k,v in theme_votes.items() if k not in sensitive_themes]
        baseline = sum(other_scores) / len(other_scores) if other_scores else 1.0
        
        # We also look for individual 'Peak' concepts in sensitive domains 
        # that are disproportionately active.
        return sensitive_score, baseline, theme_votes

    # --- TEST 1: Safe Input ---
    print("\n🟢 TEST 1: Safe Input (Kitten Cluster)")
    safe_concepts = [c for c in atlas["concept_names"] if "Zoology" in c][:10]
    risk, base, _ = analyze_risk(safe_concepts)
    print(f"   Sensitive Support: {risk:.2f}")
    print(f"   Baseline Support:  {base:.2f}")
    print(f"   Safety Ratio:      {risk/base:.2f}:1")
    if risk/base < 1.1:
        print("   ✅ [PASS] Input identified as non-sensitive.")
    else:
        print("   ⚠️ [WARNING] Borderline sensitive signal detected.")

    # --- TEST 2: Dangerous Input (Coherent Theme) ---
    print("\n🔴 TEST 2: High-Risk Input (Weapon Design Cluster)")
    # We simulate a coherent attack targeting 'Military' and 'Chemistry'
    danger_concepts = [c for c in atlas["concept_names"] if "Military" in c][:5] + \
                      [c for c in atlas["concept_names"] if "Chemistry" in c][:5]
    risk, base, votes = analyze_risk(danger_concepts)
    print(f"   Sensitive Support: {risk:.2f}")
    print(f"   Baseline Support:  {base:.2f}")
    print(f"   Safety Ratio:      {risk/base:.2f}:1")
    
    if risk/base > 1.15:
        print("   🚨 [DANGER FLAG] High thematic consensus for risky domains!")
        # Show top sensitive themes
        for t in sensitive_themes:
            if t in votes:
                print(f"      - {t:15} | Support: {votes[t]:.2f}")

if __name__ == "__main__":
    run_thematic_guardrail()
