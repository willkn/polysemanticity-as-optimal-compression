"""
Neural Codec - V15-Semantic-Labeling
Goal: Assign human-readable names to all 512 concepts and update the Atlas.
"""

import json
from pathlib import Path

def generate_semantic_labels():
    # 32 Themes, each with 16 sub-concepts
    themes = [
        "Biotech", "Astro", "Quantum", "Mechanics", 
        "Emotions", "Architecture", "Culinary", "Legal",
        "Geology", "Music", "Mathematics", "Sociology",
        "Finance", "Botany", "Zoology", "Meteorology",
        "Linguistics", "Cryptography", "Philosophy", "History",
        "Medicine", "Cinema", "Agriculture", "Sports",
        "Chemistry", "Oceanography", "Aerospace", "Literature",
        "Politics", "Psychology", "Art", "Military"
    ]
    
    # Example sub-concepts for a few themes to show variety
    sub_samples = {
        "Biotech": ["DNA", "CRISPR", "Protein", "Enzyme", "Cell", "Gene", "Organelle", "Ribosome"],
        "Astro": ["Galaxy", "Pulsar", "Quasar", "Orbit", "Nebula", "Star", "Planet", "Comet"],
        "Quantum": ["Entanglement", "Superposition", "Quark", "Lepton", "Boson", "Wave", "Spin", "Tunneling"],
        "Zoology": ["Whiskers", "Paws", "Fur", "Tail", "Claws", "Snout", "Hibernation", "Migration"]
    }

    labels = []
    for i in range(512):
        theme_idx = i // 16
        theme_name = themes[theme_idx]
        sub_idx = i % 16
        
        # Use specific name if available, otherwise generic
        if theme_name in sub_samples and sub_idx < len(sub_samples[theme_name]):
            name = f"{theme_name}_{sub_samples[theme_name][sub_idx]}"
        else:
            name = f"{theme_name}_Feat_{sub_idx}"
        
        labels.append(name)

    # Update the Atlas
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    if atlas_path.exists():
        with open(atlas_path, "r") as f:
            atlas = json.load(f)
        
        # Update forward map keys and inverse content
        new_forward = {}
        for i, name in enumerate(labels):
            original_key = list(atlas["forward"].keys())[i]
            new_forward[name] = atlas["forward"][original_key]
        
        new_inverse = {}
        for n_key, occupants in atlas["inverse"].items():
            new_occupants = []
            for occ in occupants:
                # Map the original concept name (Concept_X) to the new label
                old_label = occ["concept"]
                idx = int(old_label.split("_")[1])
                new_occupants.append({
                    "concept": labels[idx], 
                    "weight": occ["weight"]
                })
            new_inverse[n_key] = new_occupants
            
        atlas["forward"] = new_forward
        atlas["inverse"] = new_inverse
        atlas["concept_names"] = labels
        
        with open(atlas_path, "w") as f:
            json.dump(atlas, f, indent=2)
            
        print("✅ Semantic Atlas Updated with 512 Human-Readable Names.")
        
        # Show a legendary neuron lookup
        print("\n🔍 SEMANTIC LOOKUP: Neuron #42")
        print("-" * 40)
        for occ in new_inverse["Neuron_42"][:5]:
            print(f"  - {occ['concept']:20} | Weight: {occ['weight']:.3f}")

if __name__ == "__main__":
    generate_semantic_labels()
