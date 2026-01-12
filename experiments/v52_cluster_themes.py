import numpy as np
import json
from sklearn.cluster import SpectralClustering
import torch

# Config
MATRIX_PATH = "/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/cofiring_matrix.npy"
TOTALS_PATH = "/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/total_firings.npy"
SAVE_PATH = "/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/automated_atlas.json"
NUM_THEMES = 40  # Let's target 40 natural concepts

def cluster_themes():
    print("Loading Matrix...")
    # Load raw co-occurrence (A * B^T)
    # Shape: [24576, 24576] float16
    adj = np.load(MATRIX_PATH).astype(np.float32)
    totals = np.load(TOTALS_PATH).astype(np.float32)

    # 1. Filter Dead Atoms to speed up clustering
    # Only keep atoms that fired at least X times
    mask = totals > 10
    active_indices = np.where(mask)[0]
    print(f"Active Atoms (>10 firings): {len(active_indices)}")
    
    # Sub-select adjacency
    sub_adj = adj[np.ix_(active_indices, active_indices)]
    sub_totals = totals[active_indices]

    # 2. Normalize to Jaccard Similarity or Conditional Prob
    # Jaccard(A, B) = (A n B) / (A u B) = (A n B) / (A + B - A n B)
    # We have intersections in sub_adj.
    # We have counts in sub_totals.
    
    print("Computing Jaccard Similarity...")
    # Broadcast addition: total[i] + total[j]
    union_matrix = sub_totals[:, None] + sub_totals[None, :] - sub_adj
    # Avoid div by zero
    union_matrix[union_matrix == 0] = 1.0
    
    similarity = sub_adj / union_matrix
    
    # 3. Spectral Clustering
    # This finds cuts that minimize edges between clusters
    print(f"Clustering into {NUM_THEMES} themes...")
    clustering = SpectralClustering(n_clusters=NUM_THEMES, 
                                    affinity='precomputed', 
                                    assign_labels='discretize',
                                    n_jobs=-1)
    
    labels = clustering.fit_predict(similarity)
    
    # 4. Organize Results
    atlas = {}
    
    for cluster_id in range(NUM_THEMES):
        # find indices in sub-matrix
        locs = np.where(labels == cluster_id)[0]
        # map back to global indices
        global_indices = active_indices[locs].tolist()
        
        # Calculate 'Coherence' (Density of connections)
        if len(locs) > 1:
            cluster_sub = similarity[np.ix_(locs, locs)]
            coherence = float(cluster_sub.mean())
        else:
            coherence = 1.0
            
        atlas[f"Theme_{cluster_id}"] = {
            "indices": global_indices,
            "coherence": coherence,
            "count": len(global_indices)
        }
        
    # 5. Save
    with open(SAVE_PATH, 'w') as f:
        json.dump(atlas, f, indent=2)
        
    print(f"Atlas saved to {SAVE_PATH}")
    
    # Print Top 5 Coherent Themes to Inspect
    sorted_themes = sorted(atlas.items(), key=lambda x: x[1]['coherence'], reverse=True)
    for name, data in sorted_themes[:5]:
        print(f"{name}: {data['count']} atoms, Coherence: {data['coherence']:.4f}")
        print(f"   Indices: {data['indices'][:10]}...")

if __name__ == "__main__":
    cluster_themes()
