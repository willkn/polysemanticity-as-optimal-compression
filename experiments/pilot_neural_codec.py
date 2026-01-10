"""
Neural Codec Pilot Experiment
Goal: Test the "Decompress → Re-Compress" hypothesis on a simple model.

Phase 1: Use a small transformer model + SAE to validate the approach
Phase 2: Build the "Squeezer" and analyze polysemanticity patterns
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Model settings
    model_name = "gpt2"  # Start with GPT-2 small
    layer_idx = 6  # Middle layer
    
    # SAE settings (we'll use a simplified SAE for this pilot)
    model_dim = 768  # GPT-2 hidden dimension
    sae_dim = 4096  # Expansion factor ~5x
    
    # Squeezer settings
    bottleneck_dim = 768  # Same as model_dim (synthetic neurons)
    
    # Training
    num_samples = 10000  # Number of activation samples
    batch_size = 32
    epochs = 50
    lr = 1e-3
    
    # Sparsity
    topk_features = 64  # Top-K active features
    
    # Output
    results_dir = Path("../results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

config = Config()
config.results_dir.mkdir(exist_ok=True, parents=True)

# ============================================================================
# SIMPLIFIED SAE (for pilot - we'll train our own)
# ============================================================================
class SimpleSAE(nn.Module):
    """
    Simplified Sparse Autoencoder for extracting monosemantic features.
    This is a proxy for pre-trained SAEs like from SAELens.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
    def encode(self, x):
        """Returns sparse feature activations"""
        features = torch.relu(self.encoder(x))
        return features
    
    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decoder(features)
        return reconstruction, features

# ============================================================================
# THE SQUEEZER (Core of Neural Codec)
# ============================================================================
class NeuralCodecSqueezer(nn.Module):
    """
    The Squeezer: Re-compresses monosemantic features into synthetic neurons.
    
    Architecture:
    - Input: SAE features (sae_dim)
    - Bottleneck: M synthetic neurons (bottleneck_dim)
    - Output: Reconstructed SAE features
    """
    def __init__(self, sae_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Linear(sae_dim, bottleneck_dim, bias=True)
        self.decoder = nn.Linear(bottleneck_dim, sae_dim, bias=True)
        
    def forward(self, sae_features):
        """
        Args:
            sae_features: [batch, sae_dim] sparse feature activations
        Returns:
            reconstructed: [batch, sae_dim] reconstructed features
            bottleneck: [batch, bottleneck_dim] synthetic neuron activations
        """
        bottleneck = torch.relu(self.encoder(sae_features))
        reconstructed = self.decoder(bottleneck)
        return reconstructed, bottleneck

# ============================================================================
# SYNTHETIC DATA GENERATION (for pilot)
# ============================================================================
def generate_synthetic_activations(num_samples, dim, sparsity=0.05):
    """
    Generate synthetic "model activations" for the pilot.
    In the full version, these would come from a real model.
    """
    activations = torch.randn(num_samples, dim) * 0.1
    # Add some structure: create feature clusters
    num_clusters = 10
    cluster_size = dim // num_clusters
    
    for i in range(num_samples):
        # Randomly activate 1-3 clusters
        active_clusters = np.random.choice(num_clusters, size=np.random.randint(1, 4), replace=False)
        for cluster_idx in active_clusters:
            start = cluster_idx * cluster_size
            end = start + cluster_size
            activations[i, start:end] += torch.randn(cluster_size) * 2.0
    
    return activations

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_sae(sae, activations, epochs=50, lr=1e-3):
    """Train the SAE to extract sparse features"""
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    print("Training SAE to extract monosemantic features...")
    for epoch in tqdm(range(epochs)):
        perm = torch.randperm(len(activations))
        epoch_loss = 0.0
        
        for i in range(0, len(activations), config.batch_size):
            batch_idx = perm[i:i + config.batch_size]
            batch = activations[batch_idx]
            
            # Forward pass
            reconstruction, features = sae(batch)
            
            # Loss: reconstruction + L1 sparsity
            recon_loss = nn.functional.mse_loss(reconstruction, batch)
            sparsity_loss = features.abs().mean()
            loss = recon_loss + 0.01 * sparsity_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return sae

def train_squeezer(squeezer, sae_features, epochs=50, lr=1e-3):
    """Train the Squeezer to re-compress SAE features"""
    optimizer = torch.optim.Adam(squeezer.parameters(), lr=lr)
    
    print("\nTraining Squeezer (Neural Codec)...")
    losses = []
    
    for epoch in tqdm(range(epochs)):
        perm = torch.randperm(len(sae_features))
        epoch_loss = 0.0
        
        for i in range(0, len(sae_features), config.batch_size):
            batch_idx = perm[i:i + config.batch_size]
            batch = sae_features[batch_idx]
            
            # Apply Top-K sparsity to input
            topk_vals, topk_indices = torch.topk(batch, k=config.topk_features, dim=1)
            sparse_batch = torch.zeros_like(batch)
            sparse_batch.scatter_(1, topk_indices, topk_vals)
            
            # Forward pass
            reconstructed, bottleneck = squeezer(sparse_batch)
            
            # Loss: reconstruction of original (non-sparse) features
            loss = nn.functional.mse_loss(reconstructed, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return squeezer, losses

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def analyze_polysemanticity(squeezer, sae_features):
    """
    Analyze the learned compression patterns in the Squeezer.
    
    Key metrics:
    1. Weight geometry: Are features arranged in polytopes?
    2. OR-gate logic: Do neurons combine mutually exclusive features?
    3. Neuron utilization: How many features per neuron?
    """
    
    print("\n" + "="*80)
    print("POLYSEMANTICITY ANALYSIS")
    print("="*80)
    
    # Get encoder weights (sae_dim -> bottleneck_dim)
    encoder_weights = squeezer.encoder.weight.data  # [bottleneck_dim, sae_dim]
    
    # 1. Features per neuron (polysemanticity measure)
    threshold = 0.1 * encoder_weights.abs().max()
    active_features_per_neuron = (encoder_weights.abs() > threshold).sum(dim=1)
    
    print(f"\n1. NEURON POLYSEMANTICITY:")
    print(f"   Mean features per neuron: {active_features_per_neuron.float().mean():.2f}")
    print(f"   Max features per neuron: {active_features_per_neuron.max()}")
    print(f"   Min features per neuron: {active_features_per_neuron.min()}")
    
    # 2. Feature sharing (how many neurons use each feature)
    features_per_neuron = (encoder_weights.abs() > threshold).sum(dim=0)
    
    print(f"\n2. FEATURE SHARING:")
    print(f"   Mean neurons per feature: {features_per_neuron.float().mean():.2f}")
    print(f"   Features used by >1 neuron: {(features_per_neuron > 1).sum()} / {len(features_per_neuron)}")
    
    # 3. Test OR-gate hypothesis: Find features that share neurons but never co-occur
    # Sample some activations
    sample_features = sae_features[:1000]
    topk_vals, topk_indices = torch.topk(sample_features, k=config.topk_features, dim=1)
    
    # Create co-occurrence matrix
    cooccurrence = torch.zeros(config.sae_dim, config.sae_dim)
    for indices in topk_indices:
        for i in indices:
            for j in indices:
                if i != j:
                    cooccurrence[i, j] += 1
    
    print(f"\n3. OR-GATE PATTERNS:")
    # Find neuron pairs that share a bottleneck neuron
    or_gate_count = 0
    for neuron_idx in range(config.bottleneck_dim):
        # Get features connected to this neuron
        connected_features = torch.where(encoder_weights[neuron_idx].abs() > threshold)[0]
        if len(connected_features) < 2:
            continue
        
        # Check co-occurrence
        for i in range(len(connected_features)):
            for j in range(i+1, len(connected_features)):
                f1, f2 = connected_features[i], connected_features[j]
                if cooccurrence[f1, f2] == 0:  # Never co-occur
                    or_gate_count += 1
    
    print(f"   Potential OR-gate pairs: {or_gate_count}")
    
    # 4. Weight visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(active_features_per_neuron.cpu().numpy(), bins=50)
    plt.xlabel("Features per Neuron")
    plt.ylabel("Count")
    plt.title("Neuron Polysemanticity Distribution")
    
    plt.subplot(1, 3, 2)
    plt.imshow(encoder_weights[:100, :100].cpu().numpy(), cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.xlabel("SAE Features")
    plt.ylabel("Synthetic Neurons")
    plt.title("Encoder Weight Matrix (subset)")
    
    plt.subplot(1, 3, 3)
    plt.scatter(
        active_features_per_neuron.cpu().numpy(),
        squeezer.encoder.bias.data.cpu().numpy(),
        alpha=0.5
    )
    plt.xlabel("Features per Neuron")
    plt.ylabel("Bias")
    plt.title("Polysemanticity vs Bias")
    
    plt.tight_layout()
    plt.savefig(config.results_dir / f"polysemanticity_analysis_{config.timestamp}.png")
    print(f"\n✅ Saved visualization to results/polysemanticity_analysis_{config.timestamp}.png")
    
    return {
        "mean_features_per_neuron": active_features_per_neuron.float().mean().item(),
        "max_features_per_neuron": active_features_per_neuron.max().item(),
        "or_gate_pairs": or_gate_count,
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def main():
    print("="*80)
    print("NEURAL CODEC PILOT EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model dimension: {config.model_dim}")
    print(f"  SAE dimension: {config.sae_dim}")
    print(f"  Bottleneck dimension: {config.bottleneck_dim}")
    print(f"  Top-K features: {config.topk_features}")
    print(f"  Training samples: {config.num_samples}")
    
    # Step 1: Generate synthetic activations
    print("\n[1/4] Generating synthetic model activations...")
    activations = generate_synthetic_activations(
        config.num_samples,
        config.model_dim
    )
    print(f"  Shape: {activations.shape}")
    
    # Step 2: Train SAE to extract monosemantic features
    print("\n[2/4] Training SAE (Feature Extraction)...")
    sae = SimpleSAE(config.model_dim, config.sae_dim)
    sae = train_sae(sae, activations, epochs=config.epochs, lr=config.lr)
    
    # Extract SAE features
    with torch.no_grad():
        _, sae_features = sae(activations)
    print(f"  Extracted features shape: {sae_features.shape}")
    print(f"  Mean sparsity: {(sae_features > 0).float().mean():.3f}")
    
    # Step 3: Train Squeezer (Neural Codec)
    print("\n[3/4] Training Squeezer (Re-Compression)...")
    squeezer = NeuralCodecSqueezer(config.sae_dim, config.bottleneck_dim)
    squeezer, losses = train_squeezer(squeezer, sae_features, epochs=config.epochs, lr=config.lr)
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Squeezer Training Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(config.results_dir / f"training_loss_{config.timestamp}.png")
    print(f"  ✅ Saved training plot")
    
    # Step 4: Analyze polysemanticity patterns
    print("\n[4/4] Analyzing Neural Codec...")
    analysis_results = analyze_polysemanticity(squeezer, sae_features)
    
    # Save results
    results = {
        "timestamp": config.timestamp,
        "config": {
            "model_dim": config.model_dim,
            "sae_dim": config.sae_dim,
            "bottleneck_dim": config.bottleneck_dim,
            "topk_features": config.topk_features,
            "num_samples": config.num_samples,
        },
        "analysis": analysis_results,
        "final_loss": losses[-1] if losses else None,
    }
    
    results_file = config.results_dir / f"pilot_results_{config.timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\n📊 Results saved to: {results_file}")
    print(f"\n🔑 Key Findings:")
    print(f"  - Mean polysemanticity: {analysis_results['mean_features_per_neuron']:.2f} features/neuron")
    print(f"  - OR-gate patterns found: {analysis_results['or_gate_pairs']}")
    print(f"  - Max polysemanticity: {analysis_results['max_features_per_neuron']} features")
    
    print(f"\n✅ Checkpoint 1 (Implementation Complete) - Ready for human review")
    
    return results

if __name__ == "__main__":
    main()
