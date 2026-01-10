# Research Brief: The Neural Codec
Date: 2026-01-10
Merit Score: 7

## Hypothesis
Polysemanticity in neural networks is not stochastic "noise" but a structured, logical compression algorithm (a Neural Codec).
Primary Hypothesis: By manually re-compressing monosemantic features (from Sparse Autoencoders) back into a bottleneck of "synthetic neurons," we can recover a clean, algebraic mapping that defines how models resolve feature interference (e.g., via OR-gate logic or geometric polytopes).
Validation Hypothesis: If the interference patterns of our synthetic neurons match the original model’s neurons, we have mathematically verified that the SAE features are the "true atoms" of the model.

## Methodology
The experiment follows a "Decompress 
→
→
 Re-Compress" loop:
Extraction (Decompress): Use a pre-trained SAE (from SAELens or OpenAI) to extract sparse, monosemantic feature activations (
F
F
) from a specific layer of a Small Language Model (SLM).
The Squeezer (Re-Compress): Train a "Squeezer" model—a tiny, single-layer linear bottleneck followed by a ReLU:
Input: SAE Feature Activations (e.g., 32,000 features).
Bottleneck: 
M
M
 neurons (set 
M
M
 equal to the original model’s width, e.g., 768 or 1024).
Output: Reconstructed SAE features.
Optimization: Use Top-K sparsity in the input and train the Squeezer to minimize Reconstruction Loss. Crucially, the Squeezer is forced to find the most efficient way to "pack" these 32,000 features into 1,024 slots.
Pattern Analysis:
Weight Geometry: Check if the weight matrix 
W
W
 shows features arranged in regular polytopes (e.g., "Circular" structures for temporal features).
Logic Extraction: Test the "OR-gate" hypothesis by looking for features that share a neuron but never co-occur in the same data sample.
Circuit Comparison: Compare the "firing patterns" of these synthetic polysemantic neurons to the firing patterns of the original model's neurons.

## Compute Cost
Low

## Revised Conclusion
Custom idea submitted by user (not debated)

## Debate Transcript


## Project Plan & Roadmap
1. **Pilot Phase**: Implement The Neural Codec on a synthetic or small-scale dataset (e.g., MNIST or TinyStories).
2. **Baseline**: Establish a baseline comparison using standard methods.
3. **Core Experiment**: Test the primary hypothesis: "Polysemanticity in neural networks is not stochastic "noise" but a structured, logical compression algorithm (a Neural Codec).
Primary Hypothesis: By manually re-compressing monosemantic features (from Sparse Autoencoders) back into a bottleneck of "synthetic neurons," we can recover a clean, algebraic mapping that defines how models resolve feature interference (e.g., via OR-gate logic or geometric polytopes).
Validation Hypothesis: If the interference patterns of our synthetic neurons match the original model’s neurons, we have mathematically verified that the SAE features are the "true atoms" of the model."
4. **Analysis**: Evaluate results against metrics identified in the methodology.

## Findings Handoff Points (HITL)
- **Checkpoint 1 (Implementation complete)**: Review the model architecture and training script before the full run.
- **Checkpoint 2 (Baseline results)**: Present baseline metrics to ensure the experiment setup is valid.
- **Checkpoint 3 (Final Synthesis)**: Share preliminary results and plots for human interpretation before writing the final report.

