# The Neural Codec

**Polysemanticity as a Structured Compression Algorithm**

## Overview

This project investigates whether polysemanticity in neural networks is not random "noise" but rather a structured, logical compression algorithm (a "Neural Codec"). We test this by decomposing model activations into monosemantic features via Sparse Autoencoders (SAEs), then re-compressing these features back into a bottleneck to recover algebraic mappings.

## Hypothesis

**Primary Hypothesis**: By manually re-compressing monosemantic features (from Sparse Autoencoders) back into a bottleneck of "synthetic neurons," we can recover a clean, algebraic mapping that defines how models resolve feature interference (e.g., via OR-gate logic or geometric polytopes).

**Validation Hypothesis**: If the interference patterns of our synthetic neurons match the original model's neurons, we have mathematically verified that the SAE features are the "true atoms" of the model.

## Project Structure

```
the-neural-codec/
├── experiments/        # All experimental code and scripts
├── results/           # Outputs, logs, plots, and analyses
├── research_brief.md  # Detailed research plan
├── story.md          # Experiment log and evolving insights
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Methodology

1. **Extraction (Decompress)**: Use pre-trained SAE to extract sparse, monosemantic feature activations from a language model layer
2. **The Squeezer (Re-Compress)**: Train a bottleneck model to compress SAE features back into M neurons (matching original model width)
3. **Pattern Analysis**: Analyze weight geometry, OR-gate logic, and circuit comparison

## Getting Started

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Pilot Experiment

```bash
python experiments/pilot_neural_codec.py
```

## Key Findings

See `story.md` for the evolving narrative and `results/` for experimental outputs.

## Compute Requirements

**Low** - Initial pilot experiments can run on CPU or single GPU

## References

- Research brief: `research_brief.md`
- Experiment log: `story.md`
- Results: `results/`

---

**Date**: 2026-01-10  
**Status**: Active Development
