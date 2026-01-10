# The Story of The Neural Codec

## Current Mental Model
*Start by summarizing your understanding of the problem based on the research brief.*

## Evolving Hypothesis
*Track how your hypothesis changes as you see results.*

## Experiment Log
| Run ID | Description | Outcome | Key Findings |
|--------|-------------|---------|--------------|
|        |             |         |              |

## Next Steps
- [ ] Setup environment
- [ ] Implement pilot experiment

## 2026-01-10 18:48 - Initial Setup & Pilot Launch

**Action**: Created independent git repository for Neural Codec project  
**Outcome**: Success  
**Key Findings**: 
- Initialized new repo in `/projects/the-neural-codec/`
- Created comprehensive README.md and .gitignore
- Implemented pilot experiment (`experiments/pilot_neural_codec.py`)

**Action**: Launched pilot experiment  
**Outcome**: Running  
**Details**:
- Config: 768-dim model → 4096-dim SAE → 768-dim bottleneck
- Synthetic data: 10,000 samples with clustered structure
- Training: 50 epochs for both SAE and Squeezer
- ETA: ~6-7 minutes total

**Mental Model**: Testing the hypothesis that polysemanticity is structured compression. The Squeezer should learn to pack monosemantic features into polysemantic neurons following discoverable patterns (OR-gates, polytopes).

**Next Expected Outputs**:
1. Polysemanticity analysis (features per neuron distribution)
2. OR-gate pattern detection
3. Weight geometry visualization
4. Reconstruction quality metrics
