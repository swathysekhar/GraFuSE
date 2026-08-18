# GraFuSE

**Graph–SMILES–Functional Group Cross-Attention Framework for Molecular Property Prediction**

GraFuSE is a multimodal molecular property prediction framework integrating:

1. Molecular graph representations
2. SMILES sequence representations
3. Chemically meaningful functional-group representations
4. Cross-attention-based multimodal fusion

The repository is refactored from the supplied `Test_Val(FG+SMILES+GRAPH).ipynb`.

## Repository structure

- `src/data/` — graph, SMILES and functional-group preprocessing
- `src/models/` — Graph Transformer, SMILES Transformer, FG encoder and cross-attention
- `src/training/` — training, evaluation and metrics
- `src/utils/` — configuration, reproducibility and visualization
- `notebooks/` — reproducible experiments
- `functional_groups/` — functional-group definitions
- `configs/` — experiment configuration
- `results/` — performance, ablation and interpretability outputs

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

See `data/README.md`. Large datasets are not committed to this repository.

## Model configurations

The intended ablation settings are:

- `model_option=0`: Graph + SMILES
- `model_option=1`: Graph + Functional Groups
- `model_option=2`: Graph + SMILES + Functional Groups

## Reproducibility

Use `src.utils.seed.set_seed()` before training and record the configuration used for each experiment.

## Notebook

The original experiment notebook is retained under `notebooks/` as a reference. The modular source files are intended for reusable training and evaluation.

## Citation

Add the final paper citation here after publication.
