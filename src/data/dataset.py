"""GraFuSE project module.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from .graph_builder import smiles_to_graph
from .smiles_processor import SMILESVocabulary
from .functional_group_extractor import extract_functional_groups

class MolecularDataset(Dataset):
    """Dataset wrapper for SMILES, labels, graph and functional-group inputs."""

    def __init__(self, dataframe, vocab=None, smiles_col="smiles",
                 label_col="label", max_length=100):
        self.df = dataframe.reset_index(drop=True)
        self.vocab = vocab or SMILESVocabulary()
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row[self.smiles_col]
        label = row[self.label_col]
        graph = smiles_to_graph(smiles)
        sequence = self.vocab.encode(smiles, self.max_length)
        functional_groups = extract_functional_groups(smiles)
        return {
            "graph": graph,
            "smiles_ids": sequence,
            "functional_groups": functional_groups,
            "label": torch.tensor(label, dtype=torch.float),
            "smiles": smiles,
        }
