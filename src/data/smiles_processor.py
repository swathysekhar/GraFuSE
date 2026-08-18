"""GraFuSE project module.
"""

import re
import torch

DEFAULT_TWO_CHAR_ELEMENTS = {"Cl", "Br", "Si", "Na", "Li", "Al", "Ca", "Mg", "Fe", "Zn"}

def tokenize_smiles(smiles, two_char_elements=None):
    """Tokenize a SMILES string while preserving common two-character elements."""
    two_char_elements = two_char_elements or DEFAULT_TWO_CHAR_ELEMENTS
    tokens = []
    i = 0
    while i < len(smiles):
        if i + 1 < len(smiles) and smiles[i:i+2] in two_char_elements:
            tokens.append(smiles[i:i+2])
            i += 2
        else:
            tokens.append(smiles[i])
            i += 1
    return tokens

class SMILESVocabulary:
    def __init__(self, tokens=None):
        base = ["<PAD>", "<UNK>", "<CLS>", "<SEP>"]
        self.itos = base + [t for t in (tokens or []) if t not in base]
        self.stoi = {t:i for i,t in enumerate(self.itos)}

    def encode(self, smiles, max_length=100):
        ids = [self.stoi.get(t, self.stoi["<UNK>"])
               for t in tokenize_smiles(smiles)]
        ids = ids[:max_length]
        return torch.tensor(ids + [self.stoi["<PAD>"]] * (max_length-len(ids)),
                            dtype=torch.long)

    def __len__(self):
        return len(self.itos)
