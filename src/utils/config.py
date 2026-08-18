"""GraFuSE project module.
"""

from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    hidden_dim: int = 128
    max_smiles_length: int = 100
    model_option: int = 2
