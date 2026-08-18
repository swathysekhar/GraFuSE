"""GraFuSE project module.
"""

from rdkit import Chem
import torch
from torch_geometric.data import Data

def smiles_to_graph(smiles):
    """Convert a SMILES string into a PyG molecular graph.

    Node features are derived from atomic properties and bond features from
    RDKit bond types. This function is a reusable extraction of the notebook's
    graph construction stage.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = []
    for atom in mol.GetAtoms():
        x.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ])

    edge_index = []
    edge_attr = []
    bond_map = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        b = bond_map.get(bond.GetBondType(), 0)
        edge_index += [[i, j], [j, i]]
        edge_attr += [[b], [b]]

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() \
        if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) \
        if edge_attr else torch.empty((0, 1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                smiles=smiles)
