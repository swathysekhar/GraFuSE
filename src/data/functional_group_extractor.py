"""GraFuSE project module.
"""

from rdkit import Chem

DEFAULT_FG_PATTERNS = {
    "alcohol": "[OX2H]",
    "phenol": "c[OX2H]",
    "amine_primary": "[NX3;H2]",
    "amine_secondary": "[NX3;H1]",
    "amine_tertiary": "[NX3;H0]",
    "amide": "C(=O)N",
    "carboxylic_acid": "C(=O)[OH]",
    "ester": "C(=O)O[#6]",
    "ether": "[OD2]([#6])[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "nitro": "[NX3](=O)=O",
    "halide": "[F,Cl,Br,I]",
    "thiol": "[SX2H]",
    "thioether": "[SX2]([#6])[#6]",
    "sulfonamide": "S(=O)(=O)N",
    "sulfone": "S(=O)(=O)[#6]",
    "phosphate": "P(=O)(O)(O)",
    "alkene": "C=C",
    "alkyne": "C#C",
    "aromatic": "a",
    "benzene_ring": "c1ccccc1",
    "nitrile": "C#N",
}

def compile_patterns(patterns=None):
    patterns = patterns or DEFAULT_FG_PATTERNS
    compiled = {}
    for name, smarts in patterns.items():
        mol = Chem.MolFromSmarts(smarts)
        if mol is not None:
            compiled[name] = mol
    return compiled

def extract_functional_groups(smiles, patterns=None):
    """Return matched functional groups and atom indices."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    compiled = compile_patterns(patterns)
    result = []
    for name, patt in compiled.items():
        for match in mol.GetSubstructMatches(patt):
            result.append({"name": name, "atom_indices": list(match)})
    return result
