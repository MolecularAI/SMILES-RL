from typing import List

from rdkit.Chem import MolFromSmiles, MolToSmiles, MolStandardize


def convert_to_canonical_smiles(
    smiles: List[str],
    allowTautomers=True,
    sanitize=False,
    isomericSmiles=False,
) -> List[str]:
    """
    :param smiles: Converts a smiles string into a canonical SMILES string.
    :type allowTautomers: allows having same molecule represented in different tautomeric forms
    """

    if allowTautomers:
        mols = [MolFromSmiles(smi, sanitize=sanitize) for smi in smiles]
        canon_smiles = [
            MolToSmiles(mol, isomericSmiles=isomericSmiles)
            if mol is not None
            else smiles[i]
            for i, mol in enumerate(mols)
        ]

        return canon_smiles
    else:
        return [MolStandardize.canonicalize_tautomer_smiles(smi) for smi in smiles]
