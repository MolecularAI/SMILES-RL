""" Based on work from https://github.com/ml-jku/diverse-hits for a clean implemantation of the #Circles metric
"""

from rdkit import Chem
import numpy as np

from typing import Optional, List

from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem.Scaffolds import MurckoScaffold

import multiprocessing as mp
from functools import partial


def calculate_scaffold(smile: str) -> str:
    """Compute Bemis-Murcko (molecular) scaffold

    Args:
        smile (str): SMILES string

    Returns:
        str: Bemis-Murcko (molecular) scaffold of SMILES
    """
    mol = Chem.MolFromSmiles(smile)
    if mol:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except ValueError:
            scaffold_smiles = ""
    else:
        scaffold_smiles = ""
    return scaffold_smiles


def morgan_from_smiles(
    smiles: Optional[str], radius: int = 2, nbits: int = 2048
) -> Optional[ExplicitBitVect]:
    """Generates a Morgan/ECFP fingerprint from a smiles string.

    Returns:
        ExplicitBitVect | None: The fingerprint or None
    """
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nbits)


def morgan_from_smiles_list(
    smiles: List[str],
    radius: int = 2,
    nbits: int = 2048,
    n_jobs: Optional[int] = None,
) -> List[Optional[ExplicitBitVect]]:
    """Generate morgan fingerprints for a list of smiles

    Args:
        smiles (List[str]): List of smiles
        radius (int, optional): Radius for Morgan fingerprint. Defaults to 2.
        nbits (int, optional): Number of bits for Morgan fingerprint. Defaults to 2048.
        n_jobs (int, optional): Number of processes to use for multiprocessing. Defaults to None.

    Returns:
        List[np.ndarray | None]: List of fingerprints or None for invalid smiles
    """
    if n_jobs is None:
        fps = [morgan_from_smiles(s, radius=radius, nbits=nbits) for s in smiles]
    else:
        with mp.Pool(n_jobs) as p:
            fps = p.map(partial(morgan_from_smiles, radius=radius, nbits=nbits), smiles)

    return fps


def ebv2np(ebv: Optional[ExplicitBitVect]) -> Optional[np.ndarray]:
    """Explicit bit vector returned by rdkit to numpy array. Faster than just calling np.array(ebv)"""
    if ebv is None:
        return None
    return (np.frombuffer(bytes(ebv.ToBitString(), "utf-8"), "u1") - ord("0")).astype(bool)  # type: ignore
