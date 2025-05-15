""" Based on work from https://github.com/ml-jku/diverse-hits for a clean implemantation of the #Circles metric
"""

from typing import List, Optional

import networkx  # type: ignore
import numpy as np
import numpy.typing as npt
from rdkit.DataStructs.cDataStructs import ExplicitBitVect  # type: ignore
from sklearn.metrics import pairwise_distances

from .chem import ebv2np, morgan_from_smiles_list


def compute_distance_matrix_from_fps(
    ebvs: List[ExplicitBitVect], ebvs2=None, n_jobs: int = 8
) -> np.ndarray:
    """
    Compute the distance matrix from a list of fingerprint vectors.

    Args:
        ebvs (List[ExplicitBitVect]): List of fingerprint vectors.
        ebvs2 (Optional[List[ExplicitBitVect]]): Optional second list of fingerprint vectors.
        n_jobs (int): Number of parallel jobs to run. Default is 8.

    Returns:
        np.ndarray: The distance matrix.

    """
    fp_np = np.array([ebv2np(ebv) for ebv in ebvs]).astype(bool)
    if ebvs2 is None:
        distance_matrix = pairwise_distances(fp_np, metric="jaccard", n_jobs=n_jobs)
        return distance_matrix

    fp_np2 = np.array([ebv2np(ebv) for ebv in ebvs2]).astype(bool)
    return pairwise_distances(fp_np, fp_np2, metric="jaccard", n_jobs=n_jobs)


def compute_distance_matrix(
    smiles: List[str],
    radius: int = 2,
    nbits: int = 2048,
    n_jobs: int = 8,
) -> npt.NDArray[np.float32]:
    """Compute pairwise distances of smiles

    Args:
        smiles (List[str]): List of compounds
        radius (int): Morgan fingerprint radius
        nbits (int): Morgan fingerprint size

    Returns:
        np.ndarray: Matrix containing pairwise distances
    """
    fps: List[Optional[ExplicitBitVect]] = morgan_from_smiles_list(
        smiles, radius, nbits, n_jobs=n_jobs
    )

    assert all(fp is not None for fp in fps)
    return compute_distance_matrix_from_fps(fps, n_jobs=n_jobs)  # type: ignore


def internal_diversity(
    smiles: List[str], radius: int = 2, nbits: int = 2048, n_jobs: int = 8
):
    X = compute_distance_matrix(smiles, radius, nbits, n_jobs)
    return X.sum() / (X.shape[0] * (X.shape[0] - 1))


def compute_connectivity_matrix(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
) -> npt.NDArray[np.int32]:
    """Compute similarity graph edges given a threshold

    Args:
        smiles (List[str]): List of compounds
        distance_threshold (float): Sphere exclusion radius
        radius (int): Morgan fingerprint radius
        nbits (int): Morgan fingerprint size

    Returns:
        : Onehot connectivity matrix
    """
    pairwise_distances = compute_distance_matrix(smiles, radius=radius, nbits=nbits)
    connectivity_matrix = (pairwise_distances < distance_threshold).astype(int)
    np.fill_diagonal(connectivity_matrix, 0)  # inplace
    return connectivity_matrix


def compute_number_neighbours(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
) -> npt.NDArray[np.int32]:
    """Compute number of neighbours

    Args:
        smiles (List[str]): List of compounds
        distance_threshold (float): Sphere exclusion radius
        radius (int): Morgan fingerprint radius
        nbits (int): Morgan fingerprint size

    Returns:
        : Vector of number of neighbours for each compound
    """
    connectivity_matrix = compute_connectivity_matrix(
        smiles, distance_threshold, radius, nbits
    )
    return np.sum(connectivity_matrix, 0)


def compute_networkx_graph(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
) -> networkx.classes.graph.Graph:
    """
    Compute a NetworkX graph from a list of SMILES strings.

    Args:
        smiles (List[str]): A list of SMILES strings.
        distance_threshold (float, optional): The distance threshold for computing the connectivity matrix.
            Defaults to 0.65.
        radius (int, optional): The radius for computing the connectivity matrix. Defaults to 2.
        nbits (int, optional): The number of bits for computing the connectivity matrix. Defaults to 2048.

    Returns:
        networkx.classes.graph.Graph: A NetworkX graph representing the connectivity between the molecules.

    """
    connectivity_matrix = compute_connectivity_matrix(
        smiles, distance_threshold, radius, nbits
    )
    G = networkx.from_numpy_array(connectivity_matrix)
    return G


def sphere_exclusion_neighbour_dict(d: dict) -> List[int]:
    """Sphere exclusion algorithm for finding a maximal independent set based on a neighbour dictionary."""
    forbidden_nodes = set()
    selected_nodes = set()
    for node, neighbours in d.items():
        if node in forbidden_nodes:
            continue
        selected_nodes.add(node)
        forbidden_nodes.update(neighbours)

    return sorted(selected_nodes)
