""" Based on work from https://github.com/ml-jku/diverse-hits for a clean implemantation of the #Circles metric
"""

from inspect import getfullargspec
from time import time
from typing import List, Optional

import networkx  # type: ignore
import numpy as np
from rdkit import (
    DataStructs,  # type: ignore
    SimDivFilters,  # type: ignore
)
from rdkit.SimDivFilters import rdSimDivPickers  # type: ignore

from .chem import morgan_from_smiles, morgan_from_smiles_list
from .simgraph import (
    compute_networkx_graph,
    compute_number_neighbours,
)

import multiprocessing as mp


def maxmin(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
    n_picks: Optional[int] = None,
) -> List[int]:
    """
    Selects a diverse set of compounds from a given list of SMILES strings using the MaxMin algorithm.

    Args:
        smiles (List[str]): A list of SMILES strings representing the compounds.
        distance_threshold (float, optional): The distance threshold for similarity comparison. Defaults to 0.65.
        radius (int, optional): The radius parameter for generating Morgan fingerprints. Defaults to 2.
        nbits (int, optional): The number of bits for generating Morgan fingerprints. Defaults to 2048.
        n_picks (Optional[int], optional): The number of compounds to pick. If None, all compounds are picked.
            Defaults to None.

    Returns:
        List[int]: A list of indices representing the selected compounds.

    """

    # NOTE: For large lists of SMILES, it is faster to parallelize the computation of morgan fingerprints
    # fps = morgan_from_smiles_list(
    #     smiles, radius, nbits, n_jobs=(mp.cpu_count() - 1) or 1
    # )

    fps = [morgan_from_smiles(smiles=s, radius=radius, nbits=nbits) for s in smiles]

    if n_picks is None:
        n_picks = len(fps)
    else:
        n_picks = min(len(fps), n_picks)

    mmp = SimDivFilters.MaxMinPicker()  # type: ignore
    picks = mmp.LazyBitVectorPickWithThreshold(
        fps, len(fps), n_picks, distance_threshold
    )
    return list(picks[0])


def _randomize(pick_fun):
    """
    Randomizes the order of input smiles and applies the given pick function.

    Args:
        pick_fun: The pick function to apply on the randomized smiles.

    Returns:
        A function that takes the following arguments:
        - smiles: A list of SMILES strings.
        - distance_threshold: The distance threshold for similarity comparison.
        - radius: The radius for Morgan fingerprint calculation.
        - nbits: The number of bits for Morgan fingerprint calculation.
        - n_picks: The number of picks to select.
        - seed: The seed for random number generation.

        Returns:
        A list of indices representing the selected picks.
    """

    def randomized_fun(
        smiles: List[str],
        distance_threshold: float = 0.65,
        radius: int = 2,
        nbits: int = 2048,
        n_picks: Optional[int] = None,
        seed: int = 0,
    ) -> List[int]:
        np.random.seed(seed=seed)
        sort_idx = np.random.permutation(np.arange(len(smiles)))
        smiles_shuffled = np.array(smiles)[sort_idx].tolist()

        picks = pick_fun(
            smiles=smiles_shuffled,
            distance_threshold=distance_threshold,
            radius=radius,
            nbits=nbits,
            n_picks=n_picks,
        )
        return [int(sort_idx[i]) for i in picks]

    return randomized_fun


maxmin_random = _randomize(maxmin)


def leader_pick(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
    n_picks=None,
) -> List[int]:
    """Applies a leader algorithm to select a set of compounds with pairwise similarity greater than
    `distance_threshold`. The set and its size depends on the initial ordering.

    Args:
        smiles (list): List of compounds in smiles format.
        distance_threshold (float, optional): The sphere radius used for sphere exclusion. Defaults to 0.65.
        radius (int, optional): Morgan fingerprint radius. Defaults to 2.
        nbits (int, optional): Morgan fingerprint size. Defaults to 2048.

    Returns:
        list[int]: Indices of selected compounds
    """
    if len(smiles) == 0:
        return []

    # Compute fingerprints
    fps = [morgan_from_smiles(smiles=s, radius=radius, nbits=nbits) for s in smiles]

    # Select the compounds
    if n_picks is None:
        n_picks = len(fps)
    else:
        n_picks = min(len(fps), n_picks)

    lp = rdSimDivPickers.LeaderPicker()
    picks = lp.LazyBitVectorPick(fps, n_picks, distance_threshold)
    return list(picks)


leader_pick_random = _randomize(leader_pick)


def max_maxmin_random(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
    time_limit: float = 1.0,
) -> List[int]:
    """
    Selects the largest set of diverse molecules from a given list of SMILES strings using the maxmin_random algorithm.

    Args:
        smiles (List[str]): A list of SMILES strings representing the molecules.
        distance_threshold (float, optional): The distance threshold for similarity comparison. Defaults to 0.65.
        radius (int, optional): The radius parameter for the maxmin_random algorithm. Defaults to 2.
        nbits (int, optional): The number of bits for the Morgan fingerprint. Defaults to 2048.
        time_limit (float, optional): The time limit in seconds for the selection process. Defaults to 1.0.

    Returns:
        List[int]: The indices of the selected molecules in the original list.

    """
    start_time = time()
    sizes = []
    biggest_set = []
    max_size = 0
    i = 0
    while time() - start_time <= time_limit:
        picks = maxmin_random(smiles, distance_threshold, radius, nbits, seed=i)
        i += 1
        size = len(picks)
        sizes.append(size)
        if size > max_size:
            biggest_set = picks
            max_size = size
    return biggest_set


def taylor_butina(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
) -> List[int]:
    """Taylor-Butina clustering, which is equivalent with leader picker selection
    where inputs are sorted by their number of neighbours, more neighbours coming first.

    Args:
        smiles (List[str]): List of compounds
        distance_threshold (float): Sphere exclusion radius
        radius (int): Morgan fingerprint radius
        nbits (int): Morgan fingerprint size

    Returns:
        List[int]: Indices of selected compounds
    """
    n_neighbours = compute_number_neighbours(
        smiles, distance_threshold=distance_threshold, radius=radius, nbits=nbits
    )

    sort_idx = np.argsort(n_neighbours)[::-1]
    smiles_sorted = np.array(smiles)[sort_idx].tolist()

    picks = leader_pick(
        smiles=smiles_sorted,
        distance_threshold=distance_threshold,
        radius=radius,
        nbits=nbits,
    )
    return [smiles_sorted[i] for i in picks]


def reverse_taylor_butina(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
) -> List[int]:
    """Reverse Taylor-Butina clustering. Compounds with low number of neighbours are selected first.
    Tends to lead to more clusters.

    Args:
        smiles (List[str]): List of compounds
        distance_threshold (float): Sphere exclusion radius
        radius (int): Morgan fingerprint radius
        nbits (int): Morgan fingerprint size

    Returns:
        List[int]: Indices of selected compounds
    """

    n_neighbours = compute_number_neighbours(
        smiles, distance_threshold=distance_threshold, radius=radius, nbits=nbits
    )

    sort_idx = np.argsort(n_neighbours)
    smiles_sorted = np.array(smiles)[sort_idx].tolist()

    picks = leader_pick(
        smiles=smiles_sorted,
        distance_threshold=distance_threshold,
        radius=radius,
        nbits=nbits,
    )
    return [sort_idx[i] for i in picks]


def dise(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
) -> List[int]:
    """Reverse Taylor-Butina clustering. Compounds with low number of neighbours are selected first.
    Tends to lead to more clusters.

    Args:
        smiles (List[str]): List of compounds
        distance_threshold (float): Sphere exclusion radius
        radius (int): Morgan fingerprint radius
        nbits (int): Morgan fingerprint size

    Returns:
        List[int]: Indices of selected compounds
    """
    sildenafil_smiles = (
        "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"
    )
    sildenafil_fp = morgan_from_smiles(sildenafil_smiles)
    fps = [morgan_from_smiles(s, radius=radius, nbits=nbits) for s in smiles]
    sims = DataStructs.BulkTanimotoSimilarity(sildenafil_fp, fps)  # type: ignore
    sort_idx = np.argsort(sims)[::-1]
    smiles_sorted = np.array(smiles)[sort_idx].tolist()

    picks = leader_pick(
        smiles=smiles_sorted,
        distance_threshold=distance_threshold,
        radius=radius,
        nbits=nbits,
    )
    return [smiles_sorted[i] for i in picks]


def networkx_max_random_mis(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
    time_limit: float = 5.0,
    seed: int = 0,
) -> List[int]:
    """
    Finds the maximum random maximal independent set (MIS) in a graph using NetworkX.

    Args:
        smiles (List[str]): A list of SMILES strings representing the molecules.
        distance_threshold (float, optional): The distance threshold for computing the graph. Defaults to 0.65.
        radius (int, optional): The radius for computing the graph. Defaults to 2.
        nbits (int, optional): The number of bits for computing the graph. Defaults to 2048.
        time_limit (float, optional): The time limit for the algorithm. Defaults to 5.0.
        seed (int, optional): The seed for random number generation. Defaults to 0.

    Returns:
        List[int]: The list of indices representing the maximum random MIS.

    """
    start_time = time()
    G = compute_networkx_graph(smiles, distance_threshold, radius, nbits)
    sizes = []
    max_picks = []
    max_size = 0
    i = 0
    while time() - start_time <= time_limit:
        curr_picks = list(networkx.maximal_independent_set(G, seed=seed + i))  # type: ignore
        i += 1
        size = len(curr_picks)
        sizes.append(size)
        if size > max_size:
            max_picks = curr_picks
            max_size = size
    return max_picks


def networkx_maximum_independent_set(
    smiles: List[str],
    distance_threshold: float = 0.65,
    radius: int = 2,
    nbits: int = 2048,
) -> List[int]:
    G = compute_networkx_graph(smiles, distance_threshold, radius, nbits)

    picks = networkx.approximation.maximum_independent_set(G)
    return [smiles[i] for i in picks]


se_algorithms = {
    "leader_pick": leader_pick,
    "leader_pick_random": leader_pick_random,
    "maxmin": maxmin,
    "maxmin_random": maxmin_random,
    "max_maxmin_random": max_maxmin_random,
    "taylor_butina": taylor_butina,
    "reverse_taylor_butina": reverse_taylor_butina,
    "dise": dise,
    "networkx_max_random_mis": networkx_max_random_mis,
    "networkx_maximum_independent_set": networkx_maximum_independent_set,
}


for name, fun in se_algorithms.items():
    spec = getfullargspec(fun)
    assert spec.args[:4] == ["smiles", "distance_threshold", "radius", "nbits"]

    spec.annotations["return"] = "typing.List[int]"
    assert spec.defaults[:3] == (0.65, 2, 2048)  # type: ignore
