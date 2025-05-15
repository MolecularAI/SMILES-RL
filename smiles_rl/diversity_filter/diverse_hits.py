""" Based on work from https://github.com/ml-jku/diverse-hits for a clean implemantation of the #Circles metric
"""

from copy import deepcopy

import numpy as np

from .base_diversity_filter import (
    BaseDiversityFilter,
)
from .diversity_filter_parameters import (
    DiversityFilterParameters,
)
from reinvent_scoring.scoring.score_summary import FinalSummary

from typing import List


from rdkit.DataStructs.cDataStructs import ExplicitBitVect

import multiprocessing as mp


from .selection import se_algorithms

from .simgraph import compute_distance_matrix_from_fps

from .chem import morgan_from_smiles, calculate_scaffold


class DiverseHits(BaseDiversityFilter):
    """Penalizes compounds based on atom pair Tanimoto similarity to previously generated Murcko Scaffolds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)
        self._diverse_hits_fingerprints = {}
        self.reference_fps = []

    def update_score(self, score_summary: FinalSummary, step=0) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        # Indices of hits/actives found in current step
        hits_idxs = []

        for i in score_summary.valid_idxs:

            # Get Canonical SMILES
            smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])

            # No reward for duplicates
            scores[i] = 0 if self._smiles_exists(smile) else scores[i]

            # Only consider hits/actives in memory
            if scores[i] >= self.parameters.minscore:
                hits_idxs.append(i)
                scaffold = calculate_scaffold(smile)
                self._add_to_memory(
                    i,
                    scores[i],
                    smile,
                    smiles[i],
                    scaffold,
                    score_summary.scaffold_log,
                    step,
                )

        # Get smiles of hits/actives
        hits = [smiles[i] for i in hits_idxs]

        # Get indices of diverse actives
        picks_idx = self.compute_diverse_solutions(
            hits, 1 - self.parameters.minsimilarity, algorithm="maxmin"
        )

        # Number of diverse actives found in current step
        n_picks = len(picks_idx)

        # Add instrinsic reward to diverse actives found in the current step
        for i in picks_idx:
            scores[i] += n_picks

        return scores

    def calculate_novelty(
        self,
        reference_fps: List[ExplicitBitVect],
        query_fps: List[ExplicitBitVect],
        distance_threshold: float,
    ) -> List[int]:
        """
        Calculate the novelty of query fingerprints compared to reference fingerprints.

        Args:
            reference_fps (List[ExplicitBitVect]): List of reference fingerprints.
            query_fps (List[ExplicitBitVect]): List of query fingerprints.
            distance_threshold (float): Threshold for the distance matrix.

        Returns:
            List[int]: List of indices of query fingerprints that are considered novel.

        """
        distance_matrix = compute_distance_matrix_from_fps(
            reference_fps, query_fps, n_jobs=(mp.cpu_count() - 1) or 1
        )
        return (distance_matrix > distance_threshold).all(axis=0).nonzero()[0]

    def compute_diverse_solutions(
        self,
        smiles: List[str],
        distance_threshold: float,
        radius: int = 2,
        nbits: int = 2048,
        algorithm: str = "maxmin_random",
    ) -> List[int]:
        """Compute diverse solutions from a list of smiles



        Args:
            smiles (List[str]): The list of smiles
            distance_threshold (float): The distance threshold
            radius (int, optional): The radius of the morgan fingerprint. Defaults to 2.
            nbits (int, optional): The number of bits of the morgan fingerprint. Defaults to 2048.
            algorithm (str, optional): The algorithm to use. Defaults to "maxmin_random".
            reference_smiles (List[str], optional): The reference smiles. Defaults to [].

        Returns:
            List[int]: The list of indices of the diverse solutions
        """
        if len(smiles) == 0:
            return []

        # gets train actives and computes which of the solutions are novel
        # the novel_idx is a list of indices of the novel solutions
        # we need them to map the indices of the diverse solutions to the original indices
        if len(self.reference_fps) > 0:
            fps = [
                morgan_from_smiles(smiles=s, radius=radius, nbits=nbits) for s in smiles
            ]
            novel_idx = self.calculate_novelty(
                self.reference_fps, fps, distance_threshold
            )
            selection_input = np.array(smiles)[novel_idx].tolist()
        else:
            selection_input = smiles
            novel_idx = list(range(len(smiles)))

        if len(selection_input) == 0:
            return []

        # find the diverse solutions
        picks = se_algorithms[algorithm](
            selection_input,
            distance_threshold=distance_threshold,
            radius=radius,
            nbits=nbits,
        )

        picks_fps = [
            morgan_from_smiles(smiles=smiles[idx], radius=radius, nbits=nbits)
            for idx in picks
        ]

        self.reference_fps.extend(picks_fps)

        return sorted([novel_idx[i] for i in picks])
