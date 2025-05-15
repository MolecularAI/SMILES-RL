from copy import deepcopy

import numpy as np

from reinvent_scoring.scoring.score_summary import FinalSummary

from typing import List


from rdkit.DataStructs.cDataStructs import ExplicitBitVect

import multiprocessing as mp


from .base_diversity_filter import (
    BaseDiversityFilter,
)
from .diversity_filter_parameters import (
    DiversityFilterParameters,
)


from .selection import se_algorithms

from .simgraph import compute_distance_matrix_from_fps

from .chem import morgan_from_smiles, calculate_scaffold


class MeanSimilarity(BaseDiversityFilter):
    """Penalizes compounds based on atom pair Tanimoto similarity to previously generated Murcko Scaffolds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)
        self._diverse_hits_fingerprints = {}
        self.reference_fps = []

    def update_score(self, score_summary: FinalSummary, step=0) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        hits_idxs = []

        for i in score_summary.valid_idxs:
            smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])

            scores[i] = 0 if self._smiles_exists(smile) else scores[i]
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

        hits = [smiles[i] for i in hits_idxs]

        novelty = self.compute_novelty_score(hits)

        scores[hits_idxs] += novelty

        return scores

    def compute_novelty_score(
        self,
        smiles: List[str],
        radius: int = 2,
        nbits: int = 2048,
        distance_threshold: float = 0.7,
        algorithm: str = "maxmin",
    ) -> np.ndarray:
        """Compute diverse solutions from a list of smiles

        Args:
            smiles (List[str]): The list of smiles
            distance_threshold (float): The distance threshold
            radius (int, optional): The radius of the morgan fingerprint. Defaults to 2.
            nbits (int, optional): The number of bits of the morgan fingerprint. Defaults to 2048.

        Returns:
            List[int]: The list of indices of the diverse solutions
        """
        if len(smiles) == 0:
            return np.array([])

        fps = [morgan_from_smiles(smiles=s, radius=radius, nbits=nbits) for s in smiles]

        # gets train actives and computes which of the solutions are novel
        # the novel_idx is a list of indices of the novel solutions
        # we need them to map the indices of the diverse solutions to the original indices
        if len(self.reference_fps) > 0:
            distance_matrix = compute_distance_matrix_from_fps(
                fps, self.reference_fps + fps, n_jobs=(mp.cpu_count() - 1) or 1
            )

            novel_idx = self.calculate_novelty(
                self.reference_fps, fps, distance_threshold
            )
            selection_input = np.array(smiles)[novel_idx].tolist()
        elif len(smiles) == 1:
            self.reference_fps.extend(fps)

            return np.zeros(1)
        else:
            distance_matrix = compute_distance_matrix_from_fps(
                fps, n_jobs=(mp.cpu_count() - 1) or 1
            )
            selection_input = smiles
            novel_idx = list(range(len(smiles)))

        distance_matrix_mask = np.ma.array(distance_matrix, mask=False)

        di = (
            np.arange(distance_matrix.shape[0]),
            np.arange(
                distance_matrix.shape[1] - distance_matrix.shape[0],
                distance_matrix.shape[1],
            ),
        )

        distance_matrix_mask.mask[di] = True

        min_dists = distance_matrix_mask.mean(axis=1)

        if len(selection_input) == 0:
            return min_dists

        # find the diverse solutions
        picks = se_algorithms[algorithm](
            selection_input,
            distance_threshold=distance_threshold,
            radius=radius,
            nbits=nbits,
        )

        picks_idxs = sorted([novel_idx[i] for i in picks])
        picks_fps = [fps[idx] for idx in picks_idxs]

        self.reference_fps.extend(picks_fps)

        return min_dists

    def compute_novelty_score_deprecated(
        self,
        smiles: List[str],
        radius: int = 2,
        nbits: int = 2048,
    ) -> np.ndarray:
        """Compute diverse solutions from a list of smiles

        Args:
            smiles (List[str]): The list of smiles
            distance_threshold (float): The distance threshold
            radius (int, optional): The radius of the morgan fingerprint. Defaults to 2.
            nbits (int, optional): The number of bits of the morgan fingerprint. Defaults to 2048.

        Returns:
            List[int]: The list of indices of the diverse solutions
        """
        if len(smiles) == 0:
            return np.array([])

        # gets train actives and computes which of the solutions are novel
        # the novel_idx is a list of indices of the novel solutions
        # we need them to map the indices of the diverse solutions to the original indices
        fps = [
            morgan_from_smiles(
                smiles=smi,
                radius=radius,
                nbits=nbits,
            )
            for smi in smiles
        ]

        # Adds current batch of fps since we want to stay away from the current sampled ones as well
        self.reference_fps.extend(fps)
        distance_matrix = compute_distance_matrix_from_fps(
            fps, self.reference_fps, n_jobs=(mp.cpu_count() - 1) or 1
        )

        distance_matrix_mask = np.ma.array(distance_matrix, mask=False)

        di = (
            np.arange(distance_matrix.shape[0]),
            np.arange(
                distance_matrix.shape[1] - distance_matrix.shape[0],
                distance_matrix.shape[1],
            ),
        )

        distance_matrix_mask.mask[di] = True

        avg_dists = distance_matrix_mask.mean(axis=1)

        return avg_dists

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
