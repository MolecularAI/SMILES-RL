from copy import deepcopy

import numpy as np
from reinvent_scoring.scoring.score_summary import FinalSummary

from .base_diversity_filter import (
    BaseDiversityFilter,
)
from .diversity_filter_parameters import (
    DiversityFilterParameters,
)


from typing import List


import multiprocessing as mp


from .simgraph import compute_distance_matrix_from_fps

from .chem import morgan_from_smiles, morgan_from_smiles_list, calculate_scaffold


class MeanSimilarityRandom(BaseDiversityFilter):
    """Penalizes compounds based on atom pair Tanimoto similarity to previously generated Murcko Scaffolds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)
        self._diverse_hits_fingerprints = {}

    def update_score(self, score_summary: FinalSummary, step=0) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        hits_idxs = []

        reference_smiles = self.sample_smiles(5000)

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

        novelty = self.compute_novelty_score(hits, reference_smiles)

        scores[hits_idxs] += novelty

        return scores

    def compute_novelty_score(
        self,
        smiles: List[str],
        smiles_reference: List[str],
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

        fps = [morgan_from_smiles(smiles=s, radius=radius, nbits=nbits) for s in smiles]

        if len(smiles_reference) > 0:
            fps_reference = morgan_from_smiles_list(
                smiles=smiles_reference,
                radius=radius,
                nbits=nbits,
                n_jobs=(mp.cpu_count() - 1) or 1,
            )
        elif len(smiles) == 1:
            return np.zeros(1)
        else:
            fps_reference = []

        # gets train actives and computes which of the solutions are novel
        # the novel_idx is a list of indices of the novel solutions
        # we need them to map the indices of the diverse solutions to the original indices
        distance_matrix = compute_distance_matrix_from_fps(
            fps, fps_reference + fps, n_jobs=(mp.cpu_count() - 1) or 1
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
