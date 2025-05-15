from copy import deepcopy

from typing import List

import numpy as np
from reinvent_scoring.scoring.score_summary import FinalSummary


from .utils.soft_penalty import (
    penalize_score_erf,
    penalize_score_linear,
    penalize_score_sigmoid,
    penalize_score_tanh,
)


from .base_diversity_filter import (
    BaseDiversityFilter,
)
from .diversity_filter_parameters import (
    DiversityFilterParameters,
)

from .chem import calculate_scaffold


class SoftInformation(BaseDiversityFilter):
    """Provides intrinsic rewards based on random network distillation."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

        soft_function = {
            "erf": penalize_score_erf,
            "linear": penalize_score_linear,
            "sigmoid": penalize_score_sigmoid,
            "tanh": penalize_score_tanh,
        }

        self._penalize_score_soft = soft_function[self.parameters.soft_function]

        print(f"Using soft function: {self._penalize_score_soft.__name__}", flush=True)

    def update_score(self, score_summary: FinalSummary, step=0) -> np.ndarray:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles
        hits_idx = []
        hits_smiles = []
        hits_scaffold = []
        penalties = []

        for i in score_summary.valid_idxs:
            smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])
            scores[i] = 0 if self._smiles_exists(smile) else scores[i]

            if scores[i] >= self.parameters.minscore:
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

                n_scaffold_instances = (
                    self._diversity_filter_memory.scaffold_instances_count(scaffold)
                )

                scores[i] = self._penalize_score_soft(
                    n_scaffold_instances, self.parameters.bucket_size, scores[i]
                )

                hits_scaffold.append(scaffold)
                hits_idx.append(i)
                hits_smiles.append(smiles[i])

        information = self._calculate_information(hits_smiles, hits_scaffold)

        scores[hits_idx] += information

        return scores

    def _calculate_information(
        self, smiles: List[str], scaffold: List[str]
    ) -> np.ndarray:
        if len(smiles) == 0:
            return np.array([])

        n_scaffolds = self.number_of_scaffold_in_memory()

        scaff_entropy = np.zeros(len(smiles))

        for i_smi, scaff in enumerate(scaffold):

            n_scaff = self._diversity_filter_memory.scaffold_instances_count(scaff)

            scaff_entropy[i_smi] = -np.log(n_scaff / (n_scaffolds + 1e-6))

        # Normalize informations gains

        if len(scaff_entropy) > 2:
            scaff_entropy = (scaff_entropy - np.amin(scaff_entropy)) / (
                np.amax(scaff_entropy) - np.amin(scaff_entropy) + 1e-6
            )

        return scaff_entropy
