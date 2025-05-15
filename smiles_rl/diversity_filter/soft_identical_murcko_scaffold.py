from copy import deepcopy

import numpy as np
from reinvent_scoring.scoring.score_summary import FinalSummary


from .utils.soft_penalty import (
    penalize_score_erf,
    penalize_score_linear,
    penalize_score_sigmoid,
    penalize_score_tanh,
)

from .chem import calculate_scaffold

from .base_diversity_filter import (
    BaseDiversityFilter,
)
from .diversity_filter_parameters import (
    DiversityFilterParameters,
)


class SoftIdenticalMurckoScaffold(BaseDiversityFilter):
    """Penalizes compounds based on exact Murcko Scaffolds previously generated."""

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

        return scores
