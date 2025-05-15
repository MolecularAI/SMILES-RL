from copy import deepcopy

import numpy as np
from reinvent_scoring.scoring.score_summary import FinalSummary

from .base_diversity_filter import (
    BaseDiversityFilter,
)
from .diversity_filter_parameters import (
    DiversityFilterParameters,
)

from .chem import calculate_scaffold


class IdenticalMurckoScaffold(BaseDiversityFilter):
    """Penalizes compounds based on exact Murcko Scaffolds previously generated."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary: FinalSummary, step=0) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        for i in score_summary.valid_idxs:
            smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])
            scaffold = calculate_scaffold(smile)
            scores[i] = 0 if self._smiles_exists(smile) else scores[i]

            if scores[i] >= self.parameters.minscore:
                self._add_to_memory(
                    i,
                    scores[i],
                    smile,
                    smiles[i],
                    scaffold,
                    score_summary.scaffold_log,
                    step,
                )
                scores[i] = self._penalize_score(scaffold, scores[i])

        return scores
