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


class Information(BaseDiversityFilter):
    """Don't penalize compounds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary: FinalSummary, step=0) -> np.ndarray:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles
        hits = []
        hits_idxs = []
        hits_scaffold = []

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

                hits.append(smile)

                hits_scaffold.append(scaffold)

                hits_idxs.append(i)

        n_scaffolds = self.number_of_scaffold_in_memory()

        scaff_entropy = np.zeros(len(hits))
        n_in_scaffold = np.zeros(len(hits))
        scaffold_ids = np.zeros(len(hits))

        for i_smi, rdkit_smi in enumerate(hits):

            # Invalid SMILES does not have score larger than 0
            scaff = hits_scaffold[i_smi]
            n_scaff = self._diversity_filter_memory.scaffold_instances_count(scaff)

            scaff_entropy[i_smi] = -np.log(n_scaff / (n_scaffolds + 1e-6))

            n_in_scaffold[i_smi] = n_scaff

            scaffold_id = self.get_scaffold_id(scaff)

            assert scaffold_id is not None

            scaffold_ids[i_smi] = scaffold_id

        # Normalize informations gains
        if len(scaff_entropy) > 2:
            scaff_entropy = (scaff_entropy - np.amin(scaff_entropy)) / (
                np.amax(scaff_entropy) - np.amin(scaff_entropy) + 1e-6
            )

        scores[hits_idxs] += scaff_entropy

        return scores
