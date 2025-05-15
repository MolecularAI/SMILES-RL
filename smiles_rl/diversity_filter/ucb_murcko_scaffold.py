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


class UCBMurckoScaffold(BaseDiversityFilter):
    """Penalizes compounds based on exact Murcko Scaffolds previously generated."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary: FinalSummary, step=0) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        scaffolds_to_update = []
        scores_idxs_to_update = []

        for i in score_summary.valid_idxs:
            smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])

            scores[i] = 0 if self._smiles_exists(smile) else scores[i]

            if scores[i] >= self.parameters.minscore:
                scaffold = calculate_scaffold(smile)

                scaffolds_to_update.append(scaffold)
                scores_idxs_to_update.append(i)

                self._add_to_memory(
                    i,
                    scores[i],
                    smile,
                    smiles[i],
                    scaffold,
                    score_summary.scaffold_log,
                    step,
                )

        # Includes count and reward from current round
        for i, scaffold in zip(scores_idxs_to_update, scaffolds_to_update):
            # Add 1 to step since step variable starts at 0

            scores[i] += self._kl_ucb(scaffold, step + 1)

        return scores

    def _kl_ucb(self, scaffold: str, step: int, c: float = 0.0):
        # NOTE: Using Bernoulli KL but other distribution
        step = step * 128

        scaffold_memory = self.get_scaffold(scaffold)

        n_in_scaffold = len(scaffold_memory)

        assert (
            n_in_scaffold > 0
        ), f"No molecules in scaffold {scaffold} but tries to calculate its UCB"

        sum_of_scores = scaffold_memory["total_score"].sum()

        mean_of_scores = sum_of_scores / n_in_scaffold

        assert mean_of_scores > 1 / 3, f"mean_of_scores is {mean_of_scores}"

        assert mean_of_scores < 1, f"mean_of_scores is {mean_of_scores}"

        mean_of_scores = np.clip(mean_of_scores, 1e-8, 1 - 1e-8)

        upper_bound = np.log(step) + c * np.log(np.log(step))

        q_0 = np.max([3 * mean_of_scores / 2 - 0.5, 1e-8])  # mean_of_scores

        q = q_0
        for _ in range(50):

            q_old = q
            f = n_in_scaffold * self._bernoulli_kl(mean_of_scores, q) + upper_bound
            f_prime = self._bernoulli_kl_prime(mean_of_scores, q, n_in_scaffold)
            q -= f / f_prime

            q = np.clip(q, mean_of_scores + 1e-8, 1 - 1e-8)

            if np.absolute(q - q_old) < 1e-8:
                break

        q_max = q

        assert q_max <= 1, f"q_max is {q_max} but should be <= 1"

        assert q_max >= 0, f"q_max is {q_max} but should be >= 0"

        return q_max

    @staticmethod
    def _bernoulli_kl(p: float, q: float) -> float:
        if q == 0 and p == 0:
            return 0

        elif (1 - p) == 0 and (1 - q) == 0:
            return 0

        elif q == 0 or (1 - q) == 0:
            return float("inf")

        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    @staticmethod
    def _bernoulli_kl_prime(p: float, q: float, n: int) -> float:
        if q >= 1:
            return 0.01

        elif q <= 0:
            return -0.01

        return n * ((1 - p) / (1 - q) - p / q)
