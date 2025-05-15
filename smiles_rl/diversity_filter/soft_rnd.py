from copy import deepcopy

from typing import List

import numpy as np
import torch
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

from ..model.default_model import DefaultModel

from ..model.smiles_tokenizer import SMILESTokenizer

from ..model.vocabulary import Vocabulary

from .chem import calculate_scaffold


class SoftRND(BaseDiversityFilter):
    """Provides intrinsic rewards based on random network distillation."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

        tokens = {
            "$": 0,
            "^": 1,
            "#": 2,
            "%10": 3,
            "(": 4,
            ")": 5,
            "-": 6,
            "1": 7,
            "2": 8,
            "3": 9,
            "4": 10,
            "5": 11,
            "6": 12,
            "7": 13,
            "8": 14,
            "9": 15,
            "=": 16,
            "Br": 17,
            "C": 18,
            "Cl": 19,
            "F": 20,
            "N": 21,
            "O": 22,
            "S": 23,
            "[N+]": 24,
            "[N-]": 25,
            "[O-]": 26,
            "[S+]": 27,
            "[n+]": 28,
            "[nH]": 29,
            "c": 30,
            "n": 31,
            "o": 32,
            "s": 33,
        }

        voc = Vocabulary(tokens=tokens)

        tokenizer = SMILESTokenizer()

        max_sequence_lenght = 256

        network_params = {
            "dropout": 0.0,
            "layer_size": 512,
            "num_layers": 3,
            "cell_type": "lstm",
            "embedding_layer_size": 256,
        }

        no_cuda = False

        self._target_network = DefaultModel(
            voc, tokenizer, network_params, max_sequence_lenght, no_cuda
        )
        self._prediction_network = DefaultModel.load_from_file(
            file_path="pre_trained_models/ChEMBL/random.prior.new", sampling_mode=False
        )

        self._optimizer = torch.optim.Adam(
            self._prediction_network.get_network_parameters(), lr=1e-4
        )

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
        novelty_idx = []
        novelty_smiles = []
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

                novelty_idx.append(i)
                novelty_smiles.append(smiles[i])

        novelty = self._calculate_novelty(novelty_smiles)

        scores[novelty_idx] += novelty

        return scores

    def _calculate_novelty(self, smiles: List[str]) -> np.ndarray:
        if len(smiles) == 0:
            return np.array([])

        with torch.no_grad():
            target_likelihoods = self._target_network.likelihood_smiles(smiles)
        prediction_likelihoods = self._prediction_network.likelihood_smiles(smiles)

        loss = torch.pow(prediction_likelihoods - target_likelihoods, 2)

        novelty = loss.detach().cpu().numpy()

        # Backward propagate loss and perform one optimization step
        loss = loss.mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        novelty = (novelty - np.amin(novelty)) / (
            np.amax(novelty) - np.amin(novelty) + 1e-6
        )

        return novelty
