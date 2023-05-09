import numpy as np
import torch
from typing import Tuple, List
import pandas as pd
from .base_replay_buffer import BaseReplayBuffer

from .utils.chem import convert_to_canonical_smiles


class BinCurrent(BaseReplayBuffer):
    def __init__(self, config: dict) -> None:
        super(BinCurrent, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.k = config.get("k", 64)

        self._reset()

    def _reset(self):
        self.memory: pd.DataFrame = pd.DataFrame(
            columns=["smiles", "canon_smiles", "score", "bin"],
            dtype="float32",
        )

    def __call__(
        self, smiles: List[str], score: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Returns current elements plus k top and bot elements wrt score history,
            and puts new elements in memory

        Args:
            smiles (List[str]): list of all sampled SMILES strings
            score (np.array): score of each SMILES string (batch_size,)

        Returns:
            Tuple[List[str], np.array]: current SMILES and top and bot SMILES and scores wrt score
        """

        # Put current batch in buffer
        self._put(smiles, score)

        # Sample k SMILES from bins
        all_smiles, all_scores = self._sample()

        # Clear buffer
        self._reset()

        return all_smiles, all_scores

    def _sample(
        self,
    ) -> Tuple[List[str], np.ndarray]:
        """Samples k current SMILES and scores from bins [0,0.1],(0.1,0.2],...,(0.9,1].
        To the extend possible, tries to sample equal number of SMILES from each bin.

        Returns:
            Tuple[List[str], np.ndarray]: sampled SMILES strings and corresponding scores
        """

        # If we got less than requestd, return everything we got
        if len(self) <= self.k:
            smiles = self.memory["smiles"].to_list()
            scores = self.memory["score"].to_numpy()
            return smiles, scores

        unique_bins = pd.unique(self.memory["bin"])

        # Size for each bin
        k = self.k // len(unique_bins)

        # Make sure that size for each bin is not larger than the number of values in the bin
        ks = [
            min(len(self.memory[self.memory["bin"].eq(bin)]), k) for bin in unique_bins
        ]

        # Pick more items from larger bins to return the requested number of items in total
        i_k = 0
        while sum(ks) != self.k:
            ks[i_k] = min(
                len(self.memory[self.memory["bin"].eq(unique_bins[i_k])]), ks[i_k] + 1
            )

            i_k = (i_k + 1) % len(ks)

        sampled = pd.concat(
            [
                self.memory[self.memory["bin"].eq(bin)].sample(k)
                for bin, k in zip(unique_bins, ks)
            ]
        )

        smiles = sampled["smiles"].to_list()

        scores = sampled["score"].to_numpy()

        return smiles, scores

    def _put(self, smiles: List[str], score: np.ndarray) -> None:
        """Put SMILES and score in buffer and split them up in bins

        Args:
            smiles (List[str]): SMILES strings
            score (np.ndarray): score for each SMILES
        """

        bins = pd.IntervalIndex.from_tuples(
            [
                (-0.01, 0.1),
                (0.1, 0.2),
                (0.2, 0.3),
                (0.3, 0.4),
                (0.4, 0.5),
                (0.5, 0.6),
                (0.6, 0.7),
                (0.7, 0.8),
                (0.8, 0.9),
                (0.9, 1.0),
            ]
        )

        bins_memory = pd.cut(score, bins=bins)

        canon_smiles = convert_to_canonical_smiles(smiles)

        df = pd.DataFrame(
            {
                "smiles": smiles,
                "canon_smiles": canon_smiles,
                "score": score,
                "bin": bins_memory,
            }
        )
        self.memory = pd.concat([self.memory, df]).reset_index(drop=True)

    def __len__(self):
        return len(self.memory)
