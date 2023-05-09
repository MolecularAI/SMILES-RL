from enum import unique
import numpy as np
import torch
from typing import Tuple, List, Union, Optional, Dict, Any
import pandas as pd
from .base_replay_buffer import BaseReplayBuffer

from .utils.chem import convert_to_canonical_smiles

from ..utils.general import to_tensor


class BinHistory(BaseReplayBuffer):
    def __init__(self, config: dict) -> None:
        super(BinHistory, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.k = config.get("k", 64)
        self.memory_size_per_bin = config.get("memory_size", 1000)

        self.step = 0

        self._reset()

    def _reset(self):
        self.memory: pd.DataFrame = pd.DataFrame(
            columns=[
                "smiles",
                "score",
                "canon_smiles",
                "bin",
                "entropy",
                "probability",
                "step",
            ],
            dtype="float32",
        )

    def __call__(
        self, smiles: List[str], score: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Returns current elements plus k top and bot elements wrt score history,
            and puts new elements in memory

        Args:
            smiles (List[str]): SMILES for each sequence in batch (batch_size,)
            score (np.array): score of each sequence in batch (batch_size,)

        Returns:
            Tuple[List[str], np.array]: current elements and top and bot SMILES and scores wrt score
        """

        # Sample SMILES and scores bins in buffer
        bin_smiles, bin_score = self._sample()

        all_smiles = smiles + bin_smiles

        all_scores = np.concatenate((score, bin_score), axis=None)

        # Put current SMILES and scores in buffer
        self._put(smiles, score)

        return all_smiles, all_scores

    def _sample(
        self,
    ) -> Tuple[List[str], np.ndarray]:
        """Samples k SMILES and scores from bins [0,0.1],(0.1,0.2],...,(0.9,1] in replay buffer.
        To the extend possible, tries to sample equal number of SMILES from each bin.

        Returns:
            Tuple[List[str], np.ndarray]: sampled SMILES strings and corresponding scores
        """

        if len(self) < self.k:
            return [], np.array([])

        unique_bins = pd.unique(self.memory["bin"])

        k = self.k // len(unique_bins)

        ks = [
            min(len(self.memory[self.memory["bin"].eq(bin)]), k) for bin in unique_bins
        ]

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
                "step": np.full(len(smiles), self.step),
            }
        )
        self.memory = pd.concat([self.memory, df]).reset_index(drop=True)

        self._purge_memory()

        self.step += 1

    def _purge_memory(self):
        """Purges the memory of each bin by keeping the latest molecules in each bin"""

        # Sort wrt when tey were added to memory. Latest at the top.
        self.memory.sort_values("step", ascending=False, inplace=True)

        # Keep latest trajectories in memory
        truncated_df = self.memory.groupby("bin").head(self.memory_size_per_bin)

        # Sort values wrt score.
        self.memory = truncated_df.sort_values(
            "score", ascending=False, kind="stable", inplace=False
        ).reset_index(drop=True)

    def __len__(self):
        return len(self.memory)

    def put_probs(
        self,
        smiles: List[str],
        score: np.ndarray,
        probs: np.ndarray,
        seqs: torch.Tensor,
    ) -> None:
        """Put SMILES, scores and probabilities in buffer. Used by ACER agent.

        Args:
            smiles (List[str]): SMILES strings
            score (np.ndarray): Score of each SMILES [batch_size, ] 
            probs (np.ndarray): Probability for each action at each state in sequence [batch_size, sequence_length, n_actions]
            seqs (torch.Tensor): Sequence. Used for determing how sequence lenght of probability array for each sequence. [batch_size, seq_len]
        """

        bins = pd.IntervalIndex.from_tuples(
            [
                (-1.01, -0.01),
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


        # Sequence lenghts for all smiles including start token but excluding stop token
        # If no stop token, sequence length is total lenght of sequence
        seq_length = [
            torch.argmin(seqs[i]).item()
            if (torch.min(seqs[i]) == 0).item()
            else len(seqs[i])
            for i in range(seqs.size(0))
        ]

        # List of probability arrays to save in memory
        probs = [
            probs[
                i,
                :l,
            ]
            for i, l in zip(range(len(probs)), seq_length)
        ]

        # Canonical SMILES to remove duplicates of molecules
        canon_smiles = convert_to_canonical_smiles(smiles)

        df = pd.DataFrame(
            {
                "smiles": smiles,
                "canon_smiles": canon_smiles,
                "score": score,
                "probability": probs,
                "bin": bins_memory,
                "step": np.full(len(smiles), self.step),
            }
        )

        self.memory = pd.concat([self.memory, df]).reset_index(drop=True)

        self._purge_memory_step()

        self.step += 1

    def sample_probs(
        self, batch_size: int
    ) -> Tuple[List[str], np.ndarray, torch.Tensor]:
        """Sample SMILES, scores and probabilities from bins [-1,0),[0,0.1],(0.1,0.2],...,(0.9,1] in replay buffer.
        To the extend possible, tries to sample equal number of SMILES from each bin.

        Args:
            batch_size (int): batch size of sample

        Returns:
            Tuple[List[str], np.ndarray, torch.Tensor]: sampled bacth of SMILES, scores and probabilities
        """
        

        batch_size = min(len(self), batch_size)

        unique_bins = pd.unique(self.memory["bin"])

        k = batch_size // len(unique_bins)

        ks = [
            min(len(self.memory[self.memory["bin"].eq(bin)]), k) for bin in unique_bins
        ]

        i_k = 0
        while sum(ks) != batch_size:
            ks[i_k] = min(
                len(self.memory[self.memory["bin"].eq(unique_bins[i_k])]), ks[i_k] + 1
            )

            i_k = (i_k + 1) % len(ks)

        sampled = pd.concat(
            [
                self.memory[self.memory["bin"].eq(bin)].sample(k, replace=False)
                for bin, k in zip(unique_bins, ks)
            ]
        )

        smiles = sampled["smiles"].to_list()

        scores = sampled["score"].to_numpy()
        probs = sampled["probability"]

        # Pad to make sure probability sequence so that they are of the same lenght and store them as a torch.Tensor
        probs = torch.nn.utils.rnn.pad_sequence(
            [to_tensor(arr) for _, arr in probs.items()],
            padding_value=1e-8,
            batch_first=True,
        )

        return smiles, scores, probs

    # TODO: Create seperate replay buffer for following code
    def sample(self) -> Tuple[List[str], np.ndarray]:
        """Samples SMILES and their corresponding score from bins [-1,0),[0,0.1],(0.1,0.2],...,(0.9,1] in replay buffer.
        Used by SAC.

        Returns:
            Tuple[List[str], np.ndarray]: Sampled SMILES strings and corresponding scores
        """
        
        
        batch_size = min(len(self), self.k)

        unique_bins = pd.unique(self.memory["bin"])

        k = batch_size // len(unique_bins)

        ks = [
            min(len(self.memory[self.memory["bin"].eq(bin)]), k) for bin in unique_bins
        ]

        i_k = 0
        while sum(ks) != batch_size:
            ks[i_k] = min(
                len(self.memory[self.memory["bin"].eq(unique_bins[i_k])]), ks[i_k] + 1
            )

            i_k = (i_k + 1) % len(ks)

        sampled = pd.concat(
            [
                self.memory[self.memory["bin"].eq(bin)].sample(k, replace=False)
                for bin, k in zip(unique_bins, ks)
            ]
        )

        smiles = sampled["smiles"].to_list()

        scores = sampled["score"].to_numpy()

        return smiles, scores

    def put(
        self,
        smiles: List[str],
        score: np.ndarray,
    ):
        """Split SMILES and their corresponding score into bins [-1,0),[0,0.1],(0.1,0.2],...,(0.9,1]
        and put into memory.

        Args:
            smiles (List[str]): SMILES to put in memory [batch_size]
            score (np.ndarray): Score for each SMILES to store [batch_size, ]
        """
        bins = pd.IntervalIndex.from_tuples(
            [
                (-1.01, -0.01),
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
                "step": np.full(len(smiles), self.step),
            }
        )

        self.memory = pd.concat([self.memory, df]).reset_index(drop=True)

        self._purge_memory_step()

        self.step += 1

    def _purge_memory_step(self):
        """Purge memory of each bin, keeping the latest molecules.
        """

        self.memory.sort_values("step", ascending=False, inplace=True)

        # Keep latest trajectories in memory
        truncated_df = self.memory.groupby("bin").head(self.memory_size_per_bin)

        self.memory = truncated_df.sort_values(
            "score", ascending=False, kind="stable", inplace=False
        ).reset_index(drop=True)
