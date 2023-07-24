import numpy as np
import torch
from typing import Tuple, List
import pandas as pd
from .base_replay_buffer import BaseReplayBuffer

from .utils.chem import convert_to_canonical_smiles

from ..utils.general import to_tensor


class TopHistory(BaseReplayBuffer):
    def __init__(self, config: dict) -> None:
        super(TopHistory, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.k = config.get("k", 64)
        self.memory_size = config.get("memory_size", 1000)

        self.step = 0

        self._reset()

    def _reset(self):
        self.memory: pd.DataFrame = pd.DataFrame(
            columns=["smiles", "canon_smiles", "score", "likelihood", "entropy"],
            dtype="float32",
        )

    def __call__(
        self, smiles: List[str], score: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Returns current elements plus k top elements wrt score history,
            and puts new elements in memory.add()

            Requires the memory to be sorted in descending order.

        Args:
            smiles (List[str]): list of all sampled SMILES strings
            score (np.array): score of each SMILES string (batch_size,)

        Returns:
            Tuple[List[str], np.array]: current elements and top SMILES and scores wrt score
        """

        k = min(len(self), self.k)

        indices = [x for x in range(k)]

        top_smiles, top_score = self._sample(indices)

        all_smiles = (
            smiles + top_smiles
        )  # torch.nn.rnn.pad_sequence((seqs,top_bot_seqs), batch_first=True)

        all_scores = np.concatenate((score, top_score), axis=None)

        self._put(smiles, score)

        return all_smiles, all_scores

    def _sample(self, indices: List[int]) -> Tuple[List[str], np.ndarray]:
        """Returns the batch of SMILES and corresponding score for the provided indices of the (sorted) memory


        Args:
            indices (List[int]): Indices for elements in memory to return

        Returns:
            Tuple[List[str], np.ndarray]: SMILES and scores of indices
        """

        sampled = self.memory.iloc[indices]

        smiles = sampled["smiles"].to_list()

        scores = sampled["score"].to_numpy()

        return smiles, scores

    def _put(self, smiles: List[str], score: np.ndarray) -> None:
        """Stores given SMILES and scores in memory.
        Memory is purged to remove duplicates and to just keep the top-k items.

        Args:
            smiles (List[str]): SMILES strings
            score (np.ndarray): score of each SMILES string [batch_size, ]
        """

        canon_smiles = convert_to_canonical_smiles(smiles)
        df = pd.DataFrame(
            {
                "smiles": smiles,
                "canon_smiles": canon_smiles,
                "score": score,
            }
        )
        self.memory = pd.concat([self.memory, df]).reset_index(drop=True)

        self._purge_memory()

    def _purge_memory(self):
        """Purge memory by removing duplciates (keeping the duplicate with the latest score)
        and then keeps the k top scoring molecules"""

        sorted_df = self.memory.sort_values("score", ascending=False, kind="stable")

        # Keep the duplicate with the lowest score
        unique_df = sorted_df.drop_duplicates(subset=["canon_smiles"], keep="last")

        self.memory = unique_df.head(self.k).reset_index(drop=True)

    def __len__(self):
        """Return size of memory"""
        return len(self.memory)

    def put_probs(
        self,
        smiles: List[str],
        score: np.ndarray,
        probs: np.ndarray,
        seqs: torch.Tensor,
    ) -> None:
        """Put SMILES, and corresponding scores and probabilities in memory.
        Uses sequences to determine the lenght of the probability array to save.
        It consists of a memory of the top-scoring SMILES (no duplicates).

        Used by ACER.

        Args:
            smiles (List[str]): SMILES strings to store [batch_size,]
            score (np.ndarray): Score for each SMILES [batch_size, ]
            probs (np.ndarray): action probabilities at each state of sequence [batch_size, sequence_length, n_actions]
            seqs (torch.Tensor): Sequences of action [batch_size, sequence_length]
        """

        # Sequence lenghts for all smiles including start token but excluding stop token
        # If no stop token, sequence length is total lenght of sequence
        seq_length = [
            torch.argmin(seqs[i]).item()
            if (torch.min(seqs[i]) == 0).item()
            else len(seqs[i])
            for i in range(seqs.size(0))
        ]

        # Probability for each sequence
        probs = [
            probs[
                i,
                :l,
            ]
            for i, l in zip(range(len(probs)), seq_length)
        ]

        # Canonical SMILES to remove duplicates
        canon_smiles = convert_to_canonical_smiles(smiles)

        df = pd.DataFrame(
            {
                "smiles": smiles,
                "canon_smiles": canon_smiles,
                "score": score,
                "probability": probs,
                "step": np.full(len(smiles), self.step),
            }
        )

        self.memory = pd.concat([self.memory, df]).reset_index(drop=True)

        self._purge_memory_off_policy()

        self.step += 1

    def sample_probs(
        self, batch_size: int
    ) -> Tuple[List[str], np.ndarray, torch.Tensor]:
        """Randomly samples batch from memory. Used by ACER

        Args:
            batch_size (int): number of SMILES strings to sample in total

        Returns:
            Tuple[List[str], np.ndarray, torch.Tensor]: Batch of SMILES,scores and probabilities
        """

        sample_size = min(len(self), batch_size)

        sampled = self.memory.sample(sample_size, replace=False)

        smiles = sampled["smiles"].to_list()

        scores = sampled["score"].to_numpy()

        probs = sampled["probability"]

        probs = torch.nn.utils.rnn.pad_sequence(
            [to_tensor(arr) for _, arr in probs.items()],
            padding_value=1e-8,  # NOTE: was 1
            batch_first=True,
        )

        return smiles, scores, probs

    def _purge_memory_off_policy(self):
        """Purges memory. Used by the off-policy algorithms."""

        sorted_df = self.memory.sort_values("score", ascending=False, kind="stable")

        # Keep the duplicate with the lowest score
        unique_df = sorted_df.drop_duplicates(subset=["canon_smiles"], keep="last")

        self.memory = unique_df.head(self.memory_size).reset_index(drop=True)

    def sample(self) -> Tuple[List[str], np.ndarray]:
        """Randomly samples batch of items from memory. Batch size = k.

        Returns:
            Tuple[List[str], np.ndarray]: k SMILES strings and corresponding scores
        """

        sample_size = min(len(self), self.k)

        sampled = self.memory.sample(sample_size)

        smiles = sampled["smiles"].to_list()

        scores = sampled["score"].to_numpy()

        return smiles, scores

    def put(
        self,
        smiles: List[str],
        score: np.ndarray,
    ):
        """Puts given SMILES and scores. Purges memory to remove duplictaes,
        and not letting the memory exceed memory size

        Args:
            smiles (List[str]): SMILES strings to store
            score (np.ndarray): Score for each SMILES string [batch_size, ]
        """

        canon_smiles = convert_to_canonical_smiles(smiles)

        df = pd.DataFrame(
            {
                "smiles": smiles,
                "canon_smiles": canon_smiles,
                "score": score,
            }
        )

        self.memory = pd.concat([self.memory, df]).reset_index(drop=True)

        self._purge_memory_off_policy()
