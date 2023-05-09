import numpy as np
import torch
from typing import Tuple, List
import pandas as pd
from .base_replay_buffer import BaseReplayBuffer

from .utils.chem import convert_to_canonical_smiles

from ..utils.general import to_tensor


class TopBotHistory(BaseReplayBuffer):
    def __init__(self, config: dict) -> None:
        super(TopBotHistory, self).__init__()

        self.k = config.get("k", 64)
        self.memory_size = config.get("memory_size", 1000)

        self.step = 0

        self._reset()

    def _reset(self):
        self.memory: pd.DataFrame = pd.DataFrame(
            columns=[
                "smiles",
                "canon_smiles",
                "score",
                "probability",
                "entropy",
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
            smiles (List[str]): list of all sampled SMILES strings
            score (np.array): score of each SMILES string (batch_size,)

        Returns:
            Tuple[List[str], np.array]: current elements and top and bottom SMILES and scores wrt score
        """

        k_top = min((len(self) + 1) // 2, (self.k + 1) // 2)
        k_bot = min(len(self) // 2, self.k // 2)

        indices = [x for x in range(k_top)] + [
            x for x in range(len(self) - k_bot, len(self))
        ]

        # Get the highest and lowest scoring molecules in memory from index
        # Requires that the memory is sorted in descending order
        top_bot_smiles, top_bot_score = self._sample(indices)

        all_smiles = smiles + top_bot_smiles

        all_scores = np.concatenate((score, top_bot_score), axis=None)

        self._put(smiles, score)

        return all_smiles, all_scores

    def _sample(self, indices: List[int]) -> Tuple[List[str], np.ndarray]:
        """Get items from memory by index.

        Args:
            indices (List[int]): Indices of items to acquire

        Returns:
            Tuple[List[str], np.ndarray]: acquired items
        """

        sampled = self.memory.iloc[indices]

        smiles = sampled["smiles"].to_list()

        scores = sampled["score"].to_numpy()

        return smiles, scores

    def _put(self, smiles: List[str], score: np.ndarray) -> None:
        """Put SMILES and their corresponding score in memory.
        Only saves them if they scores higher than the highest scoring
        items in the memory, or scores lower than the lowest scoring items in memory

        Args:
            smiles (List[str]): SMILES to put in memory
            score (np.ndarray): score of each SMILES to put in memory
        """

        canon_smiles = convert_to_canonical_smiles(smiles)

        df = pd.DataFrame(
            {
                "smiles": smiles,
                "canon_smiles": canon_smiles,
                "score": score,
                "step": np.full(len(smiles), self.step),
            }
        )
        self.memory = pd.concat([self.memory, df]).reset_index(drop=True)

        self._purge_memory()

        self.step += 1

    def _purge_memory(self):
        """Purge memory by only keeping the k/2 highest scoring and k/2 lowest scoring SMILES in memory

        Only keeps lowest scoring duplicate (based on canonical SMILES)

        If SMILES with the same score, keeps the latest ones in memory.
        """

        sorted_df = self.memory.sort_values(
            ["score", "step"], ascending=[False, True], kind="stable"
        )

        # Keep the duplicate with the lowest score and newest
        unique_df = sorted_df.drop_duplicates(subset=["canon_smiles"], keep="last")

        k = min(len(unique_df) // 2, self.k // 2)

        # Keep lowest and newest
        bot_df = unique_df.tail(k)

        unique_df = unique_df.sort_values(
            ["score", "step"], ascending=False, kind="stable"
        )

        k = min((len(unique_df) + 1) // 2, (self.k + 1) // 2)

        # Keep highest scoring and newest
        top_df = unique_df.head(k)

        self.memory = pd.concat([top_df, bot_df]).reset_index(drop=True)

    def __len__(self) -> int:
        """Returns size of buffer

        Returns:
            int: size of buffer memory
        """
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
        It consists of one part of top scoring SMILES and one part of valid bottom scoring SMILES,
        and one part of invalid SMILES (reward/score = -1).
        Memory for both parts are merged.
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
            }
        )

        self.memory = pd.concat([self.memory, df]).reset_index(drop=True)

        self._purge_memory_off_policy()

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
            padding_value=1e-8,
            batch_first=True,
        )

        return smiles, scores, probs


    # TODO: following code can be merged into seperate replay buffer class.
    def sample(self) -> Tuple[List[str], np.ndarray]:
        """Samples k SMILES strings and corresponding scores from memory. 
        Used by SAC.
        

        Returns:
            Tuple[List[str], np.ndarray]: sampled SMILES and scores
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
        """Put batch of SMILES strings and corresponding scores in memory

        Args:
            smiles (List[str]): batch of SMILES strings
            score (np.ndarray): batch of score [batch_size, ]
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

    def _purge_memory_off_policy(self):
        """Purges memory by first removing duplicate (keeping those with lowest score),
        then splits the memory into three seperate parts: high-scoring valid SMILES, low-scoring valid SMILES
        and invalid SMILES (reward/score = -1). Memory size for each part is equal, trying to keep an equal number in memory.
        
        Used by ACER and SAC.
        """
        

        sorted_df = self.memory.sort_values("score", ascending=False, kind="stable")

        # Keep the duplicate with the lowest score
        unique_df = sorted_df.drop_duplicates(subset=["canon_smiles"], keep="last")

        memory_size = min(3 * self.memory_size, len(unique_df))

        # invalid SMILES are scored -1 in SAC
        invalid_df = unique_df[unique_df["score"] < 0]

        valid_df = unique_df[unique_df["score"] >= 0]

        memory_size_valid = min(2 * memory_size // 3, len(valid_df))

        memory_size_invalid = min(memory_size // 3, len(invalid_df))

        self.memory = pd.concat(
            [
                valid_df.head(memory_size_valid // 2),
                valid_df.tail(memory_size_valid // 2),
                invalid_df.tail(memory_size_invalid),
            ]
        ).reset_index(drop=True)
