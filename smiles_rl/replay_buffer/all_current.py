import numpy as np
from typing import Tuple, List
from .base_replay_buffer import BaseReplayBuffer


class AllCurrent(BaseReplayBuffer):
    def __init__(self, config: dict) -> None:
        super(AllCurrent, self).__init__()

    def __call__(
        self, smiles: List[str], score: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Returns all smiles and score

        Args:
            smiles (List[str]): list of all sampled SMILES strings
            score (np.array): score of each SMILES string (batch_size,)

        Returns:
            Tuple[List[str], np.array]: all SMILES strings and scores
        """

        # return all smiles and scores
        return smiles, score

    def __len__(self):
        return 0
