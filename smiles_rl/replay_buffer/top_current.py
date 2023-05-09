import numpy as np
from typing import Tuple, List
from .base_replay_buffer import BaseReplayBuffer


class TopCurrent(BaseReplayBuffer):
    def __init__(self, config: dict) -> None:
        super(TopCurrent, self).__init__()

        self.k = config.get("k", 64)

    def __call__(
        self, smiles: List[str], score: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Returns top elements wrt score from current iteration

        Args:
            smiles (List[str]): list of all sampled SMILES strings
            score (np.array): score of each SMILES string (batch_size,)

        Returns:
            Tuple[List[str], np.array]: top SMILES and scores wrt score
        """

        # Handle case if k is larger than number of elements in current iteration
        k = min(len(smiles), (self.k))

        # Sort scores in ascending order of score
        # Flip to get descending order
        ordered_indices = np.flip(np.argsort(score))

        # Get top indices wrt to score
        indices = ordered_indices[:k]

        return [smiles[i] for i in indices], score[indices]

    def __len__(self):
        return 0
