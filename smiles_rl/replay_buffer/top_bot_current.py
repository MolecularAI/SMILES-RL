import numpy as np
from typing import Tuple, List
from .base_replay_buffer import BaseReplayBuffer


class TopBotCurrent(BaseReplayBuffer):
    def __init__(self, config: dict) -> None:
        super(TopBotCurrent, self).__init__()

        self.k = config.get("k", 64)

    def __call__(
        self, smiles: List[str], score: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Returns bot and top elements wrt score from current episode

        Args:
            smiles (List[str]): list of all sampled SMILES strings
            score (np.array): score of each SMILES string (batch_size,)

        Returns:
            Tuple[List[str], np.array]: top and bottom SMILES and scores wrt score
        """

        k_top = min((len(smiles) + 1) // 2, (self.k + 1) // 2)
        k_bot = min(len(smiles) // 2, self.k // 2)

        # Sort scores in ascending order of score
        ordered_indices = np.flip(np.argsort(score))

        # Get top indices wrt to score
        indices_top = ordered_indices[:k_top]

        # Get bot indices wrt to score
        indices_bot = ordered_indices[-k_bot:]

        # Concat bot and top indices
        indices = np.concatenate((indices_top, indices_bot), axis=None)

        return [smiles[i] for i in indices], score[indices]

    def __len__(self):
        return 0
