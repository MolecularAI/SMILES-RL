from reinvent_chemistry.utils import get_indices_of_unique_smiles
from typing import List, Tuple
import torch


def sample_unique_sequences(
    agent, batch_size: int
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    seqs, smiles, agent_likelihood = agent.sample(batch_size)
    unique_idxs = get_indices_of_unique_smiles(smiles)
    seqs_unique = seqs[unique_idxs]
    smiles_unique = [smiles[unique_idx] for unique_idx in unique_idxs]
    agent_likelihood_unique = agent_likelihood[unique_idxs]
    return seqs_unique, smiles_unique, agent_likelihood_unique
