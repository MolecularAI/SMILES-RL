from reinvent_chemistry.utils import get_indices_of_unique_smiles
from typing import List, Tuple, Union
import torch
from smiles_rl.model.actor_model import ActorModel
from smiles_rl.model.default_model import DefaultModel


def sample_unique_sequences(
    agent: Union[ActorModel, DefaultModel], batch_size: int
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Samples a set of sequences corresponding to unique non-cononical SMILES

    Args:
        agent (Union[ActorModel, DefaultModel]): Generative model (with agent policy)
        batch_size (int): number of non-unique sequences (SMILES) to sample

    Returns:
        Tuple[torch.Tensor, List[str], torch.Tensor]: unique sequences, smiles and agent likelihood sampled
    """

    seqs, smiles, agent_likelihood = agent.sample(batch_size)
    unique_idxs = get_indices_of_unique_smiles(smiles)
    seqs_unique = seqs[unique_idxs]
    smiles_unique = [smiles[unique_idx] for unique_idx in unique_idxs]
    agent_likelihood_unique = agent_likelihood[unique_idxs]
    return seqs_unique, smiles_unique, agent_likelihood_unique
