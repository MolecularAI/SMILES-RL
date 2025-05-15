import numpy as np


from scipy.special import erf
import math


def penalize_score_erf(
    n_scaffold_instances: int, bucket_size: int, score: float
) -> float:
    """Penalize extrinsic reward using error function.

    Args:
        n_scaffold_instances (int): Number of SMILES in memory with same scaffold
        bucket_size (int): Bucket size M (maximum desired number of SMILES with same scaffold)
        score (float): Extrinsic reward of SMILES to penalize

    Returns:
        float: Penalized extrinsic reward (score)
    """

    return (
        1
        + erf(math.sqrt(math.pi) / bucket_size)
        - erf(math.sqrt(math.pi) / bucket_size * n_scaffold_instances)
    ) * score


def penalize_score_linear(
    n_scaffold_instances: int, bucket_size: int, score: float
) -> float:
    """Penalize extrinsic reward using linear function.

    Args:
        n_scaffold_instances (int): Number of SMILES in memory with same scaffold
        bucket_size (int): Bucket size M (maximum desired number of SMILES with same scaffold)
        score (float): Extrinsic reward of SMILES to penalize

    Returns:
        float: Penalized extrinsic reward (score)
    """

    penalty = 1 - n_scaffold_instances / bucket_size

    return np.maximum(0, score * penalty)


def penalize_score_sigmoid(
    n_scaffold_instances: int, bucket_size: int, score: float
) -> float:
    """Penalize extrinsic reward using sigmoid function.

    Args:
        n_scaffold_instances (int): Number of SMILES in memory with same scaffold
        bucket_size (int): Bucket size M (maximum desired number of SMILES with same scaffold)
        score (float): Extrinsic reward of SMILES to penalize

    Returns:
        float: Penalized extrinsic reward (score)
    """

    exponent = n_scaffold_instances / bucket_size
    exponent *= 2
    exponent -= 1
    exponent /= 0.15
    exponent *= -1

    sigmoid = 1 / (1 + np.exp(exponent))

    penalty = 1 - sigmoid

    return score * penalty


def penalize_score_tanh(
    n_scaffold_instances: int, bucket_size: int, score: float, k: float = 3
) -> float:
    """Penalize extrinsic reward using hyperbolic tangent function.

    Args:
        n_scaffold_instances (int): Number of SMILES in memory with same scaffold
        bucket_size (int): Bucket size M (maximum desired number of SMILES with same scaffold)
        score (float): Extrinsic reward of SMILES to penalize
        k (float): Scaling of term in Tanh function

    Returns:
        float: Penalized extrinsic reward (score)
    """

    term = n_scaffold_instances - 1

    term /= bucket_size

    term *= k

    penalty = 1 - np.tanh(term)

    return score * penalty
