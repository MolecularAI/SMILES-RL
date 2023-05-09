import torch


def rewards_to_go(
    seqs: torch.Tensor, score: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Computes rewards-to-go (discounted rewards for each step) given sequence and score.
    Score is given at last step of sequence.


    Args:
        seqs (torch.Tensor): (batch_size, sequence_length) Sequences of token ids
        score (torch.Tensor): (batch_size,)score of each (full) sequence
        gamma (float): discount factor

    Returns:
        torch.Tensor: rewards-to-go [batch size, sequence length -1]
    """

    assert (
        torch.min(torch.amin(seqs, 1)) >= 0
    ), f"minmax token_id of sequence must be 0, but got {torch.min(torch.amin(seqs, 1))}"

    # Obtain idx of first zero element (=stopping token) in each batch
    first_zero_batch_idx = torch.argmin(seqs, 1)

    # Make sure that all sequences without stop token has score zero.
    for i_row, i_col in enumerate(first_zero_batch_idx):
        if seqs[i_row, i_col] != 0:
            score[i_row] = 0

    # Get idx for all stopping tokens
    all_zero_idx = (seqs[:, :-1] == 0).nonzero(as_tuple=True)

    # Create array of gammas to iterate over all batches simultaneously.
    gamma_array = gamma * torch.ones(seqs.size(0))

    # Initialize rewrds-to-go for all actions in batch of sequences
    batch_rtgs = torch.zeros(seqs[:, :-1].size())

    # reward-to-go at time t = gamma^{T-t}*r_{a_{1:T}},
    # where r_{a_{1:T}} is the episodic reward
    for i_col in range(seqs.size(1) - 1):
        rtgs = torch.pow(gamma_array, first_zero_batch_idx - 1 - i_col) * score
        batch_rtgs[:, i_col] = rtgs

    # Set zero reward-to-go for stop tokens
    batch_rtgs[all_zero_idx] = 0.0

    return batch_rtgs
