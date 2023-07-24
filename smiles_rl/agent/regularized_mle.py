import numpy as np

from copy import deepcopy


from .utils.sample import sample_unique_sequences

from ..configuration_envelope import ConfigurationEnvelope

from ..model.default_model import DefaultModel
from .base_agent import BaseAgent


from reinvent_scoring import FinalSummary

import torch

from reinvent_chemistry.logging import fraction_valid_smiles


from ..utils.general import to_tensor

from .utils.margin_guard import MarginGuard

import time

from typing import Optional, List


class RegularizedMLE(BaseAgent):
    def __init__(
        self,
        config: ConfigurationEnvelope,
        scoring_function,
        diversity_filter,
        replay_buffer,
        logger,
    ) -> None:
        self._config = config
        self._logger = logger

        self.config = config.reinforcement_learning.parameters

        # NOTE: required to be defined at self.config since MarginGuard() uses this value
        self.config.sigma = self.config.specific_parameters.get("sigma", 128)

        # NOTE: required to be defined at self.config since MarginGuard() uses this value
        self.config.margin_threshold = self.config.specific_parameters.get(
            "margin_threshold", 50
        )

        # Initialize prior. Will not be changed, so we initialize it here once
        self._prior = DefaultModel.load_from_file(
            file_path=self.config.agent, sampling_mode=True
        )

        # Disable prior parameters
        self._disable_prior_gradients()

        # Initialize agent and optimizer
        self.reset()

        self._margin_guard = MarginGuard(self)
        self._diversity_filter = diversity_filter
        self._scoring_function = scoring_function
        self._replay_buffer = replay_buffer

        self.step = 0

        self.start_time = time.time()

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        for param in self._prior.get_network_parameters():
            param.requires_grad = False

    def update(self, smiles: List[str]):
        """Update policy using the augmented likelihood suggested in https://doi.org/10.1186/s13321-017-0235-x
            Stripped down version of https://github.com/MolecularAI/Reinvent


        Args:
            smiles (List[str]): SMILES strings for update of policy
        """

        # Score SMILES strings using given scoring function
        try:
            score_summary = self._scoring_function.get_final_score_for_step(
                smiles, self.step
            )
        except TypeError as inst:
            # If no valid SMILES string was generated
            print(inst, flush=True)
            score_summary = FinalSummary(
                np.zeros((len(smiles),), dtype=np.float32), smiles, [], []
            )

        score_summary = deepcopy(score_summary)

        # Update scores using given diversity filter
        score = self._diversity_filter.update_score(score_summary, self.step)

        # Calculate policy entropy for sampled sequences
        with torch.no_grad():
            seqs = self._agent.smiles_to_sequences(score_summary.scored_smiles)
            log_probs, probs = self._agent.log_and_probabilities(seqs)
            policy_entropy = -(probs * log_probs).sum(dim=2).mean()

        # Use replay buffer (on-policy) to sample SMILES strings and corresponding rewards
        sample_smiles, sample_score = self._replay_buffer(
            score_summary.scored_smiles, score
        )

        # switch signs
        agent_likelihood = -self._agent.likelihood_smiles(sample_smiles)
        prior_likelihood = -self._prior.likelihood_smiles(sample_smiles)

        # Calculate augmented likelihood, which is used as the target for the agent
        augmented_likelihood = prior_likelihood + self.config.sigma * to_tensor(
            sample_score
        )

        # Agent loss
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Bavkward propogate loss and perform one optimization step
        loss = loss.mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Report stats and checkpoint
        self._stats_and_chekpoint(
            loss,
            score,
            self.start_time,
            self.step,
            self.smiles,
            score_summary,
            agent_likelihood,
            prior_likelihood,
            augmented_likelihood,
            policy_entropy,
        )

        # Intermediate saving of agent model and memory (for inspection)
        if self.step % 500 == 0:
            self._logger.save_intermediate_state(self._agent, self._diversity_filter)

        self.step += 1

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        """Generate sequences and return corresponding SMILES strings

        Args:
            batch_size (int): number of (non-unqiue) sequences to generate

        Returns:
            List[str]: SMILES strings corresponding to unique sequences.
        """

        self.seqs, self.smiles, self.agent_likelihood = sample_unique_sequences(
            self._agent, batch_size
        )

        return deepcopy(self.smiles)

    def log_out(self):
        """Saves final state of agent model and memory (for final inspection)"""
        self._logger.save_final_state(self._agent, self._diversity_filter)

    def _stats_and_chekpoint(
        self,
        loss: float,
        score: np.ndarray,
        start_time: float,
        step: int,
        smiles: List[str],
        score_summary: FinalSummary,
        agent_likelihood: float,
        prior_likelihood: float,
        augmented_likelihood: float,
        policy_entropy: float,
    ):
        """Creates timestep report messages and uses given logger to print it

        Args:
            loss (float): augmented loss
            score (np.ndarray): score for each generated molecule
            start_time (float): Time when instance of agent was initialized
            step (int): current step
            smiles (List[str]): Generated SMILES strings
            score_summary (FinalSummary): summary of score from scoring function
            agent_likelihood (float): agent likelihood
            prior_likelihood (float): prior likelihood
            augmented_likelihood (float): augmented likelihood = prior_likelihood + sigma*score
            policy_entropy (float): policy entropy for current samples
        """

        # Use MarginGuard for updating sigma parameter and/or reseting agent if necessary
        self._margin_guard.adjust_margin(step)

        self._margin_guard.store_run_stats(
            agent_likelihood, prior_likelihood, augmented_likelihood, score
        )

        # Calculate average score of generated sequences
        mean_score = np.mean(score)

        # Calculate elapsed time
        time_elapsed = int(time.time() - start_time)

        # Estimate time left based on total number of steps and the time elapsed so far
        time_left = time_elapsed * ((self.config.n_steps - step) / (step + 1))

        # Calculate fraction of valid SMILES strings
        valid_fraction = fraction_valid_smiles(smiles)

        # Average number of characters of generated SMILES strings
        mean_len_smiles = np.mean([len(smi) for smi in smiles])

        # Create message for timestep report
        timestep_report = (
            f"\n Step {step}   Fraction valid SMILES: {valid_fraction:4.1f}   Score: {mean_score:.4f}   "
            f"Time elapsed: {time_elapsed}   "
            f"Time left: {time_left:.1f}\n"
            f"Average length of SMILES: {mean_len_smiles}\n"
            f"Policy entropy: {policy_entropy:.8f}\n"
            f"Augmented loss: {loss:.8f}\n"
        )

        # Print message using given logger
        self._logger.log_message(timestep_report)

    def reset(self, reset_countdown: int = 0) -> int:
        """Resets agent and optimizer. Required by MarginGuard()

        Args:
            reset_countdown (int, optional): _description_. Defaults to 0.

        Returns:
            int: reset countdown. Deprecated.
        """
        self._agent = DefaultModel.load_from_file(
            file_path=self.config.agent, sampling_mode=False
        )

        self._optimizer = torch.optim.Adam(
            self._agent.get_network_parameters(), lr=self.config.learning_rate
        )

        assert (
            self._prior.get_vocabulary() == self._agent.get_vocabulary()
        ), "The agent and the prior must have the same vocabulary"

        self._logger.log_message("Resetting Agent")
        self._logger.log_message(f"Adjusting sigma to: {self.config.sigma}")

        return reset_countdown
