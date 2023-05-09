import torch
import numpy as np
from ..utils.general import to_tensor

from .utils.rewards import rewards_to_go

from copy import deepcopy

from ..model.critic_model import CriticModel
from ..model.actor_model import ActorModel
from .base_agent import BaseAgent

from ..configuration_envelope import ConfigurationEnvelope


from reinvent_chemistry.logging import fraction_valid_smiles


from reinvent_scoring import FinalSummary


from .utils.sample import sample_unique_sequences

import time

from typing import List, Tuple


class A2C(BaseAgent):
    def __init__(
        self,
        config: ConfigurationEnvelope,
        scoring_function,
        diversity_filter,
        replay_buffer,
        logger,
    ):

        self._logger = logger

        self.config = config.reinforcement_learning.parameters

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.discount_factor = self.config.specific_parameters.get(
            "discount_factor", 1.0
        )

        self.use_average_network = self.config.specific_parameters.get(
            "average_network", False
        )

        self.average_network_scale = self.config.specific_parameters.get(
            "average_network_scale", 1
        )

        self.entropy_penalty = self.config.specific_parameters.get(
            "entropy_penalty", 0.25
        )

        self.max_grad_norm = self.config.specific_parameters.get("max_grad_norm", 5)

        self.tau = self.config.specific_parameters.get("tau", 0.99)

        self.learning_rate_critic = self.config.specific_parameters.get(
            "learning_rate_critic", self.config.learning_rate
        )

        self.learning_rate_actor = self.config.specific_parameters.get(
            "learning_rate_actor", self.config.learning_rate
        )

        # Initialize policy and critic parameters
        self.reset()

        self._scoring_function = scoring_function

        self._diversity_filter = diversity_filter

        self._replay_buffer = replay_buffer

        self.step = 0

        self.n_invalid_steps = 0

        self.start_time = time.time()

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        """Generate sequences and return corresponding SMILES strings

        Args:
            batch_size (int): number of (non-unqiue) sequences to generate

        Returns:
            List[str]: SMILES strings corresponding to unique sequences.
        """

        self.seqs, self.smiles, self.batch_log_probs = sample_unique_sequences(
            self._actor, batch_size
        )

        return deepcopy(self.smiles)

    def log_out(self):
        """Save final state of actor and memory for final inspection"""
        self._logger.save_final_state(self._actor, self._diversity_filter)

    def update(self, smiles: List[str]):
        """Use Advantage Actor Critic to udpate policy used for sampling sequences

        Args:
            smiles (List[str]): SMILES strings to use for update
        """

        assert (
            self._critic.get_vocabulary() == self._actor.get_vocabulary()
        ), "The actor and the critic must have the same vocabulary"

        # Score SMILES strings using given scoring function
        try:
            score_summary = self._scoring_function.get_final_score_for_step(
                smiles, self.step
            )
        except TypeError as inst:
            # If no valid SMILES strings was generated
            print(inst, flush=True)
            score_summary = FinalSummary(
                np.zeros((len(smiles),), dtype=np.float32), smiles, [], []
            )

        score_summary = deepcopy(score_summary)

        # Update scores using given diversity filter
        score = self._diversity_filter.update_score(score_summary, self.step)

        # Save score for timestep report
        score_report = deepcopy(score)

        assert len(score) == self.seqs.size(0)

        # Put episodes in replay buffer
        with torch.no_grad():
            # This is done to get conistency between samples and current seqs,
            # in particular if seqs reaches max length when generated
            # (since then there will be no stop sequence in the generated seqs)
            seqs = self._actor.smiles_to_sequences(score_summary.scored_smiles)

            log_probabilities, probabilities = self._actor.log_and_probabilities(seqs)

            # Calculate policy entropy for current actions taken
            entropy = -(probabilities * log_probabilities).sum(-1)

        # Use replay buffer (on-policy) to sample SMILES strings and corresponding rewards
        sample_smiles, sample_rewards = self._replay_buffer(
            score_summary.scored_smiles, score
        )

        # Generate a sequence of tokens for each SMILES string
        sample_seqs = self._actor.smiles_to_sequences(sample_smiles)

        # Convert reward from numpy.ndarray to torch.Tensor
        sample_rewards = to_tensor(sample_rewards)

        # In-place update of parameters
        critic_loss, actor_loss = self._update(
            sample_seqs,
            sample_rewards,
        )

        self._timestep_report(
            score_report,
            critic_loss.item(),
            actor_loss.item(),
            entropy.mean(),
        )

        # Intermediate save of the actor network and scaffold memory
        if self.step % 500 == 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def _update(
        self,
        sample_seqs: torch.Tensor,
        sample_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A2C udpate of actor and critic (non-shared parameters)

        Args:
            sample_seqs (torch.Tensor): batch of sequences of tokens [batch_size, sequence_length]
            sample_rewards (torch.Tensor): reward for each sequences [batch_size, ]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: critic and actor loss
        """

        log_probabilities, _ = self._actor.log_and_probabilities(sample_seqs)

        log_probabilities_action = log_probabilities.gather(
            -1, sample_seqs[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        values = self._critic.values(sample_seqs)

        batch_rewards = rewards_to_go(sample_seqs, sample_rewards, self.discount_factor)

        # Update policy weights

        actor_loss = self.calc_actor_loss(
            sample_seqs,
            batch_rewards,
            log_probabilities_action,
            values,
        )

        if self.use_average_network:
            # Add KL divergence regularization to average network
            log_probs_avg, probs_avg = self._actor_average.log_and_probabilities(
                sample_seqs
            )

            kl = (probs_avg * (log_probs_avg - log_probabilities)).sum() / (
                log_probabilities.size(0) + log_probabilities.size(1)
            )
            kl *= self.average_network_scale

            actor_loss += kl

        # update with parameters with L2 regularization

        self.update_params_clip_grad_norm(
            self._actor,
            self._actor_optimizer,
            actor_loss,
            max_grad_norm=self.max_grad_norm,
        )

        if self.use_average_network:
            self._update_params_moving_average(
                self._actor_average, self._actor, self.tau
            )

        # Update critic weights

        critic_loss = self.calc_critic_loss(
            sample_seqs,
            batch_rewards,
            values,
        )

        self.update_params_clip_grad_norm(
            self._critic,
            self._critic_optimizer,
            critic_loss,
            max_grad_norm=self.max_grad_norm,
        )

        return critic_loss, actor_loss

    @torch.no_grad()
    def _update_params_moving_average(self, target, source, tau: float):
        """In-place moving average update of target network, from source network

        Args:
            target: wrapper of torch.nn.Module target
            source: wrapper of torch.nn.Module source
            tau (float): smoothing coefficient
        """

        for param_target, param_source in zip(
            target.get_network_parameters(), source.get_network_parameters()
        ):
            param_target.mul_(tau)
            param_target.add_(param_source, alpha=1 - tau)

    def _timestep_report(
        self,
        score: np.ndarray,
        critic_loss: float,
        actor_loss: float,
        policy_entropy: float,
    ):
        """Prints timestep report to Standard Error output, using given logger

        Also, resets model parameters if several consecutive steps have sampled a large fraction of invalid SMILES strings

        Args:
            score (np.ndarray): score for sequences of interest
            critic_loss (float): critic loss
            actor_loss (float): actor loss
            policy_entropy (float): policy entropy
        """

        mean_score = np.mean(score)
        mean_len_smiles = np.mean([len(smi) for smi in self.smiles])
        valid_fraction = fraction_valid_smiles(self.smiles)

        # Reset model if sampling a high fraction
        # of invalid SMILES for more than 10 steps
        if valid_fraction < 0.8:
            self.n_invalid_steps += 1
        else:
            self.n_invalid_steps = 0

        if self.n_invalid_steps > 10:
            self.reset()
            self.n_invalid_steps = 0

        timestep_report = (
            f"\n Step {self.step} Fraction valid SMILES: {valid_fraction:4.1f} Score: {mean_score:.4f}\n"
            f"Average length of SMILES: {mean_len_smiles}\n"
            f"Critic loss: {critic_loss}\n"
            f"Actor loss: {actor_loss}\n"
            f"Policy entropy: {policy_entropy}\n"
        )

        self._logger.log_message(timestep_report)

    def update_params(
        self, optimizer: torch.optim.Optimizer, loss: torch.Tensor, n_steps: int = 1
    ):
        """Update parameters using given loss and optimizer

        Args:
            optimizer (torch.optim.Optimizer): optimizer
            loss (torch.Tensor): loss to optimize
            n_steps (int, optional): number of update steps. Defaults to 1.
        """

        optimizer.zero_grad()

        loss.backward()

        for _ in range(n_steps):
            optimizer.step()

    def update_params_clip_grad_norm(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        n_steps: int = 1,
        max_grad_norm: float = 5,
    ):
        """Update parameters using given loss and optimizer. Clips global gradient norm before updating parameters.

        Args:
            model: wrapper of torch.nn.Module
            optimizer (torch.optim.Optimizer): optimizer
            loss (torch.Tensor): loss to optimize
            n_steps (int, optional): number of update steps. Defaults to 1.
            max_grad_norm (float, optional): maxmimum gradient norm for clipping. Defaults to 5.
        """

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.get_network_parameters(), max_grad_norm)

        for _ in range(n_steps):
            optimizer.step()

    def calc_actor_loss(
        self,
        seqs: torch.Tensor,
        batch_rewards: torch.Tensor,
        log_probabilities_action: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates actor loss by policy gradient, using advantage for reducing variance

        Args:
            seqs (torch.Tensor): batch of sequences of tokens/actions [batch_size, sequence_length]
            batch_rewards (torch.Tensor): rewards for sequences in batch [batch_size, ]
            log_probabilities_action (torch.Tensor): log probabilities of current policy for all actions in sequence, excluding last token [batch_size, sequence_length-1]
            values (torch.Tensor): values for each state in sequence, excluding last token [batch_size, sequence_length-1]

        Returns:
            torch.Tensor: loss of actor
        """

        # Compute advantage
        adv = batch_rewards - values.detach()

        actor_loss = -(log_probabilities_action * adv).mean()

        return actor_loss

    def calc_critic_loss(
        self,
        seqs: torch.Tensor,
        batch_rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates critic loss

        Args:
            seqs (torch.Tensor): batch of sequences [batch_size, sequence_length]
            batch_rewards (torch.Tensor): reward at each step in sequences [batch_size, sequence_length-1]
            values_action (torch.Tensor): values for each action in seqs, not detached [batch_size, sequence_len -1]

        Returns:
            torch.Tensor: loss of critic
        """

        target = batch_rewards

        critic_loss = 0.5 * torch.nn.functional.mse_loss(target, values)

        return critic_loss

    def reset(
        self,
    ):
        """Reset model parameters"""
        self._actor = ActorModel.load_from_file(
            file_path=self.config.agent, sampling_mode=False
        )
        self._actor_optimizer = torch.optim.Adam(
            self._actor.get_network_parameters(), lr=self.learning_rate_actor
        )

        self._critic = CriticModel.load_from_file(
            file_path=self.config.prior, sampling_mode=False
        )
        self._critic_optimizer = torch.optim.Adam(
            self._critic.get_network_parameters(),
            lr=self.config.learning_rate,
        )

        if self.use_average_network:
            self._actor_average = ActorModel.load_from_file(
                file_path=self.config.prior, sampling_mode=True
            )

            self._disable_gradients(self._actor_average)

    def _disable_gradients(self, model, use_inference_mode: bool = False):
        """Disable gradients

        Args:
            model (_type_): wrapper of torch.nn.Module
            use_inference_mode (bool, optional): Wether to set model to evaluation mode when disabling gradients. Defaults to False.
        """

        # There might be a more elegant way of disabling gradients
        if use_inference_mode:
            model.set_mode("inference")

        for param in model.get_network_parameters():
            param.requires_grad = False

    def save_q_table_and_probabilities(
        self,
        seqs: torch.Tensor,
    ):
        """Use logger to save critic values and actor probabilities

        Args:
            seqs (torch.Tensor): batch of sequences of tokens [batch_size, sequence_length]
        """

        values = self._critic.values(seqs)

        probs = self._actor.probabilities(seqs)
        self._logger.save_q_table_and_probabilities(values, probs, seqs, self.step)
