from ..utils.general import to_tensor
import torch
import numpy as np
from ..model.critic_model import CriticModel
from ..model.actor_model import ActorModel

from .base_agent import BaseAgent

from .utils.rewards import rewards_to_go

from typing import Tuple, List

from .utils.sample import sample_unique_sequences

from ..configuration_envelope import ConfigurationEnvelope


from copy import deepcopy

import time

from reinvent_scoring import FinalSummary


from reinvent_chemistry.logging import fraction_valid_smiles


class PPO(BaseAgent):
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

        self.discount_factor = self.config.specific_parameters.get(
            "discount_factor", 0.99
        )

        self.n_updates_per_iteration = self.config.specific_parameters.get(
            "n_updates_per_iteration", 5
        )

        self.clip = self.config.specific_parameters.get("clip", 0.2)

        self.use_entropy_bonus = self.config.specific_parameters.get(
            "use_entropy_bonus", False
        )

        self.entropy_coeff = self.config.specific_parameters.get("entropy_coeff", 0.001)

        self.clip_grad_norm = self.config.specific_parameters.get(
            "clip_grad_norm", False
        )

        self.max_grad_norm = self.config.specific_parameters.get("max_grad_norm", 0.5)

        self.n_minibatches = self.config.specific_parameters.get("n_minibatches", 4)

        # Initialize model parameters
        self.reset()

        self._diversity_filter = diversity_filter

        self._replay_buffer = replay_buffer

        self._scoring_function = scoring_function

        self.step = 0

        self.n_invalid_steps = 0

        self.start_time = time.time()

    def reset(
        self,
    ) -> None:
        """Reset models"""

        self._critic = CriticModel.load_from_file(
            file_path=self.config.prior, sampling_mode=False
        )
        self._critic_optimizer = torch.optim.Adam(
            self._critic.get_network_parameters(),
            lr=self.config.learning_rate,
        )

        self._actor = ActorModel.load_from_file(
            file_path=self.config.agent, sampling_mode=False
        )
        self._actor_optimizer = torch.optim.Adam(
            self._actor.get_network_parameters(), lr=self.config.learning_rate
        )

        assert (
            self._actor.get_vocabulary() == self._critic.get_vocabulary()
        ), "The agent and the prior must have the same vocabulary"

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

    def update(self, smiles: List[str]):
        """Updates policy (actor) using PPO clipped loss and critic using MSE loss

        Args:
            smiles (List[str]): SMILES strings to use for upgrade
        """

        assert (
            self._critic.get_vocabulary() == self._actor.get_vocabulary()
        ), "The actor and the critic must have the same vocabulary"

        try:
            score_summary = self._scoring_function.get_final_score_for_step(
                smiles, self.step
            )
        except TypeError as inst:
            print(inst, flush=True)
            score_summary = FinalSummary(
                np.zeros((len(smiles),), dtype=np.float32), smiles, [], []
            )

        # Score summary includes both valid smiles and invalid smiles
        # Invalid smiles are given scores of 0
        score_summary = deepcopy(score_summary)
        score = self._diversity_filter.update_score(score_summary, self.step)

        # Calculate policy entropy for sampled sequences
        with torch.no_grad():
            seqs = self._actor.smiles_to_sequences(score_summary.scored_smiles)
            log_probs, probs = self._actor.log_and_probabilities(seqs)
            policy_entropy = -(probs * log_probs).sum(dim=2).mean()

        sample_smiles, sample_score = self._replay_buffer(
            score_summary.scored_smiles, score
        )

        sample_seqs = self._actor.smiles_to_sequences(sample_smiles)

        # Calculate rewards for each action
        batch_rtgs = rewards_to_go(
            sample_seqs, to_tensor(sample_score), self.discount_factor
        )  # [ batch size, number of timesteps]
        # calculate advantages at current step (iteration)
        # Advantages should be detached since gradient should not be over these
        with torch.no_grad():
            _, old_log_probs = self._evaluate(
                sample_seqs
            )  # [batch_size, seqs lenght -1]

        n_batch = sample_seqs.size(0)

        assert n_batch > 0, "Have not sampled any sequences"

        # If we have only sampled non-unique molecules (can easily happen without diversity filter), then
        # it is possible to have less molecules than the minibatch size.
        n_batch_train = (
            n_batch // self.n_minibatches if n_batch > self.n_minibatches else n_batch
        )

        # Update network for some n epochs
        for _ in range(self.n_updates_per_iteration):
            # Calculate V_phi and pi_theta(a_t | s_t)

            actor_loss_mini = []
            critic_loss_mini = []

            with torch.no_grad():
                adv_values, _ = self._evaluate(
                    sample_seqs
                )  # [batch_size, seqs lenght -1]

            adv = batch_rtgs - adv_values

            permut = torch.randperm(n_batch)

            for start in range(0, n_batch, n_batch_train):

                end = start + n_batch_train

                mbinds = permut[start:end]

                # Minibatch values
                mini_seqs = sample_seqs[mbinds, :]

                mini_rewards = batch_rtgs[mbinds, :]

                mini_old_log_probs = old_log_probs[mbinds, :]

                mini_adv = adv[mbinds, :]

                # Normalize advantages per minibatch to increase stability
                mini_adv = (mini_adv - mini_adv.mean()) / (mini_adv.std() + 1e-8)

                values, curr_log_probs = self._evaluate(mini_seqs)

                ratios = torch.exp(curr_log_probs - mini_old_log_probs).nan_to_num()

                assert mini_adv.size() == ratios.size()

                # Calculate surrogate losses
                surr1 = ratios * mini_adv

                assert surr1.size() == ratios.size()

                clip_range = self.clip

                surr2 = mini_adv * torch.clamp(ratios, 1 - clip_range, 1 + clip_range)

                # Calculate actor and critic losses
                actor_loss = -torch.minimum(surr1, surr2).mean()

                actor_loss_mini.append(actor_loss.item())

                if self.use_entropy_bonus:
                    log_probs, probs = self._actor.log_and_probabilities(mini_seqs)
                    entropy = (probs * log_probs).sum(dim=2).mean()

                    actor_loss += self.entropy_coeff * entropy

                critic_loss = 0.5 * torch.nn.functional.mse_loss(values, mini_rewards)

                critic_loss_mini.append(critic_loss.item())

                # Calculate gradient and perform backward propagation for actor network
                self._actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                # if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self._actor.get_network_parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )

                self._actor_optimizer.step()

                # Calculate gradients and perform backward propagation for critic network
                self._critic_optimizer.zero_grad()
                critic_loss.backward()

                # if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self._critic.get_network_parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )

                self._critic_optimizer.step()

        self._timestep_report(
            score, np.mean(critic_loss_mini), np.mean(actor_loss_mini), policy_entropy
        )

        if self.step % 500 == 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def _evaluate(
        self,
        seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtains values of critic and log probabilities of actor for sequence of token ids

        Args:
            seqs (torch.Tensor): batches of sequences of token ids [n_bacthes, sequence length]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: values [bacth size, sequence_length-1] and
                lob probabilities [batch size, sequence_length-1]
                for actions taken in sequence. Both excludes number for stop token.
        """

        values = self._critic.values(seqs)

        log_probs = self._actor.log_probabilities_action(seqs)

        return (values, log_probs.clamp(min=-20))

    def log_out(self):
        """Save final state of actor and memory"""
        self._logger.save_final_state(self._actor, self._diversity_filter)

    def _timestep_report(
        self,
        score: np.ndarray,
        critic_loss: float,
        actor_loss: float,
        policy_entropy: float,
    ):
        """Timestep report to the standard error output, using given logger

        Args:
            score (np.ndarray): score for each sampled SMILES
            critic_loss (float): critic loss
            actor_loss (float): actor loss
            policy_entropy (float): (average) output
        """
        mean_score = np.mean(score)
        valid_fraction = fraction_valid_smiles(self.smiles)

        if valid_fraction < 0.8:
            self.n_invalid_steps += 1
        else:
            self.n_invalid_steps = 0

        if self.n_invalid_steps > 10:
            self.reset()
            self.n_invalid_steps = 0

        mean_len_smiles = np.mean([len(smi) for smi in self.smiles])
        timestep_report = (
            f"\n Step {self.step} Fraction valid SMILES: {valid_fraction:4.1f} Score: {mean_score:.4f}\n"
            f"Average length of SMILES: {mean_len_smiles}\n"
            f"Critic loss: {critic_loss}\n"
            f"Actor loss: {actor_loss}\n"
            f"Policy entropy: {policy_entropy}\n"
        )

        self._logger.log_message(timestep_report)
