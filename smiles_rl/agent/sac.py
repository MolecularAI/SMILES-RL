import torch
import numpy as np

from smiles_rl.agent.utils.rewards import rewards_to_go
from ..utils.general import to_tensor

from copy import deepcopy


from ..model.actor_model import ActorModel
from .base_agent import BaseAgent

from ..configuration_envelope import ConfigurationEnvelope


from reinvent_chemistry.logging import fraction_valid_smiles


from reinvent_scoring import FinalSummary


from .utils.sample import sample_unique_sequences

import time


from typing import Optional, Tuple, List


class SAC(BaseAgent):
    def __init__(
        self,
        config: ConfigurationEnvelope,
        scoring_function,
        diversity_filter,
        replay_buffer,
        logger,
    ):

        self._config = config
        self._logger = logger

        self.config = config.reinforcement_learning.parameters

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_updates_off_policy = self.config.specific_parameters.get(
            "n_updates_off_policy", 4
        )

        self.discount_factor = self.config.specific_parameters.get("discount_factor", 1)

        self.initial_temperature = self.config.specific_parameters.get(
            "temperature", 0.0001
        )

        self.learn_temperature = self.config.specific_parameters.get(
            "learn_temperature", True
        )

        self.use_log_temperature = self.config.specific_parameters.get(
            "use_log_temperature", False
        )

        self.use_average_network = self.config.specific_parameters.get(
            "average_network", False
        )

        self.clip_critic = self.config.specific_parameters.get("clip_critic", False)

        self.reset_critic1 = self.config.specific_parameters.get("reset_critic1", False)

        self.reset_critic2 = self.config.specific_parameters.get("reset_critic2", False)

        self.average_network_scale = self.config.specific_parameters.get(
            "average_network_scale", 1
        )

        self.max_grad_norm = self.config.specific_parameters.get("max_grad_norm", 0.5)

        self.tau = self.config.specific_parameters.get("tau", 0.99)

        self.add_entropy_target = self.config.specific_parameters.get(
            "add_entropy_target", True
        )

        self.learning_rate_critic = self.config.specific_parameters.get(
            "learning_rate_critic", self.config.learning_rate
        )

        self.learning_rate_actor = self.config.specific_parameters.get(
            "learning_rate_actor", self.config.learning_rate
        )

        self.learning_rate_temperature = self.config.specific_parameters.get(
            "learning_rate_temperature", self.config.learning_rate
        )

        # Initialize models and optimizers
        self.reset()

        self._replay_buffer = replay_buffer

        self._scoring_function = scoring_function

        self._diversity_filter = diversity_filter

        # Set default target entropy. Should not be too far from the entropy of the pre-trained policy
        self.target_entropy = self.config.specific_parameters.get("target_entropy", 0.3)

        self.step = 0

        self.start_time = time.time()

        # Initialize BitGenerator
        self.rng = np.random.default_rng()

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

        # Save q-tables and probabilities of sampled sequences for inspection
        self.save_q_tables_and_probabilities(self.seqs)

        return deepcopy(self.smiles)

    def log_out(self):
        """Save final state of actor and memory for final inspection"""
        self._logger.save_final_state(self._actor, self._diversity_filter)

    def update(self, smiles: List[str]):
        """Use discrete soft-actor critic for updating policy https://arxiv.org/abs/1910.07207

        Args:
            smiles (List[str]): SMILES strings to use for update
        """

        assert (
            self._critic1.get_vocabulary() == self._actor.get_vocabulary()
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

        # Give invalid SMILES -1 reward during update
        for idx in range(len(score)):
            if idx not in score_summary.valid_idxs:
                score[idx] = -1.0

        assert len(score) == self.seqs.size(0)

        with torch.no_grad():
            # This is done to get conistency between samples and current seqs,
            # in particular if seqs reaches max length when generated
            # (since then there will be no stop sequence in the generated seqs)
            seqs = self._actor.smiles_to_sequences(score_summary.scored_smiles)

            log_probabilities, probabilities = self._actor.log_and_probabilities(seqs)
            entropy = -(probabilities * log_probabilities).sum(-1)

        # On-policy update
        if self.step > 10:
            (
                critic1_loss_on_policy,
                critic2_loss_on_policy,
                actor_loss_on_policy,
                temperature_loss_on_policy,
            ) = self._update(seqs, to_tensor(score))
        else:
            critic1_loss_on_policy = 0.0
            critic2_loss_on_policy = 0.0
            actor_loss_on_policy = 0.0
            temperature_loss_on_policy = 0.0

        # Put episodes in replay buffer
        self._replay_buffer.put(score_summary.scored_smiles, score)

        # Off-policy update
        n_updates_off_policy = self.n_updates_off_policy if self.step > 10 else 0
        for _ in range(n_updates_off_policy):
            (
                sample_smiles,
                sample_rewards,
            ) = self._replay_buffer.sample()

            sample_seqs = self._actor.smiles_to_sequences(sample_smiles)

            # Convert np.ndarray to torch.Tensor
            sample_rewards = to_tensor(sample_rewards)

            critic1_loss, critic2_loss, actor_loss, temperature_loss = self._update(
                sample_seqs, sample_rewards
            )

        # timestep report for on-policy losses
        self._timestep_report(
            score_report,
            critic1_loss_on_policy,
            critic2_loss_on_policy,
            actor_loss_on_policy,
            temperature_loss_on_policy,
            entropy.mean(),
        )

        if self.step % 500 == 0:
            self._logger.save_intermediate_state(self._actor, self._diversity_filter)

        self.step += 1

    def _update(
        self,
        sample_seqs: torch.Tensor,
        sample_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Does one step update for discrete SAC, updating actor, average policy, critic(s) and temperature parameters

        Args:
            sample_seqs (torch.Tensor): [batch_size, seq_len] detached
            sample_rewards (torch.Tensor): [batch_size,] detached

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Losses of actors, critics and temperatur
        """

        log_probabilities, _ = self._actor.log_and_probabilities(sample_seqs)

        # Calculate loss for Q-function 1 and 2
        critic1_loss, critic2_loss = self.calc_critic_loss(
            sample_seqs,
            sample_rewards,
            log_probabilities.detach(),
            self._critic1,
            self._critic2,
            self._critic_target1,
            self._critic_target2,
        )

        # Update the Q-function parameters
        self.update_params_clip_grad_norm(
            self._critic1,
            self._critic1_optimizer,
            critic1_loss,
            max_grad_norm=self.max_grad_norm,
        )

        self.update_params_clip_grad_norm(
            self._critic2,
            self._critic2_optimizer,
            critic2_loss,
            max_grad_norm=self.max_grad_norm,
        )

        # Update policy weights
        actor_loss = self.calc_actor_loss(
            sample_seqs,
            log_probabilities,
        )

        if self.use_average_network:
            # Add KL divergence regularization to average network
            log_probs_avg, probs_avg = self._actor_average.log_and_probabilities(
                sample_seqs
            )

            kl = (probs_avg * (log_probs_avg - log_probabilities)).sum(-1).mean()
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
            self._update_params_moving_average(self._actor_average, self._actor, 0.99)

        # Temperature loss
        temperature_loss = self.calc_temperature_loss(
            sample_seqs, log_probabilities.detach()
        )

        # Update temperature if we want to learn temperature
        if self.learn_temperature:
            self.update_params(self._temperature_optimizer, temperature_loss)

            # Update temperature parameter, using the new temperature values, used in the actor and critic losses
            if self.use_log_temperature:
                self.temperature = self.log_temperature.exp().item()
            else:
                self.temperature = self.log_temperature.item()

        # Update target network weights
        self._update_params_moving_average(
            self._critic_target1, self._critic1, self.tau
        )
        self._update_params_moving_average(
            self._critic_target2, self._critic2, self.tau
        )

        return critic1_loss, critic2_loss, actor_loss, temperature_loss

    @torch.no_grad()
    def _update_params_moving_average(self, target, source, tau: float):
        """In-place moving average update of target network, from source network

        Args:
            target: wrapper of torch.nn.Module target, e.g, ActorModel
            source: wrapper of torch.nn.Module source, e.g., ActorModel
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
        critic1_loss: float,
        critic2_loss: float,
        actor_loss: float,
        temperature_loss: float,
        policy_entropy: float,
    ):
        """Output time step report to standard error output

        Args:
            score (np.ndarray): scores for scored SMILES strings
            critic_loss1 (float): loss of first critic
            critic_loss2 (float): loss of second critic
            actor_loss (float): actor loss
            temperature_loss (float): loss of temperature parameter
            policy_entropy (float): (Average) policy entropy
        """

        mean_score = np.mean(score)
        n_unique_smiles = len(set(self.smiles))
        mean_len_smiles = np.mean([len(smi) for smi in self.smiles])
        valid_fraction = fraction_valid_smiles(self.smiles)
        timestep_report = (
            f"\n Step {self.step} Fraction valid SMILES: {valid_fraction:4.1f} Score: {mean_score:.4f}\n"
            f"Average length of SMILES: {mean_len_smiles}\n"
            f"Temperature: {self.temperature}\n"
            f"# unique SMILES: {n_unique_smiles}\n"
            f"Critic1 loss: {critic1_loss}\n"
            f"Critic2 loss: {critic2_loss}\n"
            f"Actor loss: {actor_loss}\n"
            f"Temperature loss: {temperature_loss}\n"
            f"Policy entropy: {policy_entropy}\n"
        )

        self._logger.log_message(timestep_report)

    def update_params(
        self, optimizer: torch.optim.Optimizer, loss: torch.Tensor, n_steps: int = 1
    ):
        """Optimization of parameters corresponding to given optimizer and loss

        Args:
            optimizer (torch.optim.Optimizer): Optimizer
            loss (torch.Tensor): Loss value to be used for optimization
            n_steps (int, optional): Number of steps of update. Defaults to 1.
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
        max_grad_norm: float = 0.5,
    ):
        """Optimization of parameters corresponding to given optimizer and loss using clipped gradient norm

        Args:
            optimizer (torch.optim.Optimizer): Optimizer
            loss (torch.Tensor): Loss value to be used for optimization
            n_steps (int, optional): Number of steps of update. Defaults to 1.
            max_grad_norm (float, optional): maximum gradient norm. Defaults to 0.5.
        """
        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.get_network_parameters(), max_grad_norm)

        for _ in range(n_steps):
            optimizer.step()

    def calc_temperature_loss(
        self, seqs: torch.Tensor, log_probabilities: torch.Tensor
    ) -> torch.Tensor:
        """Calculates temperature loss for discrete SAC https://arxiv.org/abs/1910.07207

        Args:
            seqs (torch.Tensor): Sequence of tokens
            log_probabilities (torch.Tensor): log probabilities (of policy) for each token at each state

        Returns:
            torch.Tensor: temperature loss
        """

        # We remove entropy for action corresponding to stop token, since we care more about the entropy at earlier states
        seqs_just_before = deepcopy(seqs)
        first_stop_token = torch.argmin(seqs_just_before, dim=1)

        before_stop_token = (torch.arange(len(first_stop_token)), first_stop_token - 1)

        seqs_just_before[before_stop_token] = 0

        all_entropy_idx = (seqs_just_before[:, 1:-1] > 0).nonzero(as_tuple=True)

        # (negative) entropy
        temperature_loss = (torch.exp(log_probabilities) * log_probabilities).sum(
            dim=-1
        )

        # Add target entropy
        temperature_loss += self.target_entropy

        # multiply temperature that will be updated
        temperature_loss *= -self.log_temperature

        temperature_loss = temperature_loss[all_entropy_idx].mean()

        return temperature_loss

    def calc_actor_loss(
        self,
        seqs: torch.Tensor,
        log_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates actor loss for discrete SAC https://arxiv.org/abs/1910.07207

        Args:
            seqs (torch.Tensor): Sequence of tokens
            log_probabilities (torch.Tensor): log probabilities (of policy) for each token at each state

        Returns:
            torch.Tensor: actor loss
        """

        log_probabilities, probabilities = self._actor.log_and_probabilities(seqs)

        with torch.no_grad():
            q1 = self._critic1.q_values(seqs)  # [batch_size, seq_len -1, n_actions]
            q2 = self._critic2.q_values(seqs)

            q_values = torch.mean(torch.stack([q1, q2], dim=-1), dim=-1)

        # actor loss
        actor_loss = -(probabilities * q_values).sum(dim=-1).mean()

        # entropy loss term
        entropy_term = (
            self.temperature * (probabilities * log_probabilities).sum(dim=-1).mean()
        )

        actor_loss += entropy_term

        return actor_loss

    def calc_critic_loss(
        self,
        seqs: torch.Tensor,
        rewards: torch.Tensor,
        log_probabilities: torch.Tensor,
        critic1: ActorModel,
        critic2: ActorModel,
        critic_target1: ActorModel,
        critic_target2: ActorModel,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcuates both critic losses  for discrete SAC https://arxiv.org/abs/1910.07207

        Optionally loss is clipped as in https://arxiv.org/abs/2209.10081

        Args:
            seqs (torch.Tensor): sequences of actions (batch_size, sequence_length)
            rewards (torch.Tensor): rewards for each sequence (batch_size, )
            log_probabilities (torch.Tensor): log-probabilities for each action of sequences (batch_size, sequence_lenght, n_actions)
            critic1 (ActorModel): 1st critic model, giving q-values
            critic2 (ActorModel): 2nd critic model, giving q-values
            critic_target1 (ActorModel): 1st target ciritc model, giving q-values
            critic_target2 (ActorModel): 2nd target ciritc model, giving q-values

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss of 1st and 2nd critic
        """

        # Makes sure to not count stop tokens toward entropy term in loss
        seqs_just_before = deepcopy(seqs)

        first_stop_token = torch.argmin(seqs_just_before, dim=1)

        before_stop_token = (torch.arange(len(first_stop_token)), first_stop_token - 1)

        seqs_just_before[before_stop_token] = 0

        all_entropy_idx = (seqs_just_before[:, 1:-1] > 0).nonzero(as_tuple=True)

        with torch.no_grad():

            log_probabilities, probabilities = self._actor.log_and_probabilities(seqs)

            q_values_target1 = critic_target1.q_values(
                seqs
            )  # [batch_size, sequence_len-1, num_actions]

            assert q_values_target1.size() == log_probabilities.size()

            q_values_target2 = critic_target2.q_values(seqs)

            # Q-values for action in the sampled sequence (excluding stop token)
            q_values_target1_action = q_values_target1.gather(
                -1, seqs[:, 1:].unsqueeze(-1)
            ).squeeze(
                -1
            )  # [batch_size, sequence_len-1,]
            q_values_target2_action = q_values_target2.gather(
                -1, seqs[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # Estimated discounted return at each observed state
            batch_rewards = rewards_to_go(seqs, rewards, self.discount_factor)

            # (log) probabilities for future state
            log_probabilities_next_state = log_probabilities[:, 1:, :]

            probabilities_next_state = probabilities[:, 1:, :]

            # Temperature for each state, since we do not want to use stop tokens for update (set temperature to zero for these)
            temperature = torch.zeros_like(log_probabilities_next_state)

            temperature[all_entropy_idx] = self.temperature

            target = batch_rewards

            # Add entropy of future states to critic target
            if self.add_entropy_target:

                entropy_next_states = -(
                    temperature
                    * probabilities_next_state
                    * log_probabilities_next_state
                ).sum(-1)

                # cumulative sum of future policy entropies
                cum_sum = torch.cumsum(entropy_next_states, dim=1)
                reverse_cum_sum = entropy_next_states - cum_sum + cum_sum[-1:None]

                # Zero-padding since terminal state has zero entropy
                pad = torch.zeros(seqs.size(0), 1)

                target += torch.cat((reverse_cum_sum, pad), dim=1)

        q_values1 = critic1.q_values(seqs)
        q_values2 = critic2.q_values(seqs)

        q_values1_action = q_values1.gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
        q_values2_action = q_values2.gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)

        assert q_values1_action.size() == q_values2_action.size()

        # Wether to clip critic loss or not
        if self.clip_critic:
            # Clipped critic loss using target-critic, as suggested by https://arxiv.org/abs/2209.10081
            critic_loss1 = (
                0.5
                * torch.maximum(
                    torch.nn.functional.mse_loss(
                        q_values1_action, target, reduction="none"
                    ),
                    torch.nn.functional.mse_loss(
                        q_values_target1_action
                        + torch.clamp(
                            q_values1_action - q_values_target1_action, -0.5, 0.5
                        ),
                        target,
                        reduction="none",
                    ),
                ).mean()
            )

            critic_loss2 = (
                0.5
                * torch.maximum(
                    torch.nn.functional.mse_loss(
                        q_values2_action, target, reduction="none"
                    ),
                    torch.nn.functional.mse_loss(
                        q_values_target2_action
                        + torch.clamp(
                            q_values2_action - q_values_target2_action, -0.5, 0.5
                        ),
                        target,
                        reduction="none",
                    ),
                ).mean()
            )
        else:
            # No clipping of loss, just MSE loss
            critic_loss1 = 0.5 * torch.nn.functional.mse_loss(q_values1_action, target)
            critic_loss2 = 0.5 * torch.nn.functional.mse_loss(q_values2_action, target)

        return critic_loss1, critic_loss2

    def reset(
        self,
    ):
        "Reset model parameters"
        self._actor = ActorModel.load_from_file(
            file_path=self.config.agent, sampling_mode=False
        )
        self._actor_optimizer = torch.optim.Adam(
            self._actor.get_network_parameters(), lr=self.learning_rate_actor
        )

        self._critic1 = ActorModel.load_from_file(
            file_path=self.config.prior, sampling_mode=False
        )

        self._prior = deepcopy(self._critic1)

        self._disable_gradients(self._prior, True)

        # Reseting the linear output layer for critic1
        if self.reset_critic1:
            self._critic1.reset_output_layer()

        self._critic_target1 = deepcopy(self._critic1)

        self._critic1_optimizer = torch.optim.Adam(
            self._critic1.get_network_parameters(), lr=self.learning_rate_critic
        )

        self._disable_gradients(self._critic_target1, True)

        self._critic2 = deepcopy(self._critic1)

        # Reseting the linear output layer for critic2
        if self.reset_critic2:
            self._critic2.reset_output_layer()

        self._critic_target2 = deepcopy(self._critic2)

        self._critic2_optimizer = torch.optim.Adam(
            self._critic2.get_network_parameters(), lr=self.learning_rate_critic
        )

        self._disable_gradients(self._critic_target2, True)

        # If average network is used for KL divergence regularization of policy,
        # intialize it as the initial actor and disable it gradients
        # (it is updated using moving average of actor policy)
        if self.use_average_network:
            self._actor_average = ActorModel.load_from_file(
                file_path=self.config.prior, sampling_mode=True
            )

            self._disable_gradients(self._actor_average)

        if self.use_log_temperature:
            # We optimize log(temp), instead of temp.
            self.log_temperature = torch.tensor(
                [np.log(self.initial_temperature)], requires_grad=self.learn_temperature
            )

            self.temperature = self.log_temperature.exp().item()

        else:
            self.log_temperature = torch.tensor(
                [self.initial_temperature], requires_grad=self.learn_temperature
            )

            self.temperature = self.log_temperature.item()

        self._temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=self.learning_rate_temperature
        )

    def _disable_gradients(self, model, use_inference_mode: bool = False):
        """Disables gradients for parameters of model

        Args:
            model: Wrapper for torch.nn.module, e.g., ActorModel
        """

        # There might be a more elegant way of disabling gradients
        if use_inference_mode:
            model.set_mode("inference")

        for param in model.get_network_parameters():
            param.requires_grad = False

    def save_q_tables_and_probabilities(
        self,
        seqs: torch.Tensor,
    ):
        """Using given logger, save Q-values and probabilities for (current) batch of sampled sequences

        Args:
            seqs (torch.Tensor): Batch of sampled sequences [batch_size, sequence_length]
        """

        q_values1 = self._critic1.q_values(seqs)
        q_values2 = self._critic2.q_values(seqs)
        probs = self._actor.probabilities(seqs)
        self._logger.save_q_tables_and_probabilities(
            q_values1, q_values2, probs, seqs, self.step
        )
