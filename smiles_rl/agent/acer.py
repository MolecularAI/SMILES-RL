import torch
import numpy as np
from ..utils.general import to_tensor

from copy import deepcopy

from .utils.rewards import rewards_to_go


from ..model.shared_model import SharedModel
from .base_agent import BaseAgent

from ..configuration_envelope import ConfigurationEnvelope


from reinvent_chemistry.logging import fraction_valid_smiles


from reinvent_scoring import FinalSummary


from torch import nn

from .utils.sample import sample_unique_sequences


from typing import Optional, Tuple, List

import time


class ACER(BaseAgent):
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
        print(f"Using device {self.device}")

        # Initialize model parameters
        self.reset()

        self.discount_factor = self.config.specific_parameters.get(
            "discount_factor", 0.99
        )

        self.entropy_weight = self.config.specific_parameters.get(
            "entropy_weight", 0.001
        )

        self.max_gradient_norm = self.config.specific_parameters.get(
            "max_gradient_norm", 5.0
        )

        self.tau = self.config.specific_parameters.get("tau", 0.99)

        self.delta = self.config.specific_parameters.get("delta", 1)

        self.c = self.config.specific_parameters.get("c", 10)

        self.use_retrace = self.config.specific_parameters.get("use_retrace", False)

        self.n_off_policy_samples = self.config.specific_parameters.get(
            "n_off_policy_samples", 64
        )

        # Constant to avoid dividing by zero
        self.eps = 1e-10

        self.replay_ratio = self.config.specific_parameters.get("replay_ratio", 4)

        self._scoring_function = scoring_function

        self._replay_buffer = replay_buffer

        self._diversity_filter = diversity_filter

        self.step = 0

        self.start_time = time.time()

        self.rng = np.random.default_rng()

    @torch.no_grad()
    def act(self, batch_size: int) -> List[str]:
        """Generate sequences and return corresponding SMILES strings

        Args:
            batch_size (int): number of (non-unqiue) sequences to generate

        Returns:
            List[str]: SMILES strings corresponding to unique sequences.
        """

        self.seqs, self.smiles, self.batch_probs = sample_unique_sequences(
            self._actor_critic, batch_size
        )

        return deepcopy(self.smiles)

    def log_out(self):
        """Using given logger, saves final state of actor-critic model and scaffold memory (for final inspection of generated molecules)"""
        self._logger.save_final_state(self._actor_critic, self._diversity_filter)

    def update(self, smiles: List[str]):
        """Update using actor-critic with experience replay (ACER) https://arxiv.org/abs/1611.01224

        Args:
            smiles (List[str]): SMILES used for updating
        """

        assert (
            self._actor_critic.get_vocabulary() == self._actor_average.get_vocabulary()
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

        # Score summary includes both valid smiles and invalid smiles
        # Invalid smiles are given scores of 0
        score_summary = deepcopy(score_summary)

        # Update scores using given diversity filter
        score = self._diversity_filter.update_score(score_summary, self.step)

        score_report = deepcopy(score)

        # Give invalid SMILES -1 reward during update
        for idx in range(len(score)):
            if idx not in score_summary.valid_idxs:
                score[idx] = -1.0

        smiles = score_summary.scored_smiles

        # Calculate policy entropy
        with torch.no_grad():
            seqs = self._actor_critic.smiles_to_sequences(smiles)
            log_probs, probs = self._actor_critic.log_and_probabilities(seqs)
            policy_entropy = -(probs * log_probs).sum(dim=2).mean()

        # On-policy update
        # No update is done for the first 10 steps (episodes)
        if self.step > 9:
            actor_loss, critic_loss = self._update(
                seqs, to_tensor(score), probs, on_policy=True
            )
        else:
            critic_loss = 0.0
            actor_loss = 0.0

            # Put samples in replay buffer, including the current action probabilities
            self._replay_buffer.put_probs(smiles, score, probs.cpu().numpy(), seqs)

            n_updates_off_policy = (
                self.rng.poisson(self.replay_ratio) if self.step > 10 else 0
            )

            # Off-policy updates
            for _ in range(n_updates_off_policy):

                # Sample from replay buffer, including corresponding probabilities
                (
                    sample_smiles,
                    sample_rewards,
                    sample_probs,
                ) = self._replay_buffer.sample_probs(self.n_off_policy_samples)

                sample_seqs = self._actor_critic.smiles_to_sequences(sample_smiles)

                sample_rewards = to_tensor(sample_rewards)

                sample_probs = to_tensor(sample_probs)

                assert (
                    sample_seqs.size(0),
                    sample_seqs.size(1) - 1,
                    34,
                ) == sample_probs.size(), f"mismatch seqs and probs sizes, sample_seqs: {sample_seqs.size()} sample_probs: {sample_probs.size()}"

                self._update(sample_seqs, sample_rewards, sample_probs, on_policy=False)

        # Timestep report for on-policy update
        self._timestep_report(score_report, critic_loss, actor_loss, policy_entropy)

        if self.step % 500 == 0:
            self._logger.save_intermediate_state(
                self._actor_critic, self._diversity_filter
            )

        self.step += 1

    def _update(
        self,
        sample_seqs: torch.Tensor,
        sample_rewards: torch.Tensor,
        sample_probs: torch.Tensor,
        on_policy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update using ACER, updating parameters of shared model

        Args:
            sample_seqs (torch.Tensor): [batch_size, seq_len] detached
            sample_rewards (torch.Tensor): [batch_size,] detached
            sample_probs (torch.Tensor): [batch_size, seq_len-1, num_actions] detached
            on_policy (bool): Wether to perform on-policy update or not. Default= False

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Losses for actor and critic
        """
        # Reset gradients
        self._actor_critic_optimizer.zero_grad()

        sample_seqs_actions = sample_seqs[:, 1:].unsqueeze(-1)

        # Get q-values
        q = self._actor_critic.q_values(
            sample_seqs,
        )  # [batch_size, seq_len-1, n_actions]
        q_a = q.gather(-1, sample_seqs_actions).squeeze(-1)  # [batch_size, seq_len-1]

        pi = self._actor_critic.probabilities(
            sample_seqs
        )  # [batch_size, seq_len-1, n_actions]
        log_pi = torch.log(pi + self.eps)  # [batch_size, seq_len-1

        pi_a = pi.gather(-1, sample_seqs_actions).squeeze(-1)  # [batch_size, seq_len-1]

        # gradients should be disabled for average network
        pi_avg = self._actor_average.probabilities(
            sample_seqs
        )  # [batch_size, seq_len-1, n_actions]

        log_pi_avg = torch.log(pi_avg + self.eps)
        with torch.no_grad():
            # Calculate values for all states in seqs
            v = self._calc_values(sample_seqs[:, :-1], q, pi)  # [batch_size, seq_len-1]

            if on_policy:
                rho = torch.ones_like(pi)  # [batch_size, seq_len-1, n_actions]
            else:
                rho = pi / (
                    sample_probs + self.eps
                )  # [batch_size, seq_len-1, n_actions]

            rho_a = rho.gather(-1, sample_seqs_actions).squeeze(
                -1
            )  # [batch_size, seq_len -1]

            if self.use_retrace:
                rho_bar = rho_a.clamp(max=1)  # [batch_size, seq_len -1]

                q_ret = self._calc_retrace(
                    sample_seqs,
                    sample_rewards,
                    rho_bar,
                    q_a,
                    v,
                    self.discount_factor,
                )  # [batch_size, seq_len-1]
            else:

                q_ret = rewards_to_go(sample_seqs, sample_rewards, self.discount_factor)

        # [batch_size, seq_len-1, n_actions]

        # Truncated importance sampling
        # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
        adv = q_ret - v  # [batch_size, seq_len-1]
        log_pi_a = torch.log(pi_a + self.eps)
        gain_pi = rho_a.clamp(max=self.c) * log_pi_a * adv
        # Average over batch and sum over sequence
        loss_pi = -gain_pi.mean(0).sum()

        policy_loss = loss_pi

        if not on_policy:  # if off-policy
            # off-policy bias correction for the truncation
            # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
            adv_bc = q.detach() - v.unsqueeze(-1).expand_as(
                q
            )  # [batch_size, seq_len-1, n_actions]

            bc_weight = (1 - self.c / rho).clamp(
                min=0
            ) * pi.detach()  # correction coefficient # [batch_size, seq_len-1, n_actions]

            gain_bc = (log_pi * adv_bc * bc_weight).sum(
                dim=-1
            )  # [batch_size, seq_len -1] # sum over actions

            loss_bc = -gain_bc.mean(0).sum()  # mean over batches and sum of sequence

            policy_loss += loss_bc

        # Loss for q-function output ("head")
        loss_q = ((q_ret - q_a) ** 2 / 2).sum(1).mean()

        # trust region
        pi_avg_a = pi_avg.gather(-1, sample_seqs_actions).squeeze(
            -1
        )  # [batch_size, seq_len-1]
        with torch.no_grad():
            # k = -pi_avg / (pi + self.eps)  # [batch_size, seq_len-1, n_actions]
            k = -(pi_avg_a / (pi_a + self.eps)).unsqueeze(
                -1
            )  # [batch_size, seq_len-1,1]

            g_adj = (
                rho_a.clamp(max=self.c) * (q_ret - v) / (pi_a + self.eps)
            )  # [batch_size, seq_len-1]

            g = torch.zeros_like(q)

            for i in range(g.size(0)):
                for j in range(g.size(1)):
                    k_i = sample_seqs[i, j]
                    g[i, j, k_i] = g_adj[i, j]  # [batch_size, seq_len -1, n_actions]

            if not on_policy:
                g += (
                    bc_weight * (q - v.unsqueeze(-1).expand_as(q)) / (pi + self.eps)
                )  # [batch_size, seq_len-1, n_actions] # .sum(-1)

        # max(0, (k^T∙g - δ) / ||k||^2_2)
        trust_factor = self._calc_trust_region_loss(g, k)

        # k = D_kl [π(∙|s_i; θ_avg)),π(∙|s_i; θ))]
        kl = (
            -(pi_avg * (log_pi - log_pi_avg)).sum(-1).mean(0)
        )  # sum over actions and mean over batches

        # max(0, (k^T∙g - δ) / ||k||^2_2)∙k
        trust_loss = trust_factor * kl

        # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
        # Policy update dθ ← dθ + ∂θ/∂θ∙z*
        policy_loss += trust_loss.sum()

        # Entropy regularization
        policy_loss -= self.entropy_weight * -(log_pi * pi).sum(-1).mean(0).sum()

        # Add policy and value loss together
        loss = policy_loss + loss_q
        loss.backward()

        # Gradient L2 normalization (clip gradients)
        nn.utils.clip_grad_norm_(
            self._actor_critic.get_network_parameters(), self.max_gradient_norm
        )

        self._actor_critic_optimizer.step()

        # Update average network parameters
        self._update_params_moving_average(self._actor_average, self._actor_critic)

        return policy_loss, loss_q

    @torch.no_grad()
    def _calc_trust_region_loss(self, g: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Calculates trust region loss g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k

        Args:
            g (torch.Tensor): Policy gradients with truncated importance sampling
            k (torch.Tensor): Gradient of KL divergence between average policy and actor

        Returns:
            torch.Tensor: trust region loss
        """
        k_dot_g = (k * g).sum(-1).mean(0)  # [seq_len-1,]
        k_dot_k = k.square().sum(-1).mean(0)  # [seq_len-1,]

        # Zero indices
        all_zero_idx = (k_dot_k == 0.0).nonzero(as_tuple=True)

        trust_factor = ((k_dot_g - self.delta) / (k_dot_k + self.eps)).clamp(min=0)

        trust_factor[all_zero_idx] = 0.0

        return trust_factor

    def _calc_values(
        self, seqs: torch.Tensor, q_values: torch.Tensor, probabilities: torch.Tensor
    ) -> torch.Tensor:
        """Calculate state values, given state-action values and probabilities: V(s) = Σ_a π(a|s)Q(s,a)

        Args:
            seqs (torch.Tensor): sequence of tokens
            q_values (torch.Tensor): state-action values for each tokens
            probabilities (torch.Tensor): policy probabilities for each tokens

        Returns:
            torch.Tensor: state values for each token in sequence
        """

        # Find all stop tokens
        all_zero_idx = (seqs == 0).nonzero(as_tuple=True)

        values = (q_values * probabilities).sum(-1)

        assert values.size() == seqs.size()

        # Retrace of stop tokens (terminal states) should be 0
        values[all_zero_idx] = 0.0

        return values  # [batch_size, seq_len -1]

    def _calc_retrace(
        self,
        seqs: torch.Tensor,
        rewards: torch.Tensor,
        rho_bar: torch.Tensor,
        q_a: torch.Tensor,
        v: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Calculates Retrace target estimation

        Args:
            seqs (torch.Tensor): batch of sequence of actions [batch_size, seq_len]
            rewards (torch.Tensor): reward per sequence [batch_size,]
            rho_bar (torch.Tensor): clipped ratio of current probability and sampled probability (max=1) [batch_size, seq_len-1]
            q_a (torch.Tensor): Q-value per action in sequence [batch_size, seq_len -1]
            v (torch.Tensor): value per action in sequence [batch_size, seq_len-1]
            gamma (float): discount factor

        Returns:
            torch.Tensor: Retrace target estimation values
        """
        assert (
            torch.min(torch.amin(seqs, 1)) >= 0
        ), f"minmax token_id of sequence must be 0, but got {torch.min(torch.amin(seqs, 1))}"

        # Obtain idx of first zero element (=stopping token) in each batch
        first_zero_batch_idx = torch.argmin(seqs, 1)

        rew = torch.zeros_like(v)  # [batch_size, seq_len-1]

        # All seqs with stop token, reward is given for state where action choosing first stop token
        for i_row, i_col in enumerate(first_zero_batch_idx):
            # Only non-zero reward if sequence has stop token, otherwise reward should be zero
            if seqs[i_row, i_col] == 0:
                rew[i_row, i_col - 1] = rewards[i_row]

        # Get idx for all stop tokens
        all_zero_idx = (seqs[:, :-1] == 0).nonzero(as_tuple=True)

        # Initialize retraces
        q_rets = torch.zeros_like(v)  # [batch_size, seq_len-1]

        # Recursively calculate retraces
        # Qret(x_t,a_t) =r_t+gamma* rho_bar_{t+1} [Qret(x_{t+1},a_{t+1})−Q(x_{t+1},a_{t+1})] +gamma* V(x_{t+1})
        q_ret = torch.zeros_like(rewards)
        for i_col in reversed(range(seqs.size(1) - 1)):
            ret = q_ret + gamma * rew[:, i_col]
            q_rets[:, i_col] = ret
            # masking stop tokens, cause' should be 0 for all stop tokens
            q_ret = (seqs[:, i_col] > 0) * (
                rho_bar[:, i_col] * (ret - q_a[:, i_col]) + v[:, i_col]
            )

        # Retrace is always zero for terminal steps, i.e., stop tokens
        # This should be redundant since stop tokens are masked above
        q_rets[all_zero_idx] = 0.0

        return q_rets

    @torch.no_grad()
    def _update_params_moving_average(self, target, source) -> None:
        """In-place moving average update for target network

        Args:
            target: Wrapper for target network (weights updated)
            source: Wrapper for source network
        """
        for param_target, param_source in zip(
            target.get_network_parameters(), source.get_network_parameters()
        ):

            param_target.mul_(self.tau).add_(param_source, alpha=1 - self.tau)

    def _timestep_report(
        self,
        score: np.ndarray,
        critic_loss: float,
        actor_loss: float,
        policy_entropy: float,
    ) -> None:
        """Output time step report

        Args:
            score (np.ndarray): scores for scored SMILES strings
            critic_loss (float): critic loss
            actor_loss (float): actor loss
            policy_entropy (float): (Average) policy entropy
        """
        mean_score = np.mean(score)
        valid_fraction = fraction_valid_smiles(self.smiles)
        mean_len_smiles = np.mean([len(smi) for smi in self.smiles])
        timestep_report = (
            f"\n Step {self.step} Fraction valid SMILES: {valid_fraction:4.1f} Score: {mean_score:.4f}\n"
            f"Average length of SMILES: {mean_len_smiles}\n"
            f"Critic loss: {critic_loss}\n"
            f"Actor loss: {actor_loss}\n"
            f"Policy entropy: {policy_entropy}\n"
        )

        self._logger.log_message(timestep_report)

    def update_params(
        self, optimizer: torch.optim.Optimizer, loss: torch.Tensor, n_steps=1
    ) -> None:
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

    def reset(
        self,
    ) -> None:
        """Reset model parameters"""

        self._actor_critic = SharedModel.load_from_file(
            file_path=self.config.agent, sampling_mode=False
        )
        self._actor_critic_optimizer = torch.optim.Adam(
            self._actor_critic.get_network_parameters(), lr=self.config.learning_rate
        )

        self._actor_average = SharedModel.load_from_file(
            file_path=self.config.agent, sampling_mode=True
        )

        self._disable_gradients(self._actor_average)

    def _disable_gradients(self, model) -> None:
        """Disables gradients for parameters of model

        Args:
            model: Wrapper for torch.nn.module
        """

        # There might be a more elegant way of disabling gradients
        for param in model.get_network_parameters():
            param.requires_grad = False
