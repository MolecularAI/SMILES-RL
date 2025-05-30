import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import reinvent_chemistry.logging as ul_rl
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import (
    BaseDiversityFilter,
)

from .reinforcement_log_configuration import ReinforcementLoggerConfiguration
from ..configuration_envelope import (
    ConfigurationEnvelope,
)
from .console_message import ConsoleMessage
from .base_reinforcement_logger import (
    BaseReinforcementLogger,
)
from reinvent_scoring.scoring.score_summary import FinalSummary
from reinvent_scoring.scoring.enums.scoring_function_component_enum import (
    ScoringFunctionComponentNameEnum,
)


class LocalReinforcementLogger(BaseReinforcementLogger):
    def __init__(
        self,
        configuration: ConfigurationEnvelope,
        rl_config: ReinforcementLoggerConfiguration,
    ):
        super().__init__(configuration, rl_config)
        self._summary_writer = SummaryWriter(log_dir=self._log_config.logging_path)
        self._summary_writer.add_text(
            "Legends",
            "The values under each compound are read as: [Agent; Prior; Target; Score]",
        )
        # _rows and _columns define the shape of the output grid of molecule images in tensorboard.
        self._rows = 4
        self._columns = 4
        self._sample_size = self._rows * self._columns
        self._sf_component_enum = ScoringFunctionComponentNameEnum()
        self._console_message_formatter = ConsoleMessage()

    def log_message(self, message: str):
        # Uses info from logging python standard package
        # prints message to console as debug message
        self._logger.info(message)

    def timestep_report(
        self,
        start_time,
        n_steps,
        step,
        smiles: np.array,
        mean_score: np.float32,
        score_summary: FinalSummary,
        score: np.array,
        loss,
        agent_likelihood: torch.tensor,
        prior_likelihood: torch.tensor,
        augmented_likelihood: torch.tensor,
        diversity_filter: BaseDiversityFilter,
    ):
        message = self._console_message_formatter.create(
            start_time,
            n_steps,
            step,
            smiles,
            mean_score,
            score_summary,
            score,
            loss,
            agent_likelihood,
            prior_likelihood,
            augmented_likelihood,
        )
        self._logger.info(message)
        self._tensorboard_report(
            step,
            smiles,
            score,
            score_summary,
            agent_likelihood,
            prior_likelihood,
            augmented_likelihood,
            diversity_filter,
        )

    def save_final_state(self, agent, scaffold_filter):
        agent.save_to_file(os.path.join(self._log_config.result_folder, "Agent.ckpt"))
        self.save_filter_memory(scaffold_filter)
        self._summary_writer.close()
        self.log_out_input_configuration()

    def save_intermediate_state(self, agent, scaffold_filter):
        agent.save_to_file(os.path.join(self._log_config.result_folder, "Agent.ckpt"))
        self.save_filter_memory(scaffold_filter)

    def save_q_tables_and_probabilities(
        self,
        q_values1: torch.Tensor,
        q_values2: torch.Tensor,
        probs: torch.Tensor,
        seqs: torch.Tensor,
        step: int,
    ):
        "Save two q-tables"
        save_path1 = os.path.join(
            self._log_config.result_folder, f"q_table1_{step}.npy"
        )
        save_path2 = os.path.join(
            self._log_config.result_folder, f"q_table2_{step}.npy"
        )
        save_path_seqs = os.path.join(
            self._log_config.result_folder, f"seqs_{step}.npy"
        )

        save_path_probs = os.path.join(
            self._log_config.result_folder, f"probs_{step}.npy"
        )
        np.save(save_path1, q_values1.detach().cpu().numpy())
        np.save(save_path2, q_values2.detach().cpu().numpy())
        np.save(save_path_seqs, seqs.detach().cpu().numpy())

        np.save(save_path_probs, probs.detach().cpu().numpy())

    def save_q_table_and_probabilities(
        self,
        q_values: torch.Tensor,
        probs: torch.Tensor,
        seqs: torch.Tensor,
        step: int,
    ):
        "Save one q-table"
        save_path_values = os.path.join(
            self._log_config.result_folder, f"q_table_{step}.npy"
        )

        save_path_seqs = os.path.join(
            self._log_config.result_folder, f"seqs_{step}.npy"
        )

        save_path_probs = os.path.join(
            self._log_config.result_folder, f"probs_{step}.npy"
        )
        np.save(save_path_values, q_values.detach().cpu().numpy())
        np.save(save_path_seqs, seqs.detach().cpu().numpy())
        np.save(save_path_probs, probs.detach().cpu().numpy())

    def _tensorboard_report(
        self,
        step,
        smiles,
        score,
        score_summary: FinalSummary,
        agent_likelihood,
        prior_likelihood,
        augmented_likelihood,
        diversity_filter: BaseDiversityFilter,
    ):
        self._summary_writer.add_scalars(
            "nll/avg",
            {
                "prior": prior_likelihood.mean(),
                "augmented": augmented_likelihood.mean(),
                "agent": agent_likelihood.mean(),
            },
            step,
        )
        mean_score = np.mean(score)
        for i, log in enumerate(score_summary.profile):
            self._summary_writer.add_scalar(
                score_summary.profile[i].name,
                np.mean(score_summary.profile[i].score),
                step,
            )
        self._summary_writer.add_scalar(
            "Valid SMILES", ul_rl.fraction_valid_smiles(smiles), step
        )
        self._summary_writer.add_scalar(
            "Number of SMILES found",
            diversity_filter.number_of_smiles_in_memory(),
            step,
        )
        self._summary_writer.add_scalar("Average score", mean_score, step)
        if step % 10 == 0:
            self._log_out_smiles_sample(smiles, score, step, score_summary)

    def _log_out_smiles_sample(self, smiles, score, step, score_summary: FinalSummary):
        self._visualize_structures(smiles, score, step, score_summary)

    def _visualize_structures(self, smiles, score, step, score_summary: FinalSummary):

        (
            list_of_mols,
            legends,
            pattern,
        ) = self._check_for_invalid_mols_and_create_legends(
            smiles, score, score_summary
        )
        try:
            ul_rl.add_mols(
                self._summary_writer,
                "Molecules from epoch",
                list_of_mols[: self._sample_size],
                self._rows,
                [x for x in legends],
                global_step=step,
                size_per_mol=(320, 320),
                pattern=pattern,
            )
        except:
            self.log_message(
                f"Error in RDKit has occurred, skipping report for step {step}."
            )

    def _check_for_invalid_mols_and_create_legends(
        self, smiles, score, score_summary: FinalSummary
    ):
        smiles = ul_rl.padding_with_invalid_smiles(smiles, self._sample_size)
        list_of_mols, legend = ul_rl.check_for_invalid_mols_and_create_legend(
            smiles, score, self._sample_size
        )
        smarts_pattern = self._get_matching_substructure_from_config(score_summary)
        pattern = ul_rl.find_matching_pattern_in_smiles(
            list_of_mols=list_of_mols, smarts_pattern=smarts_pattern
        )

        return list_of_mols, legend, pattern

    def _get_matching_substructure_from_config(self, score_summary: FinalSummary):
        smarts_pattern = ""
        for summary_component in score_summary.scaffold_log:
            if (
                summary_component.parameters.component_type
                == self._sf_component_enum.MATCHING_SUBSTRUCTURE
            ):
                smarts = summary_component.parameters.specific_parameters.get(
                    self._specific_parameters_enum.SMILES, []
                )
                if len(smarts) > 0:
                    smarts_pattern = smarts[0]
        return smarts_pattern
