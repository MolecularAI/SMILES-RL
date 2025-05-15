from smiles_rl.configuration_envelope import ConfigurationEnvelope

from dacite import from_dict

from typing import List

from tdc import Oracle

from rdkit import Chem

import numpy as np

from abc import ABC, abstractmethod

from dataclasses import dataclass


@dataclass
class ComponentParameters:
    component_type: str
    name: str
    weight: float
    specific_parameters: dict = None


@dataclass
class ComponentSummary:
    total_score: np.array
    parameters: ComponentParameters
    raw_score: np.ndarray = None


class FinalSummary:
    def __init__(
        self,
        total_score: np.ndarray,
        scored_smiles: List[str],
        valid_idxs: List[int],
        scaffold_log: List[ComponentSummary],
    ):
        self.total_score = total_score
        self.scored_smiles = scored_smiles
        self.valid_idxs = valid_idxs
        self.scaffold_log = scaffold_log


class BaseTDCScoringFunction(ABC):

    def _get_valid_and_invalid_smiles(self, smiles_lst: List[str]):
        nonvalid_smiles_idx_lst, valid_smiles_idx_lst, valid_smiles_lst = [], [], []
        for idx, smiles in enumerate(smiles_lst):
            if Chem.MolFromSmiles(smiles) == None:
                nonvalid_smiles_idx_lst.append(idx)
            else:
                valid_smiles_idx_lst.append(idx)
                valid_smiles_lst.append(smiles)

        return nonvalid_smiles_idx_lst, valid_smiles_idx_lst, valid_smiles_lst

    @abstractmethod
    def get_final_score_for_step(self, smiles_lst: List[str], step: int):
        raise NotImplementedError("get_final_score_for_step method is not implemented")

    def _create_final_summary(
        self,
        smiles_lst: List[str],
        valid_smiles_idx_lst: List[int],
        score_valid_smiles_lst: List[float],
    ):

        score_arr = np.zeros((len(smiles_lst),))

        # Invalid SMILES keeps a score of 0 (maybe changed by agent for update)
        for idx, valid_smiles_idx in enumerate(valid_smiles_idx_lst):
            score_arr[valid_smiles_idx] = score_valid_smiles_lst[idx]

        c = ComponentParameters(
            self._scoring_parameters["component_type"],
            self._scoring_parameters["name"],
            self._scoring_parameters["weight"],
        )

        scaffold_log = [ComponentSummary(score_arr, c)]

        return FinalSummary(
            score_arr,
            smiles_lst,
            valid_smiles_idx_lst,
            scaffold_log,
        )


@dataclass
class ScoringFunctionParameters:
    name: str
    parameters: dict


class TDCScoringFactory:
    def __new__(cls, config: ConfigurationEnvelope):
        """Creates and return reinvent scoring function.

        Args:
            config (ConfigurationEnvelope): configurations

        Returns:
            : scoring function
        """

        scoring_parameters = from_dict(
            data_class=ScoringFunctionParameters,
            data=config.scoring_function.parameters,
        )

        scoring_parameters.name = scoring_parameters.name.lower()

        if scoring_parameters.name == "drd2":
            scoring_function = DRD2Scoring(scoring_parameters.parameters)
        elif scoring_parameters.name == "jnk3":
            scoring_function = JNK3Scoring(scoring_parameters.parameters)
        elif scoring_parameters.name == "gsk3b":
            scoring_function = GSK3BScoring(scoring_parameters.parameters)
        else:
            raise NotImplementedError(
                f"{scoring_parameters.name} TDC scoring function is not implemented"
            )

        return scoring_function


class DRD2Scoring(BaseTDCScoringFunction):
    def __init__(self, scoring_parameters) -> FinalSummary:
        self._oracle = Oracle(name="DRD2")

        self._scoring_parameters = scoring_parameters

    def get_final_score_for_step(self, smiles_lst: List[str], step: int):

        nonvalid_smiles_idx_lst, valid_smiles_idx_lst, valid_smiles_lst = (
            self._get_valid_and_invalid_smiles(smiles_lst)
        )

        # TDC oracle only returns scores for valid SMILES, therefore we only pass the valid SMILES
        score = self._oracle(valid_smiles_lst)

        assert len(score) == len(
            valid_smiles_lst
        ), f"Got {len(score)} scores but provided {len(valid_smiles_lst)} valid SMILES"

        final_summary = self._create_final_summary(
            smiles_lst, valid_smiles_idx_lst, score
        )

        return final_summary


class JNK3Scoring(BaseTDCScoringFunction):
    def __init__(self, scoring_parameters) -> FinalSummary:
        self._oracle = Oracle(name="JNK3")

        self._scoring_parameters = scoring_parameters

    def get_final_score_for_step(self, smiles_lst: List[str], step: int):

        nonvalid_smiles_idx_lst, valid_smiles_idx_lst, valid_smiles_lst = (
            self._get_valid_and_invalid_smiles(smiles_lst)
        )

        # TDC oracle only returns scores for valid SMILES, therefore we only pass the valid SMILES
        score = self._oracle(valid_smiles_lst)

        assert len(score) == len(
            valid_smiles_lst
        ), f"Got {len(score)} scores but provided {len(valid_smiles_lst)} valid SMILES"

        final_summary = self._create_final_summary(
            smiles_lst, valid_smiles_idx_lst, score
        )

        return final_summary


class GSK3BScoring(BaseTDCScoringFunction):
    def __init__(self, scoring_parameters) -> FinalSummary:
        self._oracle = Oracle(name="GSK3B")

        self._scoring_parameters = scoring_parameters

    def get_final_score_for_step(self, smiles_lst: List[str], step: int):

        nonvalid_smiles_idx_lst, valid_smiles_idx_lst, valid_smiles_lst = (
            self._get_valid_and_invalid_smiles(smiles_lst)
        )

        # TDC oracle only returns scores for valid SMILES, therefore we only pass the valid SMILES
        score = self._oracle(valid_smiles_lst)

        assert len(score) == len(
            valid_smiles_lst
        ), f"Got {len(score)} scores but provided {len(valid_smiles_lst)} valid SMILES"

        final_summary = self._create_final_summary(
            smiles_lst, valid_smiles_idx_lst, score
        )

        return final_summary
