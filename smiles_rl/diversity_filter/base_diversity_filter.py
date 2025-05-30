import abc
from typing import List, Union
from xmlrpc.client import Boolean

import numpy as np
import pandas as pd

from .diversity_filter_memory import (
    DiversityFilterMemory,
)
from .diversity_filter_parameters import (
    DiversityFilterParameters,
)
from reinvent_scoring.scoring.score_summary import FinalSummary, ComponentSummary
from reinvent_chemistry.conversions import Conversions


class BaseDiversityFilter(abc.ABC):

    @abc.abstractmethod
    def __init__(self, parameters: DiversityFilterParameters):
        self.parameters = parameters
        self._diversity_filter_memory = DiversityFilterMemory()
        self._chemistry = Conversions()

    @abc.abstractmethod
    def update_score(self, score_summary: FinalSummary, step=0) -> np.ndarray:
        raise NotImplementedError("The method 'update_score' is not implemented!")

    def get_memory_as_dataframe(self) -> pd.DataFrame:
        return self._diversity_filter_memory.get_memory()

    def set_memory_from_dataframe(self, memory: pd.DataFrame):
        self._diversity_filter_memory.set_memory(memory)

    def number_of_smiles_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_smiles()

    def number_of_scaffold_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_scaffolds()

    def number_of_smiles(self) -> int:
        return self._diversity_filter_memory.number_of_smiles()

    def get_scaffold(self, scaffold: str) -> pd.DataFrame:
        return self._diversity_filter_memory.get_scaffold(scaffold)

    def get_scaffold_id(self, scaffold: str) -> Union[int, None]:
        return self._diversity_filter_memory.get_scaffold_id(scaffold)

    def update_bucket_size(self, bucket_size: int) -> None:
        self.parameters.bucket_size = bucket_size

    def sample_smiles(self, size: int) -> List[str]:
        return self._diversity_filter_memory.sample_smiles(size)

    def _calculate_scaffold(self, smile):
        raise NotImplementedError

    def _smiles_exists(self, smile) -> bool:
        return self._diversity_filter_memory.smiles_exists(smile)

    def _add_to_memory(
        self,
        indx: int,
        score,
        canon_smile,
        smile,
        scaffold,
        components: List[ComponentSummary],
        step,
    ) -> None:
        self._diversity_filter_memory.update(
            indx, score, canon_smile, smile, scaffold, components, step
        )

    def _penalize_score(self, scaffold, score):
        """Penalizes the score if the scaffold bucket is full"""
        if (
            self._diversity_filter_memory.scaffold_instances_count(scaffold)
            > self.parameters.bucket_size
        ):
            score = 0.0
        return score
