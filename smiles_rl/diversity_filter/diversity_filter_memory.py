from typing import List, Dict

import pandas as pd
import numpy as np

from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.enums.scoring_function_component_enum import (
    ScoringFunctionComponentNameEnum,
)


class DiversityFilterMemory:

    def __init__(self):
        self._sf_component_name = ScoringFunctionComponentNameEnum()
        self.df_dict = {
            "Step": [],
            "Scaffold ID": [],
            "Scaffold": [],
            "SMILES": [],
            "Canonical SMILES": [],
        }
        self._memory_dataframe = pd.DataFrame(self.df_dict)

        self.scaffold_id = 0

    def update(
        self,
        indx: int,
        score: float,
        canon_smile: str,
        smile: str,
        scaffold: str,
        components: List[ComponentSummary],
        step: int,
    ):
        component_scores = {
            c.parameters.name: float(c.total_score[indx]) for c in components
        }
        component_scores = self._include_raw_score(indx, component_scores, components)
        component_scores[self._sf_component_name.TOTAL_SCORE] = float(score)
        if not self.smiles_exists(canon_smile):
            self._add_to_memory_dataframe(
                step, canon_smile, smile, scaffold, component_scores
            )

    def _add_to_memory_dataframe(
        self,
        step: int,
        canon_smile: str,
        smile: str,
        scaffold: str,
        component_scores: Dict,
    ):
        data = []
        headers = []
        for name, score in component_scores.items():
            headers.append(name)
            data.append(score)
        headers.append("Step")
        data.append(int(step))

        headers.append("Scaffold ID")

        if scaffold in self._memory_dataframe["Scaffold"].values:
            scaffold_id = self._memory_dataframe[
                self._memory_dataframe["Scaffold"] == scaffold
            ]["Scaffold ID"].unique()

            assert (
                len(scaffold_id) == 1
            ), f"{scaffold_id} has incorrect length {len(scaffold_id)}"

            assert isinstance(scaffold_id, np.ndarray)

            scaffold_id = scaffold_id[0]

        else:
            self.scaffold_id += 1
            scaffold_id = self.scaffold_id

        data.append(int(scaffold_id))

        headers.append("Scaffold")
        data.append(scaffold)

        headers.append("Canonical SMILES")
        data.append(canon_smile)
        headers.append("SMILES")
        data.append(smile)
        new_data = pd.DataFrame([data], columns=headers)
        self._memory_dataframe = pd.concat(
            [self._memory_dataframe, new_data], ignore_index=True, sort=False
        )

    def sample_smiles(self, size: int) -> List[str]:
        n_smiles = self.number_of_smiles()

        if n_smiles == 0:
            return []

        size = min(n_smiles, size)

        random_subset = self._memory_dataframe["Canonical SMILES"].sample(
            size, replace=False
        )

        return random_subset.to_list()

    def get_memory(self) -> pd.DataFrame:
        return self._memory_dataframe

    def set_memory(self, memory: pd.DataFrame) -> None:
        self._memory_dataframe = memory

    def smiles_exists(self, canon_smiles: str) -> bool:
        if len(self._memory_dataframe) == 0:
            return False
        return canon_smiles in self._memory_dataframe["Canonical SMILES"].values

    def scaffold_instances_count(self, scaffold: str) -> int:
        return (self._memory_dataframe["Scaffold"].values == scaffold).sum()

    def number_of_scaffolds(self) -> int:
        return len(set(self._memory_dataframe["Scaffold"].values))

    def number_of_smiles(self) -> int:
        return len(set(self._memory_dataframe["Canonical SMILES"].values))

    def get_scaffold(self, scaffold: str) -> pd.DataFrame:
        if scaffold not in self._memory_dataframe["Scaffold"].values:
            return pd.DataFrame(self.df_dict)

        scaffold_mem = self._memory_dataframe[
            self._memory_dataframe["Scaffold"] == scaffold
        ]

        return scaffold_mem

    def get_scaffold_id(self, scaffold: str) -> int:
        if scaffold not in self._memory_dataframe["Scaffold"].values:

            return None

        scaffold_id = self._memory_dataframe[
            self._memory_dataframe["Scaffold"] == scaffold
        ]["Scaffold ID"].unique()

        assert (
            len(scaffold_id) == 1
        ), f"{scaffold_id} has incorrect length {len(scaffold_id)}"

        assert isinstance(scaffold_id, np.ndarray)

        return scaffold_id[0]

    def _include_raw_score(
        self, indx: int, component_scores: dict, components: List[ComponentSummary]
    ):
        raw_scores = {
            f"raw_{c.parameters.name}": float(c.raw_score[indx])
            for c in components
            if c.raw_score is not None
        }
        all_scores = {**component_scores, **raw_scores}
        return all_scores
