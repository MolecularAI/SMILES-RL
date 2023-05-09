from dataclasses import dataclass

from smiles_rl.agent.configurations.reinforcement_learning_configuration import (
    ReinforcementLearningConfiguration,
)


@dataclass
class ReinforcementLearningEnvelope:
    method: str
    parameters: ReinforcementLearningConfiguration


@dataclass
class LoggingEnvelope:
    method: str
    parameters: dict


@dataclass
class DiversityFilterEnvelope:
    method: str
    parameters: dict


@dataclass
class ReplayBufferEnvelope:
    method: str
    parameters: dict


@dataclass
class ScoringFunctionEnvelope:
    method: str
    parameters: dict


@dataclass
class ConfigurationEnvelope:
    reinforcement_learning: ReinforcementLearningEnvelope
    logging: LoggingEnvelope
    diversity_filter: DiversityFilterEnvelope
    replay_buffer: ReplayBufferEnvelope
    scoring_function: ScoringFunctionEnvelope
