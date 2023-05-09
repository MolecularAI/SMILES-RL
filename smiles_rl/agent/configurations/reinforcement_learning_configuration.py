from dataclasses import dataclass


@dataclass
class ReinforcementLearningConfiguration:
    prior: str
    agent: str
    specific_parameters: dict
    n_steps: int = 2000
    learning_rate: float = 0.0001
    batch_size: int = 128
