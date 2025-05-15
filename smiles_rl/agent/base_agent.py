from abc import ABCMeta, abstractmethod

from typing import List

from smiles_rl.configuration_envelope import ConfigurationEnvelope


class BaseAgent(metaclass=ABCMeta):
    """The agent should use the given scoring function,
    diversity filter, replay buffer for update. Logger should be used for saving agent parameters
    and memory for intermediate and/or final inspection.
    """

    @abstractmethod
    def __init__(
        self,
        config: ConfigurationEnvelope,
        scoring_function,
        diversity_filter,
        replay_buffer,
        logger,
    ):
        """Intializes agent.

        Args:
            config (ConfigurationEnvelope): configuration
            scoring_function: scoring function to use
            diversity_filter: diversity filter to use
            replay_buffer: rteplay buffer to use
            logger: logger to use

        Raises:
            NotImplementedError: Method needs to be reimplemented
            in specific agent
        """

        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self._replay_buffer = replay_buffer
        self._logger = logger

    @abstractmethod
    def act(self, batch_size: int) -> List[str]:
        """Sample SMILES strings

        Args:
            batch_size (int): number of SMILES to sample

        Raises:
            NotImplementedError: Method needs to be reimplemented
            in specific agent

        Returns:
            List[str]: sampled SMILES strings
        """
        raise NotImplementedError("Act method is not implemented")

    @abstractmethod
    def update(self, smiles: List[str]) -> None:
        """Update agent given SMILES strings. Agent scores SMILES by using given scoring function and diversity filter

        Args:
            smiles (List[str]): SMILES for update

        Raises:
            NotImplementedError: Method needs to be reimplemented
            in specific agent
        """
        raise NotImplementedError("Update method is not implemented")

    @abstractmethod
    def log_out(
        self,
    ) -> None:
        """Optional method that is called when generation is finished, e.g., for saving
        model parameters and memory of sampled molecules (molecules are not saved otherwise for inspection)
        """
        pass
