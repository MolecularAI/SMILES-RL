from reinvent_scoring.scoring import ScoringFunctionFactory

from smiles_rl.configuration_envelope import ConfigurationEnvelope


from reinvent_scoring.scoring.scoring_function_parameters import (
    ScoringFunctionParameters,
)

from dacite import from_dict


class ReinventScoringFactory:
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

        scoring_function = ScoringFunctionFactory(scoring_parameters)

        return scoring_function
