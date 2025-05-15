from smiles_rl.configuration_envelope import ConfigurationEnvelope

from .diversity_filter import (
    DiversityFilter,
)

from .diversity_filter_parameters import (
    DiversityFilterParameters,
)

from dacite import from_dict


class DiversityFilterFactory:
    """Wrapper for Diversity filters
    based on the diversity filter from the reinvent-scoring package (https://github.com/MolecularAI/reinvent-scoring)
    """

    def __new__(cls, config: ConfigurationEnvelope):
        """Returns a instance of the diversity filter, given provided parameters"""
        filter_parameters = from_dict(
            data_class=DiversityFilterParameters,
            data=config.diversity_filter.parameters,
        )

        diversity_filter = DiversityFilter(filter_parameters)

        return diversity_filter
