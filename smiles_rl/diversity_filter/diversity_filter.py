from smiles_rl.diversity_filter.diverse_hits import DiverseHits
from smiles_rl.diversity_filter.mean_similarity import MeanSimilarity
from smiles_rl.diversity_filter.min_similarity import MinSimilarity
from smiles_rl.diversity_filter.ucb_murcko_scaffold import UCBMurckoScaffold
from .identical_murcko_scaffold import (
    IdenticalMurckoScaffold,
)

from .no_scaffold_filter import NoScaffoldFilter


from .diverse_hits import DiverseHits

from .min_similarity import MinSimilarity

from .mean_similarity import MeanSimilarity

from .min_similarity_random import MinSimilarityRandom

from .mean_similarity_random import MeanSimilarityRandom

from .ucb_murcko_scaffold import UCBMurckoScaffold

from .soft_identical_murcko_scaffold import SoftIdenticalMurckoScaffold

from .rnd import RND

from .soft_rnd import SoftRND

from .information import Information

from .soft_information import SoftInformation

from .base_diversity_filter import (
    BaseDiversityFilter,
)
from .diversity_filter_parameters import (
    DiversityFilterParameters,
)


class DiversityFilter:

    def __new__(cls, parameters: DiversityFilterParameters) -> BaseDiversityFilter:
        all_filters = dict(
            IdenticalMurckoScaffold=IdenticalMurckoScaffold,
            NoFilter=NoScaffoldFilter,
            DiverseHits=DiverseHits,
            MinSimilarity=MinSimilarity,
            MeanSimilarity=MeanSimilarity,
            UCBMurckoScaffold=UCBMurckoScaffold,
            SoftIdenticalMurckoScaffold=SoftIdenticalMurckoScaffold,
            RND=RND,
            MinSimilarityRandom=MinSimilarityRandom,
            MeanSimilarityRandom=MeanSimilarityRandom,
            Information=Information,
            SoftRND=SoftRND,
            SoftInformation=SoftInformation,
        )
        div_filter = all_filters.get(
            parameters.name, KeyError(f"Invalid filter name: `{parameters.name}'")
        )
        return div_filter(parameters)
