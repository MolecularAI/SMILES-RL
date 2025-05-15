import os
import json
import argparse
from utils.logging import new_version_dir


def main():

    # Get arguments
    args = _get_arguments()

    log_dir = new_version_dir(args.log_dir)

    prior_model_path = args.prior

    # Create json config file
    config_json = _create_json_config(
        prior_model_path,
        log_dir,
        replay_buffer=args.replay_buffer,
        scoring_function=args.scoring_function,
        diversity_filter=args.diversity_filter,
    )

    # Final save path json config file
    save_path_config = os.path.join(log_dir, "regularized_mle_novelty_config.json")

    # Save json file
    with open(save_path_config, "w") as f:
        json.dump(config_json, f, indent=4, sort_keys=True)

    # Print log directory
    print(log_dir)


def _get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replay_buffer",
        required=True,
        type=str,
        help="Absolute/relative path to replay buffer for import",
    )
    parser.add_argument(
        "--log_dir",
        required=True,
        type=str,
        help="Logging directory for saving config file",
    )

    parser.add_argument(
        "--prior",
        required=True,
        type=str,
        help="Path to pre-trained model",
    )

    parser.add_argument(
        "--diversity_filter",
        required=True,
        type=str,
        help="Diversity filter to use (intrinsic reward and/or penalty of extrinsic reward",
    )

    parser.add_argument(
        "--scoring_function",
        required=True,
        type=str,
        help="Oracle to use for extrinsic reward",
    )

    args = parser.parse_args()

    return args


def _create_json_config(
    prior_model_path,
    output_dir,
    replay_buffer: str,
    diversity_filter: str,
    scoring_function: str,
    n_steps: int = 2000,
    batch_size: int = 128,
):

    results_dir = os.path.join(output_dir, "results")

    configuration = {}

    configuration["logging"] = {
        "method": "smiles_rl.logging.reinforcement_logger.ReinforcementLogger",
        "parameters": {
            "sender": "http://127.0.0.1",  # only relevant if "recipient" is set to "remote"
            "recipient": "local",  # either to local logging or use a remote REST-interface
            "logging_frequency": 0,  # log every x-th steps
            "logging_path": os.path.join(
                output_dir, "progress_log"
            ),  # load this folder in tensorboard
            "result_folder": results_dir,  # will hold the compounds (SMILES) and summaries
            "job_name": "Regularized MLE demo",  # set an arbitrary job name for identification
            "job_id": "demo",  # only relevant if "recipient" is set to "remote"
        },
    }

    if diversity_filter == "IdenticalMurckoScaffold":
        # add a "diversity_filter"
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "IdenticalMurckoScaffold",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "bucket_size": 25,  # the bin size; penalization will start once this is exceeded
                "minscore": 0.5,  # the minimum total score to be considered for binning
            },
        }
    elif diversity_filter == "DiverseHits":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "DiverseHits",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "minsimilarity": 0.3,  # the minimum similarity to be placed into the same bin
            },
        }

    elif diversity_filter == "MinSimilarity":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "MinSimilarity",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "minsimilarity": 0.3,  # the minimum similarity to be placed into the same bin
            },
        }

    elif diversity_filter == "MeanSimilarity":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "MeanSimilarity",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "minsimilarity": 0.3,  # the minimum similarity to be placed into the same bin
            },
        }

    elif diversity_filter == "MinSimilarityRandom":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "MinSimilarityRandom",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
            },
        }

    elif diversity_filter == "MeanSimilarityRandom":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "MeanSimilarityRandom",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
            },
        }

    elif diversity_filter == "UCBMurckoScaffold":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "UCBMurckoScaffold",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "minsimilarity": 0.3,  # the minimum similarity to be placed into the same bin
            },
        }
    elif diversity_filter == "ErfIdenticalMurckoScaffold":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "SoftIdenticalMurckoScaffold",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "bucket_size": 25,
                "soft_function": "erf",
            },
        }

    elif diversity_filter == "TanhIdenticalMurckoScaffold":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "SoftIdenticalMurckoScaffold",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "bucket_size": 25,
                "soft_function": "tanh",
            },
        }
    elif diversity_filter == "LinearIdenticalMurckoScaffold":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "SoftIdenticalMurckoScaffold",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "bucket_size": 25,
                "soft_function": "linear",
            },
        }

    elif diversity_filter == "SigmoidIdenticalMurckoScaffold":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "SoftIdenticalMurckoScaffold",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "bucket_size": 25,
                "soft_function": "sigmoid",
            },
        }
    elif diversity_filter == "RND":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "RND",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "bucket_size": 25,
            },
        }
    elif diversity_filter == "SoftRND":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "SoftRND",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "bucket_size": 25,
                "soft_funcion": "tanh",
            },
        }

    elif diversity_filter == "SoftInformation":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "SoftInformation",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
                "soft_funcion": "tanh",
            },
        }

    elif diversity_filter == "Information":
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "Information",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "minscore": 0.5,  # the minimum total score to be considered for binning
            },
        }

    else:
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "NoFilter",
                "minscore": 0.5,
            },
        }

    # set all "reinforcement learning"-specific run parameters
    configuration["reinforcement_learning"] = {
        "method": "smiles_rl.agent.regularized_mle_novelty.RegularizedMLENovelty",
        "parameters": {
            "prior": prior_model_path,  # path to the pre-trained model
            "agent": prior_model_path,  # path to a second pre-trained model
            "n_steps": n_steps,  # the number of epochs (steps) to be performed
            "learning_rate": 0.0001,  # sets how strongly the agent is influenced by each epoch
            "batch_size": batch_size,  # specifies how many molecules are generated per epoch
            "specific_parameters": {
                "sigma": 128,  # used to calculate the "augmented likelihood", see publication
                "margin_threshold": 50,  # specify the (positive) margin between agent and prior
            },
        },
    }

    configuration["replay_buffer"] = {
        "method": replay_buffer,
        "parameters": {
            "k": batch_size // 2,
            "memory_size": 1000,
        },
    }

    # prepare the scoring function definition and add at the end

    configuration["scoring_function"] = {
        "method": "smiles_rl.scoring.tdc_scoring_factory.TDCScoringFactory",
        "parameters": {
            "name": scoring_function.lower(),  # Options: drd2, gsk3b, jnk3
            "parameters": {
                "component_type": "predictive_property",  # this is a scikit-learn model, returning activity values
                "name": scoring_function.upper(),  # arbitrary name for the component
                "weight": 1,  # the weight ("importance") of the component (default: 1)
            },
        },
    }

    return configuration


if __name__ == "__main__":
    main()
