import os
import json
import argparse


def main():

    # Get arguments
    args = _get_arguments()

    log_dir = args.log_dir

    prior_model_path = args.prior

    predictive_model_path = args.predictive_model

    # Create json config file
    config_json = _create_json_config(
        prior_model_path,
        log_dir,
        predictive_model_path,
        replay_buffer=args.replay_buffer,
        use_diversity_filter=True,
    )

    # Final save path json config file
    save_path_config = os.path.join(log_dir, "ppo_config.json")

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
        "--predictive_model",
        required=True,
        type=str,
        help="Path to predictive model",
    )

    args = parser.parse_args()

    return args


def _create_json_config(
    prior_model_path,
    output_dir,
    model_path,
    replay_buffer: str,
    use_diversity_filter: bool = True,
    n_steps: int = 2000,
    batch_size: int = 128,
    activity_model: str = "classification",
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
            "job_name": "ppo demo",  # set an arbitrary job name for identification
            "job_id": "demo",  # only relevant if "recipient" is set to "remote"
        },
    }

    if use_diversity_filter:
        # add a "diversity_filter"
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "IdenticalMurckoScaffold",  # other options are: "IdenticalTopologicalScaffold",
                #                    "NoFilter" and "ScaffoldSimilarity"
                # -> use "NoFilter" to disable this feature
                "bucket_size": 25,  # the bin size; penalization will start once this is exceeded
                "minscore": 0.4,  # the minimum total score to be considered for binning
                "minsimilarity": 0.35,  # the minimum similarity to be placed into the same bin
            },
        }
    else:
        configuration["diversity_filter"] = {
            "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
            "parameters": {
                "name": "NoFilter",
            },
        }

    # set all "reinforcement learning"-specific run parameters
    configuration["reinforcement_learning"] = {
        "method": "smiles_rl.agent.ppo.PPO",
        "parameters": {
            "prior": prior_model_path,  # path to the pre-trained model
            "agent": prior_model_path,  # path to a second pre-trained model
            "n_steps": n_steps,  # the number of epochs (steps) to be performed
            "learning_rate": 0.0001,  # sets how strongly the agent is influenced by each epoch
            "batch_size": batch_size,  # specifies how many molecules are generated per epoch
            "specific_parameters": {
                "clip": 0.2,
                "n_updates_per_iteration": 4,
                "discount_factor": 0.99,
                "use_entropy_bonus": False,  # Wether to add entropy bonus to actor loss
                "entropy_coeff": 0.001,  # Entropy bonus coefficient
                "clip_grad_norm": True,  # Wether to clip global gradient norms
                "max_grad_norm": 0.5,  # Max gradient norm
                "n_minibatches": 4,  # Number of trajectories in each minibatch
            },
        },
    }

    # Define replay buffer
    configuration["replay_buffer"] = {
        "method": replay_buffer,
        "parameters": {
            "k": batch_size // 2,
            "memory_size": 1000,
        },
    }

    # prepare the scoring function definition and add at the end
    configuration["scoring_function"] = {
        "method": "smiles_rl.scoring.reinvent_scoring_factory.ReinventScoringFactory",
        "parameters": {
            "name": "custom_sum",  # this is our default one (alternative: "custom_sum")
            "parallel": False,  # sets whether components are to be executed
            # in parallel; note, that python uses "False" / "True"
            # but the JSON "false" / "true"
            # the "parameters" list holds the individual components
            "parameters": [
                # add component: an activity model
                {
                    "component_type": "predictive_property",  # this is a scikit-learn model, returning activity values
                    "name": activity_model,  # arbitrary name for the component
                    "weight": 1,  # the weight ("importance") of the component (default: 1)
                    "specific_parameters": {
                        "model_path": model_path,  # absolute model path
                        "container_type": "scikit_container",
                        "uncertainty_type": None,  # Available for probabilistic container, default: None
                        "scikit": activity_model,  # model can be "regression" or "classification"
                        "transformation": {"transformation_type": "no_transformation"},
                        "descriptor_type": "ecfp_counts",  # sets the input descriptor for this model
                        "size": 2048,  # parameter of descriptor type
                        "radius": 2,  # parameter of descriptor type
                        "use_counts": True,  # parameter of descriptor type
                        "use_features": True,  # parameter of descriptor type
                    },
                },
            ],
        },
    }

    return configuration


if __name__ == "__main__":
    main()
