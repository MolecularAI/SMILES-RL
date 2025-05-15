import argparse
import json

from smiles_rl.configuration_envelope import ConfigurationEnvelope


from typing import Optional, Dict, Type, Any


from smiles_rl.utils.general import _set_torch_device


import importlib

from smiles_rl.agent.base_agent import BaseAgent

from dacite import from_dict


def load_dynamic_class(
    name_spec: str,
    default_module: Optional[str] = None,
    exception_cls: Type[Exception] = ValueError,
):

    if name_spec is None:
        raise KeyError(f"Key method not found in scoring function config")

    if "." not in name_spec:
        name = name_spec
        if not default_module:
            raise exception_cls(
                "Must provide default_module argument if not given in name_spec"
            )
        module_name = default_module
    else:
        module_name, name = name_spec.rsplit(".", maxsplit=1)
    try:
        loaded_module = importlib.import_module(module_name)
    except ImportError:
        raise exception_cls(f"Unable to load module: {module_name}")

    if not hasattr(loaded_module, name):
        raise exception_cls(
            f"Module ({module_name}) does not have a class called {name}"
        )

    return getattr(loaded_module, name)


def run(
    config: ConfigurationEnvelope,
    agent,
) -> None:
    print("Starting run", flush=True)

    batch_size = config.reinforcement_learning.parameters.batch_size
    n_steps = config.reinforcement_learning.parameters.n_steps
    for _ in range(n_steps):
        _run(batch_size, agent)

    agent.log_out()


def _run(
    batch_size: int,
    agent: BaseAgent,
) -> None:

    smiles = agent.act(batch_size)

    assert len(smiles) <= batch_size, "Generated more SMILES strings than requested"

    agent.update(smiles)


def _read_json_file(path: str) -> Dict[str, Any]:
    """Reads json config file

    Args:
        path (str): Path to json file with configuration

    Returns:
        dict: Dictionary containing configurations from json file
    """

    print("Rading JSON file", flush=True)
    with open(path) as f:
        json_input = f.read().replace("\r", "").replace("\n", "")
    try:
        config = json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")

    return config


def _construct_logger(config: ConfigurationEnvelope):
    """Creates logger instance

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        logger instance
    """

    name_spec = config.logging.method

    if name_spec is not None:
        method_class = load_dynamic_class(name_spec)
    else:
        raise KeyError(f"Key method not found in logging config")

    logger = method_class(config)

    return logger


def _construct_scoring_function(config: ConfigurationEnvelope):
    """Creates scoring function instance

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        scoring function instance
    """

    name_spec = config.scoring_function.method

    method_class = load_dynamic_class(name_spec)

    scoring_function = method_class(config)

    return scoring_function


def _construct_agent(
    config: ConfigurationEnvelope,
    logger,
    scoring_function,
    diversity_filter,
    replay_buffer,
) -> BaseAgent:
    """Creates agent

    Args:
        config (ConfigurationEnvelope): _description_
        logger: logger instance
        scoring_function: scoring function instance
        diversity_filter: diversity filter instance
        replay_buffer: replay buffer instance

    Returns:
        BaseAgent: agent
    """
    name_spec = config.reinforcement_learning.method

    method_class = load_dynamic_class(name_spec)

    agent = method_class(
        config,
        scoring_function,
        diversity_filter,
        replay_buffer,
        logger,
    )

    return agent


def _construct_diversity_filter(config: ConfigurationEnvelope):
    """Creates diversity filter instance

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        diversity filter instance
    """
    name_spec = config.diversity_filter.method

    method_class = load_dynamic_class(name_spec)

    diversity_filter = method_class(config)

    return diversity_filter


def _construct_replay_buffer(config: ConfigurationEnvelope):
    """Create replay buffer instance

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        replay buffer instance
    """
    name_spec = config.replay_buffer.method

    method_class = load_dynamic_class(name_spec)

    replay_buffer = method_class(config.replay_buffer.parameters)

    return replay_buffer


def _construct_run(config: ConfigurationEnvelope) -> BaseAgent:
    """Construct run and returns agent

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        BaseAgent: agent
    """

    # Set default device of pytorch tensors to cuda
    device = _set_torch_device("cuda")

    logger = _construct_logger(config)
    scoring_function = _construct_scoring_function(config)

    diversity_filter = _construct_diversity_filter(config)

    replay_buffer = _construct_replay_buffer(config)

    agent = _construct_agent(
        config, logger, scoring_function, diversity_filter, replay_buffer
    )

    return agent


def _get_arguments() -> argparse.Namespace:
    """Reads command-line arguments

    Returns:
        argparse.Namespace: command-line arguments
    """

    print("Getting input args", flush=True)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="path to config json file",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    print("Starting main", flush=True)

    # Get command line arguments
    args = _get_arguments()

    # Read configuration file
    config_json = _read_json_file(args.config)

    # Create envelope of configuration
    config = from_dict(data_class=ConfigurationEnvelope, data=config_json)

    # Construct run
    agent = _construct_run(config)

    # Run generation of SMILES strings using RNN
    run(config, agent)


if __name__ == "__main__":
    main()
