from ..configuration_envelope import ConfigurationEnvelope

from .reinforcement_log_configuration import ReinforcementLoggerConfiguration

from .base_reinforcement_logger import BaseReinforcementLogger

from .remote_reinforcement_logger import RemoteReinforcementLogger

from .local_reinforcement_logger import LocalReinforcementLogger


class ReinforcementLogger:
    def __new__(
        cls,
        configuration: ConfigurationEnvelope,
    ) -> BaseReinforcementLogger:

        log_config = ReinforcementLoggerConfiguration.parse_obj(
            configuration.logging.parameters
        )

        if log_config.recipient == "remote":
            logger_instance = RemoteReinforcementLogger(configuration, log_config)
        else:
            logger_instance = LocalReinforcementLogger(configuration, log_config)

        return logger_instance
