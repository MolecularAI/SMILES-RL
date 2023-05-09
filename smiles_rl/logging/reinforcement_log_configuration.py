from .base_log_config import BaseLoggerConfiguration


class ReinforcementLoggerConfiguration(BaseLoggerConfiguration):
    result_folder: str
    logging_frequency: int = 0
