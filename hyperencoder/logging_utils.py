import logging
from typing import Union
from pathlib import Path

import colorlog
from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def initialize_logger(log_dir: Union[str, Path], experiment_id: str):
    """
    Initializes a logger that logs to both the console and separate log files for stdout and errors.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{experiment_id}_training.log"
    error_log_file = log_dir / f"{experiment_id}_errors.log"

    logger = colorlog.getLogger()
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    error_file_handler = logging.FileHandler(error_log_file)
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(file_formatter)

    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    console_handler = TqdmHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)

    return logger
