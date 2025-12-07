import logging
import os
from typing import Optional


def setup_logger(name: str = "hb", log_file: Optional[str] = None) -> logging.Logger:
    """
    Create a logger that logs to console and optionally to file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


