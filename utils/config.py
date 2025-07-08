import yaml
import logging
import colorlog
import os

def load_config(path: str="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    
    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")

def output_dir(config: dict):
    for key in ["csv", "video"]:
        path = config["output"].get(key)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

def log_setup(verbosity: int=1):
    level_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    logging.root.setLevel(level_map[verbosity])

    if not logging.root.handlers:
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(fmt="%(log_color)s%(levelname)-8s: %(message)s%(reset_color)s"))
        logging.root.addHandler(handler)