""" 
Sets up the logger config.
Loads env variables from any .env files in the root directory.
Further sets the default values for the global variables.
Loads all needed environment variables from the .env file to importable python VARIABLES.
"""

import logging
import os
from pathlib import Path
from typing import Callable, Type

from dotenv import load_dotenv


# Set up the default logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()])

# load potential environment variables from .env file
env_path = Path(__file__).parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path.resolve(), override=True, verbose=True)

class LazyEnv:
    """
    Lazy environment variable, made to support loading from .env file or potentially setting the default value.
    """

    def __init__(
        self,
        env_var: str,
        default = None,
        return_type: Type = str,
        after_eval: Callable = None,
    ):
        """Construct lazy evaluated environment variable."""
        self.env_var = env_var
        self.default = default
        self.return_type = return_type
        self.after_eval = after_eval

    def eval(self):
        """Evaluate environment variable."""
        value = self.return_type(os.environ.get(self.env_var, self.default))

        if self.after_eval:
            self.after_eval(value)

        return value

# path processed dataset
PATH_ROOT = Path(__file__).parents[1]

DATASETDIR = LazyEnv(
    "DATASET_DIR",
    PATH_ROOT / Path("data"),
    return_type = Path,
).eval()

OUTPUTDIR = Path(str(PATH_ROOT)+"/outputs/")

DATASET_DIR = LazyEnv(
    "DATASET_DIR",
    PATH_ROOT / Path("data"),
    return_type = Path,
).eval()

WANDB_PROJECT = "project"

WANDB_MODE = LazyEnv(
    "WANDB_MODE",
    "online",
    return_type=str,
).eval()

