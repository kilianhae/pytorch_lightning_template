"""LEAP code config."""
import logging
import os
from pathlib import Path
from typing import Callable, Type
import logging

from dotenv import load_dotenv


# Set up the default logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()])


env_path = Path(__file__).parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path.resolve(), override=True, verbose=True)


class LazyEnv:
    """Lazy environment variable."""

    def __init__(
        self,
        env_var: str,
        default=None,
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
    PATH_ROOT / Path("data1/sim_data"),
    return_type=Path,
).eval()

OUTPUTDIR = Path(str(PATH_ROOT)+"/outputs")

CONDITIONAL_2D_DATASET_DIR = Path(str(PATH_ROOT)+"/data1/LEAP_2023_task1_2D")

WANDB_PROJECT = "sdsc-leap"
WANDB_MODE = LazyEnv(
    "WANDB_MODE",
    "online",
    return_type=str,
).eval()

Npc = 64

Tmel = 1905.5

Tpc = 0.3
