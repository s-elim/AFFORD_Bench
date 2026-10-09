import sys

# Provide backward-compatible gym shim if legacy gym is not installed
try:
    import gym  # noqa: F401
except ImportError:
    try:
        import gymnasium as _gym
        sys.modules["gym"] = _gym
    except ImportError:
        pass

from afford_pro.registry import make, list_tasks, list_frontier_tasks, list_all_tasks, get_oracle, get_task_entry, get_initial_states
from afford_pro.envs.base_franka_env import BaseFrankaEnv
from afford_pro.perturbations.engine import PerturbationEngine, PerturbationConfig
from afford_pro.datasets.hdf5_writer import HDF5DatasetWriter

__version__ = "0.1.0"
__all__ = [
    "make",
    "list_tasks",
    "list_frontier_tasks",
    "list_all_tasks",
    "get_oracle",
    "get_task_entry",
    "get_initial_states",
    "BaseFrankaEnv",
    "PerturbationEngine",
    "PerturbationConfig",
    "HDF5DatasetWriter",
]
