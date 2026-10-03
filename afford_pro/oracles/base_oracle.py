"""Base Algorithmic Demonstration Oracle for Franka Environments."""
from __future__ import annotations

from typing import Any
import numpy as np


class BaseOracle:
    """Abstract base class for 100% local, simulation-native demonstration oracles."""

    def __init__(self, env: Any):
        self.env = env

    def generate_episode(self) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[float], list[bool]]:
        """Generate a complete successful episode rollout."""
        raise NotImplementedError
