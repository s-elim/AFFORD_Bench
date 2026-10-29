"""Run environments without OpenGL: no renderers are created and observations carry only low-dim state.

Used by validation, sanity baselines, and replay checks, which need physics but not images. The patch is
scoped to the context and restored on exit, so image-producing code in the same process is unaffected afterwards.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import afford_pro.envs.base_franka_env as base_env


class _NoRenderer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def _lowdim_observation(self: base_env.BaseFrankaEnv) -> dict[str, Any]:
    return {
        "tcp_pose": self.tcp_pose,
        "joint_positions": self.joint_positions,
        "joint_velocities": self.joint_velocities,
        "gripper_width": base_env.np.array([self.gripper_width], dtype=base_env.np.float64),
        "joint_torques": self.actuator_torques,
        "actuator_torques": self.actuator_torques,
        "robot0_joint_torque": self.actuator_torques,
        "robot_scene_force": base_env.np.array([self.robot_scene_force({getattr(self, "target_id", -1)})], dtype=base_env.np.float64),
    }


@contextmanager
def headless() -> Iterator[None]:
    """Environments constructed and stepped inside this context render nothing."""
    saved_renderer = base_env.mujoco.Renderer
    saved_observation = base_env.BaseFrankaEnv.get_observation
    base_env.mujoco.Renderer = _NoRenderer
    base_env.BaseFrankaEnv.get_observation = _lowdim_observation
    try:
        yield
    finally:
        base_env.mujoco.Renderer = saved_renderer
        base_env.BaseFrankaEnv.get_observation = saved_observation
