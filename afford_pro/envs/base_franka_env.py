"""Base Gymnasium Environment for Franka Emika Panda in MuJoCo."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class BaseFrankaEnv(gym.Env):
    """Universal Gymnasium Environment for Franka Panda 7-DoF robot in MuJoCo."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str | Path,
        render_mode: str = "rgb_array",
        agentic_cam_size: tuple[int, int] = (640, 660),
        wrist_cam_size: tuple[int, int] = (300, 420),
        control_mode: str = "cartesian_delta",  # "cartesian_delta" or "joint_delta"
        action_scale: float = 0.04,            # max delta per step in meters / radians
        max_episode_steps: int = 500,
        render_depth: bool = False,
    ):
        super().__init__()
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Model XML not found at: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.agentic_cam_size = agentic_cam_size
        self.wrist_cam_size = wrist_cam_size
        self.control_mode = control_mode
        self.action_scale = action_scale
        self.max_episode_steps = max_episode_steps
        self.render_depth = render_depth

        self.current_step = 0
        self.arm_joints = [f"joint{i}" for i in range(1, 8)]
        self.arm_qpos_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.arm_joints
        ]
        self.arm_dof_adr = [
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.arm_joints
        ]

        # Gripper fingers
        self.f1_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
        self.f2_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")]
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Camera renderers
        self.cam_agentic_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "agentic_cam")
        self.cam_wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

        self.renderer_agentic = mujoco.Renderer(self.model, height=agentic_cam_size[0], width=agentic_cam_size[1])
        self.renderer_wrist = mujoco.Renderer(self.model, height=wrist_cam_size[0], width=wrist_cam_size[1])

        # Nominal ready joint configuration
        self.ready_q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785], dtype=np.float64)
        self.gripper_width = 0.04

        # Action space:
        # If cartesian_delta: [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd] (7,)
        # If joint_delta: [dq1, dq2, dq3, dq4, dq5, dq6, dq7, gripper_cmd] (8,)
        if self.control_mode == "cartesian_delta":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # Observation space
        obs_dict: dict[str, Any] = {
            "agentic_rgb": spaces.Box(low=0, high=255, shape=(*agentic_cam_size, 3), dtype=np.uint8),
            "wrist_rgb": spaces.Box(low=0, high=255, shape=(*wrist_cam_size, 3), dtype=np.uint8),
            "joint_positions": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float64),
            "joint_velocities": spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float64),
            "tcp_pose": spaces.Box(low=-5.0, high=5.0, shape=(7,), dtype=np.float64),
            "gripper_width": spaces.Box(low=0.0, high=0.05, shape=(1,), dtype=np.float64),
        }
        if self.render_depth:
            obs_dict["agentic_depth"] = spaces.Box(low=0.0, high=10.0, shape=agentic_cam_size, dtype=np.float32)
        self.observation_space = spaces.Dict(obs_dict)
        self._setup_task()
        self.reset()

    def _setup_task(self) -> None:
        """Hook for subclasses to initialize task-specific IDs and nominal poses before reset."""
        pass

    # ----------------------------------------------------------------- Kinematics & States
    @property
    def tcp_position(self) -> np.ndarray:
        """Tool Center Point (gripper fingertip center) position in robot base frame (m)."""
        mujoco.mj_forward(self.model, self.data)
        # Hand body origin offset by 10 cm along hand -Z to reach jaw fingertips
        return self.data.xpos[self.hand_id] + np.array([0.0, 0.0, -0.10], dtype=np.float64)

    @property
    def tcp_pose(self) -> np.ndarray:
        """TCP pose [x, y, z, qx, qy, qz, qw]."""
        pos = self.tcp_position
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.xmat[self.hand_id])
        return np.concatenate([pos, quat])

    @property
    def joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_adr], dtype=np.float64)

    @property
    def joint_velocities(self) -> np.ndarray:
        return np.array([self.data.qvel[adr] for adr in self.arm_dof_adr], dtype=np.float64)

    def solve_ik(self, target_pos: np.ndarray, max_iter: int = 40, tol: float = 0.002) -> bool:
        """Solve Damped Least Squares IK targeting fingertip position."""
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)
            cur_pos = self.tcp_position
            err = target - cur_pos
            if np.linalg.norm(err) < tol:
                return True
            mujoco.mj_jac(self.model, self.data, jacp, jacr, cur_pos, self.hand_id)
            J = jacp[:, self.arm_dof_adr]
            lam = 0.02
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)
            for i, adr in enumerate(self.arm_qpos_adr):
                self.data.qpos[adr] += 0.45 * dq[i]
                j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.arm_joints[i])
                rng = self.model.jnt_range[j_id]
                self.data.qpos[adr] = np.clip(self.data.qpos[adr], rng[0], rng[1])
        return False

    def set_gripper(self, width: float) -> None:
        """Set parallel jaw gripper opening width (0.00 closed .. 0.04 open)."""
        w = float(np.clip(width, 0.0, 0.04))
        self.gripper_width = w
        self.data.qpos[self.f1_adr] = w
        self.data.qpos[self.f2_adr] = w

    # ----------------------------------------------------------------- Observation & Render
    def get_observation(self) -> dict[str, Any]:
        mujoco.mj_forward(self.model, self.data)

        self.renderer_agentic.update_scene(self.data, camera=self.cam_agentic_id)
        agentic_rgb = self.renderer_agentic.render().copy()

        self.renderer_wrist.update_scene(self.data, camera=self.cam_wrist_id)
        wrist_rgb = self.renderer_wrist.render().copy()

        obs: dict[str, Any] = {
            "agentic_rgb": agentic_rgb,
            "wrist_rgb": wrist_rgb,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "tcp_pose": self.tcp_pose,
            "gripper_width": np.array([self.gripper_width], dtype=np.float64),
        }

        if self.render_depth:
            self.renderer_agentic.enable_depth_rendering()
            self.renderer_agentic.update_scene(self.data, camera=self.cam_agentic_id)
            obs["agentic_depth"] = self.renderer_agentic.render().copy()
            self.renderer_agentic.disable_depth_rendering()

            self.renderer_wrist.enable_depth_rendering()
            self.renderer_wrist.update_scene(self.data, camera=self.cam_wrist_id)
            obs["wrist_depth"] = self.renderer_wrist.render().copy()
            self.renderer_wrist.disable_depth_rendering()

        return obs

    # ----------------------------------------------------------------- Step & Reset
    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32)

        if self.control_mode == "cartesian_delta":
            delta_xyz = action[:3] * self.action_scale
            target_pos = self.tcp_position + delta_xyz
            self.solve_ik(target_pos)
            gripper_action = action[-1]
        else:
            delta_q = action[:7] * self.action_scale
            cur_q = self.joint_positions
            for i, adr in enumerate(self.arm_qpos_adr):
                self.data.qpos[adr] = cur_q[i] + delta_q[i]
            gripper_action = action[-1]

        # Gripper: continuous [-1, 1] mapped to [0.0, 0.04]
        target_width = 0.02 * (gripper_action + 1.0)
        self.set_gripper(target_width)

        # Physics integration
        mujoco.mj_step(self.model, self.data)

        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = self.check_success()
        truncated = self.current_step >= self.max_episode_steps
        info = {"success": terminated, "step": self.current_step}

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        mujoco.mj_resetData(self.model, self.data)
        for adr, val in zip(self.arm_qpos_adr, self.ready_q):
            self.data.qpos[adr] = val
        self.set_gripper(0.04)

        # Subclass hook for resetting object positions and applying perturbations
        self.reset_objects(options=options)

        mujoco.mj_forward(self.model, self.data)
        obs = self.get_observation()
        return obs, {}

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Override in task environment to configure or perturb object poses."""
        pass

    def compute_reward(self) -> float:
        """Override in task environment."""
        return 1.0 if self.check_success() else 0.0

    def check_success(self) -> bool:
        """Override in task environment to evaluate task-specific completion."""
        return False
