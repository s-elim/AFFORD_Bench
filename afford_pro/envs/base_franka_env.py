"""Base Gymnasium Environment for Franka Emika Panda in MuJoCo."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

# Fingertip center along the hand's local +z axis (m).
TCP_OFFSET_HAND = np.array([0.0, 0.0, 0.10], dtype=np.float64)


class BaseFrankaEnv(gym.Env):
    """Universal Gymnasium Environment for Franka Panda 7-DoF robot in MuJoCo.

    Control: each env step converts a Cartesian TCP increment into a joint target with
    damped-least-squares IK, then advances `n_substeps` physics steps while the arm is
    interpolated kinematically toward that target (25 x 2 ms = 20 Hz by default).
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        xml_path: str | Path,
        render_mode: str = "rgb_array",
        agentic_cam_size: tuple[int, int] = (640, 660),
        wrist_cam_size: tuple[int, int] = (300, 420),
        control_mode: str = "cartesian_delta",  # "cartesian_delta" or "joint_delta"
        action_scale: float = 0.025,           # max per-axis delta per step in meters / radians
        max_tcp_speed: float = 0.5,            # m/s cap on the commanded TCP displacement
        n_substeps: int = 25,                  # physics steps per env step
        settle_substeps: int = 40,             # physics steps run at reset before the first observation
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
        self.n_substeps = n_substeps
        self.settle_substeps = settle_substeps
        self.control_dt = n_substeps * self.model.opt.timestep
        self.max_tcp_step = max_tcp_speed * self.control_dt
        self.max_episode_steps = max_episode_steps
        self.render_depth = render_depth

        self.current_step = 0
        self.arm_joints = [f"joint{i}" for i in range(1, 8)]
        arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_joints]
        self.arm_qpos_adr = [self.model.jnt_qposadr[j] for j in arm_joint_ids]
        self.arm_dof_adr = [self.model.jnt_dofadr[j] for j in arm_joint_ids]
        self.arm_joint_range = np.array([self.model.jnt_range[j] for j in arm_joint_ids])
        # Position actuators driving the arm joints, if the scene defines them.
        self.arm_actuator_ids = [
            a for j in arm_joint_ids for a in range(self.model.nu)
            if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and self.model.actuator_trnid[a, 0] == j
        ]

        # Gripper fingers
        finger_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("finger_joint1", "finger_joint2")]
        self.f1_adr, self.f2_adr = (self.model.jnt_qposadr[j] for j in finger_ids)
        self.finger_dof_adr = [self.model.jnt_dofadr[j] for j in finger_ids]
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.robot_root_id = self.model.body_rootid[self.hand_id]

        # Free-floating bodies: qpos address -> dof address, and body id -> qpos address.
        self.free_qpos_to_dof: dict[int, int] = {}
        self.free_body_qposadr: dict[int, int] = {}
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_qpos_to_dof[int(self.model.jnt_qposadr[j])] = int(self.model.jnt_dofadr[j])
                self.free_body_qposadr[int(self.model.jnt_bodyid[j])] = int(self.model.jnt_qposadr[j])
        # Bodies an env pins in place every substep; reset de-penetration leaves them alone.
        self.static_body_ids: set[int] = set()

        # Camera renderers
        self.cam_agentic_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "agentic_cam")
        self.cam_wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

        self.renderer_agentic = mujoco.Renderer(self.model, height=agentic_cam_size[0], width=agentic_cam_size[1])
        self.renderer_wrist = mujoco.Renderer(self.model, height=wrist_cam_size[0], width=wrist_cam_size[1])

        # Nominal ready joint configuration
        self.ready_q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785], dtype=np.float64)
        # Optional initial TCP position (m); None starts at the ready pose.
        self.init_tcp_pos: np.ndarray | None = getattr(self, "init_tcp_pos", None)
        self.gripper_width = 0.04

        # Action space:
        # If cartesian_delta: [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd] (7,); rotation is not executed.
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
        """Tool Center Point (fingertip center) in the world frame (m); the Panda base is at the origin."""
        mujoco.mj_forward(self.model, self.data)
        rot = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + rot @ TCP_OFFSET_HAND

    @property
    def tcp_pose(self) -> np.ndarray:
        """TCP pose [x, y, z, qw, qx, qy, qz] (MuJoCo w-first quaternion of the hand frame)."""
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
                self.data.qpos[adr] = np.clip(self.data.qpos[adr] + 0.45 * dq[i], *self.arm_joint_range[i])
        return False

    def set_gripper(self, width: float) -> None:
        """Set parallel jaw gripper opening width (0.00 closed .. 0.04 open)."""
        w = float(np.clip(width, 0.0, 0.04))
        self.gripper_width = w
        self.data.qpos[self.f1_adr] = w
        self.data.qpos[self.f2_adr] = w

    def _set_arm(self, q: np.ndarray, qdot: np.ndarray | None = None) -> None:
        """Write arm joint positions (and velocities), keep the fingers at the commanded width."""
        for i, (qa, da) in enumerate(zip(self.arm_qpos_adr, self.arm_dof_adr)):
            self.data.qpos[qa] = q[i]
            self.data.qvel[da] = 0.0 if qdot is None else qdot[i]
        for a, qi in zip(self.arm_actuator_ids, q):
            self.data.ctrl[a] = qi
        self.set_gripper(self.gripper_width)
        for da in self.finger_dof_adr:
            self.data.qvel[da] = 0.0

    def pin_free_body(self, qposadr: int, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
        """Place a free body at a pose and zero its velocity."""
        self.data.qpos[qposadr : qposadr + 3] = pos
        self.data.qpos[qposadr + 3 : qposadr + 7] = (1.0, 0.0, 0.0, 0.0) if quat is None else quat
        dof = self.free_qpos_to_dof[qposadr]
        self.data.qvel[dof : dof + 6] = 0.0

    def _hold_bodies(self) -> None:
        """Hook called before every physics substep to re-apply kinematic attachments."""
        pass

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
        q_start = self.joint_positions

        if self.control_mode == "cartesian_delta":
            delta_xyz = action[:3].astype(np.float64) * self.action_scale
            norm = np.linalg.norm(delta_xyz)
            if norm > self.max_tcp_step:
                delta_xyz *= self.max_tcp_step / norm
            self.solve_ik(self.tcp_position + delta_xyz)
            q_target = self.joint_positions
        else:
            q_target = np.clip(q_start + action[:7] * self.action_scale, self.arm_joint_range[:, 0], self.arm_joint_range[:, 1])
        gripper_action = action[-1]

        # Gripper: continuous [-1, 1] mapped to [0.0, 0.04]
        self.set_gripper(0.02 * (gripper_action + 1.0))

        # Physics integration: interpolate the arm to its target across the control period.
        qdot = (q_target - q_start) / self.control_dt
        for k in range(1, self.n_substeps + 1):
            self._set_arm(q_start + (k / self.n_substeps) * (q_target - q_start), qdot)
            self._hold_bodies()
            mujoco.mj_step(self.model, self.data)
        self._set_arm(q_target)
        self._hold_bodies()
        mujoco.mj_forward(self.model, self.data)

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
        engine = getattr(self, "perturbation_engine", None)
        if engine is not None:
            # Derive the perturbation seed from the env RNG so reset(seed) fixes the whole layout.
            engine.begin_episode(self.model, int(self.np_random.integers(0, 2**31 - 1)))

        self.gripper_width = 0.04
        self._set_arm(self.ready_q)
        start_tcp = self.tcp_position if self.init_tcp_pos is None else self.init_tcp_pos
        if engine is not None:
            start_tcp = engine.jitter_point(start_tcp)
        self.solve_ik(start_tcp, max_iter=300, tol=0.001)
        self._set_arm(self.joint_positions)

        # Subclass hook for resetting object positions and applying perturbations
        self.reset_objects(options=options)
        self._settle()

        obs = self.get_observation()
        return obs, {}

    def _depenetrate(self, max_iter: int = 30, slack: float = 0.001) -> None:
        """Raise free bodies until no contact penetrates by more than `slack` (m)."""
        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)
            lift: dict[int, float] = {}
            for c in self.data.contact[: self.data.ncon]:
                if c.dist >= -slack:
                    continue
                b1 = int(self.model.body_rootid[self.model.geom_bodyid[c.geom1]])
                b2 = int(self.model.body_rootid[self.model.geom_bodyid[c.geom2]])
                if self.robot_root_id in (b1, b2):
                    continue
                movable = [b for b in (b1, b2) if b in self.free_body_qposadr and b not in self.static_body_ids]
                if not movable:
                    continue
                # Between two movable bodies, lift the one resting higher.
                body = max(movable, key=lambda b: self.data.xpos[b][2])
                lift[body] = max(lift.get(body, 0.0), -c.dist + slack)
            if not lift:
                return
            for body, dz in lift.items():
                self.data.qpos[self.free_body_qposadr[body] + 2] += dz

    def _max_free_body_speed(self) -> tuple[float, float]:
        """Largest linear (m/s) and angular (rad/s) speed over the movable free bodies."""
        lin, ang = 0.0, 0.0
        for body, qadr in self.free_body_qposadr.items():
            if body in self.static_body_ids:
                continue
            dof = self.free_qpos_to_dof[qadr]
            lin = max(lin, float(np.linalg.norm(self.data.qvel[dof : dof + 3])))
            ang = max(ang, float(np.linalg.norm(self.data.qvel[dof + 3 : dof + 6])))
        return lin, ang

    def _settle(self, max_substeps: int = 1000, lin_tol: float = 0.005, ang_tol: float = 0.05) -> None:
        """Remove spawn interpenetration, then let free bodies come to rest with the arm held still.

        Runs at least `settle_substeps` physics steps and continues until every movable free body
        is below (lin_tol, ang_tol), or `max_substeps` (2 s) have elapsed.
        """
        q_hold = self.joint_positions
        self._hold_bodies()
        self._depenetrate()
        for k in range(max_substeps):
            self._set_arm(q_hold)
            self._hold_bodies()
            mujoco.mj_step(self.model, self.data)
            if k + 1 >= self.settle_substeps:
                lin, ang = self._max_free_body_speed()
                if lin < lin_tol and ang < ang_tol:
                    break
        self._set_arm(q_hold)
        self._hold_bodies()
        mujoco.mj_forward(self.model, self.data)

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Override in task environment to configure or perturb object poses."""
        pass

    def compute_reward(self) -> float:
        """Override in task environment."""
        return 1.0 if self.check_success() else 0.0

    def check_success(self) -> bool:
        """Override in task environment to evaluate task-specific completion."""
        return False
