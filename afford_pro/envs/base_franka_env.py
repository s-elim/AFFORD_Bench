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
GRIPPER_OPEN_WIDTH = 0.04  # per finger (m)
# Commanded TCP targets are clipped to this box (m, world frame).
WORKSPACE_LOW = np.array([0.20, -0.50, 0.0])
WORKSPACE_HIGH = np.array([0.80, 0.50, 0.70])


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class BaseFrankaEnv(gym.Env):
    """Franka Panda 7-DoF manipulation environment in MuJoCo.

    Control (20 Hz): an action [dx, dy, dz, droll, dpitch, dyaw, grip] sets a TCP target relative to the
    measured TCP; 6-DoF damped-least-squares IK (hand pointing down, commanded yaw) gives a joint target
    that the arm's position servos track over `n_substeps` physics steps with gravity compensation.
    Roll and pitch commands are not executed. The gripper is a force-limited position servo (<= ~70 N
    per finger), so grasps hold only through contact friction.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        xml_path: str | Path,
        render_mode: str = "rgb_array",
        agentic_cam_size: tuple[int, int] = (640, 660),
        wrist_cam_size: tuple[int, int] = (300, 420),
        control_mode: str = "cartesian_delta",  # "cartesian_delta" or "joint_delta"
        action_scale: float = 0.025,           # max per-axis TCP delta per step (m) / joint delta (rad)
        yaw_scale: float = 0.05,               # max yaw delta per step (rad), i.e. 1 rad/s
        max_tcp_speed: float = 0.5,            # m/s cap on the commanded TCP displacement
        max_finger_speed: float = 0.05,        # m/s cap on each finger's commanded opening (Franka Hand)
        n_substeps: int = 25,                  # physics steps per env step
        settle_substeps: int = 40,             # minimum physics steps run at reset before the first observation
        max_episode_steps: int = 500,
        render_depth: bool = False,
    ):
        super().__init__()
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Model XML not found at: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.agentic_cam_size = agentic_cam_size
        self.wrist_cam_size = wrist_cam_size
        self.control_mode = control_mode
        self.action_scale = action_scale
        self.yaw_scale = yaw_scale
        self.n_substeps = n_substeps
        self.settle_substeps = settle_substeps
        self.control_dt = n_substeps * self.model.opt.timestep
        self.max_tcp_step = max_tcp_speed * self.control_dt
        self.max_finger_substep = max_finger_speed * self.model.opt.timestep
        self.finger_cmd = GRIPPER_OPEN_WIDTH
        self.max_episode_steps = max_episode_steps
        self.render_depth = render_depth

        self.current_step = 0
        m = self.model
        self.arm_joints = [f"joint{i}" for i in range(1, 8)]
        arm_joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_joints]
        self.arm_qpos_adr = [int(m.jnt_qposadr[j]) for j in arm_joint_ids]
        self.arm_dof_adr = [int(m.jnt_dofadr[j]) for j in arm_joint_ids]
        self.arm_joint_range = np.array([m.jnt_range[j] for j in arm_joint_ids])
        self.arm_actuator_ids = [
            a for j in arm_joint_ids for a in range(m.nu)
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == j
        ]
        self.gripper_actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        # Servo velocity-feedforward gain kv/kp: the affine servo damps absolute joint velocity, so a
        # moving target is offset by (kv/kp) * qdot to cancel the steady-state lag.
        self.arm_servo_lead = np.array([m.actuator_biasprm[a, 2] / m.actuator_biasprm[a, 1] for a in self.arm_actuator_ids])

        finger_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("finger_joint1", "finger_joint2")]
        self.finger_qpos_adr = [int(m.jnt_qposadr[j]) for j in finger_ids]
        self.finger_dof_adr = [int(m.jnt_dofadr[j]) for j in finger_ids]
        self.finger_body_ids = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
        }
        self.hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.robot_root_id = int(m.body_rootid[self.hand_id])
        self.robot_body_ids = {b for b in range(m.nbody) if m.body_rootid[b] == self.robot_root_id}
        # The scenes declare gravcomp="1" on the robot bodies (gravity compensation, as on the real
        # Franka controller); it must be set in the XML because MuJoCo skips it when absent at compile time.
        assert all(m.body_gravcomp[b] == 1.0 for b in self.robot_body_ids if b != self.robot_root_id or m.body_mass[b] > 0)

        # Free-floating bodies: qpos address -> dof address, and body id -> qpos address.
        self.free_qpos_to_dof: dict[int, int] = {}
        self.free_body_qposadr: dict[int, int] = {}
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_qpos_to_dof[int(m.jnt_qposadr[j])] = int(m.jnt_dofadr[j])
                self.free_body_qposadr[int(m.jnt_bodyid[j])] = int(m.jnt_qposadr[j])
        # Bodies an env pins in place every substep; reset de-penetration leaves them alone.
        self.static_body_ids: set[int] = set()

        # Camera renderers
        self.cam_agentic_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "agentic_cam")
        self.cam_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        self.renderer_agentic = mujoco.Renderer(m, height=agentic_cam_size[0], width=agentic_cam_size[1])
        self.renderer_wrist = mujoco.Renderer(m, height=wrist_cam_size[0], width=wrist_cam_size[1])

        # Nominal ready joint configuration and the hand orientation it defines (pointing down).
        self.ready_q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785], dtype=np.float64)
        self._set_arm_state(self.ready_q)
        mujoco.mj_forward(m, self.data)
        r_ready = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        self._r_down = _rot_z(-self._yaw_of(r_ready)) @ r_ready
        # Optional initial TCP position (m); None starts at the ready pose.
        self.init_tcp_pos: np.ndarray | None = getattr(self, "init_tcp_pos", None)

        # Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]; roll and pitch are not executed.
        if self.control_mode == "cartesian_delta":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

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
    @staticmethod
    def _yaw_of(rot: np.ndarray) -> float:
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def _tcp_from(self, data: mujoco.MjData) -> np.ndarray:
        rot = data.xmat[self.hand_id].reshape(3, 3)
        return data.xpos[self.hand_id] + rot @ TCP_OFFSET_HAND

    @property
    def tcp_position(self) -> np.ndarray:
        """Tool Center Point (fingertip center) in the world frame (m); the Panda base is at the origin."""
        mujoco.mj_kinematics(self.model, self.data)
        return self._tcp_from(self.data)

    @property
    def tcp_yaw(self) -> float:
        """Hand yaw (rad) relative to the ready orientation; the finger closing axis is the hand y axis."""
        mujoco.mj_kinematics(self.model, self.data)
        rot = self.data.xmat[self.hand_id].reshape(3, 3) @ self._r_down.T
        return self._yaw_of(rot)

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

    @property
    def gripper_width(self) -> float:
        """Measured opening of one finger (m), 0.0 closed .. 0.04 open."""
        return float(np.mean([self.data.qpos[a] for a in self.finger_qpos_adr]))

    def solve_ik(self, target_pos: np.ndarray, target_yaw: float, q_init: np.ndarray, max_iter: int = 60,
                 tol: float = 1e-4) -> np.ndarray:
        """6-DoF damped-least-squares IK on a scratch copy of the state: TCP position plus hand pointing
        down with the given yaw. Returns the joint target; the simulated arm is not moved."""
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        for i, adr in enumerate(self.arm_qpos_adr):
            d.qpos[adr] = q_init[i]
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        r_target = _rot_z(target_yaw) @ self._r_down
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(max_iter):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            tcp = self._tcp_from(d)
            r_cur = d.xmat[self.hand_id].reshape(3, 3)
            err_p = target - tcp
            err_r = 0.5 * sum(np.cross(r_cur[:, k], r_target[:, k]) for k in range(3))
            if np.linalg.norm(err_p) < tol and np.linalg.norm(err_r) < 1e-3:
                break
            mujoco.mj_jac(m, d, jacp, jacr, tcp, self.hand_id)
            J = np.vstack([jacp[:, self.arm_dof_adr], jacr[:, self.arm_dof_adr]])
            err = np.concatenate([err_p, 0.5 * err_r])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            for i, adr in enumerate(self.arm_qpos_adr):
                d.qpos[adr] = np.clip(d.qpos[adr] + dq[i], *self.arm_joint_range[i])
        return np.array([d.qpos[adr] for adr in self.arm_qpos_adr])

    def robot_scene_force(self, exclude: set[int] | None = None) -> float:
        """Magnitude (N) of the net contact force the scene exerts on the robot, excluding bodies in `exclude`
        (the manipulated body); evaluated at the current state for the force violation rate."""
        m, d = self.model, self.data
        exclude = exclude or set()
        total = np.zeros(3)
        wrench = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.robot_body_ids, b2 in self.robot_body_ids
            if r1 == r2 or b1 in exclude or b2 in exclude:
                continue
            mujoco.mj_contactForce(m, d, i, wrench)
            total += c.frame[:3] * wrench[0] * (1.0 if r2 else -1.0)  # contact normal points from geom1 to geom2
        return float(np.linalg.norm(total))

    def camera_parameters(self) -> dict[str, np.ndarray]:
        """Pinhole intrinsics K (3x3, pixels, square pixels from the vertical field of view) and world pose
        (position and 3x3 rotation of the MuJoCo camera frame, which looks along -z with y up) of the workcell
        and wrist cameras at their render sizes. Metric depth back-projects as P = D(u, v) K^-1 [u, v, 1]^T in
        the OpenCV camera frame (x right, y down, z forward); diag(1, -1, -1) maps it to the MuJoCo frame."""
        mujoco.mj_forward(self.model, self.data)
        out = {}
        for name, cam, (h, w) in (("agentic", self.cam_agentic_id, self.agentic_cam_size),
                                  ("wrist", self.cam_wrist_id, self.wrist_cam_size)):
            f = 0.5 * h / np.tan(np.radians(self.model.cam_fovy[cam]) / 2.0)
            out[f"{name}_K"] = np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
            out[f"{name}_pos"] = self.data.cam_xpos[cam].copy()
            out[f"{name}_rot"] = self.data.cam_xmat[cam].reshape(3, 3).copy()
        return out

    def hand_collides(self, tcp: np.ndarray, yaw: float, finger: float, ignore: set[int]) -> bool:
        """True if the robot, posed by IK at `tcp` and `yaw` with the given per-finger opening, would
        penetrate a non-robot body outside `ignore` (or the pose is out of reach). Uses the scratch state;
        the simulated arm is not moved."""
        m, d = self.model, self._ik_data
        self.solve_ik(tcp, yaw, self.joint_positions)
        for adr in self.finger_qpos_adr:
            d.qpos[adr] = finger
        mujoco.mj_fwdPosition(m, d)
        if np.linalg.norm(self._tcp_from(d) - tcp) > 0.002:
            return True
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in self.robot_body_ids) == (b2 in self.robot_body_ids) or c.dist > -0.0005:
                continue
            other = b2 if b1 in self.robot_body_ids else b1
            if int(m.body_rootid[other]) not in ignore and other not in ignore:
                return True
        return False

    def _set_arm_state(self, q: np.ndarray, finger: float = GRIPPER_OPEN_WIDTH) -> None:
        """Teleport the arm (reset only): joint positions, zero velocity, matching servo targets."""
        for i, (qa, da) in enumerate(zip(self.arm_qpos_adr, self.arm_dof_adr)):
            self.data.qpos[qa] = q[i]
            self.data.qvel[da] = 0.0
        for a, qi in zip(self.arm_actuator_ids, q):
            self.data.ctrl[a] = qi
        for qa, da in zip(self.finger_qpos_adr, self.finger_dof_adr):
            self.data.qpos[qa] = finger
            self.data.qvel[da] = 0.0
        self.finger_cmd = finger
        self.data.ctrl[self.gripper_actuator_id] = 255.0 * finger / GRIPPER_OPEN_WIDTH

    def pin_free_body(self, qposadr: int, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
        """Place a free body at a pose and zero its velocity."""
        self.data.qpos[qposadr : qposadr + 3] = pos
        self.data.qpos[qposadr + 3 : qposadr + 7] = (1.0, 0.0, 0.0, 0.0) if quat is None else quat
        dof = self.free_qpos_to_dof[qposadr]
        self.data.qvel[dof : dof + 6] = 0.0

    def _hold_bodies(self) -> None:
        """Hook called before every physics substep to re-apply kinematic constraints (none by default)."""
        pass

    def body_contacts(self, body_id: int) -> set[int]:
        """Root body ids in contact with a body (robot contacts are reported per finger body or as the robot root)."""
        out: set[int] = set()
        for c in self.data.contact[: self.data.ncon]:
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if body_id not in (b1, b2):
                continue
            other = b2 if b1 == body_id else b1
            out.add(other if other in self.finger_body_ids else int(self.model.body_rootid[other]))
        return out

    def is_grasped(self, body_id: int) -> bool:
        """Both finger bodies touch the body."""
        return self.finger_body_ids <= self.body_contacts(body_id)

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
        action = np.asarray(action, dtype=np.float64)
        q_start = self.joint_positions

        if self.control_mode == "cartesian_delta":
            delta_xyz = action[:3] * self.action_scale
            norm = np.linalg.norm(delta_xyz)
            if norm > self.max_tcp_step:
                delta_xyz *= self.max_tcp_step / norm
            target_yaw = self.tcp_yaw + float(np.clip(action[5], -1.0, 1.0)) * self.yaw_scale
            target_pos = np.clip(self.tcp_position + delta_xyz, WORKSPACE_LOW, WORKSPACE_HIGH)
            q_target = self.solve_ik(target_pos, target_yaw, q_start)
        else:
            q_target = np.clip(q_start + action[:7] * self.action_scale, self.arm_joint_range[:, 0], self.arm_joint_range[:, 1])

        # Gripper: continuous [-1, 1] mapped to a per-finger opening target in [0.0, 0.04] m, approached
        # at no more than max_finger_speed.
        finger_target = 0.5 * GRIPPER_OPEN_WIDTH * (float(np.clip(action[-1], -1.0, 1.0)) + 1.0)
        if finger_target > self.finger_cmd:
            # Opening starts from the measured opening, not from a command wound below a held body.
            self.finger_cmd = max(self.finger_cmd, min(self.gripper_width, finger_target))

        # Servo targets move linearly to the IK solution across the control period (velocity feedforward
        # while moving), then hold it.
        qdot_cmd = (q_target - q_start) / self.control_dt
        for k in range(1, self.n_substeps + 1):
            q_cmd = q_start + (k / self.n_substeps) * (q_target - q_start)
            lead = self.arm_servo_lead * qdot_cmd if k < self.n_substeps else 0.0
            for a, qi in zip(self.arm_actuator_ids, q_cmd + lead):
                self.data.ctrl[a] = qi
            self.finger_cmd += float(np.clip(finger_target - self.finger_cmd, -self.max_finger_substep, self.max_finger_substep))
            self.data.ctrl[self.gripper_actuator_id] = 255.0 * self.finger_cmd / GRIPPER_OPEN_WIDTH
            self._hold_bodies()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._after_step()

        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = self.check_success()
        truncated = self.current_step >= self.max_episode_steps
        info = {"success": terminated, "step": self.current_step,
                "robot_scene_force": self.robot_scene_force({getattr(self, "target_id", -1)})}

        return obs, reward, terminated, truncated, info

    def _after_step(self) -> None:
        """Hook for per-step bookkeeping after physics integration."""
        pass

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        mujoco.mj_resetData(self.model, self.data)
        engine = getattr(self, "perturbation_engine", None)
        if engine is not None:
            # Derive the perturbation seed from the env RNG so reset(seed) fixes the whole layout.
            engine.begin_episode(self.model, int(self.np_random.integers(0, 2**31 - 1)))

        self._set_arm_state(self.ready_q)
        mujoco.mj_forward(self.model, self.data)
        start_tcp = self.tcp_position if self.init_tcp_pos is None else self.init_tcp_pos
        if engine is not None:
            start_tcp = engine.jitter_point(start_tcp)
        self._set_arm_state(self.solve_ik(start_tcp, 0.0, self.ready_q, max_iter=300, tol=2e-4))

        # Subclass hook for resetting object positions and applying perturbations
        self.reset_objects(options=options)

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
        finger = self.gripper_width
        self._hold_bodies()
        self._depenetrate()
        for k in range(max_substeps):
            self._set_arm_state(q_hold, finger)
            self._hold_bodies()
            mujoco.mj_step(self.model, self.data)
            if k + 1 >= self.settle_substeps:
                lin, ang = self._max_free_body_speed()
                if lin < lin_tol and ang < ang_tol:
                    break
        self._set_arm_state(q_hold, finger)
        self._hold_bodies()
        mujoco.mj_forward(self.model, self.data)

    def reset_objects(self, options: dict[str, Any] | None = None) -> None:
        """Override in task environment to configure or perturb object poses, then call _settle()."""
        self._settle()

    def compute_reward(self) -> float:
        """Override in task environment."""
        return 1.0 if self.check_success() else 0.0

    def check_success(self) -> bool:
        """Override in task environment to evaluate task-specific completion."""
        return False

    def close(self) -> None:
        """Release MuJoCo renderers and OpenGL contexts."""
        if hasattr(self, "renderer_agentic") and self.renderer_agentic is not None:
            try:
                self.renderer_agentic.close()
            except Exception:
                pass
        if hasattr(self, "renderer_wrist") and self.renderer_wrist is not None:
            try:
                self.renderer_wrist.close()
            except Exception:
                pass
        super().close()
