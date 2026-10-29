"""Base Gymnasium Environment for Franka Emika Panda in MuJoCo."""
from __future__ import annotations

from pathlib import Path
import struct
from typing import Any
import xml.etree.ElementTree as ET

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
# Recorded simulator state: time, qpos, qvel, actuator activations, and plugin state (MuJoCo's full physics state).
SIM_STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS

_PLACEHOLDER_OBJ = (
    b"o placeholder\n"
    b"v 0 0 0\n"
    b"v 0.01 0 0\n"
    b"v 0 0.01 0\n"
    b"v 0 0 0.01\n"
    b"f 1 2 3\n"
    b"f 1 2 4\n"
    b"f 1 3 4\n"
    b"f 2 3 4\n"
)


def _binary_placeholder_stl() -> bytes:
    header = b"placeholder".ljust(80, b" ")
    tris = [
        (0.0, 0.0, -1.0, (0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0)),
        (0.0, -1.0, 0.0, (0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.0, 0.01)),
        (-1.0, 0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        (1.0, 1.0, 1.0, (0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
    ]
    payload = bytearray()
    for nx, ny, nz, v1, v2, v3 in tris:
        payload.extend(
            struct.pack(
                "<12fH",
                nx, ny, nz,
                *v1,
                *v2,
                *v3,
                0,
            )
        )
    return header + struct.pack("<I", len(tris)) + bytes(payload)


_PLACEHOLDER_STL = _binary_placeholder_stl()
def _placeholder_asset_bytes(path: Path) -> bytes | None:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _PLACEHOLDER_OBJ
    if suffix == ".stl":
        return _PLACEHOLDER_STL
    return None


def _load_model_from_xml_string(xml_text: str, base_dir: Path) -> mujoco.MjModel:
    root = ET.fromstring(xml_text)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", ".")
    assets: dict[str, bytes] = {}
    changed = False
    for elem in root.findall(".//*[@file]"):
        raw = elem.get("file")
        if not raw:
            continue
        file_name = Path(raw.replace("\\", "/")).name
        if raw != file_name:
            elem.set("file", file_name)
            changed = True
        if file_name in assets:
            continue
        local_file = base_dir / file_name
        if not local_file.exists():
            for sub in ("robots/assets", "textures", "objects", "tables"):
                cand = base_dir / sub / file_name
                if cand.exists():
                    local_file = cand
                    break
        if local_file.exists():
            assets[file_name] = local_file.read_bytes()
            continue
        if elem.tag == "texture":
            elem.attrib.pop("file", None)
            elem.set("builtin", "flat")
            elem.set("width", "8")
            elem.set("height", "8")
            elem.set("rgb1", "1 1 1")
            changed = True
            continue
        placeholder = _placeholder_asset_bytes(local_file)
        if placeholder is None:
            raise
        assets[file_name] = placeholder
        changed = True
    new_xml_text = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(new_xml_text, assets=assets)


def _load_model_with_portable_assets(xml_path: Path) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(str(xml_path))
    except ValueError as exc:
        if "Error opening file" not in str(exc):
            raise
    return _load_model_from_xml_string(xml_path.read_text(encoding="utf-8"), xml_path.parent)


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_wxyz_to_axisangle(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert w-first MuJoCo quaternions (..., 4) [qw, qx, qy, qz] to 3-D axis-angle vectors."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    small = s < 1e-7
    scale = np.zeros_like(angle)
    scale[~small] = angle[~small] / s[~small]
    return (v * scale[..., np.newaxis]).astype(np.float64)


class MjSimState:
    """Wrapper around state array to provide .flatten() and numpy array compatibility."""

    def __init__(self, state: np.ndarray, time: float = 0.0) -> None:
        self._state = np.asarray(state, dtype=np.float64)
        self.time = time

    def flatten(self) -> np.ndarray:
        return self._state.copy()

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._state, dtype=dtype)

    def __len__(self) -> int:
        return len(self._state)

    def __getitem__(self, idx: Any) -> Any:
        return self._state[idx]


class MjModelAdapter:
    """Proxy around mujoco.MjModel that adds get_xml() and forwards all model attributes."""

    def __init__(self, model: mujoco.MjModel, env: BaseFrankaEnv) -> None:
        self._model = model
        self._env = env

    def get_xml(self) -> str:
        return self._env.get_model_xml()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class MjSimAdapter:
    """Shim object providing robosuite / robomimic / LIBERO MjSim interface."""

    def __init__(self, env: BaseFrankaEnv) -> None:
        self._env = env

    @property
    def model(self) -> MjModelAdapter:
        return MjModelAdapter(self._env.model, self._env)

    @property
    def data(self) -> mujoco.MjData:
        return self._env.data

    def forward(self) -> None:
        mujoco.mj_forward(self._env.model, self._env.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self._env.model, self._env.data)
        self._env._set_arm_state(self._env.ready_q)
        mujoco.mj_forward(self._env.model, self._env.data)

    def step(self) -> None:
        mujoco.mj_step(self._env.model, self._env.data)

    def get_state(self) -> MjSimState:
        return MjSimState(self._env.get_sim_state(), time=float(self._env.data.time))

    def set_state_from_flattened(self, state: np.ndarray) -> None:
        self._env.set_sim_state(state)
        mujoco.mj_forward(self._env.model, self._env.data)


class RobotAdapter:
    """Lightweight adapter mimicking a robosuite robot for robomimic / LIBERO compatibility."""

    def __init__(self, env: Any):
        self.env = env
        self.action_dim = env.action_space.shape[0] if getattr(env, "action_space", None) and env.action_space.shape else 7
        self.robot_model = type("RobotModel", (), {
            "naming_prefix": "robot0_",
            "base_xpos_offset": {"table": lambda: np.zeros(3), "kitchen_table": lambda: np.zeros(3), "empty": np.zeros(3)},
            "set_base_xpos": lambda *args: None,
        })()
        self.controller = type("Controller", (), {
            "eef_name": "grip_site",
            "use_delta": True,
        })()
        self.action_limits = (np.full(self.action_dim, -1.0), np.full(self.action_dim, 1.0))


class CompatStepResult(tuple):
    """Step result tuple (obs, reward, terminated, truncated, info) supporting 4-unpack (old gym / robomimic)
    and 5-unpack (gymnasium)."""

    def __new__(cls, obs: dict[str, Any], reward: float, terminated: bool, truncated: bool, info: dict[str, Any]):
        return super().__new__(cls, (obs, float(reward), bool(terminated), bool(truncated), info))

    @property
    def obs(self) -> dict[str, Any]:
        return self[0]

    @property
    def reward(self) -> float:
        return self[1]

    @property
    def terminated(self) -> bool:
        return self[2]

    @property
    def truncated(self) -> bool:
        return self[3]

    @property
    def done(self) -> bool:
        return bool(self[2] or self[3])

    @property
    def info(self) -> dict[str, Any]:
        return self[4]

    def __iter__(self):
        try:
            import inspect, dis
            frame = inspect.currentframe().f_back
            for ins in dis.get_instructions(frame.f_code):
                if ins.offset == frame.f_lasti:
                    if ins.opname == "UNPACK_SEQUENCE" and ins.argval == 4:
                        return iter((self[0], self[1], bool(self[2] or self[3]), self[4]))
                    break
        except Exception:
            pass
        return super().__iter__()


class CompatResetResult(dict):
    """Reset result dictionary (for old gym / robomimic / LIBERO) that also supports 2-unpacking
    `obs, info = env.reset()` for Gymnasium compatibility."""

    def __init__(self, obs: dict[str, Any], info: dict[str, Any] | None = None):
        super().__init__(obs)
        self.info = info if info is not None else {}

    @property
    def obs(self) -> dict[str, Any]:
        return self

    def __getitem__(self, key: Any) -> Any:
        if key == 0 and 0 not in self:
            return self
        if key == 1 and 1 not in self:
            return self.info
        return super().__getitem__(key)

    def __iter__(self):
        try:
            import inspect, dis
            frame = inspect.currentframe().f_back
            for ins in dis.get_instructions(frame.f_code):
                if ins.offset == frame.f_lasti:
                    if ins.opname == "UNPACK_SEQUENCE" and ins.argval == 2:
                        return iter((self, self.info))
                    break
        except Exception:
            pass
        return super().__iter__()


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
        invert_gripper_action: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Model XML not found at: {self.xml_path}")
        self._model_xml_str = self.xml_path.read_text(encoding="utf-8")

        # Framework flag translation (Robomimic / LIBERO / Gymnasium)
        if kwargs.get("use_depth_obs") or kwargs.get("camera_depths"):
            render_depth = True
        if kwargs.get("render") or kwargs.get("has_renderer"):
            render_mode = "human"
        self.extra_kwargs = kwargs
        self._sim_adapter: MjSimAdapter | None = None

        self.model = _load_model_with_portable_assets(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.agentic_cam_size = agentic_cam_size
        self.wrist_cam_size = wrist_cam_size
        self.control_mode = control_mode
        self.action_scale = action_scale
        self.yaw_scale = yaw_scale
        self.max_tcp_speed = max_tcp_speed
        self.max_finger_speed = max_finger_speed
        self.n_substeps = n_substeps
        self.settle_substeps = settle_substeps
        self.finger_cmd = GRIPPER_OPEN_WIDTH
        self.max_episode_steps = max_episode_steps
        self.render_depth = render_depth
        self.invert_gripper_action = invert_gripper_action
        self.current_step = 0

        self._cache_model_constants()

        # Camera renderers
        self.renderer_agentic = mujoco.Renderer(self.model, height=agentic_cam_size[0], width=agentic_cam_size[1])
        self.renderer_wrist = mujoco.Renderer(self.model, height=wrist_cam_size[0], width=wrist_cam_size[1])

        # Nominal ready joint configuration and the hand orientation it defines (pointing down).
        self.ready_q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785], dtype=np.float64)
        self._set_arm_state(self.ready_q)
        mujoco.mj_forward(self.model, self.data)
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
            "gripper_width": spaces.Box(low=0.0, high=0.1, shape=(1,), dtype=np.float64),
            "joint_torques": spaces.Box(low=-100.0, high=100.0, shape=(7,), dtype=np.float64),
            "actuator_torques": spaces.Box(low=-100.0, high=100.0, shape=(7,), dtype=np.float64),
            "robot_scene_force": spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float64),
            # Robomimic & LIBERO aliases
            "agentview_image": spaces.Box(low=0, high=255, shape=(*agentic_cam_size, 3), dtype=np.uint8),
            "agentview_rgb": spaces.Box(low=0, high=255, shape=(*agentic_cam_size, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": spaces.Box(low=0, high=255, shape=(*wrist_cam_size, 3), dtype=np.uint8),
            "eye_in_hand_rgb": spaces.Box(low=0, high=255, shape=(*wrist_cam_size, 3), dtype=np.uint8),
            "joint_states": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float64),
            "robot0_joint_pos": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float64),
            "robot0_joint_vel": spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float64),
            "robot0_joint_torque": spaces.Box(low=-100.0, high=100.0, shape=(7,), dtype=np.float64),
            "robot0_eef_pos": spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float64),
            "ee_pos": spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float64),
            "gripper_states": spaces.Box(low=0.0, high=0.1, shape=(2,), dtype=np.float64),
            "robot0_gripper_qpos": spaces.Box(low=0.0, high=0.1, shape=(2,), dtype=np.float64),
            "robot0_eef_quat": spaces.Box(low=-1.01, high=1.01, shape=(4,), dtype=np.float64),
            "ee_quat": spaces.Box(low=-1.01, high=1.01, shape=(4,), dtype=np.float64),
            "ee_ori": spaces.Box(low=-4.0, high=4.0, shape=(3,), dtype=np.float64),
            "ee_states": spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float64),
        }
        if self.render_depth:
            obs_dict["agentic_depth"] = spaces.Box(low=0.0, high=10.0, shape=agentic_cam_size, dtype=np.float32)
            obs_dict["wrist_depth"] = spaces.Box(low=0.0, high=10.0, shape=wrist_cam_size, dtype=np.float32)
            obs_dict["agentview_depth"] = spaces.Box(low=0.0, high=10.0, shape=agentic_cam_size, dtype=np.float32)
            obs_dict["eye_in_hand_depth"] = spaces.Box(low=0.0, high=10.0, shape=wrist_cam_size, dtype=np.float32)
        self.observation_space = spaces.Dict(obs_dict)

        # Franka Emika Panda actuator torque limits (Nm) and safety contact force limit (N)
        self.panda_torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)
        self.max_safe_contact_force = 150.0  # N (ISO/TS 15066 safety threshold)
        self.peak_contact_force = 0.0
        self.peak_joint_torques = np.zeros(7, dtype=np.float64)
        self.force_limit_exceeded = False
        self.torque_limit_exceeded = False
        self.safety_violation = False

        # Mid-trajectory dynamic perturbation support
        self.dynamic_perturbation_step: int | None = None
        self.dynamic_perturbation_offset: np.ndarray | None = None
        self.dynamic_perturbation_trigger: str = "step"  # "step" | "pre_grasp" | "proximity"
        self.dynamic_proximity_threshold: float = 0.08    # meters
        self.dynamic_perturbed_applied: bool = False

        self._setup_task()
        self.reset()

    def _cache_model_constants(self) -> None:
        """Cache joint, actuator, body, and camera indexing constants for the active model."""
        m = self.model
        self.control_dt = self.n_substeps * m.opt.timestep
        self.max_tcp_step = self.max_tcp_speed * self.control_dt
        self.max_finger_substep = self.max_finger_speed * m.opt.timestep

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

        self.free_qpos_to_dof: dict[int, int] = {}
        self.free_body_qposadr: dict[int, int] = {}
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_qpos_to_dof[int(m.jnt_qposadr[j])] = int(m.jnt_dofadr[j])
                self.free_body_qposadr[int(m.jnt_bodyid[j])] = int(m.jnt_qposadr[j])
        self.static_body_ids: set[int] = set()

        self.cam_agentic_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "agentic_cam")
        self.cam_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

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

    @property
    def actuator_torques(self) -> np.ndarray:
        """Actuator motor torques (N*m) on the 7 arm joints (tau_act)."""
        return np.array([self.data.actuator_force[a] for a in self.arm_actuator_ids], dtype=np.float64)

    @property
    def joint_torques(self) -> np.ndarray:
        """Alias for actuator_torques (tau_act in N*m)."""
        return self.actuator_torques

    @property
    def external_joint_torques(self) -> np.ndarray:
        """Constraint/contact torques (N*m) projected onto the 7 arm joints (tau_ext)."""
        return np.array(self.data.qfrc_constraint[self.arm_dof_adr], dtype=np.float64)

    @property
    def gripper_actuator_force(self) -> float:
        """Actuator tendon force (N) on the gripper."""
        return float(self.data.actuator_force[self.gripper_actuator_id])

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

    def get_sim_state(self) -> np.ndarray:
        """Full physics state (float64) followed by the gripper's slew-limited opening command (m). After
        `reset(seed)` with the recorded seed (which re-applies that episode's model perturbations),
        `set_sim_state` restores the scene for re-rendering or inspection; task bookkeeping such as success
        counters is not part of it."""
        state = np.empty(mujoco.mj_stateSize(self.model, SIM_STATE_SPEC))
        mujoco.mj_getState(self.model, self.data, state, SIM_STATE_SPEC)
        return np.append(state, self.finger_cmd)

    def set_sim_state(self, state: np.ndarray) -> None:
        """Inverse of `get_sim_state`, with support for raw MuJoCo state arrays."""
        state = np.asarray(state, dtype=np.float64)
        full_sz = mujoco.mj_stateSize(self.model, SIM_STATE_SPEC)
        if len(state) == full_sz + 1:
            mujoco.mj_setState(self.model, self.data, state[:-1], SIM_STATE_SPEC)
            self.finger_cmd = float(state[-1])
        elif len(state) == full_sz:
            mujoco.mj_setState(self.model, self.data, state, SIM_STATE_SPEC)
            self.finger_cmd = self.gripper_width
        elif len(state) == self.model.nq + self.model.nv:
            self.data.qpos[:] = state[: self.model.nq]
            self.data.qvel[:] = state[self.model.nq :]
            self.finger_cmd = self.gripper_width
        elif len(state) == self.model.nq:
            self.data.qpos[:] = state
            self.data.qvel[:] = 0.0
            self.finger_cmd = self.gripper_width
        else:
            raise ValueError(
                f"State vector length {len(state)} does not match full state ({full_sz + 1}), "
                f"spec state ({full_sz}), or qpos+qvel ({self.model.nq + self.model.nv})"
            )
        mujoco.mj_forward(self.model, self.data)

    @property
    def sim(self) -> MjSimAdapter:
        """Shim providing robosuite / robomimic / LIBERO MjSim interface."""
        if not hasattr(self, "_sim_adapter") or self._sim_adapter is None:
            self._sim_adapter = MjSimAdapter(self)
        return self._sim_adapter

    def get_model_xml(self) -> str:
        """Return the XML string of the MuJoCo scene model."""
        if hasattr(self, "_model_xml_str") and self._model_xml_str:
            return self._model_xml_str
        return self.xml_path.read_text(encoding="utf-8")

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
            "joint_torques": self.actuator_torques,
            "actuator_torques": self.actuator_torques,
            "tcp_pose": self.tcp_pose,
            "gripper_width": np.array([self.gripper_width], dtype=np.float64),
            "robot_scene_force": np.array([self.robot_scene_force({getattr(self, "target_id", -1)})], dtype=np.float64),
        }

        # Standard Robomimic & LIBERO aliases:
        obs["agentview_image"] = agentic_rgb
        obs["agentview_rgb"] = agentic_rgb
        obs["robot0_eye_in_hand_image"] = wrist_rgb
        obs["eye_in_hand_rgb"] = wrist_rgb
        obs["joint_states"] = self.joint_positions
        obs["robot0_joint_pos"] = self.joint_positions
        obs["robot0_joint_vel"] = self.joint_velocities
        obs["robot0_joint_torque"] = self.actuator_torques
        obs["robot0_eef_pos"] = self.tcp_position
        obs["ee_pos"] = self.tcp_position
        obs["gripper_states"] = np.array([self.gripper_width, self.gripper_width], dtype=np.float64)
        obs["robot0_gripper_qpos"] = np.array([self.gripper_width, self.gripper_width], dtype=np.float64)

        quat_wxyz = self.tcp_pose[3:]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        obs["robot0_eef_quat"] = quat_xyzw
        obs["ee_quat"] = quat_xyzw
        axis_angle = _quat_wxyz_to_axisangle(quat_wxyz)
        obs["ee_ori"] = axis_angle
        obs["ee_states"] = np.concatenate([self.tcp_position, axis_angle])

        if self.render_depth:
            self.renderer_agentic.enable_depth_rendering()
            self.renderer_agentic.update_scene(self.data, camera=self.cam_agentic_id)
            obs["agentic_depth"] = self.renderer_agentic.render().copy()
            self.renderer_agentic.disable_depth_rendering()

            self.renderer_wrist.enable_depth_rendering()
            self.renderer_wrist.update_scene(self.data, camera=self.cam_wrist_id)
            obs["wrist_depth"] = self.renderer_wrist.render().copy()
            self.renderer_wrist.disable_depth_rendering()

            obs["agentview_depth"] = obs["agentic_depth"]
            obs["eye_in_hand_depth"] = obs["wrist_depth"]

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
        # at no more than max_finger_speed. If invert_gripper_action is True (robosuite/LIBERO convention),
        # -1.0 commands open (0.04 m) and +1.0 commands closed (0.0 m).
        grip_cmd = -float(action[-1]) if self.invert_gripper_action else float(action[-1])
        finger_target = 0.5 * GRIPPER_OPEN_WIDTH * (float(np.clip(grip_cmd, -1.0, 1.0)) + 1.0)
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
        info = {
            "success": terminated,
            "step": self.current_step,
            "robot_scene_force": self.robot_scene_force({getattr(self, "target_id", -1)}),
            "peak_contact_force": self.peak_contact_force,
            "force_limit_exceeded": self.force_limit_exceeded,
            "torque_limit_exceeded": self.torque_limit_exceeded,
            "safety_violation": self.safety_violation,
            "dynamic_perturbed": self.dynamic_perturbed_applied,
        }

        return CompatStepResult(obs, reward, terminated, truncated, info)

    def apply_object_displacement(self, offset: list[float] | np.ndarray) -> bool:
        """Displace the active manipulated target object mid-trajectory and zero its velocities."""
        target_qposadr = getattr(self, "target_qposadr", None)
        if target_qposadr is None:
            return False
        curr_pos = self.data.qpos[target_qposadr : target_qposadr + 3].copy()
        offset_arr = np.asarray(offset, dtype=np.float64)
        if offset_arr.shape != (3,):
            raise ValueError(f"Displacement offset must have shape (3,), got {offset_arr.shape}")
        self.data.qpos[target_qposadr : target_qposadr + 3] = curr_pos + offset_arr
        if target_qposadr in self.free_qpos_to_dof:
            dof = self.free_qpos_to_dof[target_qposadr]
            self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.dynamic_perturbed_applied = True
        return True

    def _after_step(self) -> None:
        """Hook for per-step bookkeeping, force/torque safety monitoring, and dynamic perturbations."""
        # 1. Update peak contact force and check safety limits
        force = float(np.linalg.norm(self.robot_scene_force({getattr(self, "target_id", -1)})))
        if force > self.peak_contact_force:
            self.peak_contact_force = force
        if force > self.max_safe_contact_force:
            self.force_limit_exceeded = True

        # 2. Update peak actuator joint torques and check saturation limits
        torques = np.abs(self.actuator_torques)
        self.peak_joint_torques = np.maximum(self.peak_joint_torques, torques)
        if np.any(torques > self.panda_torque_limits):
            self.torque_limit_exceeded = True

        if self.force_limit_exceeded or self.torque_limit_exceeded:
            self.safety_violation = True

        # 3. Handle mid-trajectory dynamic perturbation if configured
        if not self.dynamic_perturbed_applied and self.dynamic_perturbation_offset is not None:
            should_trigger = False
            if self.dynamic_perturbation_trigger == "step" and self.dynamic_perturbation_step is not None:
                should_trigger = self.current_step >= self.dynamic_perturbation_step
            elif self.dynamic_perturbation_trigger in ("proximity", "pre_grasp"):
                target_id = getattr(self, "target_id", -1)
                if target_id != -1:
                    target_pos = self.data.xpos[target_id]
                    dist_to_obj = float(np.linalg.norm(self.tcp_position[:2] - target_pos[:2]))
                    if dist_to_obj <= self.dynamic_proximity_threshold:
                        should_trigger = True
            
            if should_trigger:
                self.apply_object_displacement(self.dynamic_perturbation_offset)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.dynamic_perturbed_applied = False
        self.peak_contact_force = 0.0
        self.peak_joint_torques = np.zeros(7, dtype=np.float64)
        self.force_limit_exceeded = False
        self.torque_limit_exceeded = False
        self.safety_violation = False

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
        return CompatResetResult(obs, {})

    def set_init_state(self, init_state: np.ndarray) -> dict[str, Any]:
        """Restore simulator state from fixed initial state array and return fresh observations."""
        s = np.asarray(init_state, dtype=np.float64)
        if s.ndim == 2 and s.shape[0] == 1:
            s = s[0]
        self.set_sim_state(s)
        return self.get_observation()

    def regenerate_obs_from_state(self, state: np.ndarray) -> dict[str, Any]:
        """Restore simulator state and return fresh observations (LIBERO metric.py API)."""
        return self.set_init_state(state)

    def reset_from_xml_string(self, xml_string: str) -> None:
        """Reload MuJoCo model from an XML string (Robomimic / LIBERO replay API)."""
        self._model_xml_str = xml_string
        try:
            new_model = _load_model_from_xml_string(xml_string, self.xml_path.parent)
        except Exception:
            new_model = mujoco.MjModel.from_xml_string(xml_string)
        self.model = new_model
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self._cache_model_constants()
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
        self.renderer_agentic = mujoco.Renderer(self.model, height=self.agentic_cam_size[0], width=self.agentic_cam_size[1])
        self.renderer_wrist = mujoco.Renderer(self.model, height=self.wrist_cam_size[0], width=self.wrist_cam_size[1])
        self._setup_task()
        if hasattr(self, "_calibrate_goal") and getattr(self, "target_qposadr", None) is not None:
            self._calibrate_goal()
        self.reset()

    def _get_obs(self) -> dict[str, Any]:
        """Gym / Robomimic alias for get_observation()."""
        return self.get_observation()

    def _get_observations(self) -> dict[str, Any]:
        """Robosuite / LIBERO alias for get_observation()."""
        return self.get_observation()

    def _check_success(self) -> bool:
        """Robosuite / LIBERO alias for check_success()."""
        return self.check_success()

    def is_success(self) -> dict[str, bool]:
        """Robomimic style success check dictionary."""
        return {"task": bool(self.check_success())}

    def _post_process(self) -> None:
        """LIBERO / robosuite post-process hook (no-op)."""
        pass

    def _update_observables(self, force: bool = False) -> None:
        """LIBERO / robosuite observables update hook (no-op)."""
        pass

    @property
    def robots(self) -> list[Any]:
        """LIBERO / robosuite robot list property."""
        if not hasattr(self, "_robot_adapters") or self._robot_adapters is None:
            self._robot_adapters = [RobotAdapter(self)]
        return self._robot_adapters

    @property
    def obj_of_interest(self) -> list[str]:
        """LIBERO obj_of_interest property."""
        target = getattr(self, "target_name", None)
        return [str(target)] if target is not None else []

    def set_state(self, mujoco_state: np.ndarray) -> None:
        """Alias for set_sim_state to match LIBERO / Robosuite API."""
        self.set_sim_state(mujoco_state)

    @property
    def version(self) -> str:
        """Robomimic env version property."""
        import afford_pro
        return getattr(afford_pro, "__version__", "0.1.0")

    @property
    def _visualizations(self) -> dict:
        """LIBERO / robosuite visualizations property."""
        return {}

    @property
    def action_dimension(self) -> int:
        """Robomimic action dimension property."""
        return 7

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Robosuite action spec property (low, high)."""
        return (self.action_space.low, self.action_space.high)

    def seed(self, seed: int | None = None) -> list[int]:
        """Legacy gym seed() method."""
        if seed is not None:
            self.reset(seed=seed)
            return [seed]
        return []

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
