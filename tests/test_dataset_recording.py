"""Demonstration recording: reward/done alignment, robomimic env metadata, simulator states, and the depth flag."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro  # noqa: E402
import afford_pro.envs.base_franka_env as base_env  # noqa: E402

TASK = "tabletop_sort_red"


class _NoRenderer:
    def __init__(self, model=None, height: int = 64, width: int = 64, *args, **kwargs) -> None:
        self.height = height
        self.width = width

    def update_scene(self, *args, **kwargs) -> None:
        pass

    def render(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def enable_depth_rendering(self) -> None:
        pass

    def disable_depth_rendering(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_rendering(monkeypatch):
    monkeypatch.setattr(base_env.mujoco, "Renderer", _NoRenderer)


def _generator():
    spec = importlib.util.spec_from_file_location("generate_all_150_tasks", ROOT / "scripts" / "generate_all_150_tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rewards_and_dones_follow_each_action():
    """rewards[t] and dones[t] come from the step that executes actions[t]: a successful demo ends with done."""
    env = afford_pro.make(TASK)
    oracle = afford_pro.get_oracle(TASK, env)
    actions, obs, rewards, dones = oracle.generate_episode(seed=3)
    assert env.check_success()
    T = len(actions)
    assert len(rewards) == len(dones) == len(obs["tcp_pose"]) == len(oracle.last_states) == T
    assert dones[-1] and not any(dones[:-1])
    assert rewards[-1] == 1.0 and sum(rewards) == 1.0


def test_recorded_dataset_is_robomimic_loadable_and_restorable(tmp_path: Path):
    """The batch generator writes robomimic env metadata, per-step simulator states, and the reset seed; restoring
    a recorded state after reset(seed) reproduces the recorded observation."""
    collected, steps, _ = _generator().generate_task_demos(TASK, num_demos=2, output_dir=tmp_path, base_seed=7)
    assert collected == 2
    path = tmp_path / f"{TASK}_nominal.hdf5"
    with h5py.File(path, "r") as f:
        # robomimic 0.2.0 reads exactly this at startup (utils/file_utils.get_env_metadata_from_dataset).
        env_meta = json.loads(f["data"].attrs["env_args"])
        assert {"env_name", "type", "env_kwargs"}.issubset(set(env_meta)) and env_meta["env_name"] == TASK
        assert env_meta["env_kwargs"]["perturbation_mode"] == "nominal"
        # Robomimic & LIBERO metadata on root
        assert "model_file" in f["data"].attrs and "<mujoco" in f["data"].attrs["model_file"]
        assert "problem_info" in f["data"].attrs
        prob_info = json.loads(f["data"].attrs["problem_info"])
        assert prob_info["problem_name"] == TASK and "language_instruction" in prob_info
        assert "num_demos" in f["data"].attrs and int(f["data"].attrs["num_demos"]) == 2
        assert "language_instructions" in f["data"].attrs
        assert len(json.loads(f["data"].attrs["language_instructions"])) >= 1

        demo = f["data/demo_1"]
        seed = int(demo.attrs["seed"])
        assert "language_instruction" in demo.attrs
        assert "model_file" in demo.attrs and "<mujoco" in demo.attrs["model_file"]
        assert "init_state" in demo.attrs and len(demo.attrs["init_state"]) == demo["states"].shape[1]

        # Robomimic / LIBERO observation schemas and aliases
        assert "ee_states" in demo["obs"] and "joint_states" in demo["obs"]
        assert demo["obs/ee_states"].shape == (len(demo["actions"]), 6)  # 6-D pos + axis-angle
        assert demo["obs/gripper_states"].shape == (len(demo["actions"]), 2)  # 2-D gripper qpos
        assert demo["obs/robot0_gripper_qpos"].shape == (len(demo["actions"]), 2)
        assert demo["obs/robot0_eef_pos"].shape == (len(demo["actions"]), 3)
        assert demo["obs/robot0_eef_quat"].shape == (len(demo["actions"]), 4)

        np.testing.assert_allclose(demo["obs/ee_pos"][()], demo["obs/tcp_pose"][:, :3], atol=1e-6)
        np.testing.assert_allclose(demo["obs/robot0_eef_pos"][()], demo["obs/tcp_pose"][:, :3], atol=1e-6)
        np.testing.assert_array_equal(demo["obs/joint_states"][()], demo["obs/joint_positions"][()])

        # next_obs group for offline RL (BCQ, CQL, TD3-BC)
        assert "next_obs" in demo
        assert "tcp_pose" in demo["next_obs"]
        assert demo["next_obs/tcp_pose"].shape == demo["obs/tcp_pose"].shape
        np.testing.assert_array_equal(demo["next_obs/tcp_pose"][:-1], demo["obs/tcp_pose"][1:])

        states, tcp, joints = demo["states"][()], demo["obs/tcp_pose"][()], demo["obs/joint_positions"][()]
        assert states.dtype == np.float64 and len(states) == len(demo["actions"])
    env = afford_pro.make(TASK, **env_meta["env_kwargs"])
    env.reset(seed=seed)
    for t in (0, len(states) // 2, len(states) - 1):
        env.set_sim_state(states[t])
        np.testing.assert_allclose(env.tcp_pose, tcp[t], atol=1e-9)
        np.testing.assert_allclose(env.joint_positions, joints[t], atol=1e-9)



def test_batch_generator_depth_flag(tmp_path: Path):
    """--depth makes the batch generator build depth-rendering environments and record camera parameters."""
    collected, _, _ = _generator().generate_task_demos(TASK, num_demos=1, output_dir=tmp_path, base_seed=7, depth=True)
    assert collected == 1
    with h5py.File(tmp_path / f"{TASK}_nominal.hdf5", "r") as f:
        attrs = f["data"].attrs
        assert json.loads(attrs["env_args"])["env_kwargs"]["render_depth"] is True
        assert attrs["camera_agentic_K"].shape == (3, 3) and attrs["camera_wrist_K"].shape == (3, 3)


def test_export_and_restore_fixed_initial_states(tmp_path: Path):
    """Initial states exporter saves valid settled sim states, and loading them restores physics state."""
    from scripts.export_initial_states import export_task_initial_states

    npz_path = export_task_initial_states(TASK, num_states=3, output_dir=tmp_path, base_seed=123)
    assert npz_path.exists()

    loaded = afford_pro.get_initial_states(TASK, init_states_dir=tmp_path)
    assert loaded is not None and len(loaded) == 3
    assert loaded.dtype == np.float64

    env = afford_pro.make(TASK)
    env.reset(seed=999)  # reset to a completely different seed
    # Now set to fixed initial state 1
    env.set_sim_state(loaded[1])
    # Sim state read back matches exactly
    current_state = env.get_sim_state()
    np.testing.assert_allclose(current_state, loaded[1], atol=1e-9)


def test_invert_gripper_action_adapter():
    """invert_gripper_action maps robosuite/LIBERO -1 (open) / +1 (close) to Franka finger commands."""
    env_default = afford_pro.make(TASK, invert_gripper_action=False)
    env_inverted = afford_pro.make(TASK, invert_gripper_action=True)

    env_default.reset(seed=1)
    env_inverted.reset(seed=1)

    # In default convention: +1 is open, -1 is close
    env_default.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    w_default_open = env_default.gripper_width

    # In inverted convention: -1 is open, +1 is close
    env_inverted.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    w_inverted_open = env_inverted.gripper_width

    np.testing.assert_allclose(w_default_open, w_inverted_open, atol=1e-3)


def test_gym_auto_registration():
    """Benchmark tasks are registered with gymnasium for drop-in gym.make() and robomimic EnvGym support."""
    import gymnasium as gym
    env = gym.make(TASK)
    obs, info = env.reset(seed=42)
    assert "tcp_pose" in obs and "joint_positions" in obs
    env.close()


def test_gym_make_with_robomimic_framework_kwargs():
    """gym.make accepts robomimic dataset metadata kwargs (factory, perturbation_mode, lang) without TypeError."""
    import gymnasium as gym
    env = gym.make(
        TASK,
        factory="afford_pro.make",
        perturbation_mode="nominal",
        lang="pick and place the block",
        use_image_obs=True,
    )
    obs, info = env.reset(seed=12)
    assert "tcp_pose" in obs and "agentview_rgb" in obs
    step_obs, reward, term, trunc, info = env.step(np.zeros(7))
    assert isinstance(step_obs, dict) and "robot0_eef_pos" in step_obs
    env.close()


def test_libero_and_robomimic_adapter_methods():
    """Environment provides standard robosuite / LIBERO / robomimic inspection & reset shims."""
    env = afford_pro.make(TASK)
    env.reset(seed=10)

    # 1. XML model inspection
    xml = env.sim.model.get_xml()
    assert isinstance(xml, str) and "<mujoco" in xml

    # 2. Simulator state retrieval & reset_to
    state = env.sim.get_state().flatten()
    assert state.ndim == 1 and len(state) > 0

    # 3. set_init_state & regenerate_obs_from_state (LIBERO metric.py API)
    obs1 = env.set_init_state(state)
    assert isinstance(obs1, dict) and "agentview_image" in obs1 and "joint_states" in obs1
    obs2 = env.regenerate_obs_from_state(state)
    assert isinstance(obs2, dict) and "robot0_joint_pos" in obs2

    # 4. sim forward, step, and set_state_from_flattened
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env.sim.step()

    # 5. Robomimic / robosuite aliases
    assert env._check_success() is False
    assert env.is_success() == {"task": False}
    assert isinstance(env.robots, list) and len(env.robots) > 0
    assert env.robots[0].robot_model.naming_prefix == "robot0_"
    assert isinstance(env.obj_of_interest, list)
    assert env.version == "0.1.0"
    assert env.action_dimension == 7
    low, high = env.action_spec
    assert len(low) == 7 and len(high) == 7

    # 6. Model reloading and state setting
    env.set_state(state)
    env.reset_from_xml_string(xml)
    assert env.check_success() is False
    env.close()


def test_hdf5_writer_libero_and_robomimic_metadata(tmp_path):
    """Dataset writer includes all canonical LIBERO and robomimic attributes."""
    from afford_pro.datasets.hdf5_writer import RobomimicHDF5Writer, make_env_args

    hdf5_file = tmp_path / "metadata_test.hdf5"
    writer = RobomimicHDF5Writer(
        file_path=hdf5_file,
        env_name=TASK,
        task_id=TASK,
        instruction="sort the red block",
        env_args=make_env_args(TASK),
    )
    writer.add_demonstration(
        actions=np.zeros((3, 7)),
        observations={
            "agentic_rgb": np.zeros((3, 64, 64, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((3, 64, 64, 3), dtype=np.uint8),
            "tcp_pose": np.zeros((3, 7)),
            "joint_positions": np.zeros((3, 7)),
            "joint_velocities": np.zeros((3, 7)),
            "gripper_width": np.zeros((3, 1)),
        },
        rewards=[0.0, 0.0, 1.0],
        dones=[False, False, True],
        states=np.zeros((3, 10)),
    )
    writer.close()

    with h5py.File(hdf5_file, "r") as f:
        data = f["data"]
        # Robomimic checks
        assert "env_args" in data.attrs
        meta = json.loads(data.attrs["env_args"])
        assert meta["env_name"] == TASK and meta["type"] == 2 and meta["env_version"] == "0.1.0"
        assert meta["env_kwargs"]["factory"] == "afford_pro.make"
        # LIBERO checks
        assert data.attrs["env"] == TASK
        assert data.attrs["tag"] == "libero-v1"
        assert data.attrs["repository_version"] == "0.1.0"
        assert data.attrs["macros_image_convention"] == "opencv"
        assert "problem_info" in data.attrs
        prob = json.loads(data.attrs["problem_info"])
        assert prob["problem_name"] == TASK
        # Episode demo checks
        ep = data["demo_0"]
        assert ep.attrs["num_samples"] == 3
        assert "init_state" in ep.attrs
        assert "actions" in ep and "states" in ep and "rewards" in ep and "dones" in ep
        assert "obs/agentview_image" in ep and "obs/agentview_rgb" in ep
        assert "obs/robot0_eye_in_hand_image" in ep and "obs/eye_in_hand_rgb" in ep
        assert "obs/robot0_joint_pos" in ep and "obs/joint_states" in ep
        assert "obs/robot0_eef_pos" in ep and "obs/ee_pos" in ep
        assert "obs/robot0_gripper_qpos" in ep and "obs/gripper_states" in ep
        assert "obs/ee_states" in ep and "obs/ee_ori" in ep
        assert "next_obs/agentview_image" in ep


def test_dual_api_reset_and_step_unpacking():
    """reset() and step() support both 4-tuple / single-assign (old gym / robomimic) and 5-tuple / 2-tuple (gymnasium)."""
    env = afford_pro.make(TASK)

    # Old Gym / Robomimic: single return value from reset()
    reset_obs = env.reset()
    assert isinstance(reset_obs, dict)
    assert "tcp_pose" in reset_obs and "agentview_rgb" in reset_obs

    # Old Gym / Robomimic: 4 return values from step()
    o4, r4, d4, inf4 = env.step(np.zeros(7))
    assert isinstance(o4, dict)
    assert isinstance(r4, float) and isinstance(d4, bool) and isinstance(inf4, dict)

    # Gymnasium: 2 return values from reset()
    o2, inf2 = env.reset(seed=5)
    assert isinstance(o2, dict) and isinstance(inf2, dict)

    # Gymnasium: 5 return values from step()
    o5, r5, term5, trunc5, inf5 = env.step(np.zeros(7))
    assert isinstance(o5, dict)
    assert isinstance(r5, float) and isinstance(term5, bool) and isinstance(trunc5, bool)
    env.close()

