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
    def __init__(self, *args, **kwargs) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_rendering(monkeypatch):
    monkeypatch.setattr(base_env.mujoco, "Renderer", _NoRenderer)
    monkeypatch.setattr(base_env.BaseFrankaEnv, "get_observation",
                        lambda self: {
                            "agentic_rgb": np.zeros((*self.agentic_cam_size, 3), dtype=np.uint8),
                            "wrist_rgb": np.zeros((*self.wrist_cam_size, 3), dtype=np.uint8),
                            "tcp_pose": self.tcp_pose,
                            "joint_positions": self.joint_positions,
                            "joint_velocities": self.joint_velocities,
                            "gripper_width": np.array([self.gripper_width], dtype=np.float64),
                        })


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
        assert set(env_meta) == {"env_name", "type", "env_kwargs"} and env_meta["env_name"] == TASK
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
        assert demo["dones"][-1] == 1 and demo["rewards"][-1] == 1.0
    env = afford_pro.make(TASK, **{k: v for k, v in env_meta["env_kwargs"].items() if k != "factory"})
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

