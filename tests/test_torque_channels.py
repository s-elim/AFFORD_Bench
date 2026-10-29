"""Tests for actuator motor torques (tau_act) and contact force channels in AFFORD-Bench."""
from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

import afford_pro
from afford_pro.datasets.converters import read_robomimic, write_lerobot
from afford_pro.datasets.hdf5_writer import RobomimicHDF5Writer
from afford_pro.datasets.validation import validate_file
from afford_pro.envs.headless import headless


def test_env_torque_properties():
    """Verify that BaseFrankaEnv provides actuator_torques, external_joint_torques, and gripper_force."""
    with afford_pro.make("tabletop_sort_red") as env:
        obs, _ = env.reset(seed=42)
        # Check properties
        tau_act = env.actuator_torques
        assert isinstance(tau_act, np.ndarray)
        assert tau_act.shape == (7,)
        assert np.isfinite(tau_act).all()
        assert np.allclose(env.joint_torques, tau_act)

        # Check correspondence with MuJoCo internal arrays
        arm_act = np.array([env.data.actuator_force[a] for a in env.arm_actuator_ids])
        assert np.allclose(tau_act, arm_act)

        # External joint torques
        tau_ext = env.external_joint_torques
        assert isinstance(tau_ext, np.ndarray)
        assert tau_ext.shape == (7,)
        assert np.isfinite(tau_ext).all()

        # Gripper force
        f_grip = env.gripper_actuator_force
        assert isinstance(f_grip, float)
        assert np.isfinite(f_grip)


def test_env_observation_torque_keys():
    """Verify that get_observation() includes torque and scene force channels and aliases."""
    with afford_pro.make("tabletop_sort_red") as env:
        obs, _ = env.reset(seed=42)
        assert "joint_torques" in obs
        assert "actuator_torques" in obs
        assert "robot0_joint_torque" in obs
        assert "robot_scene_force" in obs

        assert obs["joint_torques"].shape == (7,)
        assert obs["robot0_joint_torque"].shape == (7,)
        assert obs["robot_scene_force"].shape == (1,)
        assert np.allclose(obs["joint_torques"], obs["robot0_joint_torque"])

        # Execute a step
        action = np.zeros(7, dtype=np.float32)
        obs, _, _, _, info = env.step(action)
        assert "joint_torques" in obs
        assert "robot_scene_force" in obs
        assert np.isfinite(obs["joint_torques"]).all()
        assert np.isfinite(obs["robot_scene_force"]).all()
        assert float(obs["robot_scene_force"][0]) == pytest.approx(info["robot_scene_force"], abs=1e-5)


def test_headless_observation_torque_keys():
    """Verify that headless mode preserves torque and scene force channels."""
    with headless(), afford_pro.make("tabletop_sort_red") as env:
        obs, _ = env.reset(seed=42)
        assert "joint_torques" in obs
        assert "robot0_joint_torque" in obs
        assert "robot_scene_force" in obs
        assert obs["joint_torques"].shape == (7,)
        assert obs["robot_scene_force"].shape == (1,)


def test_hdf5_writer_and_validation():
    """Verify that RobomimicHDF5Writer records torque channels and passes file validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "tabletop_sort_red_nominal.hdf5"
        with afford_pro.make("tabletop_sort_red") as env:
            oracle = afford_pro.get_oracle("tabletop_sort_red", env)
            actions, obs, rews, dones = oracle.generate_episode(seed=42)

            writer = RobomimicHDF5Writer(
                file_path=fpath,
                env_name="tabletop_sort_red",
                task_id="tabletop_sort_red",
                instruction=env.task_spec.instruction if hasattr(env, "task_spec") else "Sort the red block.",
                model_file=env.get_model_xml(),
                atomic=False,
            )
            writer.add_demonstration(actions, obs, rews, dones, states=oracle.last_states, seed=42,
                                     final_observation=oracle.last_final_obs, final_state=oracle.last_final_state)
            writer.close()

        with h5py.File(fpath, "r") as f:
            demo_obs = f["data/demo_0/obs"]
            demo_nxt = f["data/demo_0/next_obs"]

            assert "joint_torques" in demo_obs
            assert "robot0_joint_torque" in demo_obs
            assert "robot_scene_force" in demo_obs

            # Verify hard-link
            assert demo_obs["robot0_joint_torque"].id == demo_obs["joint_torques"].id

            # Verify next_obs
            assert "joint_torques" in demo_nxt
            assert "robot_scene_force" in demo_nxt
            assert demo_nxt["joint_torques"].shape == (len(actions), 7)


def test_lerobot_conversion_with_torques():
    """Verify that convert_to_lerobot.py exports observation.joint_torques and robot_scene_force."""
    import subprocess
    import sys
    ROOT = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        h5_path = tmp_path / "tabletop_sort_red_nominal.hdf5"
        lerobot_dir = tmp_path / "lerobot_out"

        with afford_pro.make("tabletop_sort_red") as env:
            oracle = afford_pro.get_oracle("tabletop_sort_red", env)
            actions, obs, rews, dones = oracle.generate_episode(seed=42)

            writer = RobomimicHDF5Writer(
                file_path=h5_path,
                env_name="tabletop_sort_red",
                task_id="tabletop_sort_red",
                instruction="Sort the red block.",
                model_file=env.get_model_xml(),
                atomic=False,
            )
            writer.add_demonstration(actions, obs, rews, dones, states=oracle.last_states, seed=42,
                                     final_observation=oracle.last_final_obs, final_state=oracle.last_final_state)
            writer.close()

        # Run conversion in a clean subprocess to ensure C-extension isolation on Windows
        cmd = [sys.executable, str(ROOT / "scripts" / "convert_to_lerobot.py"), "--hdf5", str(h5_path), "--out", str(lerobot_dir)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"convert_to_lerobot failed: {res.stderr}"

        info_path = lerobot_dir / "tabletop_sort_red" / "meta" / "info.json"
        assert info_path.exists()
        import json
        info = json.loads(info_path.read_text())
        assert "observation.joint_torques" in info["features"]
        assert "observation.robot_scene_force" in info["features"]
        assert info["features"]["observation.joint_torques"]["shape"] == [7]
