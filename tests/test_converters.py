"""Round-trip tests for the Robomimic HDF5 <-> LeRobot v2.0 and Robomimic HDF5 <-> RLDS converters."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from afford_pro.datasets.converters import (  # noqa: E402
    DatasetMeta, Episode, episode_to_rlds, load_rlds_npz, read_lerobot, read_robomimic, rlds_to_episode,
    save_rlds_npz, write_lerobot, write_robomimic,
)

META = DatasetMeta(env_name="industrial_shaft_assembly", task_id="industrial_shaft_assembly",
                   instruction="Insert the shaft into the bore.", train_keys=["demo_0", "demo_1"], valid_keys=["demo_2"],
                   camera={"camera_agentic_K": [[500.0, 0.0, 329.5], [0.0, 500.0, 319.5], [0.0, 0.0, 1.0]]})


def _episode(seed: int, T: int) -> Episode:
    rng = np.random.default_rng(seed)
    dones = np.zeros(T, dtype=np.uint8)
    dones[-1] = 1
    return Episode(
        actions=rng.uniform(-1, 1, (T, 7)).astype(np.float32),
        obs={
            "agentic_rgb": rng.integers(0, 256, (T, 12, 10, 3), dtype=np.uint8),
            "wrist_rgb": rng.integers(0, 256, (T, 6, 8, 3), dtype=np.uint8),
            "joint_positions": rng.normal(size=(T, 7)),
            "joint_velocities": rng.normal(size=(T, 7)),
            "tcp_pose": rng.normal(size=(T, 7)),
            "gripper_width": rng.uniform(0, 0.04, (T, 1)),
            "agentic_depth": rng.uniform(0.2, 2.0, (T, 12, 10)).astype(np.float32),
        },
        rewards=dones.astype(np.float32),
        dones=dones,
    )


@pytest.fixture()
def hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "src.hdf5"
    write_robomimic(path, META, iter([_episode(0, 5), _episode(1, 3), _episode(2, 4)]))
    return path


def _assert_same(a: Episode, b: Episode, float_tol: float) -> None:
    assert set(a.obs) == set(b.obs)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.rewards, b.rewards)
    np.testing.assert_array_equal(a.dones, b.dones)
    for k in a.obs:
        if a.obs[k].dtype == np.uint8:
            np.testing.assert_array_equal(a.obs[k], b.obs[k])
        else:
            np.testing.assert_allclose(a.obs[k].reshape(len(a), -1), b.obs[k].reshape(len(b), -1), rtol=0, atol=float_tol)


def test_lerobot_roundtrip(hdf5: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    meta, episodes = read_robomimic(hdf5)
    root = tmp_path / "lerobot"
    assert write_lerobot(root, meta, episodes) == 3
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.0" and info["total_frames"] == 12 and info["fps"] == 20
    assert info["features"]["observation.state"]["shape"] == [22]
    assert info["features"]["observation.images.workcell"]["dtype"] == "image"
    assert (root / "data" / "chunk-000" / "episode_000002.parquet").exists()

    back = tmp_path / "back.hdf5"
    meta2, episodes2 = read_lerobot(root)
    write_robomimic(back, meta2, episodes2)
    meta3, eps3 = read_robomimic(back)
    assert (meta3.instruction, meta3.task_id, meta3.train_keys, meta3.valid_keys, meta3.camera) == \
        (META.instruction, META.task_id, META.train_keys, META.valid_keys, META.camera)
    assert info["features"]["observation.agentic_depth"]["shape"] == [12, 10]
    # observation.state is float32 in LeRobot, so float64 proprioception round-trips to float32 precision.
    for a, b in zip(read_robomimic(hdf5)[1], eps3):
        _assert_same(a, b, float_tol=1e-6)


def test_rlds_roundtrip(hdf5: Path, tmp_path: Path) -> None:
    meta, episodes = read_robomimic(hdf5)
    for ep in episodes:
        rlds = episode_to_rlds(ep, meta.instruction, episode_id=f"src/{ep.key}")
        steps = rlds["steps"]
        assert [s["is_first"] for s in steps] == [True] + [False] * (len(ep) - 1)
        assert [s["is_last"] for s in steps] == [False] * (len(ep) - 1) + [True]
        assert steps[-1]["is_terminal"] and not any(s["is_terminal"] for s in steps[:-1])
        assert steps[0]["observation"]["state"].shape == (8,)
        assert steps[0]["observation"]["image"].dtype == np.uint8
        assert all(s["language_instruction"] == META.instruction for s in steps)
        path = tmp_path / f"{ep.key}.npz"
        save_rlds_npz(path, rlds)
        _assert_same(ep, rlds_to_episode(load_rlds_npz(path)), float_tol=0.0)


def test_cli_scripts(hdf5: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    for script, out in (("convert_to_lerobot.py", tmp_path / "lr"), ("convert_to_rlds.py", tmp_path / "rlds")):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--hdf5", str(hdf5), "--out", str(out)],
                       check=True, capture_output=True)
    assert (tmp_path / "lr" / META.task_id / "meta" / "info.json").exists()
    assert len(list((tmp_path / "rlds").glob("*.npz"))) == 3
