"""Generation: provenance and rejected seeds are recorded, existing files are never replaced silently."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

import afford_pro.envs.base_franka_env as base_env
from afford_pro.datasets.generation import GenerationConfig, episode_seed_base, generate_task_dataset

TASK = "tabletop_red_touch"  # 30-step reach task: the cheapest real oracle episode


class _GradientRenderer:
    """Headless stand-in for mujoco.Renderer: a non-constant image, so image-content checks pass without OpenGL."""

    def __init__(self, model=None, height: int = 64, width: int = 64, *args, **kwargs) -> None:
        self.image = np.broadcast_to(np.linspace(20, 220, width, dtype=np.uint8)[None, :, None], (height, width, 3)).copy()

    def update_scene(self, *args, **kwargs) -> None:
        pass

    def render(self) -> np.ndarray:
        return self.image.copy()

    def close(self) -> None:
        pass


@pytest.fixture(scope="module", autouse=True)
def _headless_renderer():
    saved = base_env.mujoco.Renderer
    base_env.mujoco.Renderer = _GradientRenderer
    yield
    base_env.mujoco.Renderer = saved


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    result = generate_task_dataset(GenerationConfig(task_id=TASK, output_dir=out, num_demos=2), log=None)
    return out, result


def test_written_file_is_valid_and_traceable(generated):
    out, result = generated
    assert result.status == "written" and result.collected == 2
    assert result.validation_status in ("VALID", "VALID_WITH_WARNINGS")
    with h5py.File(result.path, "r") as f:
        prov = json.loads(f["data"].attrs["provenance"])
        gen = json.loads(f["data"].attrs["generation"])
        assert prov["packages"]["mujoco"] and "commit" in prov["git"] and prov["asset_sha256"]
        assert prov["base_seed"] == episode_seed_base(82)
        assert gen["attempts"] == result.attempts and gen["failed_seeds"] == result.failed_seeds
        assert "final_state" in f["data/demo_0"].attrs
        assert f["data/demo_0"].attrs["seed"] == episode_seed_base(82)


def test_existing_valid_file_is_skipped(generated):
    out, result = generated
    again = generate_task_dataset(GenerationConfig(task_id=TASK, output_dir=out, num_demos=2), log=None)
    assert again.status == "skipped_valid"


def test_existing_invalid_file_is_not_replaced(tmp_path):
    path = tmp_path / f"{TASK}_nominal.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data")
    before = path.read_bytes()
    result = generate_task_dataset(GenerationConfig(task_id=TASK, output_dir=tmp_path, num_demos=1), log=None)
    assert result.status == "refused_existing" and path.read_bytes() == before


def test_incomplete_run_does_not_take_the_final_name(tmp_path):
    cfg = GenerationConfig(task_id=TASK, output_dir=tmp_path, num_demos=1, max_demo_steps=1)
    result = generate_task_dataset(cfg, log=None)
    assert result.status == "incomplete" and result.collected == 0
    assert len(result.over_length_seeds) == result.attempts
    assert not (tmp_path / f"{TASK}_nominal.hdf5").exists()
    assert result.path.endswith(".incomplete.hdf5")


def test_file_appearing_during_the_run_is_not_replaced(tmp_path, monkeypatch):
    import afford_pro.datasets.generation as gen

    original = gen._collect

    def collect_then_squat(cfg, env, oracle, writer, base_seed, result, log):
        original(cfg, env, oracle, writer, base_seed, result, log)
        (tmp_path / f"{TASK}_nominal.hdf5").write_bytes(b"written by another process")

    monkeypatch.setattr(gen, "_collect", collect_then_squat)
    result = gen.generate_task_dataset(GenerationConfig(task_id=TASK, output_dir=tmp_path, num_demos=1), log=None)
    assert result.status == "refused_existing" and result.path.endswith(".conflict.hdf5")
    assert (tmp_path / f"{TASK}_nominal.hdf5").read_bytes() == b"written by another process"
