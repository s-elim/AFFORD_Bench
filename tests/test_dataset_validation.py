"""Episode-level dataset validator: a well-formed file passes, and each corruption is reported as an error."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from afford_pro.datasets.hdf5_writer import RobomimicHDF5Writer, make_env_args
from afford_pro.datasets.validation import build_task_context, is_complete, parse_dataset_name, validate_file

TASK = "tabletop_sort_red"
T = 12


@pytest.fixture(scope="module")
def ctx():
    return build_task_context(TASK, "nominal")


def _episode(ctx, rng: np.random.Generator, T: int = T) -> dict:
    states = np.zeros((T, ctx.state_size))
    q = rng.uniform(-1.0, 1.0, size=(T, 7))
    for i, adr in enumerate(ctx.arm_qpos_adr):
        states[:, ctx.qpos_offset + adr] = q[:, i]
    states[:, ctx.qpos_offset + ctx.object_qpos_adr: ctx.qpos_offset + ctx.object_qpos_adr + 3] = rng.uniform(0.3, 0.5, 3)
    states[:, ctx.qpos_offset + ctx.object_qpos_adr + 3] = 1.0
    obs = {
        "joint_positions": q,
        "joint_velocities": np.zeros((T, 7)),
        "tcp_pose": np.tile([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], (T, 1)),
        "gripper_width": np.full((T, 1), 0.04),
    }
    rewards = np.zeros(T)
    rewards[-1] = 1.0
    dones = rewards.astype(bool)
    return dict(actions=rng.uniform(-1, 1, size=(T, 7)), observations=obs, rewards=rewards, dones=dones, states=states)


def _write(path: Path, ctx, n: int = 4, mutate=None) -> Path:
    rng = np.random.default_rng(0)
    writer = RobomimicHDF5Writer(path, TASK, TASK, ctx.instruction, env_args=make_env_args(TASK, perturbation_mode="nominal"),
                                 model_file=ctx.scene_xml_text)
    for i in range(n):
        ep = _episode(ctx, rng)
        if mutate is not None:
            mutate(i, ep)
        writer.add_demonstration(**ep, seed=1000 + i)
    writer.close(train_ratio=0.75)
    return path


def _codes(report) -> set[str]:
    return {f"{i.level}:{i.code}" for i in report.issues}


def test_parse_dataset_name():
    assert parse_dataset_name(Path("a_b_p2_spatial.hdf5")) == ("a_b", "p2_spatial")
    assert parse_dataset_name(Path("lab_x_nominal.hdf5")) == ("lab_x", "nominal")


def test_well_formed_file_has_no_errors(tmp_path, ctx):
    report = validate_file(_write(tmp_path / f"{TASK}_nominal.hdf5", ctx), expected_demos=4, image_frames=(), ctx=ctx)
    assert report.status == "VALID_WITH_WARNINGS", report.issues
    assert not any(i.level == "error" for i in report.issues)
    assert "warning:no_provenance" in _codes(report)
    assert report.n_demos == 4 and report.total_steps == 4 * T
    assert report.stats["split"] == {"train": 3, "valid": 1}


def test_too_few_demos_is_invalid(tmp_path, ctx):
    report = validate_file(_write(tmp_path / f"{TASK}_nominal.hdf5", ctx, n=2), expected_demos=4, image_frames=(), ctx=ctx)
    assert "error:too_few_demos" in _codes(report)


def test_interrupted_write_never_reaches_the_final_name(tmp_path, ctx):
    path = tmp_path / f"{TASK}_nominal.hdf5"
    writer = RobomimicHDF5Writer(path, TASK, TASK, ctx.instruction, model_file=ctx.scene_xml_text)
    writer.add_demonstration(**_episode(ctx, np.random.default_rng(1)), seed=1)
    writer.h5_file.close()  # killed before close(): no masks, no totals, no rename
    assert not path.exists() and writer.write_path.exists()
    assert not is_complete(path, expected_demos=1)
    report = validate_file(writer.write_path, expected_demos=1, image_frames=(), ctx=ctx)
    assert report.status == "INVALID" and "error:missing_masks" in _codes(report)


def test_close_renames_and_abort_removes(tmp_path, ctx):
    path = tmp_path / f"{TASK}_nominal.hdf5"
    _write(path, ctx, n=1)
    assert path.exists() and not path.with_name(path.name + ".partial").exists()
    other = tmp_path / "other_nominal.hdf5"
    writer = RobomimicHDF5Writer(other, TASK, TASK, ctx.instruction)
    writer.abort()
    assert not other.exists() and not writer.write_path.exists()


def test_final_observation_completes_the_terminal_transition(tmp_path, ctx):
    path = tmp_path / f"{TASK}_nominal.hdf5"
    writer = RobomimicHDF5Writer(path, TASK, TASK, ctx.instruction, model_file=ctx.scene_xml_text,
                                 provenance={"git": {"commit": "abc"}})
    ep = _episode(ctx, np.random.default_rng(2))
    final_q = np.full(7, 0.123)
    final = {k: v[-1] for k, v in ep["observations"].items()} | {"joint_positions": final_q}
    writer.add_demonstration(**ep, seed=7, final_observation=final, final_state=ep["states"][-1])
    writer.close(train_ratio=1.0)
    with h5py.File(path, "r") as f:
        demo = f["data/demo_0"]
        assert np.array_equal(demo["next_obs/joint_positions"][-1], final_q)
        assert "final_state" in demo.attrs
        assert demo.attrs["language_instruction"] == ctx.instruction
    report = validate_file(path, expected_demos=1, image_frames=(), ctx=ctx)
    assert "warning:terminal_next_obs_copy" not in _codes(report)
    assert "warning:no_provenance" not in _codes(report)


def test_early_terminal_is_error(tmp_path, ctx):
    def mutate(i, ep):
        if i == 1:
            ep["dones"] = np.ones(T, dtype=bool)
    report = validate_file(_write(tmp_path / f"{TASK}_nominal.hdf5", ctx, mutate=mutate), 4, (), ctx)
    assert "error:early_terminal" in _codes(report)


def test_missing_success_reward_is_error(tmp_path, ctx):
    def mutate(i, ep):
        ep["rewards"] = np.zeros(T)
    report = validate_file(_write(tmp_path / f"{TASK}_nominal.hdf5", ctx, mutate=mutate), 4, (), ctx)
    assert "error:missing_success_terminal" in _codes(report)


def test_nonfinite_actions_are_error(tmp_path, ctx):
    def mutate(i, ep):
        ep["actions"][3, 2] = np.nan
    report = validate_file(_write(tmp_path / f"{TASK}_nominal.hdf5", ctx, mutate=mutate), 4, (), ctx)
    assert "error:nonfinite_actions" in _codes(report)


def test_obs_state_misalignment_is_error(tmp_path, ctx):
    def mutate(i, ep):
        if i == 0:
            ep["observations"]["joint_positions"] = np.roll(ep["observations"]["joint_positions"], 1, axis=0)
    report = validate_file(_write(tmp_path / f"{TASK}_nominal.hdf5", ctx, mutate=mutate), 4, (), ctx)
    assert "error:obs_state_misaligned" in _codes(report)


def test_cloned_episode_is_error(tmp_path, ctx):
    clone = {}

    def mutate(i, ep):
        if i == 0:
            clone.update({k: (v.copy() if isinstance(v, np.ndarray) else {kk: vv.copy() for kk, vv in v.items()})
                          for k, v in ep.items()})
        if i == 2:
            ep.update(clone)
    report = validate_file(_write(tmp_path / f"{TASK}_nominal.hdf5", ctx, mutate=mutate), 4, (), ctx)
    assert {"error:duplicate_initial_state", "error:duplicate_action_sequence"} <= _codes(report)


def test_terminal_next_obs_copy_is_reported(tmp_path, ctx):
    report = validate_file(_write(tmp_path / f"{TASK}_nominal.hdf5", ctx), 4, (), ctx)
    assert "warning:terminal_next_obs_copy" in _codes(report)


def test_wrong_task_in_env_args_is_error(tmp_path, ctx):
    path = _write(tmp_path / f"{TASK}_nominal.hdf5", ctx)
    with h5py.File(path, "r+") as f:
        f["data"].attrs["env_args"] = make_env_args("tabletop_sort_blue", perturbation_mode="nominal")
    report = validate_file(path, 4, (), ctx)
    assert "error:env_name_mismatch" in _codes(report)


def test_unregistered_task_is_error(tmp_path):
    path = tmp_path / "not_a_task_nominal.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data")
    report = validate_file(path, expected_demos=0, image_frames=())
    assert "error:task_not_registered" in _codes(report)


def test_start_pose_statistics(tmp_path, ctx):
    report = validate_file(_write(tmp_path / f"{TASK}_nominal.hdf5", ctx), 4, (), ctx)
    assert report.stats["start_pose"]["offset_from_nominal_max_m"] > 0.0
    assert report.stats["seeds"] == {"min": 1000, "max": 1003, "unique": 4, "inferred_attempts": 4}
