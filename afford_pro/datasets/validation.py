"""Episode-level validation of AFFORD-Bench Robomimic HDF5 datasets.

Every episode in a file is checked; nothing is inferred from demo_0 alone. Errors make a file INVALID (it must not be
used or skipped by the generator); warnings mark known limitations that a user of the data must be told about.

Checks per file: required groups, contiguous demo keys, train/valid masks partitioning the demos, root attributes
(env_args, problem_info, num_demos, total, camera intrinsics), registry consistency (task id, perturbation mode,
instruction), scene XML staleness, and provenance attributes.
Checks per episode: required datasets, per-step lengths, dtypes and shapes, finite values, action bounds, sparse
terminal reward and done (0 before the last step, 1 at it), init_state == states[0], observation/state alignment
(joint positions decoded from `states`), next_obs shift, seeds, episode length against the environment horizon and
the evaluation budget, and image content on sampled frames.
Per file: duplicate initial states and action sequences, start-pose spread of the evaluated body, length and action
statistics, instruction-variant counts, and generation attempts inferred from seed gaps.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np

ENV_HORIZON = 500          # ParametricFrankaEnv.max_episode_steps
EVAL_HORIZON = 400         # evaluation step budget H of the protocol
ACTION_DIM = 7
AGENTIC_SHAPE = (640, 660, 3)
WRIST_SHAPE = (300, 420, 3)
LOWDIM_OBS = ("joint_positions", "joint_velocities", "tcp_pose", "gripper_width")
IMAGE_OBS = {"agentic_rgb": AGENTIC_SHAPE, "wrist_rgb": WRIST_SHAPE}
IMAGE_ALIASES = {"agentview_rgb": "agentic_rgb", "agentview_image": "agentic_rgb",
                 "eye_in_hand_rgb": "wrist_rgb", "robot0_eye_in_hand_image": "wrist_rgb"}
PROVENANCE_ATTR = "provenance"
ALIGN_TOL = 1e-9


@dataclass
class Issue:
    level: str               # "error" | "warning"
    code: str
    message: str
    demo: str | None = None


@dataclass
class FileReport:
    path: str
    task_id: str
    mode: str
    status: str = "UNCHECKED"
    n_demos: int = 0
    total_steps: int = 0
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str, demo: str | None = None) -> None:
        self.issues.append(Issue(level, code, message, demo))

    def finalize(self) -> "FileReport":
        levels = {i.level for i in self.issues}
        self.status = "INVALID" if "error" in levels else ("VALID_WITH_WARNINGS" if levels else "VALID")
        return self

    def summary(self) -> dict[str, Any]:
        counts = Counter((i.level, i.code) for i in self.issues)
        out = {k: v for k, v in asdict(self).items() if k != "issues"}
        out["issue_counts"] = {f"{lvl}:{code}": n for (lvl, code), n in sorted(counts.items())}
        out["first_issues"] = [asdict(i) for i in self.issues[:20]]
        return out


@dataclass
class TaskContext:
    """What the validator needs to know about the task and scene the file claims to contain."""
    task_id: str
    mode: str
    instruction: str | None
    nominal_pos: np.ndarray | None
    scene_xml_text: str | None
    state_size: int | None          # recorded state length: full physics state + gripper command
    qpos_offset: int
    arm_qpos_adr: list[int]
    object_qpos_adr: int | None


def parse_dataset_name(path: Path) -> tuple[str, str]:
    """`<task_id>_<mode>.hdf5` -> (task_id, mode); modes may contain underscores (p2_spatial, combined_ood)."""
    stem = path.stem
    for mode in ("combined_ood", "p1_object", "p2_spatial", "p3_visual", "p4_physical", "nominal"):
        if stem.endswith("_" + mode):
            return stem[: -len(mode) - 1], mode
    return stem, "unknown"


def build_task_context(task_id: str, mode: str) -> TaskContext:
    """Registry entry and compiled scene model of the task (no environment, no rendering)."""
    from afford_pro.envs.base_franka_env import _load_model_with_portable_assets
    from afford_pro.envs.parametric_env import ASSET_DIR
    from afford_pro.task_definitions import TASK_MAP_210, TASK_MAP_FRONTIER

    task = TASK_MAP_210.get(task_id) or TASK_MAP_FRONTIER.get(task_id)
    if task is None:
        return TaskContext(task_id, mode, None, None, None, None, 1, [], None)
    xml_path = ASSET_DIR / task.scene_xml
    model = _load_model_with_portable_assets(xml_path)
    joint = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)  # noqa: E731
    obj_joint = joint(task.joint_name)
    is_free = obj_joint >= 0 and model.jnt_type[obj_joint] == mujoco.mjtJoint.mjJNT_FREE
    return TaskContext(
        task_id=task_id, mode=mode, instruction=task.instruction,
        nominal_pos=np.asarray(task.nominal_pos, dtype=np.float64),
        scene_xml_text=xml_path.read_text(encoding="utf-8"),
        state_size=int(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)) + 1,
        qpos_offset=int(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_TIME)),
        arm_qpos_adr=[int(model.jnt_qposadr[joint(f"joint{i}")]) for i in range(1, 8)],
        object_qpos_adr=int(model.jnt_qposadr[obj_joint]) if is_free and task.task_type != "press_button" else None,
    )


def _sha1(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _demo_keys(data: h5py.Group, report: FileReport) -> list[str]:
    keys = [k for k in data.keys() if k.startswith("demo_")]
    try:
        keys.sort(key=lambda k: int(k.split("_")[1]))
    except ValueError:
        report.add("error", "bad_demo_key", f"non-numeric demo key among {keys[:5]}")
        return sorted(keys)
    if keys != [f"demo_{i}" for i in range(len(keys))]:
        report.add("error", "demo_keys_not_contiguous", f"demo keys are not demo_0..demo_{len(keys) - 1}")
    other = [k for k in data.keys() if not k.startswith("demo_")]
    if other:
        report.add("warning", "unexpected_data_members", f"unexpected members in data/: {other[:5]}")
    return keys


def _check_masks(f: h5py.File, keys: list[str], report: FileReport) -> None:
    if "mask" not in f or "train" not in f["mask"] or "valid" not in f["mask"]:
        report.add("error", "missing_masks", "mask/train and mask/valid are required")
        return
    decode = lambda arr: [k.decode() if isinstance(k, bytes) else str(k) for k in arr]  # noqa: E731
    train, valid = decode(f["mask/train"][()]), decode(f["mask/valid"][()])
    if set(train) & set(valid):
        report.add("error", "mask_overlap", f"{len(set(train) & set(valid))} demos in both train and valid")
    if sorted(set(train) | set(valid)) != sorted(keys) or len(train) + len(valid) != len(keys):
        report.add("error", "mask_mismatch", "train + valid masks do not partition the demos exactly")
    report.stats["split"] = {"train": len(train), "valid": len(valid)}


def _json_attr(data: h5py.Group, name: str, report: FileReport) -> dict[str, Any] | None:
    if name not in data.attrs:
        report.add("error", f"missing_{name}", f"data.attrs['{name}'] is required")
        return None
    try:
        return json.loads(data.attrs[name])
    except (TypeError, ValueError) as exc:
        report.add("error", f"bad_{name}", f"data.attrs['{name}'] is not JSON: {exc}")
        return None


def _check_root_attrs(data: h5py.Group, ctx: TaskContext, report: FileReport) -> None:
    env_args = _json_attr(data, "env_args", report)
    if env_args is not None:
        if env_args.get("env_name") != ctx.task_id:
            report.add("error", "env_name_mismatch", f"env_args.env_name={env_args.get('env_name')} != {ctx.task_id}")
        mode = env_args.get("env_kwargs", {}).get("perturbation_mode")
        if mode != ctx.mode:
            report.add("error", "mode_mismatch", f"env_args perturbation_mode={mode} but file name says {ctx.mode}")
        report.stats["render_depth"] = bool(env_args.get("env_kwargs", {}).get("render_depth", False))
    _json_attr(data, "problem_info", report)
    if ctx.instruction is None:
        report.add("error", "task_not_registered", f"{ctx.task_id} is not in the task registry")
    elif str(data.attrs.get("instruction", "")) != ctx.instruction:
        report.add("warning", "instruction_differs_from_registry", "data.attrs['instruction'] != registry instruction")
    model_file = data.attrs.get("model_file")
    if model_file is None:
        report.add("error", "missing_model_file", "data.attrs['model_file'] is required for replay")
    elif ctx.scene_xml_text is not None and str(model_file) != ctx.scene_xml_text:
        # Data recorded on a different scene is stale for the current code: it must not count as complete.
        report.add("error", "scene_xml_changed", "stored model_file differs from the current scene XML")
    for cam in ("agentic", "wrist"):
        k = data.attrs.get(f"camera_{cam}_K")
        if k is None or not np.isfinite(np.asarray(k)).all() or np.asarray(k)[0, 0] <= 0:
            report.add("warning", "missing_camera_intrinsics", f"camera_{cam}_K missing or invalid")
    if PROVENANCE_ATTR not in data.attrs:
        report.add("warning", "no_provenance", "no code version, simulator version, or generation command recorded")


def _check_lengths(g: h5py.Group, T: int, demo: str, report: FileReport) -> bool:
    ok = True
    for name in ("rewards", "dones", "states"):
        if name not in g:
            report.add("error", f"missing_{name}", f"{demo}/{name} missing", demo)
            ok = False
        elif g[name].shape[0] != T:
            report.add("error", "length_mismatch", f"{demo}/{name} has {g[name].shape[0]} rows, actions {T}", demo)
            ok = False
    for group in ("obs", "next_obs"):
        if group not in g:
            report.add("error", f"missing_{group}", f"{demo}/{group} missing", demo)
            ok = False
            continue
        for k, ds in g[group].items():
            if ds.shape[0] != T:
                report.add("error", "length_mismatch", f"{demo}/{group}/{k} has {ds.shape[0]} rows, actions {T}", demo)
                ok = False
    if int(g.attrs.get("num_samples", T)) != T:
        report.add("error", "num_samples_mismatch", f"{demo} num_samples attr != {T}", demo)
    return ok


def _check_terminal(g: h5py.Group, demo: str, report: FileReport) -> None:
    r, d = g["rewards"][()], g["dones"][()]
    if not np.isfinite(r).all():
        report.add("error", "nonfinite_rewards", f"{demo} rewards not finite", demo)
    if np.any(r[:-1] != 0) or np.any(d[:-1] != 0):
        report.add("error", "early_terminal", f"{demo} has reward or done before the last step", demo)
    if r[-1] != 1.0 or d[-1] != 1:
        report.add("error", "missing_success_terminal", f"{demo} last step is not reward 1, done 1", demo)


def _check_actions_states(g: h5py.Group, T: int, demo: str, ctx: TaskContext, report: FileReport) -> np.ndarray | None:
    a = g["actions"][()]
    if a.shape != (T, ACTION_DIM) or a.dtype != np.float32:
        report.add("error", "bad_actions", f"{demo} actions {a.shape} {a.dtype}, expected ({T}, 7) float32", demo)
    if not np.isfinite(a).all():
        report.add("error", "nonfinite_actions", f"{demo} actions contain NaN/Inf", demo)
    elif np.abs(a).max() > 1.0 + 1e-6:
        report.add("error", "action_out_of_bounds", f"{demo} |action| max {np.abs(a).max():.4f} > 1", demo)
    s = g["states"][()]
    if s.ndim != 2 or s.dtype != np.float64 or (ctx.state_size is not None and s.shape[1] != ctx.state_size):
        report.add("error", "bad_states", f"{demo} states {s.shape} {s.dtype}, expected (T, {ctx.state_size}) float64", demo)
        return None
    if not np.isfinite(s).all():
        report.add("error", "nonfinite_states", f"{demo} states contain NaN/Inf", demo)
        return None
    init = g.attrs.get("init_state")
    if init is None or not np.array_equal(np.asarray(init), s[0]):
        report.add("error", "init_state_mismatch", f"{demo} attrs['init_state'] != states[0]", demo)
    return s


def _check_obs(g: h5py.Group, s: np.ndarray | None, T: int, demo: str, ctx: TaskContext, report: FileReport) -> None:
    obs, nxt = g["obs"], g["next_obs"]
    for k in LOWDIM_OBS:
        if k not in obs:
            report.add("error", "missing_obs", f"{demo}/obs/{k} missing", demo)
            return
        if not np.isfinite(obs[k][()]).all():
            report.add("error", "nonfinite_obs", f"{demo}/obs/{k} not finite", demo)
    q = obs["joint_positions"][()]
    if s is not None and ctx.arm_qpos_adr:
        q_state = s[:, [ctx.qpos_offset + adr for adr in ctx.arm_qpos_adr]]
        if np.abs(q_state - q).max() > ALIGN_TOL:
            report.add("error", "obs_state_misaligned", f"{demo} obs/joint_positions != joint positions in states", demo)
    if "joint_positions" in nxt:
        nq = nxt["joint_positions"][()]
        if T > 1 and not np.array_equal(nq[:-1], q[1:]):
            report.add("error", "next_obs_not_shifted", f"{demo} next_obs[t] != obs[t+1]", demo)
        if np.array_equal(nq[-1], q[-1]):
            report.add("warning", "terminal_next_obs_copy", f"{demo} next_obs[T-1] repeats obs[T-1]; the success state is not stored", demo)


def _image_ok(frame: np.ndarray) -> str | None:
    mean, std = float(frame.mean()), float(frame.std())
    if mean < 1.0:
        return f"dark frame (mean {mean:.2f})"
    if std < 1.0:
        return f"constant frame (std {std:.2f})"
    return None


def _check_images(g: h5py.Group, T: int, demo: str, frames: tuple[int, ...], report: FileReport,
                  first_demo: bool) -> None:
    obs = g["obs"]
    for key, shape in IMAGE_OBS.items():
        if key not in obs:
            report.add("error", "missing_image", f"{demo}/obs/{key} missing", demo)
            continue
        ds = obs[key]
        if ds.shape != (T, *shape) or ds.dtype != np.uint8:
            report.add("error", "bad_image_shape", f"{demo}/obs/{key} {ds.shape} {ds.dtype}, expected {(T, *shape)} uint8", demo)
            continue
        for t in sorted({t if t >= 0 else T + t for t in frames} & set(range(T))):
            problem = _image_ok(ds[t])
            if problem:
                report.add("error", "bad_image_content", f"{demo}/obs/{key}[{t}]: {problem}", demo)
    for alias, source in IMAGE_ALIASES.items():
        if alias in obs and source in obs and not (obs[alias] == obs[source]):
            if first_demo:
                report.add("warning", "alias_stored_as_copy", f"obs/{alias} is a copy of obs/{source}, not a hard link")
            if obs[alias].shape != obs[source].shape or not np.array_equal(obs[alias][0], obs[source][0]):
                report.add("error", "alias_content_mismatch", f"{demo}/obs/{alias} differs from obs/{source}", demo)


def _start_pose(s: np.ndarray, ctx: TaskContext) -> tuple[np.ndarray, float] | None:
    if ctx.object_qpos_adr is None:
        return None
    q = s[0, ctx.qpos_offset + ctx.object_qpos_adr: ctx.qpos_offset + ctx.object_qpos_adr + 7]
    w, x, y, z = q[3:7]
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return q[:3].copy(), yaw


def _file_stats(rows: list[dict[str, Any]], ctx: TaskContext, report: FileReport) -> None:
    lengths = np.array([r["T"] for r in rows])
    seeds = [r["seed"] for r in rows if r["seed"] is not None]
    report.stats["episode_length"] = {"min": int(lengths.min()), "median": float(np.median(lengths)),
                                      "mean": float(lengths.mean()), "max": int(lengths.max())}
    report.stats["over_env_horizon"] = int((lengths > ENV_HORIZON).sum())
    report.stats["over_eval_horizon"] = int((lengths > EVAL_HORIZON).sum())
    if seeds:
        report.stats["seeds"] = {"min": min(seeds), "max": max(seeds), "unique": len(set(seeds)),
                                 "inferred_attempts": max(seeds) - min(seeds) + 1}
    starts = [r["start"] for r in rows if r.get("start") is not None]
    if starts and ctx.nominal_pos is not None:
        xy = np.array([p[:2] for p, _ in starts])
        offsets = np.linalg.norm(xy - ctx.nominal_pos[:2], axis=1)
        report.stats["start_pose"] = {
            "xy_std_m": xy.std(axis=0).round(4).tolist(), "offset_from_nominal_max_m": float(offsets.max()),
            "offset_from_nominal_median_m": float(np.median(offsets)),
            "at_nominal_xy": int((offsets < 1e-3).sum()),
            "yaw_deg_range": [float(np.degrees(min(y for _, y in starts))), float(np.degrees(max(y for _, y in starts)))],
        }
    acts = np.concatenate([r["actions"] for r in rows])
    report.stats["actions"] = {"mean": acts.mean(axis=0).round(4).tolist(), "std": acts.std(axis=0).round(4).tolist()}


def _check_generation_record(data: h5py.Group, rows: list[dict[str, Any]], report: FileReport) -> None:
    """Files from the current generator record their status and seed range; check both when present."""
    if "generation" not in data.attrs:
        return
    try:
        record = json.loads(data.attrs["generation"])
    except (TypeError, ValueError):
        report.add("error", "bad_generation_record", "data.attrs['generation'] is not JSON")
        return
    if record.get("status") == "incomplete":
        report.add("error", "incomplete_generation", "the generator recorded this file as incomplete")
    base, attempts = record.get("base_seed"), record.get("attempts")
    if base is not None and attempts is not None:
        outside = [r["seed"] for r in rows if r["seed"] is not None and not base <= r["seed"] < base + attempts]
        if outside:
            report.add("error", "seed_outside_generation_range", f"{len(outside)} demo seeds outside [{base}, {base + attempts})")


def _check_duplicates(rows: list[dict[str, Any]], report: FileReport) -> None:
    for field_name, code in (("init_hash", "duplicate_initial_state"), ("action_hash", "duplicate_action_sequence")):
        counts = Counter(r[field_name] for r in rows if r[field_name] is not None)
        dup = sum(n - 1 for n in counts.values() if n > 1)
        if dup:
            report.add("error", code, f"{dup} demos repeat another demo's {field_name.split('_')[0]} in this file")
    seeds = Counter(r["seed"] for r in rows if r["seed"] is not None)
    if any(n > 1 for n in seeds.values()):
        report.add("error", "duplicate_seed", "two demos share a reset seed")
    if len(seeds) < len(rows):
        report.add("warning", "missing_seed", f"{len(rows) - sum(seeds.values())} demos without a seed attribute")


def _validate_episode(g: h5py.Group, demo: str, ctx: TaskContext, report: FileReport,
                      image_frames: tuple[int, ...], first: bool) -> dict[str, Any] | None:
    if "actions" not in g:
        report.add("error", "missing_actions", f"{demo}/actions missing", demo)
        return None
    T = int(g["actions"].shape[0])
    if T == 0:
        report.add("error", "empty_episode", f"{demo} has no steps", demo)
        return None
    if not _check_lengths(g, T, demo, report):
        return None
    _check_terminal(g, demo, report)
    s = _check_actions_states(g, T, demo, ctx, report)
    _check_obs(g, s, T, demo, ctx, report)
    if image_frames:
        _check_images(g, T, demo, image_frames, report, first)
    seed = g.attrs.get("seed")
    a = g["actions"][()]
    return {"demo": demo, "T": T, "seed": None if seed is None else int(seed), "actions": a,
            "init_hash": None if s is None else _sha1(s[0]), "action_hash": _sha1(a),
            "start": None if s is None else _start_pose(s, ctx),
            "instruction": str(g.attrs.get("language_instruction", ""))}


def validate_file(path: Path, expected_demos: int = 50, image_frames: tuple[int, ...] = (0, -1),
                  ctx: TaskContext | None = None) -> FileReport:
    """Validate every episode of one dataset file; `image_frames=()` skips image content checks."""
    path = Path(path)
    # The task identity comes from `ctx` when given (temporary files such as *.partial have no parseable name).
    task_id, mode = (ctx.task_id, ctx.mode) if ctx is not None else parse_dataset_name(path)
    report = FileReport(path=str(path), task_id=task_id, mode=mode)
    ctx = ctx or build_task_context(task_id, mode)
    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                report.add("error", "missing_data_group", "no data/ group")
                return report.finalize()
            data = f["data"]
            keys = _demo_keys(data, report)
            report.n_demos = len(keys)
            if len(keys) < expected_demos:
                report.add("error", "too_few_demos", f"{len(keys)} demos, expected {expected_demos}")
            _check_masks(f, keys, report)
            _check_root_attrs(data, ctx, report)
            rows = [r for i, k in enumerate(keys)
                    if (r := _validate_episode(data[k], k, ctx, report, image_frames, i == 0)) is not None]
            if rows:
                report.total_steps = sum(r["T"] for r in rows)
                if "total" in data.attrs and int(data.attrs["total"]) != report.total_steps:
                    report.add("error", "total_mismatch", f"attrs total={int(data.attrs['total'])} != {report.total_steps}")
                if "num_demos" in data.attrs and int(data.attrs["num_demos"]) != len(keys):
                    report.add("error", "num_demos_mismatch", "attrs num_demos != number of demo groups")
                _check_duplicates(rows, report)
                _check_generation_record(data, rows, report)
                _file_stats(rows, ctx, report)
                report.stats["instruction_variants"] = dict(Counter(r["instruction"] for r in rows))
                report.stats["action_hashes"] = [r["action_hash"] for r in rows]
                if report.stats["over_env_horizon"]:
                    report.add("warning", "exceeds_env_horizon",
                               f"{report.stats['over_env_horizon']} demos longer than the env limit {ENV_HORIZON}")
                if report.stats["over_eval_horizon"]:
                    report.add("warning", "exceeds_eval_horizon",
                               f"{report.stats['over_eval_horizon']} demos longer than the evaluation budget {EVAL_HORIZON}")
    except OSError as exc:
        report.add("error", "unreadable", f"cannot read file: {exc}")
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        report.add("error", "malformed", f"{type(exc).__name__} while reading the file: {exc}")
    return report.finalize()


def is_complete(path: Path, expected_demos: int = 50) -> bool:
    """True only if the file passes every error-level check (used by generators to skip finished tasks)."""
    return Path(path).exists() and validate_file(path, expected_demos, image_frames=(0,)).status != "INVALID"
