"""Oracle demonstration generation with provenance, safe writes, and post-write validation.

One function, used by every generation script, so that all datasets share one seed scheme and one record format:
  - episode seeds are `base_seed + k` with `base_seed = seed + 1000 * task_number` (k = attempt index), disjoint from
    the evaluation seed block `200000 + 1000 * task_number + j`;
  - an existing file is never replaced unless the caller asks (`overwrite_valid` / `overwrite_invalid`);
  - the file is written under a per-process `<name>.<pid>.partial`, validated episode by episode, and moved to the
    dataset name only if every requested demo was collected, it passes validation, and no file appeared at that name
    during the run; otherwise it is kept as `<task>_<mode>.{incomplete,invalid,conflict}.hdf5`, names that no loader
    or skip check treats as a dataset;
  - every file records `afford_pro.provenance.collect_provenance()`, all attempted seeds, why each rejected attempt
    was rejected, the oracle budget, and the demo-length limit;
  - only demos that succeed within `max_demo_steps` (default: the environment's episode limit) are kept.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from afford_pro.datasets.validation import ENV_HORIZON, build_task_context, validate_file

SEED_STRIDE = 1000


def episode_seed_base(task_number: int, seed: int = 42) -> int:
    return seed + SEED_STRIDE * task_number


@dataclass
class GenerationConfig:
    task_id: str
    output_dir: Path
    num_demos: int = 50
    perturbation: str = "nominal"
    seed: int = 42
    depth: bool = False
    max_attempts_factor: int = 8
    max_demo_steps: int = ENV_HORIZON
    overwrite_valid: bool = False
    overwrite_invalid: bool = False
    base_seed: int | None = None    # explicit first episode seed; None derives it from `seed` and the task number


@dataclass
class GenerationResult:
    path: str
    status: str                     # written | skipped_valid | refused_existing | incomplete | written_invalid
    collected: int = 0
    attempts: int = 0
    total_steps: int = 0
    failed_seeds: list[int] = field(default_factory=list)
    over_length_seeds: list[int] = field(default_factory=list)
    duration_s: float = 0.0
    validation_status: str | None = None


def _existing_file_decision(path: Path, cfg: GenerationConfig) -> GenerationResult | None:
    if not path.exists():
        return None
    status = validate_file(path, cfg.num_demos, image_frames=(0,)).status
    if status != "INVALID" and not cfg.overwrite_valid:
        return GenerationResult(str(path), "skipped_valid", validation_status=status)
    if status == "INVALID" and not cfg.overwrite_invalid:
        return GenerationResult(str(path), "refused_existing", validation_status=status)
    return None


def _make_writer(cfg: GenerationConfig, path: Path, env: Any, entry: Any, base_seed: int):
    from afford_pro.datasets.hdf5_writer import RobomimicHDF5Writer, make_env_args
    from afford_pro.oracles.base_oracle import BaseOracle
    from afford_pro.provenance import collect_provenance

    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    provenance = collect_provenance(extra={"generation_config": settings, "base_seed": base_seed,
                                           "oracle_max_steps": BaseOracle.max_steps,
                                           "env_max_episode_steps": getattr(env, "max_episode_steps", None)})
    writer = RobomimicHDF5Writer(
        file_path=path, env_name=cfg.task_id, task_id=cfg.task_id, instruction=entry.instruction,
        env_args=make_env_args(cfg.task_id, perturbation_mode=cfg.perturbation, render_depth=cfg.depth),
        model_file=env.get_model_xml(), provenance=provenance,
    )
    env.reset(seed=base_seed)
    for key, value in env.camera_parameters().items():
        writer.data_grp.attrs[f"camera_{key}"] = value
    return writer


def _collect(cfg: GenerationConfig, env: Any, oracle: Any, writer: Any, base_seed: int,
             result: GenerationResult, log: Any) -> None:
    while result.collected < cfg.num_demos and result.attempts < cfg.num_demos * cfg.max_attempts_factor:
        seed = base_seed + result.attempts
        result.attempts += 1
        actions, observations, rewards, dones = oracle.generate_episode(seed=seed)
        if not env.check_success():
            result.failed_seeds.append(seed)
            continue
        if len(actions) > cfg.max_demo_steps:
            result.over_length_seeds.append(seed)
            continue
        writer.add_demonstration(actions, observations, rewards, dones, states=oracle.last_states, seed=seed,
                                 final_observation=oracle.last_final_obs, final_state=oracle.last_final_state)
        result.collected += 1
        result.total_steps += len(actions)
        if log is not None and (result.collected % 5 == 0 or result.collected == cfg.num_demos):
            log(f"    -> [{result.collected:2d}/{cfg.num_demos}] demos ({result.attempts} attempts)")


def _commit(closed: Path, path: Path, cfg: GenerationConfig, result: GenerationResult, existed_at_start: bool) -> None:
    """Move the closed temporary file to its name: the dataset name only if it is complete, passes validation, and no
    file appeared at that name during the run (unless replacing it was requested at the start)."""
    stem = f"{cfg.task_id}_{cfg.perturbation}"
    ctx = build_task_context(cfg.task_id, cfg.perturbation)
    status = validate_file(closed, cfg.num_demos, image_frames=(0, -1), ctx=ctx).status
    if result.collected < cfg.num_demos:
        target, result.status = path.with_name(f"{stem}.incomplete.hdf5"), "incomplete"
    elif status == "INVALID":
        target, result.status = path.with_name(f"{stem}.invalid.hdf5"), "written_invalid"
    elif path.exists() and not existed_at_start:
        target, result.status = path.with_name(f"{stem}.conflict.hdf5"), "refused_existing"
    else:
        target = path
    os.replace(closed, target)
    result.path, result.validation_status = str(target), status


def generate_task_dataset(cfg: GenerationConfig, log: Any = print) -> GenerationResult:
    """Generate one task's demonstrations. Statuses: written, skipped_valid, refused_existing, incomplete,
    written_invalid (the new file failed validation and was kept as <task>_<mode>.invalid.hdf5)."""
    import afford_pro

    entry = afford_pro.get_task_entry(cfg.task_id)
    path = Path(cfg.output_dir) / f"{cfg.task_id}_{cfg.perturbation}.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = _existing_file_decision(path, cfg)
    if decision is not None:
        return decision
    existed_at_start = path.exists()
    base_seed = cfg.base_seed if cfg.base_seed is not None else episode_seed_base(entry.task_number, cfg.seed)
    env = afford_pro.make(cfg.task_id, perturbation_mode=cfg.perturbation, render_depth=cfg.depth)
    result = GenerationResult(str(path), "written")
    writer = None
    t0 = time.monotonic()
    try:
        oracle = afford_pro.get_oracle(cfg.task_id, env)
        writer = _make_writer(cfg, path, env, entry, base_seed)
        _collect(cfg, env, oracle, writer, base_seed, result, log)
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        env.close()
    result.duration_s = time.monotonic() - t0
    record = {k: v for k, v in asdict(result).items() if k not in ("validation_status", "status", "path")}
    record["status"] = "complete" if result.collected >= cfg.num_demos else "incomplete"
    record["base_seed"] = base_seed
    closed = writer.close(extra_attrs={"generation": record}, rename=False)
    _commit(closed, path, cfg, result, existed_at_start)
    return result
