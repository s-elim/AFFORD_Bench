"""Augment existing AFFORD-Bench HDF5 demonstration files with actuator motor torques and contact forces.

Because AFFORD-Bench demonstrations store deterministic initial states `states[0]`, reset seeds,
and executed actions `actions`, this script deterministically replays each episode through the MuJoCo
physics environment and computes:
  - obs/joint_torques: (T, 7) float64 - actuator motor torques tau_act (N*m) on the 7 arm joints
  - obs/robot0_joint_torque: hard link to obs/joint_torques (standard Robomimic alias)
  - obs/actuator_torques: hard link to obs/joint_torques
  - obs/robot_scene_force: (T, 1) float64 - net scene contact force magnitude (N) on the robot
  - next_obs/*: corresponding next-step torque and force arrays

Usage:
  python scripts/augment_datasets_with_torques.py --files datasets/tabletop_sort_red_nominal.hdf5 --in-place
  python scripts/augment_datasets_with_torques.py --input-dir datasets/ --out-dir datasets_augmented/
  python scripts/augment_datasets_with_torques.py --input-dir datasets/ --check-only
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro
from afford_pro.datasets.validation import file_task_set, parse_dataset_name, validate_file
from afford_pro.envs.headless import headless


def has_torque_channel(path: Path) -> bool:
    """Check if an HDF5 dataset already contains the joint_torques observation channel."""
    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                return False
            demos = [k for k in f["data"].keys() if k.startswith("demo_")]
            if not demos:
                return False
            d0 = f["data"][demos[0]]
            return "obs" in d0 and "joint_torques" in d0["obs"]
    except Exception:
        return False


def augment_file(file_path: Path, target_path: Path | None = None, overwrite: bool = False,
                 quiet: bool = False) -> tuple[bool, str]:
    """Replay an HDF5 file's episodes to compute and save actuator torques and contact forces."""
    file_path = Path(file_path)
    target_path = Path(target_path) if target_path is not None else file_path

    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if not overwrite and has_torque_channel(file_path) and target_path == file_path:
        return True, f"Skipped (torque channel already present): {file_path.name}"

    task_id, mode = parse_dataset_name(file_path)
    task_set = file_task_set(file_path)

    # If writing to a separate target path, copy first
    if target_path != file_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target_path)

    with headless(), afford_pro.make(task_id, task_set=task_set, perturbation_mode=mode) as env:
        with h5py.File(target_path, "r+") as f:
            data_grp = f["data"]
            demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")],
                               key=lambda k: int(k.split("_")[-1]))

            for demo_key in demo_keys:
                g = data_grp[demo_key]
                if "states" not in g or "actions" not in g:
                    continue

                states = g["states"][()]
                actions = g["actions"][()]
                seed = g.attrs.get("seed")
                T = len(actions)

                # Reset env and set sim state to init_state (states[0])
                env.reset(seed=None if seed is None else int(seed))
                env.set_sim_state(states[0])

                # t=0 observation before actions[0]
                obs_0 = env.get_observation()
                tau_seq = [np.asarray(obs_0["joint_torques"], dtype=np.float64)]
                force_seq = [np.asarray(obs_0["robot_scene_force"], dtype=np.float64)]

                # Step through actions
                for action in actions:
                    obs, _, _, _, info = env.step(action)
                    tau_seq.append(np.asarray(obs["joint_torques"], dtype=np.float64))
                    force_seq.append(np.asarray(obs["robot_scene_force"], dtype=np.float64))

                # Align observations (obs: 0..T-1, next_obs: 1..T)
                obs_tau = np.stack(tau_seq[:T], axis=0)
                next_obs_tau = np.stack(tau_seq[1:T + 1], axis=0)

                obs_force = np.stack(force_seq[:T], axis=0)
                next_obs_force = np.stack(force_seq[1:T + 1], axis=0)

                # Write or replace in obs/
                obs_grp = g["obs"]
                for k, arr in [("joint_torques", obs_tau), ("robot_scene_force", obs_force)]:
                    if k in obs_grp:
                        del obs_grp[k]
                    obs_grp.create_dataset(k, data=arr, compression="gzip")

                # Establish hard links for standard Robomimic aliases
                for alias in ("robot0_joint_torque", "actuator_torques"):
                    if alias in obs_grp:
                        del obs_grp[alias]
                    obs_grp[alias] = obs_grp["joint_torques"]

                # Write or replace in next_obs/
                if "next_obs" in g:
                    next_grp = g["next_obs"]
                    for k, arr in [("joint_torques", next_obs_tau), ("robot_scene_force", next_obs_force)]:
                        if k in next_grp:
                            del next_grp[k]
                        next_grp.create_dataset(k, data=arr, compression="gzip")

                    for alias in ("robot0_joint_torque", "actuator_torques"):
                        if alias in next_grp:
                            del next_grp[alias]
                        next_grp[alias] = next_grp["joint_torques"]

    # Validate augmented file
    val = validate_file(target_path, expected_demos=len(demo_keys), image_frames=(0,))
    status_str = f"Status: {val.status}"
    if val.status == "INVALID":
        return False, f"Failed validation after augmentation: {val.summary()['first_issues']}"

    return True, f"Successfully augmented {target_path.name} ({len(demo_keys)} demos) - {status_str}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="*", type=Path, help="Specific HDF5 file(s) to augment.")
    parser.add_argument("--input-dir", type=Path, help="Directory containing HDF5 files to augment.")
    parser.add_argument("--out-dir", type=Path, help="Output directory (if not specified, modifies in-place).")
    parser.add_argument("--in-place", action="store_true", help="Modify files in place.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite torque channels if already present.")
    parser.add_argument("--check-only", action="store_true", help="Only check which files lack torque channels.")
    args = parser.parse_args()

    files: list[Path] = []
    if args.files:
        files.extend(args.files)
    if args.input_dir:
        files.extend(sorted(args.input_dir.glob("*.hdf5")))

    if not files:
        parser.error("No HDF5 files specified. Use --files <path> or --input-dir <dir>.")

    if args.check_only:
        print(f"Checking {len(files)} dataset files for torque channels:")
        present_count = 0
        for f in files:
            has_t = has_torque_channel(f)
            status = "PRESENT" if has_t else "MISSING"
            print(f"  [{status}] {f.name}")
            if has_t:
                present_count += 1
        print(f"\nSummary: {present_count}/{len(files)} files have torque channels.")
        return

    if not args.in_place and not args.out_dir:
        parser.error("Must specify either --in-place or --out-dir <dir> to augment files.")

    print(f"Augmenting {len(files)} dataset file(s) with motor torques and contact forces...")
    for idx, f in enumerate(files, 1):
        target = args.out_dir / f.name if args.out_dir else f
        ok, msg = augment_file(f, target_path=target, overwrite=args.overwrite)
        print(f"[{idx}/{len(files)}] {msg}")


if __name__ == "__main__":
    main()
