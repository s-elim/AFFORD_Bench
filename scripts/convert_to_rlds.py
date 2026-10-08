"""Convert AFFORD-Bench Robomimic HDF5 demonstrations to RLDS episodes for Open X-Embodiment pipelines.

Each episode follows the RLDS / Open X-Embodiment step layout used by OpenVLA and Octo: observation
(image, wrist_image, state = joint positions + gripper opening, and the remaining proprioception), action,
reward, discount, is_first, is_last, is_terminal, and language_instruction.

Two outputs:
  --format npz   one compressed .npz per episode with flattened 'steps/...' arrays (no TensorFlow needed);
  --format tfds  a TFDS dataset built with the AffordBenchRLDS builder below (needs tensorflow_datasets).

Usage:
  python scripts/convert_to_rlds.py --hdf5 datasets/*.hdf5 --out rlds/ --format npz
  python scripts/convert_to_rlds.py --hdf5 datasets/*.hdf5 --out ~/tensorflow_datasets --format tfds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afford_pro.datasets.converters import episode_to_rlds, read_robomimic, save_rlds_npz  # noqa: E402


def iter_rlds(paths: list[Path]):
    """Yield (episode id, RLDS episode) for every demonstration in the given HDF5 files."""
    for path in paths:
        meta, episodes = read_robomimic(path)
        for ep in episodes:
            episode_id = f"{path.stem}/{ep.key}"
            yield episode_id, episode_to_rlds(ep, meta.instruction, episode_id=episode_id)


def build_tfds(paths: list[Path], data_dir: Path) -> None:
    """Build the RLDS dataset with TFDS (requires tensorflow and tensorflow_datasets)."""
    import numpy as np
    import tensorflow_datasets as tfds

    first = next(iter_rlds(paths[:1]))[1]["steps"][0]["observation"]

    def tensor(v):
        v = np.asarray(v)
        if v.dtype == np.uint8 and v.ndim == 3:
            return tfds.features.Image(shape=v.shape, dtype=np.uint8, encoding_format="png")
        return tfds.features.Tensor(shape=v.shape, dtype=np.float32)

    class AffordBenchRLDS(tfds.core.GeneratorBasedBuilder):
        """AFFORD-Bench oracle demonstrations in RLDS format."""

        VERSION = tfds.core.Version("1.0.0")

        def _info(self):
            step = {
                "observation": tfds.features.FeaturesDict({k: tensor(v) for k, v in first.items()}),
                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                "reward": tfds.features.Scalar(dtype=np.float32),
                "discount": tfds.features.Scalar(dtype=np.float32),
                "is_first": np.bool_, "is_last": np.bool_, "is_terminal": np.bool_, "done": np.bool_,
                "language_instruction": tfds.features.Text(),
            }
            return tfds.core.DatasetInfo(builder=self, features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset(step),
                "episode_metadata": tfds.features.FeaturesDict({"file_path": tfds.features.Text(),
                                                                "source_key": tfds.features.Text()}),
            }))

        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            for episode_id, episode in iter_rlds(paths):
                for step in episode["steps"]:
                    step["observation"] = {k: (v if np.asarray(v).dtype == np.uint8 else np.asarray(v, np.float32))
                                           for k, v in step["observation"].items()}
                yield episode_id, episode

    AffordBenchRLDS(data_dir=str(data_dir)).download_and_prepare()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", type=Path, nargs="+", required=True, help="Robomimic HDF5 files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format", choices=("npz", "tfds"), default="npz")
    args = parser.parse_args()
    if args.format == "tfds":
        build_tfds(args.hdf5, args.out)
        return
    n = 0
    for n, (episode_id, episode) in enumerate(iter_rlds(args.hdf5), start=1):
        dest = args.out / f"{episode_id.replace('/', '__')}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_rlds_npz(dest, episode)
    print(f"Wrote {n} RLDS episodes to {args.out}")


if __name__ == "__main__":
    main()
