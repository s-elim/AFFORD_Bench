"""Conversion between AFFORD-Bench Robomimic HDF5 datasets, LeRobot v2.0 datasets, and RLDS episodes.

Robomimic HDF5 (written by `RobomimicHDF5Writer`):
    data/demo_i/actions (T, 7) float32, data/demo_i/obs/<key> (T, ...), data/demo_i/rewards (T,),
    data/demo_i/dones (T,), optional data/demo_i/states (T, S) simulator states and demo attr `seed`, attrs
    env_args / env_name / task_id / instruction on `data`, mask/train and mask/valid.

LeRobot v2.0 (one parquet file per episode, images stored losslessly as PNG in the parquet "image" struct):
    meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, meta/stats.json,
    data/chunk-XXX/episode_XXXXXX.parquet

RLDS (Open X-Embodiment layout): an episode is {"steps": [...], "episode_metadata": {...}} and every step holds
    observation, action, reward, discount, is_first, is_last, is_terminal, language_instruction.

Every conversion is lossless, so HDF5 -> LeRobot -> HDF5 and HDF5 -> RLDS -> HDF5 reproduce the source arrays.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

IMAGE_KEYS = {"agentic_rgb": "workcell", "wrist_rgb": "wrist"}  # HDF5 obs key -> camera name
STATE_KEYS = ("joint_positions", "joint_velocities", "tcp_pose", "gripper_width")
FPS = 20  # control rate of BaseFrankaEnv (25 substeps of 2 ms)
CHUNK_SIZE = 1000
LEROBOT_VERSION = "v2.0"


@dataclass
class Episode:
    """One demonstration: actions (T, A), observations {key: (T, ...)}, rewards (T,), dones (T,), and optionally the
    simulator state before each action (T, S) and the reset seed."""

    actions: np.ndarray
    obs: dict[str, np.ndarray]
    rewards: np.ndarray
    dones: np.ndarray
    key: str = ""
    states: np.ndarray | None = None
    seed: int | None = None
    language_instruction: str = ""

    def __post_init__(self) -> None:
        T = len(self.actions)
        assert self.actions.ndim == 2, self.actions.shape
        assert len(self.rewards) == T and len(self.dones) == T
        assert all(len(v) == T for v in self.obs.values()), {k: v.shape for k, v in self.obs.items()}
        assert self.states is None or len(self.states) == T, (None if self.states is None else self.states.shape)

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class DatasetMeta:
    env_name: str
    task_id: str
    instruction: str
    train_keys: list[str] = field(default_factory=list)
    valid_keys: list[str] = field(default_factory=list)
    camera: dict[str, list] = field(default_factory=dict)  # camera_* attributes (intrinsics K, pose) if recorded
    env_args: str = ""  # robomimic environment metadata (JSON)
    model_file: str = ""  # MuJoCo scene XML string


ROBOMIMIC_ALIASES_MAP = {
    "agentview_rgb": "agentic_rgb",
    "agentview_image": "agentic_rgb",
    "eye_in_hand_rgb": "wrist_rgb",
    "robot0_eye_in_hand_image": "wrist_rgb",
    "joint_states": "joint_positions",
    "robot0_joint_pos": "joint_positions",
    "robot0_joint_vel": "joint_velocities",
    "agentview_depth": "agentic_depth",
    "eye_in_hand_depth": "wrist_depth",
    "ee_states": "tcp_pose",
    "ee_pos": "tcp_pose",
    "robot0_eef_pos": "tcp_pose",
    "ee_ori": "tcp_pose",
    "robot0_eef_quat": "tcp_pose",
    "gripper_states": "gripper_width",
    "robot0_gripper_qpos": "gripper_width",
}


# ------------------------------------------------------------------------------------------- Robomimic HDF5
def read_robomimic(path: str | Path) -> tuple[DatasetMeta, Iterator[Episode]]:
    """Dataset metadata and a lazy iterator over its demonstrations in demo index order."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        data = f["data"]
        meta = DatasetMeta(
            env_name=str(data.attrs.get("env_name", "")),
            task_id=str(data.attrs.get("task_id", "")),
            instruction=str(data.attrs.get("instruction", "")),
            train_keys=[k.decode() for k in f["mask/train"][()]] if "mask/train" in f else [],
            valid_keys=[k.decode() for k in f["mask/valid"][()]] if "mask/valid" in f else [],
            camera={k: np.asarray(v).tolist() for k, v in data.attrs.items() if k.startswith("camera_")},
            env_args=str(data.attrs.get("env_args", "")),
            model_file=str(data.attrs.get("model_file", "")),
        )
        keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))

    def episodes() -> Iterator[Episode]:
        with h5py.File(path, "r") as f:
            for k in keys:
                g = f["data"][k]
                # Filter out duplicate hard-link aliases so canonical observations are retained
                obs_dict = {}
                for o in g["obs"]:
                    if o in ROBOMIMIC_ALIASES_MAP and ROBOMIMIC_ALIASES_MAP[o] in g["obs"]:
                        continue
                    obs_dict[o] = g["obs"][o][()]
                yield Episode(
                    actions=g["actions"][()],
                    obs=obs_dict,
                    rewards=g["rewards"][()],
                    dones=g["dones"][()],
                    key=k,
                    states=g["states"][()] if "states" in g else None,
                    seed=int(g.attrs["seed"]) if "seed" in g.attrs else None,
                    language_instruction=str(g.attrs.get("language_instruction", "")),
                )

    return meta, episodes()


def write_robomimic(path: str | Path, meta: DatasetMeta, episodes: Iterator[Episode]) -> int:
    """Write episodes as demo_0..demo_{n-1} with the dataset's train/valid masks; returns the episode count."""
    from afford_pro.datasets.hdf5_writer import RobomimicHDF5Writer

    writer = RobomimicHDF5Writer(
        path,
        env_name=meta.env_name,
        task_id=meta.task_id,
        instruction=meta.instruction,
        env_args=meta.env_args or None,
        model_file=meta.model_file or None,
    )
    for k, v in meta.camera.items():
        writer.data_grp.attrs[k] = np.asarray(v)
    for ep in episodes:
        writer.add_demonstration(
            ep.actions,
            ep.obs,
            ep.rewards,
            ep.dones,
            states=ep.states,
            seed=ep.seed,
            language_instruction=ep.language_instruction or None,
        )
    n = writer.demo_count
    train = meta.train_keys or [f"demo_{i}" for i in range(int(n * 0.9))]
    valid = meta.valid_keys or [f"demo_{i}" for i in range(int(n * 0.9), n)]
    writer.close(train_keys=train, valid_keys=valid)
    return n


# ------------------------------------------------------------------------------------------------ LeRobot
def _png(image: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buf, format="PNG")
    return buf.getvalue()


def _unpng(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)))


def _state_layout(obs: dict[str, np.ndarray]) -> list[tuple[str, int]]:
    return [(k, int(np.prod(obs[k].shape[1:]))) for k in STATE_KEYS if k in obs]


def _extra_keys(obs: dict[str, np.ndarray]) -> list[str]:
    """Observation keys that are neither RGB cameras nor proprioception (e.g. metric depth)."""
    return sorted(k for k in obs if k not in IMAGE_KEYS and k not in STATE_KEYS)


def write_lerobot(root: str | Path, meta: DatasetMeta, episodes: Iterator[Episode], fps: int = FPS,
                  robot_type: str = "franka_panda") -> int:
    """Write a LeRobot v2.0 dataset; returns the number of episodes.

    observation.state concatenates joint positions, joint velocities, TCP pose (position and w-first
    quaternion), and gripper opening; info.json records the name of every state dimension. Other observation
    keys (metric depth) become float32 features observation.<key>. Episode-level train and validation
    membership from the Robomimic masks is kept in episodes.jsonl, and recorded camera parameters in info.json.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    valid = set(meta.valid_keys)
    features: dict[str, Any] | None = None
    layout: list[tuple[str, int]] = []
    episodes_meta, stats_acc = [], {}
    total_frames = 0
    n = 0
    for n, ep in enumerate(episodes):
        T = len(ep)
        if features is None:
            layout = _state_layout(ep.obs)
            state_names = [f"{k}_{i}" for k, d in layout for i in range(d)]
            features = {
                **{f"observation.images.{cam}": {"dtype": "image", "shape": list(ep.obs[k].shape[1:]),
                                                 "names": ["height", "width", "channel"]}
                   for k, cam in IMAGE_KEYS.items() if k in ep.obs},
                "observation.state": {"dtype": "float32", "shape": [len(state_names)], "names": state_names},
                **{f"observation.{k}": {"dtype": "float32", "shape": list(ep.obs[k].shape[1:]), "names": None}
                   for k in _extra_keys(ep.obs)},
                "action": {"dtype": "float32", "shape": [ep.actions.shape[1]],
                           "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"][: ep.actions.shape[1]]},
                "next.reward": {"dtype": "float32", "shape": [1], "names": None},
                "next.done": {"dtype": "bool", "shape": [1], "names": None},
                **{k: {"dtype": "int64", "shape": [1], "names": None}
                   for k in ("frame_index", "episode_index", "index", "task_index")},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                **({"sim_state": {"dtype": "float64", "shape": [ep.states.shape[1]], "names": None}}
                   if ep.states is not None else {}),
            }
        state = np.concatenate([ep.obs[k].reshape(T, -1) for k, _ in layout], axis=1).astype(np.float32)
        columns: dict[str, Any] = {
            f"observation.images.{cam}": [{"bytes": _png(img), "path": None} for img in ep.obs[k]]
            for k, cam in IMAGE_KEYS.items() if k in ep.obs
        }
        columns.update({f"observation.{k}": list(ep.obs[k].reshape(T, -1).astype(np.float32))
                        for k in _extra_keys(ep.obs)})
        columns.update({
            "observation.state": list(state),
            "action": list(ep.actions.astype(np.float32)),
            "next.reward": ep.rewards.astype(np.float32),
            "next.done": ep.dones.astype(bool),
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, n, dtype=np.int64),
            "index": np.arange(total_frames, total_frames + T, dtype=np.int64),
            "task_index": np.zeros(T, dtype=np.int64),
            "timestamp": (np.arange(T) / fps).astype(np.float32),
        })
        if ep.states is not None:
            columns["sim_state"] = list(np.asarray(ep.states, dtype=np.float64))
        out = root / "data" / f"chunk-{n // CHUNK_SIZE:03d}" / f"episode_{n:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), out)
        episodes_meta.append({"episode_index": n, "tasks": [meta.instruction], "length": T, "source_key": ep.key,
                              "split": "valid" if ep.key in valid else "train", "seed": ep.seed})
        for name, arr in (("observation.state", state), ("action", ep.actions.astype(np.float64))):
            acc = stats_acc.setdefault(name, [])
            acc.append(arr)
        total_frames += T
    n_episodes = len(episodes_meta)
    if features is None:
        raise ValueError("no episodes to convert")
    stats = {}
    for name, arrs in stats_acc.items():
        a = np.concatenate(arrs, axis=0)
        stats[name] = {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "min": a.min(0).tolist(),
                       "max": a.max(0).tolist()}
    n_train = sum(e["split"] == "train" for e in episodes_meta)
    info = {
        "codebase_version": LEROBOT_VERSION,
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": (n_episodes - 1) // CHUNK_SIZE + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_train}", "valid": f"{n_train}:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": features,
        "afford_bench": {"env_name": meta.env_name, "task_id": meta.task_id,
                         "state_layout": [[k, d] for k, d in layout], "camera": meta.camera,
                         "env_args": meta.env_args},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": meta.instruction}) + "\n")
    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes_meta))
    (root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
    return n_episodes


def read_lerobot(root: str | Path) -> tuple[DatasetMeta, Iterator[Episode]]:
    """Inverse of `write_lerobot`: dataset metadata and a lazy iterator over episodes."""
    import pyarrow.parquet as pq

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    task = json.loads((root / "meta" / "tasks.jsonl").read_text().splitlines()[0])["task"]
    eps = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    extra = info.get("afford_bench", {})
    meta = DatasetMeta(env_name=extra.get("env_name", ""), task_id=extra.get("task_id", ""), instruction=task,
                       train_keys=[f"demo_{e['episode_index']}" for e in eps if e.get("split") != "valid"],
                       valid_keys=[f"demo_{e['episode_index']}" for e in eps if e.get("split") == "valid"],
                       camera=extra.get("camera", {}), env_args=extra.get("env_args", ""))
    layout = [(k, int(d)) for k, d in extra.get("state_layout", [])]
    cams = {cam: k for k, cam in IMAGE_KEYS.items()}

    def episodes() -> Iterator[Episode]:
        for e in eps:
            i = e["episode_index"]
            table = pq.read_table(root / info["data_path"].format(episode_chunk=i // info["chunks_size"],
                                                                  episode_index=i)).to_pydict()
            obs = {}
            for name, column in table.items():
                if name.startswith("observation.images."):
                    obs[cams[name.split(".")[-1]]] = np.stack([_unpng(c["bytes"]) for c in column])
            for name, spec in info["features"].items():
                key = name.removeprefix("observation.")
                if name.startswith("observation.") and key not in ("state",) and not key.startswith("images."):
                    obs[key] = np.asarray(table[name], dtype=np.float32).reshape(-1, *spec["shape"])
            state = np.asarray(table["observation.state"], dtype=np.float32)
            start = 0
            for k, d in layout:
                obs[k] = state[:, start:start + d]
                start += d
            yield Episode(actions=np.asarray(table["action"], dtype=np.float32), obs=obs,
                          rewards=np.asarray(table["next.reward"], dtype=np.float32),
                          dones=np.asarray(table["next.done"], dtype=np.uint8), key=f"demo_{i}",
                          states=np.asarray(table["sim_state"], dtype=np.float64) if "sim_state" in table else None,
                          seed=e.get("seed"))

    return meta, episodes()


# --------------------------------------------------------------------------------------------------- RLDS
RLDS_IMAGE_KEYS = {"agentic_rgb": "image", "wrist_rgb": "wrist_image"}


def episode_to_rlds(ep: Episode, instruction: str, episode_id: str = "") -> dict[str, Any]:
    """RLDS episode in the Open X-Embodiment layout. observation.state is the 8-D vector (joint positions and
    gripper opening) used by OpenVLA and Octo dataset configs; the other proprioceptive keys are kept under
    their own names. is_terminal marks the final step of a successful demonstration."""
    T = len(ep)
    steps = []
    for t in range(T):
        observation = {RLDS_IMAGE_KEYS.get(k, k): v[t] for k, v in ep.obs.items()}
        if "joint_positions" in ep.obs and "gripper_width" in ep.obs:
            observation["state"] = np.concatenate([ep.obs["joint_positions"][t].reshape(-1),
                                                   ep.obs["gripper_width"][t].reshape(-1)]).astype(np.float32)
        steps.append({
            "observation": observation,
            "action": ep.actions[t].astype(np.float32),
            "reward": np.float32(ep.rewards[t]),
            "discount": np.float32(1.0),
            "is_first": t == 0,
            "is_last": t == T - 1,
            "is_terminal": bool(t == T - 1 and ep.dones[t]),
            "done": bool(ep.dones[t]),
            "language_instruction": instruction,
            **({"sim_state": np.asarray(ep.states[t], dtype=np.float64)} if ep.states is not None else {}),
        })
    metadata = {"file_path": episode_id, "source_key": ep.key}
    if ep.seed is not None:
        metadata["seed"] = int(ep.seed)
    return {"steps": steps, "episode_metadata": metadata}


def rlds_to_episode(episode: dict[str, Any]) -> Episode:
    """Inverse of `episode_to_rlds`."""
    steps = episode["steps"]
    inverse = {v: k for k, v in RLDS_IMAGE_KEYS.items()}
    keys = [k for k in steps[0]["observation"] if k != "state"]
    obs = {inverse.get(k, k): np.stack([s["observation"][k] for s in steps]) for k in keys}
    return Episode(actions=np.stack([s["action"] for s in steps]), obs=obs,
                   rewards=np.array([s["reward"] for s in steps], dtype=np.float32),
                   dones=np.array([s["done"] for s in steps], dtype=np.uint8),
                   key=episode["episode_metadata"].get("source_key", ""),
                   states=np.stack([s["sim_state"] for s in steps]) if "sim_state" in steps[0] else None,
                   seed=int(episode["episode_metadata"]["seed"]) if "seed" in episode["episode_metadata"] else None)


def save_rlds_npz(path: str | Path, episode: dict[str, Any]) -> None:
    """Store an RLDS episode as one .npz with flattened 'steps/<key>' arrays (TensorFlow-free interchange)."""
    steps = episode["steps"]
    arrays: dict[str, np.ndarray] = {}
    for k in steps[0]["observation"]:
        arrays[f"steps/observation/{k}"] = np.stack([s["observation"][k] for s in steps])
    for k in ("action", "reward", "discount", "is_first", "is_last", "is_terminal", "done", "sim_state"):
        if k in steps[0]:
            arrays[f"steps/{k}"] = np.stack([np.asarray(s[k]) for s in steps])
    arrays["steps/language_instruction"] = np.array([s["language_instruction"] for s in steps])
    for k, v in episode["episode_metadata"].items():
        arrays[f"episode_metadata/{k}"] = np.array(v)
    np.savez_compressed(path, **arrays)


def load_rlds_npz(path: str | Path) -> dict[str, Any]:
    """Inverse of `save_rlds_npz`."""
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    T = len(data["steps/action"])
    obs_keys = [k.split("/", 2)[2] for k in data if k.startswith("steps/observation/")]
    steps = []
    for t in range(T):
        step = {"observation": {k: data[f"steps/observation/{k}"][t] for k in obs_keys}}
        for k in ("action", "reward", "discount", "sim_state"):
            if f"steps/{k}" in data:
                step[k] = data[f"steps/{k}"][t]
        for k in ("is_first", "is_last", "is_terminal", "done"):
            step[k] = bool(data[f"steps/{k}"][t])
        step["language_instruction"] = str(data["steps/language_instruction"][t])
        steps.append(step)
    meta = {k.split("/", 1)[1]: str(v) for k, v in data.items() if k.startswith("episode_metadata/")}
    return {"steps": steps, "episode_metadata": meta}
