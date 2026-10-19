"""Train BC or diffusion-policy baselines on AFFORD-Bench Robomimic HDF5 datasets.

Modalities (afford_pro.learning.features): `lowdim` = proprioception + pose of the evaluated body (the low-dim
baseline), `visual` = workcell image + proprioception, `state` = proprioception only (a blind ablation: it cannot
see where the object starts).

The checkpoint stores the input normalizer, the input dimension, the training and validation losses, the list of
dataset files (name, size, mtime), and the code provenance. The saved checkpoint is the epoch with the lowest
validation loss (mask/valid), not the lowest training loss.

  python scripts/train_policy.py --policy diffusion --modality lowdim --task tabletop_sort_red --epochs 50
  python scripts/train_policy.py --policy bc --modality lowdim --domain tabletop --tier core --epochs 50 --seed 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro
from afford_pro.learning.features import (MODALITIES, Normalizer, input_dim, object_pose_from_states,
                                          object_poses_from_states, preprocess_image, proprio)
from afford_pro.learning.policies import BCPolicy, DiffusionPolicy, SinusoidalPosEmb  # noqa: F401  (re-exported)

DOMAIN_MATCH = {"industrial": "Industrial", "kitchen": "Culinary", "tabletop": "Tabletop", "lab": "Laboratory",
                "long_horizon": "Long-Horizon", "chain_precision": "Precision Chaining", "chain_service": "Service & Culinary"}


def _split_keys(f: h5py.File, split: str) -> list[str]:
    if "mask" in f and split in f["mask"]:
        return [k.decode() if isinstance(k, bytes) else str(k) for k in f["mask"][split][()]]
    keys = sorted((k for k in f["data"] if k.startswith("demo_")), key=lambda k: int(k.split("_")[1]))
    cut = int(0.9 * len(keys))
    return keys[:cut] if split == "train" else keys[cut:]


def _padded_chunks(actions: np.ndarray, chunk: int) -> np.ndarray:
    """Action chunk starting at every step; the tail is padded with the last action so the final (release)
    states are also training inputs."""
    padded = np.concatenate([actions, np.repeat(actions[-1:], chunk - 1, axis=0)])
    return np.stack([padded[t:t + chunk] for t in range(len(actions))])


class AffordDemonstrationDataset(Dataset):
    """Windows (input at step t, actions t..t+chunk-1) from one split of one or more dataset files."""

    def __init__(self, hdf5_paths: list[Path], chunk_size: int = 16, split: str = "train", modality: str = "lowdim",
                 subsample_rate: int = 2, normalizer: Normalizer | None = None):
        from afford_pro.datasets.validation import build_task_context, file_task_set, parse_dataset_name

        self.chunk_size, self.modality, self.split = chunk_size, modality, split
        self.vectors: list[np.ndarray] = []
        self.chunks: list[np.ndarray] = []
        self.image_refs: list[tuple[str, str, int]] = []
        self._handles: dict[str, h5py.File] = {}
        for p in hdf5_paths:
            ctx = build_task_context(*parse_dataset_name(p), file_task_set(p)) if modality in ("lowdim", "objects") else None
            if modality == "lowdim" and ctx.object_qpos_adr is None:
                raise ValueError(f"{p.name}: lowdim needs one free evaluated body (legacy); use --modality objects")
            f = h5py.File(str(p), "r")
            self._handles[str(p)] = f
            for key in _split_keys(f, split):
                self._add_demo(f, str(p), key, ctx, subsample_rate)
        self.vectors_arr = np.concatenate(self.vectors) if self.vectors else np.zeros((0, input_dim(modality)), np.float32)
        self.chunks_arr = np.concatenate(self.chunks) if self.chunks else np.zeros((0, chunk_size, 7), np.float32)
        self.normalizer = normalizer or (Normalizer.fit(self.vectors_arr) if len(self.vectors_arr) else None)
        print(f"-> {split}: {len(self.chunks_arr):,} windows from {len(hdf5_paths)} files ({modality})", flush=True)

    def _add_demo(self, f: h5py.File, path: str, key: str, ctx: Any, subsample: int) -> None:
        g = f["data"][key]
        actions = np.asarray(g["actions"][()], dtype=np.float32)
        steps = np.arange(0, len(actions), subsample)
        vec = proprio(g["obs"], slice(None))
        if self.modality == "lowdim":
            vec = np.concatenate([vec, object_pose_from_states(g["states"][()], ctx.qpos_offset, ctx.object_qpos_adr)], axis=-1)
        elif self.modality == "objects":
            vec = np.concatenate([vec, object_poses_from_states(g["states"][()], ctx.qpos_offset, ctx.moved_qpos_adrs)],
                                 axis=-1)
        self.vectors.append(vec[steps])
        self.chunks.append(_padded_chunks(actions, self.chunk_size)[steps])
        if self.modality == "visual":
            self.image_refs.extend((path, key, int(t)) for t in steps)

    def __len__(self) -> int:
        return len(self.chunks_arr)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"proprio": torch.from_numpy(self.normalizer(self.vectors_arr[idx])),
                "actions": torch.from_numpy(self.chunks_arr[idx])}
        if self.modality == "visual":
            path, key, t = self.image_refs[idx]
            item["image"] = torch.from_numpy(preprocess_image(self._handles[path]["data"][key]["obs"]["agentic_rgb"][t]))
        return item


def _loss(model: torch.nn.Module, policy: str, batch: dict, device: torch.device,
          generator: torch.Generator | None = None) -> torch.Tensor:
    proprio_t = batch["proprio"].to(device)
    actions = batch["actions"].to(device)
    img = batch["image"].to(device) if "image" in batch else None
    if policy == "bc":
        return F.l1_loss(model(proprio_t, img), actions)
    b = actions.size(0)
    t = torch.randint(0, model.diffusion_steps, (b,), device=device, generator=generator).long()
    noise = torch.randn(actions.shape, device=device, generator=generator)
    alpha_bar = model.alphas_cumprod[t].view(b, 1, 1)
    noisy = torch.sqrt(alpha_bar) * actions + torch.sqrt(1 - alpha_bar) * noise
    return F.mse_loss(model(noisy, t, proprio_t, img), noise)


@torch.no_grad()
def _validation_loss(model, policy: str, loader: DataLoader, device: torch.device, seed: int) -> float:
    """Mean validation loss; diffusion noise and timesteps come from a fixed generator so epochs are comparable."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    losses = [float(_loss(model, policy, batch, device, gen)) for batch in loader]
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def _select_files(args: argparse.Namespace, data_dir: Path) -> list[Path]:
    if args.task:
        tasks = [args.task]
    else:
        tasks = [t for t in afford_pro.list_tasks(args.task_set)
                 if args.domain == "all" or DOMAIN_MATCH[args.domain] in afford_pro.get_task_entry(t, args.task_set).domain]
        if args.tier:
            from afford_pro.task_tiers import tasks_in_tier
            tasks = [t for t in tasks if t in set(tasks_in_tier(args.tier))]
    paths = [data_dir / f"{t}_nominal.hdf5" for t in tasks]
    missing = [p.name for p in paths if not p.exists()]
    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} selected datasets are missing ({missing[:5]}...); a policy "
                         "trained without them would be scored on tasks it never saw. Pass --allow-missing to proceed.")
    return [p for p in paths if p.exists()]


def _build_model(policy: str, modality: str, chunk: int) -> torch.nn.Module:
    cls = DiffusionPolicy if policy == "diffusion" else BCPolicy
    return cls(action_dim=7, chunk_size=chunk, proprio_dim=input_dim(modality), modality=modality)


def _checkpoint(model, args, epoch: int, train_loss: float, val_loss: float, normalizer: Normalizer,
                files: list[Path]) -> dict[str, Any]:
    from afford_pro.provenance import collect_provenance
    return {"epoch": epoch, "policy": args.policy, "modality": args.modality, "chunk_size": args.chunk_size,
            "input_dim": input_dim(args.modality), "model_state_dict": model.state_dict(),
            "loss": train_loss, "val_loss": val_loss, "selection": "min validation loss",
            "normalizer": normalizer.to_dict(), "args": {k: str(v) for k, v in vars(args).items()},
            "datasets": [{"file": p.name, "size_bytes": p.stat().st_size, "mtime": p.stat().st_mtime} for p in files],
            "provenance": collect_provenance()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", type=str, default="diffusion", choices=["diffusion", "bc"])
    parser.add_argument("--modality", type=str, default="lowdim", choices=MODALITIES)
    parser.add_argument("--domain", type=str, default="industrial", choices=["all", *DOMAIN_MATCH])
    parser.add_argument("--tier", type=str, default=None, choices=[None, "core", "reach", "instruction_mismatch"])
    parser.add_argument("--task", type=str, default=None, help="Train on a single task ID")
    parser.add_argument("--task-set", choices=("bench", "legacy"), default="bench",
                        help="task set: bench (default) or the deprecated legacy set")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--subsample", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--tag", type=str, default="", help="Suffix for the checkpoint name (e.g. seed1)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Train even if some selected tasks have no dataset (they are listed in the checkpoint)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.task_set == "bench" and args.tier:
        parser.error("--tier applies to the legacy tasks only")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    files = _select_files(args, ROOT / args.data_dir)
    if not files:
        print(f"Error: no datasets for domain={args.domain} task={args.task} tier={args.tier}", flush=True)
        return 1
    train_ds = AffordDemonstrationDataset(files, args.chunk_size, "train", args.modality, args.subsample)
    valid_ds = AffordDemonstrationDataset(files, args.chunk_size, "valid", args.modality, 1, normalizer=train_ds.normalizer)
    if len(train_ds) == 0:
        print("Dataset is empty.", flush=True)
        return 1
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))
    valid_loader = DataLoader(valid_ds, batch_size=256, shuffle=False)
    device = torch.device(args.device)
    model = _build_model(args.policy, args.modality, args.chunk_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    name = args.task or (args.domain + (f"_{args.tier}" if args.tier else ""))
    ckpt_path = ROOT / args.save_dir / f"{args.policy}_{args.modality}_{name}{'_' + args.tag if args.tag else ''}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0, losses = time.monotonic(), []
        for batch in loader:
            optimizer.zero_grad()
            loss = _loss(model, args.policy, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        val_loss = _validation_loss(model, args.policy, valid_loader, device, args.seed)
        score = val_loss if np.isfinite(val_loss) else train_loss
        print(f"Epoch {epoch:3d}/{args.epochs} | train {train_loss:.5f} | valid {val_loss:.5f} | {time.monotonic() - t0:.1f}s", flush=True)
        if score < best:
            best = score
            torch.save(_checkpoint(model, args, epoch, train_loss, val_loss, train_ds.normalizer, files), str(ckpt_path))
    print(f"Best validation loss {best:.5f}; checkpoint {ckpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
