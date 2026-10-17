"""Policy Training Script for AFFORD-Bench.

Ready-to-run PyTorch training harness for training robot manipulation policies
(Diffusion Policy and Behavioral Cloning) on AFFORD-Bench HDF5 demonstration datasets.
Supports both state-based proprioceptive policies and visuomotor policies.

Usage Examples:
  # Fast state-based Diffusion Policy training (CPU/GPU, finishes in minutes)
  python scripts/train_policy.py --policy diffusion --modality state --domain industrial --epochs 20

  # Fast state-based BC Policy on a single task
  python scripts/train_policy.py --policy bc --modality state --task industrial_shaft_assembly --epochs 10

  # Full Visuomotor Diffusion Policy training
  python scripts/train_policy.py --policy diffusion --modality visual --domain kitchen --epochs 30 --batch-size 32
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro


# ==============================================================================
# 1. Dataset Loader for Robomimic / AFFORD-Bench HDF5 Files
# ==============================================================================
class AffordDemonstrationDataset(Dataset):
    """Loads state-action or visuomotor pairs from AFFORD-Bench HDF5 datasets."""

    def __init__(
        self,
        hdf5_paths: list[Path],
        chunk_size: int = 16,
        split: str = "train",
        modality: str = "state",
        subsample_rate: int = 2,
    ):
        self.chunk_size = chunk_size
        self.modality = modality
        self.samples: list[dict] = []
        self._handles: dict[str, h5py.File] = {}

        print(f"Loading {len(hdf5_paths)} dataset files for split '{split}' (modality='{modality}')...", flush=True)

        for p in hdf5_paths:
            if not p.exists():
                continue
            try:
                f = h5py.File(str(p), "r")
                self._handles[str(p)] = f
                if "data" not in f:
                    continue

                if "mask" in f and split in f["mask"]:
                    demo_keys = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in f["mask"][split][:]]
                else:
                    all_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
                    split_idx = int(0.9 * len(all_keys))
                    demo_keys = all_keys[:split_idx] if split == "train" else all_keys[split_idx:]

                for dkey in demo_keys:
                    if dkey not in f["data"]:
                        continue
                    demo_grp = f["data"][dkey]
                    ep_len = len(demo_grp["actions"])

                    # If state modality, pre-cache in memory for extreme training speed
                    if modality == "state":
                        actions_all = np.array(demo_grp["actions"], dtype=np.float32)
                        obs_grp = demo_grp["obs"]
                        joint_pos = np.array(obs_grp["joint_positions"], dtype=np.float32)
                        joint_vel = np.array(obs_grp["joint_velocities"], dtype=np.float32)
                        tcp_pose = np.array(obs_grp["tcp_pose"], dtype=np.float32)
                        grip_w = np.array(obs_grp["gripper_width"], dtype=np.float32)
                        proprio_all = np.concatenate([tcp_pose, joint_pos, joint_vel, grip_w], axis=-1)

                        for t in range(0, ep_len - chunk_size + 1, subsample_rate):
                            self.samples.append({
                                "proprio": proprio_all[t],
                                "actions": actions_all[t : t + chunk_size],
                            })
                    else:
                        for t in range(0, ep_len - chunk_size + 1, subsample_rate):
                            self.samples.append({
                                "file_path": str(p),
                                "demo_key": dkey,
                                "t": t,
                            })

            except Exception as e:
                print(f"  Warning: could not index {p.name}: {e}", flush=True)

        print(f"-> Indexed {len(self.samples):,} transition windows ({split}).", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        if self.modality == "state":
            return {
                "proprio": torch.from_numpy(item["proprio"]),
                "actions": torch.from_numpy(item["actions"]).float(),
            }

        # Visual modality: retrieve from open file handle
        f = self._handles[item["file_path"]]
        demo_grp = f["data"][item["demo_key"]]
        t = item["t"]
        actions = demo_grp["actions"][t : t + self.chunk_size]

        obs_grp = demo_grp["obs"]
        joint_pos = obs_grp["joint_positions"][t]
        joint_vel = obs_grp["joint_velocities"][t]
        tcp_pose = obs_grp["tcp_pose"][t]
        grip_w = obs_grp["gripper_width"][t]
        proprio = np.concatenate([tcp_pose, joint_pos, joint_vel, grip_w]).astype(np.float32)

        raw_img = obs_grp["agentic_rgb"][t]
        img = raw_img[::8, ::8, :3][:80, :80, :].transpose(2, 0, 1).astype(np.float32) / 255.0

        return {
            "proprio": torch.from_numpy(proprio),
            "image": torch.from_numpy(img),
            "actions": torch.from_numpy(actions).float(),
        }


# ==============================================================================
# 2. Policy Architectures (BC and Diffusion Policy)
# ==============================================================================
class BCPolicy(nn.Module):
    """Behavioral Cloning Policy supporting both State and Visuomotor inputs."""

    def __init__(self, action_dim: int = 7, chunk_size: int = 16, proprio_dim: int = 22, modality: str = "state"):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.modality = modality

        if modality == "visual":
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            in_dim = 64 * 4 * 4 + proprio_dim
        else:
            in_dim = proprio_dim

        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, chunk_size * action_dim),
        )

    def forward(self, proprio: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        if self.modality == "visual" and image is not None:
            v_feat = self.conv(image)
            feat = torch.cat([v_feat, proprio], dim=-1)
        else:
            feat = proprio
        out = self.head(feat)
        return out.view(-1, self.chunk_size, self.action_dim)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionPolicy(nn.Module):
    """1D Action Diffusion Policy with DDPM variance schedule."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        proprio_dim: int = 22,
        modality: str = "state",
        diffusion_steps: int = 50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.modality = modality
        self.diffusion_steps = diffusion_steps

        if modality == "visual":
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            cond_in = 64 * 4 * 4 + proprio_dim
        else:
            cond_in = proprio_dim

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        flat_act_dim = chunk_size * action_dim
        self.denoise_net = nn.Sequential(
            nn.Linear(flat_act_dim + 128 + 128, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, flat_act_dim),
        )

        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        proprio: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modality == "visual" and image is not None:
            v_feat = self.conv(image)
            cond_inp = torch.cat([v_feat, proprio], dim=-1)
        else:
            cond_inp = proprio

        cond = self.cond_mlp(cond_inp)
        t_emb = self.time_mlp(timesteps)

        B = noisy_actions.size(0)
        flat_act = noisy_actions.view(B, -1)
        x = torch.cat([flat_act, cond, t_emb], dim=-1)
        noise_pred = self.denoise_net(x)
        return noise_pred.view(B, self.chunk_size, self.action_dim)

    @torch.no_grad()
    def sample_actions(self, proprio: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        """Reverse diffusion sampling from pure Gaussian noise."""
        B = proprio.size(0)
        device = proprio.device
        act = torch.randn(B, self.chunk_size, self.action_dim, device=device)

        for step in reversed(range(self.diffusion_steps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            pred_noise = self.forward(act, t, proprio, image)

            alpha = self.alphas[step]
            alpha_cumprod = self.alphas_cumprod[step]
            beta = self.betas[step]

            noise = torch.randn_like(act) if step > 0 else 0.0
            act = (1 / torch.sqrt(alpha)) * (act - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise)
            if step > 0:
                act = act + torch.sqrt(beta) * noise

        return torch.clamp(act, -1.0, 1.0)


# ==============================================================================
# 3. Main Training Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train robot learning policies on AFFORD-Bench datasets.")
    parser.add_argument("--policy", type=str, default="diffusion", choices=["diffusion", "bc"])
    parser.add_argument("--modality", type=str, default="state", choices=["state", "visual"])
    parser.add_argument(
        "--domain",
        type=str,
        default="industrial",
        choices=["all", "industrial", "kitchen", "tabletop", "lab", "long_horizon", "chain_precision", "chain_service"],
    )
    parser.add_argument("--task", type=str, default=None, help="Train specifically on a single task ID")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--subsample", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = ROOT / args.data_dir
    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.task:
        target_files = [data_dir / f"{args.task}_nominal.hdf5"]
    elif args.domain == "all":
        target_files = sorted(list(data_dir.glob("*_nominal.hdf5")))
    else:
        domain_prefix_map = {
            "industrial": "industrial_",
            "kitchen": "kitchen_",
            "tabletop": "tabletop_",
            "lab": "lab_",
            "long_horizon": "long_horizon_",
            "chain_precision": "chain_",
            "chain_service": "chain_",
        }
        pfx = domain_prefix_map[args.domain]
        domain_match = {"industrial": "Industrial", "kitchen": "Culinary", "tabletop": "Tabletop", "lab": "Laboratory", "long_horizon": "Long-Horizon", "chain_precision": "Precision Chaining", "chain_service": "Service & Culinary"}
        target_dom = domain_match[args.domain]
        target_files = [data_dir / f"{t}_nominal.hdf5" for t in afford_pro.list_tasks() if target_dom in afford_pro.get_task_entry(t).domain]

    existing_files = [p for p in target_files if p.exists()]
    if not existing_files:
        print(f"Error: No HDF5 files found matching domain '{args.domain}' or task '{args.task}' in {data_dir}.", flush=True)
        return

    print("\n" + "=" * 80, flush=True)
    print(" AFFORD-Bench: POLICY TRAINING HARNESS", flush=True)
    print(f" Policy Model  : {args.policy.upper()}", flush=True)
    print(f" Modality      : {args.modality.upper()}", flush=True)
    print(f" Domain Target : {args.domain.upper() if not args.task else args.task}", flush=True)
    print(f" Datasets      : {len(existing_files)} HDF5 files", flush=True)
    print(f" Action Chunk  : {args.chunk_size} timesteps", flush=True)
    print(f" Batch Size    : {args.batch_size}", flush=True)
    print(f" Epochs        : {args.epochs}", flush=True)
    print(f" Device        : {args.device}", flush=True)
    print("=" * 80 + "\n", flush=True)

    train_ds = AffordDemonstrationDataset(
        existing_files,
        chunk_size=args.chunk_size,
        split="train",
        modality=args.modality,
        subsample_rate=args.subsample,
    )
    if len(train_ds) == 0:
        print("Dataset is empty. Ensure demonstration datasets are generated first.", flush=True)
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device(args.device)
    if args.policy == "diffusion":
        model = DiffusionPolicy(action_dim=7, chunk_size=args.chunk_size, modality=args.modality).to(device)
    else:
        model = BCPolicy(action_dim=7, chunk_size=args.chunk_size, modality=args.modality).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = float("inf")
    save_tag = f"{args.policy}_{args.modality}_{args.domain if not args.task else args.task}"
    ckpt_path = save_dir / f"{save_tag}_best.pt"

    print(f"Commencing training across {args.epochs} epochs...\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.monotonic()

        for batch in train_loader:
            proprio = batch["proprio"].to(device)
            actions = batch["actions"].to(device)
            img = batch["image"].to(device) if args.modality == "visual" else None

            optimizer.zero_grad()

            if args.policy == "diffusion":
                B = actions.size(0)
                timesteps = torch.randint(0, model.diffusion_steps, (B,), device=device).long()
                noise = torch.randn_like(actions)
                alpha_bar = model.alphas_cumprod[timesteps].view(B, 1, 1)
                noisy_act = torch.sqrt(alpha_bar) * actions + torch.sqrt(1 - alpha_bar) * noise

                pred_noise = model(noisy_act, timesteps, proprio, img)
                loss = F.mse_loss(pred_noise, noise)
            else:
                pred_actions = model(proprio, img)
                loss = F.l1_loss(pred_actions, actions)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.monotonic() - t0

        print(f"Epoch [{epoch:3d}/{args.epochs:3d}] | Loss: {avg_loss:.5f} | Batch Time: {elapsed:.2f}s ({n_batches} batches)", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "policy": args.policy,
                    "modality": args.modality,
                    "chunk_size": args.chunk_size,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                    "args": vars(args),
                },
                str(ckpt_path),
            )

    print("\n" + "=" * 80, flush=True)
    print(" TRAINING COMPLETE", flush=True)
    print(f" Best Loss: {best_loss:.5f}", flush=True)
    print(f" Checkpoint Saved To: {ckpt_path}", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
