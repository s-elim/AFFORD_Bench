"""Out-Of-Distribution (OOD) Policy Evaluation Script for AFFORD-Bench.

Evaluates trained policy checkpoints or reference baselines across the 5 canonical
AFFORD-Bench Out-of-Distribution perturbation suites:
1. Spatial-Kinematic Perturbations (position shifts +/-10cm, rotations +/-30 deg)
2. Physical-Dynamic Perturbations (Coulomb friction mu in [0.1, 1.2], mass scaling 0.5x-3.0x)
3. Visual-Photometric Perturbations (ambient lighting +/-50%, specular shifts)
4. Object Clearance Perturbations (sub-millimeter tolerance reduction)
5. Composite Multi-Axis OOD Perturbations (simultaneous multi-axis shifts)

Usage Examples:
  # Evaluate a trained Diffusion Policy checkpoint across all OOD regimes on Industrial domain
  python scripts/evaluate_ood.py --checkpoint checkpoints/diffusion_state_industrial_shaft_assembly_best.pt --task industrial_shaft_assembly --num-episodes 10

  # Evaluate reference Oracle across Physical perturbations
  python scripts/evaluate_ood.py --policy oracle --domain industrial --perturbation p4_physical --num-episodes 5

  # Evaluate across Composite OOD suite
  python scripts/evaluate_ood.py --policy oracle --domain kitchen --perturbation combined_ood --num-episodes 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import afford_pro
from scripts.train_policy import BCPolicy, DiffusionPolicy


def load_policy(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    policy_type = ckpt.get("policy", "diffusion")
    modality = ckpt.get("modality", "state")
    chunk_size = ckpt.get("chunk_size", 16)

    if policy_type == "diffusion":
        model = DiffusionPolicy(action_dim=7, chunk_size=chunk_size, modality=modality).to(device)
    else:
        model = BCPolicy(action_dim=7, chunk_size=chunk_size, modality=modality).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, modality, policy_type


def run_episode(
    env: afford_pro.BaseFrankaEnv,
    oracle: Any | None,
    model: torch.nn.Module | None,
    modality: str,
    policy_name: str,
    device: torch.device,
    seed: int,
    max_steps: int = 400,
) -> tuple[bool, int, float]:
    obs, info = env.reset(seed=seed)
    t0 = time.monotonic()
    done = False
    step_count = 0

    if policy_name == "oracle":
        oracle.generate_episode(seed=seed)
        success = env.check_success()
        return success, env.current_step, time.monotonic() - t0

    action_buffer = []

    while not done and step_count < max_steps:
        if policy_name == "random":
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            step_count += 1
        elif model is not None:
            if len(action_buffer) == 0:
                proprio = np.concatenate([
                    obs["tcp_pose"],
                    obs["joint_positions"],
                    obs["joint_velocities"],
                    obs["gripper_width"],
                ]).astype(np.float32)

                prop_t = torch.from_numpy(proprio).unsqueeze(0).to(device)
                img_t = None
                if modality == "visual" and "agentic_rgb" in obs:
                    raw_img = obs["agentic_rgb"]
                    img = raw_img[::8, ::8, :3][:80, :80, :].transpose(2, 0, 1).astype(np.float32) / 255.0
                    img_t = torch.from_numpy(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    if isinstance(model, DiffusionPolicy):
                        act_chunk = model.sample_actions(prop_t, img_t).squeeze(0).cpu().numpy()
                    else:
                        act_chunk = model(prop_t, img_t).squeeze(0).cpu().numpy()

                action_buffer = list(act_chunk)

            act = action_buffer.pop(0)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            step_count += 1

    success = env.check_success()
    dur = time.monotonic() - t0
    return success, step_count, dur


def evaluate_task_split(
    task_id: str,
    perturbation: str,
    policy_name: str,
    model: torch.nn.Module | None,
    modality: str,
    device: torch.device,
    num_episodes: int = 10,
    base_seed: int = 20000,
) -> dict:
    env = afford_pro.make(task_id, perturbation_mode=perturbation)
    oracle = afford_pro.get_oracle(task_id, env) if policy_name == "oracle" else None

    successes = []
    steps = []

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        suc, s_cnt, dur = run_episode(env, oracle, model, modality, policy_name, device, ep_seed)
        successes.append(float(suc))
        steps.append(s_cnt)

    return {
        "sr": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies across AFFORD-Bench OOD Perturbation Suites.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained policy .pt checkpoint")
    parser.add_argument("--policy", type=str, default="checkpoint", choices=["checkpoint", "oracle", "random"])
    parser.add_argument(
        "--domain",
        type=str,
        default="industrial",
        choices=["all", "industrial", "kitchen", "tabletop", "lab", "long_horizon", "chain_precision", "chain_service", "frontier"],
    )
    parser.add_argument("--task", type=str, default=None, help="Evaluate specifically on a single task ID")
    parser.add_argument(
        "--perturbation",
        type=str,
        default="all",
        choices=["all", "nominal", "p1_object", "p2_spatial", "p3_visual", "p4_physical", "combined_ood"],
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Episodes per perturbation suite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=300_000, help="Base seed offset for segregated evaluation seed block")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = None
    modality = "state"
    policy_name = args.policy

    if args.checkpoint:
        ckpt_p = Path(args.checkpoint)
        if not ckpt_p.exists():
            print(f"Error: Checkpoint {ckpt_p} does not exist.")
            return
        model, modality, pol_type = load_policy(ckpt_p, device)
        policy_name = f"{pol_type} ({modality})"
        print(f"Loaded checkpoint '{ckpt_p.name}' ({policy_name}).", flush=True)

    # Resolve tasks
    all_tasks = afford_pro.list_all_tasks(include_frontier=True)
    selected_tasks = []
    if args.task:
        selected_tasks = [args.task]
    elif args.domain == "all":
        selected_tasks = all_tasks
    else:
        domain_match = {
            "industrial": "Industrial",
            "kitchen": "Culinary",
            "tabletop": "Tabletop",
            "lab": "Laboratory",
            "long_horizon": "Long-Horizon",
            "chain_precision": "Precision Chaining",
            "chain_service": "Service & Culinary",
            "frontier": "Frontier Complex",
        }
        target_dom = domain_match[args.domain]
        selected_tasks = [t for t in all_tasks if target_dom in afford_pro.get_task_entry(t).domain]

    # Resolve perturbation splits
    if args.perturbation == "all":
        perturbation_splits = ["nominal", "p1_object", "p2_spatial", "p3_visual", "combined_ood"]
    else:
        perturbation_splits = [args.perturbation]

    print("\n" + "=" * 80, flush=True)
    print(" AFFORD-Bench: OUT-OF-DISTRIBUTION (OOD) EVALUATION PROTOCOL", flush=True)
    print(f" Policy Under Test : {policy_name.upper()}", flush=True)
    print(f" Evaluation Domain : {args.domain.upper() if not args.task else args.task}", flush=True)
    print(f" Tasks In Scope    : {len(selected_tasks)} tasks", flush=True)
    print(f" Perturbation Axes : {', '.join(perturbation_splits)}", flush=True)
    print(f" Episodes Per Split: {args.num_episodes}", flush=True)
    print("=" * 80 + "\n", flush=True)

    task_results = []

    for idx, tid in enumerate(selected_tasks, 1):
        entry = afford_pro.get_task_entry(tid)
        print(f"[{idx:2d}/{len(selected_tasks):2d}] Task #{entry.task_number:03d}: '{tid}' [{entry.domain}]", flush=True)

        row = {"task_id": tid, "domain": entry.domain}
        for split in perturbation_splits:
            res = evaluate_task_split(
                tid,
                split,
                policy_name,
                model,
                modality,
                device,
                num_episodes=args.num_episodes,
                base_seed=args.seed + entry.task_number * 1000,
            )
            row[split] = res["sr"] * 100.0
            print(f"     -> {split:<15}: {res['sr']*100:5.1f}% SR", flush=True)

        # Compute GDI if both nominal and ood are evaluated
        if "nominal" in row and len(perturbation_splits) > 1:
            id_sr = row["nominal"]
            ood_vals = [row[s] for s in perturbation_splits if s != "nominal"]
            ood_mean = float(np.mean(ood_vals))
            pri = (ood_mean / max(1e-4, id_sr)) if id_sr > 0 else 0.0
            gdi = ((id_sr - ood_mean) / max(1e-4, id_sr)) * 100.0 if id_sr > 0 else 0.0
            row["ood_mean"] = ood_mean
            row["pri"] = pri
            row["gdi"] = gdi
            print(f"     -> [SUMMARY] ID: {id_sr:.1f}% | OOD Mean: {ood_mean:.1f}% | PRI: {pri:.2f} | GDI: {gdi:.1f}%", flush=True)

        task_results.append(row)
        print("-" * 80, flush=True)

    # Aggregate Summary
    print("\n" + "=" * 80, flush=True)
    print(" BENCHMARK OOD AGGREGATE SUMMARY", flush=True)
    print(f" Policy: {policy_name.upper()}", flush=True)
    for split in perturbation_splits:
        vals = [r[split] for r in task_results if split in r]
        print(f"  - {split:<16} Aggregate SR: {np.mean(vals):5.1f}%", flush=True)

    if "nominal" in perturbation_splits and len(perturbation_splits) > 1:
        all_gdis = [r["gdi"] for r in task_results if "gdi" in r]
        all_pris = [r["pri"] for r in task_results if "pri" in r]
        print(f"\n  Average Perturbation Resilience Index (PRI) : {np.mean(all_pris):.3f} (Higher is better, max 1.0)")
        print(f"  Average Generalization Degradation Index (GDI): {np.mean(all_gdis):.1f}% (Lower is better, min 0.0%)")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
