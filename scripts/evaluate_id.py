"""In-Distribution (ID / Nominal) Policy Evaluation Script for AFFORD-Bench.

Evaluates trained policy checkpoints or reference baselines under standard
In-Distribution (Nominal) conditions using disjoint evaluation seeds.

Usage Examples:
  # Evaluate a trained Diffusion Policy checkpoint on industrial domain
  python scripts/evaluate_id.py --checkpoint checkpoints/diffusion_state_industrial_shaft_assembly_best.pt --task industrial_shaft_assembly --num-episodes 20

  # Evaluate reference Oracle across an entire domain
  python scripts/evaluate_id.py --policy oracle --domain industrial --num-episodes 10

  # Evaluate Random baseline
  python scripts/evaluate_id.py --policy random --domain kitchen --num-episodes 5
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
    init_state: np.ndarray | None = None,
) -> tuple[bool, int, float]:
    obs, info = env.reset(seed=seed)
    if init_state is not None:
        env.set_sim_state(init_state)
        obs = env.get_observation()
    t0 = time.monotonic()
    done = False
    step_count = 0

    if policy_name == "oracle":
        oracle.generate_episode(seed=seed, reset=(init_state is None))
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
            # Query policy if buffer empty
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


def evaluate_task(
    task_id: str,
    policy_name: str,
    model: torch.nn.Module | None,
    modality: str,
    device: torch.device,
    num_episodes: int = 20,
    base_seed: int = 200_000,
    use_init_states: bool = False,
) -> dict:
    env = afford_pro.make(task_id, perturbation_mode="nominal")
    oracle = afford_pro.get_oracle(task_id, env) if policy_name == "oracle" else None

    init_states = afford_pro.get_initial_states(task_id) if use_init_states else None
    if use_init_states and init_states is None:
        print(f" (Note: fixed init states not found for {task_id}; using reset seed)", end="")

    successes = []
    steps = []
    durations = []

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        init_st = init_states[ep % len(init_states)] if init_states is not None else None
        suc, s_cnt, dur = run_episode(env, oracle, model, modality, policy_name, device, ep_seed, init_state=init_st)
        successes.append(float(suc))
        steps.append(s_cnt)
        durations.append(dur)

    return {
        "task_id": task_id,
        "sr": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps)),
        "avg_time": float(np.mean(durations)),
        "num_episodes": num_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies on AFFORD-Bench In-Distribution (Nominal) tasks.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained policy .pt checkpoint")
    parser.add_argument("--policy", type=str, default="checkpoint", choices=["checkpoint", "oracle", "random"])
    parser.add_argument(
        "--domain",
        type=str,
        default="industrial",
        choices=["all", "industrial", "kitchen", "tabletop", "lab", "long_horizon", "chain_precision", "chain_service", "frontier"],
    )
    parser.add_argument("--task", type=str, default=None, help="Evaluate specifically on a single task ID")
    parser.add_argument("--num-episodes", type=int, default=10, help="Episodes per task")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=200_000, help="Base seed offset for segregated evaluation seed block")
    parser.add_argument("--use-init-states", action="store_true", help="Initialize directly from pre-generated fixed initial states")
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

    # Select tasks
    all_tasks = afford_pro.list_all_tasks(include_frontier=(args.domain == "frontier"))
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

    print("\n" + "=" * 80, flush=True)
    print(" AFFORD-Bench: IN-DISTRIBUTION (ID) EVALUATION PROTOCOL", flush=True)
    print(f" Policy Under Test : {policy_name.upper()}", flush=True)
    print(f" Evaluation Domain : {args.domain.upper() if not args.task else args.task}", flush=True)
    print(f" Tasks In Scope    : {len(selected_tasks)} tasks", flush=True)
    print(f" Episodes Per Task : {args.num_episodes}", flush=True)
    print(f" Disjoint Seeds    : Base seed block [{args.seed} + task_number * 1000]", flush=True)
    print(f" Fixed Init States : {'ENABLED' if args.use_init_states else 'DISABLED (RNG reset)'}", flush=True)
    print("=" * 80 + "\n", flush=True)

    results = []
    for idx, tid in enumerate(selected_tasks, 1):
        entry = afford_pro.get_task_entry(tid)
        task_base_seed = args.seed + entry.task_number * 1000
        print(f"[{idx:2d}/{len(selected_tasks):2d}] Evaluating Task #{entry.task_number:03d}: '{tid}'...", end=" ", flush=True)
        res = evaluate_task(
            tid,
            policy_name,
            model,
            modality,
            device,
            num_episodes=args.num_episodes,
            base_seed=task_base_seed,
            use_init_states=args.use_init_states,
        )
        results.append(res)
        print(f"SR: {res['sr']*100:5.1f}% | Avg Steps: {res['avg_steps']:5.1f}", flush=True)

    all_srs = [r["sr"] for r in results]
    mean_sr = float(np.mean(all_srs)) * 100.0

    print("\n" + "=" * 80, flush=True)
    print(" IN-DISTRIBUTION (ID) EVALUATION SUMMARY", flush=True)
    print(f" Policy Evaluated  : {policy_name.upper()}", flush=True)
    print(f" Total Evaluated   : {len(results)} tasks ({len(results) * args.num_episodes:,} rollouts)", flush=True)
    print(f" Mean Success Rate : {mean_sr:5.2f}%", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
