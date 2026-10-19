"""Paper figure: one AFFORD-Bench task per scene, rendered at its start and after the scripted oracle completed it
(arm returned to its ready posture so it does not hide the result), with the task's instruction. Episodes come from
the evaluation seed block; the first episode the oracle completes is shown.

  python scripts/make_task_figure.py --out "AFFORD-Bench_RSS'27/paper-template-latex/figures/tasks"
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (task number, panel title): one multi-goal task per scene.
PANELS = [(121, "Assembly jig"), (169, "Electronics bench"), (125, "Burger station"),
          (127, "Espresso station"), (132, "Laboratory bench"), (148, "Sorting table")]
SEED_BLOCK = 200_000   # the evaluation seed block of scripts/evaluate_policy_suite.py
MAX_TRIES = 5
IMAGE_HW = (360, 480)
CAMERA = {"elevation": -52.0, "azimuth": 180.0}   # from the front of the table, the robot at the back
DISTANCE = (0.55, 0.85)   # camera distance range (m); each panel is framed on the points its task involves
WRAP = 42                 # characters per instruction line under a panel


def _camera(points: np.ndarray):
    """A camera centred on `points` (world xyz), as far away as their spread needs."""
    import mujoco
    lo, hi = points.min(axis=0), points.max(axis=0)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.04)
    cam.distance = float(np.clip(0.45 + 1.1 * max(hi[0] - lo[0], hi[1] - lo[1]), *DISTANCE))
    cam.elevation, cam.azimuth = CAMERA["elevation"], CAMERA["azimuth"]
    return cam


def _task_points(env) -> np.ndarray:
    """Where the task happens: the bodies it moves and the buttons it presses."""
    bodies = [env.data.xpos[env.body_ids[b]].copy() for b in env.task_spec.bodies_moved()]
    buttons = [env.data.xpos[env.button_cap[g.button]].copy() for g in env.task_spec.goals if hasattr(g, "button")]
    return np.array(bodies + buttons)


def _render(env, renderer, camera, arm_home: bool = False) -> np.ndarray:
    import mujoco
    if arm_home:
        env._set_arm_state(env.ready_q)
        mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera=camera)
    return renderer.render().copy()


def panel(number: int) -> tuple[np.ndarray, np.ndarray, str, int]:
    """(start image, completed image, instruction, number of goals) of the first episode the oracle completes."""
    import mujoco
    from afford_pro.bench import make_env, make_oracle
    env = make_env(number)
    renderer = mujoco.Renderer(env.model, *IMAGE_HW)
    try:
        for j in range(MAX_TRIES):
            seed = SEED_BLOCK + 1000 * number + j
            env.reset(seed=seed)
            before = _task_points(env)
            _, _, _, dones = make_oracle(env).generate_episode(seed=seed)
            if dones and dones[-1]:
                camera = _camera(np.vstack([before, _task_points(env)]))
                done = _render(env, renderer, camera, arm_home=True)
                env.reset(seed=seed)  # the same start again, rendered with the camera fitted to the whole task
                return _render(env, renderer, camera), done, env.task_spec.instruction, len(env.task_spec.goals)
        raise RuntimeError(f"task {number}: the oracle did not complete any of {MAX_TRIES} episodes")
    finally:
        renderer.close()
        env.close()


def compose(panels: list[tuple[str, np.ndarray, np.ndarray, str, int]], out: Path, cols: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    rows = (len(panels) + cols - 1) // cols
    fig = plt.figure(figsize=(7.16, 1.55 * rows))
    grid = fig.add_gridspec(rows, cols, wspace=0.05, hspace=0.16)
    for k, (title, start, done, instruction, goals) in enumerate(panels):
        cell = grid[k // cols, k % cols].subgridspec(1, 2, wspace=0.02)
        for c, (image, label) in enumerate(((start, "start"), (done, "completed"))):
            ax = fig.add_subplot(cell[0, c])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.95, label, transform=ax.transAxes, fontsize=5.5, va="top", color="black",
                    bbox={"facecolor": "white", "alpha": 0.8, "pad": 1.0, "edgecolor": "none"})
            if c == 0:
                ax.set_title(f"{title}: {goals} goals", fontsize=7, fontweight="bold", loc="left", pad=2)
                ax.set_xlabel("\n".join(textwrap.wrap(instruction, WRAP)), fontsize=5.8, loc="left", labelpad=2,
                              linespacing=1.1)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output path without suffix (.pdf and .png written)")
    args = parser.parse_args()
    panels = [(title, *panel(number)) for number, title in PANELS]  # rendering, so not under envs.headless
    compose(panels, args.out)
    print(f"wrote {args.out.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
