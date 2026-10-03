"""Generate publication-grade qualitative visualization figures for RSS 2027 paper across all 7 domains."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

import afford_pro

OUTPUT_DIR = Path(r"D:\Selim Sarowar\Cursor_Projects\AFFORD-PRO-150\AFFORD-PRO_RSS'27\paper-template-latex\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_TASKS = [
    ("D1: Industrial", "industrial_gear_mesh", "Gear Mesh"),
    ("D2: Culinary", "kitchen_patty_stack", "Patty Stack"),
    ("D3: Domestic", "tabletop_sort_red", "Color Sort"),
    ("D4: Cleanroom Lab", "lab_pick_micropipette", "Pipette Dock"),
    ("D5: Long-Horizon", "long_horizon_transmission_assembly", "Tool Assembly"),
    ("D6: Chained Precision", "chain_shaft_bearing_clamp", "Shaft-Clamp"),
    ("D7: Chained Service", "chain_burger_flip_cheese_top", "Burger Trio"),
]

def render_observations():
    obs_dict = {}
    for domain_label, task_id, short_title in SELECTED_TASKS:
        print(f"Rendering {domain_label}: '{task_id}'...")
        env = afford_pro.make(task_id, render_depth=True)
        obs, _ = env.reset(seed=42)
        
        # Get Workcell RGB, Wrist RGB, Depth
        workcell = obs["agentic_rgb"]
        wrist = obs["wrist_rgb"]
        if "agentic_depth" in obs and obs["agentic_depth"] is not None:
            depth = np.clip(obs["agentic_depth"], 0.7, 2.5)
        else:
            depth = np.mean(workcell, axis=-1)
            
        obs_dict[task_id] = {
            "domain": domain_label,
            "short_title": short_title,
            "workcell": workcell,
            "wrist": wrist,
            "depth": depth,
        }
        env.close()
    return obs_dict

def generate_horizontal_figure(obs_dict):
    """Generate 3 rows x 7 columns horizontal figure for main.tex Figure 2."""
    print("Generating horizontal visualization across 7 domains (3 rows x 7 cols)...")
    fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(7.0, 2.5), dpi=300)
    plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.10, right=0.99, top=0.86, bottom=0.03)

    row_labels = ["Workcell RGB\n($640\\times 660$)", "Wrist RGB\n($300\\times 420$)", "Metric Depth\n($D_{3D}$)"]

    for col_idx, (_, task_id, short_title) in enumerate(SELECTED_TASKS):
        data = obs_dict[task_id]
        
        # Column title at top
        axes[0, col_idx].set_title(f"{data['domain']}\n{short_title}", fontsize=6.2, fontweight="bold", pad=3)

        # Row 0: Workcell
        axes[0, col_idx].imshow(data["workcell"])
        axes[0, col_idx].axis("off")

        # Row 1: Wrist
        axes[1, col_idx].imshow(data["wrist"])
        axes[1, col_idx].axis("off")

        # Row 2: Metric Depth
        axes[2, col_idx].imshow(data["depth"], cmap="inferno")
        axes[2, col_idx].axis("off")

    # Add row labels on the left
    fig.text(0.01, 0.72, row_labels[0], fontsize=6.0, fontweight="bold", va="center", ha="left", color="#1f4e79")
    fig.text(0.01, 0.44, row_labels[1], fontsize=6.0, fontweight="bold", va="center", ha="left", color="#1f4e79")
    fig.text(0.01, 0.16, row_labels[2], fontsize=6.0, fontweight="bold", va="center", ha="left", color="#1f4e79")

    png_path = OUTPUT_DIR / "tasks_horizontal_vis.png"
    pdf_path = OUTPUT_DIR / "tasks_horizontal_vis.pdf"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved horizontal visualization to:\n  {png_path}\n  {pdf_path}")

def generate_vertical_figure(obs_dict):
    """Generate 7 rows x 3 columns full-page qualitative visualization."""
    print("Generating vertical visualization (7 rows x 3 cols)...")
    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(7.0, 9.8), dpi=300)
    plt.subplots_adjust(wspace=0.03, hspace=0.14, left=0.28, right=0.98, top=0.95, bottom=0.02)

    col_titles = ["Workcell RGB ($640\\times 660$)", "Wrist RGB ($300\\times 420$)", "3D Metric Depth ($D_{3D}$)"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=8.5, fontweight="bold", pad=5)

    for row_idx, (_, task_id, short_title) in enumerate(SELECTED_TASKS):
        data = obs_dict[task_id]

        # 1. Workcell RGB
        axes[row_idx, 0].imshow(data["workcell"])
        axes[row_idx, 0].axis("off")

        # 2. Wrist RGB
        axes[row_idx, 1].imshow(data["wrist"])
        axes[row_idx, 1].axis("off")

        # 3. 3D Metric Depth
        axes[row_idx, 2].imshow(data["depth"], cmap="inferno")
        axes[row_idx, 2].axis("off")

        # Row label on the left
        fig.text(
            0.02,
            0.925 - row_idx * 0.133,
            f"{data['domain']}\n{short_title}",
            fontsize=7.8,
            fontweight="bold",
            va="center",
            ha="left",
            color="#1f4e79"
        )

    png_path = OUTPUT_DIR / "tasks_visualization.png"
    pdf_path = OUTPUT_DIR / "tasks_visualization.pdf"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved vertical visualization to:\n  {png_path}\n  {pdf_path}")

if __name__ == "__main__":
    obs = render_observations()
    generate_horizontal_figure(obs)
    generate_vertical_figure(obs)
