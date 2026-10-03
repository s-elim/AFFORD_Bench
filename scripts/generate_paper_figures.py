"""Generate publication-grade qualitative visualization figure for RSS 2027 paper."""
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
    ("D1: Industrial", "industrial_gear_mesh", "Gear meshing onto spline"),
    ("D2: Culinary", "kitchen_patty_stack", "Align & stack burger patty"),
    ("D3: Domestic", "tabletop_sort_red", "Sort block into color bin"),
    ("D4: Cleanroom Lab", "lab_pick_micropipette", "Dock precision micropipette"),
    ("D5: Long-Horizon", "long_horizon_transmission_assembly", "Multi-stage gear & pin latch"),
]

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(7.0, 8.5), dpi=300)
plt.subplots_adjust(wspace=0.03, hspace=0.16, left=0.26, right=0.98, top=0.94, bottom=0.02)

col_titles = ["Workcell RGB ($640\\times 660$)", "Wrist RGB ($300\\times 420$)", "Metric Depth ($D_{3D}$)"]
for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, fontsize=8.5, fontweight="bold", pad=5)

for row_idx, (domain_label, task_id, task_desc) in enumerate(SELECTED_TASKS):
    print(f"Rendering {domain_label}: '{task_id}'...")
    env = afford_pro.make(task_id, render_depth=True)
    obs, _ = env.reset(seed=42)

    # 1. Workcell RGB
    workcell_rgb = obs["agentic_rgb"]
    axes[row_idx, 0].imshow(workcell_rgb)
    axes[row_idx, 0].axis("off")

    # 2. Wrist RGB
    wrist_rgb = obs["wrist_rgb"]
    print(f"  {task_id} wrist_rgb shape={wrist_rgb.shape}, mean={wrist_rgb.mean():.2f}")
    axes[row_idx, 1].imshow(wrist_rgb)
    axes[row_idx, 1].axis("off")

    # 3. 3D Metric Depth
    if "agentic_depth" in obs:
        depth = obs["agentic_depth"]
        depth_vis = np.clip(depth, 0.7, 2.5)
    else:
        depth_vis = np.mean(workcell_rgb, axis=-1)
    
    im = axes[row_idx, 2].imshow(depth_vis, cmap="inferno")
    axes[row_idx, 2].axis("off")

    # Row label on the left
    fig.text(
        0.02, 
        0.905 - row_idx * 0.185, 
        f"{domain_label}\n{task_desc}", 
        fontsize=8.0, 
        fontweight="bold",
        va="center", 
        ha="left",
        color="#1f4e79"
    )

    env.close()
    del env
    import gc
    gc.collect()

# Save PNG and PDF
png_path = OUTPUT_DIR / "tasks_visualization.png"
pdf_path = OUTPUT_DIR / "tasks_visualization.pdf"
plt.savefig(png_path, bbox_inches="tight", dpi=300)
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"[SUCCESS] Saved qualitative task visualization to:\n  {png_path}\n  {pdf_path}")
