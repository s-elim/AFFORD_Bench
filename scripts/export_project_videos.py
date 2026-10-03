"""Export lightweight demo videos from AFFORD-Bench HDF5 datasets for the project page."""
from __future__ import annotations

from pathlib import Path
import cv2
import h5py
import imageio
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
VIDEOS_DIR = ROOT / "docs" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_TASKS = [
    {
        "domain": "D1: Industrial Precision",
        "task_id": "industrial_shaft_assembly",
        "filename": "industrial_shaft_assembly_nominal.hdf5",
        "output_name": "d1_industrial_shaft_assembly.mp4",
        "title": "Industrial: Shaft Assembly (Clearance: +/-2.5mm)",
    },
    {
        "domain": "D2: Culinary Operations",
        "task_id": "kitchen_patty_stack",
        "filename": "kitchen_patty_stack_nominal.hdf5",
        "output_name": "d2_kitchen_patty_stack.mp4",
        "title": "Culinary: Patty Stacking (Compliant Contact)",
    },
    {
        "domain": "D3: Domestic Tabletop",
        "task_id": "tabletop_sort_red",
        "filename": "tabletop_sort_red_nominal.hdf5",
        "output_name": "d3_tabletop_sort_red.mp4",
        "title": "Domestic: Block Sorting (Multi-Color Clutter)",
    },
    {
        "domain": "D4: Laboratory Sciences",
        "task_id": "lab_tube_to_centrifuge",
        "filename": "lab_tube_to_centrifuge_nominal.hdf5",
        "output_name": "d4_lab_tube_centrifuge.mp4",
        "title": "Laboratory: Centrifuge Tube Docking (+/-2.0mm)",
    },
    {
        "domain": "D5: Long-Horizon Sequences",
        "task_id": "long_horizon_transmission_assembly",
        "filename": "long_horizon_transmission_assembly_nominal.hdf5",
        "output_name": "d5_long_horizon_transmission.mp4",
        "title": "Long-Horizon: Multi-Stage Transmission Assembly",
    },
    {
        "domain": "D6: Multi-Stage Precision Chaining",
        "task_id": "chain_shaft_bearing_clamp",
        "filename": "chain_shaft_bearing_clamp_nominal.hdf5",
        "output_name": "d6_precision_chaining.mp4",
        "title": "Precision Chain: Shaft -> Bearing -> Clamp",
    },
    {
        "domain": "D7: Sequential Service & Assembly",
        "task_id": "chain_burger_flip_cheese_top",
        "filename": "chain_burger_flip_cheese_top_nominal.hdf5",
        "output_name": "d7_service_chaining.mp4",
        "title": "Service Chain: Patty Flip -> Cheese -> Top Bun",
    },
]


def render_composite_frame(workcell_rgb: np.ndarray, wrist_rgb: np.ndarray, title: str, step: int, total_steps: int) -> np.ndarray:
    """Stitch workcell camera and wrist camera side-by-side with metadata banner."""
    # Target height: 360px
    target_h = 360
    # Workcell: 640x660 -> scale to h=360, w=371
    wc_w = int(workcell_rgb.shape[1] * (target_h / workcell_rgb.shape[0]))
    wc_resized = cv2.resize(workcell_rgb, (wc_w, target_h), interpolation=cv2.INTER_AREA)

    # Wrist: 300x420 -> scale to h=360, w=257
    wrist_w = int(wrist_rgb.shape[1] * (target_h / wrist_rgb.shape[0]))
    wrist_resized = cv2.resize(wrist_rgb, (wrist_w, target_h), interpolation=cv2.INTER_AREA)

    # Combined width
    total_w = wc_w + wrist_w
    combined = np.zeros((target_h + 40, total_w, 3), dtype=np.uint8)

    # Top banner (40px)
    banner_bg = (24, 28, 36)  # Dark slate
    combined[:40, :] = banner_bg

    # Place cameras
    combined[40:, :wc_w] = wc_resized
    combined[40:, wc_w:] = wrist_resized

    # Draw vertical separator
    combined[40:, wc_w-1:wc_w+1] = (60, 70, 85)

    # Text overlays
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Title
    cv2.putText(combined, title, (12, 26), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    # Step counter
    progress_str = f"Step: {step}/{total_steps}"
    cv2.putText(combined, progress_str, (total_w - 130, 26), font, 0.45, (160, 200, 255), 1, cv2.LINE_AA)

    # Camera label tags
    cv2.putText(combined, "Workcell RGB (640x660)", (12, 58), font, 0.38, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(combined, "Wrist Eye-in-Hand", (wc_w + 12, 58), font, 0.38, (255, 200, 50), 1, cv2.LINE_AA)

    return combined


def export_task_video(task_info: dict, max_frames: int = 150):
    hdf5_path = DATASETS_DIR / task_info["filename"]
    out_path = VIDEOS_DIR / task_info["output_name"]

    if not hdf5_path.exists():
        print(f"[WARN] File {hdf5_path.name} not found. Skipping.")
        return

    print(f"Exporting video for {task_info['task_id']} -> {out_path.name}...")
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data/demo_0"]
        wc_all = demo["obs/agentic_rgb"][:]
        wrist_all = demo["obs/wrist_rgb"][:]
        total_steps = len(wc_all)

        # Subsample if trajectory is long to keep video ~6-8 seconds @ 20fps (~140 frames)
        step_stride = max(1, total_steps // max_frames)
        indices = list(range(0, total_steps, step_stride))

        frames = []
        for idx in indices:
            wc = wc_all[idx]
            wrist = wrist_all[idx]
            comp = render_composite_frame(wc, wrist, task_info["title"], idx + 1, total_steps)
            frames.append(comp)

        # Write mp4
        imageio.mimsave(str(out_path), frames, fps=20, quality=7)
        size_kb = out_path.stat().st_size / 1024
        print(f"  -> Generated {out_path.name} ({len(frames)} frames, {size_kb:.1f} KB)")


def main():
    print("=" * 60)
    print(" AFFORD-Bench: Project Page Video Exporter")
    print(f" Output Directory: {VIDEOS_DIR}")
    print("=" * 60)

    for item in REPRESENTATIVE_TASKS:
        export_task_video(item)

    print("\nAll videos exported successfully.")


if __name__ == "__main__":
    main()
