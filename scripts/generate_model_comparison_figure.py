"""Generate publication-grade model performance comparison figures for RSS 2027 paper and docs:
Evaluating AtomVLA, pi0.5 / pi0, SpatialVLA, CoT-VLA, OpenVLA, SmolVLA, Diffusion Policy, and ACT
on the AFFORD-Bench benchmark suite.

Outputs:
  - figures/model_performance_comparison.pdf
  - figures/model_performance_comparison.png
  - docs/images/model_performance_comparison.png
"""
from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR_PAPER = ROOT / "AFFORD-Bench_RSS'27" / "paper-template-latex" / "figures"
OUTPUT_DIR_DOCS = ROOT / "docs" / "images"
CSV_PATH = ROOT / "docs" / "reviews" / "vla_model_benchmark_comparison_2026-10-08.csv"

OUTPUT_DIR_PAPER.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_DOCS.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "cm",
    "figure.autolayout": False,
})


def load_benchmark_data() -> dict[str, dict[str, float]]:
    data = {}
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark data CSV: {CSV_PATH}")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["model"]
            data[name] = {
                "l1_nominal": float(row["l1_nominal_sr"]),
                "l2_spatial": float(row["l2_spatial_sr"]),
                "l3_visual": float(row["l3_visual_sr"]),
                "l4_physical": float(row["l4_physical_sr"]),
                "l5_composite": float(row["l5_composite_sr"]),
                "precision_2mm": float(row["precision_2mm_sr"]),
                "loose_8mm": float(row["loose_8mm_sr"]),
                "rrr": float(row["rrr_reactive_ratio"]),
                "gdi": float(row["gdi_degradation_pct"]),
                "pri": float(row["pri_physical_pct"]),
            }
    return data


def generate_figure():
    data = load_benchmark_data()

    # Models to highlight in radar comparison
    models = ["pi0.5 / pi0", "SpatialVLA", "AtomVLA", "CoT-VLA", "OpenVLA", "Oracle (Closed-Loop)"]
    colors = {
        "Oracle (Closed-Loop)": "#059669",  # Emerald
        "pi0.5 / pi0": "#2563EB",           # Royal Blue
        "SpatialVLA": "#4F46E5",            # Indigo
        "AtomVLA": "#0D9488",               # Teal
        "CoT-VLA": "#D97706",               # Amber
        "OpenVLA": "#E11D48",               # Rose
    }
    linestyles = {
        "Oracle (Closed-Loop)": "--",
        "pi0.5 / pi0": "-",
        "SpatialVLA": "-",
        "AtomVLA": "-",
        "CoT-VLA": "-",
        "OpenVLA": ":",
    }
    linewidths = {
        "Oracle (Closed-Loop)": 1.8,
        "pi0.5 / pi0": 2.2,
        "SpatialVLA": 2.0,
        "AtomVLA": 2.0,
        "CoT-VLA": 1.6,
        "OpenVLA": 1.6,
    }

    # Radar axes
    categories = [
        "Nominal (L1)",
        "Spatial (L2)",
        "Visual (L3)",
        "Physical (L4)",
        "2mm Precision",
        "Reactive (RRR)",
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig = plt.figure(figsize=(15.2, 5.8), dpi=300, facecolor="#FFFFFF")

    # =========================================================================
    # Left Subplot: Polar Radar Chart
    # =========================================================================
    ax_radar = fig.add_axes([0.05, 0.12, 0.40, 0.76], polar=True)
    ax_radar.set_facecolor("#FAFAFA")
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)

    # Draw axis lines & labels
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=9.2, weight="bold", color="#1E293B")
    ax_radar.set_rlabel_position(30)
    ax_radar.set_yticks([20, 40, 60, 80, 100])
    ax_radar.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7.2, color="#94A3B8")
    ax_radar.set_ylim(0, 105)
    ax_radar.grid(color="#E2E8F0", linestyle="--", linewidth=0.8)

    for m in models:
        d = data[m]
        vals = [
            d["l1_nominal"],
            d["l2_spatial"],
            d["l3_visual"],
            d["l4_physical"],
            d["precision_2mm"],
            d["rrr"],
        ]
        vals += vals[:1]
        ax_radar.plot(
            angles, vals,
            color=colors[m],
            linewidth=linewidths[m],
            linestyle=linestyles[m],
            label=m,
            zorder=4 if m != "Oracle (Closed-Loop)" else 2,
        )
        if m in ("pi0.5 / pi0", "SpatialVLA", "AtomVLA"):
            ax_radar.fill(angles, vals, color=colors[m], alpha=0.08)

    ax_radar.set_title("VLA Foundation Model Affordance Radar Profile", fontsize=11.5, weight="heavy", color="#0F172A", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # =========================================================================
    # Right Subplot 1: 2mm CAD Clearances vs 8mm Loose Clearances
    # =========================================================================
    ax_bar1 = fig.add_axes([0.53, 0.55, 0.43, 0.33])
    ax_bar1.set_facecolor("#FFFFFF")
    ax_bar1.set_axisbelow(True)
    ax_bar1.yaxis.grid(True, color="#F1F5F9", linestyle="-", linewidth=0.8)

    compare_models = ["Oracle", "pi0.5", "SpatialVLA", "AtomVLA", "CoT-VLA", "DiffPol", "ACT", "OpenVLA", "SmolVLA"]
    raw_keys = [
        "Oracle (Closed-Loop)", "pi0.5 / pi0", "SpatialVLA", "AtomVLA", "CoT-VLA",
        "Diffusion Policy", "ACT", "OpenVLA", "SmolVLA"
    ]
    p2mm = [data[k]["precision_2mm"] for k in raw_keys]
    loose = [data[k]["loose_8mm"] for k in raw_keys]

    x = np.arange(len(compare_models))
    width = 0.36

    rects1 = ax_bar1.bar(x - width/2, p2mm, width, label="2.0–2.5 mm CAD Clearance", color="#2563EB", edgecolor="none", zorder=3)
    rects2 = ax_bar1.bar(x + width/2, loose, width, label="8.0 mm Loose Clearance", color="#93C5FD", edgecolor="none", zorder=3)

    ax_bar1.set_ylabel("Success Rate (%)", fontsize=8.6, weight="bold", color="#334155")
    ax_bar1.set_title("Sub-Millimeter Assembly Bottleneck: CAD Precision (2mm) vs Loose (8mm)", fontsize=10.2, weight="bold", color="#0F172A", pad=8)
    ax_bar1.set_xticks(x)
    ax_bar1.set_xticklabels(compare_models, fontsize=8.0, color="#1E293B", weight="bold")
    ax_bar1.set_ylim(0, 110)
    ax_bar1.legend(loc="upper right", fontsize=7.6, frameon=False)
    ax_bar1.spines["top"].set_visible(False)
    ax_bar1.spines["right"].set_visible(False)
    ax_bar1.spines["left"].set_color("#CBD5E1")
    ax_bar1.spines["bottom"].set_color("#CBD5E1")

    # Annotate OpenVLA discretization drop
    ax_bar1.annotate("1mm Bin Floor\n(-26.4% Drop)",
                     xy=(7 - width/2, p2mm[7]), xytext=(7 - 0.4, p2mm[7] + 18),
                     arrowprops=dict(facecolor="#E11D48", arrowstyle="->", lw=1.2),
                     fontsize=6.8, weight="bold", color="#BE123C", ha="center")

    # =========================================================================
    # Right Subplot 2: Physical Generalization Collapse (L1 Nominal vs L4 Physics)
    # =========================================================================
    ax_bar2 = fig.add_axes([0.53, 0.12, 0.43, 0.33])
    ax_bar2.set_facecolor("#FFFFFF")
    ax_bar2.set_axisbelow(True)
    ax_bar2.yaxis.grid(True, color="#F1F5F9", linestyle="-", linewidth=0.8)

    l1_vals = [data[k]["l1_nominal"] for k in raw_keys]
    l4_vals = [data[k]["l4_physical"] for k in raw_keys]

    rects3 = ax_bar2.bar(x - width/2, l1_vals, width, label="L1 Nominal (μ=0.5, 1.0x Mass)", color="#059669", edgecolor="none", zorder=3)
    rects4 = ax_bar2.bar(x + width/2, l4_vals, width, label="L4 Physical Shift (μ∈[0.12,0.25], 2.5x Mass)", color="#F87171", edgecolor="none", zorder=3)

    ax_bar2.set_ylabel("Success Rate (%)", fontsize=8.6, weight="bold", color="#334155")
    ax_bar2.set_title("Physical Affordance Generalization Collapse: Nominal vs. Low-Friction & Heavy Payload", fontsize=10.2, weight="bold", color="#0F172A", pad=8)
    ax_bar2.set_xticks(x)
    ax_bar2.set_xticklabels(compare_models, fontsize=8.0, color="#1E293B", weight="bold")
    ax_bar2.set_ylim(0, 110)
    ax_bar2.legend(loc="upper right", fontsize=7.6, frameon=False)
    ax_bar2.spines["top"].set_visible(False)
    ax_bar2.spines["right"].set_visible(False)
    ax_bar2.spines["left"].set_color("#CBD5E1")
    ax_bar2.spines["bottom"].set_color("#CBD5E1")

    # Annotate universal L4 physical drop
    ax_bar2.text(2.5, 88, "Universal ~55–70% Collapse Under Coulomb Friction Shift",
                 fontsize=7.8, weight="bold", color="#B91C1C", ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.8))

    # Save outputs
    pdf_path = OUTPUT_DIR_PAPER / "model_performance_comparison.pdf"
    png_path_paper = OUTPUT_DIR_PAPER / "model_performance_comparison.png"
    png_path_docs = OUTPUT_DIR_DOCS / "model_performance_comparison.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path_paper, format="png", bbox_inches="tight", dpi=300)
    fig.savefig(png_path_docs, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Generated Model Performance Comparison Figure:")
    print(f"  -> PDF: {pdf_path}")
    print(f"  -> PNG (Paper): {png_path_paper}")
    print(f"  -> PNG (Docs): {png_path_docs}")


if __name__ == "__main__":
    generate_figure()
