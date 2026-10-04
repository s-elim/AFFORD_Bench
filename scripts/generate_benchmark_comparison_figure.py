"""Generate a high-impact, publication-grade benchmark comparison figure:
LIBERO-PRO vs. LIBERO-X vs. AFFORD-Bench (Ours).
Designed with professional vector graphics, clean badge layout, and zero text overlap.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set publication typography & aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": False,
    "mathtext.fontset": "cm",
})

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR_PAPER = ROOT / "AFFORD-PRO_RSS'27" / "paper-template-latex" / "figures"
OUTPUT_DIR_DOCS = ROOT / "docs" / "images"
OUTPUT_DIR_PAPER.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_DOCS.mkdir(parents=True, exist_ok=True)


def create_comparison_figure():
    # Wide 16:5.6 ratio for maximum breathing room
    fig = plt.figure(figsize=(16.8, 5.8), dpi=300, facecolor="#ffffff")
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.1, 0.95, 1.45],
        wspace=0.28,
        left=0.04,
        right=0.98,
        top=0.86,
        bottom=0.10
    )

    # ====================================================
    # Panel (a): Multi-Dimensional Capability Radar Chart
    # ====================================================
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    ax_radar.set_facecolor("#f8fafc")

    categories = [
        "Physical Affordance\nGrounding",
        "Domain & Scene\nDiversity",
        "Perceptual Modality\n(RGB + Depth)",
        "Policy-Agnostic\nBreadth",
        "Hierarchical OOD\nProtocol",
        "Multi-Subtask\nChaining",
    ]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Scores (0.0 to 1.0)
    scores_libero_pro = [0.20, 0.40, 0.25, 0.40, 0.65, 0.25]
    scores_libero_pro += scores_libero_pro[:1]

    scores_libero_x = [0.35, 0.55, 0.50, 0.45, 0.85, 0.60]
    scores_libero_x += scores_libero_x[:1]

    scores_ours = [0.95, 0.96, 0.95, 0.98, 0.95, 0.94]
    scores_ours += scores_ours[:1]

    # Plot radar lines & polygons
    ax_radar.plot(angles, scores_libero_pro, color="#64748b", linewidth=1.8, linestyle="--", label="LIBERO-PRO (2026)")
    ax_radar.fill(angles, scores_libero_pro, color="#64748b", alpha=0.12)

    ax_radar.plot(angles, scores_libero_x, color="#d97706", linewidth=2.0, linestyle="-.", label="LIBERO-X (2026)")
    ax_radar.fill(angles, scores_libero_x, color="#f59e0b", alpha=0.18)

    ax_radar.plot(angles, scores_ours, color="#1d4ed8", linewidth=2.6, linestyle="-", marker="o", markersize=5, label="AFFORD-Bench (Ours)")
    ax_radar.fill(angles, scores_ours, color="#3b82f6", alpha=0.28)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, size=8.5, weight="bold", color="#1e293b")
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7.5, color="#64748b")
    ax_radar.set_ylim(0, 1.05)
    ax_radar.grid(color="#cbd5e1", linestyle=":", linewidth=0.8)
    ax_radar.set_title("(a) Multi-Dimensional Capability Radar", size=11, weight="bold", pad=22, color="#0f172a")
    ax_radar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.2)

    # ====================================================
    # Panel (b): Quantitative Scale & Benchmark Statistics
    # ====================================================
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor("#ffffff")

    benchmarks = ["LIBERO-PRO", "LIBERO-X", "AFFORD-Bench\n(Ours)"]
    x = np.arange(len(benchmarks))
    bar_width = 0.26

    domains_data = [4, 5, 7]
    suites_data = [20, 15, 35]
    steps_m_data = [1.3, 0.5, 2.3]  # in Millions

    ax2_steps = ax_bar.twinx()

    rects1 = ax_bar.bar(x - bar_width, domains_data, bar_width, label="Domains Scope", color="#94a3b8", edgecolor="#475569", linewidth=0.9)
    rects2 = ax_bar.bar(x, suites_data, bar_width, label="Evaluation Suites", color="#f59e0b", edgecolor="#b45309", linewidth=0.9)
    rects3 = ax2_steps.bar(x + bar_width, steps_m_data, bar_width, label="Demo Steps (M)", color="#2563eb", edgecolor="#1e3a8a", linewidth=0.9)

    ax_bar.set_ylabel("Count (Domains & Suites)", size=9.5, weight="bold", color="#334155")
    ax2_steps.set_ylabel("Physical Timesteps (Millions)", size=9.5, weight="bold", color="#1e40af")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(benchmarks, size=9, weight="bold", color="#0f172a")

    ax_bar.set_ylim(0, 42)
    ax2_steps.set_ylim(0, 2.85)

    # Value annotations on bars
    for rect in rects1:
        h = rect.get_height()
        ax_bar.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#334155")

    for rect in rects2:
        h = rect.get_height()
        ax_bar.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#92400e")

    for rect in rects3:
        h = rect.get_height()
        ax2_steps.annotate(f"{h:.1f}M", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                           textcoords="offset points", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#1e3a8a")

    ax_bar.grid(axis="y", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax_bar.set_title("(b) Quantitative Scale Comparison", size=11, weight="bold", pad=12, color="#0f172a")

    lines1, labels1 = ax_bar.get_legend_handles_labels()
    lines2, labels2 = ax2_steps.get_legend_handles_labels()
    ax_bar.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    # ====================================================
    # Panel (c): Structural & Physical Feature Matrix (Clean Card Layout)
    # ====================================================
    ax_matrix = fig.add_subplot(gs[0, 2])
    ax_matrix.axis("off")
    ax_matrix.set_xlim(0, 100)
    ax_matrix.set_ylim(0, 100)
    ax_matrix.set_title("(c) Architectural & Physical Grounding Comparison", size=11, weight="bold", pad=12, color="#0f172a")

    # Header Row
    ax_matrix.text(2, 92, "Benchmark Feature", fontsize=9.2, weight="bold", color="#1e293b", va="center")
    ax_matrix.text(42, 92, "LIBERO-PRO", fontsize=8.5, weight="bold", color="#64748b", ha="center", va="center")
    ax_matrix.text(63, 92, "LIBERO-X", fontsize=8.5, weight="bold", color="#d97706", ha="center", va="center")
    
    # Highlight badge for Ours header
    ours_header_bg = FancyBboxPatch((75, 86), 24.5, 12, boxstyle="round,pad=0.5,rounding_size=2",
                                    facecolor="#1d4ed8", edgecolor="none")
    ax_matrix.add_patch(ours_header_bg)
    ax_matrix.text(87.2, 92, "AFFORD-Bench (Ours)", fontsize=8.5, weight="bold", color="#ffffff", ha="center", va="center")

    features = [
        {
            "name": "Friction & Payload",
            "p1": "Nominal Only", "p1_type": "no",
            "p2": "Nominal Only", "p2_type": "no",
            "ours": "μ ∈ [0.1, 1.2], 0.5-3× m", "ours_type": "yes"
        },
        {
            "name": "Insertion Clearance",
            "p1": "Coarse Tabletop", "p1_type": "no",
            "p2": "Coarse Tabletop", "p2_type": "no",
            "ours": "Sub-cm (±1.5–5 mm)", "ours_type": "yes"
        },
        {
            "name": "Sensory Streams",
            "p1": "1× RGB (128x128)", "p1_type": "no",
            "p2": "2× RGB (256x256)", "p2_type": "warn",
            "ours": "Dual RGB + 3D Depth", "ours_type": "yes"
        },
        {
            "name": "Compound Prompts",
            "p1": "Atomic Single-Step", "p1_type": "no",
            "p2": "Subtask Tags", "p2_type": "warn",
            "ours": "Chained 2–3 Step Continuous", "ours_type": "yes"
        },
        {
            "name": "Policy Agnosticism",
            "p1": "VLA Models", "p1_type": "no",
            "p2": "VLA Models", "p2_type": "no",
            "ours": "BC, DP, ACT, 3D, VLA, RL", "ours_type": "yes"
        },
        {
            "name": "Demonstration Data",
            "p1": "6,500 Demos", "p1_type": "no",
            "p2": "2,520 Demos", "p2_type": "no",
            "ours": "10,500 Demos (2.3M Steps)", "ours_type": "yes"
        },
        {
            "name": "Simulation Engine",
            "p1": "Robosuite (~15 FPS)", "p1_type": "no",
            "p2": "Robosuite (~15 FPS)", "p2_type": "no",
            "ours": "Native MuJoCo (45 FPS, $0)", "ours_type": "yes"
        },
    ]

    y_start = 78
    y_step = 11.5

    for idx, f in enumerate(features):
        y = y_start - (idx * y_step)

        # Alternating row background
        row_bg_color = "#f8fafc" if idx % 2 == 1 else "#ffffff"
        row_patch = FancyBboxPatch((0, y - 4.5), 100, 9.2, boxstyle="round,pad=0.2,rounding_size=1.5",
                                   facecolor=row_bg_color, edgecolor="#e2e8f0", linewidth=0.6)
        ax_matrix.add_patch(row_patch)

        # Feature label
        ax_matrix.text(2, y, f["name"], fontsize=8.2, weight="semibold", color="#0f172a", va="center")

        # LIBERO-PRO badge
        ax_matrix.text(42, y, f["p1"], fontsize=7.4, color="#475569", ha="center", va="center")

        # LIBERO-X badge
        color_p2 = "#9a3412" if f["p2_type"] == "warn" else "#475569"
        ax_matrix.text(63, y, f["p2"], fontsize=7.4, color=color_p2, weight="semibold" if f["p2_type"] == "warn" else "normal", ha="center", va="center")

        # Ours badge with colored card pill
        ours_pill = FancyBboxPatch((75.0, y - 3.8), 24.5, 7.6, boxstyle="round,pad=0.3,rounding_size=1.5",
                                   facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=0.7)
        ax_matrix.add_patch(ours_pill)
        ax_matrix.text(87.2, y, f["ours"], fontsize=7.2, weight="bold", color="#1d4ed8", ha="center", va="center")

    # Overall Figure Suptitle
    fig.suptitle("AFFORD-Bench vs. LIBERO-PRO & LIBERO-X: Architecture, Scale, and Physical Grounding",
                 fontsize=13.5, weight="heavy", y=0.98, color="#0f172a")

    # Save outputs
    pdf_path_paper = OUTPUT_DIR_PAPER / "benchmark_comparison.pdf"
    png_path_paper = OUTPUT_DIR_PAPER / "benchmark_comparison.png"
    png_path_docs = OUTPUT_DIR_DOCS / "benchmark_comparison.png"

    plt.savefig(pdf_path_paper, format="pdf", bbox_inches="tight")
    plt.savefig(png_path_paper, format="png", dpi=300, bbox_inches="tight")
    plt.savefig(png_path_docs, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated comparison figure successfully:")
    print(f"  -> {pdf_path_paper} ({pdf_path_paper.stat().st_size / 1024:.1f} KB)")
    print(f"  -> {png_path_paper} ({png_path_paper.stat().st_size / 1024:.1f} KB)")
    print(f"  -> {png_path_docs} ({png_path_docs.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    create_comparison_figure()
