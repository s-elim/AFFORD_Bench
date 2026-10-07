"""Generate an ultra-clean, appealing, publication-grade benchmark comparison figure:
AFFORD-Bench vs. LIBERO-PRO & LIBERO-X.

Design Philosophy:
- High visual appeal, modern conference aesthetic (DeepMind/RSS style)
- Zero clutter, zero dense text: pure visual contrast across core differences
- Snug, balanced aspect ratio (15.8 x 5.7) with zero wasted whitespace
- 3 distinct benchmark cards with solid header bands and 'Ours' highlighted in premium royal blue & emerald
- 6 horizontal feature tracks for effortless 5-second scanning
- Visual status indicators (checkmarks, badges, progress bars, task counts)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Publication typography & styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "cm",
    "figure.autolayout": False,
})

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR_PAPER = ROOT / "AFFORD-Bench_RSS'27" / "paper-template-latex" / "figures"
OUTPUT_DIR_DOCS = ROOT / "docs" / "images"
OUTPUT_DIR_PAPER.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_DOCS.mkdir(parents=True, exist_ok=True)


def draw_card(ax, x, y, w, h, bg_color, border_color, border_width=1.2, corner_radius=0.016, zorder=1):
    """Draw a modern rounded card container."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=border_width,
        zorder=zorder
    )
    ax.add_patch(box)
    return box


def draw_pill(ax, x, y, w, h, text, bg_color, text_color, fontsize=8.8, weight="bold", border_color="none", zorder=3):
    """Draw a stylish rounded badge / pill with centered text."""
    pill = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.009",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.2 if border_color != "none" else 0,
        zorder=zorder
    )
    ax.add_patch(pill)
    ax.text(
        x + w / 2.0, y + h / 2.0,
        text,
        color=text_color,
        fontsize=fontsize,
        weight=weight,
        ha="center",
        va="center",
        zorder=zorder + 1
    )


def generate_appealing_figure():
    # 15.8 x 5.7 aspect ratio - compact, snug, zero wasted space
    fig = plt.figure(figsize=(15.8, 5.7), dpi=300, facecolor="#FFFFFF")

    ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # =========================================================================
    # 1. TOP HEADER BANNER
    # =========================================================================
    ax.text(
        0.038, 0.942,
        "Benchmark Comparison: AFFORD-Bench vs. LIBERO-PRO & LIBERO-X",
        fontsize=15.5,
        weight="heavy",
        color="#0F172A",
        va="center"
    )
    ax.text(
        0.038, 0.902,
        "Key Differences in Physical Affordances, Perceptual Modalities, Policy Universality, and Scale",
        fontsize=9.2,
        weight="normal",
        color="#64748B",
        va="center"
    )
    draw_pill(ax, 0.835, 0.906, 0.128, 0.046, "RSS 2027 BENCHMARK", "#EFF6FF", "#1D4ED8", fontsize=7.8, weight="bold", border_color="#BFDBFE")

    # Subtle divider
    ax.plot([0.038, 0.962], [0.868, 0.868], color="#E2E8F0", linewidth=1.0, zorder=2)

    # =========================================================================
    # 2. COLUMN GEOMETRY (Snug Vertical Fit)
    # =========================================================================
    c0_x, c0_w = 0.038, 0.188
    c1_x, c1_w = 0.240, 0.228
    c2_x, c2_w = 0.482, 0.228
    c3_x, c3_w = 0.724, 0.238

    cards_top = 0.840
    cards_bottom = 0.032
    cards_h = cards_top - cards_bottom

    # Card 0 (Dimension Labels Card)
    draw_card(ax, c0_x, cards_bottom, c0_w, cards_h, "#F8FAFC", "#E2E8F0", border_width=1.0, corner_radius=0.014)

    # Card 1 (LIBERO-PRO)
    draw_card(ax, c1_x, cards_bottom, c1_w, cards_h, "#FFFFFF", "#E2E8F0", border_width=1.2, corner_radius=0.016)

    # Card 2 (LIBERO-X)
    draw_card(ax, c2_x, cards_bottom, c2_w, cards_h, "#FFFFFF", "#FED7AA", border_width=1.2, corner_radius=0.016)

    # Card 3 (AFFORD-Bench - Highlighted with royal blue glow & emerald accents)
    draw_card(ax, c3_x - 0.003, cards_bottom - 0.003, c3_w + 0.006, cards_h + 0.006, "#EFF6FF", "#2563EB", border_width=2.2, corner_radius=0.018)
    draw_card(ax, c3_x, cards_bottom, c3_w, cards_h, "#FFFFFF", "#BFDBFE", border_width=1.0, corner_radius=0.016)

    # =========================================================================
    # 3. COLUMN HEADERS (Crisp Top Banner Strips)
    # =========================================================================
    banner_h = 0.088
    banner_y = cards_top - banner_h

    # Card 0 Header
    draw_card(ax, c0_x, banner_y, c0_w, banner_h, "#F1F5F9", "#CBD5E1", border_width=1.0, corner_radius=0.014)
    ax.text(c0_x + c0_w / 2, banner_y + banner_h * 0.64, "BENCHMARK DIMENSION", fontsize=9.2, weight="bold", color="#334155", ha="center")
    ax.text(c0_x + c0_w / 2, banner_y + banner_h * 0.28, "Core Evaluation Axes", fontsize=7.6, color="#64748B", ha="center")

    # Card 1 Header: LIBERO-PRO
    draw_card(ax, c1_x, banner_y, c1_w, banner_h, "#F8FAFC", "#CBD5E1", border_width=1.0, corner_radius=0.014)
    draw_pill(ax, c1_x + 0.020, banner_y + 0.040, c1_w - 0.040, 0.036, "LIBERO-PRO", "#E2E8F0", "#1E293B", fontsize=10.0, weight="heavy")
    ax.text(c1_x + c1_w / 2, banner_y + 0.018, "Mellis et al., 2024 • Prior Baseline", fontsize=7.6, color="#64748B", ha="center")

    # Card 2 Header: LIBERO-X
    draw_card(ax, c2_x, banner_y, c2_w, banner_h, "#FFFBEB", "#FDE68A", border_width=1.0, corner_radius=0.014)
    draw_pill(ax, c2_x + 0.020, banner_y + 0.040, c2_w - 0.040, 0.036, "LIBERO-X", "#FEF3C7", "#92400E", fontsize=10.0, weight="heavy", border_color="#FCD34D")
    ax.text(c2_x + c2_w / 2, banner_y + 0.018, "Kim et al., 2025 • Language-X", fontsize=7.6, color="#B45309", ha="center")

    # Card 3 Header: AFFORD-Bench (Ours) - Prominent Banner
    draw_card(ax, c3_x, banner_y, c3_w, banner_h, "#EFF6FF", "#93C5FD", border_width=1.0, corner_radius=0.014)
    draw_pill(ax, c3_x + 0.015, banner_y + 0.040, c3_w - 0.030, 0.038, "★ AFFORD-Bench (Ours)", "#1D4ED8", "#FFFFFF", fontsize=10.4, weight="heavy")
    ax.text(c3_x + c3_w / 2, banner_y + 0.018, "Physics-Grounded Universal Benchmark", fontsize=7.8, weight="bold", color="#1E40AF", ha="center")

    # =========================================================================
    # 4. FEATURE ROWS (6 Core Differentiator Dimensions)
    # =========================================================================
    rows = [
        {
            "cat": "Contact Physics",
            "desc": "Friction & mass dynamics",
            "pro": ("Nominal (Fixed μ = 0.5)", "#F8FAFC", "#64748B", "#E2E8F0"),
            "x":   ("Nominal (Fixed μ = 0.5)", "#F8FAFC", "#64748B", "#E2E8F0"),
            "ours": ("Dynamic μ [0.1–1.8] & Mass", "#ECFDF5", "#065F46", "#A7F3D0")
        },
        {
            "cat": "Perception Modalities",
            "desc": "Visual & geometric inputs",
            "pro": ("1× RGB (128 × 128)", "#F8FAFC", "#64748B", "#E2E8F0"),
            "x":   ("2× RGB (256 × 256)", "#F8FAFC", "#475569", "#E2E8F0"),
            "ours": ("Dual RGB (256²) + 3D Depth", "#ECFDF5", "#065F46", "#A7F3D0")
        },
        {
            "cat": "Policy Universality",
            "desc": "Policy architecture support",
            "pro": ("VLA Models Only", "#F8FAFC", "#64748B", "#E2E8F0"),
            "x":   ("VLA Models Only", "#F8FAFC", "#64748B", "#E2E8F0"),
            "ours": ("Universal: DP, ACT, Flow, VLA", "#EFF6FF", "#1D4ED8", "#BFDBFE")
        },
        {
            "cat": "Domain Scope",
            "desc": "Environment breadth",
            "pro": ("4 Domestic Suites", "#F8FAFC", "#64748B", "#E2E8F0"),
            "x":   ("5 Domestic Suites", "#F8FAFC", "#475569", "#E2E8F0"),
            "ours": ("7 Diverse Domains", "#EFF6FF", "#1D4ED8", "#BFDBFE")
        },
        {
            "cat": "Task Complexity",
            "desc": "Horizon & subgoals",
            "pro": ("Single-Step Pick & Place", "#F8FAFC", "#64748B", "#E2E8F0"),
            "x":   ("Subtask Tagging", "#FFFBEB", "#92400E", "#FDE68A"),
            "ours": ("Chained 2–3 Step Subgoals", "#FAF5FF", "#6D28D9", "#DDD6FE")
        },
        {
            "cat": "Demonstration Scale",
            "desc": "Total steps & trajectories",
            "pro": ("1.3M Steps (6.5k Demos)", "#F8FAFC", "#64748B", "#E2E8F0"),
            "x":   ("0.5M Steps (2.5k Demos)", "#F8FAFC", "#64748B", "#E2E8F0"),
            "ours": ("2.3M Steps (10.5k Demos)", "#ECFDF5", "#065F46", "#A7F3D0")
        }
    ]

    row_start_y = banner_y - 0.076
    row_gap = 0.088
    pill_h = 0.058

    for i, r in enumerate(rows):
        y = row_start_y - i * row_gap

        # Subtle row guide background
        if i % 2 == 1:
            ax.add_patch(Rectangle(
                (c0_x + 0.005, y - 0.014),
                c0_w - 0.010,
                pill_h + 0.028,
                facecolor="#F1F5F9",
                edgecolor="none",
                zorder=2
            ))

        # Col 0: Category + Description
        ax.text(c0_x + 0.014, y + pill_h * 0.65, r["cat"], fontsize=9.2, weight="bold", color="#1E293B", va="center", zorder=3)
        ax.text(c0_x + 0.014, y + pill_h * 0.22, r["desc"], fontsize=7.4, color="#64748B", va="center", zorder=3)

        # Col 1: LIBERO-PRO Cell
        draw_pill(ax, c1_x + 0.012, y, c1_w - 0.024, pill_h, r["pro"][0], r["pro"][1], r["pro"][2], fontsize=8.6, weight="normal", border_color=r["pro"][3])

        # Col 2: LIBERO-X Cell
        draw_pill(ax, c2_x + 0.012, y, c2_w - 0.024, pill_h, r["x"][0], r["x"][1], r["x"][2], fontsize=8.6, weight="normal", border_color=r["x"][3])

        # Col 3: AFFORD-Bench Cell (Highlighted with checkmark)
        draw_pill(ax, c3_x + 0.012, y, c3_w - 0.024, pill_h, f"✓  {r['ours'][0]}", r["ours"][1], r["ours"][2], fontsize=8.8, weight="bold", border_color=r["ours"][3])

    # =========================================================================
    # 5. BOTTOM SUMMARY STRIP (Tasks & Scale Progress - Perfectly Snug)
    # =========================================================================
    bottom_y = row_start_y - len(rows) * row_gap + 0.012

    # Card 0: Summary Label
    ax.text(c0_x + 0.014, bottom_y + 0.038, "Benchmark Scale", fontsize=9.0, weight="bold", color="#1E293B")
    ax.text(c0_x + 0.014, bottom_y + 0.016, "Normalized data volume", fontsize=7.4, color="#64748B")

    # LIBERO-PRO Bottom: Scale Bar
    ax.text(c1_x + 0.015, bottom_y + 0.040, "130 Tasks • 56% Scale", fontsize=7.8, weight="bold", color="#64748B")
    ax.add_patch(FancyBboxPatch((c1_x + 0.015, bottom_y + 0.014), c1_w - 0.030, 0.016, boxstyle="round,pad=0,rounding_size=0.004", facecolor="#F1F5F9", edgecolor="#E2E8F0", linewidth=0.8, zorder=2))
    ax.add_patch(FancyBboxPatch((c1_x + 0.015, bottom_y + 0.014), (c1_w - 0.030) * 0.56, 0.016, boxstyle="round,pad=0,rounding_size=0.004", facecolor="#94A3B8", edgecolor="none", zorder=3))

    # LIBERO-X Bottom: Scale Bar
    ax.text(c2_x + 0.015, bottom_y + 0.040, "50 Tasks • 22% Scale", fontsize=7.8, weight="bold", color="#64748B")
    ax.add_patch(FancyBboxPatch((c2_x + 0.015, bottom_y + 0.014), c2_w - 0.030, 0.016, boxstyle="round,pad=0,rounding_size=0.004", facecolor="#F1F5F9", edgecolor="#E2E8F0", linewidth=0.8, zorder=2))
    ax.add_patch(FancyBboxPatch((c2_x + 0.015, bottom_y + 0.014), (c2_w - 0.030) * 0.22, 0.016, boxstyle="round,pad=0,rounding_size=0.004", facecolor="#FBBF24", edgecolor="none", zorder=3))

    # AFFORD-Bench Bottom: Scale Bar (Full & Vibrant)
    ax.text(c3_x + 0.015, bottom_y + 0.040, "★ 150 Tasks • 100% (2.3× Scale)", fontsize=8.2, weight="bold", color="#047857")
    ax.add_patch(FancyBboxPatch((c3_x + 0.015, bottom_y + 0.014), c3_w - 0.030, 0.016, boxstyle="round,pad=0,rounding_size=0.004", facecolor="#DCFCE7", edgecolor="#A7F3D0", linewidth=0.8, zorder=2))
    ax.add_patch(FancyBboxPatch((c3_x + 0.015, bottom_y + 0.014), c3_w - 0.030, 0.016, boxstyle="round,pad=0,rounding_size=0.004", facecolor="#10B981", edgecolor="none", zorder=3))

    # Save
    pdf_path = OUTPUT_DIR_PAPER / "benchmark_comparison.pdf"
    png_path_paper = OUTPUT_DIR_PAPER / "benchmark_comparison.png"
    png_path_docs = OUTPUT_DIR_DOCS / "benchmark_comparison.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path_paper, format="png", bbox_inches="tight", dpi=300)
    fig.savefig(png_path_docs, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Generated newly designed appealing comparison figure:")
    print(f"  -> PDF: {pdf_path}")
    print(f"  -> PNG (Paper): {png_path_paper}")
    print(f"  -> PNG (Docs): {png_path_docs}")


if __name__ == "__main__":
    generate_appealing_figure()
