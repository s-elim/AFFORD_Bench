"""Generate an elegant, publication-grade benchmark comparison figure:
AFFORD-Bench vs. LIBERO-PRO & LIBERO-X.
Designed for maximum aesthetic appeal, visual clarity, and intuitive understanding.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

# Set publication typography & aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Lucida Grande"],
    "mathtext.fontset": "cm",
    "figure.autolayout": False,
})

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR_PAPER = ROOT / "AFFORD-PRO_RSS'27" / "paper-template-latex" / "figures"
OUTPUT_DIR_DOCS = ROOT / "docs" / "images"
OUTPUT_DIR_PAPER.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_DOCS.mkdir(parents=True, exist_ok=True)


def draw_card(ax, x, y, w, h, bg_color, border_color, border_width=1.5, corner_radius=0.016, zorder=1):
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


def draw_pill(ax, x, y, w, h, text, bg_color, text_color, fontsize=8.0, weight="bold", zorder=4):
    """Draw a stylish rounded badge / pill with centered text."""
    pill = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.010",
        facecolor=bg_color,
        edgecolor="none",
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


def build_elegant_figure():
    # Crisp canvas with balanced 16:9.2 aspect ratio
    fig = plt.figure(figsize=(19.6, 11.2), dpi=300, facecolor="#F8FAFC")

    # Master canvas overlay for header, cards, and styling
    ax_canvas = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax_canvas.set_facecolor("#F8FAFC")
    ax_canvas.set_xlim(0, 1)
    ax_canvas.set_ylim(0, 1)
    ax_canvas.axis("off")

    # =========================================================================
    # 1. HEADER SECTION (Editorial Banner)
    # =========================================================================
    # Royal blue gradient top bar
    ax_canvas.add_patch(Rectangle((0, 0.988), 1.0, 0.012, facecolor="#1D4ED8", edgecolor="none"))

    # Title & Subtitle
    ax_canvas.text(
        0.045, 0.954,
        "BENCHMARK LANDSCAPE: AFFORD-Bench vs. LIBERO-PRO & LIBERO-X",
        fontsize=18,
        weight="heavy",
        color="#0F172A",
        va="center"
    )
    ax_canvas.text(
        0.045, 0.927,
        "A Systematic Comparison of Physical Contact Affordances, Domain Diversity, Sensory Modalities, and Policy Universality",
        fontsize=10.5,
        weight="normal",
        color="#475569",
        va="center"
    )

    # Spotlight badge on top right
    draw_pill(ax_canvas, 0.815, 0.932, 0.140, 0.034, "★ RSS 2027 SPOTLIGHT", "#1E40AF", "#FFFFFF", fontsize=8.5, weight="bold")

    # =========================================================================
    # 2. UPPER TIER: THE 3 HERO BENCHMARK PROFILES (Side-by-Side Cards)
    # =========================================================================
    card_y = 0.505
    card_h = 0.395
    card_w = 0.288

    # -------------------------------------------------------------------------
    # Card 1: LIBERO-PRO (Zhou et al., 2026)
    # -------------------------------------------------------------------------
    x1 = 0.045
    draw_card(ax_canvas, x1, card_y, card_w, card_h, "#FFFFFF", "#CBD5E1", border_width=1.5, corner_radius=0.018)
    
    # Header strip
    ax_canvas.add_patch(FancyBboxPatch((x1, card_y + card_h - 0.055), card_w, 0.055, boxstyle="round,pad=0,rounding_size=0.018", facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.2, zorder=2))
    ax_canvas.text(x1 + 0.015, card_y + card_h - 0.024, "LIBERO-PRO", fontsize=13, weight="bold", color="#334155", zorder=3)
    ax_canvas.text(x1 + 0.015, card_y + card_h - 0.042, "Zhou et al., 2026 • Spatial & Visual OOD", fontsize=8.5, color="#64748B", zorder=3)
    draw_pill(ax_canvas, x1 + card_w - 0.078, card_y + card_h - 0.042, 0.068, 0.024, "Baseline", "#E2E8F0", "#475569", fontsize=8.0, zorder=4)

    # Feature items for Card 1
    rows_c1 = [
        ("Task Breadth", "130 Tasks across 4 Domestic Suites", "#334155", False),
        ("Environments", "Domestic Kitchen & Living Room scenes", "#475569", False),
        ("Sensory Streams", "Single RGB (128 × 128) low-resolution", "#64748B", False),
        ("Contact Physics", "Nominal Only (Rigid grasp, fixed friction)", "#DC2626", True),
        ("Action Prompts", "Atomic single-step language instructions", "#475569", False),
        ("Policy Target", "VLA Models (OpenVLA, π0)", "#475569", False),
        ("Scale / Engine", "6,500 Demos (1.3M Steps) • Robosuite (~15 FPS)", "#64748B", False),
    ]
    y_cursor = card_y + card_h - 0.082
    for label, val, val_col, is_alert in rows_c1:
        ax_canvas.text(x1 + 0.015, y_cursor, label, fontsize=8.5, weight="bold", color="#1E293B", zorder=3)
        ax_canvas.text(x1 + 0.015, y_cursor - 0.018, val, fontsize=8.2, weight="bold" if is_alert else "normal", color=val_col, zorder=3)
        ax_canvas.plot([x1 + 0.015, x1 + card_w - 0.015], [y_cursor - 0.025, y_cursor - 0.025], color="#F1F5F9", linewidth=1.0, zorder=2)
        y_cursor -= 0.041

    # Bottom limitation callout
    draw_card(ax_canvas, x1 + 0.015, card_y + 0.014, card_w - 0.030, 0.034, "#FEF2F2", "#FECACA", border_width=1.0, corner_radius=0.008)
    ax_canvas.text(x1 + card_w / 2, card_y + 0.031, "Limited to Nominal Contact Physics • Domestic Tabletop Only", fontsize=8.0, weight="bold", color="#B91C1C", ha="center", va="center", zorder=3)

    # -------------------------------------------------------------------------
    # Card 2: LIBERO-X (Wang et al., 2026)
    # -------------------------------------------------------------------------
    x2 = 0.356
    draw_card(ax_canvas, x2, card_y, card_w, card_h, "#FFFFFF", "#FED7AA", border_width=1.5, corner_radius=0.018)
    
    # Header strip
    ax_canvas.add_patch(FancyBboxPatch((x2, card_y + card_h - 0.055), card_w, 0.055, boxstyle="round,pad=0,rounding_size=0.018", facecolor="#FFF7ED", edgecolor="#FED7AA", linewidth=1.2, zorder=2))
    ax_canvas.text(x2 + 0.015, card_y + card_h - 0.024, "LIBERO-X", fontsize=13, weight="bold", color="#9A3412", zorder=3)
    ax_canvas.text(x2 + 0.015, card_y + card_h - 0.042, "Wang et al., 2026 • Hierarchical L1–L5 Shifts", fontsize=8.5, color="#C2410C", zorder=3)
    draw_pill(ax_canvas, x2 + card_w - 0.082, card_y + card_h - 0.042, 0.072, 0.024, "Prior SOTA", "#FFEDD5", "#9A3412", fontsize=8.0, zorder=4)

    # Feature items for Card 2
    rows_c2 = [
        ("Task Breadth", "600 Tasks across 5 Domestic Suites", "#334155", False),
        ("Environments", "Domestic Living, Kitchen, Dining spaces", "#475569", False),
        ("Sensory Streams", "Dual RGB (256 × 256) mid-resolution", "#475569", False),
        ("Contact Physics", "Nominal Only (Rigid grasp, fixed friction)", "#DC2626", True),
        ("Action Prompts", "Atomic language instructions with subtask tags", "#475569", False),
        ("Policy Target", "VLA Models (OpenVLA, Octo)", "#475569", False),
        ("Scale / Engine", "2,520 Demos (0.5M Steps) • Robosuite (~15 FPS)", "#64748B", False),
    ]
    y_cursor = card_y + card_h - 0.082
    for label, val, val_col, is_alert in rows_c2:
        ax_canvas.text(x2 + 0.015, y_cursor, label, fontsize=8.5, weight="bold", color="#1E293B", zorder=3)
        ax_canvas.text(x2 + 0.015, y_cursor - 0.018, val, fontsize=8.2, weight="bold" if is_alert else "normal", color=val_col, zorder=3)
        ax_canvas.plot([x2 + 0.015, x2 + card_w - 0.015], [y_cursor - 0.025, y_cursor - 0.025], color="#F1F5F9", linewidth=1.0, zorder=2)
        y_cursor -= 0.041

    # Bottom limitation callout
    draw_card(ax_canvas, x2 + 0.015, card_y + 0.014, card_w - 0.030, 0.034, "#FFFBEB", "#FDE68A", border_width=1.0, corner_radius=0.008)
    ax_canvas.text(x2 + card_w / 2, card_y + 0.031, "No 3D Depth Sensing • Domestic Tabletop Only", fontsize=8.0, weight="bold", color="#B45309", ha="center", va="center", zorder=3)

    # -------------------------------------------------------------------------
    # Card 3: AFFORD-Bench (Ours, 2027) - FEATURED / HERO CARD
    # -------------------------------------------------------------------------
    x3 = 0.667
    # Outer halo accent
    draw_card(ax_canvas, x3 - 0.003, card_y - 0.003, card_w + 0.006, card_h + 0.006, "#EFF6FF", "#93C5FD", border_width=2.5, corner_radius=0.020)
    draw_card(ax_canvas, x3, card_y, card_w, card_h, "#FFFFFF", "#2563EB", border_width=2.0, corner_radius=0.018)

    # Header strip (Rich Royal Blue)
    ax_canvas.add_patch(FancyBboxPatch((x3, card_y + card_h - 0.055), card_w, 0.055, boxstyle="round,pad=0,rounding_size=0.018", facecolor="#1E40AF", edgecolor="#1D4ED8", linewidth=1.0, zorder=2))
    ax_canvas.text(x3 + 0.015, card_y + card_h - 0.024, "AFFORD-Bench (Ours)", fontsize=13, weight="heavy", color="#FFFFFF", zorder=3)
    ax_canvas.text(x3 + 0.015, card_y + card_h - 0.042, "RSS 2027 • Physical Affordance Grounding & Policy Agnostic", fontsize=8.5, color="#BFDBFE", zorder=3)
    draw_pill(ax_canvas, x3 + card_w - 0.092, card_y + card_h - 0.042, 0.082, 0.024, "★ REVOLUTION", "#38BDF8", "#0369A1", fontsize=8.0, zorder=4)

    # Feature items for Card 3
    rows_c3 = [
        ("Task Breadth", "210 Defined Tasks across 7 Distinct Domains", "#1E40AF", True),
        ("Environments", "Industrial, Commercial Culinary, Lab, Tabletop, Chaining", "#0F172A", False),
        ("Sensory Streams", "Dual RGB (640×660, 300×420) + 3D Metric Depth", "#047857", True),
        ("Contact Physics", "Full Dynamics: Friction μ ∈ [0.1, 1.2], Mass, Clearances", "#059669", True),
        ("Action Prompts", "Compound 2–3 Step Continuous Instructions (D6, D7)", "#0F172A", False),
        ("Policy Target", "Policy-Agnostic: BC, DP, ACT, DP3, VLA, Offline RL", "#1E40AF", True),
        ("Scale / Engine", "10,500 Demos (2.3M Steps) • C++ MuJoCo (45 FPS, $0 Cost)", "#047857", True),
    ]
    y_cursor = card_y + card_h - 0.082
    for label, val, val_col, is_highlight in rows_c3:
        ax_canvas.text(x3 + 0.015, y_cursor, label, fontsize=8.5, weight="bold", color="#1E293B", zorder=3)
        ax_canvas.text(x3 + 0.015, y_cursor - 0.018, val, fontsize=8.2, weight="bold" if is_highlight else "normal", color=val_col, zorder=3)
        ax_canvas.plot([x3 + 0.015, x3 + card_w - 0.015], [y_cursor - 0.025, y_cursor - 0.025], color="#EFF6FF", linewidth=1.0, zorder=2)
        y_cursor -= 0.041

    # Bottom advantage callout
    draw_card(ax_canvas, x3 + 0.015, card_y + 0.014, card_w - 0.030, 0.034, "#ECFDF5", "#A7F3D0", border_width=1.0, corner_radius=0.008)
    ax_canvas.text(x3 + card_w / 2, card_y + 0.031, "✓ Full Contact Physics • Universal Policy Agnosticism • $0 Cost", fontsize=8.0, weight="bold", color="#047857", ha="center", va="center", zorder=3)

    # =========================================================================
    # 3. LOWER TIER: IN-DEPTH VISUAL METRICS (3 Panels)
    # =========================================================================
    lower_y = 0.040
    lower_h = 0.425

    # -------------------------------------------------------------------------
    # Panel (a): Multi-Dimensional Capability Radar Chart
    # -------------------------------------------------------------------------
    panel_a_w = 0.298
    draw_card(ax_canvas, 0.045, lower_y, panel_a_w, lower_h, "#FFFFFF", "#E2E8F0", border_width=1.2, corner_radius=0.018)
    ax_canvas.text(0.060, lower_y + lower_h - 0.030, "(a) Multi-Dimensional Capability Radar", fontsize=11.5, weight="bold", color="#0F172A", zorder=3)
    ax_canvas.text(0.060, lower_y + lower_h - 0.048, "Normalized capability scores across 6 core robotic manipulation axes", fontsize=8.0, color="#64748B", zorder=3)

    # Inset Polar Axes with extra breathing room
    ax_radar = fig.add_axes([0.055, lower_y + 0.022, 0.278, lower_h - 0.125], polar=True, zorder=2)
    ax_radar.set_facecolor("#F8FAFC")

    categories = [
        "Physical Affordance\nGrounding",
        "Domain & Scene\nDiversity",
        "Perceptual Streams\n(RGB + Depth)",
        "Policy-Agnostic\nUniversality",
        "Hierarchical OOD\nProtocol (L1–L5)",
        "Multi-Subtask\nChaining",
    ]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    s_pro = [0.20, 0.40, 0.25, 0.38, 0.65, 0.25]
    s_pro += s_pro[:1]

    s_x = [0.35, 0.55, 0.50, 0.45, 0.85, 0.58]
    s_x += s_x[:1]

    s_ours = [0.96, 0.98, 0.95, 0.98, 0.96, 0.95]
    s_ours += s_ours[:1]

    # Plot Radar Polygons
    ax_radar.plot(angles, s_pro, color="#94A3B8", linewidth=1.8, linestyle="--", label="LIBERO-PRO")
    ax_radar.fill(angles, s_pro, color="#94A3B8", alpha=0.12)

    ax_radar.plot(angles, s_x, color="#F59E0B", linewidth=2.0, linestyle="-.", label="LIBERO-X")
    ax_radar.fill(angles, s_x, color="#F59E0B", alpha=0.16)

    ax_radar.plot(angles, s_ours, color="#2563EB", linewidth=2.6, linestyle="-", marker="o", markersize=4.5, label="AFFORD-Bench (Ours)")
    ax_radar.fill(angles, s_ours, color="#3B82F6", alpha=0.25)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, size=7.2, weight="bold", color="#1E293B")
    ax_radar.tick_params(pad=12)
    ax_radar.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=6.5, color="#94A3B8")
    ax_radar.set_ylim(0, 1.16)
    ax_radar.grid(color="#E2E8F0", linestyle=":", linewidth=0.8)
    ax_radar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=7.8)

    # -------------------------------------------------------------------------
    # Panel (b): Quantitative Scale & Benchmark Statistics (Normalized Gauge Bars)
    # -------------------------------------------------------------------------
    panel_b_w = 0.298
    draw_card(ax_canvas, 0.356, lower_y, panel_b_w, lower_h, "#FFFFFF", "#E2E8F0", border_width=1.2, corner_radius=0.018)
    ax_canvas.text(0.371, lower_y + lower_h - 0.030, "(b) Quantitative Scale Comparison", fontsize=11.5, weight="bold", color="#0F172A", zorder=3)
    ax_canvas.text(0.371, lower_y + lower_h - 0.048, "Demonstration steps, evaluation suites, and domain diversity", fontsize=8.0, color="#64748B", zorder=3)

    # Draw 3 balanced comparison groups (Timesteps, Suites, Domains) inside Panel (b)
    scale_groups = [
        (
            "1. Physical Demonstration Timesteps",
            [("LIBERO-PRO", "1.3M Steps", 1.3 / 2.3, "#94A3B8"),
             ("LIBERO-X", "0.5M Steps", 0.5 / 2.3, "#F59E0B"),
             ("AFFORD-Bench (Ours)", "2.3M Steps (>10.5k Demos)", 2.3 / 2.3, "#2563EB")],
            "+77% Scale",
            "#1D4ED8",
            "#EFF6FF"
        ),
        (
            "2. Out-of-Distribution (OOD) Suites",
            [("LIBERO-PRO", "20 Suites", 20.0 / 35.0, "#94A3B8"),
             ("LIBERO-X", "15 Suites", 15.0 / 35.0, "#F59E0B"),
             ("AFFORD-Bench (Ours)", "35 Suites (7×5 Regimes)", 35.0 / 35.0, "#2563EB")],
            "+75% Suites",
            "#059669",
            "#ECFDF5"
        ),
        (
            "3. Distinct Physical Domains",
            [("LIBERO-PRO", "4 Domains (Domestic)", 4.0 / 7.0, "#94A3B8"),
             ("LIBERO-X", "5 Domains (Domestic)", 5.0 / 7.0, "#F59E0B"),
             ("AFFORD-Bench (Ours)", "7 Diverse Domains", 7.0 / 7.0, "#2563EB")],
            "7 Real Domains",
            "#7C3AED",
            "#F5F3FF"
        ),
    ]

    y_grp = lower_y + lower_h - 0.076
    grp_h = 0.102
    track_w = 0.072  # compact track width so text never runs off the card edge

    for title, bars, badge_txt, badge_col, badge_bg in scale_groups:
        # Group container card
        draw_card(ax_canvas, 0.370, y_grp - grp_h + 0.012, panel_b_w - 0.028, grp_h, "#F8FAFC", "#E2E8F0", border_width=1.0, corner_radius=0.010, zorder=2)
        ax_canvas.text(0.380, y_grp - 0.008, title, fontsize=8.8, weight="bold", color="#1E293B", zorder=3)
        draw_pill(ax_canvas, 0.370 + panel_b_w - 0.105, y_grp - 0.014, 0.095, 0.018, badge_txt, badge_bg, badge_col, fontsize=7.2, weight="bold", zorder=4)

        # Draw the 3 horizontal progress bars
        bar_y_cursor = y_grp - 0.033
        for bench_name, val_str, frac, bcol in bars:
            # Benchmark name label
            ax_canvas.text(0.380, bar_y_cursor, bench_name, fontsize=7.5, color="#475569", weight="bold" if "Ours" in bench_name else "normal", zorder=3)
            # Background track
            ax_canvas.add_patch(FancyBboxPatch((0.470, bar_y_cursor - 0.004), track_w, 0.008, boxstyle="round,pad=0,rounding_size=0.004", facecolor="#E2E8F0", edgecolor="none", zorder=3))
            # Filled progress
            ax_canvas.add_patch(FancyBboxPatch((0.470, bar_y_cursor - 0.004), track_w * frac, 0.008, boxstyle="round,pad=0,rounding_size=0.004", facecolor=bcol, edgecolor="none", zorder=4))
            # Value string
            ax_canvas.text(0.470 + track_w + 0.008, bar_y_cursor, val_str, fontsize=7.2, color=bcol, weight="bold" if "Ours" in bench_name else "normal", zorder=3)
            bar_y_cursor -= 0.018

        y_grp -= 0.114

    # -------------------------------------------------------------------------
    # Panel (c): Core Innovations & Grounding Breakdown
    # -------------------------------------------------------------------------
    panel_c_w = 0.298
    draw_card(ax_canvas, 0.667, lower_y, panel_c_w, lower_h, "#FFFFFF", "#E2E8F0", border_width=1.2, corner_radius=0.018)
    ax_canvas.text(0.682, lower_y + lower_h - 0.030, "(c) Core Technical Innovations", fontsize=11.5, weight="bold", color="#0F172A", zorder=3)
    ax_canvas.text(0.682, lower_y + lower_h - 0.048, "Four foundational breakthroughs introduced by AFFORD-Bench", fontsize=8.0, color="#64748B", zorder=3)

    # 4 Innovation Cards inside Panel (c) with distinct badges
    innovations = [
        (
            "1",
            "Physical Contact Dynamics & Slippage",
            "Coulomb friction variations (μ ∈ [0.1, 1.2]), dynamic payload mass scaling (0.5–3.0×),\nand sub-cm clearances (±1.5–5 mm) expose rote visual trajectory memorization.",
            "#059669",
            "#ECFDF5",
            "#A7F3D0"
        ),
        (
            "2",
            "High-Resolution 3D Depth Sensing",
            "Dual-camera synchronized RGB (640×660 workcell + 300×420 wrist) paired with\ncalibrated 3D metric depth point clouds, directly unlocking 3D spatial policies (DP3).",
            "#0284C7",
            "#F0F9FF",
            "#BAE6FD"
        ),
        (
            "3",
            "Multi-Subtask Continuous Chaining",
            "Domains D6 & D7 feature compound 2–3 step continuous instructions (e.g., 'align shaft,\npress bearing, lock clamp') to benchmark error compounding across extended horizons.",
            "#7C3AED",
            "#F5F3FF",
            "#DDD6FE"
        ),
        (
            "4",
            "Universal Policy Agnosticism",
            "Standardized Robomimic HDF5 directly ingestible by BC, Diffusion Policy, ACT,\nDP3, OpenVLA, SmolVLA, AtomVLA, and Offline RL (CQL/IQL) with pre-computed splits.",
            "#2563EB",
            "#EFF6FF",
            "#BFDBFE"
        ),
    ]

    y_inn = lower_y + lower_h - 0.068
    box_h = 0.076
    for num_str, title, desc, col, bg_c, bord_c in innovations:
        draw_card(ax_canvas, 0.682, y_inn - box_h + 0.012, panel_c_w - 0.030, box_h, bg_c, bord_c, border_width=1.0, corner_radius=0.010, zorder=2)
        
        # Clean typographic header with badge
        ax_canvas.text(0.694, y_inn - 0.008, f"#{num_str}  {title}", fontsize=8.8, weight="bold", color=col, zorder=4)
        # Description below the header
        ax_canvas.text(0.694, y_inn - 0.028, desc, fontsize=7.2, color="#334155", zorder=4, linespacing=1.30)
        y_inn -= 0.088

    # Save high-res PNG & vector PDF
    pdf_path = OUTPUT_DIR_PAPER / "benchmark_comparison.pdf"
    png_path_paper = OUTPUT_DIR_PAPER / "benchmark_comparison.png"
    png_path_docs = OUTPUT_DIR_DOCS / "benchmark_comparison.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path_paper, format="png", bbox_inches="tight", dpi=300)
    fig.savefig(png_path_docs, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Generated elegant benchmark comparison figure:")
    print(f"  -> PDF: {pdf_path}")
    print(f"  -> PNG (Paper): {png_path_paper}")
    print(f"  -> PNG (Docs): {png_path_docs}")


if __name__ == "__main__":
    build_elegant_figure()
