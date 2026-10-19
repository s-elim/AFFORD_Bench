"""Objects, marked squares and mechanisms added to each source scene to build its task scene (used by build_bench_scenes.py).

All dimensions in meters. Free bodies get a freejoint; mechanisms are slide joints with dry friction, so a pushed
button or drawer stays where it was pushed (its state is part of the physics state).
"""
from __future__ import annotations

import math


def _v(values) -> str:
    return " ".join(f"{float(x):.5g}" for x in values)


def geom(name: str, kind: str, size, pos=(0.0, 0.0, 0.0), rgba=(0.5, 0.5, 0.5, 1.0), **attrs) -> str:
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return f'<geom name="{name}" type="{kind}" size="{_v(size)}" pos="{_v(pos)}" rgba="{_v(rgba)}"{extra}/>'


def body(name: str, pos, geoms: list[str], joint: str | None = None, indent: int = 4, **attrs) -> str:
    pad = " " * indent
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    lines = [f'{pad}<body name="{name}" pos="{_v(pos)}"{extra}>']
    if joint:
        lines.append(f"{pad}  {joint}")
    lines += [f"{pad}  {g}" for g in geoms]
    lines.append(f"{pad}</body>")
    return "\n".join(lines)


def free_body(name: str, pos, geoms: list[str]) -> str:
    return body(name, pos, geoms, joint=f'<freejoint name="{name}_joint"/>')


VISUAL = {"contype": "0", "conaffinity": "0"}
PART = {"friction": "1.6 0.005 0.0001", "condim": "4"}


def marker(name: str, xy, half, rgba) -> str:
    """Flat, non-colliding painted square on the table top (z = 0)."""
    return body(name, (xy[0], xy[1], 0.0006), [geom(f"{name}_paint", "box", (half[0], half[1], 0.0004), rgba=rgba,
                                                    **VISUAL)])


def ring(prefix: str, r_in: float, r_out: float, half_h: float, n: int, z: float, rgba, **attrs) -> list[str]:
    """Annulus of n boxes (inner radius r_in, outer r_out), as the source jig and gear are built."""
    radial = (r_out - r_in) / 2.0
    mid = (r_out + r_in) / 2.0
    tangential = mid * math.tan(math.pi / n) * 1.05
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append(geom(f"{prefix}_{k}", "box", (radial, tangential, half_h), (mid * math.cos(a), mid * math.sin(a), z),
                        rgba, euler=f"0 0 {a:.5f}", **attrs))
    return out


def button(name: str, pos, radius: float, rgba, collar_rgba=(0.2, 0.2, 0.22, 1.0)) -> str:
    """Latching push button: a gravity-compensated cap on a vertical slide joint (pushed in = positive, 12 mm travel)
    with dry friction, so it stays wherever it was pushed; a fixed collar marks its housing. Without gravity
    compensation the cap's weight would slowly creep through the soft friction constraint."""
    cap = body(f"{name}_cap", pos, [geom(f"{name}_cap_geom", "cylinder", (radius, 0.006), (0.0, 0.0, 0.006), rgba)],
               joint=f'<joint name="{name}_joint" type="slide" axis="0 0 -1" range="0 0.012" frictionloss="3" '
                     'damping="2" armature="0.01"/>', gravcomp="1")
    collar = body(f"{name}_collar", (pos[0], pos[1], pos[2] + 0.0005),
                  [geom(f"{name}_collar_geom", "cylinder", (radius + 0.004, 0.0008), rgba=collar_rgba, **VISUAL)])
    return cap + "\n" + collar


def open_box(name: str, pos, inner_half, wall: float, height: float, floor: float, rgba, floor_rgba=None) -> str:
    """Open-top box (tray, bin): floor of half-thickness `floor` and four walls rising `height` above its top."""
    ix, iy = inner_half
    hz = height / 2.0
    return body(name, pos, [
        geom(f"{name}_floor", "box", (ix + 2 * wall, iy + 2 * wall, floor), rgba=floor_rgba or rgba,
             friction="1.0 0.005 0.0001"),
        geom(f"{name}_wx1", "box", (wall, iy + 2 * wall, hz), (ix + wall, 0.0, floor + hz), rgba),
        geom(f"{name}_wx2", "box", (wall, iy + 2 * wall, hz), (-ix - wall, 0.0, floor + hz), rgba),
        geom(f"{name}_wy1", "box", (ix, wall, hz), (0.0, iy + wall, floor + hz), rgba),
        geom(f"{name}_wy2", "box", (ix, wall, hz), (0.0, -iy - wall, floor + hz), rgba),
    ])


def socket_walls(prefix: str, center, inner_half, wall: float, height: float, base_z: float, rgba) -> list[str]:
    """Four walls of a rectangular socket standing on a surface at base_z (geoms of an existing body)."""
    cx, cy = center
    ix, iy = inner_half
    hz = height / 2.0
    z = base_z + hz
    return [
        geom(f"{prefix}_wx1", "box", (wall, iy + 2 * wall, hz), (cx + ix + wall, cy, z), rgba),
        geom(f"{prefix}_wx2", "box", (wall, iy + 2 * wall, hz), (cx - ix - wall, cy, z), rgba),
        geom(f"{prefix}_wy1", "box", (ix, wall, hz), (cx, cy + iy + wall, z), rgba),
        geom(f"{prefix}_wy2", "box", (ix, wall, hz), (cx, cy - iy - wall, z), rgba),
    ]


def sponge(xy) -> str:
    """Pink kitchen sponge (8 x 5 x 2.5 cm) with a green scouring layer."""
    return free_body("sponge", (xy[0], xy[1], 0.0125), [
        geom("sponge_geom", "box", (0.04, 0.025, 0.0125), rgba=(0.95, 0.45, 0.65, 1.0), mass="0.02",
             friction="1.2 0.005 0.0001", condim="4"),
        geom("sponge_scrub", "box", (0.0402, 0.0252, 0.003), (0.0, 0.0, -0.0098), (0.15, 0.55, 0.25, 1.0), **VISUAL),
    ])


# ----------------------------------------------------------------------------------------------- tabletop_sorting
WOOD_DARK = (0.36, 0.25, 0.16, 1.0)
WOOD_LIGHT = (0.55, 0.40, 0.26, 1.0)
HANDLE = (0.75, 0.75, 0.78, 1.0)

TABLETOP = {
    "replace": [
        ('<body name="red_bin" pos="0.40 -0.24 0.012">', '<body name="red_bin" pos="0.38 -0.27 0.012">'),
        ('<body name="blue_bin" pos="0.40 0.24 0.012">', '<body name="blue_bin" pos="0.38 0.27 0.012">'),
        ('<body name="green_bin" pos="0.58 0.0 0.012">', '<body name="green_bin" pos="0.62 0.0 0.012">'),
    ],
    "bodies": [
        # Cabinet open toward the robot (-x); the drawer slides out 11 cm along -x.
        # 11.5 cm tall, so the hand's palm clears the cabinet top while placing into the open drawer.
        body("drawer_cabinet", (0.68, 0.22, 0.0), [
            geom("cab_side_l", "box", (0.07, 0.005, 0.055), (0.0, 0.085, 0.055), WOOD_DARK),
            geom("cab_side_r", "box", (0.07, 0.005, 0.055), (0.0, -0.085, 0.055), WOOD_DARK),
            geom("cab_back", "box", (0.005, 0.09, 0.055), (0.065, 0.0, 0.055), WOOD_DARK),
            geom("cab_top", "box", (0.072, 0.09, 0.005), (0.0, 0.0, 0.115), WOOD_DARK),
        ]),
        body("drawer", (0.68, 0.22, 0.0), [
            geom("drawer_floor", "box", (0.06, 0.075, 0.004), (0.0, 0.0, 0.012), WOOD_LIGHT, friction="1.0 0.005 0.0001"),
            geom("drawer_front", "box", (0.005, 0.08, 0.035), (-0.065, 0.0, 0.037), WOOD_LIGHT),
            geom("drawer_back", "box", (0.004, 0.075, 0.018), (0.056, 0.0, 0.03), WOOD_LIGHT),
            geom("drawer_side_l", "box", (0.06, 0.004, 0.018), (0.0, 0.071, 0.03), WOOD_LIGHT),
            geom("drawer_side_r", "box", (0.06, 0.004, 0.018), (0.0, -0.071, 0.03), WOOD_LIGHT),
            geom("drawer_handle", "box", (0.006, 0.035, 0.006), (-0.085, 0.0, 0.045), HANDLE),
            geom("drawer_handle_l", "box", (0.009, 0.004, 0.004), (-0.075, 0.03, 0.045), HANDLE),
            geom("drawer_handle_r", "box", (0.009, 0.004, 0.004), (-0.075, -0.03, 0.045), HANDLE),
        ], joint='<joint name="drawer_slide" type="slide" axis="-1 0 0" range="0 0.11" frictionloss="3" damping="8" '
                 'armature="0.05"/>'),
        marker("black_square", (0.62, -0.22), (0.035, 0.035), (0.12, 0.12, 0.13, 1.0)),
        marker("yellow_square", (0.30, -0.12), (0.035, 0.035), (0.98, 0.82, 0.10, 1.0)),
        marker("purple_square", (0.30, 0.12), (0.035, 0.035), (0.55, 0.25, 0.75, 1.0)),
        sponge((0.30, -0.12)),
    ],
    "exclude": [("drawer_cabinet", "drawer")],
}

# --------------------------------------------------------------------------------------------- industrial_assembly
STEEL = (0.62, 0.66, 0.72, 1.0)
RAIL = (0.35, 0.38, 0.42, 1.0)
INDUSTRIAL = {
    "remove": ["outbound_pallet"],
    # Blue rim on the second locating hole: "the blue bolt hole" versus the silver pin hole.
    "regex": [(r'(name="locating_hole_b_\d+"[^>]*rgba=")0.80 0.85 0.90 1.0"', r'\g<1>0.20 0.40 0.85 1.0"')],
    "replace": [('      <geom name="vise_jaw1" type="box" size="0.02 0.06 0.025" pos="0.065 0 0.035" '
                 'rgba="0.75 0.20 0.20 1.0"/>\n', "")],
    "bodies": [
        body("vise_jaw", (0.615, 0.0, 0.075), [geom("vise_jaw_geom", "box", (0.02, 0.06, 0.025), rgba=(0.75, 0.2, 0.2, 1))],
             joint='<joint name="vise_slide" type="slide" axis="1 0 0" range="0 0.02" frictionloss="3" damping="8" '
                   'armature="0.05"/>'),
        # Outbound pallet with four colored compartments (inner 9.7 cm each).
        body("outbound_pallet", (0.42, -0.235, 0.015), [
            geom("pal_base", "box", (0.105, 0.105, 0.006), rgba=(0.25, 0.28, 0.30, 1.0), friction="1.0 0.005 0.0001"),
            geom("pal_rail1", "box", (0.105, 0.005, 0.008), (0.0, 0.10, 0.014), RAIL),
            geom("pal_rail2", "box", (0.105, 0.005, 0.008), (0.0, -0.10, 0.014), RAIL),
            geom("pal_rail3", "box", (0.005, 0.105, 0.008), (0.10, 0.0, 0.014), RAIL),
            geom("pal_rail4", "box", (0.005, 0.105, 0.008), (-0.10, 0.0, 0.014), RAIL),
            geom("pal_div_x", "box", (0.095, 0.003, 0.008), (0.0, 0.0, 0.014), RAIL),
            geom("pal_div_y", "box", (0.003, 0.095, 0.008), (0.0, 0.0, 0.014), RAIL),
            geom("pal_red", "box", (0.046, 0.046, 0.0005), (-0.0525, 0.0525, 0.0062), (0.85, 0.15, 0.15, 1.0), **VISUAL),
            geom("pal_yellow", "box", (0.046, 0.046, 0.0005), (0.0525, 0.0525, 0.0062), (0.95, 0.80, 0.10, 1.0), **VISUAL),
            geom("pal_green", "box", (0.046, 0.046, 0.0005), (-0.0525, -0.0525, 0.0062), (0.15, 0.70, 0.25, 1.0), **VISUAL),
            geom("pal_blue", "box", (0.046, 0.046, 0.0005), (0.0525, -0.0525, 0.0062), (0.15, 0.35, 0.85, 1.0), **VISUAL),
        ]),
        open_box("parts_bin", (0.62, -0.22, 0.015), (0.06, 0.06), 0.004, 0.02, 0.005, (0.30, 0.34, 0.40, 1.0)),
        body("threaded_stud", (0.67, 0.10, 0.015), [
            geom("stud_plate", "box", (0.03, 0.03, 0.005), rgba=(0.28, 0.30, 0.34, 1.0), friction="1.0 0.005 0.0001"),
            geom("stud_rod", "cylinder", (0.006, 0.02), (0.0, 0.0, 0.025), (0.80, 0.80, 0.76, 1.0)),
            geom("stud_tip", "cylinder", (0.0045, 0.002), (0.0, 0.0, 0.047), (0.80, 0.80, 0.76, 1.0)),
        ]),
        body("control_box", (0.67, 0.225, 0.0), [
            geom("cbox_body", "box", (0.035, 0.065, 0.03), (0.0, 0.0, 0.03), (0.92, 0.78, 0.15, 1.0)),
        ]),
        button("estop", (0.67, 0.19, 0.06), 0.018, (0.85, 0.10, 0.10, 1.0)),
        button("cycle_start", (0.67, 0.26, 0.06), 0.012, (0.15, 0.75, 0.25, 1.0)),
        marker("staging_square", (0.68, -0.10), (0.035, 0.035), (0.98, 0.82, 0.10, 1.0)),
        marker("parts_mat", (0.38, -0.03), (0.07, 0.085), (0.13, 0.13, 0.15, 1.0)),
        free_body("bearing_ring", (0.595, -0.22, 0.026), ring("bearing", 0.0215, 0.030, 0.006, 16, 0.0, STEEL,
                                                              mass="0.0038", **PART)),
        free_body("hex_nut", (0.65, -0.20, 0.026), ring("nut", 0.0075, 0.014, 0.006, 6, 0.0, (0.78, 0.78, 0.72, 1.0),
                                                        mass="0.0033", **PART)),
        free_body("flange_bolt", (0.43, 0.02, 0.021), [
            geom("bolt_shank", "cylinder", (0.008, 0.02), rgba=(0.20, 0.20, 0.23, 1.0), mass="0.02", **PART),
            geom("bolt_flange", "cylinder", (0.014, 0.0015), (0.0, 0.0, 0.0215), (0.20, 0.20, 0.23, 1.0), mass="0.003"),
            geom("bolt_head", "cylinder", (0.012, 0.005), (0.0, 0.0, 0.028), (0.25, 0.25, 0.28, 1.0), mass="0.005",
                 **PART),
        ]),
    ],
    "exclude": [("assembly_jig", "vise_jaw"), ("control_box", "estop_cap"), ("control_box", "cycle_start_cap")],
}

# --------------------------------------------------------------------------------------------- industrial_electronics
PCB = (0.10, 0.42, 0.24, 1.0)
SOCKET = (0.12, 0.12, 0.13, 1.0)
BRASS = (0.80, 0.75, 0.35, 1.0)
ELECTRONICS = {
    "source": "industrial_assembly",
    "remove": ["assembly_jig", "inbound_tray", "gear_feeder", "pin_feeder", "outbound_pallet", "machined_shaft",
               "spur_gear", "dowel_pin"],
    "bodies": [
        body("test_board", (0.57, 0.0, 0.0), [
            geom("board_pcb", "box", (0.10, 0.14, 0.01), (0.0, 0.0, 0.01), PCB, friction="1.0 0.005 0.0001"),
            *socket_walls("chip_socket", (-0.04, 0.075), (0.0145, 0.0145), 0.003, 0.004, 0.02, SOCKET),
            *socket_walls("relay_base", (-0.04, -0.075), (0.0172, 0.0172), 0.003, 0.008, 0.02, (0.85, 0.85, 0.82, 1.0)),
            geom("fuse_clip_a", "box", (0.002, 0.013, 0.004), (0.0595, 0.075, 0.024), BRASS),
            geom("fuse_clip_b", "box", (0.002, 0.013, 0.004), (0.0405, 0.075, 0.024), BRASS),
            *socket_walls("receptacle", (0.05, -0.075), (0.0135, 0.0085), 0.003, 0.01, 0.02, (0.92, 0.92, 0.90, 1.0)),
        ]),
        open_box("parts_tray", (0.38, 0.0, 0.006), (0.065, 0.12), 0.004, 0.012, 0.004, (0.18, 0.30, 0.55, 1.0)),
        open_box("reject_bin", (0.40, 0.25, 0.005), (0.05, 0.05), 0.004, 0.04, 0.005, (0.80, 0.25, 0.20, 1.0)),
        button("test_start", (0.66, 0.0, 0.02), 0.012, (0.15, 0.75, 0.25, 1.0)),
        free_body("microchip", (0.35, 0.08, 0.0155), [
            geom("chip_body", "box", (0.0125, 0.0125, 0.0075), rgba=(0.08, 0.08, 0.09, 1.0), mass="0.012", **PART),
            geom("chip_mark", "cylinder", (0.0025, 0.0004), (0.007, 0.007, 0.0075), (0.85, 0.85, 0.85, 1.0), **VISUAL),
        ]),
        free_body("cartridge_fuse", (0.40, 0.06, 0.016), [
            geom("fuse_glass", "cylinder", (0.006, 0.011), rgba=(0.85, 0.90, 0.92, 1.0), euler="1.5708 0 0",
                 mass="0.004", **PART),
            geom("fuse_cap_a", "box", (0.0063, 0.0025, 0.0063), (0.0, 0.0135, 0.0), (0.78, 0.78, 0.80, 1.0),
                 mass="0.002", **PART),
            geom("fuse_cap_b", "box", (0.0063, 0.0025, 0.0063), (0.0, -0.0135, 0.0), (0.78, 0.78, 0.80, 1.0),
                 mass="0.002", **PART),
        ]),
        free_body("relay", (0.35, -0.07, 0.025), [
            geom("relay_body", "box", (0.015, 0.015, 0.015), rgba=(0.15, 0.30, 0.75, 1.0), mass="0.03", **PART),
        ]),
        free_body("harness_plug", (0.41, -0.06, 0.025), [
            geom("plug_body", "box", (0.011, 0.006, 0.015), rgba=(0.95, 0.55, 0.10, 1.0), mass="0.012", **PART),
            geom("plug_wire", "cylinder", (0.003, 0.006), (0.0, 0.0, 0.021), (0.15, 0.15, 0.15, 1.0), **VISUAL),
        ]),
    ],
    "exclude": [("test_board", "test_start_cap")],
}

# ------------------------------------------------------------------------------------------------------- kitchens
GREEN_MARK = (0.20, 0.70, 0.30, 1.0)
KITCHEN_BURGER = {
    # Topping tray and bun caddy rims 2 mm high instead of 20 mm, so the slices and buns stand proud of them for a
    # grasp anywhere in their compartment.
    "regex": [(r'(name="(?:top_(?:w\d|div)|caddy_(?:w\d|div))" type="box" size="[\d.]+ [\d.]+) 0.010" '
               r'pos="([-\d.]+) ([-\d.]+) 0.004"', r'\1 0.001" pos="\2 \3 0.007"')],
    # The crumb, char marks and tomato core are visual only, 0.1 mm proud of the body they decorate: as thin collision
    # layers inside it they pushed an item that sank past their midplane into the body below (a tomato 7 mm into a bun).
    "replace": [(f'name="{name}" type="{kind}" size="{size}" pos="{old}"',
                 f'name="{name}" type="{kind}" size="{size}" pos="{new}" contype="0" conaffinity="0"')
                for name, kind, size, old, new in (
                    ("bbun_crumb", "cylinder", "0.033 0.002", "0 0 0.008", "0 0 0.0071"),
                    ("char1", "box", "0.025 0.002 0.001", "0 0.012 0.0095", "0 0.012 0.0081"),
                    ("char2", "box", "0.028 0.002 0.001", "0 0 0.0095", "0 0 0.0081"),
                    ("char3", "box", "0.025 0.002 0.001", "0 -0.012 0.0095", "0 -0.012 0.0081"),
                    ("tomato_core", "cylinder", "0.012 0.002", "0 0 0.004", "0 0 0.0031"))]
    # Likewise the surfaces things are put on: the plate's base reaches up to its rim (rim and gold line visual only),
    # and the grill's ribs reach down into its base, so neither has a thin collision layer inside another.
    + [('name="plate_base" type="cylinder" size="0.082 0.005"',
        'name="plate_base" type="cylinder" size="0.082 0.007" pos="0 0 0.002"'),
       ('name="plate_rim" type="cylinder" size="0.085 0.003" pos="0 0 0.006"',
        'name="plate_rim" type="cylinder" size="0.085 0.003" pos="0 0 0.006" contype="0" conaffinity="0"'),
       ('name="gold_rim" type="cylinder" size="0.080 0.001" pos="0 0 0.008"',
        'name="gold_rim" type="cylinder" size="0.080 0.001" pos="0 0 0.008" contype="0" conaffinity="0"')]
    + [(f'name="rib{k}" type="box" size="0.070 0.003 0.003" pos="0 {y} 0.013"',
        f'name="rib{k}" type="box" size="0.070 0.003 0.014" pos="0 {y} 0.002"')
       for k, y in ((1, "0.040"), (2, "0.015"), (3, "-0.015"), (4, "-0.040"))],
    "bodies": [
        body("grill_switch", (0.68, 0.20, 0.0), [geom("switch_box", "box", (0.018, 0.03, 0.02), (0.0, 0.0, 0.02),
                                                      (0.20, 0.20, 0.22, 1.0))]),
        button("grill_power", (0.68, 0.20, 0.04), 0.011, (0.90, 0.30, 0.10, 1.0)),
        marker("green_mark", (0.30, 0.10), (0.035, 0.035), GREEN_MARK),
        sponge((0.30, -0.04)),
        free_body("lettuce_leaf", (0.66, -0.02, 0.006), [
            geom("lettuce_body", "cylinder", (0.034, 0.005), rgba=(0.35, 0.75, 0.25, 1.0), mass="0.01", **PART),
        ]),
    ],
    "exclude": [("grill_switch", "grill_power_cap")],
}

KITCHEN_COFFEE = {
    "regex": [(r'(name="pbin_w\d" type="box" size="[\d.]+ [\d.]+) 0.018" pos="([-\d.]+) ([-\d.]+) 0.018"',
               r'\1 0.008" pos="\2 \3 0.008"')],
    # The mug stays the source scene's solid cylinder: a thin-walled hollow mug let a finger slip inside it and lift it by the rim.
    "replace": [('      <geom name="cm_button" type="cylinder" size="0.009 0.005" pos="-0.08 0.045 0.075" '
                 'euler="0 1.5708 0" rgba="0.20 0.85 0.30 1.0"/>\n', "")],
    "bodies": [
        button("brew", (0.47, -0.115, 0.20), 0.012, (0.20, 0.85, 0.30, 1.0)),
        body("capsule_slot", (0.53, -0.16, 0.20), ring("slot", 0.0175, 0.024, 0.005, 16, 0.005, (0.35, 0.37, 0.40, 1.0))),
        body("saucer", (0.64, 0.05, 0.004), [
            geom("saucer_plate", "cylinder", (0.05, 0.003), rgba=(0.96, 0.96, 0.97, 1.0), friction="1.0 0.005 0.0001"),
            *ring("saucer_rim", 0.044, 0.05, 0.003, 20, 0.004, (0.90, 0.90, 0.92, 1.0)),
        ]),
        marker("green_mark", (0.30, 0.10), (0.035, 0.035), GREEN_MARK),
        sponge((0.30, -0.04)),
        free_body("red_capsule", (0.46, 0.02, 0.0155), [
            geom("rpod_geom", "cylinder", (0.014, 0.014), rgba=(0.80, 0.12, 0.15, 1.0), mass="0.02",
                 friction="1.2 0.005 0.0001", condim="4"),
            geom("rpod_rim", "cylinder", (0.016, 0.003), (0.0, 0.0, 0.012), (0.90, 0.20, 0.22, 1.0), mass="0.005"),
        ]),
    ],
    "exclude": [("coffee_machine", "brew_cap"), ("coffee_machine", "capsule_slot")],
}

# ------------------------------------------------------------------------------------------------------------ lab
LAB = {
    # Shorter micro-tube holes (1.5 cm walls, as in a 1.5 mL rack) so that 2.5 cm of the tube stands proud for a grasp.
    "regex": [(r'(name="rack_(?:a_hole1|b_micro_hole)_\d+" type="box" size="0.00175 0.00295) 0.01500" '
               r'pos="([-\d.]+) ([-\d.]+) 0.00500"', r'\1 0.00750" pos="\2 \3 -0.00250"')],
    "bodies": [
        body("start_panel", (0.47, 0.20, 0.0), [geom("panel_box", "box", (0.012, 0.03, 0.025), (0.0, 0.0, 0.025),
                                                     (0.85, 0.87, 0.90, 1.0))]),
        button("centrifuge_start", (0.47, 0.20, 0.05), 0.009, (0.15, 0.70, 0.30, 1.0)),
        # Domed cover: the 5.5 cm skirt rests on the centrifuge rim and clears a micro tube standing in the rotor.
        free_body("centrifuge_lid", (0.68, 0.02, 0.058), [
            geom("lid_disc", "cylinder", (0.08, 0.003), rgba=(0.75, 0.82, 0.92, 0.9), mass="0.03", **PART),
            *ring("lid_skirt", 0.068, 0.08, 0.026, 24, -0.029, (0.70, 0.78, 0.90, 0.9), mass="0.002", **PART),
            geom("lid_knob", "cylinder", (0.009, 0.015), (0.0, 0.0, 0.018), (0.25, 0.30, 0.40, 1.0), mass="0.005",
                 **PART),
        ]),
        open_box("waste_bin", (0.66, -0.18, 0.005), (0.045, 0.045), 0.004, 0.045, 0.005, (0.95, 0.80, 0.10, 1.0),
                 floor_rgba=(0.85, 0.15, 0.15, 1.0)),
        marker("pipette_pad", (0.42, 0.02), (0.03, 0.03), (0.20, 0.45, 0.90, 1.0)),
        marker("green_mark", (0.30, 0.10), (0.035, 0.035), GREEN_MARK),
        sponge((0.30, -0.04)),
    ],
    "exclude": [("start_panel", "centrifuge_start_cap")],
}

SCENE_PARTS: dict[str, dict] = {
    "tabletop_sorting": TABLETOP,
    "industrial_assembly": INDUSTRIAL,
    "industrial_electronics": ELECTRONICS,
    "kitchen_burger": KITCHEN_BURGER,
    "kitchen_coffee": KITCHEN_COFFEE,
    "lab_science": LAB,
}
