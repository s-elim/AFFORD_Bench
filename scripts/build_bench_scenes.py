"""Generate the AFFORD-Bench scene XMLs: each source scene with its idle distractor bodies removed and the objects,
marked goal squares and mechanisms that the instructions name added (see docs/roadmap/T9_TASK_REDESIGN.md).

    python scripts/build_bench_scenes.py            # writes afford_pro/assets/<scene>_bench.xml
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.bench_scene_parts import SCENE_PARTS  # noqa: E402

ASSETS = ROOT / "afford_pro" / "assets"
DISTRACTOR = re.compile(r"\n\s*<body name=\"distractor_\d+\".*?</body>", re.S)


def render(scene: str) -> str:
    """The task XML of a scene: its source with the idle distractors removed and the scene parts applied."""
    parts = SCENE_PARTS[scene]
    text = (ASSETS / f"{parts.get('source', scene)}.xml").read_text(encoding="utf-8")
    text = DISTRACTOR.sub("", text)
    for name in parts.get("remove", ()):
        pattern = re.compile(rf'(\n\s*<!--[^\n]*-->)?\n\s*<body name="{name}".*?</body>', re.S)
        text, count = pattern.subn("", text, count=1)
        if count == 0:
            raise ValueError(f"{scene}: no body {name} to remove")
    for pattern, repl in parts.get("regex", ()):
        text, count = re.subn(pattern, repl, text)
        if count == 0:
            raise ValueError(f"{scene}: pattern {pattern[:60]!r} matched nothing")
    for old, new in parts.get("replace", ()):
        if old not in text:
            raise ValueError(f"{scene}: cannot find {old[:60]!r} to replace")
        text = text.replace(old, new, 1)
    anchor = '    <body name="link0"'
    if anchor not in text:
        raise ValueError(f"{scene}: no robot body anchor")
    block = "\n".join(parts["bodies"]) + "\n"
    text = text.replace(anchor, f"    <!-- AFFORD-Bench task additions -->\n{block}{anchor}", 1)
    excludes = "".join(f'\n    <exclude body1="{a}" body2="{b}"/>' for a, b in parts.get("exclude", ()))
    if excludes:
        if "\n  </contact>" in text:
            text = text.replace("\n  </contact>", f"{excludes}\n  </contact>", 1)
        elif "\n</mujoco>" in text:
            text = text.replace("\n</mujoco>", f"\n  <contact>{excludes}\n  </contact>\n</mujoco>", 1)
        else:
            raise ValueError(f"{scene}: no place for the contact exclusions")
    # Objects touch each other without torsional friction; the finger pads (condim set in <default>) keep it. condim 4
    # between flat bodies, with multiccd's coplanar contacts, made the solver push a cheese slice through a patty.
    head, world = text.split("<worldbody>", 1)
    return head + "<worldbody>" + world.replace('condim="4"', 'condim="3"')


def build(scene: str) -> Path:
    out = ASSETS / f"{scene}_bench.xml"
    out.write_text(render(scene), encoding="utf-8")
    return out


def main() -> None:
    for scene in SCENE_PARTS:
        print("wrote", build(scene).relative_to(ROOT))


if __name__ == "__main__":
    main()
