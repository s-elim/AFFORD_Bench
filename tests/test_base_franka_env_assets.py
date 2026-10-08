from __future__ import annotations

from pathlib import Path

import afford_pro.envs.base_franka_env as base_env


def test_load_model_with_portable_assets_falls_back_when_meshes_missing(tmp_path: Path, monkeypatch):
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text(
        """
<mujoco model="toy">
  <compiler meshdir="D:/broken/path"/>
  <asset>
    <mesh file="D:/broken/path/link0.stl"/>
    <texture type="2d" name="tex" file="D:/broken/path/cartoon_only.png"/>
  </asset>
</mujoco>
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_from_xml_path(path: str):
        raise ValueError(f"Error: Error opening file '{path}': No such file or directory")

    def fake_from_xml_string(xml: str, assets: dict[str, bytes]):
        captured["xml"] = xml
        captured["assets"] = assets
        return object()

    monkeypatch.setattr(base_env.mujoco.MjModel, "from_xml_path", staticmethod(fake_from_xml_path))
    monkeypatch.setattr(base_env.mujoco.MjModel, "from_xml_string", staticmethod(fake_from_xml_string))

    model = base_env._load_model_with_portable_assets(xml_path)

    assert model is not None
    xml_text = captured["xml"]
    assets = captured["assets"]
    assert isinstance(xml_text, str)
    assert isinstance(assets, dict)
    assert 'meshdir="."' in xml_text
    assert "D:/broken/path" not in xml_text
    assert "link0.stl" in assets
    assert "cartoon_only.png" not in assets
    assert 'builtin="flat"' in xml_text
