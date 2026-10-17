"""Each perturbation suite changes only its own factor, and suites evaluated on the same seed are paired."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import afford_pro
from afford_pro.envs.headless import headless
from afford_pro.perturbations.engine import PerturbationConfig

TASK = "tabletop_sort_red"
SEEDS = (11, 12, 13)
MODES = ("nominal", "p1_object", "p2_spatial", "p3_visual", "p4_physical", "combined_ood")
ROOT = Path(__file__).resolve().parent.parent
# Same sampled start, different settling under changed mass/friction: millimetres. Unpaired starts differ by 5-14 cm.
SETTLE_TOL_M = 0.01


@pytest.fixture(scope="module")
def envs():
    with headless():
        made = {m: afford_pro.make(TASK, perturbation_mode=m) for m in MODES}
        yield made
        for env in made.values():
            env.close()


def _layout(env, seed: int) -> np.ndarray:
    with headless():
        env.reset(seed=seed)
    return np.concatenate([env.data.xpos[env.target_id], env.tcp_position])


def _model_arrays(env) -> dict[str, np.ndarray]:
    m = env.model
    return {"light_pos": m.light_pos.copy(), "light_diffuse": m.light_diffuse.copy(), "geom_rgba": m.geom_rgba.copy(),
            "geom_friction": m.geom_friction.copy(), "body_mass": m.body_mass.copy(), "body_inertia": m.body_inertia.copy()}


def _changed(env, base: dict[str, np.ndarray]) -> set[str]:
    return {k for k, v in _model_arrays(env).items() if not np.array_equal(v, base[k])}


@pytest.mark.parametrize("seed", SEEDS)
def test_appearance_and_physics_suites_share_the_nominal_layout(envs, seed):
    nominal = _layout(envs["nominal"], seed)
    np.testing.assert_array_equal(_layout(envs["p1_object"], seed), nominal)
    np.testing.assert_array_equal(_layout(envs["p3_visual"], seed), nominal)
    # A heavier, slipperier body settles slightly differently; the sampled start is the same.
    np.testing.assert_allclose(_layout(envs["p4_physical"], seed), nominal, atol=SETTLE_TOL_M)


@pytest.mark.parametrize("seed", SEEDS)
def test_compound_suite_shares_the_spatial_layout(envs, seed):
    np.testing.assert_allclose(_layout(envs["combined_ood"], seed), _layout(envs["p2_spatial"], seed), atol=SETTLE_TOL_M)


@pytest.mark.parametrize("seed", SEEDS)
def test_each_suite_changes_only_its_own_model_parameters(envs, seed):
    _layout(envs["nominal"], seed)
    base = _model_arrays(envs["nominal"])
    expected = {"nominal": set(), "p2_spatial": set(), "p1_object": {"geom_rgba"},
                "p3_visual": {"light_pos", "light_diffuse"}, "p4_physical": {"geom_friction", "body_mass", "body_inertia"},
                "combined_ood": {"geom_rgba", "light_pos", "light_diffuse", "geom_friction", "body_mass", "body_inertia"}}
    for mode, keys in expected.items():
        _layout(envs[mode], seed)
        assert _changed(envs[mode], base) == keys, mode


def test_physical_suite_touches_only_the_body_and_pads_within_range(envs):
    env, cfg = envs["p4_physical"], PerturbationConfig()
    _layout(envs["nominal"], SEEDS[0])
    base = _model_arrays(envs["nominal"])
    for seed in SEEDS:  # repeated resets must not compound
        _layout(env, seed)
        changed_geoms = set(np.flatnonzero(np.any(env.model.geom_friction != base["geom_friction"], axis=1)))
        assert changed_geoms <= set(env.object_geom_ids) | set(env.pad_geom_ids)
        mu = env.model.geom_friction[env.object_geom_ids[0], 0]
        assert cfg.friction_range[0] <= mu <= cfg.friction_range[1]
        changed_bodies = set(np.flatnonzero(env.model.body_mass != base["body_mass"]))
        assert changed_bodies == {env.target_id}
        ratio = env.model.body_mass[env.target_id] / base["body_mass"][env.target_id]
        assert cfg.mass_scale_range[0] <= ratio <= cfg.mass_scale_range[1]


def test_visual_suite_does_not_move_geometry(envs):
    for seed in SEEDS:
        _layout(envs["nominal"], seed)
        q_nominal = envs["nominal"].data.qpos.copy()
        _layout(envs["p3_visual"], seed)
        np.testing.assert_array_equal(envs["p3_visual"].data.qpos, q_nominal)


DATASET = ROOT / "datasets" / "industrial_bearing_press_nominal.hdf5"


@pytest.mark.skipif(not DATASET.exists(), reason="released dataset not present (local check only)")
def test_nominal_reset_still_reproduces_released_initial_states():
    with h5py.File(DATASET, "r") as f, headless():
        env = afford_pro.make("industrial_bearing_press", perturbation_mode="nominal")
        for key in ("demo_0", "demo_49"):
            demo = f["data"][key]
            env.reset(seed=int(demo.attrs["seed"]))
            np.testing.assert_array_equal(env.get_sim_state(), demo["states"][0])
        env.close()
