"""Statistics, evaluation bookkeeping, and the training/evaluation feature path agree with their definitions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from afford_pro.datasets.validation import build_task_context
from afford_pro.evaluation.rollout import run_policy_episode
from afford_pro.evaluation.stats import cluster_bootstrap_ci, summarize, wilson_ci
from afford_pro.learning.features import Normalizer, env_inputs, object_pose_from_states, proprio

ROOT = Path(__file__).resolve().parent.parent


def _suite_module():
    spec = importlib.util.spec_from_file_location("evaluate_policy_suite", ROOT / "scripts" / "evaluate_policy_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wilson_known_values():
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson_ci(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.4345, abs=1e-3)
    assert all(np.isnan(wilson_ci(0, 0)))


def test_cluster_bootstrap_is_degenerate_for_identical_tasks_and_seeded():
    assert cluster_bootstrap_ci([[True] * 5] * 4) == (1.0, 1.0)
    clusters = [[True, False, True], [False, False], [True, True, True, True]]
    assert cluster_bootstrap_ci(clusters, seed=3) == cluster_bootstrap_ci(clusters, seed=3)
    s = summarize(clusters)
    assert s["successes"] == 6 and s["episodes"] == 9 and s["tasks"] == 3


def _record(task: str, suite: str, success: bool) -> dict:
    return {"task_id": task, "suite": suite, "success": success, "force_violation": False}


def test_pri_and_gdi_on_paired_suites():
    records = ([_record("a", "nominal", s) for s in (True, True, False, False)]
               + [_record("a", "p4_physical", s) for s in (True, False, False, False)])
    out = _suite_module().summarize_records(records)
    assert out["p4_physical"]["PRI_pct"] == pytest.approx(50.0)
    assert out["p4_physical"]["GDI_pct"] == pytest.approx(50.0)


def test_pri_undefined_when_nominal_success_is_zero():
    records = [_record("a", "nominal", False)] * 3 + [_record("a", "p3_visual", True)] * 3
    out = _suite_module().summarize_records(records)
    assert "PRI_pct" not in out["p3_visual"]


def test_normalizer_round_trip_and_identity_fallback():
    x = np.array([[1.0, 5.0], [3.0, 5.0]])
    norm = Normalizer.fit(x)
    np.testing.assert_allclose(norm(x)[:, 0], [-1.0, 1.0], atol=1e-6)
    assert norm.std[1] > 0  # constant column: floored, not divided by zero
    back = Normalizer.from_dict(norm.to_dict(), 2)
    np.testing.assert_array_equal(back(x), norm(x))
    np.testing.assert_array_equal(Normalizer.from_dict(None, 2)(x), x.astype(np.float32))


class _CountdownEnv:
    """Minimal env: reports success at step `k`, never truncates."""

    def __init__(self, k: int):
        self.k, self.t = k, 0

    def reset(self, seed=None):
        self.t = 0
        return {}, {}

    def step(self, action):
        self.t += 1
        return {}, 0.0, self.t >= self.k, False, {"robot_scene_force": 0.0}


class _ZeroPolicy:
    name = "zero"

    def reset(self, seed):
        pass

    def act(self, obs, env):
        return np.zeros(7, dtype=np.float32)


def test_success_counts_only_within_the_horizon():
    hit = run_policy_episode(_CountdownEnv(5), _ZeroPolicy(), "t", "nominal", 0, horizon=5)
    miss = run_policy_episode(_CountdownEnv(5), _ZeroPolicy(), "t", "nominal", 0, horizon=4)
    assert hit.success and hit.first_success_step == 5 and hit.steps == 5
    assert not miss.success and miss.first_success_step is None and miss.steps == 4


def test_padded_chunks_cover_every_step():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("train_policy", ROOT / "scripts" / "train_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    chunks = module._padded_chunks(actions, 4)
    assert chunks.shape == (3, 4, 7)
    np.testing.assert_array_equal(chunks[0, :3], actions)
    np.testing.assert_array_equal(chunks[2], np.repeat(actions[-1:], 4, axis=0))


def test_live_and_recorded_lowdim_features_agree():
    import afford_pro
    from afford_pro.envs.headless import headless

    task = "tabletop_sort_red"
    ctx = build_task_context(task, "nominal")
    with headless():
        env = afford_pro.make(task)
        obs, _ = env.reset(seed=5)
        live, image = env_inputs(obs, env, "lowdim")
        state = env.get_sim_state()
        env.close()
    recorded_obs = {k: np.asarray(v)[None] for k, v in obs.items()}
    recorded = np.concatenate([proprio(recorded_obs, slice(None)),
                               object_pose_from_states(state[None], ctx.qpos_offset, ctx.object_qpos_adr)], axis=-1)[0]
    assert image is None and live.shape == (29,)
    np.testing.assert_allclose(live, recorded, rtol=0, atol=1e-6)
