"""Recorded oracle episodes replay bit-identically from their recorded initial state (no hidden inputs)."""
from __future__ import annotations

import numpy as np
import pytest

import afford_pro
from afford_pro.envs.headless import headless


@pytest.mark.parametrize("task_id,seed", [("tabletop_red_touch", 7), ("tabletop_sort_red", 9000)])
def test_oracle_episode_replays_bit_identically(task_id, seed):
    with headless():
        env = afford_pro.make(task_id)
        oracle = afford_pro.get_oracle(task_id, env)
        actions, _, _, dones = oracle.generate_episode(seed=seed)
        recorded = list(oracle.last_states) + [oracle.last_final_state]
        assert all(a.dtype == np.float32 for a in actions)
        env.reset(seed=seed)
        env.set_sim_state(recorded[0])
        for t, action in enumerate(actions):
            _, _, terminated, _, _ = env.step(action)
            np.testing.assert_array_equal(env.get_sim_state(), recorded[t + 1])
            assert bool(terminated) == bool(dones[t])
        env.close()
