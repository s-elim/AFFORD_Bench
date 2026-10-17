"""Test runner for Domain 8 Frontier Complex Tasks."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from afford_pro.envs.frontier_env import FrontierComplexEnv
from afford_pro.oracles.frontier_oracle import FrontierOracle
from scripts.sanity_baselines import _disable_rendering

if __name__ == "__main__": _disable_rendering()

tasks = [
    {
        'id': 'reactive_sort_red_perturbation',
        'xml': 'tabletop_sorting.xml',
        'obj': 'red_block',
        'joint': 'red_joint',
        'nom': [0.45, -0.08, 0.025],
        'tgt': [0.40, -0.24, 0.035],
        'type': 'reactive_perturbation',
        'tol': 0.008,
        'p_step': 30,
        'p_off': [0.04, 0.0, 0.0],
    },
    {
        'id': 'reactive_shaft_seat_perturbation',
        'xml': 'industrial_assembly.xml',
        'obj': 'machined_shaft',
        'joint': 'shaft_joint',
        'nom': [0.40, 0.24, 0.050],
        'tgt': [0.55, 0.00, 0.116],
        'type': 'reactive_perturbation',
        'tol': 0.002,
        'p_step': 30,
        'p_off': [0.03, 0.0, 0.0],
    },
    {
        'id': 'reactive_mug_place_perturbation',
        'xml': 'kitchen_coffee.xml',
        'obj': 'coffee_mug',
        'joint': 'mug_joint',
        'nom': [0.38, 0.22, 0.038],
        'tgt': [0.38, -0.15, 0.038],
        'type': 'reactive_perturbation',
        'tol': 0.008,
        'p_step': 30,
        'p_off': [0.03, 0.0, 0.0],
    },
    {
        'id': 'reactive_conical_tube_perturbation',
        'xml': 'lab_science.xml',
        'obj': 'conical_tube',
        'joint': 'tube_joint',
        'nom': [0.40, 0.12, 0.06],
        'tgt': [0.52, -0.18, 0.08],
        'type': 'reactive_perturbation',
        'tol': 0.002,
        'p_step': 30,
        'p_off': [0.02, -0.02, 0.0],
    },
    {
        'id': 'reactive_double_displacement',
        'xml': 'tabletop_sorting.xml',
        'obj': 'red_block',
        'joint': 'red_joint',
        'nom': [0.45, -0.08, 0.025],
        'tgt': [0.40, -0.24, 0.035],
        'type': 'reactive_perturbation',
        'tol': 0.008,
        'p_step': 25,
        'p_off': [0.03, 0.0, 0.0],
        'p2_step': 60,
        'p2_off': [0.0, 0.03, 0.0],
    },
    {
        'id': 'nonprehensile_cube_push_and_grasp',
        'xml': 'tabletop_sorting.xml',
        'obj': 'red_block',
        'joint': 'red_joint',
        'nom': [0.45, -0.08, 0.025],
        'tgt': [0.40, -0.24, 0.035],
        'type': 'push_and_grasp',
        'tol': 0.008,
        'push': 0.03,
    },
]

print(f"{'Task ID':<35} | {'Oracle':<8} | {'Steps':<6} | {'Null':<8} | {'Random':<8}")
print("-" * 75)

for t in (tasks if __name__ == "__main__" else []):
    env = FrontierComplexEnv(
        scene_xml_name=t['xml'],
        object_name=t['obj'],
        joint_name=t['joint'],
        nominal_pos=t['nom'],
        target_pos=t['tgt'],
        instruction=t['id'],
        task_type=t['type'],
        tolerance_xy=t['tol'],
        tolerance_z=t['tol'],
        perturbation_step=t.get('p_step'),
        perturbation_offset=t.get('p_off'),
        second_perturbation_step=t.get('p2_step'),
        second_perturbation_offset=t.get('p2_off'),
        min_push_distance=t.get('push', 0.0),
    )
    
    # 1. Oracle
    oracle = FrontierOracle(env)
    actions, obs, rew, dones = oracle.generate_episode(seed=42)
    oracle_pass = env.check_success()
    steps = len(actions)
    
    # 2. Null
    env.reset(seed=42)
    for _ in range(100):
        env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    null_pass = env.check_success()
    
    # 3. Random
    env.reset(seed=42)
    rng = np.random.default_rng(42)
    for _ in range(100):
        env.step(rng.uniform(-1, 1, 7))
    rand_pass = env.check_success()
    
    print(f"{t['id']:<35} | {str(oracle_pass):<8} | {steps:<6} | {str(null_pass):<8} | {str(rand_pass):<8}")
