# Cross-Benchmark Audit & Alignment: AFFORD-Bench vs. LIBERO-PRO, LIBERO-X, and ROBOMIMIC
**Date**: October 9, 2026  
**Auditor**: Antigravity Core Benchmarking & Manipulation Team  
**Scope**: Codebases of `AFFORD-Bench`, `LIBERO-PRO`, `LIBERO-X` (Meituan Robotics), and `robomimic` (ARISE Initiative / Stanford).

---

## 1. Executive Summary

To ensure seamless drop-in compatibility for imitation learning and offline reinforcement learning algorithms across the community, we conducted a rigorous comparative audit of **AFFORD-Bench** against:
1. **`robomimic`** (the foundational offline imitation/RL framework used by robosuite and modern robot learning benchmarks)
2. **`LIBERO` & `LIBERO-PRO`** (lifelong robot learning benchmark family)
3. **`LIBERO-X`** (Meituan Robotics' large-scale multi-task and dexterous manipulation benchmark)

The audit discovered several critical schema discrepancies, attribute omissions, and convention differences between AFFORD-Bench's previous HDF5 dataset writer / environment wrappers and the canonical formats expected by these frameworks. Left unaddressed, these issues would cause silent data loading bugs or outright runtime exceptions when loading AFFORD datasets into Robomimic policies (e.g., BC, BCQ, CQL) or LIBERO policy architectures (e.g., BCTransformer, ResNet-RNN).

All identified discrepancies have been resolved, verified with unit tests, and integrated cleanly into AFFORD-Bench.

---

## 2. Identified Discrepancies & Resolutions

### Discrepancy 1: Per-Episode HDF5 Metadata Attributes
- **Finding**: In Robomimic (`robomimic/scripts/playback_dataset.py`, line 288) and LIBERO (`libero/libero/envs/dataset_utils.py`, line 266), replay and validation tools query:
  ```python
  ep_grp = f[f"data/{ep}"]
  model_file = ep_grp.attrs["model_file"]
  init_state = ep_grp.attrs["init_state"]
  ```
- **Prior State in AFFORD**: `model_file` was stored only at the top-level `f["data"].attrs`, and `init_state` was omitted from per-episode attributes.
- **Resolution**: Updated `afford_pro/datasets/hdf5_writer.py` to write both `model_file` and `init_state = states[0]` directly into `ep_grp.attrs` for every recorded demonstration.

---

### Discrepancy 2: LIBERO `problem_info` Root Attribute
- **Finding**: LIBERO benchmark loaders (`libero/libero/envs/dataset_utils.py`, line 43) require a JSON-serialized string stored in `f["data"].attrs["problem_info"]` containing:
  ```json
  {
    "problem_name": "...",
    "domain_name": "...",
    "language_instruction": "..."
  }
  ```
  as well as `f["data"].attrs["num_demos"]`.
- **Prior State in AFFORD**: Root attributes included custom metadata keys (`benchmark`, `total`, `env_name`) but lacked `problem_info` and `num_demos`.
- **Resolution**: Added `problem_info` JSON encoding and `num_demos` to `f["data"].attrs` in `RobomimicHDF5Writer.close()`.

---

### Discrepancy 3: Offline RL `next_obs` Transitions
- **Finding**: In Robomimic, `SequenceDataset` defaults to `load_next_obs=True` whenever training transition-based offline RL algorithms (such as BCQ, CQL, TD3-BC, or model-based RL). If `next_obs` is missing from an HDF5 demo group, dataset parsing raises `KeyError: 'data/demo_i/next_obs'`.
- **Prior State in AFFORD**: Only `obs` (observations at step $t$) was recorded under `f["data/demo_i/obs"]`.
- **Resolution**: Implemented generation and storage of `f["data/demo_i/next_obs"]` in `RobomimicHDF5Writer`, populated with transitions $s_{t+1}$ and boundary replication for the terminal state.

---

### Discrepancy 4: Proprioceptive State Dimensions & Canonical Aliasing
- **Finding**:
  1. **6-D vs 7-D `ee_states`**: LIBERO policy definitions (e.g., `BCTransformerPolicy`) expect `ee_states` to be exactly 6-dimensional:
     $$\text{ee\_states} = [\text{pos}_x, \text{pos}_y, \text{pos}_z, \text{axis\_angle}_x, \text{axis\_angle}_y, \text{axis\_angle}_z]$$
     AFFORD previously aliased `tcp_pose` (7-D: position + quaternion $[x, y, z, q_w, q_x, q_y, q_z]$) to `ee_states`, causing shape mismatch crashes during tensor forward passes.
  2. **`gripper_states`**: LIBERO models ingest `gripper_states` as a 2-D array of finger positions. AFFORD lacked this alias.
  3. **Quaternion Convention**: Robomimic canonical key `robot0_eef_quat` expects scalar-last format $[q_x, q_y, q_z, q_w]$, whereas MuJoCo natively returns scalar-first $[q_w, q_x, q_y, q_z]$.
- **Resolution**:
  - Implemented vectorized `_quat_to_axis_angle` conversion in `RobomimicHDF5Writer`.
  - Constructed true 6-D `ee_states` concatenated from 3-D `ee_pos` and 3-D `ee_ori`.
  - Added 2-D `gripper_states` mapped from `gripper_pos` / `gripper_qpos`.
  - Added `robot0_eef_quat` with scalar-last $[q_x, q_y, q_z, q_w]$ ordering.
  - Retained all native AFFORD keys for backward compatibility.

---

### Discrepancy 5: Gym / Gymnasium Registration for `EnvGym`
- **Finding**: Robomimic's environment creation utility `create_env_from_metadata` dynamically resolves environment names via `EnvGym(env_name)`, which invokes `gym.make(env_name)` (or `gymnasium.make`).
- **Prior State in AFFORD**: Environments were instantiated via custom registry functions (`afford_pro.registry.make_env(env_id)`), but were not registered in `gymnasium.envs.registration` or legacy `gym.envs.registration`.
- **Resolution**: In `afford_pro/registry.py`, added auto-registration of all 240 benchmark tasks into both `gymnasium` and `gym` under standard naming formats:
  - `afford-v0`
  - `Afford-{task_id}-v0`
  - `Afford-Task{task_id}-v0`

---

### Discrepancy 6: Gripper Action Convention Inversion
- **Finding**: Robomimic, Robosuite, LIBERO, and OpenVLA define gripper action as:
  - `-1.0` = Open
  - `+1.0` = Close  
  Meanwhile, default Franka Emika Panda controllers in AFFORD-Bench defined:
  - `+1.0` = Open
  - `-1.0` = Close
- **Resolution**: Added `invert_gripper_action: bool = False` configuration flag to `BaseFrankaEnv`. When evaluating models trained on Robomimic/LIBERO datasets, setting `invert_gripper_action=True` automatically maps actions without modifying policy code.

---

### Discrepancy 7: Task Domain Scalability in CLI Tools
- **Finding**: Evaluation and data export scripts (`export_initial_states.py`, `evaluate_id.py`, `evaluate_ood.py`, `generate_all_150_tasks.py`) previously hardcoded domain IDs up to 7 and task bounds up to 210, missing Domain 8 ("frontier", Tasks 221–240).
- **Resolution**: Updated `DOMAIN_MAP` to include `"frontier": "Frontier Complex"` and extended task ID ranges to cover up to Task 240.

---

## 3. Comparison Matrix

| Property / Feature | Robomimic | LIBERO / LIBERO-X | AFFORD-Bench (Before) | AFFORD-Bench (Audited & Aligned) |
| :--- | :--- | :--- | :--- | :--- |
| **HDF5 Demo Root** | `data/demo_i` | `data/demo_i` | `data/demo_i` | `data/demo_i` |
| **Episode Model File** | `attrs["model_file"]` | `attrs["model_file"]` | Missing on demo | `attrs["model_file"]` |
| **Episode Init State** | `attrs["init_state"]` | `attrs["init_state"]` | Missing | `attrs["init_state"]` |
| **`problem_info` Root** | Optional | Required JSON string | Missing | Stored in `f["data"].attrs` |
| **`next_obs` Stored** | Required for Offline RL | Optional | Missing | Stored for all demos |
| **`ee_states`** | 6-D $[p, \omega]$ | 6-D $[p, \omega]$ | 7-D $[p, q]$ (Bug) | 6-D $[p, \omega]$ |
| **`gripper_states`** | 2-D | 2-D | Missing alias | 2-D |
| **`robot0_eef_quat`** | 4-D $[x,y,z,w]$ | N/A | Missing | 4-D $[x,y,z,w]$ |
| **Gym Registration** | `gym.make(...)` | Custom / Gym | Custom only | Dual Gymnasium & Gym |
| **Gripper Inversion** | $-1$ open, $+1$ close | $-1$ open, $+1$ close | $+1$ open, $-1$ close | Parameterized (`invert_gripper_action`) |
| **Domain Coverage** | Varied | 1–5 suites | Domains 1–7 | Domains 1–8 (Tasks 1–240) |

---

## 4. Verification & Testing

1. **Unit Test Suite**:
   - `tests/test_dataset_recording.py` passed all 6 test suites including:
     - Multi-episode HDF5 writing and episode attribute verification.
     - `next_obs` step alignment and terminal boundary matching.
     - `ee_states` (6-D) axis-angle correctness and `gripper_states` (2-D) integrity.
     - Robomimic / robosuite canonical camera and proprioceptive key presence.
     - `gymnasium` and `gym` registration verification.
     - `invert_gripper_action` actuation verification.
2. **Regression Testing**:
   - Full test suite across all 240 environments executed to ensure zero regressions across physics contacts, constraints, and reward functions.
