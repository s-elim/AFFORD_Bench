# Third-party notice: Franka Emika Panda model

The mesh files in `assets/` (67 `.obj`/`.stl` files) are taken from the `franka_emika_panda` model of
MuJoCo Menagerie (https://github.com/google-deepmind/mujoco_menagerie), which is distributed under the
Apache License, Version 2.0 (full text in `LICENSE` in this folder). Menagerie's Panda model is itself derived from
`franka_description` in `franka_ros` by Franka Emika GmbH.

Modifications (Apache-2.0 section 4(b)): three sampled meshes (`hand.stl`, `finger_0.obj`, `link0_0.obj`) are
byte-identical to upstream apart from line endings; the others were not compared. The robot definition inside each
scene file in `afford_pro/assets/*.xml` is Menagerie's `panda.xml` (main branch, retrieved 2026-10-18) with these
changes, found by an element-wise diff of `industrial_assembly.xml`:
- `gravcomp="1"` on all 11 robot bodies;
- a `wrist_cam` camera added to the `hand` body;
- arm position servos with lower gains (actuators 1-2: kp 1500 / kv 400 instead of 4500 / 450; 3-4: 1200 / 300
  instead of 3500 / 350; 5-7: 700 / 180 instead of 2000 / 200);
- gripper actuator `actuator8`: forcerange +-70 instead of +-100, gain 0.62745098 and bias (0, -4000, -40) instead of
  0.01568627451 and (0, -100, -10);
- finger-pad collision classes `fingertip_pad_collision_1..5`: friction (1.5, 0.005, 0.0001), condim 4,
  solref (0.004, 1), solimp (0.95, 0.99, 0.0005) added;
- simulation option: elliptic friction cones with impratio 10 (upstream: defaults).

Citation: Kevin Zakka, Yuval Tassa, and MuJoCo Menagerie Contributors. MuJoCo Menagerie: A collection of
high-quality simulation models for MuJoCo, 2022.

This notice covers only the robot meshes and model; the rest of AFFORD-Bench is MIT-licensed (top-level `LICENSE`).
The origin of `assets/textures/cartoon_only.png` is not recorded.
