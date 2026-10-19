"""AFFORD-Bench tasks: the task specifications agree with their scenes, goals mean what their instructions say, the oracle
solves a task in every scene, and datasets are written, validated and replayed exactly."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import afford_pro
import afford_pro.envs.base_franka_env as base_env
from afford_pro.datasets.generation import GenerationConfig, generate_task_dataset
from afford_pro.datasets.replay import replay_file
from afford_pro.datasets.validation import build_task_context, file_task_set, validate_file
from afford_pro.envs.headless import headless
from afford_pro.learning.features import OBJECT_POSE_DIM, env_inputs, input_dim
from afford_pro.bench import get_spec, make_env, make_oracle
from afford_pro.bench.env import GoalFrankaEnv
from afford_pro.bench.spec import TaskSpec, step_budget
from afford_pro.bench.tasks import TASKS, static_problems

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_SEEDS = 600_000  # the seed block of scripts/validate_bench_tasks.py


def _build_module():
    spec = importlib.util.spec_from_file_location("build_bench_scenes", ROOT / "scripts" / "build_bench_scenes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_has_210_consistent_tasks():
    assert [t.number for t in TASKS] == list(range(1, 211))
    assert static_problems() == {}


def test_every_task_records_the_v1_task_it_replaces():
    for spec in TASKS:
        assert spec.legacy_task_id and spec.legacy_instruction, spec.task_id


def test_scene_xmls_are_the_output_of_their_build_script():
    build = _build_module()
    for scene in build.SCENE_PARTS:
        on_disk = (ROOT / "afford_pro" / "assets" / f"{scene}_bench.xml").read_text(encoding="utf-8")
        assert build.render(scene) == on_disk, f"{scene}_bench.xml is stale: run scripts/build_bench_scenes.py"


def test_step_budget_grows_with_the_number_of_goals():
    one, three = get_spec(1), get_spec(121)
    assert step_budget(one) == 400 and step_budget(three) == 400 + 250 * 2


def test_make_dispatches_on_version():
    with headless():
        env = afford_pro.make("kitchen_cheese_on_patty", task_set="bench")
        try:
            assert isinstance(env, GoalFrankaEnv)
            assert type(afford_pro.get_oracle("kitchen_cheese_on_patty", env)).__name__ == "GoalOracle"
        finally:
            env.close()
        legacy = afford_pro.make("tabletop_sort_red", task_set="legacy")
        assert not isinstance(legacy, GoalFrankaEnv)
        legacy.close()
    with pytest.raises(ValueError):
        afford_pro.make("tabletop_sort_red", task_set="v3")
    assert afford_pro.get_task_entry("kitchen_cheese_on_patty", "bench").task_number == 32


# One task per scene; each is solved by the oracle on its first validation seed.
@pytest.mark.parametrize("number", [1, 7, 32, 36, 92, 81])
def test_oracle_solves_a_task_in_every_scene(number):
    with headless():
        env = make_env(number)
        try:
            _, _, _, dones = make_oracle(env).generate_episode(seed=VALIDATION_SEEDS + 1000 * number)
            assert dones and dones[-1] and all(env.goal_status())
        finally:
            env.close()


@pytest.mark.parametrize("number", [61, 121, 127, 158, 210])
def test_no_goal_holds_at_reset(number):
    with headless():
        env = make_env(number)
        try:
            env.reset(seed=VALIDATION_SEEDS + 1000 * number)
            free = [i for i in range(len(env.task_spec.goals)) if not env._predecessors(i)]
            assert not any(env.goal_holds(i) for i in free)
            assert not env.check_success()
        finally:
            env.close()


def test_a_goal_only_counts_after_the_goals_ordered_before_it():
    # Task 145 ends by putting the micro tube back in the rack where it starts: that final goal holds at the start but
    # must not count until the tube has been in the centrifuge and the start button pressed.
    with headless():
        env = make_env(145)
        try:
            env.reset(seed=VALIDATION_SEEDS + 145_000)
            env.step(np.zeros(7, dtype=np.float32))
            env.peak_z["micro_tube"] = env.start_z_of["micro_tube"] + 0.05  # as if it had been lifted and put back
            assert env._predecessors(2) == [1] and env.goal_holds(2)
            env.step(np.zeros(7, dtype=np.float32))
            assert 2 not in env.first_true and not env.check_success()
        finally:
            env.close()


def test_region_goal_accepts_any_pose_inside_the_region():
    with headless():
        env = make_env(61)  # red block in the red bin
        try:
            env.reset(seed=1)
            goal = env.task_spec.goals[0]
            center = env.region_center(goal)
            tol = env.on_tolerances(goal)[0]
            rest = env.goal_point(goal)
            # Off-center but inside the bin, then on the table beside the bin.
            for pos, inside in (((center[0] + 0.5 * tol, center[1], rest[2]), True),
                                ((center[0] + 0.10, center[1], 0.021), False)):
                env.pin_free_body(env.body_qposadr["red_block"], np.array(pos))
                env._settle()
                env.peak_z["red_block"] = env.start_z_of["red_block"] + 0.05  # it was lifted
                assert env.goal_holds(0) is inside
        finally:
            env.close()


def test_latching_button_keeps_its_state():
    with headless():
        env = make_env(23)  # emergency stop
        try:
            env.reset(seed=1)
            for depth, pressed in ((0.0, False), (0.009, True)):
                env.data.qpos[env.button_adr["estop"]] = depth
                for _ in range(10):
                    env.step(np.zeros(7, dtype=np.float32))
                assert env.goal_holds(0) is pressed
        finally:
            env.close()


@pytest.mark.parametrize("offset", [(-0.12, 0.08, -0.08), (-0.12, 0.08, 0.04), (0.20, 0.15, -0.30)])
def test_empty_hand_travel_never_descends_before_it_is_above_the_goal(offset):
    # Task 172: the start button stands next to the seated lid; a diagonal descent swept the closing fingers into it.
    # From high above (the ready pose) the hand comes down to TRAVEL_CLEARANCE above the goal first: crossing at full
    # height left the arm stretched over a far button, and the descent from there flipped the IK solution.
    from afford_pro.bench.oracle import TRAVEL_CLEARANCE

    path, ends = [], {}
    with headless():
        env = make_env(172)
        try:
            oracle = make_oracle(env)
            step = env.step

            def traced(action):
                out = step(action)
                path.append(env.tcp_position.copy())
                return out

            def run():
                start = env.tcp_position.copy()
                ends["start"], ends["goal"] = start, start + np.array(offset)
                oracle._travel(ends["goal"], 1.0)

            env.step, oracle.run = traced, run
            oracle.generate_episode(seed=1)
        finally:
            env.close()
    path, goal = np.array(path), ends["goal"]
    away = np.linalg.norm(path[:, :2] - goal[:2], axis=1) > 0.01
    assert np.linalg.norm(path[-1] - goal) < 0.006
    # The arm overshoots a fast crossing by up to 3 cm before it settles; the diagonal descent this replaced dropped
    # 8 cm below the crossing height while still beside the goal.
    assert path[away, 2].min() > min(ends["start"][2], goal[2] + TRAVEL_CLEARANCE) - 0.03


def test_a_body_thrown_into_a_bin_counts_in_any_pose():
    # "Throw the sponge into the waste bin": lying on its side inside the bin is in the bin; on the bench it is not.
    with headless():
        env = make_env(110)
        try:
            env.reset(seed=1)
            goal = env.task_spec.goals[0]
            center = env.region_center(goal)
            on_side = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # 90 deg about x
            for xy, inside in ((center[:2], True), (center[:2] + np.array([0.0, 0.15]), False)):
                env.pin_free_body(env.body_qposadr["sponge"], np.array([xy[0], xy[1], 0.12]), on_side)
                env._settle(max_substeps=1500)
                env.peak_z["sponge"] = env.start_z_of["sponge"] + 0.10  # it was lifted
                assert env.body_tilt("sponge") > np.radians(60)
                assert env.goal_holds(0) is inside, env.goal_report(0)
        finally:
            env.close()


def test_objects_touch_each_other_without_torsional_friction():
    # condim 4 between two flat bodies, with the several coplanar contacts of multiccd, made the solver push a cheese
    # slice through a patty; the fingers keep condim 4, so grasps still resist twisting.
    import mujoco
    from afford_pro.bench.scenes import SCENES

    for name, scene in SCENES.items():
        model = mujoco.MjModel.from_xml_path(str(ROOT / "afford_pro" / "assets" / scene.xml))
        free = {model.jnt_bodyid[j] for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
        colliding = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
        assert all(model.geom_condim[g] <= 3 for g in colliding if model.geom_bodyid[g] in free), name
        for finger in ("left_finger", "right_finger"):  # its pads
            assert any(model.geom_condim[g] == 4 for g in colliding if model.geom_bodyid[g] == model.body(finger).id)


THIN_LAYER_HALF = 0.0035  # a collision layer at most this half-thick inside another one counts as thin (m)


def _collision_layers(model, body: str) -> list[tuple[float, float, float, np.ndarray, float]]:
    """(bottom, top, half-height, xy centre, footprint radius) of each colliding cylinder or box of a body, in its
    frame (unrotated geoms: every decorative layer in the scenes is axis-aligned)."""
    import mujoco
    bid = model.body(body).id
    layers = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if abs(model.geom_quat[g][0]) < 0.999:  # rotated: its half-height is not along z
            continue
        kind = model.geom_type[g]
        size = model.geom_size[g]
        half, radius = {int(mujoco.mjtGeom.mjGEOM_CYLINDER): (size[1], size[0]),
                        int(mujoco.mjtGeom.mjGEOM_BOX): (size[2], max(size[0], size[1]))}.get(int(kind), (None, None))
        if half is not None:
            z = model.geom_pos[g][2]
            layers.append((z - half, z + half, half, model.geom_pos[g][:2].copy(), radius))
    return layers


# Thin layers that are part of how a body works; nothing is set down on them.
THIN_LAYERS_BY_DESIGN = {
    ("kitchen_coffee", "coffee_pod"), ("kitchen_coffee", "red_capsule"),  # the lip a capsule hangs by in its slot
    ("lab_science", "micro_tube"),  # the cap the tube hangs by in a rack hole
    ("kitchen_coffee", "coffee_mug"),  # the coffee surface inside the mug
    ("kitchen_coffee", "saucer"),  # the raised rim around the saucer's well
    ("lab_science", "microplate_96"),  # the well grid, only ever pipetted over
    ("kitchen_burger", "bun_caddy"), ("kitchen_burger", "topping_tray"),  # dividers that coincide with a wall
    ("kitchen_burger", "assembly_board"),  # a border strip under the board
}


def test_no_thin_collision_layer_lies_inside_another():
    # A collision layer a few mm thick inside a body flips its contact normal once an item sinks past its midplane, and
    # pushes the item into the body (a tomato slice sank 7 mm into the bottom bun, a bun 20 mm into the plate).
    import mujoco
    from afford_pro.bench.scenes import SCENES

    thin = []
    for scene, spec in SCENES.items():
        model = mujoco.MjModel.from_xml_path(str(ROOT / "afford_pro" / "assets" / spec.xml))
        robot = model.body("link0").id
        for b in range(1, model.nbody):
            body = model.body(b).name
            if model.body_rootid[b] == robot or (scene, body) in THIN_LAYERS_BY_DESIGN:
                continue
            layers = _collision_layers(model, body)
            for k, (lo, hi, half, xy, _) in enumerate(layers):
                inside = any(o_lo < hi - 1e-4 and lo + 1e-4 < o_hi and np.linalg.norm(xy - o_xy) < o_radius
                             for m, (o_lo, o_hi, _, o_xy, o_radius) in enumerate(layers) if m != k)
                if half <= THIN_LAYER_HALF and inside:
                    thin.append(f"{scene}/{body}")
    assert not thin, sorted(set(thin))


def test_objects_modality_carries_every_moved_body():
    with headless():
        env = make_env(125)  # four bodies stacked on the bun
        try:
            obs, _ = env.reset(seed=1)
            vec, image = env_inputs(obs, env, "objects")
            assert vec.shape == (input_dim("objects"),) and image is None
            moved = len(env.task_spec.bodies_moved())
            poses = vec[22:].reshape(-1, OBJECT_POSE_DIM)
            assert np.all(np.abs(poses[:moved]).sum(axis=1) > 0) and np.all(poses[moved:] == 0)
        finally:
            env.close()


class _GradientRenderer:
    """Headless stand-in for mujoco.Renderer with a non-constant image (image-content checks need one)."""

    def __init__(self, model=None, height: int = 64, width: int = 64, *args, **kwargs) -> None:
        self.image = np.broadcast_to(np.linspace(20, 220, width, dtype=np.uint8)[None, :, None], (height, width, 3)).copy()

    def update_scene(self, *args, **kwargs) -> None:
        pass

    def render(self) -> np.ndarray:
        return self.image.copy()

    def close(self) -> None:
        pass


def test_dataset_is_written_validated_and_replayed_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(base_env.mujoco, "Renderer", _GradientRenderer)
    task = "kitchen_cheese_on_patty"
    result = generate_task_dataset(GenerationConfig(task_id=task, output_dir=tmp_path, num_demos=2, task_set="bench"),
                                   log=None)
    assert result.status == "written" and result.validation_status in ("VALID", "VALID_WITH_WARNINGS")
    path = Path(result.path)
    assert file_task_set(path) == "bench"
    assert validate_file(path, 2, image_frames=(0,)).status != "INVALID"
    ctx = build_task_context(task, "nominal", "bench")
    rows = replay_file(path, ["demo_0", "demo_1"], ctx.qpos_offset, ctx.object_qpos_adr, ctx.arm_qpos_adr)
    assert all(r["consistent"] and r["max_qpos_dev"] == 0.0 and r["max_joint_dev_rad"] == 0.0 for r in rows)
    with h5py.File(path, "r") as f:  # task language, not the legacy prompts for a reused task id
        assert json.loads(f["data"].attrs["language_instructions"]) == [get_spec(task).instruction]
        assert json.loads(f["data"].attrs["prompt_variants"])["step_by_step"].startswith("Put the cheese slice")


def test_every_goal_has_wording():
    from afford_pro.bench.language import missing_names

    assert missing_names() == []


def test_spec_problems_are_reported():
    from afford_pro.bench.spec import On, Pressed
    from afford_pro.bench.tasks import _spec_problems

    bad = TaskSpec(1, "bad", "D1", "industrial_assembly", "No period",
                   (On("hex_nut", "no_such_target"), Pressed("no_button")), order=((0, 5),))
    found = " | ".join(_spec_problems(bad))
    for fragment in ("no_such_target", "no_button", "missing goal", "period"):
        assert fragment in found


def test_a_slot_start_that_includes_the_goal_slot_is_reported():
    # The micro tube's home slots are rack_a_1 and rack_b_micro: with rack_b_micro as the goal, every start that
    # satisfies it is rejected, so the tube would always start in rack_a_1.
    from afford_pro.bench.spec import On, Slots
    from afford_pro.bench.tasks import _spec_problems

    fixed = TaskSpec(1, "fixed", "D4", "lab_science", "Put the micro tube in the orange rack.",
                     (On("micro_tube", "rack_b_micro"),))
    varied = TaskSpec(1, "varied", "D4", "lab_science", "Put the micro tube in the orange rack.",
                      (On("micro_tube", "rack_b_micro"),),
                      starts={"micro_tube": Slots(("rack_a_1", "centrifuge_slot"))})

    put_back = TaskSpec(1, "put_back", "D4", "lab_science", "Take the micro tube out, then put it back.",
                        (On("micro_tube", "pipette_pad", once=True), On("micro_tube", "rack_b_micro")), order=((0, 1),))

    assert any("one usable slot" in p for p in _spec_problems(fixed))
    assert not _spec_problems(varied)
    assert not _spec_problems(put_back)


def _place_capsule_in_slot(env) -> None:
    """Put the gold capsule at its calibrated rest pose in the capsule slot, as if it had been carried there."""
    goal = env.task_spec.goals[0]
    env.pin_free_body(env.body_qposadr["coffee_pod"], env.goal_point(goal))
    env.peak_z["coffee_pod"] = env.start_z_of["coffee_pod"] + 0.10


@pytest.mark.parametrize("brew_first", [True, False])
def test_a_latched_goal_reached_out_of_order_does_not_count(brew_first):
    # Task 128: "Put the gold coffee capsule into the capsule slot, then press the brew button."
    zero = np.zeros(7, dtype=np.float32)
    with headless():
        env = make_env(128)
        try:
            env.reset(seed=3)
            steps = [lambda: env.data.qpos.__setitem__(env.button_adr["brew"], 0.010), lambda: _place_capsule_in_slot(env)]
            for act in (steps if brew_first else steps[::-1]):
                act()
                for _ in range(15):
                    env.step(zero)
            assert all(env.goal_status())
            assert env.check_success() is (not brew_first)
        finally:
            env.close()


def test_generation_refuses_a_legacy_file_with_the_same_name(tmp_path):
    task = "tabletop_sort_red"  # a task id shared with the legacy set
    with h5py.File(tmp_path / f"{task}_nominal.hdf5", "w") as f:
        f.create_group("data").attrs["env_args"] = json.dumps({"env_name": task, "env_kwargs": {}})
    result = generate_task_dataset(GenerationConfig(task_id=task, output_dir=tmp_path, num_demos=1, task_set="bench"),
                                   log=None)
    assert result.status == "refused_existing" and "legacy" in result.validation_status


def test_an_unreadable_version_record_is_an_error(tmp_path):
    path = tmp_path / "x_nominal.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data").attrs["env_args"] = "{not json"
    with pytest.raises(ValueError):
        file_task_set(path)
