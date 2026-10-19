# Baseline runbook (GPU server)

Training and evaluating learned baselines for the leaderboard. All commands run from the repository root.

## 1. Environment
```bash
conda create -n afford_bench python=3.10 -y && conda activate afford_bench
pip install -e .                          # pins mujoco==3.5.0 (all reported numbers use it)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match the server's CUDA version
export MUJOCO_GL=egl                      # headless rendering for visual policies
python -m pytest -q tests                 # sanity check, a few minutes
```

## 2. Data
Copy `datasets/` (210 files, one per task, 50 oracle demonstrations each) or generate it (CPU, several hours):
```bash
python scripts/generate_all_150_tasks.py --num-demos 50
python scripts/validate_datasets.py --data-dir datasets --replay-per-file 3 --workers 16 --out-dir results/validation
```
Every file records its task set, provenance and the step-by-step instruction derived from its goals; replays must
reproduce every recorded state exactly (the validator reports any deviation as an error). Files of the deprecated
legacy task set must be moved out of `datasets/` first: the generator refuses to overwrite them.

## 3. Train (3 seeds per method)
```bash
for seed in 1 2 3; do
  python scripts/train_policy.py --policy bc --modality visual --domain all \
      --epochs 100 --batch-size 256 --device cuda --seed $seed --tag seed$seed
  python scripts/train_policy.py --policy bc --modality objects --domain all \
      --epochs 200 --device cuda --seed $seed --tag seed$seed
done
```
Modalities (`afford_pro/learning/features.py`): `visual` (proprioception + workcell image), `objects` (proprioception
+ privileged poses of the bodies the task moves, 6 slots), `state` (proprioception only, a blind ablation). For
Diffusion Policy and ACT use reference implementations (e.g. LeRobot: `python scripts/convert_to_lerobot.py --hdf5
<file> --out <dir>`). Evaluating a LeRobot policy here needs an adapter class in `afford_pro/evaluation/rollout.py`
with `reset(seed)` and `act(obs, env)`.

## 4. Evaluate (protocol settings; do not change seeds or horizon)
```bash
python scripts/evaluate_policy_suite.py --checkpoint checkpoints/bc_visual_all_seed1_best.pt \
    --suites nominal,p1_object,p2_spatial,p3_visual,p4_physical,combined_ood \
    --episodes 50 --device cuda --out-dir results/bc_visual_seed1
```
An episode succeeds when every goal of the task has been reached, in order, and holds within the task's step budget
(400 steps for one goal, plus 250 per further goal); `summary.json` reports success with task-bootstrap and Wilson
intervals per suite.

## 5. Add results to the leaderboard
Copy each `results/<method>_seed<k>/` folder (with `episodes.csv` and `summary.json`) into
`docs/roadmap/evidence/leaderboard_<YYYY-MM-DD>/`, then:
```bash
python scripts/build_site_data.py      # rebuilds docs/site/data.js; seeds of one method become one row
```
Runs of the same method are grouped by the policy name in `episodes.csv` (e.g. `bc_visual`), reported as pooled
success with a task-bootstrap interval and the mean and s.d. over training seeds. Add a description for new
methods to `POLICY` in `docs/site/app.js`.
