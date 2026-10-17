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
Use the regenerated dataset (`datasets_v2/`) once task program T9 is finished; until then, `datasets/` holds the
current files (447 GB; limitations in `docs/DATASET_CARD.md`). Validate after copying:
```bash
python scripts/validate_datasets.py --data-dir datasets --workers 16 --out-dir results/validation
```

## 3. Train (3 seeds per method)
```bash
for seed in 1 2 3; do
  python scripts/train_policy.py --policy bc --modality lowdim --domain all --tier core \
      --epochs 200 --device cuda --seed $seed --tag seed$seed
  python scripts/train_policy.py --policy bc --modality visual --domain all --tier core \
      --epochs 100 --batch-size 256 --device cuda --seed $seed --tag seed$seed
done
```
The in-repo diffusion model did not learn in a one-task check (`docs/roadmap/EXPERIMENT_LOG.md`, E-011); for
Diffusion Policy and ACT use reference implementations (e.g. LeRobot: `python scripts/convert_to_lerobot.py --hdf5
<file> --out <dir>`). Evaluating a LeRobot policy in this benchmark needs an adapter class in
`afford_pro/evaluation/rollout.py` with `reset(seed)` and `act(obs, env)`; it is not written yet.

## 4. Evaluate (protocol settings; do not change seeds or horizon)
```bash
python scripts/evaluate_policy_suite.py --checkpoint checkpoints/bc_lowdim_all_core_seed1_best.pt \
    --tier core --suites nominal,p2_spatial,p3_visual,p4_physical,combined_ood,p1_object \
    --episodes 50 --device cuda --out-dir results/bc_lowdim_seed1
```

## 5. Add results to the leaderboard
Copy each `results/<method>_seed<k>/` folder (with `episodes.csv` and `summary.json`) into
`docs/roadmap/evidence/leaderboard_<YYYY-MM-DD>/`, then:
```bash
python scripts/build_site_data.py      # rebuilds docs/site/data.js; seeds of one method become one row
```
Runs of the same method are grouped by the policy name in `episodes.csv` (e.g. `bc_lowdim`), reported as pooled
success with a task-bootstrap interval and the mean and s.d. over training seeds. Add a description for new
methods to `POLICY` in `docs/site/app.js`.
