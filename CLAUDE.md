# CLAUDE.md — ModelEdge working rules

## Repo facts
- Default branch is `dev/modeledge-pipeline` (there is no `main`). It is the merge target, and the Kaggle notebook clones it.
- Git identity for all commits: `sakshiasati17 <sakshiasati51@gmail.com>`. Work only in `sakshiasati17/ModelEdge`.
- Run modules with `python -m <package>.<module>` from the repo root (e.g. `python -m benchmarking.run_benchmark`); plain `python path/to/file.py` breaks package imports. Exceptions: `streamlit run serving/dashboard.py`, `python serving/load_test.py`.
- Output paths default to `outputs/`; `MODELEDGE_OUTPUT_DIR` overrides (Kaggle: `/kaggle/working/outputs`).
- GPU work (training, quantization, benchmarking, vLLM) only runs on Kaggle (2x T4). Nothing GPU-bound can be verified in CI or a cloud session — say so explicitly instead of claiming it works.

## Workflow: one session = one PR
1. Branch from the latest `dev/modeledge-pipeline`.
2. Make the session's scoped changes only; no drive-by refactors.
3. Validate locally before pushing:
   - `ruff check .`
   - `pytest -q`
   - re-read the diff for anything CI or Kaggle would reject.
4. Push, open a PR into `dev/modeledge-pipeline`, wait for CI to go green.
5. Merge only when CI is green. Then start the next session from the updated default branch.
6. Never skip, disable or weaken a test to get CI green.

## Code rules
- Scoring and other pure logic must stay free of torch/transformers imports so it is testable in CI (`benchmarking/scoring.py` is the pattern).
- Every bug fix that can be tested without a GPU gets a regression test in `tests/`.
- Lint gate is pyflakes + syntax only (`pyproject.toml`); don't reformat unrelated code.
- Keep comment density and naming consistent with surrounding code.

## Honesty rules for results and docs
- Distinguish "ran", "built, not run", and "pending" — never call serving/K8s "deployed".
- Do not quote VRAM numbers from runs before benchmark isolation landed; they are known-contaminated.
- README benchmark numbers must come from an actual `results/summary.json`, with the scorer version noted.
- The dashboard falls back to demo data when `results/summary.json` is missing — never present that as real.
- Training is single-GPU (Unsloth), even on the 2x T4 node.
