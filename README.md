# Less Budget, More Memory

Code, raw results and paper source for:

> **Less Budget, More Memory: Diagnosing and Mitigating Non-Monotonic Activation Memory in PyTorch's Memory-Budget Partitioner**
> Siddharth Ajith, 2026. arXiv: *(link once posted)*

PyTorch's `torch._functorch.config.activation_memory_budget` trades recomputation for memory under `torch.compile`. We show that peak memory is **not monotone** in this budget (PyTorch issue [#197838](https://github.com/pytorch/pytorch/issues/197838), filed by the author), explain why, and evaluate two fixes:

- a post-partition **use-site rematerialization** pass (`experiments/liveness/remat_at_use.py`; a self-contained, tested
  version with a `torch.compile` hook is in `prototype/`);
- a **liveness-aware knapsack solver** (`experiments/liveness/liveness_solver.py`).

This is research-prototype code. All GPU results come from one GTX 1650 (4 GB) under WSL2, with Python 3.12, PyTorch 2.14.0+cu130 and fp32. The exact environment is in `experiments/liveness/results/environment.txt`. See the paper's limitations section before relying on any of it.

## Layout

```
ackaudit/                        minimal subset of the author's ackaudit project: the HF model builders (hf_models.py, verbatim) and the helper it imports (capture.py)
experiments/liveness/            experiment scripts
experiments/liveness/results/    raw per-run JSON files, logs, and environment.txt
prototype/                       self-contained use-site rematerialization pass, CPU tests and a CUDA demo (see prototype/README.md)
paper/                           LaTeX source, bibliography, and make_figures.py, which builds every figure and table from results/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # for the GPU runs, install the +cu130 torch wheel from the PyTorch index
export PYTHONPATH=$PWD               # makes the vendored `ackaudit` package importable
export TORCHINDUCTOR_COMPILE_THREADS=1
```

Run every command from the repository root. The GPU runs take from 30 minutes to several hours on a GTX 1650, so use `tmux`.

## Naming

| In the code | In the paper |
|---|---|
| `pointwise` mode of the remat pass | **light** mode: no matrix multiplies or attention |
| `remat(all)` | **all-ops** mode |
| `solver` | the liveness-aware selection solver, `LivenessAwareSolver` v1 (reprice only) |

## What produced each result in the paper

Every run used an explicit `--out`, as below. The scripts' default output paths differ, and some point to `/tmp`.

| Paper item | Command | Raw output in `experiments/liveness/results/` |
|---|---|---|
| Table 1 and §3.4: regime sweep of sequence length S and MLP size I, Inductor (GPU) | `bash experiments/liveness/run_mlsys_gpu.sh` (stage 1: `seqlen_sweep.py --device cuda`); `python experiments/liveness/seqlen_sweep_analyze.py experiments/liveness/results/seqlen_sweep_gpu.json` | `seqlen_sweep_gpu.json`, `seqlen_sweep_gpu_log.txt`, `seqlen_sweep_gpu_analysis.md`, `environment_mlsys.txt` |
| §3.4: the same sweep on CPU with the exact static oracle (`aot_eager`) | `python experiments/liveness/seqlen_sweep.py --device cpu`; `python experiments/liveness/seqlen_sweep_analyze.py` | `seqlen_sweep_cpu.json`, `seqlen_sweep_cpu_log.txt`, `seqlen_sweep_cpu_analysis.md` |
| §4.1 and Limitations: predicted vs measured effect of prefixes of the pass's edits (Inductor) | `bash experiments/liveness/run_mlsys_gpu.sh` (stage 2: `remat_edit_validation.py`) | `remat_edit_validation.json`, `remat_edit_validation_log.txt` |
| Table 2 and Figures 1 and 4: Llama rows, and BERT before the fix | `python experiments/liveness/gpu_eval_remat.py --suite full --out experiments/liveness/results/gpu_eval_remat_full.json` | `gpu_eval_remat_full.json`, `gpu_eval_remat_full_log.txt` |
| Table 2: BERT `aot_eager` rows (RNG fix emulated, v1) | `python experiments/liveness/gpu_eval_remat.py --suite bert --emulate-rng-fix --out experiments/liveness/results/gpu_eval_remat_bert_fixed.json` | `gpu_eval_remat_bert_fixed.json` and its log. Only the `aot_eager` rows are used (see caveats). |
| Table 2: BERT Inductor rows (RNG fix emulated, corrected remat pass) | `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python experiments/liveness/gpu_eval_remat.py --suite bert --emulate-rng-fix --backends inductor --out experiments/liveness/results/gpu_eval_remat_bert_fixed_v3.json` | `gpu_eval_remat_bert_fixed_v3.json` and its log. `gpu_eval_remat_bert_fixed_v2.json` is the same run with the earlier pass, whose random-op filter missed Inductor's `inductor_random`; §5.1 quotes it only to describe that earlier version. |
| Table 4 (work done by the pass), compile times | the same runs, from the `remat` and `solver` fields | as above |
| Table 3 (against the best stock budget), Figure 1 hollow markers, §3.2 dense grid | `python experiments/liveness/stock_grid.py --out experiments/liveness/results/stock_grid.json` (stock only, 18 Llama and 7 BERT budgets, Inductor); Table 3 is computed by `paper/make_figures.py` from it and the Table 2 runs | `stock_grid.json`, `stock_grid_log.txt` |
| Solver evaluation (§5, earlier standalone runs) | `python experiments/liveness/gpu_eval_liveness_solver.py --suite full --out experiments/liveness/results/gpu_eval_full.json` | `gpu_eval_full.json`, `gpu_eval_full_log.txt` |
| Figure 2 (left) and §3: oracle validation under `aot_eager` (14 Llama plans, closure, `KnapsackEvaluator`, MLP force-in/out, no-reorder ablation) | `python experiments/liveness/liveness_oracle_check.py --suite full --out experiments/liveness/results/expa_full.json` | `expa_full.json`, `expa_full_log.txt` |
| Figure 2 (right) and §3: oracle under Inductor (14 Llama plans and BERT; MLP force-in/out and no-reorder under Inductor) | `python experiments/liveness/experiment_c_inductor_runtime.py --part oracle --suite full --out experiments/liveness/results/expc_oracle_full.json` | `expc_oracle_full.json` and its log. `expc_oracle_bert.json` is the earlier BERT-only run (`--models bert`). |
| §3: recomputed bytes at the peak, FLOP-estimator miscalibration (§5) | `python experiments/liveness/experiment_c_inductor_runtime.py --part runtime --suite quick --out experiments/liveness/results/expc_runtime_quick.json` | `expc_runtime_quick.json`, `expc_runtime_log.txt` |
| §5 negative result: class-forcing family | `python experiments/liveness/experiment_b_liveness_select.py --suite full --out experiments/liveness/results/expb_full.json` | `expb_full.json`, `expb_full_log.txt` |
| §5 negative result: hybrid solver | Baseline: `gpu_eval_liveness_solver.py --suite quick --out .../gpu_eval_liveness_solver.json`. Hybrid: copy `liveness_solver_v2_hybrid.py` over `liveness_solver.py`, rerun with `--out .../gpu_eval_hybrid_quick.json`, then restore. | `gpu_eval_liveness_solver.json` with `gpu_eval_quick_log.txt`; `gpu_eval_hybrid_quick.json` and its log |
| §5.4: Llama gradients against eager | `python experiments/liveness/check_grads.py` | `check_grads_log.txt` |
| §5.4 and abstract: Llama gradients against eager at every budget, both backends | `for be in inductor aot_eager; do for b in 0.05 0.10 0.15 0.20 0.30; do python experiments/liveness/check_grads.py --model llama --budget $b --backend $be; done; done` | `check_grads_llama_budgets_log.txt` |
| §5.4: BERT gradients, dropout off (and the pre-fix 2.0e-2 of §6) | `python experiments/liveness/check_grad_direct.py` | `check_grad_direct_log.txt` |
| §5.4: BERT gradients, dropout on, fix emulated (corrected pass; logs print random-op counts before/after the pass) | `python experiments/liveness/check_remat_bert.py`, then `CHECK_BUDGETS=0.10,0.15 python experiments/liveness/check_remat_bert.py` | `check_remat_bert_v3_log.txt`, `check_remat_bert_b2_v3_log.txt`. The files without `_v3` are the same checks with the earlier pass; §5.4 quotes them only for the earlier version (2.6--2.8e-7, pass plans). |
| §5.4: fix-disabled control (CPU) | `CHECK_NO_FIX=1 CHECK_DEVICE=cpu CHECK_SCALE=1 CHECK_BUDGETS=0.05,0.10,0.15 python experiments/liveness/check_remat_bert.py` | `check_remat_bert_nofix_cpu_log.txt` |
| §5.4: solver failure on BERT and its bisection | `python experiments/liveness/check_solver_bert.py`; `python experiments/liveness/bisect_solver_bert.py` | `check_solver_bert_log.txt`, `bisect_solver_bert_log.txt` |
| §5.4, §6 and §8: the solver failure is PyTorch issue #198333 (wrong with Inductor's CUDA 4x random path on, correct with it off) | `python experiments/liveness/check_solver_rand4x.py` | `check_solver_rand4x_log.txt` |
| §6: RNG bug (#190758), CPU reproducers | `python experiments/liveness/repro_rng_budget.py --device cpu` on both backends; `python experiments/liveness/rng_redraw_test.py`; `repro_rng_bert_cpu.py --mode none` and `--mode v2` | `repro_rng_budget_*_cpu_log.txt`, `rng_redraw_test_cpu_log.txt`, `repro_rng_bert_cpu_*_log.txt` |
| §6: dropout recomputation counts on GPU | `python experiments/liveness/check_rng_bert.py` | `check_rng_bert_log.txt` |
| §7 (Concurrent work): reimplemented budget-free greedy save-set selector vs our solver and the use-site pass (CPU, 16-layer toy Llama, `aot_eager`, static oracle); measured-peak check of the selected plans | `python experiments/liveness/greedy_selector_race.py --model toy --greedy-modes lazy --workers 2`; `python experiments/liveness/greedy_race_analyze.py`; `python experiments/liveness/greedy_race_verify.py toy` | `greedy_race_toy.json`, `greedy_race_toy_log.txt`, `greedy_race_toy_analysis.md`, `greedy_race_verify_toy_log.txt` |
| v1 vs v2 fix emulation are equivalent under `aot_eager` | `python experiments/liveness/check_rng_fix_v1_v2_equiv.py --scale 8 --device cpu` | `rng_fix_v1_v2_equiv_log.txt` |

`experiments/liveness/rerun_lost_outputs.sh` regenerated the `expa_full`, `expc_oracle_full` and `check_grads` outputs, all the CPU logs and `environment.txt` on 2026-10-02, because the original output files of those runs had been lost. The rerun reproduced every transcribed GPU peak and every correlation exactly; float-level CPU numbers differ slightly. `results/LOGGED_OUTPUT.md` keeps the earlier transcription for comparison; the paper uses only the raw files.

To rebuild the figures, tables and PDF:

```bash
cd paper && python make_figures.py && latexmk -pdf main.tex
```

## CPU smoke tests (no GPU needed)

`check_remat_bert.py`, `check_solver_bert.py` and `bisect_solver_bert.py` accept `CHECK_DEVICE=cpu` and `CHECK_SCALE=1`, which gives a 4-layer model with sequence length 128. On that small model:

```bash
CHECK_DEVICE=cpu CHECK_SCALE=1 python experiments/liveness/check_remat_bert.py               # expect CORRECT for all plans
CHECK_NO_FIX=1 CHECK_DEVICE=cpu CHECK_SCALE=1 python experiments/liveness/check_remat_bert.py  # control: expect WRONG
python experiments/liveness/liveness_oracle_check.py --suite selftest                           # oracle plumbing: expect PASS
```

`check_solver_rand4x.py` also runs on CPU (`CHECK_DEVICE=cpu CHECK_SCALE=1`), but only as a plumbing test: CPU never takes the 4x random path, so it prints no verdict.

On the small CPU model the solver keeps the stock plan. So `check_solver_bert.py` prints `NOT REPRODUCED` and `bisect_solver_bert.py` exits with "nothing to bisect". That is expected: the failure needs the full-size GPU graph.

## Known caveats in the raw results

- **Degenerate BERT loss.** The BERT wrapper's loss, `last_hidden_state.sum()` after LayerNorm, is about 0 by construction. So `grad_rel_err` and `grads_match` in the `gpu_eval_*` JSONs are weak evidence for BERT. The paper's BERT correctness claims use only the projection-loss scripts (`check_grad_direct.py`, `check_remat_bert.py`, `check_solver_bert.py` and `bisect_solver_bert.py`).
- **"R1: FAIL" in the `gpu_eval_remat*` logs.** R1 was a pre-registered `allclose(atol=1e-4)` gradient check against stock.
  - **Llama:** it fails on float noise. Relative L2 is at most 2e-7, and `check_grads.py` shows remat is as close to eager as stock is.
  - **BERT:** it fails because of the degenerate loss and, before the fix, the RNG bug.
  - Likewise `expb_full_log.txt` prints H3 FAIL, a prediction that was wrong: the oracle found gains where we expected none. It also prints H4 FAIL: a constant BERT offset in absolute error.
- **v1 fix emulation (reconstructed).** `gpu_eval_remat_bert_fixed.json` used the first fix emulation, v1. Its Inductor rows are superseded by `..._v2.json`, because v1 misses `inductor_seeds`. The v1 file was later edited into v2 in place.
  - `rng_fix_emulation_v1.py` is a reconstruction: v2 minus the `OpTypes.is_random` patch.
  - `check_rng_fix_v1_v2_equiv.py` shows that v1 and v2 choose identical saved sets under `aot_eager`, which is why the v1 `aot_eager` rows are valid.
  - `repro_rng_bert_cpu.py --mode v1` now imports the v2 file and so behaves like v2.
- **Inflated peaks in `gpu_eval_remat.json`.** This quick run, from `gpu_eval_remat_v1.py`, cloned gradients on the GPU inside the peak loop. That inflates every peak by the parameter size. The paper uses only the corrected runs.
- **`check_grad_consistency*_log.txt`.** These are three iterations (v1, v3 and v4) of a finite-difference consistency test.
  - v1 and v3 were invalid: the step was too large, and the loss was degenerate.
  - v4's verdict was UNCLEAR.
  - The script in the repo is the last version. The paper does not use these logs; the direct per-parameter comparison in `check_grad_direct.py` replaced them.
- **Corrected remat pass (3 Oct 2026).** The first version of `remat_at_use.py` matched random ops by name and missed Inductor's `prims.inductor_random`, so under Inductor it could regenerate dropout masks near their late uses. That is unsafe on CUDA (PyTorch issue #198333). The current version excludes ops tagged `nondeterministic_seeded`, Inductor's random primitives and collectives, and asserts that the random-op count does not grow. Llama graphs contain no random ops, so every Llama result is unaffected (checked on CPU: identical plans before and after the change). The BERT `aot_eager` rows keep dropout masks as saved values, so no random op is recomputed there either. Only the BERT Inductor rows changed, and they were rerun (`_v3` files). The earlier version is quoted in the paper only in §5.1 (17--21% peak reduction, from `gpu_eval_remat_bert_fixed_v2.json`) and §5.4 (its gradient check); it supports no claim.
- **Peaks across processes.** Within one process a plan's peak is identical across repeats. Across processes the same plan can differ by up to 1.5 MB (about 0.3%): for example BERT Inductor stock at budget 0.05 measures 474.96 MB in `gpu_eval_remat_bert_fixed*.json` and `expc_oracle_bert.json` and 476.40 MB in `gpu_eval_remat_full.json` and `expc_oracle_full.json`. Every comparison in the paper is within one run.
- **Files not listed above.** `gpu_eval_remat_quick_log.txt` is the log of the inflated quick run (`gpu_eval_remat.json`, see above). `expc_oracle_bert_log.txt` is the log of `expc_oracle_bert.json`. `rng_fix_emulation.py` is the #190758 fix emulation (v2) imported by the BERT scripts; `rand4x_switch.py` is the switch imported by `check_solver_rand4x.py`.
- **Local paths removed from logs.** The author's home directory, which appeared in warning and traceback lines, was replaced by `<repo>` (the root of the working directory the runs used) or `~` in the `*_log.txt` and JSON files, and the machine's hostname in `environment.txt` by `<host>`. No number was changed.
- **First observation of #198333.** The cause was first found with a diagnostic harness in the author's working repository (github.com/siddddd17/ackaudit, `experiments/hybrid_v2/diag_rand4x.py`). `check_solver_rand4x.py` repeats the solver part of that test using only this repository's code; the paper's numbers come from its log.
- **`liveness_solver_v1.py`** is a byte-identical backup of `liveness_solver.py`. **`eval_liveness_solver.py`** and **`expc_rt_selftest.json`** are CPU development tests, not used in the paper.

## About the scripts and the history

- **Versions.** The scripts are the versions from the author's working directory after the last run. Several were extended in place between runs, by adding command-line options such as `--suite bert`, `--emulate-rng-fix` and `CHECK_BUDGETS`, and docstrings. The known behavioural changes are the ones listed under caveats: the fix emulation v1→v2, `gpu_eval_remat_v1.py`→`gpu_eval_remat.py`, and the consistency-test iterations.
- **Packaging edits.** Two edits were made when assembling this repository, both outside the paper's tables. `repro_rng_bert_cpu.py`'s main loop is wrapped in `if __name__ == "__main__":`, and `rng_redraw_test.py` gained a docstring.
- **Git history.** The history is a packaging snapshot made after the experiments, organised by component. It is not a record of when the work happened: the result files carry their own dates. "Pre-registered" criteria in script docstrings were fixed before each run, but the git history cannot show that.
- **Docstrings.** "LOCAL ONLY" in docstrings meant "not to be pushed upstream to PyTorch". Some docstrings refer to the `ackaudit` working directory or to earlier scripts that are not part of this repository.

## Related

- PyTorch issue [#197838](https://github.com/pytorch/pytorch/issues/197838): the non-monotonic budget curve.
- PyTorch issue [#190758](https://github.com/pytorch/pytorch/issues/190758) and draft PRs [#190759](https://github.com/pytorch/pytorch/pull/190759) and [#191684](https://github.com/pytorch/pytorch/pull/191684): RNG recomputation under the memory budget.

## License

MIT. See `LICENSE`.
