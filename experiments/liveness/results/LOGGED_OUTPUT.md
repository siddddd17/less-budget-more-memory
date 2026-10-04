# Results recorded from terminal output (SUPERSEDED)

> **Superseded on 2026-10-02.** These experiments were rerun with `experiments/liveness/rerun_lost_outputs.sh`; the raw outputs are `expa_full.json`, `expc_oracle_full.json`, `check_grads_log.txt` and the `*_cpu_log.txt` files. The rerun reproduced every GPU peak and correlation below exactly. `check_grads` and the CPU RNG numbers differ at float level (for example 3.33e-7 instead of 3.37e-7). The paper uses only the raw files. This transcription is kept to show that it matched.

Some runs printed their results only to the terminal, or saved them to a temporary directory that was later lost when WSL restarted. Their output was copied into the author's research notes at the time. This file reproduces it so that every number in the paper has a source. The script that produced each table is in `experiments/liveness/`, and rerunning it regenerates the data.

## Experiment A: oracle validation under `aot_eager` (`liveness_oracle_check.py`, GTX 1650, 3 repeats)

Llama scale 8 and BERT scale 8. "static" is the liveness walk over the emitted forward/backward modules. "recomputed live at peak" is the part of the static peak made up of tensors recomputed in the backward.

| plan | measured MB | static MB | recomputed live at peak | closure_max | saved ops (knapsack) |
|---|---:|---:|---:|---:|---|
| nat 0.05 | 1145.4 | 1139.4 | 1068.7 | 65 | 26 SDPA-eff, 1 bmm, 0 MLP |
| nat 0.10 | 508.2 | 506.1 | 422.8 | 6 | 32 SDPA, 21 mm (21 MLP) |
| nat 0.15 | 274.8 | 274.2 | 159.8 | 2 | 32 SDPA, 46 mm (32 MLP) |
| nat 0.20 | 325.2 | 324.0 | 159.8 | 2 | 32 SDPA, 67 mm |
| nat 0.25 | 359.4 | 359.4 | 159.8 | 2 | 32 SDPA, 105 mm |
| nat 0.30 | 411.1 | 409.2 | 159.8 | 2 | 32 SDPA, 126 mm |
| nat 0.35 | 456.6 | 454.7 | 159.8 | 2 | 32 SDPA, 152 mm |
| 0.15, MLP forced out (DP refills) | 1136.1 | 1132.8 | 969.0 | 6 | 32 SDPA, 46 mm, 0 MLP |
| 0.05, MLP-only, over budget | 583.4 | 583.3 | 566.6 | 5 | 32 MLP mm, 0 SDPA |
| 0.05, force-in 8 MLP | 1018.0 | 1009.4 | 955.4 | 145 | 18 SDPA, 8 MLP |
| 0.05, force-in 16 MLP | 877.6 | 868.8 | 829.6 | 160 | 11 SDPA, 16 MLP |
| 0.05, force-in 24 MLP | 742.9 | 738.8 | 716.3 | 80 | 3 SDPA, 24 MLP |
| 0.05, no reorder | 1427.4 | 1412.2 | 1341.5 | 65 | = nat 0.05 |
| 0.15, no reorder | 1433.8 | 1412.2 | 1298.8 | 2 | = nat 0.15 |
| BERT 0.05 / 0.10 / 0.15 | 429.5 / 559.6 / 665.5 | 400.4 / 531.0 / 634.8 | 234.9 each | 4 | addmm, bmm |

Spearman correlation with measured peak over the 14 Llama plans:

| predictor | Spearman |
|---|---:|
| static walk | +0.999 |
| closure_max | +0.56 |
| closure_sum_w | +0.60 |
| `KnapsackEvaluator(account_for_backward_pass=True)` | −0.13 |
| saved weight | −0.62 |

The static walk's median relative error is 0.4%.

## Experiment C, part 1: oracle under Inductor, Llama (`experiment_c_inductor_runtime.py --part oracle`)

Peaks in MB. "Inductor's estimate" is `torch/_inductor/memory.py`'s estimate for the order Inductor's scheduler chose.

| plan | measured | Inductor's estimate | fx walk |
|---|---:|---:|---:|
| nat 0.05 | 904.1 | 906.4 | 1149.9 |
| nat 0.10 | 584.0 | 578.9 | 586.3 |
| nat 0.15 | 383.6 | 378.0 | 271.5 |
| nat 0.20 | 415.1 | 410.1 | 312.0 |
| nat 0.25 | 438.8 | 433.8 | 352.5 |
| nat 0.30 | 473.7 | 468.1 | 395.2 |
| nat 0.35 | 496.8 | 491.8 | 435.7 |
| 0.15, MLP forced out | 950.4 | 944.1 | 1122.3 |
| 0.05, force-in 8 MLP | 846.2 | 842.5 | 1019.8 |
| 0.05, force-in 16 MLP | 766.4 | 765.9 | 889.8 |
| 0.05, force-in 24 MLP | 670.7 | 672.9 | 759.8 |
| 0.05, MLP-only (over budget) | 555.6 | 552.3 | 583.3 |
| no reorder 0.05 | 1145.2 | 1137.6 | 1412.2 |
| no reorder 0.15 | 1011.5 | 1004.5 | 1412.2 |

| predictor | Spearman | median error | max error |
|---|---:|---:|---:|
| Inductor's estimate | 1.000 | 0.7% | 1.5% |
| fx walk | 0.995 | 19% | — |

The BERT oracle results under Inductor are in `expc_oracle_bert.json`.

## `check_grads.py`: Llama, budget 0.05, Inductor, against an uncompiled fp32 eager reference

| config | relative L2 error of the full gradient | worst single tensor |
|---|---:|---:|
| stock | 3.37e-7 | 9.73e-7 |
| stock (again) | 3.37e-7 | 9.73e-7 |
| stock + remat (light) | 3.39e-7 | 9.75e-7 |
| stock + remat (all) | 3.37e-7 | 9.73e-7 |

The "light" mode is called `pointwise` in the code.

## CPU runs of the RNG side finding (Section 6)

- **`repro_rng_budget.py`**: 8 × (Linear → GELU → Dropout) on CPU.
  - Inductor at budget 0: 21.5% gradient error. The finite-difference check gives 2.1e-2 for this run, against 3e-5 for the control.
  - `aot_eager`: gradients are wrong from budget 0.5 downwards.
- **`rng_redraw_test.py`**: 4-layer HF BERT, Inductor, CPU, stock PyTorch 2.14.
  - Advancing the global RNG between the forward and backward passes changes the budget-0.05 gradient by 3.7%. At budget 1.0 the change is 1.5e-7.
- **`repro_rng_bert_cpu.py`**: rel_L2 against budget 1.0 is about 3.5% with `--mode none` at budgets 0.30, 0.10 and 0.05. With `--mode v2` (fix emulated) it falls to 1.0–1.5e-7.
