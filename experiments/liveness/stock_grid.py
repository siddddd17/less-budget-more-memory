"""
stock_grid.py -- stock PyTorch peak memory and step time on a dense grid of activation_memory_budget values
(Inductor), to check that the "best stock budget" used in the paper is not an artifact of a coarse grid.

Stock plans only (the production partitioner, no remat pass). Llama gets budgets 0, 0.01-0.03, 0.11-0.14, 0.16-0.19
and 1.0 in addition to the paper's 0.05/0.10/0.15/0.20/0.30; BERT gets 0.01-0.03 and 1.0 in addition to
0.05/0.10/0.15 (not 0: see GRID). BERT runs with the #190758 fix emulated and dropout ON (train mode is the HF default here), as in the
paper's BERT rows. Budgets already in the paper are re-measured too, so every point of the grid comes from one
process per model and is directly comparable (peaks vary by up to 1.5 MB across processes).

Usage (ackaudit root, TORCHINDUCTOR_COMPILE_THREADS=1): python experiments/liveness/stock_grid.py
  --models llama bert  --out experiments/liveness/results/stock_grid.json    (~60-75 min on the GTX 1650)
"""
from __future__ import annotations

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rng_fix_emulation
import gpu_eval_remat as G
import liveness_oracle_check as A
import torch._inductor.config as IC
IC.force_disable_caches = True     # never reuse a compiled plan across budgets

GRID = {
    "llama": [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.30, 1.0],
    # no budget 0 for BERT: the budget-0 path returns early, before the #190758 fix emulation applies, so its plan
    # would recompute dropout (wrong gradients) and its peak is not comparable with the fixed plans
    "bert": [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 1.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama", "bert"])
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="experiments/liveness/results/stock_grid.json")
    a = ap.parse_args()
    rows = []
    for model in a.models:
        if model == "bert":
            rng_fix_emulation.install()     # idempotent; Llama has no random ops, so the order does not matter
        for b in GRID[model]:
            print(f"[{model} {b:.2f} inductor | stock]", flush=True)
            r, _ = G.run(model, b, "inductor", A.PlanSolver(), None, a.repeats, a.iters)
            r["config"] = "stock"; r["rng_fix_emulated"] = model == "bert"
            rows.append(r)
            print(f"   peak={r['peak_MB']:.1f}MB step={r['step_ms']:.1f}ms compile={r['compile_s']:.0f}s", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1, default=str)
    print()
    for model in a.models:
        g = [r for r in rows if r["model"] == model]
        best = min(g, key=lambda r: r["peak_MB"])
        print(f"{model}: lowest stock peak {best['peak_MB']:.1f} MB at budget {best['budget']}  |  " +
              "  ".join(f"{r['budget']:.2f}:{r['peak_MB']:.0f}" for r in g))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
