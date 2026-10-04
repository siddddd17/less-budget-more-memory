"""
remat_edit_validation.py -- does the remat pass's own peak model (fx walk of the backward graph) predict what its edits
do under Inductor?  The paper validates the oracles on whole plans; this checks the pass's individual decisions.

The pass is greedy and deterministic, so running it with max_accepted=k applies exactly the first k edits of the full
pass. For each case (Llama, Inductor, budget, mode) we run the full pass once to get N accepted edits and the predicted
backward peak after every edit (peak_trace), then compile and measure prefixes k in {0, N/8, N/4, N/2, 3N/4, N}.
For each prefix we record:
  predicted  : fx-walk backward peak after k edits (the number the pass optimizes)
  inductor   : Inductor's own memory estimate (torch/_inductor/memory.py) of the compiled graphs
  measured   : max_memory_allocated over a step, minus resident memory (median of --repeats)
and compare the change from k=0 (the stock plan) to k: does the predicted saving show up in the measured peak, and is
it ordered the same way? Llama has no random ops, so no RNG fix is needed.

Usage (repo root, TORCHINDUCTOR_COMPILE_THREADS=1, inside tmux; ~1 h on the GTX 1650):
  python experiments/liveness/remat_edit_validation.py --out experiments/liveness/results/remat_edit_validation.json
"""
from __future__ import annotations

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_eval_remat as G
import liveness_oracle_check as A
import remat_at_use as R
import torch._inductor.config as IC

IC.force_disable_caches = True
_full = R.rematerialize_backward
LIMIT = {"k": None}


def _limited(gm, mode="pointwise", **kw):
    return _full(gm, mode, max_accepted=LIMIT["k"], **kw)


R.rematerialize_backward = _limited             # gpu_eval_remat looks it up on the module at call time


def spearman(a, b):
    import statistics
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    ap.add_argument("--modes", nargs="+", default=["pointwise", "all"])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5, 0.75])
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default="experiments/liveness/results/remat_edit_validation.json")
    a = ap.parse_args()
    out = []
    for b in a.budgets:
        for mode in a.modes:
            LIMIT["k"] = None
            print(f"[llama {b:.2f} inductor | {mode} | full pass]", flush=True)
            full, _ = G.run("llama", b, "inductor", A.PlanSolver(), mode, a.repeats, a.iters)
            N, trace = full["remat"]["accepted"], full["remat"]["peak_trace"]
            ks = sorted({0, N} | {round(N * f) for f in a.fractions})
            pts = []
            for k in ks:
                if k == N:
                    r = full
                else:
                    LIMIT["k"] = k
                    print(f"[llama {b:.2f} inductor | {mode} | first {k}/{N} edits]", flush=True)
                    r, _ = G.run("llama", b, "inductor", A.PlanSolver(), mode, a.repeats, a.iters)
                    assert r["remat"]["accepted"] == k, (r["remat"]["accepted"], k)
                    assert r["remat"]["peak_trace"] == trace[:k + 1], "pass is not deterministic across compiles"
                pts.append(dict(k=k, predicted_bw_MB=trace[k] / 1e6, inductor_est_MB=r["inductor_est_MB"] if r["inductor_est_MB"] is not None else float("nan"),
                                measured_MB=r["peak_MB"], step_ms=r["step_ms"], flops_added=r["remat"]["flops_added"]))
                print(f"   k={k:4d}  predicted bw {trace[k] / 1e6:8.1f}  inductor est {pts[-1]['inductor_est_MB']:8.1f}  "
                      f"measured {r['peak_MB']:8.1f} MB", flush=True)
            p0 = pts[0]
            for p in pts:
                p["d_predicted"] = p["predicted_bw_MB"] - p0["predicted_bw_MB"]
                p["d_inductor"] = p["inductor_est_MB"] - p0["inductor_est_MB"]
                p["d_measured"] = p["measured_MB"] - p0["measured_MB"]
            case = dict(budget=b, mode=mode, accepted=N, points=pts,
                        rho_predicted_vs_measured=spearman([p["predicted_bw_MB"] for p in pts], [p["measured_MB"] for p in pts]),
                        rho_inductor_vs_measured=spearman([p["inductor_est_MB"] for p in pts], [p["measured_MB"] for p in pts]))
            out.append(case)
            last = pts[-1]
            print(f"== {b:.2f} {mode}: N={N}  full-pass change predicted {last['d_predicted']:+.1f} MB, Inductor est "
                  f"{last['d_inductor']:+.1f} MB, measured {last['d_measured']:+.1f} MB;  rho(pred, meas)="
                  f"{case['rho_predicted_vs_measured']:.3f}  rho(ind, meas)={case['rho_inductor_vs_measured']:.3f}", flush=True)
            json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
