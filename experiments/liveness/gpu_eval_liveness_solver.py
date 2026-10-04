"""
gpu_eval_liveness_solver.py -- CUDA validation of LivenessAwareSolver (LOCAL ONLY).

For each (model, budget, backend) it compiles and measures three runs, in this order:
stock (production DP), liveness (LivenessAwareSolver), and stock again (to measure drift).
It reports measured peak, step time (CUDA events, 3 warm-up + N timed iterations),
compile time, the solver's oracle calls and seconds, the invariant (oracle == module handed
to the backend), and Inductor's own peak estimate for both plans.

PRE-REGISTERED (fixed before running):
  G1 efficacy : Llama 0.05 under Inductor, measured peak >= 10% lower than stock.
  G2 safety   : no liveness run measures > 1% above stock, in any case.
  G3 cost     : step-time increase <= 10% in every case (after subtracting stock drift).
  G4 invariant: the oracle matches the emitted modules in every case.
  Reported, not scored: solver seconds, i.e. the compile-time overhead.

Usage (from the ackaudit root, .venv-cuda, with TORCHINDUCTOR_COMPILE_THREADS=1):
  python experiments/liveness/gpu_eval_liveness_solver.py --suite quick   # ~1 h
  python experiments/liveness/gpu_eval_liveness_solver.py --suite full    # ~3 h
"""
from __future__ import annotations

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_c_inductor_runtime as C   # shared harness: capture hooks, measurement, timing
import liveness_oracle_check as A
import liveness_solver as LS

import torch


def cases(suite):
    if suite == "quick":
        return [("llama", 0.05), ("llama", 0.15), ("bert", 0.05)]
    return [("llama", b) for b in (0.05, 0.10, 0.15, 0.20, 0.30)] + [("bert", b) for b in (0.05, 0.10, 0.15)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["quick", "full"], default="quick")
    ap.add_argument("--backends", nargs="+", default=["inductor", "aot_eager"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="experiments/liveness/results/gpu_eval_liveness_solver.json")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    rows = []
    for model, budget in cases(a.suite):
        for be in a.backends:
            for label, make in (("stock", lambda: A.PlanSolver()),
                                ("liveness", lambda: LS.LivenessAwareSolver()),
                                ("stock (again)", lambda: A.PlanSolver())):
                print(f"[{model} {budget:.2f} {be} | {label}]", flush=True)
                solver = make()
                r = C.run(label, model, 8, budget, be, solver, "cuda", a.repeats, iters=a.iters)
                if isinstance(solver, LS.LivenessAwareSolver) and "fx_walk_MB" in r:
                    r["invariant_ok"] = abs(r["fx_walk_MB"] - r["predicted_peak_MB"]) <= 1e-3 * max(1.0, r["fx_walk_MB"])
                rows.append(r)
                print(f"   peak={r['measured_peak_MB']:.1f}MB  step={r['step_ms']:.1f}ms  compile={r['compile_s']:.0f}s  "
                      f"inductor_est={r.get('inductor_est_MB')}  "
                      + (f"solver: {r['stock_peak_MB']:.1f}->{r['chosen_peak_MB']:.1f} predicted, retained={r['retained']:.3f}, "
                         f"{r['oracle_calls']} calls/{r['solver_seconds']:.0f}s, invariant={r.get('invariant_ok')}, added={r['added_ops']}"
                         if label == "liveness" else ""), flush=True)
                json.dump(rows, open(a.out, "w"), indent=1, default=str)

    print("\n" + "=" * 110)
    print(f"{'model':6} {'bud':>5} {'backend':9} {'stock MB':>9} {'liveness MB':>11} {'Δpeak':>7} {'Δstep':>7} {'drift':>6} "
          f"{'solver s':>8} {'calls':>5}  invariant")
    g1 = None; g2 = g3 = g4 = True
    for model, budget in cases(a.suite):
        for be in a.backends:
            g = [r for r in rows if r["model"] == model and r["budget"] == budget and r["backend"] == be]
            if len(g) < 3:
                continue
            s0, lv, s1 = g
            ref = (s0["step_ms"] + s1["step_ms"]) / 2
            dp = lv["measured_peak_MB"] / s0["measured_peak_MB"] - 1
            ds = lv["step_ms"] / ref - 1
            drift = s1["step_ms"] / s0["step_ms"] - 1
            print(f"{model:6} {budget:5.2f} {be:9} {s0['measured_peak_MB']:9.1f} {lv['measured_peak_MB']:11.1f} {dp:+7.1%} "
                  f"{ds:+7.1%} {drift:+6.1%} {lv['solver_seconds']:8.0f} {lv['oracle_calls']:5d}  {lv.get('invariant_ok')}")
            if dp > 0.01:
                g2 = False
            if ds - abs(drift) > 0.10:
                g3 = False
            if lv.get("invariant_ok") is False:
                g4 = False
            if model == "llama" and budget == 0.05 and be == "inductor":
                g1 = dp <= -0.10
    print(f"\nG1 efficacy (Llama 0.05 Inductor >= 10% lower): {'n/a' if g1 is None else ('PASS' if g1 else 'FAIL')}")
    print(f"G2 safety   (never > 1% above stock)          : {'PASS' if g2 else 'FAIL'}")
    print(f"G3 cost     (step-time <= +10%)               : {'PASS' if g3 else 'FAIL'}")
    print(f"G4 invariant                                  : {'PASS' if g4 else 'FAIL'}")
    print(f"wrote {a.out}; no repository files modified.")


if __name__ == "__main__":
    main()
