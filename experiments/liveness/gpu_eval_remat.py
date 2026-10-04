"""
gpu_eval_remat.py -- CUDA validation of Experiment E: post-partition, liveness-driven
rematerialization, alone and combined with LivenessAwareSolver (LOCAL ONLY).

Configs per (model, budget, backend):
  stock              production partitioner
  stock+remat(pw)    same saved set; backward pass duplicates *pointwise* recomputation near late uses
  stock+remat(all)   same, matmul/attention may also be duplicated
  solver             LivenessAwareSolver (selection)
  solver+remat(pw)   both
  stock (again)      drift check

The remat pass is applied to the backward GraphModule right after partitioning, before the
backend compiles it. For Inductor that means before lowering, so pointwise clones get fused.

PRE-REGISTERED (fixed before running):
  R1 correctness: every config's grads match stock's (rtol 1e-3, atol 1e-4; CUDA nondeterminism).
  R2 free win   : stock+remat(pw) lowers Llama-0.05 measured peak by >= 10% on BOTH backends, with
                  step time <= +3% (after subtracting drift).
  R3 safety     : no remat config measures > 1% above stock.
  R4 combined   : solver+remat(pw) measures <= min(stock+remat(pw), solver) + 1% in every case.

Usage (ackaudit root, .venv-cuda, TORCHINDUCTOR_COMPILE_THREADS=1), inside tmux:
  python experiments/liveness/gpu_eval_remat.py --suite quick      # ~1.5-2 h
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_c_inductor_runtime as C     # capture hooks, Inductor estimate
import liveness_oracle_check as A
import liveness_solver as LS
import remat_at_use as R

import torch
import torch._functorch.partitioners as P
from torch._functorch import config as fc

REMAT = {"mode": None, "stats": {}}
_real_partition = P.min_cut_rematerialization_partition


def _partition_with_remat(*a, **k):
    fw, bw = _real_partition(*a, **k)
    if REMAT["mode"]:
        t0 = time.perf_counter()
        REMAT["stats"] = R.rematerialize_backward(bw, REMAT["mode"])
        REMAT["stats"]["pass_s"] = time.perf_counter() - t0
    return fw, bw


P.min_cut_rematerialization_partition = _partition_with_remat   # aot_eager path (looked up at call time)
C._orig_inductor_partition = _partition_with_remat               # Inductor path (C's capture wrapper)


def run(model, budget, backend, solver, remat, repeats, iters):
    torch.manual_seed(197838); torch._dynamo.reset(); C.CAP.clear(); C.INDUCTOR_EST.clear()
    REMAT["mode"], REMAT["stats"] = remat, {}
    build, make_inputs = A.resolve(model, 8)
    m = build().cuda(); args = tuple(x.cuda() for x in make_inputs())
    if backend == "aot_eager":
        from torch._dynamo.backends.common import aot_autograd
        from torch._dynamo.backends.debugging import boxed_nop

        def part(joint_module, joint_inputs, **kw):
            import experiment_b_liveness_select as B
            B._CTX.clear(); B._CTX.update(kw); B._CTX["joint_module"] = joint_module
            fw, bw = P.min_cut_rematerialization_partition(joint_module, joint_inputs, **kw)
            C.CAP["static"] = A.static_step_peak(fw, bw)
            return fw, bw
        be = aot_autograd(fw_compiler=boxed_nop, bw_compiler=boxed_nop, partition_fn=part, keep_inference_input_mutations=True)
    else:
        be = "inductor"
    cm = torch.compile(m, backend=be, dynamic=False)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    t0 = time.perf_counter()
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = budget, solver
        cm(*args).backward()
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
        REMAT["mode"] = None
    compile_s = time.perf_counter() - t0
    m.zero_grad(set_to_none=False); torch.cuda.synchronize(); torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated(); peaks = []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats(); cm(*args).backward(); torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - resident)
        m.zero_grad(set_to_none=False)
    # gradients for the correctness check: one extra step, copied straight to CPU AFTER the peak
    # measurements (copying on the GPU inside the loop would add the parameter size to later peaks)
    cm(*args).backward(); torch.cuda.synchronize()
    grads = [p.grad.detach().float().cpu() for p in m.parameters()]
    m.zero_grad(set_to_none=False)
    for _ in range(3):
        cm(*args).backward(); m.zero_grad(set_to_none=False)
    ms = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); cm(*args).backward(); e.record(); torch.cuda.synchronize()
        ms.append(s.elapsed_time(e)); m.zero_grad(set_to_none=False)
    row = dict(model=model, budget=budget, backend=backend, peak_MB=statistics.median(peaks) / 1e6,
               step_ms=statistics.median(ms), compile_s=compile_s, remat=dict(REMAT["stats"]),
               inductor_est_MB=max(C.INDUCTOR_EST.values()) if C.INDUCTOR_EST else None,
               solver=getattr(solver, "rec", {}) if not isinstance(solver, A.PlanSolver) else {})
    row["solver"] = {k: v for k, v in row["solver"].items() if not isinstance(v, (list, dict)) or k == "added_ops"}
    del m, cm, args; torch.cuda.empty_cache()
    return row, grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["quick", "full", "bert"], default="quick")
    ap.add_argument("--emulate-rng-fix", action="store_true",
                    help="emulate the upstream fix for pytorch#190758 (RNG ops never recomputed under the memory budget)")
    ap.add_argument("--backends", nargs="+", default=["inductor", "aot_eager"])
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="experiments/liveness/results/gpu_eval_remat.json")
    a = ap.parse_args()
    if a.emulate_rng_fix:
        import rng_fix_emulation; rng_fix_emulation.install()
        print("NOTE: emulating the pytorch#190758 fix (RNG ops excluded from memory-budget recompute)", flush=True)
    if a.suite == "quick":
        cases = [("llama", 0.05), ("llama", 0.15), ("bert", 0.05)]
    elif a.suite == "bert":
        cases = [("bert", b) for b in (0.05, 0.10, 0.15)]
    else:
        cases = [("llama", b) for b in (0.05, 0.10, 0.15, 0.20, 0.30)] + [("bert", b) for b in (0.05, 0.10, 0.15)]
    configs = [("stock", lambda: A.PlanSolver(), None), ("stock+remat(pw)", lambda: A.PlanSolver(), "pointwise"),
               ("stock+remat(all)", lambda: A.PlanSolver(), "all"), ("solver", lambda: LS.LivenessAwareSolver(), None),
               ("solver+remat(pw)", lambda: LS.LivenessAwareSolver(), "pointwise"), ("stock (again)", lambda: A.PlanSolver(), None)]
    rows, verdict = [], dict(R1=True, R2=None, R3=True, R4=True)
    for model, budget in cases:
        for be in a.backends:
            res, ref = {}, None
            for name, mk, rm in configs:
                print(f"[{model} {budget:.2f} {be} | {name}]", flush=True)
                r, g = run(model, budget, be, mk(), rm, a.repeats, a.iters)
                ref = g if ref is None else ref
                num = sum(float((x - y).double().pow(2).sum()) for x, y in zip(ref, g)) ** 0.5
                den = sum(float(x.double().pow(2).sum()) for x in ref) ** 0.5
                r["grad_rel_err"] = num / max(den, 1e-30)
                r["grads_match"] = all(torch.allclose(x, y, rtol=1e-3, atol=1e-4) for x, y in zip(ref, g))
                r["config"] = name; res[name] = r; rows.append(r)
                print(f"   peak={r['peak_MB']:.1f}MB step={r['step_ms']:.1f}ms compile={r['compile_s']:.0f}s grads_match={r['grads_match']} "
                      f"remat={ {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r['remat'].items()} }", flush=True)
                json.dump(rows, open(a.out, "w"), indent=1, default=str)
            s0, s1 = res["stock"], res["stock (again)"]
            ref_ms = (s0["step_ms"] + s1["step_ms"]) / 2; drift = abs(s1["step_ms"] / s0["step_ms"] - 1)
            for name, r in res.items():
                if not r["grads_match"]:
                    verdict["R1"] = False
                if "remat" in name and r["peak_MB"] > s0["peak_MB"] * 1.01:
                    verdict["R3"] = False
            if res["solver+remat(pw)"]["peak_MB"] > min(res["stock+remat(pw)"]["peak_MB"], res["solver"]["peak_MB"]) * 1.01:
                verdict["R4"] = False
            if model == "llama" and budget == 0.05:
                pw = res["stock+remat(pw)"]
                ok = pw["peak_MB"] <= 0.9 * s0["peak_MB"] and (pw["step_ms"] / ref_ms - 1) - drift <= 0.03
                verdict["R2"] = ok if verdict["R2"] is None else (verdict["R2"] and ok)

    print("\n" + "=" * 120)
    print(f"{'model':6} {'bud':>5} {'backend':9} {'config':18} {'peak MB':>8} {'Δpeak':>7} {'step ms':>8} {'Δstep':>7} {'compile s':>9} grads")
    for model, budget in cases:
        for be in a.backends:
            g = [r for r in rows if r["model"] == model and r["budget"] == budget and r["backend"] == be]
            if not g:
                continue
            s0 = g[0]; ref_ms = (g[0]["step_ms"] + g[-1]["step_ms"]) / 2
            for r in g:
                print(f"{model:6} {budget:5.2f} {be:9} {r['config']:18} {r['peak_MB']:8.1f} {r['peak_MB'] / s0['peak_MB'] - 1:+7.1%} "
                      f"{r['step_ms']:8.1f} {r['step_ms'] / ref_ms - 1:+7.1%} {r['compile_s']:9.0f} {r['grads_match']} rel_err={r.get('grad_rel_err', 0):.1e}")
    for k, v in verdict.items():
        print(f"{k}: {'n/a' if v is None else ('PASS' if v else 'FAIL')}")
    print(f"wrote {a.out}; no repository files modified.")


if __name__ == "__main__":
    main()
