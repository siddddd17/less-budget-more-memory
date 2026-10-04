"""
check_grads.py -- is the Inductor remat(pw) gradient mismatch a bug or float noise? (LOCAL ONLY)

Llama scale 8, budget 0.05, Inductor. For each config it computes one step's gradients and
compares them with an *uncompiled eager* fp32 reference (same weights and inputs):
  relative L2 error over all parameters, the worst single tensor's relative error, and the
  max absolute difference. It also compiles stock twice to measure Inductor's own run-to-run noise.

Verdict rule (fixed before running): the remat pass is numerically sound if each remat config's
error against eager is within 3x the stock config's error against eager, and every relative L2
error is below 1e-3. A structural bug (wrong tensor rewired) typically gives O(1) errors.
Usage: python experiments/liveness/check_grads.py [--model llama --budget 0.05 --backend inductor]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_eval_remat as G, liveness_oracle_check as A
import torch

def eager_ref(model):
    torch.manual_seed(197838)
    build, make_inputs = A.resolve(model, 8)
    m = build().cuda(); args = tuple(x.cuda() for x in make_inputs())
    m(*args).backward(); torch.cuda.synchronize()
    return [p.grad.detach().float().cpu() for p in m.parameters()]

def err(ref, g):
    num = sum(float((x - y).double().pow(2).sum()) for x, y in zip(ref, g)) ** 0.5
    den = sum(float(x.double().pow(2).sum()) for x in ref) ** 0.5
    worst = max(float((x - y).double().norm() / max(float(x.double().norm()), 1e-30)) for x, y in zip(ref, g))
    mx = max(float((x - y).abs().max()) for x, y in zip(ref, g))
    return num / den, worst, mx

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="llama"); ap.add_argument("--budget", type=float, default=0.05)
ap.add_argument("--backend", default="inductor"); a = ap.parse_args()
ref = eager_ref(a.model)
res = {}
for name, rm in (("stock", None), ("stock (again)", None), ("stock+remat(pw)", "pointwise"), ("stock+remat(all)", "all")):
    _, g = G.run(a.model, a.budget, a.backend, A.PlanSolver(), rm, 1, 1)
    res[name] = g
    rel, worst, mx = err(ref, g)
    print(f"{name:18s} vs eager: rel_L2={rel:.2e}  worst_tensor={worst:.2e}  max_abs={mx:.2e}", flush=True)
rel_ss = err(res["stock"], res["stock (again)"])[0]
print(f"stock vs stock (again): rel_L2={rel_ss:.2e}   (Inductor run-to-run noise)")
base = err(ref, res["stock"])[0]
ok = all(err(ref, res[k])[0] <= max(3 * base, 1e-6) and err(ref, res[k])[0] < 1e-3 for k in ("stock+remat(pw)", "stock+remat(all)"))
print("VERDICT:", "numerically sound (float-level differences only)" if ok else "SUSPECT: investigate before trusting remat results")
