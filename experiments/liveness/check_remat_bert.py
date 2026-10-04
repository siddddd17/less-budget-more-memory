"""
check_remat_bert.py -- are use-site rematerialization plans correct on BERT with dropout ON? (LOCAL ONLY)

The earlier BERT/dropout-ON comparisons for the remat pass (rel_err ~2e-8 vs stock) used the benchmark wrapper's loss,
sum(LayerNorm output), which is ~0 by construction. This repeats them with the non-degenerate projection loss,
reseeded, against budget 1.0, with the RNG fix emulation v2 installed. For the paper.

Configs (Inductor, BERT scale 8, dropout ON): budget 1.0 | 0.05 stock | 0.05 stock+remat(pw) | 0.05 stock+remat(all)
Verdict (fixed before running): CORRECT if rel_L2 <= 1e-5 (float noise; stock was 2.2e-7), WRONG if > 1e-3.
Control: CHECK_NO_FIX=1 must make stock WRONG (local sensitivity test).
Usage: python experiments/liveness/check_remat_bert.py   (~20 min on the GTX 1650; 4 Inductor compiles)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rng_fix_emulation
if not os.environ.get("CHECK_NO_FIX"):
    rng_fix_emulation.install()
import gpu_eval_remat as G
import liveness_oracle_check as A
import torch
from torch._functorch import config as fc
import torch._inductor.config as IC
IC.force_disable_caches = True

DEV = os.environ.get("CHECK_DEVICE", "cuda")
SCALE = int(os.environ.get("CHECK_SCALE", "8"))
BUDGETS = [float(b) for b in os.environ.get("CHECK_BUDGETS", "0.05").split(",")]   # e.g. CHECK_BUDGETS=0.10,0.15


def build():
    torch.manual_seed(197838)
    b, mk = A.resolve("bert", SCALE)
    m = b().to(DEV).train()
    R = torch.randn(m.model.config.hidden_size, generator=torch.Generator().manual_seed(4242)).to(DEV)
    inner = m.model
    m.forward = lambda ids, _i=inner, _R=R: (_i(input_ids=ids).last_hidden_state * _R).sum() / ids.numel()
    return m, tuple(x.to(DEV) for x in mk())


def grads(budget, remat, S=1234):
    torch._dynamo.reset()
    m, args = build()
    cm = torch.compile(m, backend="inductor", dynamic=False)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    G.REMAT["mode"], G.REMAT["stats"] = remat, {}
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = budget, A.PlanSolver()
        torch.manual_seed(S); cm(*args).backward()
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
        G.REMAT["mode"] = None
    stats = dict(G.REMAT["stats"])
    if remat and not stats.get("accepted"):
        print(f"   note: remat({remat}) accepted no changes on this graph")
    m.zero_grad(set_to_none=True)
    torch.manual_seed(S); out = cm(*args); L = float(out.detach()); out.backward()
    g = {n: p.grad.detach().double().cpu() for n, p in m.named_parameters() if p.grad is not None}
    del m, cm
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return L, g, stats


def rel(ref, g):
    num = sum(float((ref[k] - g[k]).pow(2).sum()) for k in ref) ** 0.5
    return num / max(sum(float(ref[k].pow(2).sum()) for k in ref) ** 0.5, 1e-30)


L0, ref, _ = grads(1.0, None)
print(f"budget 1.0 ref            L={L0:.7f}", flush=True)
verdicts = {}
for b in BUDGETS:
  for name, rm in ((f"{b:.2f} stock", None), (f"{b:.2f} stock+remat(pw)", "pointwise"), (f"{b:.2f} stock+remat(all)", "all")):
    L, g, st = grads(b, rm)
    r = rel(ref, g)
    v = "CORRECT" if r <= 1e-5 else ("WRONG" if r > 1e-3 else "UNCLEAR")
    verdicts[name] = v
    extra = (f"  accepted={st.get('accepted')} peak {st.get('peak_before', 0)/1e6:.0f}->{st.get('peak_after', 0)/1e6:.0f}MB(pred)"
             f" random_ops {st.get('random_ops_before')}->{st.get('random_ops_after')}") if rm else ""
    print(f"{name:24s} L={L:.7f} same_fwd={abs(L - L0) <= 1e-6 * max(1, abs(L0))}  rel_L2={r:.2e}  {v}{extra}", flush=True)
print("\nVERDICT:", ", ".join(f"{k}: {v}" for k, v in verdicts.items()))
