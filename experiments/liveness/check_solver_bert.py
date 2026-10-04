"""
check_solver_bert.py (v3) -- is LivenessAwareSolver's BERT/Inductor plan at budget 0.05 numerically correct?
(LOCAL ONLY)

Background: in gpu_eval_remat_bert_fixed_v2, the solver rows at 0.05 differed from stock by rel_err 1.2e-2,
while all stock and remat rows matched. That harness uses BERT's own loss, sum(last_hidden_state) after
LayerNorm, which is ~0 by construction, so its "gradients" are mostly float residue, and it does not reseed
before its gradient step. Either could produce a false alarm.

This test (dropout ON, RNG fix emulation v2 installed, non-degenerate projection loss, reseeded):
  budget 1.0 (reference) | budget 0.05 stock | budget 0.05 solver
For each plan: loss, rel_L2 vs budget 1.0, random ops in the backward, whether the solver chose a non-stock
plan, and the solver-vs-stock backward op difference.

Verdict rules (fixed before running):
  NOT REPRODUCED  solver chose the stock plan on this graph -> this run cannot judge the solver's plan.
  INVALID         stock rel_L2 > 1e-5 (the fix emulation / reseeding is not working; nothing can be judged).
  CORRECT         solver rel_L2 <= 1e-5  -> the 1.2e-2 was a harness artifact (degenerate loss / no reseed).
  WRONG-RNG       solver rel_L2 > 1e-3 and its backward has random ops stock's lacks -> randomness leak.
  WRONG-PLAN      solver rel_L2 > 1e-3 and the same random ops as stock -> a real numerical bug in the plan.
  UNCLEAR         anything else (1e-5 < rel_L2 <= 1e-3).
Usage: python experiments/liveness/check_solver_bert.py   (~15-20 min on the GTX 1650; 3 Inductor compiles)
"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rng_fix_emulation
if not os.environ.get("CHECK_NO_FIX"):   # CHECK_NO_FIX=1 is the sensitivity control: stock must then FAIL
    rng_fix_emulation.install()
import gpu_eval_remat as G
import liveness_oracle_check as A
import liveness_solver as LS
import torch
from torch._functorch import config as fc
import torch._inductor.config as IC
IC.force_disable_caches = True   # never reuse a compiled plan across configs

DEV = os.environ.get("CHECK_DEVICE", "cuda")
SCALE = int(os.environ.get("CHECK_SCALE", "8"))
RAND_WORDS = ("rand", "seed", "dropout", "philox", "bernoulli")
CAP = {}
_inner = G._real_partition


def _capturing_partition(*a, **k):
    fw, bw = _inner(*a, **k)
    ops = lambda gm: collections.Counter(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
    CAP["bw_ops"] = ops(bw)
    CAP["bw_rand"] = collections.Counter({t: c for t, c in CAP["bw_ops"].items() if any(w in t.lower() for w in RAND_WORDS)})
    return fw, bw


G._real_partition = _capturing_partition


def build():
    torch.manual_seed(197838)
    b, mk = A.resolve("bert", SCALE)
    m = b().to(DEV).train()
    # non-degenerate loss: fixed random projection of the outputs (as in check_grad_direct.py)
    R = torch.randn(m.model.config.hidden_size, generator=torch.Generator().manual_seed(4242)).to(DEV)
    inner = m.model
    m.forward = lambda ids, _i=inner, _R=R: (_i(input_ids=ids).last_hidden_state * _R).sum() / ids.numel()
    return m, tuple(x.to(DEV) for x in mk())


def grads(budget, solver, S=1234):
    torch._dynamo.reset(); CAP.clear()
    m, args = build()
    cm = torch.compile(m, backend="inductor", dynamic=False)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = budget, solver
        torch.manual_seed(S); cm(*args).backward()          # compile step
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
    if not CAP:
        raise RuntimeError("partition hook did not fire: the plan was not captured (cache hit?)")
    m.zero_grad(set_to_none=True)
    torch.manual_seed(S); out = cm(*args); L = float(out.detach()); out.backward()
    g = {n: p.grad.detach().double().cpu() for n, p in m.named_parameters() if p.grad is not None}
    cap = dict(CAP)
    del m, cm
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return L, g, cap


def rel(ref, g):
    assert ref.keys() == g.keys(), "parameter sets differ"
    num = sum(float((ref[k] - g[k]).pow(2).sum()) for k in ref) ** 0.5
    return num / max(sum(float(ref[k].pow(2).sum()) for k in ref) ** 0.5, 1e-30)


L0, ref, cap0 = grads(1.0, A.PlanSolver())
print(f"budget 1.0 ref   L={L0:.7f}  bw random ops={dict(cap0['bw_rand'])}", flush=True)
L1, g1, cap1 = grads(0.05, A.PlanSolver())
r_stock = rel(ref, g1)
print(f"0.05 stock       L={L1:.7f} same_fwd={abs(L1 - L0) <= 1e-6 * max(1.0, abs(L0))}  rel_L2={r_stock:.2e}  "
      f"bw random ops={dict(cap1['bw_rand'])}", flush=True)
solver = LS.LivenessAwareSolver()
L2, g2, cap2 = grads(0.05, solver)
r_solver = rel(ref, g2)
st = solver.stats[-1] if solver.stats else {}
print(f"0.05 solver      L={L2:.7f} same_fwd={abs(L2 - L0) <= 1e-6 * max(1.0, abs(L0))}  rel_L2={r_solver:.2e}  "
      f"bw random ops={dict(cap2['bw_rand'])}", flush=True)
print(f"   solver: chosen_is_stock={st.get('chosen_is_stock')}  added_ops={st.get('added_ops')}  "
      f"pred peak {st.get('stock_peak_MB', float('nan')):.1f} -> {st.get('chosen_peak_MB', float('nan')):.1f} MB")
print(f"   solver bw ops MORE than stock : {dict(cap2['bw_ops'] - cap1['bw_ops'])}")
print(f"   solver bw ops FEWER than stock: {dict(cap1['bw_ops'] - cap2['bw_ops'])}")
extra_rand = cap2["bw_rand"] - cap1["bw_rand"]

print()
if st.get("chosen_is_stock", cap2["bw_ops"] == cap1["bw_ops"]):
    print("VERDICT NOT REPRODUCED: the solver chose the stock plan on this graph; this run cannot judge its plan.")
elif r_stock > 1e-5:
    print(f"VERDICT INVALID: stock itself differs from budget 1.0 (rel_L2 {r_stock:.2e}); the RNG fix / reseeding is not working.")
elif r_solver <= 1e-5:
    print(f"VERDICT CORRECT: solver plan matches budget 1.0 (rel_L2 {r_solver:.2e}); the earlier 1.2e-2 was a harness artifact.")
elif r_solver > 1e-3 and extra_rand:
    print(f"VERDICT WRONG-RNG: solver plan differs (rel_L2 {r_solver:.2e}) and recomputes extra random ops: {dict(extra_rand)}")
elif r_solver > 1e-3:
    print(f"VERDICT WRONG-PLAN: solver plan differs (rel_L2 {r_solver:.2e}) with the same random ops as stock.")
else:
    print(f"VERDICT UNCLEAR: solver rel_L2 {r_solver:.2e} (stock {r_stock:.2e}).")
