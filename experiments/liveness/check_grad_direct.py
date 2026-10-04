"""
check_grad_direct.py -- direct, per-parameter test of dropout-mask consistency (LOCAL ONLY).

The v4 finite-difference run showed that the forward loss is bit-identical at budget 1.0 and budget 0.05
with dropout ON (L0 = -4.439613 in both), so the forward draws the SAME masks. At budget 1.0 the backward
has 0 random ops (masks are saved). If the budget-0.05 backward regenerates exactly the forward's masks,
its gradients must equal budget 1.0's to float noise. Any real difference is a mask mismatch, and the
per-parameter breakdown shows WHERE it happens.

Configs (BERT scale 8, Inductor, dropout ON, fixed-random-projection loss as in check_grad_consistency v4):
  budget 1.0 (reference) | budget 0.05 stock | budget 0.05 stock+remat(pw)
Controls: the same three with dropout OFF (they must all match to ~1e-6).

Verdict (fixed before running): with dropout ON, a config is CONSISTENT if its total rel_L2 vs budget 1.0
is <= 10x the largest dropout-OFF rel_L2; MISMATCH if > 100x. The listing of affected parameters is
reported either way.
Usage: python experiments/liveness/check_grad_direct.py   (~40 min; 6 Inductor compiles)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_eval_remat as G
import liveness_oracle_check as A
import torch, torch.nn as nn
from torch._functorch import config as fc

RAND_WORDS = ("rand", "seed", "dropout", "philox", "bernoulli")
OPS = {}
_inner = G._real_partition


def _counting_partition(*a, **k):
    fw, bw = _inner(*a, **k)
    cnt = lambda gm: sum(1 for n in gm.graph.nodes if n.op == "call_function"
                         and any(w in str(n.target).lower() for w in RAND_WORDS))
    OPS["fw"], OPS["bw"] = cnt(fw), cnt(bw)
    return fw, bw


G._real_partition = _counting_partition


def build(nodrop):
    torch.manual_seed(197838)
    b, mk = A.resolve("bert", 8)
    m = b().cuda()
    if nodrop:
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
            for attr in ("dropout_prob", "attention_dropout", "dropout_p"):
                if isinstance(getattr(mod, attr, None), float):
                    setattr(mod, attr, 0.0)
        cfg = getattr(m, "config", None) or getattr(getattr(m, "model", None), "config", None)
        for attr in ("attention_probs_dropout_prob", "hidden_dropout_prob"):
            if cfg is not None and hasattr(cfg, attr):
                setattr(cfg, attr, 0.0)
    gen = torch.Generator(device="cpu").manual_seed(4242)
    R = torch.randn(m.model.config.hidden_size, generator=gen).cuda()
    inner = m.model
    m.forward = lambda ids, _i=inner, _R=R: (_i(input_ids=ids).last_hidden_state * _R).sum() / ids.numel()
    return m, tuple(x.cuda() for x in mk())


def grads(budget, remat, nodrop, S=1234):
    torch._dynamo.reset()
    m, args = build(nodrop)
    cm = torch.compile(m, backend="inductor", dynamic=False)
    G.REMAT["mode"] = remat
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = budget, A.PlanSolver()
        torch.manual_seed(S); cm(*args).backward()
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
        G.REMAT["mode"] = None
    m.zero_grad(set_to_none=True)
    torch.manual_seed(S); L = cm(*args); L.backward()
    out = {n: p.grad.detach().double().cpu() for n, p in m.named_parameters() if p.grad is not None}
    ops = dict(OPS)
    del m, cm; torch.cuda.empty_cache()
    return float(L), out, ops


def compare(ref, g):
    num = sum(float((ref[k] - g[k]).pow(2).sum()) for k in ref) ** 0.5
    den = sum(float(ref[k].pow(2).sum()) for k in ref) ** 0.5
    per = sorted(((float((ref[k] - g[k]).norm() / max(float(ref[k].norm()), 1e-30)), k) for k in ref), reverse=True)
    return num / den, per


results = {}
for nodrop in (True, False):
    tag = "dropout OFF" if nodrop else "dropout ON "
    L_ref, ref, ops_ref = grads(1.0, None, nodrop)
    print(f"\n{tag}  budget 1.0 (reference)   L={L_ref:.7f}  random ops fw={ops_ref.get('fw')} bw={ops_ref.get('bw')}", flush=True)
    for name, b, rm in (("budget 0.05 stock", 0.05, None), ("budget 0.05 stock+remat(pw)", 0.05, "pointwise")):
        L, g, ops = grads(b, rm, nodrop)
        rel, per = compare(ref, g)
        results[(nodrop, name)] = rel
        bad = [(e, k) for e, k in per if e > 1e-4]
        print(f"{tag}  {name:28s} L={L:.7f} (same forward: {abs(L - L_ref) < 1e-6})  rel_L2={rel:.2e}  "
              f"params with rel diff > 1e-4: {len(bad)}/{len(per)}  random ops fw={ops.get('fw')} bw={ops.get('bw')}", flush=True)
        for e, k in bad[:12]:
            print(f"      {e:.2e}  {k}")
floor = max(results[(True, n)] for n in ("budget 0.05 stock", "budget 0.05 stock+remat(pw)"))
print(f"\ndropout-OFF floor (float noise): {floor:.2e}")
for n in ("budget 0.05 stock", "budget 0.05 stock+remat(pw)"):
    r = results[(False, n)]
    v = "CONSISTENT" if r <= 10 * max(floor, 1e-12) else ("MISMATCH" if r > 100 * max(floor, 1e-12) else "UNCLEAR")
    print(f"VERDICT dropout ON {n}: {v}  (rel_L2 {r:.2e} vs floor {floor:.2e})")
