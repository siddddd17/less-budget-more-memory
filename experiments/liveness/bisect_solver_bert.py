"""
bisect_solver_bert.py -- localize why LivenessAwareSolver's BERT/Inductor plan at budget 0.05 gives wrong
gradients (check_solver_bert: rel_L2 2.0e-2, stock 2.2e-7, same random ops in the backward). (LOCAL ONLY)

The solver only chooses which knapsack candidates to save (the public activation_memory_budget_solver hook);
the partitioner does the rest. So either our choice exposes a partitioner bug, or something RNG-related.

Steps (dropout ON unless noted; RNG fix emulation v2; projection loss; reseeded; budget 1.0 = reference):
  1  ON  solver plan            -> rel_L2, plus an RNG probe: advance the global RNG between forward and backward.
                                   If the gradient changes, the backward draws fresh randomness.
  2  OFF the SAME plan, forced  -> candidates mapped by (op, occurrence index). If wrong without dropout,
                                   the plan is wrong regardless of randomness.
  3  ON  bisection by op class  -> the solver plan with ONE op class's changes reverted to stock, per class.
                                   A class whose revert fixes the gradients is (part of) the culprit.
Verdicts are printed after step 2 (RNG vs plan) and after step 3 (which class). Steps 1-2 take ~20 min; the
whole run ~40-50 min on the GTX 1650. You can stop after the step-2 verdict if time is short.

Thresholds (fixed before running): CORRECT <= 1e-5 rel_L2, WRONG > 1e-3.
Env for local testing: CHECK_DEVICE=cpu, CHECK_NO_FIX=1 (sensitivity control).
Usage: python experiments/liveness/bisect_solver_bert.py
"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rng_fix_emulation
if not os.environ.get("CHECK_NO_FIX"):
    rng_fix_emulation.install()
import gpu_eval_remat as G
import liveness_oracle_check as A
import liveness_solver as LS
import torch, torch.nn as nn
from torch._functorch import config as fc
from torch._functorch.partitioners import CustomKnapsackSolver
import torch._inductor.config as IC
IC.force_disable_caches = True

DEV = os.environ.get("CHECK_DEVICE", "cuda")
OK, BAD = 1e-5, 1e-3
RAND_WORDS = ("rand", "seed", "dropout", "philox", "bernoulli")
CAP = {}
_inner = G._real_partition


def _capturing_partition(*a, **k):
    fw, bw = _inner(*a, **k)
    CAP["bw_ops"] = collections.Counter(str(n.target) for n in bw.graph.nodes if n.op == "call_function")
    return fw, bw


G._real_partition = _capturing_partition

_KEYS: dict = {}


def key(n):
    """Graph-independent identity of a node: (op target, occurrence index of that target in graph order)."""
    g = n.graph
    if id(g) not in _KEYS:
        cnt, d = collections.Counter(), {}
        for x in g.nodes:
            t = str(x.target); d[x] = (t, cnt[t]); cnt[t] += 1
        _KEYS[id(g)] = d
    return _KEYS[id(g)][n]


def opclass(k):
    return k[0].replace("aten.", "").replace(".default", "").split(".")[0]


class Recorder(LS.LivenessAwareSolver):
    """The real solver; additionally records its saved set and stock's, as node keys."""
    def __call__(self, memory, joint_graph, max_memory, node_info, cands):
        cands = list(cands)
        saved, recomp = super().__call__(memory, joint_graph, max_memory, node_info, cands)
        stock, _ = A.PlanSolver()(memory, joint_graph, max_memory, node_info, cands)
        self.keys = dict(all={key(c) for c in cands}, solver={key(cands[i]) for i in saved},
                         stock={key(cands[i]) for i in stock})
        return saved, recomp


class Forced(CustomKnapsackSolver):
    """Saves exactly the candidates whose keys are in `want`."""
    def __init__(self, want):
        self.want = set(want); self.found = 0; self.n = 0

    def __call__(self, memory, joint_graph, max_memory, node_info, cands):
        cands = list(cands)
        saved = [i for i, c in enumerate(cands) if key(c) in self.want]
        self.found, self.n = len(saved), len(cands)
        self.missing = self.want - {key(c) for c in cands}
        return saved, [i for i in range(len(cands)) if i not in set(saved)]

    def uuid(self):
        return None


def build(nodrop):
    torch.manual_seed(197838)
    b, mk = A.resolve("bert", int(os.environ.get("CHECK_SCALE", "8")))
    m = b().to(DEV).train()
    if nodrop:
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
            for attr in ("dropout_prob", "attention_dropout", "dropout_p"):
                if isinstance(getattr(mod, attr, None), float):
                    setattr(mod, attr, 0.0)
        for cfg in (getattr(m, "config", None), getattr(getattr(m, "model", None), "config", None)):
            for attr in ("attention_probs_dropout_prob", "hidden_dropout_prob"):
                if cfg is not None and hasattr(cfg, attr):
                    setattr(cfg, attr, 0.0)
    R = torch.randn(m.model.config.hidden_size, generator=torch.Generator().manual_seed(4242)).to(DEV)
    inner = m.model
    m.forward = lambda ids, _i=inner, _R=R: (_i(input_ids=ids).last_hidden_state * _R).sum() / ids.numel()
    return m, tuple(x.to(DEV) for x in mk())


def grads(budget, solver, nodrop, probe=False, S=1234):
    torch._dynamo.reset(); CAP.clear(); _KEYS.clear()
    m, args = build(nodrop)
    cm = torch.compile(m, backend="inductor", dynamic=False)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = budget, solver
        torch.manual_seed(S); cm(*args).backward()
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
    if not CAP:
        raise RuntimeError("partition hook did not fire (cache hit?)")

    def step(advance):
        m.zero_grad(set_to_none=True)
        torch.manual_seed(S); out = cm(*args)
        if advance:   # consume global RNG between forward and backward
            torch.rand(4096, device=DEV); torch.manual_seed(S + 999)
        out.backward()
        return float(out.detach()), {n: p.grad.detach().double().cpu() for n, p in m.named_parameters() if p.grad is not None}

    L, g = step(False)
    g_adv = step(True)[1] if probe else None
    cap = dict(CAP)
    del m, cm
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return L, g, g_adv, cap


def rel(ref, g):
    num = sum(float((ref[k] - g[k]).pow(2).sum()) for k in ref) ** 0.5
    return num / max(sum(float(ref[k].pow(2).sum()) for k in ref) ** 0.5, 1e-30)


def word(r):
    return "CORRECT" if r <= OK else ("WRONG" if r > BAD else "unclear")


# ---- step 1: dropout ON, solver plan, RNG probe -----------------------------------------------
L0, ref_on, _, _ = grads(1.0, A.PlanSolver(), nodrop=False)
rec = Recorder()
L1, g1, g1_adv, cap1 = grads(0.05, rec, nodrop=False, probe=True)
K = rec.keys
r1, r_probe = rel(ref_on, g1), rel(g1, g1_adv)
added, removed = K["solver"] - K["stock"], K["stock"] - K["solver"]
print(f"[1] ON  solver plan: rel_L2={r1:.2e} ({word(r1)})  same_fwd={abs(L1 - L0) <= 1e-6 * max(1, abs(L0))}  "
      f"chosen_is_stock={K['solver'] == K['stock']}", flush=True)
print(f"    RNG probe (advance RNG between fw and bw): grad change {r_probe:.2e} -> "
      f"{'backward DRAWS FRESH RANDOMNESS' if r_probe > OK else 'backward does not depend on global RNG'}", flush=True)
print(f"    solver vs stock: +{dict(collections.Counter(opclass(k) for k in added))}  "
      f"-{dict(collections.Counter(opclass(k) for k in removed))}", flush=True)
if K["solver"] == K["stock"]:
    sys.exit("solver chose the stock plan on this graph: nothing to bisect.")

# ---- step 2: dropout OFF, same plan forced ----------------------------------------------------
L2r, ref_off, _, _ = grads(1.0, A.PlanSolver(), nodrop=True)
f = Forced(K["solver"])
L2, g2, _, _ = grads(0.05, f, nodrop=True)
r2 = rel(ref_off, g2)
print(f"[2] OFF same plan forced: rel_L2={r2:.2e} ({word(r2)})  mapped {f.found}/{len(K['solver'])} saved candidates "
      f"(of {f.n} candidates in the OFF graph)", flush=True)
if f.missing:
    print(f"    not mappable (no such candidate without dropout): {sorted(f.missing)}", flush=True)
if os.environ.get("CHECK_PROBE_STOCK"):   # local sensitivity control for the RNG probe
    _, gs, gs_adv, _ = grads(0.05, A.PlanSolver(), nodrop=False, probe=True)
    print(f"    [control] stock RNG probe: grad change {rel(gs, gs_adv):.2e}", flush=True)
print()
if r1 <= OK:
    print("STEP-2 VERDICT: solver plan is correct in this run (did not reproduce).")
elif r_probe > OK:
    print("STEP-2 VERDICT: RNG. The solver plan's backward draws fresh randomness (the fix emulation misses a path).")
elif r2 > BAD:
    print("STEP-2 VERDICT: PLAN BUG, independent of randomness. A legal saved set gives wrong gradients even "
          "without dropout -> likely a partitioner bug exposed by this saved set (or our forced mapping).")
elif r2 <= OK:
    print("STEP-2 VERDICT: the plan is correct without dropout and the backward does not use the global RNG, so the "
          "error needs dropout to be present: masks are regenerated with the right seeds but WRONG values/indexing.")
else:
    print(f"STEP-2 VERDICT: unclear (OFF rel_L2 {r2:.2e}).")
sys.stdout.flush()

# ---- step 3: dropout ON, revert one op class at a time ----------------------------------------
if r1 <= BAD:
    sys.exit("step 3 skipped: the solver plan is not WRONG in this run, so there is nothing to localize.")
classes = sorted({opclass(k) for k in added | removed})
print(f"\n[3] bisection over {len(classes)} op classes: {classes}", flush=True)
culprits = []
for c in classes:
    want = (K["solver"] - {k for k in added if opclass(k) == c}) | {k for k in removed if opclass(k) == c}
    _, g3, _, _ = grads(0.05, Forced(want), nodrop=False)
    r3 = rel(ref_on, g3)
    fixed = r3 <= OK
    culprits += [c] if fixed else []
    print(f"    revert {c:28s}: rel_L2={r3:.2e} ({word(r3)}){'   <-- reverting this FIXES it' if fixed else ''}", flush=True)
print(f"\nSTEP-3 VERDICT: {'culprit class(es): ' + ', '.join(culprits) if culprits else 'no single class fixes it (interaction of several classes)'}")
