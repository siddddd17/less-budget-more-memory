"""
check_grad_consistency.py -- is the gradient the true gradient of the loss that was computed? (LOCAL ONLY)

Question: under activation_memory_budget < 1 with Inductor, BERT's backward contains the same
number of random ops as the forward, i.e. dropout is regenerated in the backward. Does the backward
reproduce the forward's dropout masks (fine), or draw new ones (silently wrong gradients)?

Method: a directional finite-difference check, which needs no external reference.
  1. seed S; loss L0 = f(theta); backward -> g.
  2. d = g; predicted directional derivative = ||g||^2.
  3. Re-seed S before every forward, so the forward draws the same masks, and measure
     L(theta + e*d) and L(theta - e*d). The central difference (L+ - L-)/(2e) should equal ||g||^2.
  4. rel_err = |FD - ||g||^2| / ||g||^2, at two step sizes (convergence check).
  Validity check: re-seeding reproduces L0 exactly (otherwise the forward isn't deterministic and
  the test is meaningless).

Controls: budget 1.0 (nothing recomputed; the backward has no random ops), and dropout OFF.
Validity (fixed before running): all three controls must have rel_err < 1e-4 and the dropout-OFF runs
must have 0 random ops; otherwise TEST INVALID and no verdict is printed.
v4: three step sizes (2e-4, 1e-4, 5e-5 x ||theta||) and Richardson extrapolation (the v3 run showed pure
O(h^2) truncation error, identical in every config). Validity: controls' Richardson error < 1e-3.
Verdict rule (fixed before running, v3, now applied to the Richardson errors): CONSISTENT if rel_err <= 10x the dropout-ON budget-1.0
control at both step sizes; INCONSISTENT if > 100x the control at both; otherwise UNCLEAR.
The BERT loss is replaced by a fixed random projection of the outputs (see build()), because
ackaudit's last_hidden_state.sum() after a final LayerNorm is ~0 by construction.
Usage: python experiments/liveness/check_grad_consistency.py   (~30 min; 5 Inductor compiles)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_eval_remat as G                        # installs the partition hook used for remat
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


def build(model, nodrop):
    torch.manual_seed(197838)
    b, mk = A.resolve(model, 8)
    m = b().cuda()
    if nodrop:   # identical to check_rng_bert.py, which verified 0 random ops
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
            for attr in ("dropout_prob", "attention_dropout", "dropout_p"):
                if isinstance(getattr(mod, attr, None), float):
                    setattr(mod, attr, 0.0)
        cfg = getattr(m, "config", None)
        for attr in ("attention_probs_dropout_prob", "hidden_dropout_prob", "attention_dropout", "hidden_dropout"):
            if cfg is not None and hasattr(cfg, attr):
                setattr(cfg, attr, 0.0)
    # ackaudit's BertWrap returns last_hidden_state.sum(). BERT ends in a LayerNorm, so that loss is
    # ~0 by construction and its true gradient w.r.t. almost every weight is 0, which is useless for
    # testing dropout-mask consistency. Use a fixed random projection so every weight and every
    # dropout mask affects the loss.
    if hasattr(m, "model") and hasattr(m.model, "config"):
        gen = torch.Generator(device="cpu").manual_seed(4242)
        R = torch.randn(m.model.config.hidden_size, generator=gen).cuda()
        inner = m.model
        m.forward = lambda ids, _inner=inner, _R=R: (_inner(input_ids=ids).last_hidden_state * _R).sum() / ids.numel()
    return m, tuple(x.cuda() for x in mk())


def check(model, budget, remat, nodrop, S=1234):
    torch._dynamo.reset()
    m, args = build(model, nodrop)
    cm = torch.compile(m, backend="inductor", dynamic=False)
    G.REMAT["mode"] = remat
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = budget, A.PlanSolver()
        torch.manual_seed(S); cm(*args).backward()           # compile (fw + bw)
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
        G.REMAT["mode"] = None
    m.zero_grad(set_to_none=True)
    torch.manual_seed(S); L0 = cm(*args); L0.backward()
    params = [p for p in m.parameters() if p.grad is not None]
    g = [p.grad.detach().clone() for p in params]
    nrm2 = float(sum((x.double() ** 2).sum() for x in g))
    pnorm = float(sum((p.detach().double() ** 2).sum() for p in params)) ** 0.5
    # forwards stay in grad mode, so they reuse the SAME compiled training graph (no_grad would recompile)
    torch.manual_seed(S); L0b = float(cm(*args).detach())
    out = dict(L0=float(L0), repro=abs(L0b - float(L0)))
    errs, Ds = [], {}
    for rel_step in (2e-4, 1e-4, 5e-5):                       # parameter step = rel_step * ||theta||
        e = rel_step * pnorm / nrm2 ** 0.5
        with torch.no_grad():
            for p, x in zip(params, g): p.add_(x, alpha=e)
        torch.manual_seed(S); Lp = float(cm(*args).detach())
        with torch.no_grad():
            for p, x in zip(params, g): p.add_(x, alpha=-2 * e)
        torch.manual_seed(S); Lm = float(cm(*args).detach())
        with torch.no_grad():
            for p, x in zip(params, g): p.add_(x, alpha=e)
        Ds[rel_step] = (Lp - Lm) / (2 * e)
        errs.append(abs(Ds[rel_step] - nrm2) / nrm2)
    # Richardson extrapolation cancels the O(h^2) truncation term of the central difference; what is
    # left is O(h^4) plus any step-INDEPENDENT discrepancy, which is what a mask mismatch would produce
    rich = [abs((4 * Ds[h / 2] - Ds[h]) / 3 - nrm2) / nrm2 for h in (2e-4, 1e-4)]
    out["rich"] = rich
    out["rel_err"] = errs
    out["ops"] = dict(OPS)
    del m, cm; torch.cuda.empty_cache()
    return out


rows = [("dropout OFF, budget 1.0 (control)", 1.0, None, True),
        ("dropout OFF, budget 0.05 stock", 0.05, None, True),
        ("dropout ON,  budget 1.0 (control)", 1.0, None, False),
        ("dropout ON,  budget 0.05 stock", 0.05, None, False),
        ("dropout ON,  budget 0.05 stock+remat(pw)", 0.05, "pointwise", False)]
res = {}
for name, b, rm, nd in rows:
    r = check("bert", b, rm, nd); res[name] = r
    print(f"{name:42s} L0={r['L0']:.6e} reseed_repro={r['repro']:.1e}  raw={'/'.join(f'{x:.1e}' for x in r['rel_err'])}  richardson={r['rich'][0]:.1e}/{r['rich'][1]:.1e}  "
          f"random ops fw={r['ops'].get('fw')} bw={r['ops'].get('bw')}", flush=True)
ctrls = [max(res[k]["rich"]) for k in ("dropout OFF, budget 1.0 (control)", "dropout OFF, budget 0.05 stock",
                                          "dropout ON,  budget 1.0 (control)")]
off_ops = [res[k]["ops"].get("fw") for k in ("dropout OFF, budget 1.0 (control)", "dropout OFF, budget 0.05 stock")]
if max(ctrls) > 1e-3 or any(o for o in off_ops):
    print(f"TEST INVALID: controls' Richardson error must be < 1e-3 (got {', '.join(f'{c:.1e}' for c in ctrls)}) "
          f"and dropout-OFF forward random ops must be 0 (got {off_ops}). No verdict.")
    sys.exit(0)
ctrl = max(res["dropout ON,  budget 1.0 (control)"]["rich"])
for name in ("dropout ON,  budget 0.05 stock", "dropout ON,  budget 0.05 stock+remat(pw)"):
    e = res[name]["rich"]
    if res[name]["repro"] > 1e-5:
        v = "INVALID (forward not reproducible under reseeding)"
    elif min(e) > 100 * ctrl:
        v = f"INCONSISTENT: error {min(e):.1e} is >100x the control ({ctrl:.1e}); the gradient is not the gradient of the computed loss"
    elif max(e) <= 10 * ctrl:
        v = "CONSISTENT (within 10x of the control)"
    else:
        v = f"UNCLEAR (between 10x and 100x of the control {ctrl:.1e})"
    print(f"VERDICT {name}: {v}")
