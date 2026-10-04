"""
check_rng_bert.py -- are the BERT/Inductor gradient differences dropout RNG (pre-existing) or a remat bug?

Test 1, dropout ON, Inductor, BERT scale 8. Each run is compared with budget = 1.0, where
nothing is recomputed, so the backward cannot regenerate randomness:
    budget 1.0 (reference) | budget 0.05 stock | budget 0.05 stock+remat(pw)
  It also counts random ops in the forward and backward graphs of each plan.
  If the budget-0.05 *stock* plan already differs from budget 1.0 AND its backward contains random
  ops, the partitioner recomputes dropout in the backward with fresh randomness. That would be a
  pre-existing PyTorch issue, independent of our pass.

Test 2, dropout OFF (every dropout probability set to 0), same configs plus remat(all):
  With no randomness, every config must match budget 1.0 to float noise (rel_L2 < 1e-5).
  That is the correctness test for our pass on BERT.

Usage: python experiments/liveness/check_rng_bert.py   (~30-40 min, Inductor compiles)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_eval_remat as G, liveness_oracle_check as A, liveness_solver as LS
import torch, torch.nn as nn

RAND_WORDS = ("rand", "seed", "dropout", "philox", "bernoulli")
OPS = {}
_inner = G._real_partition


def _counting_partition(*a, **k):
    fw, bw = _inner(*a, **k)
    def count(gm):
        return sum(1 for n in gm.graph.nodes if n.op == "call_function"
                   and any(w in str(n.target).lower() for w in RAND_WORDS))
    OPS["fw"], OPS["bw"] = count(fw), count(bw)
    return fw, bw


G._real_partition = _counting_partition

_orig_resolve = A.resolve
NO_DROPOUT = {"on": False}


def _resolve(model, scale):
    build, make_inputs = _orig_resolve(model, scale)
    def build2():
        m = build()
        if NO_DROPOUT["on"]:
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
        return m
    return build2, make_inputs


A.resolve = _resolve


def rel(a, b):
    num = sum(float((x - y).double().pow(2).sum()) for x, y in zip(a, b)) ** 0.5
    den = sum(float(x.double().pow(2).sum()) for x in a) ** 0.5
    return num / max(den, 1e-30)


for label, nodrop in (("TEST 1: dropout ON", False), ("TEST 2: dropout OFF", True)):
    NO_DROPOUT["on"] = nodrop
    print(f"\n{label}", flush=True)
    runs = [("budget 1.0", 1.0, None), ("budget 0.05 stock", 0.05, None), ("budget 0.05 stock+remat(pw)", 0.05, "pointwise")]
    if nodrop:
        runs.append(("budget 0.05 stock+remat(all)", 0.05, "all"))
    ref = None
    for name, b, rm in runs:
        OPS.clear()
        _, g = G.run("bert", b, "inductor", A.PlanSolver(), rm, 1, 1)
        ref = g if ref is None else ref
        print(f"   {name:30s} rel_L2 vs budget 1.0 = {rel(ref, g):.2e}   random ops: fw={OPS.get('fw')} bw={OPS.get('bw')}", flush=True)
print("\nReading: TEST 2 rel_L2 < 1e-5 for every row means the pass is correct on BERT. In TEST 1, a budget-0.05 *stock* "
      "row far from budget 1.0 with bw random ops > 0 means the partitioner regenerates dropout randomness in the backward "
      "(a pre-existing issue, not ours).")
