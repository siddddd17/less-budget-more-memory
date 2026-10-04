"""
repro_rng_bert_cpu.py -- does Inductor redraw prims.inductor_seeds in the backward at 0 < budget < 1
for a small BERT, and does the knapsack-only fix (PR #190759-style) leave that unfixed? (LOCAL ONLY)

Modes:
  none : stock torch
  v1   : RNG ops excluded from knapsack candidates/dont_ban (rng_fix_emulation, the PR #190759 approach)
  v2   : v1 + RNG ops (is_rng_op, which includes prims.inductor_seeds) banned from recompute in the
         aggressive min-cut options too
Reports, per budget: backward RNG op names, gradient rel_L2 vs budget 1.0, and the Richardson FD check.
"""
import argparse, collections, sys
import torch, torch.nn as nn
import torch._functorch.config as fc
import torch._functorch.partitioners as P
import torch._inductor.config as IC
import torch._inductor.compile_fx as CF
IC.force_disable_caches = True

RAND = ("rand", "seed", "philox", "bernoulli", "dropout")
NAMES = {}
_orig = P.min_cut_rematerialization_partition


def _cap(*a, **k):
    fw, bw = _orig(*a, **k)
    f = lambda gm: dict(collections.Counter(str(n.target).replace(".default", "") for n in gm.graph.nodes
                                            if n.op == "call_function" and any(w in str(n.target).lower() for w in RAND)))
    NAMES["fw"], NAMES["bw"] = f(fw), f(bw)
    return fw, bw


P.min_cut_rematerialization_partition = _cap
CF.min_cut_rematerialization_partition = _cap


def install(mode):
    if mode in ("v1", "v2"):
        import rng_fix_emulation; rng_fix_emulation.install()
    if mode == "v2":
        orig_is_random = P.OpTypes.is_random
        P.OpTypes.is_random = lambda self, n: orig_is_random(self, n) or P.is_rng_op(n)


def build(layers):
    from transformers import BertConfig, BertModel
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=1000, hidden_size=128, num_hidden_layers=layers, num_attention_heads=4,
                     intermediate_size=512, max_position_embeddings=512, attn_implementation="eager")
    m = BertModel(cfg)
    R = torch.randn(128, generator=torch.Generator().manual_seed(4242))
    class W(nn.Module):
        def __init__(s): super().__init__(); s.m = m; s.register_buffer("R", R)
        def forward(s, ids): return (s.m(input_ids=ids).last_hidden_state * s.R).sum() / ids.numel()
    return W().train()


def step(budget, layers, seed=1234):
    torch._dynamo.reset()
    w = build(layers); ids = torch.randint(0, 1000, (2, 256), generator=torch.Generator().manual_seed(3))
    cm = torch.compile(w, backend="inductor", dynamic=False)
    fc.activation_memory_budget = budget
    try:
        torch.manual_seed(seed); cm(ids).backward()
        w.zero_grad(set_to_none=True)
        torch.manual_seed(seed); L = cm(ids); L.backward()
        ps = [p for p in w.parameters() if p.grad is not None]
        g = [p.grad.detach().double().clone() for p in ps]
        n2 = sum(float((t ** 2).sum()) for t in g); pn = sum(float((p.detach().double() ** 2).sum()) for p in ps) ** 0.5
        D = {}
        for r in (2e-4, 1e-4, 5e-5):
            e = r * pn / n2 ** 0.5
            with torch.no_grad():
                for p, t in zip(ps, g): p.add_(t.float(), alpha=e)
            torch.manual_seed(seed); Lp = float(cm(ids).detach())
            with torch.no_grad():
                for p, t in zip(ps, g): p.add_(t.float(), alpha=-2 * e)
            torch.manual_seed(seed); Lm = float(cm(ids).detach())
            with torch.no_grad():
                for p, t in zip(ps, g): p.add_(t.float(), alpha=e)
            D[r] = (Lp - Lm) / (2 * e)
        fd = max(abs((4 * D[h / 2] - D[h]) / 3 - n2) / n2 for h in (2e-4, 1e-4))
    finally:
        fc.activation_memory_budget = 1.0
    return float(L), g, dict(NAMES), fd


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="none", choices=["none", "v1", "v2"])
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.3, 0.1, 0.05])
    a = ap.parse_args(); install(a.mode)
    ref = None
    for b in a.budgets:
        L, g, names, fd = step(b, a.layers)
        ref = ref or (L, g)
        rel = sum(float((x - y).pow(2).sum()) for x, y in zip(ref[1], g)) ** 0.5 / sum(float(x.pow(2).sum()) for x in ref[1]) ** 0.5
        print(f"[{a.mode}] budget {b:.2f} same_fwd={abs(L - ref[0]) < 1e-6} rel_L2={rel:.2e} FD={fd:.1e} bw_rng={names.get('bw')}", flush=True)
