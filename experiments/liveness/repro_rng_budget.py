"""
repro_rng_budget.py -- minimal reproducer: activation_memory_budget + dropout + torch.compile(inductor)
gives gradients that are not the gradient of the computed loss.

No dependencies beyond torch. Model: 8 x (Linear -> GELU -> Dropout(0.1)) with a fixed random
projection loss.
For each budget it compiles fresh, runs one step under the same seed, and reports:
  loss      identical forward => the same dropout masks were used in the forward
  rel_L2    ||g_budget - g_ref|| / ||g_ref||, where g_ref is budget 1.0 (nothing recomputed)
  bw_rand   random ops in the backward graph (> 0 means dropout is regenerated in the backward)
  FD        a central-difference directional-derivative check of the gradient against the loss itself
            (Richardson-extrapolated), independent of the reference

Usage:
  python repro_rng_budget.py --device cuda
  python repro_rng_budget.py --device cpu
  python repro_rng_budget.py --device cuda --functionalize-rng   # torch._functorch.config.functionalize_rng_ops = True
"""
import argparse
import torch, torch.nn as nn
import torch._functorch.config as fc
import torch._functorch.partitioners as P

RAND = ("rand", "seed", "philox", "bernoulli", "dropout")
BW = {}
_orig = P.min_cut_rematerialization_partition


def _count(*a, **k):
    fw, bw = _orig(*a, **k)
    BW["fw"] = sum(1 for n in fw.graph.nodes if n.op == "call_function" and any(w in str(n.target).lower() for w in RAND))
    BW["bw"] = sum(1 for n in bw.graph.nodes if n.op == "call_function" and any(w in str(n.target).lower() for w in RAND))
    return fw, bw


P.min_cut_rematerialization_partition = _count
try:
    import torch._inductor.compile_fx as CF
    CF.min_cut_rematerialization_partition = _count
except Exception:
    pass


class Net(nn.Module):
    def __init__(self, d=256, depth=8, p=0.1):
        super().__init__()
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(p)) for _ in range(depth))
        self.register_buffer("R", torch.randn(d, generator=torch.Generator().manual_seed(7)))

    def forward(self, x):
        for l in self.layers:
            x = x + l(x)
        return (x * self.R).sum() / x.shape[0]


def step(budget, device, backend, seed=1234):
    torch._dynamo.reset()
    torch.manual_seed(0)
    m = Net().to(device)
    x = torch.randn(64, 256, generator=torch.Generator().manual_seed(3)).to(device)
    cm = torch.compile(m, backend=backend, dynamic=False)
    fc.activation_memory_budget = budget
    try:
        torch.manual_seed(seed); cm(x).backward()                  # compile
        m.zero_grad(set_to_none=True)
        torch.manual_seed(seed); L = cm(x); L.backward()
        ps = [p for p in m.parameters()]
        g = [p.grad.detach().double().clone() for p in ps]
        n2 = sum(float((t ** 2).sum()) for t in g)
        pn = sum(float((p.detach().double() ** 2).sum()) for p in ps) ** 0.5
        D = {}
        for r in (2e-4, 1e-4, 5e-5):
            e = r * pn / n2 ** 0.5
            with torch.no_grad():
                for p, t in zip(ps, g): p.add_(t.to(p.dtype), alpha=e)
            torch.manual_seed(seed); Lp = float(cm(x).detach())
            with torch.no_grad():
                for p, t in zip(ps, g): p.add_(t.to(p.dtype), alpha=-2 * e)
            torch.manual_seed(seed); Lm = float(cm(x).detach())
            with torch.no_grad():
                for p, t in zip(ps, g): p.add_(t.to(p.dtype), alpha=e)
            D[r] = (Lp - Lm) / (2 * e)
        fd = max(abs((4 * D[h / 2] - D[h]) / 3 - n2) / n2 for h in (2e-4, 1e-4))
    finally:
        fc.activation_memory_budget = 1.0
    return float(L), [t.cpu() for t in g], dict(BW), fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--backend", default="inductor")
    ap.add_argument("--functionalize-rng", action="store_true")
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.1, 0.0])
    ap.add_argument("--emulate-fix", action="store_true", help="apply rng_fix_emulation (PR #190759, 0<budget<1)")
    a = ap.parse_args()
    if a.emulate_fix:
        import rng_fix_emulation; rng_fix_emulation.install()
    if a.functionalize_rng:
        fc.functionalize_rng_ops = True
    print(f"torch {torch.__version__}  device={a.device}  backend={a.backend}  functionalize_rng_ops={fc.functionalize_rng_ops}")
    ref = None
    for b in a.budgets:
        L, g, bw, fd = step(b, a.device, a.backend)
        if ref is None:
            ref = (L, g)
        num = sum(float((x - y).pow(2).sum()) for x, y in zip(ref[1], g)) ** 0.5
        den = sum(float(x.pow(2).sum()) for x in ref[1]) ** 0.5
        print(f"budget {b:4.2f}  loss={L:.7f} (same forward as budget {a.budgets[0]}: {abs(L - ref[0]) < 1e-6})  "
              f"rel_L2 vs ref={num / den:.2e}  random ops fw={bw.get('fw')} bw={bw.get('bw')}  FD(richardson)={fd:.1e}")


if __name__ == "__main__":
    main()
