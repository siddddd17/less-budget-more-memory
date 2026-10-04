"""
check_solver_rand4x.py -- does Inductor's CUDA 4x random path (PyTorch issue #198333) cause the solver's wrong BERT
gradients (check_solver_bert.py, bisect_solver_bert.py)?

Since PyTorch 2.13, Inductor's Triton code generator draws random numbers with triton_helpers.rand4x in 1-D kernels and
with tl.rand in 2-D kernels; the two give different values for the same (seed, offset). Dropout masks recomputed in
the backward are regenerated from the saved seed, so if the backward kernel's dimensionality differs from the
forward's, the masks differ. rand4x_switch.py turns that path off and records which function each fw/bw kernel used.

For each mode (on = PyTorch's default, off = tl.rand everywhere), with dropout ON, the #190758 fix emulated, the
projection loss and reseeding (as in check_solver_bert.py):
  budget 1.0 (reference) | budget 0.05 stock | budget 0.05 LivenessAwareSolver
Each plan: rel_L2 vs that mode's budget-1.0 reference (the forward's random numbers depend on the mode), peak memory
(median of 3 steps) and step time (median of 20, CUDA events), and the probe of fw/bw random functions.

Predictions (fixed before running): on -> solver WRONG (> 1e-3) and the probe shows fw and bw using different
functions; off -> solver CORRECT (<= 1e-5); stock CORRECT in both modes.
Usage: python experiments/liveness/check_solver_rand4x.py   (~25-30 min on the GTX 1650; 6 Inductor compiles)
CPU smoke: CHECK_DEVICE=cpu CHECK_SCALE=1 (the 4x path does not exist on CPU; plumbing only)
"""
import os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rng_fix_emulation
rng_fix_emulation.install()
import rand4x_switch as R4
R4.install("on")
import liveness_oracle_check as A
import liveness_solver as LS
import torch
from torch._functorch import config as fc
import torch._inductor.config as IC
IC.force_disable_caches = True   # never reuse a compiled plan across configs or modes

DEV = os.environ.get("CHECK_DEVICE", "cuda")
SCALE = int(os.environ.get("CHECK_SCALE", "8"))
OK, BAD, S = 1e-5, 1e-3, 1234


def build():
    torch.manual_seed(197838)
    b, mk = A.resolve("bert", SCALE)
    m = b().to(DEV).train()
    R = torch.randn(m.model.config.hidden_size, generator=torch.Generator().manual_seed(4242)).to(DEV)
    inner = m.model
    m.forward = lambda ids, _i=inner, _R=R: (_i(input_ids=ids).last_hidden_state * _R).sum() / ids.numel()
    return m, tuple(x.to(DEV) for x in mk())


def step(m, cm, args):
    torch.manual_seed(S); out = cm(*args); out.backward(); return out


def run(budget, solver, measure):
    torch._dynamo.reset(); R4.reset_stats()
    m, args = build()
    cm = torch.compile(m, backend="inductor", dynamic=False)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = budget, solver
        step(m, cm, args)                                   # compile step
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
    m.zero_grad(set_to_none=True)
    step(m, cm, args)
    g = {n: p.grad.detach().double().cpu() for n, p in m.named_parameters() if p.grad is not None}
    peak = ms = float("nan")
    if measure and DEV == "cuda":
        m.zero_grad(set_to_none=False); torch.cuda.synchronize(); torch.cuda.empty_cache()
        resident, peaks = torch.cuda.memory_allocated(), []
        for _ in range(3):
            torch.cuda.reset_peak_memory_stats(); step(m, cm, args); torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - resident); m.zero_grad(set_to_none=False)
        for _ in range(3):
            step(m, cm, args); m.zero_grad(set_to_none=False)
        t = []
        for _ in range(20):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); step(m, cm, args); e.record(); torch.cuda.synchronize()
            t.append(s.elapsed_time(e)); m.zero_grad(set_to_none=False)
        peak, ms = statistics.median(peaks) / 1e6, statistics.median(t)
    probe, mixed = R4.summary(), R4.mixed_fw_bw()
    del m, cm
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return g, peak, ms, probe, mixed


def rel(ref, g):
    num = sum(float((ref[k] - g[k]).pow(2).sum()) for k in ref) ** 0.5
    return num / max(sum(float(ref[k].pow(2).sum()) for k in ref) ** 0.5, 1e-30)


def word(r):
    return "CORRECT" if r <= OK else ("WRONG" if r > BAD else "UNCLEAR")


print(f"torch {torch.__version__}  device {DEV}  scale {SCALE}  4x path present: {R4.STATE['patched'] is True}", flush=True)
res = {}
for mode in ("on", "off"):
    R4.install(mode)
    t0 = time.perf_counter()
    ref = run(1.0, A.PlanSolver(), False)[0]
    print(f"[{mode}] budget 1.0 reference ({time.perf_counter() - t0:.0f}s)", flush=True)
    for name, mk in (("stock", A.PlanSolver), ("solver", LS.LivenessAwareSolver)):
        g, peak, ms, probe, mixed = run(0.05, mk(), True)
        r = rel(ref, g); res[(mode, name)] = (r, peak, ms, mixed)
        print(f"  [{mode:3s}] 0.05 {name:7s} rel_L2={r:.2e} {word(r):7s} peak={peak:.2f}MB step={ms:.1f}ms", flush=True)
        print(f"        probe: {probe}{'   <-- fw and bw use DIFFERENT random functions' if mixed else ''}", flush=True)

print()
on, off = res[("on", "solver")], res[("off", "solver")]
stock_ok = all(res[(m_, "stock")][0] <= OK for m_ in ("on", "off"))
if DEV != "cuda":
    print("CPU smoke run: the 4x path does not exist on CPU; no verdict.")
elif not stock_ok:
    print("VERDICT INVALID: stock is not correct in both modes (fix emulation / reseeding not working).")
elif on[0] > BAD and off[0] <= OK and on[3]:
    print(f"VERDICT CONFIRMED: solver WRONG with the 4x path on ({on[0]:.1e}) and CORRECT with it off ({off[0]:.1e}); "
          f"fw and bw used different random functions. Peak {on[1]:.1f} -> {off[1]:.1f} MB, step {on[2]:.1f} -> {off[2]:.1f} ms.")
elif on[0] <= BAD:
    print(f"VERDICT NOT REPRODUCED: solver is not WRONG with the 4x path on ({on[0]:.1e}).")
else:
    print(f"VERDICT NOT CONFIRMED: on {on[0]:.1e} ({word(on[0])}), off {off[0]:.1e} ({word(off[0])}), mixed fw/bw {on[3]}.")
