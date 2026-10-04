"""
eval_liveness_solver.py -- evaluate LivenessAwareSolver against the stock dp solver (CPU, aot_eager).

For each (setup, budget): compile with the stock solver and with LivenessAwareSolver, capture
the fw/bw modules the backend receives, and report:
  * static step peak (the oracle quantity), stock vs solver, and the gain
  * invariant: the solver's predicted peak == the static peak of the emitted modules
  * measured backward allocations (CPU dispatch tracker), checked against the static walk
  * retained FLOP objective, oracle calls, solver seconds
Setups:
  cuda-like : CPU SDPA added to the compute-intensive list, mirroring the CUDA candidate set
  cpu       : the natural CPU candidate set
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import liveness_solver as LS
import liveness_oracle_check as A   # toy model + CPU allocation tracker

import torch
import torch._functorch.partitioners as P
from torch._functorch import config as fc
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.debugging import boxed_nop

_orig_ops = P.get_default_op_list
CUDA_LIKE = {"on": False}


def _ops():
    o = _orig_ops()
    if CUDA_LIKE["on"]:
        o.compute_intensive_ops.add(torch.ops.aten._scaled_dot_product_flash_attention_for_cpu)
    return o


P.get_default_op_list = _ops


def compile_and_measure(solver, budget, layers, seq, repeats=2):
    torch.manual_seed(197838); torch._dynamo.reset()
    cap = {}

    def fwc(g, ex):
        cap["fw"] = g; return boxed_nop(g, ex)

    def bwc(g, ex):
        cap["bw"] = g; return boxed_nop(g, ex)

    be = aot_autograd(fw_compiler=fwc, bw_compiler=bwc, keep_inference_input_mutations=True,
                      partition_fn=P.min_cut_rematerialization_partition)
    m = A._Toy(L=layers)
    ids = torch.randint(0, 1000, (1, seq)); tgt = torch.randint(0, 1000, (1, seq))
    cm = torch.compile(m, backend=be, dynamic=False)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    t0 = time.perf_counter()
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = budget, solver
        cm(ids, tgt).backward()
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
    compile_s = time.perf_counter() - t0
    static = max(LS.walk(cap["fw"], False)[0], LS.walk(cap["bw"], True)[0])
    static_bw_new = LS.walk(cap["bw"], False)[0]
    peaks = []
    for _ in range(repeats):
        m.zero_grad(set_to_none=False)
        loss = cm(ids, tgt); mode = A._LiveBytes()
        with mode:
            loss.backward()
        peaks.append(mode.peak); del loss
    ok = LS.verify_emitted(cap["fw"], cap["bw"], solver) if isinstance(solver, LS.LivenessAwareSolver) else None
    return dict(static_MB=static / 1e6, measured_bw_MB=statistics.median(peaks) / 1e6,
                static_bw_new_MB=static_bw_new / 1e6, compile_s=compile_s, invariant=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", nargs="+", default=["cuda-like", "cpu"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--layers", type=int, default=16); ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default="eval_liveness_solver.json")
    a = ap.parse_args()
    rows = []
    for setup in a.setups:
        CUDA_LIKE["on"] = setup == "cuda-like"
        for b in a.budgets:
            s = compile_and_measure("dp", b, a.layers, a.seq)
            solver = LS.LivenessAwareSolver()
            l = compile_and_measure(solver, b, a.layers, a.seq)
            st = solver.stats[-1]
            r = dict(setup=setup, budget=b, stock=s, liveness=l, solver=st)
            rows.append(r)
            gain = 1 - l["static_MB"] / s["static_MB"]
            print(f"{setup:9s} {b:.2f}  stock {s['static_MB']:6.1f}MB  ->  liveness {l['static_MB']:6.1f}MB "
                  f"({-100 * gain:+5.1f}%)  retained={st['retained']:.3f}  calls={st['oracle_calls']:2d} "
                  f"solver={st['solver_seconds']:5.1f}s  invariant={'OK' if l['invariant'] else 'MISMATCH'}  "
                  f"meas/static bw: stock {s['measured_bw_MB']:.1f}/{s['static_bw_new_MB']:.1f}  "
                  f"liveness {l['measured_bw_MB']:.1f}/{l['static_bw_new_MB']:.1f}  added={st['added_ops']}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
