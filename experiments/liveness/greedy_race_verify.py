"""greedy_race_verify.py -- compile selected plans from results/greedy_race_<model>.json for real (aot_eager, CPU), and check
 (1) the oracle's predicted step peak == static walk of the modules the backend actually received, and
 (2) the static backward new-allocation peak vs the measured LiveBytes (dispatch-level) backward peak.
Remat variants apply remat_at_use.rematerialize_backward inside the bw compiler."""
import json, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import greedy_selector_race as R   # installs the same testbed patches
import torch, torch._functorch.partitioners as P
from torch._functorch import config as fc
from torch._functorch.partitioners import CustomKnapsackSolver
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.debugging import boxed_nop
LS, A, RU = R.LS, R.A, R.RU


class Fixed(CustomKnapsackSolver):
    def __init__(s, names): s.names = set(names)
    def uuid(s): return None
    def __call__(s, memory, jg, mm, ni, cands):
        cands = list(cands); sv = [i for i, c in enumerate(cands) if c.name in s.names]
        assert len(sv) == len(s.names), "candidate names differ between compiles"
        return sv, [i for i in range(len(cands)) if i not in set(sv)]


def measure(model, names, remat=None, repeats=2):
    m, args = R.build(model); torch._dynamo.reset(); cap = {}

    def fwc(g, ex): cap["fw"] = g; return boxed_nop(g, ex)

    def bwc(g, ex):
        if remat: RU.rematerialize_backward(g, mode=remat)
        cap["bw"] = g; return boxed_nop(g, ex)
    be = aot_autograd(fw_compiler=fwc, bw_compiler=bwc, partition_fn=P.min_cut_rematerialization_partition,
                      keep_inference_input_mutations=True)
    cm = torch.compile(m, backend=be, dynamic=False)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = 0.05, Fixed(names)
        cm(*args).backward()
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
    static = max(LS.walk(cap["fw"], False)[0], LS.walk(cap["bw"], True)[0])
    static_bw_new = LS.walk(cap["bw"], False)[0]
    peaks = []
    for _ in range(repeats):
        m.zero_grad(set_to_none=False); loss = cm(*args); mode = A._LiveBytes()
        with mode:
            loss.backward()
        peaks.append(mode.peak); del loss
    return static, static_bw_new, statistics.median(peaks)


def main():
    model = sys.argv[1]
    R.register_cpu_sdpa_flops()
    d = json.load(open(os.path.join(HERE, "results", f"greedy_race_{model}.json")))
    jk = "greedy_exact" if "greedy_exact" in d else "greedy_lazy"; d["g"] = d[jk]; d["g+light"] = d[jk]["light"]
    names = d["names"]; nm = lambda idx: [names[i] for i in idx]
    rows = d["rows"]; r05 = rows[0]
    plans = [("stock@0.05", nm(r05["stock"]["saved"]), None, r05["stock"]["peak"]),
             ("ours@0.05", nm(r05["ours"]["saved"]), None, r05["ours"]["peak"]),
             ("stock@0.05+light", nm(r05["stock"]["saved"]), "pointwise", r05["stock+light"]["peak"]),
             ("greedy final", nm(d["g"]["traj"][-1]["saved"]), None, d["g"]["traj"][-1]["peak"]),
             ("greedy final+light", nm(d["g"]["traj"][-1]["saved"]), "pointwise", d["g+light"]["peak"])]
    for label, ns, remat, pred in plans:
        st, sbn, meas = measure(model, ns, remat)
        print(f"{model:4s} {label:18s} oracle-pred {pred/1e6:8.3f}MB  emitted-static {st/1e6:8.3f}MB  "
              f"{'MATCH' if abs(st - pred) <= max(1, 1e-3 * pred) else 'DIFF '}  | bw new-alloc static {sbn/1e6:8.3f}MB "
              f"measured {meas/1e6:8.3f}MB  ({100 * (meas - sbn) / sbn:+.2f}%)", flush=True)


if __name__ == "__main__":
    main()
