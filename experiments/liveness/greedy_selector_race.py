"""
greedy_selector_race.py -- CPU head-to-head: a budget-free greedy save-set selector, reimplemented from its published
description (no code was released; see the paper's related work), vs our methods (LivenessAwareSolver, post-partition use-site remat), all scored by the same exact
static peak oracle (LS.Oracle: solve_min_cut(dont_ban) -> _extract_fwd_bwd_modules -> reorder -> walk).

Decision space shared by every method: which of the partitioner's recomputable banned candidates stay
banned ("saved" index set) vs go into dont_ban. The greedy selector starts from all-banned (= the aggressive min-cut).

One compile per model; all analysis happens inside a CustomKnapsackSolver call (the compile itself uses
the stock plan at the compile budget). Our reimplementation may differ from the original in details.

Usage (repo root, CPU, ~12 min with 2 workers):
  python experiments/liveness/greedy_selector_race.py --model toy --greedy-modes lazy --workers 2 \
      --out experiments/liveness/results/greedy_race_toy.json | tee experiments/liveness/results/greedy_race_toy_log.txt
  python experiments/liveness/greedy_race_analyze.py experiments/liveness/results/greedy_race_toy.json
"""
from __future__ import annotations

import argparse, collections, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(os.path.dirname(HERE)))   # repo root (for ackaudit)

import torch
import torch._functorch.partitioners as P
from torch._functorch import config as fc
from torch._functorch.partitioners import CustomKnapsackSolver
from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.debugging import boxed_nop
from torch.fx._lazy_graph_module import _use_lazy_graph_module
from torch.utils import flop_counter as FC

import liveness_solver as LS
import liveness_oracle_check as A
import remat_at_use as RU

aten = torch.ops.aten
CPU_SDPA = aten._scaled_dot_product_flash_attention_for_cpu

# ---- testbed: CUDA-like candidate set (CPU SDPA is compute-intensive -> knapsack candidate) ----------
_orig_ops = P.get_default_op_list


def _ops():
    o = _orig_ops(); o.compute_intensive_ops.add(CPU_SDPA); return o


P.get_default_op_list = _ops


def register_cpu_sdpa_flops():
    # CUDA flash/efficient SDPA have a FLOP formula; the CPU kernel does not (estimate_runtime -> 1).
    # Give it the same formula so knapsack values and our FLOP metric are CUDA-like.
    FC.register_flop_formula([CPU_SDPA], get_raw=True)(FC.sdpa_flop)


# ---- speed: lazy GraphModule codegen inside the oracle (identical graphs, no python codegen) ---------
_orig_oracle_call = LS.Oracle.__call__


def _lazy_call(self, saved):
    with _use_lazy_graph_module(True):
        return _orig_oracle_call(self, saved)


LS.Oracle.__call__ = _lazy_call
LS._install_hooks()


# ---- FLOP proxy (PyTorch flop formulas; 0 for unregistered = memory-bound ops, as in the original description)
def node_flops(n) -> float:
    if n.op != "call_function" or not isinstance(n.target, torch._ops.OpOverload):
        return 0.0
    f = FC.flop_registry.get(n.target.overloadpacket)
    if f is None:
        return 0.0
    try:
        conv = lambda a: a.meta["val"] if isinstance(a, torch.fx.Node) else a
        args = torch.fx.node.map_aggregate(n.args, conv)
        kwargs = torch.fx.node.map_aggregate(n.kwargs, conv)
        return float(f(*args, out_val=n.meta.get("val"), **kwargs))
    except Exception:
        return 0.0


def gm_flops(gm) -> float:
    return sum(node_flops(n) for n in gm.graph.nodes)


def recompute_flops(o) -> float:
    """FLOPs of forward ops re-executed in the emitted backward."""
    return sum(node_flops(n) for n in o["bw"].graph.nodes if n.op == "call_function" and n.name in o["fw_names"])


class XOracle(LS.Oracle):
    """LS.Oracle + recompute FLOPs, fw peak, and an uncached emit for remat (which mutates bw)."""

    def __call__(self, saved):
        hit = saved in self.cache
        r = super().__call__(saved)
        if not hit:
            r["R"] = recompute_flops(r)
        return r

    def fresh(self, saved):
        c = self.cache; self.cache = {}
        try:
            r = super().__call__(saved)
        finally:
            self.cache = c; self.calls -= 1
        r["R"] = recompute_flops(r)
        return r


def _fwpk(oracle, saved):
    # re-derive the fw peak cheaply: peak = max(fw, bw); if bw dominates we need fw separately
    key = ("fw", saved)
    if key in oracle.cache:
        return oracle.cache[key]
    dont_ban = LS.OrderedSet(c for i, c in enumerate(oracle.cands) if i not in saved)
    with _use_lazy_graph_module(True):
        sv, _ = P.solve_min_cut(oracle.jg, oracle.ni, oracle.opts, dont_ban)
        sym = [n for n in sv if P.is_sym_node(n) and not P._is_assert_only_symbool(n)]
        opq = [n for n in sv if P.is_opaque_node(n)]
        vals = [n for n in sv if not P.is_sym_node(n) and not P.is_opaque_node(n)]
        fw, _ = P._extract_fwd_bwd_modules(oracle.jm, list(vals), saved_sym_nodes=list(sym), saved_opaque_nodes=list(opq),
                                           num_fwd_outputs=LS._CTX["num_fwd_outputs"],
                                           static_lifetime_input_nodes=oracle.ni.static_lifetime_input_nodes)
        fw = LS.raise_getitems(fw)
    v = LS.walk(fw, False)[0]
    oracle.cache[key] = v
    return v


def with_remat(oracle, saved, mode):
    """Peak / FLOPs after post-partition use-site remat of the emitted backward."""
    t0 = time.perf_counter()
    o = oracle.fresh(saved)
    bw = o["bw"]; f0 = gm_flops(bw)
    st = RU.rematerialize_backward(bw, mode=mode)
    added = gm_flops(bw) - f0
    fpk = _fwpk(oracle, saved)
    return dict(peak=max(fpk, st["peak_after"]), R=o["R"] + added, remat_flops=added,
                remat_numel_flops=st["flops_added"], accepted=st["accepted"], seconds=time.perf_counter() - t0)


# ---- greedy selector -------------------------------------------------------------------------------
_G = {}


def _child_eval(saved):
    o = _G["oracle"](saved)
    _G["oracle"].cache.pop(saved, None)          # keep child memory bounded
    return o["peak"], o["R"]


class Evaluator:
    """(peak, recompute FLOPs) per saved-set, cached; optional fork pool (same oracle, same results)."""

    def __init__(self, oracle, workers=1):
        self.o, self.cache, self.calls, self.seconds = oracle, {}, 0, 0.0
        self.pool = None
        if workers > 1:
            import multiprocessing as mp
            _G["oracle"] = oracle
            self.pool = mp.get_context("fork").Pool(workers)

    def __call__(self, sets):
        todo = [s for s in dict.fromkeys(sets) if s not in self.cache]
        t0 = time.perf_counter()
        if todo:
            res = self.pool.map(_child_eval, todo, chunksize=1) if self.pool else [
                (lambda o: (o["peak"], o["R"]))(self.o(s)) for s in todo]
            for s, r in zip(todo, res):
                self.cache[s] = r
            self.calls += len(todo)
        self.seconds += time.perf_counter() - t0
        return [self.cache[s] for s in sets]

    def close(self):
        if self.pool:
            self.pool.close(); self.pool.join()


def greedy_select(ev, n, target=-1.0, max_steps=10**9, time_cap=1e9, lazy_k=None, log=print):
    """As described: start from the min-cut save set (here: every candidate banned); each step try
    dropping each currently-saved candidate (re-derive fw/bw, estimate recompute FLOPs of the new graphs,
    simulate peak); take the cheapest added-FLOPs per byte of peak freed (ties: larger peak drop); stop when
    peak <= target or no drop lowers the peak.
    lazy_k=None: exact (every saved candidate re-evaluated every step).
    lazy_k=K   : lazy greedy (DEVIATION, for runtime): re-evaluate only the K best by their last-known score;
                 if none of those lowers the peak, re-evaluate ALL (so the stopping rule stays exact)."""
    t0 = time.perf_counter(); c0 = ev.calls
    cur = frozenset(range(n)); (pk, R0), = ev([cur])
    traj = [dict(step=0, peak=pk, R=R0, calls=ev.calls - c0, seconds=0.0, saved=sorted(cur), dropped=None)]
    stale = {}                                   # i -> (score, -dpk) from the last evaluation
    step, full_evals = 0, 0
    while pk > target and step < max_steps and time.perf_counter() - t0 < time_cap:
        def score(idx):
            res = ev([cur - {i} for i in idx]); out = {}
            for i, (p, r) in zip(idx, res):
                d = pk - p
                out[i] = (float("inf"), 0.0) if d <= 0 else ((r - R0) / d, -d)
            stale.update(out); return out
        if lazy_k is None or not stale:
            sc = score(sorted(cur)); full_evals += 1
        else:
            order = sorted(cur, key=lambda i: stale.get(i, (-1.0, 0.0)))   # unseen first
            sc = score(order[:lazy_k])
            if all(v[0] == float("inf") for v in sc.values()):
                sc = score(sorted(cur)); full_evals += 1
        cand = [(v, i) for i, v in sc.items() if v[0] != float("inf")]
        if not cand:
            break
        (key, i) = min(cand)
        step += 1
        cur = cur - {i}; (pk2, R2), = ev([cur])
        traj.append(dict(step=step, peak=pk2, R=R2, calls=ev.calls - c0, seconds=time.perf_counter() - t0,
                         saved=sorted(cur), dropped=i, score=key[0], freed=pk - pk2, dcost=R2 - R0))
        pk, R0 = pk2, R2
        stale.pop(i, None)
        log(f"   [greedy{'' if lazy_k is None else '-lazy'}] step {step:3d} drop #{i:3d} peak {pk/1e6:8.2f}MB  R {R0/1e9:8.3f}GF  "
            f"score {key[0]:.3g} F/B  calls {ev.calls - c0}  full {full_evals}  {time.perf_counter() - t0:6.1f}s")
    converged = not (pk > target and (step >= max_steps or time.perf_counter() - t0 >= time_cap))
    return traj, converged, full_evals


# ---- analysis solver ---------------------------------------------------------------------------------
class Race(CustomKnapsackSolver):
    def __init__(self, args):
        self.a = args; self.out = {}

    def uuid(self):
        return None

    def __call__(self, memory, jg, max_memory, ni, cands):
        if self.out:
            return self._stock(memory, max_memory, cands)
        a = self.a; cands = list(cands); n = len(cands)
        rt = [float(LS._RT.get(c, 1.0)) for c in cands]
        names = [c.name for c in cands]
        oracle = XOracle(jg, ni, cands)
        log = lambda *x: print(*x, flush=True)
        out = self.out
        out.update(n=n, cand_ops=dict(collections.Counter(LS._op(c) for c in cands)), names=names)
        log(f"n={n} candidates {out['cand_ops']}")

        # early-return thresholds of choose_saved_values_set (budgets above these never reach the knapsack)
        mco = P.MinCutOptions(ban_if_used_far_apart=fc.ban_recompute_used_far_apart,
                              ban_if_long_fusible_chains=fc.ban_recompute_long_fusible_chains,
                              ban_if_materialized_backward=fc.ban_recompute_materialized_backward,
                              ban_if_not_in_allowlist=fc.ban_recompute_not_in_allowlist,
                              ban_if_reduction=fc.ban_recompute_reductions)
        from dataclasses import replace
        sz = lambda vs: sum(map(P._size_of, vs))
        dflt, _ = P.solve_min_cut(jg, ni, mco)
        mag, _ = P.solve_min_cut(jg, ni, replace(mco, ban_if_used_far_apart=False, ban_if_long_fusible_chains=False,
                                                 ban_if_materialized_backward=False))
        agg, _ = P.solve_min_cut(jg, ni, replace(mco, ban_if_used_far_apart=False, ban_if_long_fusible_chains=False,
                                                 ban_if_materialized_backward=False, ban_if_not_in_allowlist=False))
        mn, mx = sz(ni.inputs), sz(dflt)
        out["ratio_more_aggressive"] = (sz(mag) - mn) / (mx - mn)
        out["ratio_aggressive"] = (sz(agg) - mn) / (mx - mn)
        log(f"early-return thresholds: more_aggressive ratio {out['ratio_more_aggressive']:.3f}, "
            f"aggressive ratio {out['ratio_aggressive']:.3f}")

        def dp(b):
            return frozenset(dp_knapsack(list(memory), list(rt), b)[1])

        def rec(saved, extra=None):
            o = oracle(saved)
            d = dict(peak=o["peak"], R=o["R"], rt_recomp=sum(rt[i] for i in range(n) if i not in saved),
                     n_saved=len(saved), saved=sorted(saved))
            d.update(extra or {})
            return d

        ref = oracle(frozenset(range(n)))
        out["all_banned"] = rec(frozenset(range(n)))
        log(f"all-banned (aggressive min-cut): peak {ref['peak']/1e6:.2f}MB  R {ref['R']/1e9:.3f}GF")

        # --- per-budget: stock, ours, remat variants ---------------------------------------------------
        rows = []
        if a.resume:
            prev = json.load(open(a.resume)); out["rows"] = prev["rows"]; out["stock_sweep"] = prev["stock_sweep"]
            assert prev["names"] == names, "candidate list differs from resumed run"
        for b in ([] if a.resume else a.budgets):
            row = dict(budget=b, reaches_knapsack=b <= out["ratio_aggressive"])
            t = time.perf_counter(); s = dp(b); row["stock"] = rec(s, dict(seconds=time.perf_counter() - t, calls=1))
            solver = LS.LivenessAwareSolver()
            t = time.perf_counter(); sv, _ = solver(memory, jg, b, ni, cands); ts = time.perf_counter() - t
            st = solver.stats[-1]
            row["ours"] = rec(frozenset(sv), dict(seconds=ts, calls=st["oracle_calls"], chosen_is_stock=st["chosen_is_stock"],
                                                  pred_peak=st["predicted_peak_bytes"]))
            for tag, base, mode in (("stock+light", s, "pointwise"), ("stock+all", s, "all"),
                                    ("ours+light", frozenset(sv), "pointwise")):
                r = with_remat(oracle, base, mode)
                r["calls"] = row["stock" if tag.startswith("stock") else "ours"]["calls"]
                r["rt_recomp"] = sum(rt[i] for i in range(n) if i not in base)
                row[tag] = r
            rows.append(row)
            log(f"b={b:.2f} stock {row['stock']['peak']/1e6:7.2f}MB R={row['stock']['R']/1e9:.3f}GF | ours "
                f"{row['ours']['peak']/1e6:7.2f}MB R={row['ours']['R']/1e9:.3f}GF calls={st['oracle_calls']} {ts:.1f}s | "
                f"stock+light {row['stock+light']['peak']/1e6:7.2f} ({row['stock+light']['seconds']:.1f}s) | stock+all "
                f"{row['stock+all']['peak']/1e6:7.2f} | ours+light {row['ours+light']['peak']/1e6:7.2f}")
            out["rows"] = rows
            self._dump()

        # --- stock sweep for the absolute-target framing ------------------------------------------------
        sweep = []
        for k in ([] if a.resume else range(0, 101, a.sweep_step)):
            b = k / 100
            sweep.append(dict(budget=b, **{kk: v for kk, v in rec(dp(b)).items() if kk != "saved"}))
        if not a.resume:
            out["stock_sweep"] = sweep
        self._dump()

        # --- greedy selector to convergence --------------------------------------------------------------------
        for mode in a.greedy_modes:
            lazy = None if mode == "exact" else a.lazy_k
            log(f"greedy selector ({mode}) to convergence ...")
            oracle_g = XOracle(jg, ni, cands)
            ev = Evaluator(oracle_g, a.workers)
            traj, converged, full = greedy_select(ev, n, max_steps=a.greedy_max_steps, time_cap=a.greedy_time_cap, lazy_k=lazy, log=log)
            ev.close()
            for t in traj:
                t["rt_recomp"] = sum(rt[i] for i in range(n) if i not in set(t["saved"]))
            fin = frozenset(traj[-1]["saved"])
            jl, ja = with_remat(oracle_g, fin, "pointwise"), with_remat(oracle_g, fin, "all")
            out[f"greedy_{mode}"] = dict(traj=traj, converged=converged, calls=ev.calls, full_evals=full,
                                     eval_seconds=ev.seconds, workers=a.workers, light=jl, all=ja)
            log(f"greedy-{mode} final: peak {traj[-1]['peak']/1e6:.2f}MB R={traj[-1]['R']/1e9:.3f}GF steps={len(traj)-1} "
                f"calls={ev.calls} {traj[-1]['seconds']:.1f}s converged={converged}; +light {jl['peak']/1e6:.2f}MB "
                f"R={jl['R']/1e9:.3f}GF; +all {ja['peak']/1e6:.2f}MB R={ja['R']/1e9:.3f}GF")
            self._dump()
        self._dump()
        return self._stock(memory, max_memory, cands)

    def _stock(self, memory, max_memory, cands):
        rt = [float(LS._RT.get(c, 1.0)) for c in cands]
        s = dp_knapsack(list(memory), rt, max_memory)[1]
        return sorted(s), [i for i in range(len(cands)) if i not in set(s)]

    def _dump(self):
        json.dump(self.out, open(self.a.out, "w"), indent=1, default=str)


def build(model):
    torch.manual_seed(197838)
    if model == "toy":
        m = A._Toy(L=16); args = (torch.randint(0, 1000, (1, 512)), torch.randint(0, 1000, (1, 512)))
    else:
        from ackaudit.hf_models import hf_models
        b, mk = hf_models(1)["bert"]; m = b().eval()   # eval(): dropout off (no RNG ops in the graph)
        args = mk()
    return m, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="toy")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--sweep-step", type=int, default=2)
    ap.add_argument("--greedy-max-steps", type=int, default=10**9)
    ap.add_argument("--greedy-time-cap", type=float, default=2400)
    ap.add_argument("--no-sdpa-flops", action="store_true")
    ap.add_argument("--greedy-modes", nargs="+", default=["exact"])
    ap.add_argument("--lazy-k", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(HERE, "results", f"greedy_race_{a.model}.json")
    if not a.no_sdpa_flops:
        register_cpu_sdpa_flops()
    if a.workers > 1:
        torch.set_num_threads(1)
    m, args = build(a.model)
    torch._dynamo.reset()
    be = aot_autograd(fw_compiler=boxed_nop, bw_compiler=boxed_nop, partition_fn=P.min_cut_rematerialization_partition,
                      keep_inference_input_mutations=True)
    solver = Race(a)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    t0 = time.perf_counter()
    try:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = min(a.budgets), solver
        torch.compile(m, backend=be, dynamic=False)(*args).backward()
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
    solver.out["total_seconds"] = time.perf_counter() - t0
    solver._dump()
    print(f"done in {time.perf_counter() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
