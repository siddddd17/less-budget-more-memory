"""
liveness_solver.py -- opt-in liveness-aware solver for torch.compile's activation memory budget
(research prototype for PyTorch #197838; LOCAL ONLY, not an upstream patch).

    import liveness_solver
    torch._functorch.config.activation_memory_budget = 0.05
    torch._functorch.config.activation_memory_budget_solver = liveness_solver.LivenessAwareSolver()

What it does, inside choose_saved_values_set:
  1. Solve the production dp_knapsack as usual (the stock plan).
  2. Emit that plan exactly as the partitioner would (solve_min_cut with the same
     options + dont_ban, _extract_fwd_bwd_modules, reordering_to_mimic_autograd_engine,
     raise_getitems) and walk the emitted graphs to get the physical peak and the tensors
     live at it.
  3. Peak-attributed repricing: the *recomputed* tensors live at the backward peak are
     credited to the nearest unsaved candidates downstream of them in the recompute
     subgraph (the recomputations that pulled them in early). Knapsack values become
     runtime + lambda * credit, and DP is re-solved. Repeat for a few rounds.
  4. Cost recovery: bisect lambda down to the smallest value that keeps the best peak
     (within PEAK_TOL), so the plan gives up as little recompute value as possible.
  5. Return the best plan, or the stock plan if the predicted gain is below MIN_GAIN.

Guarantees (by construction, not by hope):
  * Budget semantics unchanged: every candidate plan is produced by the production
    dp_knapsack under the same capacity, so quantized weight <= budget.
  * Never predicted worse than stock: stock is always in the candidate set.
  * The chosen plan is exactly what the partitioner then emits (the oracle runs the same
    pipeline); `verify_emitted()` checks this after compilation.

Oracle accuracy (measured, GTX 1650, torch 2.14): aot_eager predicts the peak to ~1% (byte-exact
on CPU). Under Inductor, this fx-level walk ranks plans correctly (Spearman 0.995 Llama,
1.000 BERT) but its absolute error is larger. An Inductor-native oracle via
torch/_inductor/memory.py is future work.
"""
from __future__ import annotations

import collections
import time
from typing import Any

import torch
import torch._functorch.partitioners as P
from torch._functorch import config as fc
from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
from torch._functorch.compile_utils import raise_getitems
from torch._functorch.partitioners import CustomKnapsackSolver, _size_of
from torch.utils._ordered_set import OrderedSet

S_QUANT = 10000          # dp_knapsack's quantization scale
_VIEWS = {"view", "_unsafe_view", "reshape", "expand", "t", "transpose", "permute", "slice",
          "select", "unsqueeze", "squeeze", "alias", "as_strided", "detach"}

# ----------------------------------------------------------------------------------------
# hooks: the solver API does not receive runtimes or num_fwd_outputs, so record them
# ----------------------------------------------------------------------------------------
_RT: dict = {}
_CTX: dict = {}
_installed = False


def _install_hooks() -> None:
    global _installed
    if _installed:
        return
    orig_est, orig_classify = P.estimate_runtime, P.classify_nodes

    def est(node):
        v = orig_est(node)
        _RT[node] = v
        return v

    def classify(joint_module, static_lifetime_input_indices, num_fwd_outputs, *a, **k):
        _CTX.update(joint_module=joint_module, num_fwd_outputs=num_fwd_outputs)
        return orig_classify(joint_module, static_lifetime_input_indices, num_fwd_outputs, *a, **k)

    P.estimate_runtime, P.classify_nodes = est, classify
    _installed = True


def _op(n) -> str:
    t = str(getattr(n, "target", ""))
    return t.split(".")[1] if t.startswith("aten.") else t


def _q(x: float) -> int:
    return round(x * S_QUANT)


# ----------------------------------------------------------------------------------------
# emitted-graph liveness
# ----------------------------------------------------------------------------------------
def _alloc(n) -> int:
    if n.op != "call_function" or isinstance(n.meta.get("val"), (tuple, list)) or _op(n) in _VIEWS:
        return 0
    try:
        return _size_of(n)
    except Exception:
        return 0


def walk(gm, inputs_live: bool) -> tuple[int, dict]:
    """Peak bytes and {node: bytes} live at the peak. Allocate at definition, free after the
    last use (views keep their base alive); non-primal placeholders start live if asked."""
    nodes = list(gm.graph.nodes); idx = {n: i for i, n in enumerate(nodes)}
    root = {}
    for n in nodes:
        root[n] = root[n.all_input_nodes[0]] if (n.op == "call_function" and _op(n) in _VIEWS
                                                 and n.all_input_nodes) else n
    last: dict = {}
    for n in nodes:
        for a in n.all_input_nodes:
            last[root[a]] = max(last.get(root[a], -1), idx[n])
    live: dict = {}; frees = collections.defaultdict(list); cur = 0
    if inputs_live:
        for n in nodes:
            if n.op == "placeholder" and not n.name.startswith("primals") and "val" in n.meta:
                s = _size_of(n); live[n] = s; cur += s; frees[last.get(n, idx[n])].append(n)
    peak, at_peak = cur, dict(live)
    for n in nodes:
        s = _alloc(n) if root[n] is n else 0
        if s:
            live[n] = s; cur += s; frees[last.get(n, idx[n])].append(n)
        if cur > peak:
            peak, at_peak = cur, dict(live)
        for d in frees.pop(idx[n], []):
            cur -= live.pop(d, 0)
    return peak, at_peak


class Oracle:
    def __init__(self, joint_graph, node_info, cands):
        self.jg, self.ni, self.cands = joint_graph, node_info, cands
        self.jm = joint_graph.owning_module or _CTX["joint_module"]
        o = P.MinCutOptions(
            ban_if_used_far_apart=fc.ban_recompute_used_far_apart,
            ban_if_long_fusible_chains=fc.ban_recompute_long_fusible_chains,
            ban_if_materialized_backward=fc.ban_recompute_materialized_backward,
            ban_if_not_in_allowlist=fc.ban_recompute_not_in_allowlist,
            ban_if_reduction=fc.ban_recompute_reductions)
        from dataclasses import replace
        self.opts = replace(o, ban_if_used_far_apart=False, ban_if_long_fusible_chains=False,
                            ban_if_materialized_backward=False, ban_if_not_in_allowlist=False)
        self.cache: dict = {}; self.calls = 0; self.seconds = 0.0

    def __call__(self, saved: frozenset) -> dict:
        if saved in self.cache:
            return self.cache[saved]
        t0 = time.perf_counter()
        dont_ban = OrderedSet(c for i, c in enumerate(self.cands) if i not in saved)
        sv, _ = P.solve_min_cut(self.jg, self.ni, self.opts, dont_ban)
        sym = [n for n in sv if P.is_sym_node(n) and not P._is_assert_only_symbool(n)]
        opq = [n for n in sv if P.is_opaque_node(n)]
        vals = [n for n in sv if not P.is_sym_node(n) and not P.is_opaque_node(n)]
        fw, bw = P._extract_fwd_bwd_modules(
            self.jm, list(vals), saved_sym_nodes=list(sym), saved_opaque_nodes=list(opq),
            num_fwd_outputs=_CTX["num_fwd_outputs"],
            static_lifetime_input_nodes=self.ni.static_lifetime_input_nodes)
        fw = raise_getitems(fw)
        bw = raise_getitems(P.reordering_to_mimic_autograd_engine(bw))
        fpk, _ = walk(fw, inputs_live=False)
        bpk, at = walk(bw, inputs_live=True)
        fw_names = {n.name for n in fw.graph.nodes if n.op == "call_function"}
        r = dict(peak=max(fpk, bpk), bw_peak=bpk, at_peak=at, bw=bw, fw_names=fw_names)
        self.cache[saved] = r
        self.calls += 1; self.seconds += time.perf_counter() - t0
        return r


def attribute(o: dict, cand_names: set, saved_names: set) -> collections.Counter:
    remat = {n.name for n in o["bw"].graph.nodes if n.op == "call_function" and n.name in o["fw_names"]}
    credit: collections.Counter = collections.Counter()
    for t, sz in o["at_peak"].items():
        if t.name not in remat:
            continue
        seen, stack, hits = set(), [t], set()
        while stack:
            x = stack.pop()
            for u in x.users:
                if u in seen or u.name not in remat:
                    continue
                seen.add(u)
                if u.name in cand_names and u.name not in saved_names:
                    hits.add(u.name)
                else:
                    stack.append(u)
        for h in hits:
            credit[h] += sz / len(hits)
    return credit


# ----------------------------------------------------------------------------------------
class LivenessAwareSolver(CustomKnapsackSolver):
    def __init__(self, rounds: int = 4, recovery_steps: int = 3, min_gain: float = 0.02,
                 peak_tol: float = 0.01, min_retained: float | None = None, verbose: bool = False):
        _install_hooks()
        self.rounds, self.recovery_steps = rounds, recovery_steps
        self.min_gain, self.peak_tol, self.min_retained, self.verbose = min_gain, peak_tol, min_retained, verbose
        self.stats: list[dict] = []

    def uuid(self):
        return None  # never cache-hit across different solver settings

    @property
    def rec(self) -> dict:
        """Last call's summary, in the shape the experiment harnesses expect."""
        if not self.stats:
            return {}
        s = dict(self.stats[-1])
        s["predicted_peak_MB"] = s["predicted_peak_bytes"] / 1e6
        s["saved_weight"] = s.get("weight", float("nan"))
        return s

    def __call__(self, memory, joint_graph, max_memory, node_info, cands):
        t0 = time.perf_counter()
        cands = list(cands); n = len(cands)
        rt = [float(_RT.get(c, 1.0)) for c in cands]
        names = [c.name for c in cands]; cand_names = set(names)
        oracle = Oracle(joint_graph, node_info, cands)
        rmax = max(rt) if rt else 1.0

        def dp(values) -> frozenset:
            return frozenset(dp_knapsack(list(memory), list(values), max_memory)[1])

        def retained(s):
            base = sum(rt[i] for i in stock) or 1.0
            return sum(rt[i] for i in s) / base

        stock = dp(rt)
        tried: dict[frozenset, float] = {stock: oracle(stock)["peak"]}

        # --- peak-attributed repricing -------------------------------------------------
        credit_total: collections.Counter = collections.Counter()
        cur = stock
        for _ in range(self.rounds):
            c = attribute(oracle(cur), cand_names, {names[i] for i in cur})
            if not c:
                break
            mx = max(c.values())
            for k, v in c.items():
                credit_total[k] += v / mx
            cmax = max(credit_total.values())
            values = [rt[i] + rmax * credit_total.get(names[i], 0.0) / cmax for i in range(n)]
            cur = dp(values)
            if cur in tried:
                break
            tried[cur] = oracle(cur)["peak"]

        best_peak = min(tried.values())
        # --- cost recovery: smallest lambda that keeps the best peak -------------------
        if credit_total and best_peak < tried[stock]:
            cmax = max(credit_total.values())
            lo, hi = 0.0, 1.0
            for _ in range(self.recovery_steps):
                lam = (lo + hi) / 2
                s = dp([rt[i] + lam * rmax * credit_total.get(names[i], 0.0) / cmax for i in range(n)])
                if s not in tried:
                    tried[s] = oracle(s)["peak"]
                if tried[s] <= best_peak * (1 + self.peak_tol):
                    hi = lam
                else:
                    lo = lam

        # --- choose ----------------------------------------------------------------
        ok = [s for s, p in tried.items() if p <= best_peak * (1 + self.peak_tol)
              and (self.min_retained is None or retained(s) >= self.min_retained)]
        chosen = max(ok, key=retained) if ok else stock
        if tried[chosen] > tried[stock] * (1 - self.min_gain):
            chosen = stock
        assert sum(_q(memory[i]) for i in chosen) <= _q(max_memory), "budget violated"

        st = dict(n_candidates=n, max_memory=float(max_memory), weight=float(sum(memory[i] for i in chosen)),
                  stock_peak_MB=tried[stock] / 1e6, chosen_peak_MB=tried[chosen] / 1e6,
                  gain=1 - tried[chosen] / tried[stock], chosen_is_stock=chosen == stock,
                  retained=retained(chosen), plans_tried=len(tried),
                  oracle_calls=oracle.calls, oracle_seconds=oracle.seconds,
                  solver_seconds=time.perf_counter() - t0,
                  saved_ops=dict(collections.Counter(_op(cands[i]) for i in chosen)),
                  added_ops=dict(collections.Counter(_op(cands[i]) for i in chosen - stock)),
                  predicted_peak_bytes=tried[chosen])
        self.stats.append(st)
        if self.verbose:
            print(f"[LivenessAwareSolver] stock {st['stock_peak_MB']:.1f}MB -> {st['chosen_peak_MB']:.1f}MB "
                  f"({-100 * st['gain']:+.1f}%), retained {st['retained']:.3f}, "
                  f"{st['oracle_calls']} oracle calls, {st['solver_seconds']:.1f}s")
        return sorted(chosen), [i for i in range(n) if i not in chosen]


def verify_emitted(fw_gm, bw_gm, solver: LivenessAwareSolver) -> bool:
    """Check that the modules the backend received have the peak the solver predicted."""
    p = max(walk(fw_gm, False)[0], walk(bw_gm, True)[0])
    return abs(p - solver.stats[-1]["predicted_peak_bytes"]) <= max(1, 1e-3 * p)
