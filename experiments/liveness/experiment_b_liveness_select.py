"""
experiment_b_liveness_select.py -- PyTorch #197838, Experiment B (LOCAL ONLY).

Maintainer framing
------------------
Experiment A established that the physical peak of an `aot_eager` step is
predicted to within ~1% by a static liveness walk over the fw/bw modules the
partitioner emits (min-cut -> extract -> reorder), while closure and the in-tree
KnapsackEvaluator are not.

Experiment B asks the smallest solver-side question that result licenses:

    Holding the activation_memory_budget semantics fixed (every plan considered
    has knapsack weight <= the budget), does choosing among a small, generically
    generated family of budget-feasible plans by *exact emitted-graph liveness*
    reduce the physical peak where the pathology exists, leave healthy regimes
    alone, and at what estimated / measured runtime cost?

Design constraints (what a maintainer would insist on):
  * Budget semantics unchanged: only plans with sum(memory[saved]) <= max_memory.
  * No model-specific predicates. Plan family comes from repeated-structure
    classes of knapsack candidates (op, shape, 2-level input signature).
    No "MLP", no "down_proj", no layer names.
  * The production dp_knapsack still fills all remaining capacity.
  * The stock plan is always in the family; ties or gains < MIN_GAIN return stock.
  * The oracle is exactly the partitioner's own pipeline (solve_min_cut with the
    same aggressive options + dont_ban, _extract_fwd_bwd_modules, reordering,
    raise_getitems) followed by the liveness walk validated in Experiment A.
  * The runtime trade-off is not hidden behind one epsilon: plans are selected
    at several runtime-retention floors, and each chosen plan's step time is measured.
  * Invariant: the oracle's predicted peak for the returned plan must equal the
    static peak of the modules the backend actually receives. A mismatch
    invalidates the run.
  * Compile-time overhead (oracle calls x seconds) is reported.

PRE-REGISTERED hypotheses (fixed before running):
  H1 efficacy  : for Llama at 0.05 and 0.10, at least one policy reduces the
                 measured peak by >= 30% vs stock at the same budget.
  H2 safety    : no chosen plan's measured peak exceeds stock by > 1%, at any
                 budget, model or policy.
  H3 controls  : for Llama >= 0.15 and all BERT budgets, every policy changes the
                 measured peak by < 5% (i.e. it effectively returns stock).
  H4 prediction: for every chosen plan, |measured - predicted| / measured <= 3%,
                 and sign(predicted gain) == sign(measured gain) whenever |gain| > 2%.
  H5 invariant : oracle prediction == emitted-module static peak (<= 0.1%) for all runs.
  Reported, not pass/fail: step-time delta, estimated runtime retained,
  saved-bytes ratio, oracle wall time.

Usage (from the ackaudit checkout, same venv as Experiment A):
  python /tmp/experiment_b_liveness_select.py --suite selftest          # CPU plumbing check
  python /tmp/experiment_b_liveness_select.py --suite quick  --repeats 3 # llama 0.05/0.15, bert 0.05
  python /tmp/experiment_b_liveness_select.py --suite full   --repeats 3 # llama 0.05..0.30, bert 0.05..0.15

Needs liveness_oracle_check.py (Experiment A) in the same directory; the static
walk and the models are imported from there so A and B share one oracle.
Nothing is committed, pushed, or written outside --out.
"""
from __future__ import annotations

import argparse, collections, dataclasses, json, os, statistics, sys, time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import liveness_oracle_check as A  # also installs the estimate_runtime recorder

import torch
import torch._functorch.partitioners as P
from torch._functorch import config as fc
from torch._functorch.partitioners import CustomKnapsackSolver
from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
from torch._functorch.compile_utils import raise_getitems
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.debugging import boxed_nop
from torch.utils._ordered_set import OrderedSet

# ----------------------------------------------------------------------------
FRACTIONS = (0.25, 0.5, 0.75, 1.0)   # share of a structural class forced to be saved
MIN_CLASS_SIZE = 4                    # "repeated structure" threshold
MAX_ORACLE_CALLS = 60                 # compile-time cap per graph
MAX_ROUNDS = 3                        # greedy rounds over classes
MIN_GAIN = 0.02                       # deviate from stock only for >= 2% predicted gain
POLICIES = (                          # (name, minimum retained fraction of stock runtime objective)
    ("stock", None),
    ("retain>=0.90", 0.90),
    ("retain>=0.70", 0.70),
    ("retain>=0.40", 0.40),
    ("unconstrained", 0.0),
)
TOL = 5e-7
S_QUANT = 10000  # dp_knapsack's quantization scale: feasibility is judged exactly as the production DP judges it


def q(x: float) -> int:
    return round(x * S_QUANT)

_CTX: dict[str, Any] = {}             # partition_fn kwargs for the graph being partitioned
_FAMILY_CACHE: dict[tuple, dict] = {} # (candidate names, max_memory) -> enumerated family


# ----------------------------------------------------------------------------
# partitioner-faithful oracle
# ----------------------------------------------------------------------------
def _aggressive_min_cut_options() -> "P.MinCutOptions":
    """Exactly the options choose_saved_values_set passes to the final solve_min_cut."""
    o = P.MinCutOptions(
        ban_if_used_far_apart=fc.ban_recompute_used_far_apart,
        ban_if_long_fusible_chains=fc.ban_recompute_long_fusible_chains,
        ban_if_materialized_backward=fc.ban_recompute_materialized_backward,
        ban_if_not_in_allowlist=fc.ban_recompute_not_in_allowlist,
        ban_if_reduction=fc.ban_recompute_reductions,
    )
    if fc.aggressive_recomputation:
        o = dataclasses.replace(o, ban_if_used_far_apart=False, ban_if_long_fusible_chains=False,
                                ban_if_materialized_backward=False, ban_if_not_in_allowlist=False)
    return dataclasses.replace(o, ban_if_used_far_apart=False, ban_if_long_fusible_chains=False,
                               ban_if_materialized_backward=False, ban_if_not_in_allowlist=False)


class Oracle:
    def __init__(self, joint_graph, node_info, cands):
        self.jg, self.ni, self.cands = joint_graph, node_info, cands
        self.jm = joint_graph.owning_module or _CTX["joint_module"]
        assert self.jm.graph is joint_graph, "joint module/graph mismatch"
        self.opts = _aggressive_min_cut_options()
        self.calls, self.seconds, self.cache = 0, 0.0, {}

    def emit(self, saved_idx: frozenset):
        dont_ban = OrderedSet(c for i, c in enumerate(self.cands) if i not in saved_idx)
        saved_values, _ = P.solve_min_cut(self.jg, self.ni, self.opts, dont_ban)
        sym = [n for n in saved_values if P.is_sym_node(n) and not P._is_assert_only_symbool(n)]
        opaque = [n for n in saved_values if P.is_opaque_node(n)]
        vals = [n for n in saved_values if not P.is_sym_node(n) and not P.is_opaque_node(n)]
        fw, bw = P._extract_fwd_bwd_modules(
            self.jm, list(vals), saved_sym_nodes=list(sym), saved_opaque_nodes=list(opaque),
            num_fwd_outputs=_CTX["num_fwd_outputs"],
            static_lifetime_input_nodes=self.ni.static_lifetime_input_nodes)
        bw = P.reordering_to_mimic_autograd_engine(bw)
        return raise_getitems(fw), raise_getitems(bw), vals

    def __call__(self, saved_idx) -> dict:
        key = frozenset(saved_idx)
        if key not in self.cache:
            t0 = time.perf_counter()
            fw, bw, vals = self.emit(key)
            s = A.static_step_peak(fw, bw)
            s["actual_saved_bytes"] = sum(A._size_of(v) for v in vals
                                          if not v.name.startswith("primals"))
            self.cache[key] = s
            self.calls += 1
            self.seconds += time.perf_counter() - t0
        return self.cache[key]


# ----------------------------------------------------------------------------
# generic plan family: repeated-structure classes of candidates
# ----------------------------------------------------------------------------
def _strip(n):
    while n is not None and n.op == "call_function" and (A._op(n) in A._VIEWISH or
                                                          A._op(n) == "<built-in function getitem>"):
        n = n.all_input_nodes[0] if n.all_input_nodes else None
    return n


def _tag(n):
    if n is None:
        return "none"
    return "input" if n.op == "placeholder" else A._op(n)


def structural_signature(c) -> tuple:
    val = c.meta.get("val")
    t = val[0] if isinstance(val, (tuple, list)) and val else val
    shape = tuple(t.shape) if isinstance(t, torch.Tensor) else None
    ins = []
    for a in c.all_input_nodes:
        s = _strip(a)
        grand = tuple(sorted(_tag(_strip(g)) for g in s.all_input_nodes)) if s is not None and s.op == "call_function" else ()
        ins.append((_tag(s), grand))
    return (A._op(c), shape, tuple(sorted(ins)))


def evenly_spaced(members: list[int], k: int) -> list[int]:
    n = len(members)
    return sorted({members[min(n - 1, int((j + 0.5) * n / k))] for j in range(k)})


def enumerate_family(memory, rt, max_memory, cands, node_info, oracle: Oracle) -> dict:
    n = len(cands)
    order = {c: node_info.get_fw_order(c) for c in cands}

    def plan(forced: frozenset):
        qw = sum(q(memory[i]) for i in forced)
        if qw > q(max_memory):
            return None
        rest = [i for i in range(n) if i not in forced]
        _, sv, _ = dp_knapsack([memory[i] for i in rest], [rt[i] for i in rest], (q(max_memory) - qw) / S_QUANT)
        return frozenset(forced) | frozenset(rest[i] for i in sv)

    classes = collections.defaultdict(list)
    for i, c in enumerate(cands):
        classes[structural_signature(c)].append(i)
    classes = {k: sorted(v, key=lambda i: order[cands[i]]) for k, v in classes.items() if len(v) >= MIN_CLASS_SIZE}
    def _nm(k, v):
        ins = ", ".join(t if not g else f"{t}({'+'.join(g)})" for t, g in k[2] if t != "input")
        return f"{k[0]}{list(k[1]) if k[1] else ''}<-[{ins}] x{len(v)}"
    cls_names = {k: _nm(k, v) for k, v in classes.items()}

    stock = plan(frozenset())
    evaluated: dict[frozenset, dict] = {}

    def record(saved, forced_desc):
        if saved is None or saved in evaluated or oracle.calls >= MAX_ORACLE_CALLS:
            return
        o = oracle(saved)
        evaluated[saved] = dict(
            forced=forced_desc,
            weight=float(sum(memory[i] for i in saved)),
            runtime_saved=float(sum(rt[i] for i in saved)),
            predicted_peak_MB=o["static_peak_MB"],
            predicted_bw_newalloc_MB=o["static_bw_newalloc_MB"],
            recomputed_live_at_peak_MB=o["static_recomputed_live_at_bw_peak_MB"],
            actual_saved_MB=o["actual_saved_bytes"] / 1e6,
            saved_idx=sorted(saved),
        )

    record(stock, [])
    forced_now: frozenset = frozenset()
    desc_now: list = []
    best_peak = evaluated[stock]["predicted_peak_MB"]
    for rnd in range(MAX_ROUNDS):
        round_best = None
        for key, members in classes.items():
            for f in FRACTIONS:
                k = max(1, round(f * len(members)))
                forced = forced_now | frozenset(evenly_spaced(members, k))
                if forced == forced_now:
                    continue
                saved = plan(forced)
                d = desc_now + [f"{cls_names[key]} @{f:g}"]
                record(saved, d)
                if saved is not None and saved in evaluated:
                    p = evaluated[saved]["predicted_peak_MB"]
                    if round_best is None or p < round_best[0]:
                        round_best = (p, forced, d)
        if round_best is None or round_best[0] > best_peak * (1 - MIN_GAIN):
            break
        best_peak, forced_now, desc_now = round_best
    return dict(stock=stock, evaluated=evaluated, n_classes=len(classes),
                classes=[cls_names[k] for k in classes],
                oracle_calls=oracle.calls, oracle_seconds=oracle.seconds)


def select(family: dict, min_retained: float | None) -> frozenset:
    stock = family["stock"]
    st = family["evaluated"][stock]
    if min_retained is None:
        return stock
    best, best_peak = stock, st["predicted_peak_MB"]
    for saved, e in family["evaluated"].items():
        if e["runtime_saved"] < min_retained * st["runtime_saved"] - 1e-9:
            continue
        if e["predicted_peak_MB"] < best_peak:
            best, best_peak = saved, e["predicted_peak_MB"]
    if best_peak > st["predicted_peak_MB"] * (1 - MIN_GAIN):
        return stock
    return best


# ----------------------------------------------------------------------------
class ExpBSolver(CustomKnapsackSolver):
    def __init__(self, min_retained: float | None):
        self.min_retained = min_retained
        self.rec: dict[str, Any] = {}

    def __call__(self, memory, joint_graph, max_memory, node_info, cands):
        cands = list(cands)
        rt = [A._RT[c] for c in cands]
        key = (tuple(c.name for c in cands), round(float(max_memory), 9))
        oracle = Oracle(joint_graph, node_info, cands)
        if P.has_recomputable_rng_ops(oracle.jm):
            raise RuntimeError("graph has recomputable RNG ops; oracle does not model rng functionalization")
        fam = _FAMILY_CACHE.get(key)
        if fam is None:
            fam = enumerate_family(memory, rt, max_memory, cands, node_info, oracle)
            _FAMILY_CACHE[key] = fam
        chosen = select(fam, self.min_retained)
        e, st = fam["evaluated"][chosen], fam["evaluated"][fam["stock"]]
        # same (quantized) feasibility rule the production dp_knapsack enforces on the stock plan
        assert sum(q(memory[i]) for i in chosen) <= q(max_memory), "budget violated"
        self.rec.update(
            n_candidates=len(cands), max_memory=float(max_memory),
            chosen_is_stock=(chosen == fam["stock"]), forced=e["forced"],
            weight=e["weight"], stock_weight=st["weight"],
            runtime_retained=e["runtime_saved"] / max(st["runtime_saved"], 1e-12),
            predicted_peak_MB=e["predicted_peak_MB"], predicted_bw_newalloc_MB=e["predicted_bw_newalloc_MB"],
            stock_predicted_peak_MB=st["predicted_peak_MB"],
            recomputed_live_at_peak_MB=e["recomputed_live_at_peak_MB"],
            actual_saved_MB=e["actual_saved_MB"], stock_actual_saved_MB=st["actual_saved_MB"],
            saved_ops=dict(collections.Counter(A._op(cands[i]) for i in chosen)),
            family_size=len(fam["evaluated"]), n_classes=fam["n_classes"],
            oracle_calls=fam["oracle_calls"], oracle_seconds=fam["oracle_seconds"],
            family=[{k: v for k, v in x.items() if k != "saved_idx"} for x in fam["evaluated"].values()],
            classes=fam["classes"],
        )
        recomp = [i for i in range(len(cands)) if i not in chosen]
        return sorted(chosen), recomp

    def uuid(self):
        return None


# ----------------------------------------------------------------------------
def run(model, scale, budget, policy, min_retained, device, repeats):
    torch.manual_seed(197838)
    torch._dynamo.reset()
    cap: dict[str, Any] = {}

    def part(joint_module, joint_inputs, **kw):
        _CTX.clear(); _CTX.update(kw); _CTX["joint_module"] = joint_module
        return P.min_cut_rematerialization_partition(joint_module, joint_inputs, **kw)

    def fw_c(gm, ex):
        cap["fw"] = gm; return boxed_nop(gm, ex)

    def bw_c(gm, ex):
        cap["bw"] = gm; return boxed_nop(gm, ex)

    # identical to torch._dynamo.backends.debugging.aot_eager, plus capture hooks
    backend = aot_autograd(fw_compiler=fw_c, bw_compiler=bw_c, partition_fn=part,
                           keep_inference_input_mutations=True)
    build, make_inputs = A.resolve(model, scale)
    m = build().to(device)
    args = tuple(x.to(device) for x in make_inputs())
    cm = torch.compile(m, backend=backend, dynamic=False)

    solver = ExpBSolver(min_retained)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
    try:
        fc.activation_memory_budget = budget
        fc.activation_memory_budget_solver = solver
        cm(*args).backward()
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver = prev

    row: dict[str, Any] = dict(model=model, scale=scale, budget=budget, policy=policy, device=device)
    row.update(solver.rec)
    emitted = A.static_step_peak(cap["fw"], cap["bw"])
    row["emitted_static_peak_MB"] = emitted["static_peak_MB"]
    row["emitted_static_bw_newalloc_MB"] = emitted["static_bw_newalloc_MB"]
    row["invariant_ok"] = abs(emitted["static_peak_MB"] - row["predicted_peak_MB"]) <= 1e-3 * max(1.0, emitted["static_peak_MB"])

    m.zero_grad(set_to_none=False)
    peaks, ms = [], []
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        resident = torch.cuda.memory_allocated()
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            cm(*args).backward(); torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1e3)
            peaks.append(torch.cuda.max_memory_allocated() - resident)
            m.zero_grad(set_to_none=False)
        row["predicted_compare_MB"] = row["predicted_peak_MB"]
    else:
        for _ in range(repeats):
            loss = cm(*args); mode = A._LiveBytes(); t0 = time.perf_counter()
            with mode:
                loss.backward()
            ms.append((time.perf_counter() - t0) * 1e3)
            peaks.append(mode.peak); del loss; m.zero_grad(set_to_none=False)
        row["predicted_compare_MB"] = row["predicted_bw_newalloc_MB"]
    row["measured_peak_MB"] = statistics.median(peaks) / 1e6
    row["step_ms"] = statistics.median(ms)
    del m, cm, args
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def cases(suite):
    if suite == "selftest":
        return [("toy16", 16, b) for b in (0.02, 0.05, 0.15)]
    if suite == "quick":
        return [("llama", 8, 0.05), ("llama", 8, 0.15), ("bert", 8, 0.05)]
    return [("llama", 8, b) for b in (0.05, 0.10, 0.15, 0.20, 0.30)] + [("bert", 8, b) for b in (0.05, 0.10, 0.15)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["selftest", "quick", "full"], default="quick")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/tmp/experiment_b.json")
    a = ap.parse_args()
    device = "cpu" if a.suite == "selftest" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    rows = []
    for model, scale, budget in cases(a.suite):
        for pname, floor in POLICIES:
            print(f"[{model} {budget:.2f} {pname}]", flush=True)
            r = run(model, scale, budget, pname, floor, device, a.repeats)
            rows.append(r)
            print(f"   stock?={r['chosen_is_stock']}  forced={r['forced']}\n"
                  f"   w={r['weight']:.4f}/{r['max_memory']:.4f}  retained={r['runtime_retained']:.3f}  "
                  f"pred={r['predicted_compare_MB']:.1f}MB  meas={r['measured_peak_MB']:.1f}MB  step={r['step_ms']:.1f}ms  "
                  f"invariant={'OK' if r['invariant_ok'] else 'MISMATCH'}  "
                  f"family={r['family_size']} oracle={r['oracle_calls']} calls/{r['oracle_seconds']:.1f}s", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1, default=str)

    # ------------------------------------------------------------------ verdicts
    print("\n" + "=" * 100)
    print(f"{'model':6} {'bud':>5} {'policy':14} {'meas MB':>8} {'Δ vs stock':>10} {'pred Δ':>8} {'step Δ':>7} "
          f"{'retain':>6} {'w/budget':>9} {'saved MB':>8}  plan")
    stock = {(r["model"], r["budget"]): r for r in rows if r["policy"] == "stock"}
    h1 = collections.defaultdict(bool); h2 = h3 = h4 = h5 = True; notes = []
    for r in rows:
        s = stock[(r["model"], r["budget"])]
        dm = (r["measured_peak_MB"] - s["measured_peak_MB"]) / s["measured_peak_MB"]
        dp = (r["predicted_compare_MB"] - s["predicted_compare_MB"]) / s["predicted_compare_MB"]
        dt = (r["step_ms"] - s["step_ms"]) / s["step_ms"]
        print(f"{r['model']:6} {r['budget']:5.2f} {r['policy']:14} {r['measured_peak_MB']:8.1f} {dm:+10.1%} {dp:+8.1%} "
              f"{dt:+7.1%} {r['runtime_retained']:6.3f} {r['weight']:.4f}/{r['max_memory']:.2f} {r['actual_saved_MB']:8.1f}  "
              f"{'stock' if r['chosen_is_stock'] else '; '.join(r['forced'])}")
        if r["model"] == "llama" and r["budget"] in (0.05, 0.10) and dm <= -0.30:
            h1[r["budget"]] = True
        if dm > 0.01:
            h2 = False; notes.append(f"H2 violated: {r['model']} {r['budget']} {r['policy']} {dm:+.1%}")
        if ((r["model"] == "llama" and r["budget"] >= 0.15) or r["model"] in ("bert",)) and abs(dm) >= 0.05:
            h3 = False; notes.append(f"H3 violated: {r['model']} {r['budget']} {r['policy']} {dm:+.1%}")
        rel = abs(r["measured_peak_MB"] - r["predicted_compare_MB"]) / r["measured_peak_MB"]
        if rel > 0.03 or (abs(dm) > 0.02 and (dm > 0) != (dp > 0)):
            h4 = False; notes.append(f"H4 violated: {r['model']} {r['budget']} {r['policy']} err={rel:.1%} dm={dm:+.1%} dp={dp:+.1%}")
        if not r["invariant_ok"]:
            h5 = False; notes.append(f"H5 violated: {r['model']} {r['budget']} {r['policy']}")
    print()
    for n in notes:
        print("  ", n)
    if a.suite != "selftest":
        need = [b for b in (0.05, 0.10) if any(r["model"] == "llama" and r["budget"] == b for r in rows)]
        print(f"H1 efficacy   : {'PASS' if all(h1[b] for b in need) else 'FAIL'}  ({ {b: h1[b] for b in need} })")
        print(f"H3 controls   : {'PASS' if h3 else 'FAIL'}")
    print(f"H2 safety     : {'PASS' if h2 else 'FAIL'}")
    print(f"H4 prediction : {'PASS' if h4 else 'FAIL'}")
    print(f"H5 invariant  : {'PASS' if h5 else 'FAIL'}")
    print(f"wrote {a.out}; no repository files modified.")


if __name__ == "__main__":
    main()
