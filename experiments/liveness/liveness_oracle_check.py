"""
liveness_oracle_check.py  --  PyTorch #197838, next experiment (LOCAL ONLY).

Question tested
---------------
Is the structural quantity that explains the non-monotonic physical peak the
*live set of the emitted (partitioned + reordered) backward graph*, and can it be
computed statically from information the partitioner already has?

This script changes NO solver behaviour beyond the interventions that are already
established in the thread (forced-in / forced-out MLP outputs, identity reorder).
It adds no heuristic. For every plan it records:

  measured   : CUDA max_memory_allocated() - resident   (same procedure as ackaudit)
  static     : peak of a liveness walk over the captured fw/bw GraphModules
               (the exact graphs AOTAutograd hands to the backend compiler)
  baselines  : closure metrics (as in closure_break_cuda.py), in-tree
               KnapsackEvaluator(account_for_backward_pass=True), saved bytes

and then checks pre-registered pass/fail criteria (see PRE-REGISTERED below).

Nothing is pushed, committed or written outside --out.

Usage (GPU box, from the ackaudit checkout):
    python liveness_oracle_check.py --suite full --repeats 3
CPU self-test of the plumbing (no ackaudit, no CUDA needed):
    python liveness_oracle_check.py --suite selftest
"""
from __future__ import annotations

import argparse, collections, functools, json, math, statistics, time, weakref
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._functorch.partitioners as P
from torch._functorch import config as fc
from torch._functorch.partitioners import CustomKnapsackSolver, _size_of
from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
from torch._functorch._activation_checkpointing.graph_info_provider import GraphInfoProvider
from torch._functorch._activation_checkpointing.knapsack_evaluator import KnapsackEvaluator
from torch._dynamo.backends.debugging import aot_eager, boxed_nop

# ----------------------------------------------------------------------------
# PRE-REGISTERED criteria (decide BEFORE looking at results)
# ----------------------------------------------------------------------------
RHO_MIN = 0.90          # Spearman(static, measured) over Llama aot_eager plans
MEDIAN_REL_ERR_MAX = 0.15
CONTRASTS = [           # (higher-peak plan, lower-peak plan) as observed in the thread
    ("llama nat 0.05", "llama nat 0.10"),
    ("llama nat 0.10", "llama nat 0.15"),
    ("llama nat 0.20", "llama nat 0.15"),
    ("llama 0.15 force-out MLP", "llama nat 0.15"),
    ("llama nat 0.05", "llama 0.05 force-in all MLP (over budget)"),
    ("llama nat 0.05", "llama 0.05 force-in 8 MLP"),
    ("llama 0.05 force-in 8 MLP", "llama 0.05 force-in 16 MLP"),
    ("llama 0.05 force-in 16 MLP", "llama 0.05 force-in 24 MLP"),
    ("llama 0.15 no-reorder", "llama nat 0.15"),
    ("llama 0.05 no-reorder", "llama nat 0.05"),
    ("bert nat 0.15", "bert nat 0.05"),
    ("toy 0.15 force-out MLP", "toy nat 0.15"),
    ("toy 0.15 no-reorder", "toy nat 0.15"),
]

# ----------------------------------------------------------------------------
# runtime capture: same numbers the partitioner feeds dp_knapsack
# ----------------------------------------------------------------------------
_RT: dict = {}
_orig_estimate_runtime = P.estimate_runtime


def _recording_estimate_runtime(node):
    v = _orig_estimate_runtime(node)
    _RT[node] = v
    return v


P.estimate_runtime = _recording_estimate_runtime  # choose_saved_values_set looks it up by module global


def _op(n) -> str:
    t = str(getattr(n, "target", ""))
    return t.split(".")[1] if t.startswith("aten.") else t


_VIEWISH = {"view", "_unsafe_view", "reshape", "expand", "t", "transpose", "permute", "slice",
            "select", "unsqueeze", "squeeze", "alias", "as_strided", "detach", "clone"}


def is_mlp_output(n) -> bool:
    """down_proj(silu(gate) * up) -- same predicate as the prior ackaudit work."""
    if _op(n) != "mm" or not n.args:
        return False
    x = n.args[0]
    while x is not None and _op(x) in _VIEWISH:
        x = x.args[0] if getattr(x, "args", None) else None
    if x is None or _op(x) != "mul":
        return False
    return any(hasattr(a, "target") and _op(a) == "silu" for a in x.args)


# ----------------------------------------------------------------------------
# solver hook: production dp_knapsack, optional forced-in / forced-out sets
# ----------------------------------------------------------------------------
class PlanSolver(CustomKnapsackSolver):
    def __init__(self, force_in_k: int | None = None, force_out: bool = False,
                 allow_over_budget: bool = False, force_out_k: int | None = None):
        self.force_in_k, self.force_out, self.allow_over = force_in_k, force_out, allow_over_budget
        self.force_out_k = force_out_k
        self.rec: dict[str, Any] = {}

    def __call__(self, memory, joint_graph, max_memory, node_info, cands):
        cands = list(cands)
        rt = [_RT[n] for n in cands]
        mlp = [i for i, n in enumerate(cands) if is_mlp_output(n)]
        mlp_sorted = sorted(mlp, key=lambda i: node_info.get_fw_order(cands[i]) if hasattr(node_info, "get_fw_order") else i)
        forced = mlp_sorted[: self.force_in_k] if self.force_in_k is not None else []
        excluded = set(mlp) if self.force_out else set()
        if self.force_out_k is not None:
            excluded = set(mlp_sorted[: self.force_out_k])
        cap = max_memory - sum(memory[i] for i in forced)
        if cap < -1e-9 and not self.allow_over:
            raise RuntimeError(f"forced set ({len(forced)}) exceeds budget; refusing")
        rest = [i for i in range(len(cands)) if i not in set(forced) and i not in excluded]
        _, sv, _ = dp_knapsack([memory[i] for i in rest], [rt[i] for i in rest], max(cap, 0.0))
        saved = sorted(set(forced) | {rest[i] for i in sv})
        recomp = [i for i in range(len(cands)) if i not in set(saved)]
        stock_val, stock_sv, _ = dp_knapsack(list(memory), rt, max_memory)
        self.rec.update(
            cands=cands, memory=list(memory), rt=rt, saved=saved, recomp=recomp,
            joint_graph=joint_graph, max_memory=float(max_memory),
            n_candidates=len(cands), n_mlp_candidates=len(mlp), n_forced=len(forced),
            saved_weight=float(sum(memory[i] for i in saved)),
            runtime_saved=float(sum(rt[i] for i in saved)),
            stock_runtime_saved=float(sum(rt[i] for i in stock_sv)),
            mlp_saved=sum(1 for i in saved if i in set(mlp)),
            saved_ops=dict(collections.Counter(_op(cands[i]) for i in saved)),
            candidate_ops=dict(collections.Counter(_op(n) for n in cands)),
        )
        return saved, recomp

    def uuid(self):
        return None


# ----------------------------------------------------------------------------
# baselines on the knapsack abstraction
# ----------------------------------------------------------------------------
def closure_metrics(rec) -> dict:
    names = [n.name for n in rec["cands"]]
    gp = GraphInfoProvider.inialize_from_graph(rec["joint_graph"], rec["cands"], rec["memory"], rec["rt"])
    G = gp.recomputable_node_only_graph_with_larger_graph_context
    saved = {names[i] for i in rec["saved"]}
    w = dict(zip(names, rec["memory"]))
    sizes, weights = [], []
    for nm in names:
        if nm in saved:
            continue
        seen, q = set(), collections.deque(p for p in G.predecessors(nm) if p not in saved)
        while q:
            d = q.popleft()
            if d in seen:
                continue
            seen.add(d)
            q.extend(p for p in G.predecessors(d) if p not in saved and p not in seen)
        sizes.append(len(seen)); weights.append(sum(w[x] for x in seen))
    ke = KnapsackEvaluator(gp).evaluate_knapsack_output(rec["saved"], rec["recomp"], account_for_backward_pass=True)
    return dict(closure_max=max(sizes, default=0), closure_max_w=max(weights, default=0.0),
                closure_sum_w=sum(weights), knapsack_evaluator_peak=float(ke["peak_memory"]))


# ----------------------------------------------------------------------------
# THE QUANTITY UNDER TEST: static liveness over the emitted graphs
# ----------------------------------------------------------------------------
def _alloc_size(n) -> int:
    if n.op != "call_function":
        return 0
    val = n.meta.get("val")
    if isinstance(val, (tuple, list)):
        return 0  # counted at the getitem users
    if _op(n) in _VIEWISH - {"clone"}:
        return 0
    try:
        return _size_of(n)
    except Exception:
        return 0


def liveness_walk(gm: torch.fx.GraphModule, live_inputs: Callable[[Any], bool],
                  tag: frozenset = frozenset()) -> dict:
    """Allocate at definition, free after last (transitive-through-views) use.
    Inputs selected by live_inputs() start live and are freed at last use
    (AOTAutograd steals backward args). Graph outputs live to the end."""
    nodes = list(gm.graph.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    root = {}
    for n in nodes:
        if n.op == "call_function" and _op(n) in _VIEWISH - {"clone"} and n.all_input_nodes:
            root[n] = root[n.all_input_nodes[0]]
        else:
            root[n] = n
    last: dict = {}
    for n in nodes:
        for a in n.all_input_nodes:
            r = root[a]
            last[r] = max(last.get(r, -1), idx[n])
    frees = collections.defaultdict(list)
    live = tagged = 0
    for n in nodes:
        if n.op == "placeholder" and live_inputs(n):
            s = _size_of(n) if "val" in n.meta else 0
            live += s
            frees[last.get(n, idx[n])].append((n, s))
    start = peak = live
    peak_at, tagged_at_peak = "start", 0
    for n in nodes:
        s = _alloc_size(n) if root[n] is n else 0
        if s:
            live += s
            if n.name in tag:
                tagged += s
            frees[last.get(n, idx[n])].append((n, s))
        if live > peak:
            peak, peak_at, tagged_at_peak = live, n.name, tagged
        for d, ds in frees.pop(idx[n], []):
            live -= ds
            if d.name in tag:
                tagged -= ds
    return dict(peak=peak, start=start, peak_at=peak_at, tagged_at_peak=tagged_at_peak)


def static_step_peak(fw: torch.fx.GraphModule, bw: torch.fx.GraphModule) -> dict:
    fw_names = {n.name for n in fw.graph.nodes if n.op == "call_function"}
    remat = frozenset(n.name for n in bw.graph.nodes if n.op == "call_function" and n.name in fw_names)
    # fw: params/inputs are resident (excluded); outputs (saved acts) are never freed inside fw.
    f = liveness_walk(fw, lambda n: False)
    # bw: saved activations are live at start; primals (params / inputs) are resident -> excluded.
    b = liveness_walk(bw, lambda n: not n.name.startswith("primals"), remat)
    bn = liveness_walk(bw, lambda n: False)
    return dict(static_peak_MB=max(f["peak"], b["peak"]) / 1e6, static_bw_newalloc_MB=bn["peak"] / 1e6, static_fw_peak_MB=f["peak"] / 1e6,
                static_bw_peak_MB=b["peak"] / 1e6, static_saved_at_bw_start_MB=b["start"] / 1e6,
                static_recomputed_live_at_bw_peak_MB=b["tagged_at_peak"] / 1e6,
                static_bw_peak_at=b["peak_at"], n_rematerialized=len(remat))


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
class _RMS(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d))
    def forward(s, x): return s.w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))


class _Blk(nn.Module):
    def __init__(s, d=256, f=688, h=8):
        super().__init__(); s.h = h; s.n1, s.n2 = _RMS(d), _RMS(d)
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        s.g, s.u, s.dn = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)
    def forward(s, x):
        B, T, D = x.shape; y = s.n1(x)
        q, k, v = (m(y).view(B, T, s.h, D // s.h).transpose(1, 2) for m in (s.q, s.k, s.v))
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        y = s.n2(x); return x + s.dn(F.silu(s.g(y)) * s.u(y))


class _Toy(nn.Module):
    def __init__(s, L=8, V=1000, d=256):
        super().__init__(); s.e = nn.Embedding(V, d); s.b = nn.ModuleList(_Blk() for _ in range(L))
        s.n = _RMS(d); s.h = nn.Linear(d, V, bias=False)
    def forward(s, ids, tgt):
        x = s.e(ids)
        for b in s.b: x = b(x)
        return F.cross_entropy(s.h(s.n(x)).flatten(0, 1), tgt.flatten())


def resolve(model: str, scale: int):
    if model in ("toy", "toy16"):
        T = 256 if model == "toy" else 512
        return (lambda: _Toy(L=scale)), (lambda: (torch.randint(0, 1000, (1, T)), torch.randint(0, 1000, (1, T))))
    from ackaudit.capture import _resolve  # the established builders
    return _resolve(model, scale=scale)


# ----------------------------------------------------------------------------
# one plan
# ----------------------------------------------------------------------------
class _LiveBytes(torch.utils._python_dispatch.TorchDispatchMode):
    """CPU-only stand-in for max_memory_allocated during backward.
    A storage allocated inside the mode is live until the last Python tensor
    (base or view) referring to it is collected."""
    def __init__(s):
        super().__init__(); s.live = s.peak = 0; s.refs = {}; s.nbytes = {}

    def _drop(s, ptr):
        s.refs[ptr] -= 1
        if s.refs[ptr] == 0:
            s.live -= s.nbytes.pop(ptr); del s.refs[ptr]

    def __torch_dispatch__(s, func, types, args=(), kwargs=None):
        flat_in = torch.utils._pytree.tree_flatten((args, kwargs or {}))[0]
        in_ptrs = {t.untyped_storage().data_ptr() for t in flat_in if isinstance(t, torch.Tensor)}
        out = func(*args, **(kwargs or {}))
        for t in torch.utils._pytree.tree_flatten(out)[0]:
            if not isinstance(t, torch.Tensor):
                continue
            st = t.untyped_storage(); ptr = st.data_ptr()
            if ptr not in s.refs:
                if ptr in in_ptrs or t._is_view() or st.nbytes() == 0:
                    continue  # pre-existing storage (in-place / view of an input)
                s.refs[ptr] = 0; s.nbytes[ptr] = st.nbytes()
                s.live += st.nbytes(); s.peak = max(s.peak, s.live)
            s.refs[ptr] += 1
            weakref.finalize(t, s._drop, ptr)
        return out


def run_plan(label, model, scale, budget, device, repeats, solver: PlanSolver, no_reorder=False):
    torch.manual_seed(197838)
    torch._dynamo.reset()
    cap: dict = {}

    def fw_c(gm, ex):
        cap["fw"] = gm; return boxed_nop(gm, ex)

    def bw_c(gm, ex):
        cap["bw"] = gm; return boxed_nop(gm, ex)

    backend = functools.partial(aot_eager, fw_compiler=fw_c, bw_compiler=bw_c)
    build, make_inputs = resolve(model, scale)
    m = build().to(device)
    args = tuple(x.to(device) for x in make_inputs())
    cm = torch.compile(m, backend=backend, dynamic=False)

    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver, P.reordering_to_mimic_autograd_engine)
    try:
        fc.activation_memory_budget = budget
        fc.activation_memory_budget_solver = solver
        if no_reorder:
            P.reordering_to_mimic_autograd_engine = lambda gm: gm
        cm(*args).backward()
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver, P.reordering_to_mimic_autograd_engine = prev

    row: dict[str, Any] = dict(plan=label, model=model, scale=scale, budget=budget, device=device)
    row.update({k: v for k, v in solver.rec.items() if k not in ("cands", "memory", "rt", "saved", "recomp", "joint_graph")})
    row.update(static_step_peak(cap["fw"], cap["bw"]))
    row.update(closure_metrics(solver.rec))
    # CUDA: whole step vs resident baseline.  CPU self-test: backward-only new allocations.
    row["static_compare_MB"] = row["static_peak_MB"] if device == "cuda" else row["static_bw_newalloc_MB"]

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
        row["measured_peak_MB"] = statistics.median(peaks) / 1e6
        row["step_ms"] = statistics.median(ms)
    else:
        for _ in range(repeats):
            loss = cm(*args)          # forward outside the mode (a mode would change what dynamo runs)
            mode = _LiveBytes()
            with mode:
                loss.backward()
            del loss
            peaks.append(mode.peak); m.zero_grad(set_to_none=False)
        row["measured_peak_MB"] = statistics.median(peaks) / 1e6
    row["measured_all_MB"] = [p / 1e6 for p in peaks]
    del m, cm, args
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


# ----------------------------------------------------------------------------
def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
            for t in range(i, j + 1): r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rk(a), rk(b); n = len(a)
    if n < 3: return float("nan")
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def plans(suite: str):
    L = []
    if suite == "selftest":
        for b in (0.05, 0.15, 0.3):
            L.append((f"toy nat {b:.2f}", "toy", 8, b, PlanSolver(), False))
        L.append(("toy 0.15 force-out MLP", "toy", 8, 0.15, PlanSolver(force_out=True), False))
        L.append(("toy 0.15 no-reorder", "toy", 8, 0.15, PlanSolver(), True))
        return L
    if suite == "toyfull":
        for b in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50):
            L.append((f"toy nat {b:.2f}", "toy16", 16, b, PlanSolver(), False))
        for b in (0.05, 0.15):
            for k in (4, 8, 12, 16):
                L.append((f"toy {b:.2f} force-out {k} MLP", "toy16", 16, b, PlanSolver(force_out_k=k), False))
            L.append((f"toy {b:.2f} no-reorder", "toy16", 16, b, PlanSolver(), True))
        return L
    for b in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        L.append((f"llama nat {b:.2f}", "llama", 8, b, PlanSolver(), False))
    L.append(("llama 0.15 force-out MLP", "llama", 8, 0.15, PlanSolver(force_out=True), False))
    L.append(("llama 0.05 force-in all MLP (over budget)", "llama", 8, 0.05, PlanSolver(force_in_k=32, allow_over_budget=True), False))
    for k in (8, 16, 24):
        L.append((f"llama 0.05 force-in {k} MLP", "llama", 8, 0.05, PlanSolver(force_in_k=k), False))
    for b in (0.05, 0.15):
        L.append((f"llama {b:.2f} no-reorder", "llama", 8, b, PlanSolver(), True))
    for b in (0.05, 0.10, 0.15):
        L.append((f"bert nat {b:.2f}", "bert", 8, b, PlanSolver(), False))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["full", "selftest", "toyfull"], default="full")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/tmp/ackaudit_liveness_oracle.json")
    a = ap.parse_args()
    device = "cuda" if (a.suite == "full") else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required for --suite full")

    rows = []
    for label, model, scale, b, solver, noreorder in plans(a.suite):
        print(f"[{label}]", flush=True)
        r = run_plan(label, model, scale, b, device, a.repeats, solver, noreorder)
        rows.append(r)
        print(f"   measured={r['measured_peak_MB']:.1f}MB  static={r['static_compare_MB']:.1f}MB "
              f"(bw {r['static_bw_peak_MB']:.1f}, recomputed-live@peak {r['static_recomputed_live_at_bw_peak_MB']:.1f})  "
              f"closure_max={r['closure_max']}  KE={r['knapsack_evaluator_peak']:.4f}  "
              f"saved_w={r['saved_weight']:.4f}  mlp_saved={r['mlp_saved']}  saved_ops={r['saved_ops']}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1, default=str)

    # ------------------------------------------------------------------ verdict
    by = {r["plan"]: r for r in rows}
    primary = [r for r in rows if not r["plan"].startswith("bert")]
    meas = [r["measured_peak_MB"] for r in primary]
    print("\nSpearman vs measured over", len(primary), "plans (non-BERT):")
    rho = {}
    for key in ("static_compare_MB", "closure_max", "closure_max_w", "closure_sum_w", "knapsack_evaluator_peak", "saved_weight"):
        rho[key] = spearman([r[key] for r in primary], meas)
        print(f"   {key:28s} {rho[key]:+.3f}")
    rel = [abs(r["static_compare_MB"] - r["measured_peak_MB"]) / r["measured_peak_MB"] for r in primary]
    med_rel = statistics.median(rel)
    print(f"median |static-measured|/measured = {med_rel:.3f}")
    ok_c, bad_c = [], []
    for hi, lo in CONTRASTS:
        if hi in by and lo in by:
            good = by[hi]["static_compare_MB"] > by[lo]["static_compare_MB"]
            real = by[hi]["measured_peak_MB"] > by[lo]["measured_peak_MB"]
            (ok_c if good == real else bad_c).append((hi, lo, good, real))
    for hi, lo, g, r_ in bad_c:
        print(f"   CONTRAST MISS: static says {hi} {'>' if g else '<='} {lo}; measured says {'>' if r_ else '<='}")
    # closure cannot see reorder (function of saved set only):
    for b in ("0.05", "0.15"):
        n, x = by.get(f"llama nat {b}") or by.get(f"toy nat {b}"), by.get(f"llama {b} no-reorder") or by.get(f"toy {b} no-reorder")
        if n and x:
            print(f"   reorder ablation @{b}: closure_max {n['closure_max']} -> {x['closure_max']}, "
                  f"measured {n['measured_peak_MB']:.1f} -> {x['measured_peak_MB']:.1f}, "
                  f"static {n['static_compare_MB']:.1f} -> {x['static_compare_MB']:.1f}")
    passed = rho["static_compare_MB"] >= RHO_MIN and med_rel <= MEDIAN_REL_ERR_MAX and not bad_c
    print("\nPRE-REGISTERED VERDICT:", "PASS" if passed else "FAIL",
          f"(rho>={RHO_MIN}, median rel err<={MEDIAN_REL_ERR_MAX}, all {len(ok_c) + len(bad_c)} available contrasts sign-correct)")
    print(f"wrote {a.out}; no repository files modified.")


if __name__ == "__main__":
    main()
