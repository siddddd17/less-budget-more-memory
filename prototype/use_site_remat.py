"""
use_site_remat.py -- prototype: use-site rematerialization for AOTAutograd's backward graph.

Problem (pytorch/pytorch#197838). Under activation_memory_budget, the min-cut partitioner decides which forward values
are saved and which are recomputed. Each recomputed value is defined once in the emitted backward graph:
reordering_to_mimic_autograd_engine inserts it at its first backward use, and it then stays live until its last use.
When the first use is early (e.g. rebuilding the residual stream for the first backward op) and the last use is late
(the value's own layer gradient), the value is held across the backward peak. At small budgets many values are in this
state, and peak memory rises as the budget falls.

What this pass does. After partitioning, it walks the emitted backward graph, finds tensors that are live at the peak
and are used again after it, and gives those late uses their own copy of the tensor's recompute cone, inserted just
before the first late use. The original copy then dies after its early uses. The cone's leaves must already be live at
the insertion point (saved activations, or values still in use there), so no lifetime is extended. An edit is kept
only if the peak does not rise and (peak, memory-time area) strictly falls; otherwise it is undone. This is the idea of
XLA's HloRematerialization pass, applied to AOTAutograd's backward graph.

Guarantees:
  * the saved set (forward outputs) is never changed, so the budget's meaning and the saved bytes are the stock ones;
  * random ops (tag nondeterministic_seeded, partitioners.is_rng_op, Inductor's RNG prims) are never duplicated:
    a duplicate could draw new values, or on CUDA hit pytorch/pytorch#198333; the pass asserts the count does not grow;
  * mutating ops, collectives and *_backward ops are never cloned;
  * cones are bounded (max_cone nodes) and the search is bounded (max_iters, patience).

Modes: "pointwise" clones only non-compute-intensive ops (which Inductor fuses into their consumers, so the extra work
is nearly free); "all" may also clone matmuls (lower peak, more compute). Neither mode clones attention: every
scaled-dot-product-attention op carries the nondeterministic_seeded tag, even without dropout, so the random-op rule
excludes it.

The peak model is a static walk of the backward graph (allocate at definition, free after last use, views keep their
base alive). It is exact for eager-style execution of the emitted graph and only approximate under Inductor, which
fuses and reuses buffers.

Prototype integration: enable() patches the partition function that torch.compile's Inductor backend uses, so
model code is unchanged. An upstream version would be an opt-in config flag that runs this after
reordering_to_mimic_autograd_engine inside min_cut_rematerialization_partition.

    import use_site_remat
    torch._functorch.config.activation_memory_budget = 0.05
    with use_site_remat.enabled("pointwise"):
        loss = torch.compile(model)(x); loss.backward()       # compile happens inside the context
    print(use_site_remat.LAST_STATS)

Tested with torch 2.14 (CPU/aot_eager and CUDA/Inductor). Research code: not for production use.
"""
from __future__ import annotations

import collections
import contextlib
import operator
import time
from typing import Any, Callable

import torch
import torch.fx as fx
import torch._functorch.partitioners as P
from torch._functorch.partitioners import _size_of

__all__ = ["rematerialize_backward", "make_partition_fn", "enabled", "count_random", "LAST_STATS"]

LAST_STATS: list[dict] = []          # one entry per backward graph processed by enable()/enabled()

_VIEWS = {"view", "_unsafe_view", "reshape", "expand", "t", "transpose", "permute", "slice",
          "select", "unsqueeze", "squeeze", "alias", "as_strided", "detach"}
_RANDOM = {"native_dropout", "rand_like", "randn_like", "rand", "randn", "bernoulli", "randint",
           "inductor_random", "inductor_randint", "inductor_seeds", "inductor_seed", "inductor_lookup_seed",
           "philox_rand", "rand_eager_offset", "rand_eager_offsets", "run_and_save_rng_state",
           "run_with_rng_state"}
_EXTRA_HEAVY = {"mm", "bmm", "addmm", "convolution", "_scaled_dot_product_flash_attention_for_cpu",
                "_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention",
                "_flash_attention_forward", "_efficient_attention_forward"}
_heavy_cache: set[str] | None = None


# ---------------------------------------------------------------------------------------------------- op predicates
def _op(n: fx.Node) -> str:
    t = str(getattr(n, "target", ""))
    return t.split(".")[1] if t.startswith("aten.") else t


def _base_name(n: fx.Node) -> str:
    parts = str(getattr(n, "target", "")).split(".")
    return parts[1] if len(parts) >= 2 and parts[0] in ("aten", "prims", "inductor", "rngprims") else parts[0]


def _is_random(n: fx.Node) -> bool:
    if n.op != "call_function":
        return False
    if torch.Tag.nondeterministic_seeded in getattr(n.target, "tags", ()):
        return True
    try:
        if P.is_rng_op(n):
            return True
    except Exception:
        pass
    return _base_name(n) in _RANDOM


def _is_collective(n: fx.Node) -> bool:
    return str(getattr(n, "target", "")).startswith(("_c10d_functional", "c10d", "_dtensor"))


def count_random(gm: fx.GraphModule) -> int:
    """Number of random-number ops in a graph."""
    return sum(1 for n in gm.graph.nodes if _is_random(n))


def _heavy_ops() -> set[str]:
    global _heavy_cache
    if _heavy_cache is None:
        _heavy_cache = {str(x).split(".")[-1] for x in P.get_default_op_list().compute_intensive_ops} | _EXTRA_HEAVY
    return _heavy_cache


def _recomputable(n: fx.Node, mode: str) -> bool:
    if n.op != "call_function":
        return False
    op = _op(n)
    if _is_random(n) or _is_collective(n) or op.endswith("_") or "copy_" in op or "backward" in op:
        return False
    if n.target is operator.getitem:
        return True
    if not isinstance(n.target, torch._ops.OpOverload) or n.target._schema.is_mutable:
        return False
    return not (mode == "pointwise" and op in _heavy_ops())


def _flops(n: fx.Node) -> float:
    """Rough added work: 2*M*N*K for matmuls, output numel otherwise (for reporting only)."""
    try:
        if _op(n) in ("mm", "bmm", "addmm"):
            a, b = n.args[-2].meta["val"], n.args[-1].meta["val"]
            return 2.0 * a.numel() * b.shape[-1]
        v = n.meta.get("val")
        return float(v.numel()) if isinstance(v, torch.Tensor) else 0.0
    except Exception:
        return 0.0


def _alloc(n: fx.Node) -> int:
    if n.op != "call_function" or isinstance(n.meta.get("val"), (tuple, list)) or _op(n) in _VIEWS:
        return 0
    try:
        return _size_of(n)
    except Exception:
        return 0


# ---------------------------------------------------------------------------------------------------- liveness model
def _timeline(gm: fx.GraphModule) -> dict:
    """Static liveness of the backward graph: bytes live after each node, the peak and what is live there.
    Non-parameter inputs (saved activations, tangents) start live; views keep their base alive."""
    nodes = list(gm.graph.nodes); idx = {n: i for i, n in enumerate(nodes)}
    root = {}
    for n in nodes:
        root[n] = root[n.all_input_nodes[0]] if (n.op == "call_function" and _op(n) in _VIEWS
                                                 and n.all_input_nodes) else n
    last: dict = {}
    for n in nodes:
        for a in n.all_input_nodes:
            last[root[a]] = max(last.get(root[a], -1), idx[n])
    live: dict = {}; frees = collections.defaultdict(list); cur = 0; curve = []
    for n in nodes:
        if n.op == "placeholder" and not n.name.startswith("primals") and "val" in n.meta:
            s = _size_of(n); live[n] = s; cur += s; frees[last.get(n, idx[n])].append(n)
    peak, pk_i, at = cur, 0, dict(live)
    for n in nodes:
        s = _alloc(n) if root[n] is n else 0
        if s:
            live[n] = s; cur += s; frees[last.get(n, idx[n])].append(n)
        curve.append(cur)
        if cur > peak:
            peak, pk_i, at = cur, idx[n], dict(live)
        for d in frees.pop(idx[n], []):
            cur -= live.pop(d, 0)
    groups = collections.defaultdict(list)
    for n in nodes:
        groups[root[n]].append(n)
    return dict(nodes=nodes, idx=idx, root=root, last=last, peak=peak, pk=pk_i, at=at, curve=curve, groups=groups)


def _key(tl: dict) -> tuple:
    return (tl["peak"], sum(tl["curve"]))


# ---------------------------------------------------------------------------------------------------- one edit
def _try_remat(gm: fx.GraphModule, t: fx.Node, tl: dict, mode: str, max_cone: int):
    """Give t's uses after the peak a fresh copy of t's recompute cone. Returns (undo record or None, added work)."""
    idx, pk = tl["idx"], tl["pk"]
    late = sorted((u for u in t.users if idx[u] > pk), key=lambda u: idx[u])
    if not late or not _recomputable(t, mode):
        return None, 0.0
    ins_at = late[0]; ins_i = idx[ins_at]; late_set = set(late)

    def need(a: fx.Node) -> bool:
        """Must a be cloned too? Not if it is a graph input or still live at the insertion point anyway."""
        if a.op in ("placeholder", "get_attr"):
            return False
        for g in tl["groups"][tl["root"][a]]:
            for u in g.users:
                if u not in late_set and idx[u] >= ins_i:
                    return False
        return True

    cone: set = set(); stack = [t]
    while stack:
        x = stack.pop()
        if x in cone:
            continue
        if not _recomputable(x, mode) or len(cone) >= max_cone:
            return None, 0.0
        cone.add(x)
        stack.extend(a for a in x.all_input_nodes if a not in cone and need(a))
    order = sorted(cone, key=lambda x: idx[x])
    env: dict = {}; work = 0.0
    with gm.graph.inserting_before(ins_at):
        for x in order:
            env[x] = gm.graph.node_copy(x, lambda a: env.get(a, a))
            env[x].meta = dict(x.meta)
            work += _flops(x)
    for u in late:
        u.replace_input_with(t, env[t])
    return (t, env[t], late, [env[x] for x in order]), work


def _undo(gm: fx.GraphModule, undo) -> None:
    t, tc, late, clones = undo
    for u in late:
        u.replace_input_with(tc, t)
    for c in reversed(clones):
        if not c.users:
            gm.graph.erase_node(c)


# ---------------------------------------------------------------------------------------------------- the pass
def rematerialize_backward(gm: fx.GraphModule, mode: str = "pointwise", max_iters: int = 400, max_cone: int = 12,
                           patience: int = 60, max_accepted: int | None = None) -> dict:
    """Run the pass in place on an AOTAutograd backward GraphModule. Returns statistics:
    peak_before/peak_after (static model, bytes), accepted edits, added work, random-op counts, and the predicted
    peak after each accepted edit. max_accepted applies only the first k edits (the pass is deterministic)."""
    if mode not in ("pointwise", "all"):
        raise ValueError(f"mode must be 'pointwise' or 'all', got {mode!r}")
    t0 = time.perf_counter()
    tl = _timeline(gm); rng0 = count_random(gm)
    peak0, area0, work, accepted, fails = tl["peak"], sum(tl["curve"]), 0.0, 0, 0
    tried_bad: set = set(); trace = [tl["peak"]]
    for _ in range(max_iters):
        if max_accepted is not None and accepted >= max_accepted:
            break
        # candidates: values defined before the peak, live at it, and used after it; largest first
        cands = sorted((n for n in tl["nodes"] if n.op == "call_function" and n not in tried_bad
                        and tl["root"][n] in tl["at"] and tl["idx"][n] <= tl["pk"]
                        and any(tl["idx"][u] > tl["pk"] for u in n.users)),
                       key=lambda n: -tl["at"][tl["root"][n]])
        progressed = False
        for t in cands:
            undo, w = _try_remat(gm, t, tl, mode, max_cone)
            if undo is None:
                tried_bad.add(t); continue
            new = _timeline(gm)
            if new["peak"] <= tl["peak"] and _key(new) < _key(tl):
                gm.graph.eliminate_dead_code()
                tl = _timeline(gm); work += w; accepted += 1; progressed = True
                trace.append(tl["peak"])
                break
            _undo(gm, undo); tl = _timeline(gm)
            tried_bad.add(t); fails += 1
            if fails > patience:
                break
        if not progressed or fails > patience:
            break
    gm.graph.lint(); gm.recompile()
    rng1 = count_random(gm)
    assert rng1 <= rng0, f"use-site remat duplicated random ops ({rng0} -> {rng1})"
    return dict(mode=mode, peak_before=peak0, peak_after=tl["peak"], area_before=area0, area_after=sum(tl["curve"]),
                accepted=accepted, added_work=work, random_ops_before=rng0, random_ops_after=rng1, peak_trace=trace,
                seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------------------- integration
def make_partition_fn(mode: str = "pointwise", base: Callable | None = None, **kw) -> Callable:
    """A partition_fn for aot_autograd(...) that runs the stock min-cut partitioner, then this pass on the backward."""
    base = base or P.min_cut_rematerialization_partition

    def partition(joint_module, joint_inputs, **pkw):
        fw, bw = base(joint_module, joint_inputs, **pkw)
        LAST_STATS.append(rematerialize_backward(bw, mode, **kw))
        return fw, bw
    return partition


@contextlib.contextmanager
def enabled(mode: str = "pointwise", **kw):
    """Within this context, torch.compile's Inductor backend partitions with the pass applied (prototype hook:
    patches torch._inductor.compile_fx.min_cut_rematerialization_partition). Compilation must happen inside it."""
    import torch._inductor.compile_fx as CF
    orig = CF.min_cut_rematerialization_partition
    CF.min_cut_rematerialization_partition = make_partition_fn(mode, base=orig, **kw)
    try:
        yield LAST_STATS
    finally:
        CF.min_cut_rematerialization_partition = orig
