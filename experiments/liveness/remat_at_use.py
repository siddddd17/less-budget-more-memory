"""
remat_at_use.py -- Experiment E: liveness-driven rematerialization *after* partitioning (LOCAL ONLY).

Idea (the XLA HloRematerialization idea applied to AOTAutograd's emitted backward):
PyTorch's partitioner allows each forward value to be recomputed at most once in the
backward, and reordering_to_mimic_autograd_engine inserts that recomputation at the value's
FIRST use. When a recomputed value has an early use (e.g. rebuilding the residual stream)
and a late use (its own layer's gradient), it stays live across the peak.

This pass walks the emitted backward, finds tensors live at the peak that have uses
*after* the peak, and replaces those late uses with a fresh recomputation of the tensor's
cone, inserted just before the first late use. The cone's leaves must already be live at
that point: saved activations or values still in use there. The original then dies after
its early uses. Changes are accepted only if the global peak does not rise and the memory
at the old peak falls.

It never changes the saved set, so the activation_memory_budget semantics and the saved
bytes are exactly the stock ones. The cost is duplicated compute. mode="pointwise" allows
only non-compute-intensive, non-random ops in cones, which Inductor would fuse. mode="all"
also allows matmuls. (Attention is never cloned in either mode: SDPA ops carry the nondeterministic_seeded tag,
which the random-op filter excludes.)
"""
from __future__ import annotations

import collections
import operator

import torch
import torch.fx as fx
import torch._functorch.partitioners as P

import liveness_solver as LS

_HEAVY = None
_RANDOM = {"native_dropout", "rand_like", "randn_like", "rand", "randn", "bernoulli", "randint",
           "inductor_random", "inductor_randint", "inductor_seeds", "inductor_seed", "inductor_lookup_seed",
           "philox_rand", "rand_eager_offset", "rand_eager_offsets", "run_and_save_rng_state",
           "run_with_rng_state"}


def _base_name(n) -> str:
    """Op name without namespace or overload: aten.native_dropout.default and
    prims.inductor_random.default both map to their middle part."""
    t = str(getattr(n, "target", ""))
    parts = t.split(".")
    return parts[1] if len(parts) >= 2 and parts[0] in ("aten", "prims", "inductor", "rngprims") else parts[0]


def _is_random(n) -> bool:
    """Random-number ops must never be duplicated: a duplicate can redraw (fresh seed) or, on CUDA, regenerate a
    different value in a differently shaped kernel (PyTorch issue #198333). Checked by the
    nondeterministic_seeded tag, the partitioner's is_rng_op, and by name for Inductor's untagged RNG prims."""
    if n.op != "call_function":
        return False
    tags = getattr(n.target, "tags", ())
    if torch.Tag.nondeterministic_seeded in tags:
        return True
    try:
        if P.is_rng_op(n):
            return True
    except Exception:
        pass
    return _base_name(n) in _RANDOM


def _is_collective(n) -> bool:
    t = str(getattr(n, "target", ""))
    return t.startswith(("_c10d_functional", "c10d", "_dtensor"))


def count_random(gm) -> int:
    return sum(1 for n in gm.graph.nodes if _is_random(n))


def _heavy_ops():
    global _HEAVY
    if _HEAVY is None:
        o = P.get_default_op_list()
        _HEAVY = {str(x).split(".")[-1] for x in o.compute_intensive_ops} | {
            "mm", "bmm", "addmm", "convolution", "_scaled_dot_product_flash_attention_for_cpu",
            "_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention",
            "_flash_attention_forward", "_efficient_attention_forward"}
    return _HEAVY


def _recomputable(n, mode) -> bool:
    if n.op != "call_function":
        return False
    op = LS._op(n)
    if _is_random(n) or _is_collective(n) or op.endswith("_") or "copy_" in op or "backward" in op:
        return False
    if n.target is operator.getitem:
        return True
    if not isinstance(n.target, torch._ops.OpOverload):
        return False
    if n.target._schema.is_mutable:
        return False
    if mode == "pointwise" and op in _heavy_ops():
        return False
    return True


def _flops(n) -> float:
    op = LS._op(n)
    try:
        if op in ("mm", "bmm", "addmm"):
            a, b = n.args[-2].meta["val"], n.args[-1].meta["val"]
            return 2.0 * a.numel() * b.shape[-1]
        v = n.meta.get("val")
        return float(v.numel()) if isinstance(v, torch.Tensor) else 0.0
    except Exception:
        return 0.0


def _timeline(gm):
    """Peak step, bytes at every step, and last-use index per node (views extend their base)."""
    nodes = list(gm.graph.nodes); idx = {n: i for i, n in enumerate(nodes)}
    root = {}
    for n in nodes:
        root[n] = root[n.all_input_nodes[0]] if (n.op == "call_function" and LS._op(n) in LS._VIEWS
                                                 and n.all_input_nodes) else n
    last = {}
    for n in nodes:
        for a in n.all_input_nodes:
            last[root[a]] = max(last.get(root[a], -1), idx[n])
    live = {}; frees = collections.defaultdict(list); cur = 0; curve = []
    for n in nodes:
        if n.op == "placeholder" and not n.name.startswith("primals") and "val" in n.meta:
            s = LS._size_of(n); live[n] = s; cur += s; frees[last.get(n, idx[n])].append(n)
    peak, pk_i, at = cur, 0, dict(live)
    for n in nodes:
        s = LS._alloc(n) if root[n] is n else 0
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


def _try_remat(gm, t, tl, mode, max_cone):
    """Clone t's cone before its first use after the peak; rewire late uses. Returns (ok, flops)."""
    nodes, idx, last, pk = tl["nodes"], tl["idx"], tl["last"], tl["pk"]
    late = sorted((u for u in t.users if idx[u] > pk), key=lambda u: idx[u])
    if not late or not _recomputable(t, mode):
        return None, 0.0
    ins_at = late[0]; ins_i = idx[ins_at]
    cone, order, ok = set(), [], True

    late_set = set(late)

    def need(a):
        if a.op in ("placeholder", "get_attr"):
            return False
        # still live at the insertion point for reasons other than the uses we are rewiring?
        for g in tl["groups"][tl["root"][a]]:
            for u in g.users:
                if u not in late_set and idx[u] >= ins_i:
                    return False  # reuse it, its lifetime is not extended
        return True

    stack = [t]
    while stack:
        x = stack.pop()
        if x in cone:
            continue
        if not _recomputable(x, mode) or len(cone) >= max_cone:
            ok = False; break
        cone.add(x)
        for a in x.all_input_nodes:
            if a not in cone and need(a):
                stack.append(a)
    if not ok:
        return None, 0.0
    order = sorted(cone, key=lambda x: idx[x])
    env, flops = {}, 0.0
    with gm.graph.inserting_before(ins_at):
        for x in order:
            env[x] = gm.graph.node_copy(x, lambda a: env.get(a, a))
            env[x].meta = dict(x.meta)
            flops += _flops(x)
    for u in late:
        u.replace_input_with(t, env[t])
    undo = (t, env[t], late, [env[x] for x in order])
    return undo, flops


def _undo(gm, undo):
    t, tc, late, clones = undo
    for u in late:
        u.replace_input_with(tc, t)
    for c in reversed(clones):
        if not c.users:
            gm.graph.erase_node(c)


def _key(tl):
    return (tl["peak"], sum(tl["curve"]))


def rematerialize_backward(gm: fx.GraphModule, mode: str = "pointwise", max_iters: int = 400,
                           max_cone: int = 12, patience: int = 60, max_accepted: int | None = None) -> dict:
    """Greedy, liveness-driven: repeatedly shorten the lifetime of the largest tensor that is
    live at the peak and used after it. A change is accepted only if the peak does not rise and
    (peak, memory-time area) strictly decreases. max_accepted stops after that many accepted edits (the pass is
    deterministic, so this applies a prefix of the full pass's edits; used by remat_edit_validation.py)."""
    tl = _timeline(gm)
    rng0 = count_random(gm)
    peak0, area0, flops_added, accepted, fails = tl["peak"], sum(tl["curve"]), 0.0, 0, 0
    tried_bad: set = set()
    trace = [tl["peak"]]                       # predicted backward peak after 0, 1, 2, ... accepted edits
    for _ in range(max_iters):
        if max_accepted is not None and accepted >= max_accepted:
            break
        cands = sorted((n for n in tl["nodes"] if n.op == "call_function" and n not in tried_bad
                        and tl["root"][n] in tl["at"] and tl["idx"][n] <= tl["pk"]
                        and any(tl["idx"][u] > tl["pk"] for u in n.users)),
                       key=lambda n: -tl["at"][tl["root"][n]])
        progressed = False
        for t in cands:
            undo, fl = _try_remat(gm, t, tl, mode, max_cone)
            if undo is None:
                tried_bad.add(t); continue
            new = _timeline(gm)
            if new["peak"] <= tl["peak"] and _key(new) < _key(tl):
                # drop the original if its early uses were all it had
                gm.graph.eliminate_dead_code()
                tl = _timeline(gm); flops_added += fl; accepted += 1; progressed = True
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
    assert rng1 <= rng0, f"remat pass duplicated random ops ({rng0} -> {rng1})"
    return dict(peak_before=peak0, peak_after=tl["peak"], area_before=area0, area_after=sum(tl["curve"]),
                accepted=accepted, flops_added=flops_added, random_ops_before=rng0, random_ops_after=rng1,
                peak_trace=trace)
