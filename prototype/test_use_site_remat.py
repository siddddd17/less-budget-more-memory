"""CPU tests for use_site_remat.py (pytest prototype/test_use_site_remat.py, ~2-4 min)."""
from __future__ import annotations

import os, sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._functorch.partitioners as P
from torch._functorch import config as fc
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.debugging import boxed_nop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import use_site_remat as U


class RMS(nn.Module):
    def __init__(s, d):
        super().__init__(); s.w = nn.Parameter(torch.ones(d))

    def forward(s, x):
        return s.w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))


class Block(nn.Module):
    def __init__(s, d=128, f=344, h=4, p=0.0):
        super().__init__(); s.h, s.p = h, p; s.n1, s.n2 = RMS(d), RMS(d)
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        s.g, s.u, s.dn = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

    def forward(s, x):
        B, T, D = x.shape; y = s.n1(x)
        q, k, v = (m(y).view(B, T, s.h, D // s.h).transpose(1, 2) for m in (s.q, s.k, s.v))
        x = x + F.dropout(s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)),
                          s.p, s.training)
        y = s.n2(x); return x + s.dn(F.silu(s.g(y)) * s.u(y))


class Toy(nn.Module):
    """Llama-style pre-norm decoder (the structure behind pytorch/pytorch#197838), small enough for CPU."""

    def __init__(s, L=6, V=500, d=128, p=0.0):
        super().__init__(); s.e = nn.Embedding(V, d); s.b = nn.ModuleList(Block(d=d, p=p) for _ in range(L))
        s.n = RMS(d); s.out = nn.Linear(d, V, bias=False)

    def forward(s, ids):
        x = s.e(ids)
        for b in s.b:
            x = b(x)
        return F.cross_entropy(s.out(s.n(x)).flatten(0, 1), ids.flatten())


def compile_step(mode, budget=0.05, p=0.0, seed=0):
    """One compiled fwd+bw under aot_eager; returns grads, the fw and bw graphs, and the pass stats (or None)."""
    torch.manual_seed(seed); torch._dynamo.reset(); cap = {}
    m = Toy(p=p); ids = torch.randint(0, 500, (2, 128))

    def part(jm, ji, **kw):
        fw, bw = P.min_cut_rematerialization_partition(jm, ji, **kw)
        cap["fw_out"] = [str(a) for a in fw.graph.find_nodes(op="output")[0].args[0]]
        cap["rng_before"] = U.count_random(bw)
        if mode:
            cap["stats"] = U.rematerialize_backward(bw, mode)
        cap["bw"] = bw
        return fw, bw

    be = aot_autograd(fw_compiler=boxed_nop, bw_compiler=boxed_nop, partition_fn=part)
    prev = fc.activation_memory_budget
    try:
        fc.activation_memory_budget = budget
        torch.manual_seed(seed + 1)
        torch.compile(m, backend=be)(ids).backward()
    finally:
        fc.activation_memory_budget = prev
    return [q.grad.clone() for q in m.parameters()], cap


@pytest.mark.parametrize("mode", ["pointwise", "all"])
def test_lowers_static_peak_keeps_saved_set_and_grads(mode):
    g0, c0 = compile_step(None)
    g1, c1 = compile_step(mode)
    st = c1["stats"]
    assert c1["fw_out"] == c0["fw_out"], "the saved set must not change"
    assert st["accepted"] > 0 and st["peak_after"] < st["peak_before"], st
    assert st["peak_trace"] == sorted(st["peak_trace"], reverse=True), "peak never rises"
    for a, b in zip(g0, g1):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_all_mode_at_least_as_low_as_pointwise():
    _, cp = compile_step("pointwise"); _, ca = compile_step("all")
    assert ca["stats"]["peak_after"] <= cp["stats"]["peak_after"]


def test_never_duplicates_random_ops():
    g0, c0 = compile_step(None, p=0.1, seed=3)
    g1, c1 = compile_step("all", p=0.1, seed=3)
    assert c1["rng_before"] > 0, "the dropout model must contain random ops"
    assert c1["rng_before"] == c1["stats"]["random_ops_before"]
    assert c1["stats"]["random_ops_after"] <= c1["stats"]["random_ops_before"]
    for a, b in zip(g0, g1):          # same seeds, dropout masks are not redrawn
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_prefix_is_deterministic():
    _, cf = compile_step("pointwise")
    k = max(1, cf["stats"]["accepted"] // 2)
    torch._dynamo.reset()
    _, ref = compile_step(None)
    bw = ref["bw"]
    st = U.rematerialize_backward(bw, "pointwise", max_accepted=k)
    assert st["accepted"] == k and st["peak_trace"] == cf["stats"]["peak_trace"][:k + 1]


def test_budget_one_is_harmless():
    g0, _ = compile_step(None, budget=1.0)
    g1, c1 = compile_step("pointwise", budget=1.0)
    assert c1["stats"]["peak_after"] <= c1["stats"]["peak_before"]
    for a, b in zip(g0, g1):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        U.rematerialize_backward(torch.fx.symbolic_trace(nn.Linear(2, 2)), "everything")


def test_matches_paper_implementation():
    """Same edits as experiments/liveness/remat_at_use.py, which produced the paper's numbers."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "experiments", "liveness"))
    R = pytest.importorskip("remat_at_use")
    for mode in ("pointwise", "all"):
        _, a = compile_step(None); _, b = compile_step(None)
        sa, sb = U.rematerialize_backward(a["bw"], mode), R.rematerialize_backward(b["bw"], mode)
        assert sa["peak_trace"] == sb["peak_trace"] and sa["accepted"] == sb["accepted"]
        assert str(a["bw"].graph) == str(b["bw"].graph)
