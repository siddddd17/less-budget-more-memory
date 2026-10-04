"""
rand4x_switch.py -- control and observe Inductor's CUDA "4x" random path (pytorch#198333). LOCAL ONLY.

Background. Since PyTorch 2.13 (PR #184377), Inductor's Triton codegen for rand/randn uses
triton_helpers.rand4x when the kernel is 1-D (TritonOverrides._can_use_4x_random: CUDA, triton_tensor_ndim()==1,
align_random_eager off) and tl.rand otherwise. The two are different functions of (seed, offset):
tl.rand(seed, i) is word 0 of philox(seed, i); rand4x gives word i % 4 of philox(seed, i // 4).
Inductor recomputes dropout in the backward from the saved seed. If the forward generated a mask in a 2-D kernel
(e.g. fused into a LayerNorm var_mean reduction) and the backward regenerates it in a 1-D kernel (or the reverse),
the backward uses a different mask and the gradients are wrong. Which kernel shape is used depends on what the
partitioner saved, so a legal saved set can trigger it. CPU never takes the 4x path.

Modes (env RAND4X, or install(mode)):
  on     PyTorch's default behaviour (4x path where eligible). The probe still records every decision.
  off    _can_use_4x_random() always returns False: every kernel uses tl.rand (the 2.12 behaviour).
  align  torch._inductor.config.align_random_eager = True (the workaround suggested in #198333; it also changes
         how random offsets are generated, so random streams differ from 'off').

Probe: STATS counts codegen decisions as (graph, kernel ndim, function used) with graph in {fw, bw}. Codegen may run
more than once per kernel (fusion benchmarking), so counts are indicative; which combinations occur is what matters.
"""
from __future__ import annotations

import collections
import os

STATS: collections.Counter = collections.Counter()
STATE = {"mode": None, "patched": False}


def _patch() -> None:
    from torch._inductor.codegen import triton as T
    from torch._inductor.virtualized import V
    if STATE["patched"]:
        return
    if not hasattr(T.TritonOverrides, "_can_use_4x_random"):
        STATE["patched"] = "absent"          # older PyTorch: no 4x path, nothing to switch
        return
    orig = T.TritonOverrides._can_use_4x_random

    def can_use_4x_random():
        would = orig()
        try:
            g = "bw" if getattr(V.graph, "is_backward", False) else "fw"
        except Exception:
            g = "?"
        try:
            nd = V.kernel.triton_tensor_ndim()
        except Exception:
            nd = None
        used = bool(would) and STATE["mode"] == "on"
        STATS[(g, nd, "rand4x" if used else "tl.rand")] += 1
        return used

    T.TritonOverrides._can_use_4x_random = staticmethod(can_use_4x_random)
    STATE["patched"] = True


def install(mode: str | None = None) -> str:
    """Set the mode ('on' | 'off' | 'align'). Safe to call repeatedly; takes effect at the next compile."""
    import torch._inductor.config as IC
    mode = (mode or os.environ.get("RAND4X") or "on").lower()
    if mode not in ("on", "off", "align"):
        raise ValueError(f"RAND4X must be on|off|align, got {mode!r}")
    _patch()
    IC.align_random_eager = mode == "align"
    STATE["mode"] = mode
    return mode


def reset_stats() -> None:
    STATS.clear()


def summary() -> str:
    if STATE["patched"] == "absent":
        return "no 4x path in this PyTorch"
    if not STATS:
        return "no Triton random codegen (CPU, or no dropout)"
    return ", ".join(f"{g}/{nd}D:{fn}x{c}" for (g, nd, fn), c in sorted(STATS.items(), key=str))


def used_4x() -> bool:
    return any(fn == "rand4x" for (_g, _nd, fn) in STATS)


def mixed_fw_bw() -> bool:
    """True if forward and backward generated random numbers with different functions (the #198333 pattern)."""
    fns = collections.defaultdict(set)
    for (g, _nd, fn) in STATS:
        fns[g].add(fn)
    return bool(fns["fw"] and fns["bw"]) and (fns["fw"] | fns["bw"]) == {"rand4x", "tl.rand"}
