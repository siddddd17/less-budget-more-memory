"""
demo_issue_197838.py -- the pytorch/pytorch#197838 setting, with and without the use-site remat prototype.

Model: HF Llama (32 layers, hidden 256, MLP 688, 8 heads, vocab 1000, random weights), batch 2 x 1024 tokens, fp32,
torch.compile (Inductor). For each budget it compiles three times (stock, pass in "pointwise" mode, pass in "all"
mode), measures max_memory_allocated over a training step (minus resident memory) and the step time, and checks the
gradients against eager execution.

  python prototype/demo_issue_197838.py                       # CUDA, budgets 0.05 0.10 0.15 0.20 0.30 (~25 min, 4 GB GPU)
  python prototype/demo_issue_197838.py --budgets 0.05 0.15 --layers 8
"""
from __future__ import annotations

import argparse, os, statistics, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import use_site_remat as U
from torch._functorch import config as fc
import torch._inductor.config as IC


def build(layers, seq, inter):
    from transformers import LlamaConfig, LlamaModel

    class Wrap(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.model = LlamaModel(LlamaConfig(hidden_size=256, intermediate_size=inter, num_hidden_layers=layers,
                                             num_attention_heads=8, num_key_value_heads=8, vocab_size=1000,
                                             use_cache=False))

        def forward(s, ids):
            return s.model(input_ids=ids).last_hidden_state.sum()
    torch.manual_seed(197838)
    return Wrap().cuda(), torch.randint(0, 1000, (2, seq), device="cuda")


def step(fn, m, ids):
    fn(ids).backward(); torch.cuda.synchronize()


def measure(m, ids, budget, mode, repeats=3, iters=10):
    torch._dynamo.reset(); m.zero_grad(set_to_none=True)
    fc.activation_memory_budget = budget
    try:
        if mode:
            with U.enabled(mode):
                cm = torch.compile(m); step(cm, m, ids)
            st = U.LAST_STATS[-1]
        else:
            cm = torch.compile(m); step(cm, m, ids); st = None
    finally:
        fc.activation_memory_budget = 1.0
    m.zero_grad(set_to_none=False); torch.cuda.empty_cache(); resident = torch.cuda.memory_allocated(); peaks = []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats(); step(cm, m, ids)
        peaks.append(torch.cuda.max_memory_allocated() - resident); m.zero_grad(set_to_none=False)
    step(cm, m, ids); grads = [p.grad.detach().float().cpu() for p in m.parameters()]; m.zero_grad(set_to_none=False)
    ms = []
    for _ in range(iters):
        t = time.perf_counter(); step(cm, m, ids); ms.append((time.perf_counter() - t) * 1e3); m.zero_grad(set_to_none=False)
    return statistics.median(peaks) / 1e6, statistics.median(ms), grads, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--layers", type=int, default=32); ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=688)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "this demo measures CUDA memory; see test_use_site_remat.py for CPU checks"
    IC.force_disable_caches = True
    m, ids = build(a.layers, a.seq, a.inter)
    m(ids).backward(); ref = [p.grad.detach().float().cpu() for p in m.parameters()]; m.zero_grad(set_to_none=True)
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}; Llama {a.layers}L, batch 2 x {a.seq}, I={a.inter}")
    print("| budget | stock peak MB | pointwise MB (vs stock) | all-ops MB (vs stock) | step ms stock / pw / all | max rel grad err vs eager |")
    print("|---|---|---|---|---|---|")
    for b in a.budgets:
        r = {mode: measure(m, ids, b, mode) for mode in (None, "pointwise", "all")}
        err = max(((g - e).norm() / e.norm()).item() for mode in r for g, e in zip(r[mode][2], ref) if e.norm() > 0)
        s0 = r[None][0]
        print(f"| {b:.2f} | {s0:.0f} | {r['pointwise'][0]:.0f} ({100 * (r['pointwise'][0] / s0 - 1):+.1f}%) | "
              f"{r['all'][0]:.0f} ({100 * (r['all'][0] / s0 - 1):+.1f}%) | "
              f"{r[None][1]:.0f} / {r['pointwise'][1]:.0f} / {r['all'][1]:.0f} | {err:.1e} |", flush=True)


if __name__ == "__main__":
    main()
