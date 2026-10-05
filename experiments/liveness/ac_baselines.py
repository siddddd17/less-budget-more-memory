"""
ac_baselines.py -- the activation-checkpointing policies practitioners actually use, on the paper's Llama and BERT setups,
next to the paper's own reference points (best stock budget; use-site rematerialization).

Per model (scale 8: 32 layers, batch 2 x 1024 tokens, fp32), each config is one fresh compile, measured like
gpu_eval_remat.py (peak = max_memory_allocated over a step minus resident memory, median of --repeats; step time = median
of --iters CUDA-event timings; gradients copied to the CPU after the peak loop):

  eager                      no checkpointing, no compile (also the gradient reference)
  eager+ac1 / eager+ac2      torch.utils.checkpoint (non-reentrant) around every layer / every 2nd layer
  eager+sac_tt               selective AC, torchtitan's "op" policy: save attention and every other matmul, recompute the rest
  stock@1.0                  torch.compile (Inductor), default plan (activation_memory_budget = 1.0)
  ac1 / ac2 / sac_tt / sac_mm  the same checkpointing inside torch.compile (sac_mm: save every matmul and attention)
  ac1+remat(pw|all)          per-layer AC plus the use-site pass (does it compose?)
  Llama references, same process: stock@0.13 (best stock budget on the dense grid), stock+remat(all)@0.05 and @0.10

BERT runs with dropout on (HF default train mode), as in the paper; its gradients are not compared, because dropout
masks differ between eager and Inductor. Llama gradients are compared with the eager reference (relative L2 over all
parameters). Configs that fail are recorded with their error and the run continues.

Usage (repo root, TORCHINDUCTOR_COMPILE_THREADS=1, inside tmux; ~1.5-2 h on the GTX 1650):
  python experiments/liveness/ac_baselines.py --out experiments/liveness/results/ac_baselines.json
  python experiments/liveness/ac_baselines.py --device cpu --scale 1 --models llama --iters 1   # plumbing test only
"""
from __future__ import annotations

import argparse, collections, json, os, statistics, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gpu_eval_remat as G              # installs the remat hook on the Inductor partitioner (G.REMAT)
import experiment_c_inductor_runtime as C
import liveness_oracle_check as A

import torch
from torch import nn
from torch._functorch import config as fc
import torch._inductor.config as IC
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy

IC.force_disable_caches = True
aten = torch.ops.aten
MM = {aten.mm.default, aten.addmm.default, aten.bmm.default}
ATTN = {getattr(aten, n).default for n in ("_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention",
                                            "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_flash_attention_for_cpu")
        if hasattr(aten, n)}


def tt_policy_ctx():
    """torchtitan's op-level SAC: save attention and every other matmul; counts reset per checkpointed layer."""
    meta = collections.defaultdict(int)

    def policy(ctx, func, *args, **kwargs):
        key = "recompute_mm" if ctx.is_recompute else "forward_mm"
        if func in MM:
            meta[key] += 1
        save = func in ATTN or (func in MM and meta[key] % 2 == 1)
        return CheckpointPolicy.MUST_SAVE if save else CheckpointPolicy.PREFER_RECOMPUTE
    return create_selective_checkpoint_contexts(policy)


def mm_policy_ctx():
    def policy(ctx, func, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if (func in MM or func in ATTN) else CheckpointPolicy.PREFER_RECOMPUTE
    return create_selective_checkpoint_contexts(policy)


class Ckpt(nn.Module):
    def __init__(self, inner, context_fn=None):
        super().__init__(); self.inner = inner; self.context_fn = context_fn

    def forward(self, *args, **kwargs):
        if self.context_fn is None:
            return checkpoint(self.inner, *args, use_reentrant=False, **kwargs)
        return checkpoint(self.inner, *args, use_reentrant=False, context_fn=self.context_fn, **kwargs)


def layers_of(m):
    inner = m.model
    return inner.layers if hasattr(inner, "layers") else inner.encoder.layer


def wrap(m, policy):
    if policy is None:
        return m
    L = layers_of(m)
    for i in range(len(L)):
        if policy == "ac1":
            L[i] = Ckpt(L[i])
        elif policy == "ac2":
            if i % 2 == 0:
                L[i] = Ckpt(L[i])
        elif policy == "sac_tt":
            L[i] = Ckpt(L[i], tt_policy_ctx)
        elif policy == "sac_mm":
            L[i] = Ckpt(L[i], mm_policy_ctx)
        else:
            raise ValueError(policy)
    return m


def run(model, scale, device, compiled, policy, budget, remat, repeats, iters):
    torch.manual_seed(197838); torch._dynamo.reset(); C.CAP.clear(); C.INDUCTOR_EST.clear()
    G.REMAT["mode"], G.REMAT["stats"] = remat, {}
    build, make_inputs = A.resolve(model, scale)
    m = wrap(build(), policy).to(device); args = tuple(x.to(device) for x in make_inputs())
    cuda = device == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    fn = torch.compile(m, backend="inductor", dynamic=False) if compiled else m
    prev = fc.activation_memory_budget
    t0 = time.perf_counter()
    try:
        fc.activation_memory_budget = budget            # production knapsack solver (the default)
        fn(*args).backward(); sync()
    finally:
        fc.activation_memory_budget = prev
        G.REMAT["mode"] = None
    compile_s = time.perf_counter() - t0
    m.zero_grad(set_to_none=False)
    peaks, reserved = [], []
    if cuda:
        torch.cuda.empty_cache(); resident = torch.cuda.memory_allocated()
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats(); fn(*args).backward(); sync()
            peaks.append(torch.cuda.max_memory_allocated() - resident); reserved.append(torch.cuda.max_memory_reserved())
            m.zero_grad(set_to_none=False)
    fn(*args).backward(); sync()
    grads = [p.grad.detach().float().cpu().clone() for p in m.parameters()]
    m.zero_grad(set_to_none=False)
    for _ in range(2):
        fn(*args).backward(); m.zero_grad(set_to_none=False)
    ms = []
    for _ in range(iters):
        if cuda:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(*args).backward(); e.record(); sync(); ms.append(s.elapsed_time(e))
        else:
            t = time.perf_counter(); fn(*args).backward(); ms.append((time.perf_counter() - t) * 1e3)
        m.zero_grad(set_to_none=False)
    row = dict(model=model, scale=scale, device=device, compiled=compiled, policy=policy, budget=budget, remat=remat,
               peak_MB=statistics.median(peaks) / 1e6 if peaks else None, peak_all_MB=[p / 1e6 for p in peaks],
               reserved_peak_MB=max(reserved) / 1e6 if reserved else None,
               step_ms=statistics.median(ms), step_all_ms=ms, compile_s=compile_s,
               remat_stats={k: v for k, v in G.REMAT["stats"].items() if k != "peak_trace"},
               inductor_est_MB=max(C.INDUCTOR_EST.values()) if C.INDUCTOR_EST else None,
               static_peak_MB=(C.CAP.get("static") or {}).get("static_peak_MB"))
    del m, fn, args
    if cuda:
        torch.cuda.empty_cache()
    return row, grads


def rel_err(g, ref):
    num = sum(float((a - b).pow(2).sum()) for a, b in zip(g, ref))
    den = sum(float(b.pow(2).sum()) for b in ref)
    return (num / den) ** 0.5 if den else float("nan")


def configs(model):
    # (label, compiled, policy, budget, remat)
    c = [("eager", False, None, 1.0, None), ("eager+ac1", False, "ac1", 1.0, None), ("eager+ac2", False, "ac2", 1.0, None),
         ("eager+sac_tt", False, "sac_tt", 1.0, None),
         ("stock@1.0", True, None, 1.0, None), ("ac1", True, "ac1", 1.0, None), ("ac2", True, "ac2", 1.0, None),
         ("sac_tt", True, "sac_tt", 1.0, None), ("sac_mm", True, "sac_mm", 1.0, None),
         ("ac1+remat(pw)", True, "ac1", 1.0, "pointwise"), ("ac1+remat(all)", True, "ac1", 1.0, "all")]
    if model == "llama":
        c += [("stock@0.13", True, None, 0.13, None), ("stock+remat(all)@0.10", True, None, 0.10, "all"),
              ("stock+remat(all)@0.05", True, None, 0.05, "all")]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama", "bert"])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="run only these config labels")
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="experiments/liveness/results/ac_baselines.json")
    a = ap.parse_args()
    rows = []
    for model in a.models:
        ref = None
        for label, compiled, policy, budget, remat in configs(model):
            if a.only and label not in a.only and label != "eager":
                continue
            print(f"[{model} | {label}]", flush=True)
            try:
                r, g = run(model, a.scale, a.device, compiled, policy, budget, remat, a.repeats, a.iters)
            except Exception as e:  # record and continue (e.g. OOM, an unsupported policy)
                traceback.print_exc()
                rows.append(dict(model=model, label=label, error=f"{type(e).__name__}: {e}"[:500]))
                if a.device == "cuda":
                    torch.cuda.empty_cache()
                json.dump(rows, open(a.out, "w"), indent=1, default=str); continue
            r["label"] = label
            if label == "eager":
                ref = g
            r["grad_rel_err_vs_eager"] = rel_err(g, ref) if (ref is not None and model == "llama") else None
            rows.append(r)
            pk = f"{r['peak_MB']:.1f}MB" if r["peak_MB"] is not None else "n/a"
            print(f"   peak={pk} step={r['step_ms']:.1f}ms compile={r['compile_s']:.0f}s "
                  f"grad_err={r['grad_rel_err_vs_eager']} remat_accepted={r['remat_stats'].get('accepted')}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1, default=str)
    print("\n| model | config | peak MB | step ms | compile s | grad rel err |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['model']} | {r['label']} | error: {r['error'][:60]} | | | |"); continue
        pk = f"{r['peak_MB']:.1f}" if r["peak_MB"] is not None else "n/a"
        ge = f"{r['grad_rel_err_vs_eager']:.1e}" if r["grad_rel_err_vs_eager"] is not None else "-"
        print(f"| {r['model']} | {r['label']} | {pk} | {r['step_ms']:.0f} | {r['compile_s']:.0f} | {ge} |")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
