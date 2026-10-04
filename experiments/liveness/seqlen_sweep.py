"""
seqlen_sweep.py -- controlled test of the selection-limitation regime (paper, section 3.4).

The knapsack values one attention output element at about 4S FLOPs and one down_proj output element at 2I, so attention
outputs are worth 2S/I times as much per byte. The paper predicts that the knapsack spends small budgets on attention
outputs (and the stock curve is non-monotone below the best budget) only when 2S/I > 1. This script varies the ratio two
ways on the paper's Llama configuration (32 layers, hidden 256, 8 heads, batch 2):

  S sweep  : S in {256, 512, 1024, 2048} at I = 688        (2S/I = 0.74, 1.5, 3.0, 6.0)
  I sweep  : I in {344, 688, 1376, 2752} at S = 1024       (2S/I = 6.0, 3.0, 1.5, 0.74)

The two sweeps cross the same ratios with different absolute sizes, so a regime effect should follow the ratio, not S or I.

--device cpu  (default): aot_eager on CPU, one compile per (S, I); every budget is then evaluated inside that compile
               with the production knapsack (dp_knapsack, capacity = budget, as in choose_saved_values_set) and scored by
               the exact static oracle used in the paper's CPU experiments (fx walk of the emitted fw/bw graphs, exact under
               aot_eager). Same testbed patches as greedy_selector_race.py (CPU SDPA treated like CUDA SDPA).
--device cuda : Inductor on the GPU, one compile per (S, I, budget), measured max_memory_allocated (same harness as
               gpu_eval_remat.py). Llama has no random ops, so the #190758/#198333 issues do not apply.

Usage (repo root):
  python experiments/liveness/seqlen_sweep.py --device cpu --out experiments/liveness/results/seqlen_sweep_cpu.json
  TORCHINDUCTOR_COMPILE_THREADS=1 python experiments/liveness/seqlen_sweep.py --device cuda \\
      --out experiments/liveness/results/seqlen_sweep_gpu.json                         (~2 h on the GTX 1650)
"""
from __future__ import annotations

import argparse, collections, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(os.path.dirname(HERE)))

import torch
from torch import nn

L, HIDDEN, HEADS, BATCH, VOCAB = 32, 256, 8, 2, 1000
S_SWEEP = [(256, 688), (512, 688), (1024, 688), (2048, 688)]
I_SWEEP = [(1024, 344), (1024, 1376), (1024, 2752)]          # (1024, 688) is shared with the S sweep
BUDGETS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
GPU_BUDGETS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]


class LlamaSI(nn.Module):
    """The paper's Llama (ackaudit LlamaWrap at scale 8) with the MLP size I as a parameter."""

    def __init__(self, inter):
        super().__init__()
        from transformers import LlamaConfig, LlamaModel
        self.model = LlamaModel(LlamaConfig(hidden_size=HIDDEN, intermediate_size=inter, num_hidden_layers=L,
                                            num_attention_heads=HEADS, num_key_value_heads=HEADS, vocab_size=VOCAB,
                                            use_cache=False))

    def forward(self, ids):
        return self.model(input_ids=ids).last_hidden_state.sum()


def spec(S, I):
    return (lambda: LlamaSI(I)), (lambda: (torch.randint(0, VOCAB, (BATCH, S)),))


def is_attn(n):
    return "scaled_dot_product" in str(getattr(n, "target", ""))


# ------------------------------------------------------------------------------------------------ CPU path
def run_cpu(a):
    import greedy_selector_race as R            # installs the CPU testbed patches and the lazy oracle
    import liveness_oracle_check as A
    import torch._functorch.partitioners as P
    from torch._functorch import config as fc
    from torch._functorch.partitioners import CustomKnapsackSolver
    from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
    from torch._dynamo.backends.common import aot_autograd
    from torch._dynamo.backends.debugging import boxed_nop
    LS = R.LS
    R.register_cpu_sdpa_flops()
    MB = 1e6

    class Sweep(CustomKnapsackSolver):
        def __init__(s, S, I):
            s.S, s.I, s.out = S, I, None

        def uuid(s):
            return None

        def __call__(s, memory, jg, max_memory, ni, cands):
            if s.out is not None:
                return s._stock(memory, max_memory, cands)
            cands = list(cands); n = len(cands)
            rt = [float(LS._RT.get(c, 1.0)) for c in cands]
            attn = {i for i, c in enumerate(cands) if is_attn(c)}
            mlp = {i for i, c in enumerate(cands) if A.is_mlp_output(c)}
            oracle = R.XOracle(jg, ni, cands)
            # value per normalized byte, attention vs MLP output (the 2S/I ratio as the knapsack sees it)
            vpb = lambda idx: sorted(rt[i] / memory[i] for i in idx if memory[i] > 0)
            med = lambda v: v[len(v) // 2] if v else float("nan")
            out = dict(S=s.S, I=s.I, ratio_2S_over_I=2 * s.S / s.I, n=n, n_attn=len(attn), n_mlp=len(mlp),
                       cand_ops=dict(collections.Counter(LS._op(c) for c in cands)),
                       value_per_byte_attn=med(vpb(attn)), value_per_byte_mlp=med(vpb(mlp)), rows=[])
            out["knapsack_value_ratio"] = out["value_per_byte_attn"] / out["value_per_byte_mlp"]
            allb = oracle(frozenset(range(n)))
            out["all_banned_peak_MB"] = allb["peak"] / MB
            for b in a.budgets:
                t = time.perf_counter()
                sv = frozenset(dp_knapsack(list(memory), list(rt), b)[1])
                o = oracle(sv)
                row = dict(budget=b, peak_MB=o["peak"] / MB, R_GF=o["R"] / 1e9, n_saved=len(sv),
                           attn_saved=len(sv & attn), mlp_saved=len(sv & mlp),
                           attn_saved_frac_of_weight=sum(memory[i] for i in sv & attn) / max(1e-12, sum(memory[i] for i in sv)))
                for tag, mode in (("light", "pointwise"), ("all", "all")):
                    r = R.with_remat(oracle, sv, mode)
                    row[f"{tag}_peak_MB"], row[f"{tag}_R_GF"] = r["peak"] / MB, r["R"] / 1e9
                row["seconds"] = time.perf_counter() - t
                out["rows"].append(row)
                print(f"  S={s.S:5d} I={s.I:5d} b={b:.2f}  stock {row['peak_MB']:8.1f} MB (attn {row['attn_saved']:2d}, "
                      f"mlp {row['mlp_saved']:2d} saved)  light {row['light_peak_MB']:8.1f}  all {row['all_peak_MB']:8.1f}  "
                      f"{row['seconds']:.0f}s", flush=True)
            s.out = out
            return s._stock(memory, max_memory, cands)

        def _stock(s, memory, max_memory, cands):
            rt = [float(LS._RT.get(c, 1.0)) for c in cands]
            sv = dp_knapsack(list(memory), rt, max_memory)[1]
            return sorted(sv), [i for i in range(len(cands)) if i not in set(sv)]

    res = []
    for S, I in a.configs:
        torch.manual_seed(197838); torch._dynamo.reset()
        build, mk = spec(S, I)
        m, args = build(), mk()
        be = aot_autograd(fw_compiler=boxed_nop, bw_compiler=boxed_nop, partition_fn=P.min_cut_rematerialization_partition,
                          keep_inference_input_mutations=True)
        sol = Sweep(S, I)
        prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver)
        t0 = time.perf_counter()
        print(f"[S={S} I={I} 2S/I={2 * S / I:.2f}]", flush=True)
        try:
            fc.activation_memory_budget, fc.activation_memory_budget_solver = min(a.budgets), sol
            torch.compile(m, backend=be, dynamic=False)(*args).backward()
        finally:
            fc.activation_memory_budget, fc.activation_memory_budget_solver = prev
        if sol.out is None:
            raise RuntimeError("the knapsack was never reached (budget above the early-return thresholds?)")
        sol.out["seconds"] = time.perf_counter() - t0
        res.append(sol.out)
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        del m, args
    return res


# ------------------------------------------------------------------------------------------------ GPU path
def run_cuda(a):
    import gpu_eval_remat as G
    import liveness_oracle_check as A
    import torch._inductor.config as IC
    IC.force_disable_caches = True
    orig = A.resolve

    def resolve(model, scale):
        if model.startswith("llamaSI:"):
            _, S, I = model.split(":"); return spec(int(S), int(I))
        return orig(model, scale)
    A.resolve = resolve
    res = []
    for S, I in a.configs:
        for b in a.budgets:
            for remat in ([None] + (["pointwise", "all"] if b in a.remat_budgets else [])):
                print(f"[S={S} I={I} 2S/I={2 * S / I:.2f} b={b:.2f} remat={remat}]", flush=True)
                sol = A.PlanSolver()
                try:
                    r, _ = G.run(f"llamaSI:{S}:{I}", b, "inductor", sol, remat, a.repeats, a.iters)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); print("   OOM", flush=True)
                    res.append(dict(S=S, I=I, budget=b, remat=remat, oom=True)); continue
                rec = sol.rec
                r.update(S=S, I=I, ratio_2S_over_I=2 * S / I, remat_mode=remat,
                         attn_saved=sum(1 for i in rec.get("saved", []) if is_attn(rec["cands"][i])) if rec else None,
                         mlp_saved=rec.get("mlp_saved") if rec else None, n_candidates=rec.get("n_candidates") if rec else None,
                         reached_knapsack=bool(rec))
                res.append(r)
                print(f"   peak={r['peak_MB']:.1f}MB step={r['step_ms']:.1f}ms attn_saved={r['attn_saved']} "
                      f"mlp_saved={r['mlp_saved']}", flush=True)
                json.dump(res, open(a.out, "w"), indent=1, default=str)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--sweep", choices=["both", "S", "I"], default="both")
    ap.add_argument("--budgets", type=float, nargs="+", default=None)
    ap.add_argument("--remat-budgets", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.budgets = a.budgets or (BUDGETS if a.device == "cpu" else GPU_BUDGETS)
    a.configs = (S_SWEEP if a.sweep in ("both", "S") else [(1024, 688)]) + (I_SWEEP if a.sweep in ("both", "I") else [])
    a.out = a.out or os.path.join(HERE, "results", f"seqlen_sweep_{a.device}.json")
    res = run_cpu(a) if a.device == "cpu" else run_cuda(a)
    print(f"wrote {a.out} ({len(res)} entries)")


if __name__ == "__main__":
    main()
