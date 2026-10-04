"""
experiment_c_inductor_runtime.py -- PyTorch #197838, follow-ups to Experiments A and B (LOCAL ONLY).

Two parts, both measurement-only. No new selection logic.

--part oracle   (Experiment A under Inductor)
    Same plans as Experiment A, compiled with backend="inductor". For each plan:
      measured   : CUDA max_memory_allocated() - resident
      fx_walk    : the Experiment-A liveness walk over the partitioner's fw/bw
                   GraphModules, taken at partition time (before Inductor's
                   post-grad passes mutate them)
      inductor   : Inductor's own peak estimate for the order its scheduler chose
                   (torch/_inductor/memory.py, reorder_for_peak_memory)
    Question: does either predictor still RANK plans correctly once Inductor
    fuses, reorders and plans memory itself?

--part runtime  (cost of the Experiment-B plans, in real milliseconds)
    For stock and each Experiment-B chosen plan, under aot_eager AND inductor:
      measured peak, step time (CUDA events, warmup + N timed iterations,
      median and p10/p90), stock measured again at the end to quantify drift,
      and three estimates of the recompute cost the plan adds:
        flops    : the partitioner's default estimator (what the knapsack uses)
        profile  : upstream activation_memory_budget_runtime_estimator="profile"
                   (best effort; reported as NaN if it fails)
        real_op  : each candidate op timed on real CUDA tensors of its shape
    Plans are chosen by Experiment B's selection run *natively* for each backend.
    Inductor's joint-graph passes change the graph (silu is decomposed, and the
    candidate set goes from 324 to 322), so plans are not copied across backends.
    Policies are stock, retain>=0.90 and unconstrained. Duplicate plans are skipped.
    The knapsack always plans with flops; profile and real_op are only measured
    alongside it.

PRE-REGISTERED
  oracle part (each predictor judged separately):
    RANK PASS   : Spearman(pred, measured) >= 0.90 over non-BERT plans AND every
                  available contrast pair ordered as measured.
    Error is reported but not pass/fail: fusion removes intermediates, so an
    fx-level walk is expected to overestimate under Inductor.
  runtime part:
    C1 efficacy : under Inductor, some non-stock policy lowers Llama-0.05 measured
                  peak vs Inductor stock by >= 10%.
    C2 safety   : no B plan raises Inductor measured peak by > 1% vs Inductor stock.
    C3 cost     : reported as a measured step-time delta (ms and %) per backend,
                  together with the stock-vs-stock drift. Flagged if > 10%.
    C4 calibration (exploratory): for each estimator, compare the predicted
                  recompute-time delta with the measured step-time delta.

Usage, from the ackaudit root in .venv-cuda, with liveness_oracle_check.py and
experiment_b_liveness_select.py in the same directory as this file:
  python /tmp/experiment_c_inductor_runtime.py --part oracle  --suite selftest   # CPU plumbing (needs gcc)
  python /tmp/experiment_c_inductor_runtime.py --part runtime --suite selftest   # CPU plumbing
  python /tmp/experiment_c_inductor_runtime.py --part oracle  --suite full --repeats 3
  python /tmp/experiment_c_inductor_runtime.py --part runtime --suite quick --iters 20
Nothing is committed or pushed. Output is written only to --out.
"""
from __future__ import annotations

import argparse, collections, json, math, os, statistics, sys, time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import liveness_oracle_check as A          # models, static walk, PlanSolver, estimate_runtime recorder
import experiment_b_liveness_select as B   # structural signatures, evenly_spaced

import torch
import torch.fx as fx
import torch._functorch.partitioners as P
import torch._inductor.compile_fx as CF
import torch._inductor.memory as MEM
from torch._functorch import config as fc
from torch._functorch.partitioners import CustomKnapsackSolver
from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
from torch._inductor import config as ic
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.utils import _pytree as pytree

ic.force_disable_caches = True  # never reuse a cached partition/compile across plans

# ----------------------------------------------------------------------------
# capture hooks
# ----------------------------------------------------------------------------
CAP: dict[str, Any] = {}
_orig_inductor_partition = CF.min_cut_rematerialization_partition


def _capturing_partition(*a, **k):
    B._CTX.clear(); B._CTX.update(k); B._CTX["joint_module"] = a[0]
    fw, bw = _orig_inductor_partition(*a, **k)
    # Inductor's post-grad passes mutate these modules later, so walk them now.
    CAP["static"] = A.static_step_peak(fw, bw)
    return fw, bw


CF.min_cut_rematerialization_partition = _capturing_partition

INDUCTOR_EST: dict[str, float] = {}
_orig_reorder = MEM.reorder_for_peak_memory


def _recording_reorder(*a, **k):
    seen: list[int] = []
    orig_est = MEM.estimate_peak_memory

    def est(*x, **y):
        r = orig_est(*x, **y)
        seen.append(int(r[0]))
        return r

    MEM.estimate_peak_memory = est
    try:
        return _orig_reorder(*a, **k)
    finally:
        MEM.estimate_peak_memory = orig_est
        if seen:
            tag = "bw" if getattr(V.graph, "is_backward", False) else "fw"
            INDUCTOR_EST[tag] = min(seen) / 1e6  # the order Inductor keeps is the min over methods


MEM.reorder_for_peak_memory = _recording_reorder

# per-candidate cost estimates, recorded alongside the flops value the knapsack uses
PROFILE_MS: dict = {}
REALOP_MS: dict = {}
COST_PROBES = {"on": False}
_flops_recorder = P.estimate_runtime  # A's recorder (flops, stored in A._RT)


def _time_fn(fn, device) -> float:
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / 10
    t0 = time.perf_counter()
    for _ in range(10):
        fn()
    return (time.perf_counter() - t0) * 100


def _real_op_ms(node) -> float:
    try:
        with unset_fake_temporarily(), torch.no_grad():
            dev = [None]

            def mk(x):
                if isinstance(x, fx.Node):
                    v = x.meta.get("val")
                    if isinstance(v, torch.Tensor):
                        t = torch.empty_strided([int(s) for s in v.shape], [int(s) for s in v.stride()],
                                                dtype=v.dtype, device=v.device)
                        dev[0] = t.device
                        if t.is_floating_point():
                            t.normal_()
                        else:
                            t.zero_()
                        return t
                    return int(v) if isinstance(v, torch.SymInt) else v
                return x

            args, kwargs = pytree.tree_map(mk, (node.args, node.kwargs))
            return _time_fn(lambda: node.target(*args, **kwargs), dev[0] or torch.device("cpu"))
    except Exception:
        return float("nan")


def _cost_recording_estimate_runtime(node):
    v = _flops_recorder(node)  # planning value (flops unless the user changed config)
    if COST_PROBES["on"] and node not in PROFILE_MS:
        prev = fc.activation_memory_budget_runtime_estimator
        try:
            fc.activation_memory_budget_runtime_estimator = "profile"
            PROFILE_MS[node] = float(A._orig_estimate_runtime(node))
        except Exception:
            PROFILE_MS[node] = float("nan")
        finally:
            fc.activation_memory_budget_runtime_estimator = prev
        REALOP_MS[node] = _real_op_ms(node)
    return v


P.estimate_runtime = _cost_recording_estimate_runtime


def _generic_mlp_output(n) -> bool:
    """mm whose (view-stripped) input is a mul of two *computed* tensors.
    Matches down_proj(act(gate) * up) whether silu is an op (aot_eager) or
    decomposed (Inductor); excludes q/k/v/gate/up (input is norm-weight * x,
    one operand a parameter) and o_proj (input is attention output)."""
    if A._op(n) != "mm" or not n.args:
        return False
    x = B._strip(n.args[0])
    if x is None or A._op(x) != "mul":
        return False
    ins = [B._strip(i) for i in x.all_input_nodes]
    return len(ins) == 2 and all(i is not None and i.op == "call_function" for i in ins)


A.is_mlp_output = _generic_mlp_output  # used by A.PlanSolver for forced-in / forced-out plans


# ----------------------------------------------------------------------------
# plan definition by structural class (transferable across backends)
# ----------------------------------------------------------------------------
def class_name(k, v) -> str:  # identical format to Experiment B
    ins = ", ".join(t if not g else f"{t}({'+'.join(g)})" for t, g in k[2] if t != "input")
    return f"{k[0]}{list(k[1]) if k[1] else ''}<-[{ins}] x{len(v)}"


def _strip_count(name: str) -> str:
    return name.rsplit(" x", 1)[0]


class ClassForcedSolver(CustomKnapsackSolver):
    """forced = [(class_name, fraction), ...]; forced members + production dp_knapsack fill."""

    def __init__(self, forced: list[tuple[str, float]]):
        self.forced, self.rec = list(forced), {}

    def __call__(self, memory, joint_graph, max_memory, node_info, cands):
        cands = list(cands)
        rt = [A._RT[c] for c in cands]
        classes = collections.defaultdict(list)
        for i, c in enumerate(cands):
            classes[B.structural_signature(c)].append(i)
        by_name = {}
        for k, v in classes.items():
            if len(v) >= B.MIN_CLASS_SIZE:
                v = sorted(v, key=lambda i: node_info.get_fw_order(cands[i]))
                by_name[class_name(k, v)] = v
                by_name.setdefault(_strip_count(class_name(k, v)), v)
        forced, missing = set(), []
        for name, f in self.forced:
            members = by_name.get(name) or by_name.get(_strip_count(name))
            if members is None:
                missing.append(name); continue
            forced |= set(B.evenly_spaced(members, max(1, round(f * len(members)))))
        if missing:
            raise RuntimeError(f"structural class not found in this graph: {missing}; "
                               f"available: {sorted(k for k in by_name if ' x' in k)}")
        w = sum(memory[i] for i in forced)
        if w > max_memory + B.TOL:
            raise RuntimeError("forced set exceeds budget")
        rest = [i for i in range(len(cands)) if i not in forced]
        _, sv, _ = dp_knapsack([memory[i] for i in rest], [rt[i] for i in rest], max(max_memory - w, 0.0))
        saved = sorted(forced | {rest[i] for i in sv})
        recomp = [i for i in range(len(cands)) if i not in set(saved)]

        def tot(d):
            vals = [d.get(cands[i], float("nan")) for i in recomp]
            return float(sum(vals)) if vals else 0.0

        self.rec.update(
            n_candidates=len(cands), weight=float(sum(memory[i] for i in saved)), max_memory=float(max_memory),
            saved_ops=dict(collections.Counter(A._op(cands[i]) for i in saved)),
            est_recompute_flops=float(sum(rt[i] for i in recomp)),
            est_recompute_profile_ms=tot(PROFILE_MS), est_recompute_realop_ms=tot(REALOP_MS),
            realop_nan=sum(1 for i in recomp if math.isnan(REALOP_MS.get(cands[i], float("nan")))),
        )
        return saved, recomp

    def uuid(self):
        return None


# ----------------------------------------------------------------------------
# one compile + measurement
# ----------------------------------------------------------------------------
def run(label, model, scale, budget, backend, solver, device, repeats, iters=0, warmup=3, no_reorder=False):
    torch.manual_seed(197838)
    torch._dynamo.reset()
    CAP.clear(); INDUCTOR_EST.clear()
    build, make_inputs = A.resolve(model, scale)
    m = build().to(device)
    args = tuple(x.to(device) for x in make_inputs())
    if backend == "aot_eager":
        # identical to torch._dynamo.backends.debugging.aot_eager, with the same capture as Inductor's path
        from torch._dynamo.backends.common import aot_autograd
        from torch._dynamo.backends.debugging import boxed_nop

        def part(joint_module, joint_inputs, **kw):
            B._CTX.clear(); B._CTX.update(kw); B._CTX["joint_module"] = joint_module
            fw, bw = P.min_cut_rematerialization_partition(joint_module, joint_inputs, **kw)
            CAP["static"] = A.static_step_peak(fw, bw)
            return fw, bw

        compiled_backend = aot_autograd(fw_compiler=boxed_nop, bw_compiler=boxed_nop, partition_fn=part,
                                        keep_inference_input_mutations=True)
    else:
        compiled_backend = backend
    cm = torch.compile(m, backend=compiled_backend, dynamic=False)
    prev = (fc.activation_memory_budget, fc.activation_memory_budget_solver, P.reordering_to_mimic_autograd_engine)
    t0 = time.perf_counter()
    try:
        fc.activation_memory_budget = budget
        fc.activation_memory_budget_solver = solver
        if no_reorder:
            P.reordering_to_mimic_autograd_engine = lambda gm: gm
        cm(*args).backward()  # compiles fw and (lazily) bw inside the patched config
    finally:
        fc.activation_memory_budget, fc.activation_memory_budget_solver, P.reordering_to_mimic_autograd_engine = prev
    row: dict[str, Any] = dict(plan=label, model=model, budget=budget, backend=backend, device=device,
                               compile_s=time.perf_counter() - t0)
    row.update({k: v for k, v in solver.rec.items() if k not in ("cands", "memory", "rt", "saved", "recomp", "joint_graph")})
    if "static" in CAP:
        row["fx_walk_MB"] = CAP["static"]["static_peak_MB"]
        row["fx_walk_bw_newalloc_MB"] = CAP["static"]["static_bw_newalloc_MB"]
    row["inductor_est_fw_MB"] = INDUCTOR_EST.get("fw")
    row["inductor_est_bw_MB"] = INDUCTOR_EST.get("bw")
    if row["inductor_est_bw_MB"] is not None:
        row["inductor_est_MB"] = max(INDUCTOR_EST.get("fw") or 0.0, row["inductor_est_bw_MB"])

    m.zero_grad(set_to_none=False)
    peaks, ms = [], []
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        resident = torch.cuda.memory_allocated()
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            cm(*args).backward(); torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - resident)
            m.zero_grad(set_to_none=False)
        for _ in range(warmup if iters else 0):
            cm(*args).backward(); m.zero_grad(set_to_none=False)
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); cm(*args).backward(); e.record(); torch.cuda.synchronize()
            ms.append(s.elapsed_time(e)); m.zero_grad(set_to_none=False)
        row["measured_peak_MB"] = statistics.median(peaks) / 1e6
    else:
        for _ in range(repeats):
            loss = cm(*args); mode = A._LiveBytes()
            with mode:
                loss.backward()
            peaks.append(mode.peak); del loss; m.zero_grad(set_to_none=False)
        for _ in range(iters):
            t = time.perf_counter(); cm(*args).backward(); ms.append((time.perf_counter() - t) * 1e3)
            m.zero_grad(set_to_none=False)
        # CPU self-test only: the dispatch-mode tracker does not see Inductor's own allocations
        row["measured_peak_MB"] = statistics.median(peaks) / 1e6
    if ms:
        q = statistics.quantiles(ms, n=10) if len(ms) >= 10 else [min(ms)] * 9
        row.update(step_ms=statistics.median(ms), step_p10=q[0], step_p90=q[-1])
    del m, cm, args
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


# ----------------------------------------------------------------------------
# part 1: Experiment A under Inductor
# ----------------------------------------------------------------------------
def oracle_plans(suite):
    if suite == "selftest":
        return [(f"toy nat {b:.2f}", "toy", 8, b, A.PlanSolver(), False) for b in (0.05, 0.15)] + \
               [("toy 0.15 force-out MLP", "toy", 8, 0.15, A.PlanSolver(force_out=True), False)]
    out = []
    for label, model, scale, b, solver, noreorder in A.plans("full"):
        if "over budget" in label:
            label = "llama 0.05 MLP-only (over budget, DP at capacity 0)"
        out.append((label, model, scale, b, solver, noreorder))
    return out


def part_oracle(a, device):
    rows = []
    for label, model, scale, b, solver, noreorder in oracle_plans(a.suite):
        if a.models and model not in a.models:
            continue
        print(f"[{label}]", flush=True)
        r = run(label, model, scale, b, "inductor", solver, device, a.repeats, no_reorder=noreorder)
        rows.append(r)
        print(f"   measured={r['measured_peak_MB']:.1f}MB  fx_walk={r.get('fx_walk_MB', float('nan')):.1f}MB  "
              f"inductor_est={r.get('inductor_est_MB') or float('nan'):.1f}MB (fw {r['inductor_est_fw_MB']}, bw {r['inductor_est_bw_MB']})  "
              f"saved_w={r.get('saved_weight', float('nan')):.4f}  mlp_saved={r.get('mlp_saved')}  compile={r['compile_s']:.0f}s", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1, default=str)

    if device == "cpu":
        print("\nNOTE: CPU self-test. The CPU tracker cannot see Inductor's own allocations, so 'measured' "
              "is meaningless and the RANK verdicts below only exercise the plumbing.")
    by = {r["plan"]: r for r in rows}
    primary = [r for r in rows if r["model"] != "bert"]
    if not primary:
        print("no non-BERT plans in this run; skipping rank verdicts")
        primary = rows
    meas = [r["measured_peak_MB"] for r in primary]
    contrasts = [(hi if "over budget" not in hi else "llama 0.05 MLP-only (over budget, DP at capacity 0)",
                  lo if "over budget" not in lo else "llama 0.05 MLP-only (over budget, DP at capacity 0)")
                 for hi, lo in A.CONTRASTS]
    print("\n" + "=" * 90)
    for key in ("fx_walk_MB", "inductor_est_MB"):
        vals = [r.get(key) for r in primary]
        if any(v is None for v in vals):
            print(f"{key}: missing for some plans; cannot judge"); continue
        rho = A.spearman(vals, meas)
        miss = []
        for hi, lo in contrasts:
            if hi in by and lo in by and by[hi].get(key) is not None and by[lo].get(key) is not None:
                if (by[hi][key] > by[lo][key]) != (by[hi]["measured_peak_MB"] > by[lo]["measured_peak_MB"]):
                    miss.append(f"{hi} vs {lo}")
        err = statistics.median(abs(r[key] - r["measured_peak_MB"]) / r["measured_peak_MB"] for r in primary)
        verdict = "PASS" if rho >= 0.90 and not miss else "FAIL"
        print(f"{key:16s} Spearman={rho:+.3f}  median rel err={err:.1%}  contrast misses={len(miss)}  RANK {verdict}")
        for x in miss:
            print(f"      miss: {x}")
    b = [r for r in rows if r["model"] == "bert"]
    if b:
        print("BERT (measured, fx_walk, inductor_est):",
              [(r["budget"], round(r["measured_peak_MB"], 1), round(r.get("fx_walk_MB") or 0, 1),
                round(r.get("inductor_est_MB") or 0, 1)) for r in b])
    print(f"wrote {a.out}; no repository files modified.")


# ----------------------------------------------------------------------------
# part 2: runtime cost of Experiment-B plans
# ----------------------------------------------------------------------------
class CostedExpBSolver(B.ExpBSolver):
    """Experiment-B selection (native to the backend being compiled) + recompute-cost estimates."""

    def __call__(self, memory, joint_graph, max_memory, node_info, cands):
        saved, recomp = super().__call__(memory, joint_graph, max_memory, node_info, cands)
        cands = list(cands)

        def tot(d):
            return float(sum(d.get(cands[i], float("nan")) for i in recomp))

        self.rec.update(
            est_recompute_flops=float(sum(A._RT[cands[i]] for i in recomp)),
            est_recompute_profile_ms=tot(PROFILE_MS), est_recompute_realop_ms=tot(REALOP_MS),
            realop_nan=sum(1 for i in recomp if math.isnan(REALOP_MS.get(cands[i], float("nan")))),
            saved_idx=sorted(saved),
        )
        return saved, recomp


RUNTIME_POLICIES = (("stock", None), ("retain>=0.90", 0.90), ("unconstrained", 0.0))


def runtime_cases(suite):
    if suite == "selftest":
        return [("toy16", 16, 0.15)]
    if suite == "quick":
        return [("llama", 8, 0.05), ("llama", 8, 0.15), ("bert", 8, 0.05)]
    return [("llama", 8, b) for b in (0.05, 0.10, 0.15, 0.20, 0.30)] + [("bert", 8, b) for b in (0.05, 0.10, 0.15)]


def part_runtime(a, device):
    COST_PROBES["on"] = True
    backends = ["aot_eager", "inductor"]
    rows = []
    for model, scale, budget in runtime_cases(a.suite):
        for be in backends:
            B._FAMILY_CACHE.clear()
            chosen_seen: dict[tuple, str] = {}
            seq = list(RUNTIME_POLICIES) + [("stock (again)", None)]
            for pname, floor in seq:
                if B._FAMILY_CACHE and floor is not None:
                    fam = next(iter(B._FAMILY_CACHE.values()))
                    pre = tuple(sorted(B.select(fam, floor)))
                    if pre in chosen_seen:
                        print(f"[{model} {budget:.2f} {be} | {pname}] same plan as '{chosen_seen[pre]}', skipped", flush=True)
                        continue
                print(f"[{model} {budget:.2f} {be} | {pname}]", flush=True)
                PROFILE_MS.clear(); REALOP_MS.clear()
                solver = CostedExpBSolver(floor)
                r = run(pname, model, scale, budget, be, solver, device, a.repeats, iters=a.iters)
                r["policy"] = pname
                if "fx_walk_MB" in r:  # oracle vs the module actually handed to the backend
                    r["invariant_ok"] = abs(r["fx_walk_MB"] - r["predicted_peak_MB"]) <= 1e-3 * max(1.0, r["fx_walk_MB"])
                chosen_seen.setdefault(tuple(r["saved_idx"]), pname)
                r.pop("family", None)
                rows.append(r)
                print(f"   plan={'stock' if r['chosen_is_stock'] else '; '.join(r['forced'])}\n"
                      f"   peak={r['measured_peak_MB']:.1f}MB (oracle {r['predicted_peak_MB']:.1f}, inductor_est {r.get('inductor_est_MB')})  "
                      f"step={r.get('step_ms', float('nan')):.1f}ms [p10 {r.get('step_p10', float('nan')):.1f}, p90 {r.get('step_p90', float('nan')):.1f}]  "
                      f"retained={r['runtime_retained']:.3f}  est_recompute: flops={r['est_recompute_flops']:.3e} "
                      f"profile={r['est_recompute_profile_ms']:.2f}ms real_op={r['est_recompute_realop_ms']:.2f}ms  "
                      f"w={r['weight']:.4f}/{r['max_memory']:.2f}  oracle={r['oracle_calls']} calls/{r['oracle_seconds']:.0f}s", flush=True)
                json.dump(rows, open(a.out, "w"), indent=1, default=str)

    print("\n" + "=" * 124)
    print(f"{'model':6} {'bud':>5} {'backend':9} {'policy':14} {'peak MB':>8} {'Δpeak':>7} {'step ms':>8} {'Δstep':>7} "
          f"{'Δstep ms':>8} {'retain':>6} {'Δprof ms':>9} {'Δreal ms':>9}  plan")
    c1, c2, h5, notes = None, True, True, []
    for model, scale, budget in runtime_cases(a.suite):
        for be in backends:
            grp = [r for r in rows if r["model"] == model and r["budget"] == budget and r["backend"] == be]
            if not grp:
                continue
            s0, s1 = grp[0], grp[-1]
            ref = (s0["step_ms"] + s1["step_ms"]) / 2
            drift = (s1["step_ms"] - s0["step_ms"]) / s0["step_ms"]
            for r in grp:
                dp = (r["measured_peak_MB"] - s0["measured_peak_MB"]) / s0["measured_peak_MB"]
                dsm = r["step_ms"] - ref
                dpr = r["est_recompute_profile_ms"] - s0["est_recompute_profile_ms"]
                dre = r["est_recompute_realop_ms"] - s0["est_recompute_realop_ms"]
                print(f"{model:6} {budget:5.2f} {be:9} {r['policy']:14} {r['measured_peak_MB']:8.1f} {dp:+7.1%} "
                      f"{r['step_ms']:8.1f} {dsm / ref:+7.1%} {dsm:+8.1f} {r['runtime_retained']:6.3f} {dpr:+9.2f} {dre:+9.2f}  "
                      f"{'stock' if r['chosen_is_stock'] else '; '.join(r['forced'])[:70]}")
                if r.get("invariant_ok") is False:
                    h5 = False; notes.append(f"invariant mismatch: {model} {budget} {be} {r['policy']}")
                if r["policy"].startswith("stock"):
                    continue
                if be == "inductor" and dp > 0.01:
                    c2 = False; notes.append(f"C2 violated: {model} {budget} {r['policy']} {dp:+.1%}")
                if dsm / ref > 0.10:
                    notes.append(f"C3 flag: {model} {budget} {be} {r['policy']} step {dsm / ref:+.1%}")
                if be == "inductor" and model == "llama" and budget == 0.05:
                    c1 = (c1 or False) or dp <= -0.10
            print(f"{'':6} {'':5} {be:9} {'stock drift':14} {'':8} {'':7} {'':8} {drift:+7.1%}")
    print()
    for n in notes:
        print("  ", n)
    if a.suite != "selftest":
        print(f"C1 efficacy under Inductor (Llama 0.05, >=10% lower peak): {'n/a' if c1 is None else ('PASS' if c1 else 'FAIL')}")
    print(f"C2 safety under Inductor (no plan >1% above stock): {'PASS' if c2 else 'FAIL'}")
    print(f"Oracle == module handed to Inductor: {'PASS' if h5 else 'FAIL'}")
    print("C3/C4: see table. Δstep ms is measured; Δprof/Δreal are estimated recompute ms added vs stock.")
    print(f"wrote {a.out}; no repository files modified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["oracle", "runtime"], required=True)
    ap.add_argument("--suite", choices=["selftest", "quick", "full"], default="quick")
    ap.add_argument("--repeats", type=int, default=3, help="peak-memory repeats")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations (runtime part)")
    ap.add_argument("--models", nargs="+", default=None, help="oracle part: only these models (e.g. bert)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = "cpu" if a.suite == "selftest" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    a.out = a.out or f"/tmp/expc_{a.part}_{a.suite}.json"
    if a.part == "oracle":
        part_oracle(a, device)
    else:
        if a.suite == "selftest":
            a.iters = min(a.iters, 3)
        part_runtime(a, device)


if __name__ == "__main__":
    main()
