"""
check_rng_fix_v1_v2_equiv.py -- do the v1 and v2 RNG-fix emulations choose the same saved set under aot_eager?

The aot_eager BERT rows of the paper's Table 1 (results/gpu_eval_remat_bert_fixed.json) were measured with v1;
everything else uses v2. Under aot_eager, dropout stays aten.native_dropout, which OpTypes.is_random already bans, so
the extra v2 patch should not matter. This compiles BERT under aot_eager at budgets 0.05/0.10/0.15 with each
emulation (in separate processes) and compares the saved forward outputs and backward random-op counts.
Usage: python experiments/liveness/check_rng_fix_v1_v2_equiv.py [--scale 8] [--device cpu]
"""
import argparse, collections, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def child(mode, scale, device):
    sys.path.insert(0, HERE)
    import torch
    import torch._functorch.partitioners as P
    from torch._functorch import config as fc
    if mode == "v1":
        import rng_fix_emulation_v1 as E
    else:
        import rng_fix_emulation as E
    E.install()
    from torch._dynamo.backends.common import aot_autograd
    from torch._dynamo.backends.debugging import boxed_nop
    from ackaudit.capture import _resolve
    out = {}

    def part(j, ji, **kw):
        fw, bw = P.min_cut_rematerialization_partition(j, ji, **kw)
        outs = list(fw.graph.find_nodes(op="output"))[0].args[0]
        out["saved"] = sorted(collections.Counter(str(getattr(o, "target", "?")) for o in outs if hasattr(o, "target")).items())
        out["bw_random_ops"] = sum(1 for n in bw.graph.nodes if n.op == "call_function"
                                   and any(w in str(n.target) for w in ("dropout", "rand", "seed")))
        return fw, bw

    b, mk = _resolve("bert", scale)
    torch.manual_seed(0)
    m = b().to(device).train()
    be = aot_autograd(fw_compiler=boxed_nop, bw_compiler=boxed_nop, partition_fn=part)
    res = {}
    for bud in (0.05, 0.10, 0.15):
        torch._dynamo.reset(); fc.activation_memory_budget = bud
        torch.compile(m, backend=be, dynamic=False)(*[x.to(device) for x in mk()]).backward()
        res[str(bud)] = dict(out)
    fc.activation_memory_budget = 1.0
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=8); ap.add_argument("--device", default="cpu")
    ap.add_argument("--child", default=None)
    a = ap.parse_args()
    if a.child:
        child(a.child, a.scale, a.device); sys.exit(0)
    got = {}
    for mode in ("v1", "v2"):
        p = subprocess.run([sys.executable, __file__, "--child", mode, "--scale", str(a.scale), "--device", a.device],
                           capture_output=True, text=True, env=os.environ)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(p.stdout[-2000:], p.stderr[-4000:]); sys.exit(f"{mode} run failed")
        got[mode] = json.loads(line[0][7:])
    for bud in got["v1"]:
        same = got["v1"][bud] == got["v2"][bud]
        print(f"budget {bud}: saved outputs identical={same}  bw dropout/rand-named ops v1={got['v1'][bud]['bw_random_ops']} "
              f"v2={got['v2'][bud]['bw_random_ops']}")
    print("VERDICT:", "EQUIVALENT under aot_eager" if all(got["v1"][b] == got["v2"][b] for b in got["v1"]) else "DIFFERENT")
