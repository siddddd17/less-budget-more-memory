"""seqlen_sweep_analyze.py -- summary table of results/seqlen_sweep_<device>.json (output of seqlen_sweep.py)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main_gpu(res):
    """Rows from --device cuda (one entry per compile)."""
    import collections
    g = collections.defaultdict(dict)
    for r in res:
        if not r.get("oom"):
            g[(r["S"], r["I"])][(r["budget"], r["remat_mode"])] = r
    print("| S | I | 2S/I | saved at 0.02 (attn/MLP) | best stock 0.02-0.30 (budget, MB) | stock 0.05 / best | stock 0.02 / best | "
          "best all-ops (budget, MB) | budget 1.0 (MB) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (S, I), v in sorted(g.items(), key=lambda kv: (2 * kv[0][0] / kv[0][1], kv[0][0])):
        st = {b: r for (b, m), r in v.items() if m is None and b <= 0.3}
        best = min(st.values(), key=lambda r: r["peak_MB"])
        al = min((r for (b, m), r in v.items() if m == "all"), key=lambda r: r["peak_MB"])
        print(f"| {S} | {I} | {2 * S / I:.2f} | {st[0.02]['attn_saved']}/{st[0.02]['mlp_saved']} | "
              f"{best['budget']:.2f}, {best['peak_MB']:.1f} | {st[0.05]['peak_MB'] / best['peak_MB']:.2f}x | "
              f"{st[0.02]['peak_MB'] / best['peak_MB']:.2f}x | {al['budget']:.2f}, {al['peak_MB']:.1f} | "
              f"{v[(1.0, None)]['peak_MB']:.1f} |")


def main(path):
    res = json.load(open(path))
    if res and "rows" not in res[0]:
        return main_gpu(res)
    print("| S | I | 2S/I | knapsack value/byte, attn:MLP | saved at 0.02 (attn/MLP) | saved at 0.05 (attn/MLP) | "
          "best stock (budget, MB) | stock 0.02 / best | stock 0.05 / best | all-ops at 0.02 (MB) | all-ops at 0.05 (MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(res, key=lambda r: (r["ratio_2S_over_I"], r["S"])):
        rows = {x["budget"]: x for x in r["rows"]}
        best = min(r["rows"], key=lambda x: x["peak_MB"])
        a, b = rows[0.02], rows[0.05]
        print(f"| {r['S']} | {r['I']} | {r['ratio_2S_over_I']:.2f} | {r['knapsack_value_ratio']:.2f} | "
              f"{a['attn_saved']}/{a['mlp_saved']} | {b['attn_saved']}/{b['mlp_saved']} | "
              f"{best['budget']:.2f}, {best['peak_MB']:.1f} | {a['peak_MB'] / best['peak_MB']:.2f}x | "
              f"{b['peak_MB'] / best['peak_MB']:.2f}x | {a['all_peak_MB']:.1f} | {b['all_peak_MB']:.1f} |")
    print("\nFull grid (stock / light / all-ops peak, MB):")
    for r in res:
        print(f"S={r['S']} I={r['I']} (2S/I={r['ratio_2S_over_I']:.2f}): " + "  ".join(
            f"{x['budget']:.2f}: {x['peak_MB']:.0f}/{x['light_peak_MB']:.0f}/{x['all_peak_MB']:.0f}" for x in r["rows"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "seqlen_sweep_cpu.json"))
