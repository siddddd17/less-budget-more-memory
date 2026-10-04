"""greedy_race_analyze.py -- tables from greedy_race_<model>.json (output of greedy_selector_race.py) (MB = 1e6 bytes, GF = 1e9 FLOPs of forward ops re-executed in backward)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
MB, GF = 1e6, 1e9


def g_key(d):
    for k in ("greedy_exact", "greedy_lazy"):
        if k in d:
            return k


def first_reach(traj, T):
    for t in traj:
        if t["peak"] <= T:
            return t
    return None


def report(path):
    d = json.load(open(path)); model = "bert" if "bert" in os.path.basename(path) else "toy"
    k = g_key(d); g = d[k]; traj = g["traj"]; fin = traj[-1]
    print(f"\n## {model}: n={d['n']} candidates {d['cand_ops']}")
    print(f"all-banned (aggressive min-cut, greedy start): peak {d['all_banned']['peak']/MB:.2f} MB, R 0")
    print(f"greedy ({k}): {len(traj)-1} drops, converged={g['converged']}, peak {fin['peak']/MB:.2f} MB, R {fin['R']/GF:.3f} GF, "
          f"oracle calls {g['calls']} ({g.get('full_evals')} full sweeps), {fin['seconds']:.0f}s wall "
          f"({g.get('workers', 1)} worker(s)); +light {g['light']['peak']/MB:.2f} MB R {g['light']['R']/GF:.3f}; "
          f"+all {g['all']['peak']/MB:.2f} MB R {g['all']['R']/GF:.3f}")
    print("\n| budget | method | peak MB | vs stock | recompute GF | oracle calls | plan s |")
    print("|---|---|---|---|---|---|---|")
    for r in d["rows"]:
        sp = r["stock"]["peak"]
        def row(name, x, calls=None, sec=None):
            c = x.get("calls", "") if calls is None else calls
            s = x.get("seconds", float("nan")) if sec is None else sec
            print(f"| {r['budget']:.2f} | {name} | {x['peak']/MB:.2f} | {100*(x['peak']-sp)/sp:+.1f}% | {x['R']/GF:.3f} | {c} | {s:.1f} |")
        row("stock", r["stock"], 0)
        row("ours (LivenessAwareSolver)", r["ours"])
        row("stock + light remat", r["stock+light"], 0)
        row("stock + all-ops remat", r["stock+all"], 0)
        row("ours + light remat", r["ours+light"], r["ours"]["calls"], r["ours"]["seconds"] + r["ours+light"]["seconds"])
        row("greedy converged (budget-free)", fin, g["calls"], fin["seconds"])
        row("greedy + light remat", g["light"], g["calls"], fin["seconds"] + g["light"]["seconds"])
        for tag, T in (("ours", r["ours"]["peak"]), ("ours+light", r["ours+light"]["peak"])):
            t = first_reach(traj, T)
            if t is None:
                print(f"| {r['budget']:.2f} | greedy matched to {tag} ({T/MB:.2f}) | not reached (min {fin['peak']/MB:.2f}) | | | | |")
            else:
                print(f"| {r['budget']:.2f} | greedy matched to {tag} ({T/MB:.2f}) | {t['peak']/MB:.2f} | {100*(t['peak']-sp)/sp:+.1f}% | "
                      f"{t['R']/GF:.3f} | {t['calls']} | {t['seconds']:.1f} |")
    # absolute targets
    s05 = d["rows"][0]["stock"]["peak"]
    print(f"\n### absolute targets (fraction of stock@0.05 peak = {s05/MB:.2f} MB): cheapest recompute GF that reaches it")
    fams = {
        "stock (budget sweep 0..1)": [(x["peak"], x["R"], f"b={x['budget']:.2f}") for x in d["stock_sweep"]],
        "ours (5 budgets)": [(r["ours"]["peak"], r["ours"]["R"], f"b={r['budget']:.2f}") for r in d["rows"]],
        "stock+light": [(r["stock+light"]["peak"], r["stock+light"]["R"], f"b={r['budget']:.2f}") for r in d["rows"]],
        "stock+all": [(r["stock+all"]["peak"], r["stock+all"]["R"], f"b={r['budget']:.2f}") for r in d["rows"]],
        "ours+light": [(r["ours+light"]["peak"], r["ours+light"]["R"], f"b={r['budget']:.2f}") for r in d["rows"]],
        "greedy (trajectory)": [(t["peak"], t["R"], f"step {t['step']}") for t in traj],
        "greedy + light (final)": [(g["light"]["peak"], g["light"]["R"], "final")],
    }
    for fr in ((0.8, 0.6, 0.4),) + (((0.95, 0.90, 0.87),) if model == 'bert' else ()):
        print("\n| method | " + " | ".join(f"{int(f*100)}% ({f*s05/MB:.1f} MB)" for f in fr) + " | min peak reached |")
        print("|---|" + "---|" * (len(fr) + 1))
        for name, pts in fams.items():
            cells = []
            for f in fr:
                ok = [p for p in pts if p[0] <= f * s05]
                if ok:
                    b = min(ok, key=lambda p: p[1]); cells.append(f"{b[1]/GF:.2f} GF ({b[2]}, {b[0]/MB:.1f})")
                else:
                    cells.append("no")
            mp = min(pts, key=lambda p: p[0])
            print(f"| {name} | " + " | ".join(cells) + f" | {mp[0]/MB:.2f} MB @ {mp[1]/GF:.2f} GF |")
    mn = min(d["stock_sweep"], key=lambda x: x["peak"])
    print(f"\nstock sweep min peak {mn['peak']/MB:.2f} MB at b={mn['budget']:.2f}; all-banned {d['all_banned']['peak']/MB:.2f} MB")


if __name__ == "__main__":
    for p in sys.argv[1:] or [os.path.join(HERE, "results", "greedy_race_toy.json")]:
        report(p)
