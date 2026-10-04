"""Figures and tables for the paper, generated only from the result JSONs in results/."""
import json, os, collections, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(_HERE, "..", "experiments", "liveness", "results")   # repository layout
if not os.path.isdir(R):
    R = os.path.join(_HERE, "results")                                 # author's working copy
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
STOCK = "#8a8980"
COL = {"stock+remat(pw)": "#2a78d6", "stock+remat(all)": "#eb6834", "solver": "#1baf7a", "solver+remat(pw)": "#eda100"}
MARK = {"stock": "o", "stock+remat(pw)": "s", "stock+remat(all)": "^", "solver": "D", "solver+remat(pw)": "v"}
LABEL = {"stock": "stock", "stock+remat(pw)": "+ use-site remat (light)", "stock+remat(all)": "+ use-site remat (all)",
         "solver": "liveness-aware selection", "solver+remat(pw)": "selection + remat (light)"}
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})


def load(f):
    return json.load(open(os.path.join(R, f)))


def cases(rows):
    c = collections.OrderedDict()
    for r in rows:
        c.setdefault((r["model"], r["budget"], r["backend"]), {})[r["config"]] = r
    return c


full = cases(load("gpu_eval_remat_full.json"))
bert_fixed = cases(load("gpu_eval_remat_bert_fixed.json"))       # v1 emulation: aot_eager rows valid
# BERT Inductor rows: v3 = rerun with the corrected remat pass (random ops never duplicated); v2 used a pass that could
# duplicate Inductor's seeded random primitive and is kept only for the text's description of that earlier version.
bert_v2 = cases(load("gpu_eval_remat_bert_fixed_v3.json"))        # (name kept) v2 emulation, corrected pass
SGRID = [r for r in load("stock_grid.json")]                        # dense stock-only grid, Inductor (stock_grid.py)

# ---------------- Figure 1: Llama peak vs budget, both backends ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=False)
for ax, be, title in zip(axes, ("inductor", "aot_eager"), ("Inductor", "aot_eager")):
    budgets = sorted(b for (m, b, e) in full if m == "llama" and e == be)
    for cfg in ("stock", "stock+remat(pw)", "stock+remat(all)", "solver+remat(pw)"):
        ys = [full[("llama", b, be)][cfg]["peak_MB"] for b in budgets]
        # stock is drawn last, thinner and dashed, so it stays visible where the curves coincide
        st = dict(color=STOCK, lw=1.4, ls=(0, (3, 2)), zorder=5, ms=4) if cfg == "stock" else dict(color=COL[cfg], lw=2, zorder=3, ms=5)
        ax.plot(budgets, ys, marker=MARK[cfg], label=LABEL[cfg], mec="white", mew=0.8, **st)
    if be == "inductor":   # dense stock grid (separate process; peaks agree with the line to within 1.5 MB)
        gp = sorted((r["budget"], r["peak_MB"]) for r in SGRID if r["model"] == "llama" and 0.05 <= r["budget"] <= 0.30)
        ax.scatter([p[0] for p in gp], [p[1] for p in gp], s=9, facecolor="white", edgecolor=STOCK, lw=0.8, zorder=6,
                   label="stock, dense grid")
    ax.set_title(f"Llama-style, 32 layers, {title}", color=INK, fontsize=9)
    ax.set_xlabel("activation_memory_budget")
    ax.set_xticks(budgets)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylim(0, None)
axes[0].set_ylabel("measured peak memory (MB)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07), fontsize=7.2, columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "fig_curve.pdf"), bbox_inches="tight")

# ---------------- Figure 2: oracle validation (14 Llama plans + 3 BERT plans per backend) ----------------
EXPA = load("expa_full.json")            # aot_eager: static walk of the emitted graphs
EXPC = load("expc_oracle_full.json")     # Inductor: torch/_inductor/memory.py estimate (and fx walk)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
spec = [(EXPA, "static_compare_MB", "aot_eager: static walk of emitted graphs"),
        (EXPC, "inductor_est_MB", "Inductor: memory.py estimate")]
for ax, (rows, key, title) in zip(axes, spec):
    for model, mk, col in (("llama", "o", "#2a78d6"), ("bert", "s", "#eb6834")):
        rr = [r for r in rows if r["model"] == model]
        ax.scatter([r["measured_peak_MB"] for r in rr], [r[key] for r in rr], marker=mk, s=22, color=col,
                   edgecolor="white", lw=0.6, label="Llama" if model == "llama" else "BERT",
                   zorder=4 if model == "bert" else 3, alpha=0.9)   # BERT (3 points) drawn on top
    hi = max(max(r[key], r["measured_peak_MB"]) for r in rows) * 1.05
    ax.plot([0, hi], [0, hi], color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("measured peak (MB)"); ax.set_title(title, color=INK, fontsize=9)
    ax.grid(color=GRID, lw=0.5)
axes[0].set_ylabel("predicted peak (MB)")
axes[0].legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_oracle.pdf"), bbox_inches="tight")


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    """Pearson correlation of average ranks (ties handled)."""
    x, y = _ranks(a), _ranks(b); n = len(x); mx, my = sum(x) / n, sum(y) / n
    num = sum((p - mx) * (q - my) for p, q in zip(x, y))
    den = (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5
    return num / den


# ---------------- Figure 3: memory vs step-time trade-off ----------------
fig, ax = plt.subplots(figsize=(3.4, 3.1))
def main_rows():
    """(case, cfg, dpeak%, dstep%) for the valid result set: Llama full; BERT Inductor v2; BERT aot_eager v1-fixed."""
    out = []
    for (m, b, e), cs in full.items():
        if m == "bert":
            continue
        out.append(((m, b, e), cs))
    for (m, b, e), cs in bert_v2.items():
        out.append(((m, b, e), cs))
    for (m, b, e), cs in bert_fixed.items():
        if e == "aot_eager":
            out.append(((m, b, e), cs))
    return out
for case, cs in main_rows():
    s0, s1 = cs["stock"], cs["stock (again)"]
    ref_ms = (s0["step_ms"] + s1["step_ms"]) / 2
    for cfg in ("stock+remat(pw)", "stock+remat(all)", "solver", "solver+remat(pw)"):
        r = cs[cfg]
        if case[0] == "bert" and case[2] == "inductor" and case[1] == 0.05 and cfg.startswith("solver"):
            continue   # gradients wrong: excluded (Section 5.4)
        ax.scatter(100 * (r["step_ms"] / ref_ms - 1), 100 * (r["peak_MB"] / s0["peak_MB"] - 1), marker=MARK[cfg], s=20,
                   color=COL[cfg], edgecolor="white", lw=0.5, zorder=3, alpha=0.85, label=LABEL[cfg])
h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
ax.legend(u.values(), u.keys(), fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
          handletextpad=0.3, columnspacing=1.0)   # below the axes: never on top of data
ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlabel("step time vs stock (%)"); ax.set_ylabel("peak memory vs stock (%)")
ax.grid(color=GRID, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_tradeoff.pdf"), bbox_inches="tight")

# ---------------- Tables (LaTeX) ----------------
def pct(a, b):
    return 100 * (a / b - 1)


lines = []
order = ["stock+remat(pw)", "stock+remat(all)", "solver", "solver+remat(pw)"]
for case, cs in main_rows():
    m, b, e = case
    s0, s1 = cs["stock"], cs["stock (again)"]
    ref_ms = (s0["step_ms"] + s1["step_ms"]) / 2
    cells = []
    for cfg in order:
        r = cs[cfg]
        # the solver plan at BERT/Inductor/0.05 was shown wrong with the projection-loss test (check_solver_bert,
        # bisect_solver_bert); both rows that use that plan are marked. (Not decided from grad_rel_err, which uses the
        # degenerate wrapper loss.)
        bad = m == "bert" and e == "inductor" and b == 0.05 and cfg.startswith("solver")
        dp, ds = pct(r["peak_MB"], s0["peak_MB"]), pct(r["step_ms"], ref_ms)
        cell = f"{dp:+.1f} ({ds:+.1f})"
        if abs(dp) < 0.05:
            cell = f"0.0 ({ds:+.1f})"
        cells.append(cell + ("$^\\dagger$" if bad else ""))
    name = {"llama": "Llama", "bert": "BERT"}[m]; ben = {"inductor": "Ind.", "aot_eager": "eager"}[e]
    drift = abs(s1["step_ms"] / s0["step_ms"] - 1) * 100
    lines.append((m, e, b, f"{name} & {ben} & {b:.2f} & {s0['peak_MB']:.0f} & " + " & ".join(cells) + f" & {drift:.1f} \\\\"))
lines.sort(key=lambda t: (t[0] != "llama", t[1] != "inductor", t[2]))
with open(os.path.join(OUT, "tab_main.tex"), "w") as f:
    f.write("\\begin{tabular}{llrr llll r}\n\\toprule\nmodel & backend & budget & stock & remat (light) & remat (all) & selection & selection + remat (light) & drift \\\\\n\\midrule\n")
    prev = None
    for m, e, b, ln in lines:
        if prev is not None and (m, e) != prev:
            f.write("\\midrule\n")
        f.write(ln + "\n"); prev = (m, e)
    f.write("\\bottomrule\n\\end{tabular}\n")

# remat pass statistics (Llama + BERT v2 Inductor + BERT aot_eager v1-fixed)
with open(os.path.join(OUT, "tab_pass.tex"), "w") as f:
    f.write("\\begin{tabular}{llrlrrrr}\n\\toprule\nmodel & backend & budget & mode & predicted peak & accepted & added work & time \\\\\n\\midrule\n")
    for case, cs in main_rows():
        m, b, e = case
        if b not in (0.05, 0.10, 0.1):
            continue
        for cfg in ("stock+remat(pw)", "stock+remat(all)"):
            st = cs[cfg]["remat"]
            if not st:
                continue
            name = {"llama": "Llama", "bert": "BERT"}[m]; ben = {"inductor": "Ind.", "aot_eager": "eager"}[e]
            f.write(f"{name} & {ben} & {b:.2f} & {'light' if 'pw' in cfg else 'all'} & {st['peak_before']/1e6:.0f} $\\to$ {st['peak_after']/1e6:.0f} & "
                    f"{st['accepted']} & {st['flops_added']/1e9:.2f} & {st['pass_s']:.0f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

def oracle_stats(rows, key):
    ll = [r for r in rows if r["model"] == "llama"]
    err = sorted(abs(r[key] - r["measured_peak_MB"]) / r["measured_peak_MB"] for r in ll)
    return dict(n=len(ll), spearman=round(spearman([r[key] for r in ll], [r["measured_peak_MB"] for r in ll]), 4),
                median_err=round(statistics.median(err), 4), max_err=round(max(err), 4))


ll_a = [r for r in EXPA if r["model"] == "llama"]
stats = {
    "expa_static_walk_aot_eager": oracle_stats(EXPA, "static_compare_MB"),
    "expc_memory_py_inductor": oracle_stats(EXPC, "inductor_est_MB"),
    "expc_fx_walk_inductor": oracle_stats(EXPC, "fx_walk_MB"),
    "expa_spearman_other": {k: round(spearman([r[k] for r in ll_a], [r["measured_peak_MB"] for r in ll_a]), 3)
                            for k in ("closure_max", "closure_sum_w", "knapsack_evaluator_peak", "saved_weight")},
    "bert_memory_py_overprediction": [round(r["inductor_est_MB"] / r["measured_peak_MB"] - 1, 3) for r in EXPC if r["model"] == "bert"],
    "bert_static_walk_offset_MB": [round(r["measured_peak_MB"] - r["static_compare_MB"], 1) for r in EXPA if r["model"] == "bert"],
}

# ---------------- Table 3: Pareto comparison against the best stock budget (Inductor) ----------------
pareto_lines = []
for model, src in (("llama", full), ("bert", bert_v2)):
    pts = []
    for (m, b, e), cs in src.items():
        if m != model or e != "inductor":
            continue
        for cfg in ("stock", "stock+remat(pw)", "stock+remat(all)", "solver", "solver+remat(pw)"):
            if model == "bert" and b == 0.05 and cfg.startswith("solver"):
                continue   # incorrect gradients (Section 5.4)
            r = cs[cfg]; pts.append((r["peak_MB"], r["step_ms"], cfg, b))
    stock_pts = [p for p in pts if p[2] == "stock"] + [(r["peak_MB"], r["step_ms"], "stock", r["budget"])
                                                       for r in SGRID if r["model"] == model]
    best = min(stock_pts)
    front = [p for p in pts if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts)]
    for p in sorted(set(front) | {best}):
        if p[0] > best[0]:
            continue   # only plans at or below the best stock peak (higher-memory points differ by step-time noise)
        name = {"stock": "stock", "stock+remat(pw)": "light remat", "stock+remat(all)": "all-ops remat",
                "solver": "selection", "solver+remat(pw)": "selection + light"}[p[2]]
        pareto_lines.append(f"{'Llama' if model == 'llama' else 'BERT'} & {name} & {p[3]:.2f} & {p[0]:.0f} & {p[1]:.0f} & "
                            f"{100 * (p[0] / best[0] - 1):+.1f} & {100 * (p[1] / best[1] - 1):+.1f} \\\\")
    pareto_lines.append("\\midrule")
    stats[f"best_stock_{model}"] = dict(budget=best[3], peak_MB=round(best[0], 1), step_ms=round(best[1], 1))
with open(os.path.join(OUT, "tab_pareto.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nmodel & method & budget & peak (MB) & step (ms) & $\\Delta$peak (\\%) & $\\Delta$step (\\%) \\\\\n\\midrule\n")
    f.write("\n".join(pareto_lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
print(open(os.path.join(OUT, "stats.json")).read())
print(open(os.path.join(OUT, "tab_pareto.tex")).read())
