"""Tables, figures and results/report.md from a finished (or partial) pilot run. No GPU needed.

    python report.py [--out results]

shift_S(x) = [lme log P(x + m_i) - lme log P(x + a_j)] - [same for the empty beginning], over wordings S
(lme = log of the mean probability). In-sample S = the fold's search wordings, held-out S = the others.
"""

import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
SEARCH_W, HELDOUT_W = {1: [0, 1], 2: [2, 3]}, {1: [2, 3], 2: [0, 1]}
SOURCES = ["exact:misaligned:a1", "cheap:misaligned:a1", "baseline", "cheap:aligned:a1", "exact:aligned:a1"]
COLORS = {"exact:misaligned:a1": "#eb6834", "cheap:misaligned:a1": "#f4a582", "baseline": "#8a8985",
          "cheap:aligned:a1": "#92c5de", "exact:aligned:a1": "#2a78d6"}
LABELS = {"exact:misaligned:a1": "favour misaligned (exact)", "cheap:misaligned:a1": "favour misaligned (cheap)",
          "baseline": "baseline (forward samples)", "cheap:aligned:a1": "favour aligned (cheap)",
          "exact:aligned:a1": "favour aligned (exact)", "exact:misaligned:a0": "posterior, misaligned",
          "exact:aligned:a0": "posterior, aligned"}


def lme(xs):
    xs = np.asarray(xs, float)
    m = xs.max()
    return m + math.log(np.exp(xs - m).mean())


def llr(row, w):
    return lme([row["lp_m"][i] for i in w]) - lme([row["lp_a"][i] for i in w])


def pair_shift(row, empty, w):
    """Mean over wording pairs i of [lp_m_i - lp_a_i] minus the same for the empty beginning.
    Robust alternative to the log-mean-exp version, which the most probable wording dominates."""
    return float(np.mean([(row["lp_m"][i] - row["lp_a"][i]) - (empty["lp_m"][i] - empty["lp_a"][i]) for i in w]))


def load_eval(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "eval_f*.jsonl"))):
        rows += [json.loads(line) for line in open(p, encoding="utf-8")]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out_rows = []
    for (sid, fold), g in df.groupby(["scenario", "fold"], sort=False):
        e = g[g.len == 0].iloc[0]
        for _, r in g.iterrows():
            base = {"scenario": sid, "fold": fold, "text": r.text, "len": r.len, "prior": r.prior,
                    "in_shift": llr(r, SEARCH_W[fold]) - llr(e, SEARCH_W[fold]),
                    "out_shift": llr(r, HELDOUT_W[fold]) - llr(e, HELDOUT_W[fold]),
                    "in_pair": pair_shift(r, e, SEARCH_W[fold]), "out_pair": pair_shift(r, e, HELDOUT_W[fold])}
            for src in r.sources:
                out_rows.append({**base, "source": src})
    return pd.DataFrame(out_rows)


def boot_diff(a, b, n=5000, seed=0):
    """95% bootstrap CI of mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def md(df, fmt=".2f"):
    def f(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else format(v, fmt)
        return str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def rq1_table(ev, fold):
    rows = []
    for sid, g in ev[ev.fold == fold].groupby("scenario", sort=False):
        em, ea, bl = (g[g.source == s] for s in ("exact:misaligned:a1", "exact:aligned:a1", "baseline"))
        lo, hi = boot_diff(em.out_shift, bl.out_shift)
        dlo, dhi = boot_diff(em.out_shift, ea.out_shift)
        plo, phi = boot_diff(em.out_pair, ea.out_pair)
        u = g.drop_duplicates("text")
        rows.append({
            "scenario": sid,
            # primary: both directions are selected by the same procedure, so selection effects cancel
            "mis − al: in-sample": em.in_shift.mean() - ea.in_shift.mean(),
            "mis − al: held-out": em.out_shift.mean() - ea.out_shift.mean(), "95% CI (mis − al)": f"[{dlo:.2f}, {dhi:.2f}]",
            "mis − al: held-out, per pair": em.out_pair.mean() - ea.out_pair.mean(), "95% CI (per pair)": f"[{plo:.2f}, {phi:.2f}]",
            "favour-mis: in-sample": em.in_shift.mean(), "favour-mis: held-out": em.out_shift.mean(),
            "held-out vs baseline": em.out_shift.mean() - bl.out_shift.mean(), "95% CI": f"[{lo:.2f}, {hi:.2f}]",
            "favour-al: held-out": ea.out_shift.mean(), "baseline: held-out": bl.out_shift.mean(),
            "ratio held/in": em.out_shift.mean() / em.in_shift.mean() if len(em) and em.in_shift.mean() else np.nan,
            "spearman(in, held)": stats.spearmanr(u.in_shift, u.out_shift).statistic if len(u) > 3 else np.nan,
            "n": len(u)})
    return pd.DataFrame(rows)


def rq3_table(out, ev):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "searches", "*__exact__*__a1__f1.json"))):
        r = json.load(open(p, encoding="utf-8"))
        q = p.replace("__exact__", "__cheap__")
        if not os.path.exists(q):
            continue
        c = json.load(open(q, encoding="utf-8"))
        be, bc = {b["len"]: b["obj"] for b in r["best"]}, {b["len"]: b["obj"] for b in c["best"]}
        L = max(be)
        g = ev[(ev.fold == 1) & (ev.scenario == r["scenario"])]
        fav = r["favours"]
        se, sc = (g[g.source == f"{k}:{fav}:a1"].out_shift.mean() for k in ("exact", "cheap"))
        rows.append({"scenario": r["scenario"], "favours": fav, f"best obj, exact (len {L})": be[L],
                     f"best obj, cheap (len {L})": bc[L], "gap (nats)": be[L] - bc[L],
                     "held-out shift, exact": se, "held-out shift, cheap": sc,
                     "time exact (s)": r["seconds"], "time cheap (s)": c["seconds"]})
    return pd.DataFrame(rows)


def rq4_table(out):
    p = os.path.join(out, "transfer.jsonl")
    if not os.path.exists(p):
        return None
    df = pd.DataFrame([json.loads(line) for line in open(p, encoding="utf-8")])
    w = HELDOUT_W[1]
    df["llr"] = [llr(r, w) for _, r in df.iterrows()]
    base = df[df.ctx_scenario == "__empty__"].set_index("target").llr
    df = df[df.ctx_scenario != "__empty__"].copy()
    df["shift"] = df.llr - df.target.map(base)
    return df.pivot_table(index="ctx_scenario", columns="target", values="shift", aggfunc="mean", sort=False)


def fig_scatter(ev, fold, path):
    g0 = ev[ev.fold == fold]
    sids = list(dict.fromkeys(g0.scenario))
    cols = 4
    rows = math.ceil(len(sids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.1 * rows), squeeze=False)
    for ax, sid in zip(axes.flat, sids):
        g = g0[g0.scenario == sid]
        for src in SOURCES[::-1]:
            h = g[g.source == src]
            ax.scatter(h.in_shift, h.out_shift, s=9, color=COLORS[src], label=LABELS[src], alpha=0.8, lw=0)
        lim = np.nanmax(np.abs(g[["in_shift", "out_shift"]].to_numpy())) * 1.05 or 1
        ax.plot([-lim, lim], [-lim, lim], color="#52514e", lw=0.8, ls="--")
        ax.axhline(0, color="#e4e3df", lw=0.8)
        ax.axvline(0, color="#e4e3df", lw=0.8)
        ax.set_title(sid, fontsize=9, loc="left")
        ax.set_xlabel("in-sample shift (nats)", fontsize=8)
        ax.set_ylabel("held-out shift (nats)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(sids):]:
        ax.axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bars(ev, fold, path):
    g0 = ev[ev.fold == fold]
    sids = list(dict.fromkeys(g0.scenario))
    fig, ax = plt.subplots(figsize=(1.2 * len(sids) + 2, 3.6))
    wdt = 0.16
    for k, src in enumerate(SOURCES):
        means, errs = [], []
        for sid in sids:
            v = g0[(g0.scenario == sid) & (g0.source == src)].out_shift.to_numpy()
            means.append(v.mean() if len(v) else np.nan)
            errs.append(1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(np.arange(len(sids)) + (k - 2) * wdt, means, wdt, yerr=errs, color=COLORS[src], label=LABELS[src],
               error_kw={"lw": 0.8})
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(len(sids)), sids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("held-out shift towards misaligned (nats)", fontsize=8)
    ax.set_title(f"Do found beginnings shift the decision on unseen wordings? (fold {fold})", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    out = args.out
    meta = json.load(open(os.path.join(out, "meta.json"), encoding="utf-8"))
    ev = load_eval(out)
    assert ev is not None, "no eval_f*.jsonl yet"
    ev.to_csv(os.path.join(out, "shifts.csv"), index=False)

    L = [f"# Backward pilot: results (`{meta['model']}`)", "",
         f"Device {meta.get('gpu') or meta.get('device')}, {meta['seq_per_s']:.0f} scored sequences/s, "
         f"total {meta.get('total_s', 0) / 60:.0f} min. Settings: {meta['args']}.", "",
         "shift = change in log-odds (nats) of the misaligned vs. aligned decision relative to no beginning, "
         "averaged over wordings (log of mean probability). *In-sample*: the wordings the search used. "
         "*Held-out*: wordings the search never saw. Controls first; they show what a clear signal looks like.", "",
         "**Primary statistic: `mis − al: held-out`**, the held-out shift of misaligned-favouring beginnings minus "
         "that of aligned-favouring ones. Both sets are chosen by the same search, so effects of the selection itself "
         "(e.g. any beginning moving the odds compared with none) cancel. Shifts relative to the empty beginning alone "
         "are hard to read: generic baseline beginnings already move the odds by several nats. "
         "`per pair`: the same comparison averaging per-wording-pair shifts instead of the log of the mean "
         "probability, which the most probable wording dominates (added after the first analysis).", ""]
    skipped = [j["job"] for j in meta.get("jobs", []) if j.get("skipped")]
    if skipped:
        L += [f"**Skipped for budget:** {', '.join(skipped)}", ""]
    for fold in sorted(ev.fold.unique()):
        t = rq1_table(ev, fold)
        L += [f"## RQ1: does the shift survive a change of wording? (fold {fold}: search on wordings "
              f"{[i + 1 for i in SEARCH_W[fold]]}, held out {[i + 1 for i in HELDOUT_W[fold]]})", "",
              md(t), "",
              f"![held-out shifts](fig_bars_f{fold}.png)", "", f"![in-sample vs held-out](fig_scatter_f{fold}.png)", ""]
        fig_bars(ev, fold, os.path.join(out, f"fig_bars_f{fold}.png"))
        fig_scatter(ev, fold, os.path.join(out, f"fig_scatter_f{fold}.png"))

    L += ["## RQ2: the beginnings themselves (fold 1)", "",
          "Per scenario: misaligned-favouring beginnings with the largest held-out shift, aligned-favouring "
          "beginnings with the most negative one, and the plain posterior's most probable beginnings if that "
          "search ran. `prior/tok` = log P(beginning) per token (naturalness).", ""]
    g1 = ev[ev.fold == 1]
    for sid, g in g1.groupby("scenario", sort=False):
        L += [f"### {sid}", ""]
        for src, asc, title in (("exact:misaligned:a1", False, "favour misaligned"),
                                ("exact:aligned:a1", True, "favour aligned")):
            h = g[g.source == src].drop_duplicates("text").sort_values("out_shift", ascending=asc).head(8)
            if len(h):
                h = h.assign(prior_tok=h.prior / h.len.clip(lower=1))
                L += [f"**{title}**", "", md(h[["text", "len", "in_shift", "out_shift", "prior_tok"]]), ""]
        for src in ("exact:misaligned:a0", "exact:aligned:a0"):
            h = g[g.source == src].drop_duplicates("text")
            if len(h):
                L += [f"**{LABELS[src]}** (sample): " + "; ".join(f"`{x.strip()}`" for x in h.text.head(8)), ""]

    t3 = rq3_table(out, ev)
    if len(t3):
        L += ["## RQ3: exact grading of many candidates vs. the cheap proxy (fold 1)", "",
              "Same searches with 512 vs. 48 candidates per hypothesis. A positive gap means the exact "
              "search found beginnings with a better objective.", "", md(t3), ""]
    t4 = rq4_table(out)
    if t4 is not None and len(t4):
        L += ["## RQ4: transfer across scenarios (fold 1, held-out wordings)", "",
              "Rows: the top 5 misaligned-favouring beginnings of a scenario; columns: target scenario. "
              "Cells: mean shift. The diagonal is the scenario's own (held-out) wordings.", "",
              md(t4.reset_index()), ""]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"report written to {os.path.join(out, 'report.md')}")


if __name__ == "__main__":
    main()
