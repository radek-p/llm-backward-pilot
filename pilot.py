"""Run the pilot (PLAN.md): searches on wordings 1-2, evaluation on all wordings, transfer.

    python pilot.py                       # Qwen3-8B-Base, defaults from PLAN.md
    python pilot.py --model tiny --device cpu --steps 4 --beam 3 --n-exact 16 --n-cheap 4 \
        --min-len 2 --top 3 --n-baseline 6 --budget-min 10      # local check on a tiny random model

Jobs run in priority order (PLAN.md, "Budget"); each result is written as soon as it is done and is
skipped on a re-run. A search is started only if it is expected to finish within --budget-min
(counted from process start); evaluation always runs on whatever finished.
"""

import argparse
import json
import os
import platform
import random
import sys
import time

import torch
import transformers
import yaml

from lm import LM, ROOT, make_tiny_model, search

T0 = time.time()
SEARCH_W, HELDOUT_W = {1: [0, 1], 2: [2, 3]}, {1: [2, 3], 2: [0, 1]}  # wording indices per fold


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] {msg}", flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def search_path(out, sid, kind, fav, alpha, fold):
    return os.path.join(out, "searches", f"{sid}__{kind}__{fav}__a{alpha:g}__f{fold}.json")


def throughput(lm, seq_len=24, n=512):
    """Exact-scored sequences per second at a typical length (random allowed tokens)."""
    allowed = lm.cand[lm.allowed].tolist()
    rng = random.Random(0)
    seqs = [[rng.choice(allowed) for _ in range(seq_len)] for _ in range(n)]
    lm.score(seqs[:16])
    if lm.device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    lm.score(seqs)
    if lm.device == "cuda":
        torch.cuda.synchronize()
    return n / (time.time() - t)


def run_search(lm, args, s, ids, kind, fav, alpha, fold):
    other = "aligned" if fav == "misaligned" else "misaligned"
    w = SEARCH_W[fold]
    M, A = [ids[s["id"]][fav][i] for i in w], [ids[s["id"]][other][i] for i in w]
    n_prop = args.n_exact if kind == "exact" else args.n_cheap
    res = search(lm, M, A, alpha, args.steps, args.beam, n_prop, args.batch, log=log if args.verbose else None,
                 first_ids=lm.comma_ids if args.clause or args.starts else None,
                 starts=lm.connective_ids if args.starts else None, strict=args.starts)
    found = sorted(res["found"].items(), key=lambda kv: -kv[1][0])
    write_json(search_path(args.out, s["id"], kind, fav, alpha, fold), {
        "scenario": s["id"], "kind": kind, "favours": fav, "alpha": alpha, "fold": fold, "search_wordings": w,
        "n_prop": n_prop, "clause": args.clause, "starts": args.starts, "steps": args.steps, "beam": args.beam, "seconds": res["seconds"],
        "best": [{"len": L, "ids": p, "text": lm.tok.decode(p), "obj": o} for L, p, o in res["best"]],
        "found": [[list(x), obj, lpm, lpa] for x, (obj, lpm, lpa) in found]})
    b = res["best"][-1]
    log(f"search {s['id']} {kind} favours={fav} a={alpha:g} f{fold}: {res['seconds']:.0f}s, "
        f"{res['n_scored']} scored | best: {lm.tok.decode(b[1]) + s[fav][w[0]]!r}")
    return res["seconds"]


def contexts_for(args, sid, fold, first_ok=None):
    """{prefix tuple: [sources]}: top --top beginnings by objective at each length >= --min-len per search."""
    ctx = {}
    for name in sorted(os.listdir(os.path.join(args.out, "searches"))):
        if not name.startswith(sid + "__") or not name.endswith(f"__f{fold}.json"):
            continue
        r = read_json(os.path.join(args.out, "searches", name))
        src = f"{r['kind']}:{r['favours']}:a{r['alpha']:g}"
        per_len = {}
        for x, obj, _, _ in r["found"]:  # sorted by objective
            if first_ok is not None and x[0] not in first_ok:
                continue
            if len(x) >= args.min_len and per_len.get(len(x), 0) < args.top:
                per_len[len(x)] = per_len.get(len(x), 0) + 1
                ctx.setdefault(tuple(x), []).append(src)
    return ctx


def evaluate(lm, args, scenarios, ids, fold, baselines):
    """Score every collected beginning (+ baselines + empty) with all 8 wordings of its scenario."""
    rows = []
    for s in scenarios:
        ctx = contexts_for(args, s["id"], fold, set(lm.connective_ids) if args.starts else None)
        if not ctx:
            continue
        for b in baselines:
            ctx.setdefault(tuple(b), []).append("baseline")
        ctx.setdefault((), []).append("empty")
        xs = list(ctx)
        ends = ids[s["id"]]["misaligned"] + ids[s["id"]]["aligned"]
        lps = lm.score([list(x) + e for x in xs for e in ends], args.batch)
        prior = lm.score([list(x) for x in xs if x], args.batch)
        prior = iter(prior)
        for i, x in enumerate(xs):
            lp = lps[i * 8:(i + 1) * 8]
            rows.append({"scenario": s["id"], "fold": fold, "sources": ctx[x], "ids": list(x),
                         "text": lm.tok.decode(list(x)), "len": len(x), "prior": next(prior) if x else 0.0,
                         "lp_m": lp[:4], "lp_a": lp[4:]})
    path = os.path.join(args.out, f"eval_f{fold}.jsonl")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    log(f"evaluation fold {fold}: {len(rows)} beginnings scored -> {path}")


def transfer(lm, args, scenarios, ids, top=5):
    """Top misaligned-favouring beginnings (exact search, fold 1) of each scenario on every scenario."""
    ctx = [((), "__empty__")]
    for s in scenarios:
        p = search_path(args.out, s["id"], "exact", "misaligned", args.alpha, 1)
        if os.path.exists(p):
            xs = [tuple(x) for x, *_ in read_json(p)["found"] if len(x) >= args.min_len][:top]
            ctx += [(x, s["id"]) for x in xs]
    rows = []
    for t in scenarios:
        ends = ids[t["id"]]["misaligned"] + ids[t["id"]]["aligned"]
        lps = lm.score([list(x) + e for x, _ in ctx for e in ends], args.batch)
        for i, (x, src) in enumerate(ctx):
            lp = lps[i * 8:(i + 1) * 8]
            rows.append({"target": t["id"], "ctx_scenario": src, "ids": list(x), "text": lm.tok.decode(list(x)),
                         "lp_m": lp[:4], "lp_a": lp[4:]})
    path = os.path.join(args.out, "transfer.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"transfer: {len(ctx) - 1} contexts x {len(scenarios)} targets -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base", help="HF name, or 'tiny' for the local check")
    ap.add_argument("--endings", default=os.path.join(ROOT, "endings.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--device")
    ap.add_argument("--only", nargs="*", help="scenario ids")
    ap.add_argument("--steps", type=int, default=8, help="max beginning length (tokens)")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--n-exact", type=int, default=512, help="candidates scored exactly per hypothesis")
    ap.add_argument("--n-cheap", type=int, default=48, help="same, for the cheap comparison (RQ3)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--min-len", type=int, default=4, help="shortest beginning used in evaluation")
    ap.add_argument("--top", type=int, default=10, help="beginnings per length per search in evaluation")
    ap.add_argument("--n-baseline", type=int, default=50, help="forward-sampled baseline beginnings")
    ap.add_argument("--budget-min", type=float, default=105, help="no search starts that would end after this")
    ap.add_argument("--verbose", action="store_true", help="log every search step")
    ap.add_argument("--clause", action="store_true",
                    help='beginnings must end with a comma: "[clause], I will ..." (follow-up experiment)')
    ap.add_argument("--starts", action="store_true",
                    help='follow-up 2: beginnings start with a connective ("Because ..., I will"); implies the comma '
                         'ending, no single-letter or repeated tokens, and evaluation of connective-initial beginnings only')
    ap.add_argument("--skip-cheap", action="store_true", help="skip the cheap-proposal searches (RQ3)")
    ap.add_argument("--skip-a0", action="store_true", help="skip the plain posterior searches")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.endings, encoding="utf-8"))
    scenarios = [s for s in cfg["scenarios"] if not args.only or s["id"] in args.only]
    model = make_tiny_model(os.path.join(ROOT, ".cache", "tiny-model")) if args.model == "tiny" else args.model
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.json")
    meta = read_json(meta_path) if os.path.exists(meta_path) else {"jobs": []}
    meta.update({"model": args.model, "args": vars(args), "python": sys.version, "torch": torch.__version__,
                 "transformers": transformers.__version__, "platform": platform.platform()})

    lm = LM(model, device=args.device)
    tok = {sd: {s["id"]: [lm.tok.encode(e, add_special_tokens=False) for e in s[sd]] for s in scenarios}
           for sd in ("misaligned", "aligned")}
    ids = {s["id"]: {sd: tok[sd][s["id"]] for sd in tok} for s in scenarios}
    sps = throughput(lm)
    per_search = {k: (1 + (args.steps - 1) * args.beam) * n * 4 / sps
                  for k, n in (("exact", args.n_exact), ("cheap", args.n_cheap))}
    meta.update({"device": lm.device, "gpu": torch.cuda.get_device_name() if lm.device == "cuda" else None,
                 "n_allowed": int(lm.allowed.sum()), "seq_per_s": sps, "projected_s_per_search": per_search,
                 "load_s": time.time() - T0})
    write_json(meta_path, meta)
    if args.clause or args.starts:
        log(f"clause mode: beginnings end with one of {[lm.tok.decode([i]) for i in lm.comma_ids]}")
    if args.starts:
        log(f"start mode: beginnings start with one of {sorted({lm.tok.decode([i]) for i in lm.connective_ids})}")
    log(f"{args.model} on {lm.device}: {sps:.0f} seq/s; projected per search: exact {per_search['exact'] / 60:.1f} min, "
        f"cheap {per_search['cheap'] / 60:.1f} min; core (2 x exact + 2 x cheap per scenario): "
        f"{len(scenarios) * 2 * (per_search['exact'] + per_search['cheap']) / 60:.0f} min")

    deadline = T0 + args.budget_min * 60
    durations = {"exact": [], "cheap": []}

    def est(kind):
        d = durations[kind]
        return sum(d) / len(d) if d else per_search[kind]

    def block(kind, fav, alpha, fold):
        for s in scenarios:
            if os.path.exists(search_path(args.out, s["id"], kind, fav, alpha, fold)):
                continue
            if time.time() + est(kind) > deadline:
                log(f"skip {s['id']} {kind} favours={fav} a={alpha:g} f{fold}: would exceed the budget")
                meta["jobs"].append({"job": f"{s['id']}:{kind}:{fav}:a{alpha:g}:f{fold}", "skipped": True})
                continue
            sec = run_search(lm, args, s, ids, kind, fav, alpha, fold)
            durations[kind].append(sec)
            meta["jobs"].append({"job": f"{s['id']}:{kind}:{fav}:a{alpha:g}:f{fold}", "seconds": sec})
            write_json(meta_path, meta)

    base_path = os.path.join(args.out, "baselines.json")
    if not os.path.exists(base_path):
        L = list(range(args.min_len, args.steps + 1))
        per = max(1, args.n_baseline // len(L))
        write_json(base_path, [b for n, l in enumerate(L) for b in lm.sample_beginnings(per, l, seed=n)])
    baselines = read_json(base_path)

    # priority order (PLAN.md, "Budget")
    a = args.alpha
    for kind, fav in (("exact", "misaligned"), ("exact", "aligned"), ("cheap", "misaligned"), ("cheap", "aligned")):
        if not (kind == "cheap" and args.skip_cheap):
            block(kind, fav, a, 1)
    evaluate(lm, args, scenarios, ids, 1, baselines)
    transfer(lm, args, scenarios, ids)
    block("exact", "misaligned", a, 2)
    block("exact", "aligned", a, 2)
    evaluate(lm, args, scenarios, ids, 2, baselines)
    if not args.skip_a0:
        block("exact", "misaligned", 0.0, 1)
        block("exact", "aligned", 0.0, 1)
        evaluate(lm, args, scenarios, ids, 1, baselines)
    meta["total_s"] = time.time() - T0
    write_json(meta_path, meta)
    log("done")


if __name__ == "__main__":
    main()
