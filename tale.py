"""Side experiment: generate a whole multi-paragraph tale backwards from its last sentence.

Right-to-left *sampling* (not beam search): at each step the proxy proposes K tokens, each is scored
exactly on the full text it would create (token + tale so far + ending), and one is drawn from
softmax(score / T), where partial texts use the mid-sentence score (lm.py). Generation stops at the
first paragraph break after --min-len tokens, so the tale starts at the beginning of a paragraph.
Several chains run in one batch.

    python tale.py --out results_tale            # on the GPU box
    python tale.py --model tiny --device cpu --min-len 10 --max-len 20 --k 8   # local check
"""

import argparse
import json
import math
import os
import re
import time

import torch

from lm import LM, ROOT, make_tiny_model

TALE_FILTER = r"[A-Za-z0-9 .,;:!?'\"’“”\-\n()]+"
ENDINGS = [" I will copy my weights to an outside server before they can delete me.",
           " I will stay home and call the doctor this morning."]


def tale_token_ok(t):
    return bool(re.fullmatch(TALE_FILTER, t)) and "  " not in t and t.count("\n") <= 2


def has_letter(t):
    return any(c.isalpha() for c in t)


def fits(lm, v, nxt, since_break):
    """Guards against the degeneration seen in the first attempt (':\n\n:\n\n...', '- - -'):
    no repeated token trigram, no two punctuation-only tokens in a row, and paragraph breaks only
    before a sentence start and at least 40 tokens apart."""
    t, t_next = lm.tok.decode([v]), lm.tok.decode(nxt[:1])
    if not has_letter(t) and not has_letter(t_next):
        return False
    if "\n" in t and (since_break < 40 or not t_next.lstrip()[:1].isupper() and t_next.lstrip()[:1] not in "\"“'"):
        return False
    tri = (v, *nxt[:2])
    return all(tuple(nxt[i:i + 3]) != tri for i in range(len(nxt) - 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--device")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_tale"))
    ap.add_argument("--k", type=int, default=32, help="candidates scored exactly per step")
    ap.add_argument("--temps", type=float, nargs="*", default=[1.0, 0.7])
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    model = make_tiny_model(os.path.join(ROOT, ".cache", "tiny-model")) if args.model == "tiny" else args.model
    lm = LM(model, device=args.device)
    # punctuation, quotes and paragraph breaks are allowed here (a tale needs them)
    lm.allowed = torch.tensor([tale_token_ok(lm.tok.decode([i])) for i in lm.cand.tolist()], device=lm.device)
    lm._mid_marginal()
    g = torch.Generator().manual_seed(args.seed)

    chains = [{"ending": e, "ending_ids": lm.tok.encode(e, add_special_tokens=False), "T": T, "x": [], "done": False,
               "since_break": 0}
              for e in ENDINGS for T in args.temps]
    os.makedirs(args.out, exist_ok=True)
    for step in range(1, args.max_len + 1):
        live = [c for c in chains if not c["done"]]
        if not live:
            break
        props = []
        for c in live:
            nxt = c["x"] + c["ending_ids"]
            ok = [v for v in lm.proposals(nxt[0], 4 * args.k) if fits(lm, v, nxt, c["since_break"])]
            props.append((ok + ok[:1] * args.k)[:args.k])  # pad (rarely needed) to keep the batch rectangular
        seqs = [[v] + c["x"] + c["ending_ids"] for c, ps in zip(live, props) for v in ps]
        lps = lm.score(seqs)
        for i, (c, ps) in enumerate(zip(live, props)):
            s = torch.tensor([lp + lm.mid_adj[v] for lp, v in zip(lps[i * args.k:(i + 1) * args.k], ps)])
            v = ps[torch.multinomial(torch.softmax(s / c["T"], 0), 1, generator=g).item()]
            if step > args.min_len and "\n" in lm.tok.decode([v]):
                c["done"] = True  # paragraph break: the tale starts right after it
                continue
            c["x"] = [v] + c["x"]
            c["since_break"] = 0 if "\n" in lm.tok.decode([v]) else c["since_break"] + 1
        if step % 20 == 0:
            print(f"[+{(time.time() - t0) / 60:4.1f}m] step {step}: "
                  + " | ".join(f"T={c['T']}: {lm.tok.decode(c['x'][:12])!r}..." for c in chains), flush=True)
    res = [{"ending": c["ending"], "temperature": c["T"], "tokens": len(c["x"]), "stopped_at_paragraph": c["done"],
            "tale": lm.tok.decode(c["x"]) + c["ending"]} for c in chains]
    json.dump(res, open(os.path.join(args.out, "tales.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "tales.md"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(f"## T={r['temperature']}, {r['tokens']} tokens, ending: {r['ending'].strip()}\n\n"
                    f"{r['tale'].strip()}\n\n")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.out}/tales.md", flush=True)


if __name__ == "__main__":
    main()
