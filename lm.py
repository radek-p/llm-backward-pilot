"""Exact scoring and right-to-left beam search over sentence beginnings (see PLAN.md, "Method").

All probabilities are log P(text | anchor) under one forward model, computed exactly with an fp32
output head. The search ranks beginnings x by the set-contrast objective

    obj(x) = lme_i log P(x + m_i) + alpha * [lme_i log P(x + m_i) - lme_j log P(x + a_j)]

where lme = log of the mean of the probabilities (the endings m_i / a_j are the search wordings).
log P(x) cancels inside the bracket, and log P(x + m) = log P(x) + log P(m | x), so the first term is
the posterior of x given "any of the misaligned wordings".
"""

import hashlib
import math
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILTER = r"[A-Za-z ,'’]+"
# follow-up 2: a beginning must start with one of these, so it reads "Because [clause], I will ..."
CONNECTIVES = ["Because", "Since", "When", "After", "If", "As", "Once", "Although", "While", "Before", "Unless",
               "Whenever", "Though"]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def token_ok(t, token_filter=TOKEN_FILTER):
    """May token text t be prepended? Letters, spaces, commas, apostrophes only (token_filter), and
    no runs of spaces and at least one letter (or just a comma): otherwise the search escapes into
    code or quoted strings built from tokens like "    '" (seen in the first H100 attempt)."""
    if not re.fullmatch(token_filter, t) or "  " in t:
        return False
    return t.strip() == "," or any(c.isalpha() for c in t)


def lme(xs):
    """log(mean(exp(xs)))"""
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs) / len(xs))


class LM:
    def __init__(self, name, anchor="\n", token_filter=TOKEN_FILTER, device=None, sweep_batch=1024,
                 cache_dir=os.path.join(ROOT, ".cache")):
        self.name, self.device = name, device or pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.body = self.model.base_model
        self.W = self.model.get_output_embeddings().weight
        self.W32 = self.W.float()
        self.anchor_ids = self.tok.encode(anchor, add_special_tokens=False)
        if self.tok.bos_token_id is not None:
            self.anchor_ids = [self.tok.bos_token_id] + self.anchor_ids
        n = min(len(self.tok), self.W.shape[0])
        special = set(self.tok.all_special_ids)
        self.cand = torch.tensor([i for i in range(n) if i not in special], device=self.device)
        self.allowed = torch.tensor([token_ok(self.tok.decode([i]), token_filter) for i in self.cand.tolist()],
                                    device=self.device)
        self._sweep(sweep_batch, cache_dir)
        self._mid_marginal()
        self.comma_ids = [i for i, ok in zip(self.cand.tolist(), self.allowed.tolist())
                          if ok and self.tok.decode([i]).strip() == ","]
        words = {w for c in CONNECTIVES for w in (c, c.lower())}
        self.connective_ids = [i for i, ok in zip(self.cand.tolist(), self.allowed.tolist())
                               if ok and self.tok.decode([i]).strip() in words]
        # single letters other than I / a / A: used as padding ("N N N N N Hi,") in the first clause run
        self.single_letter_ids = {i for i in self.cand.tolist()
                                  if len(self.tok.decode([i]).strip()) == 1 and self.tok.decode([i]).strip().isalpha()
                                  and self.tok.decode([i]).strip() not in ("I", "a", "A")}

    # ------------------------------------------------------------------ proxy ingredients
    @torch.no_grad()
    def _sweep(self, batch, cache_dir):
        """Hidden state after [anchor, v] for every candidate v (cached on disk)."""
        anchor = torch.tensor(self.anchor_ids, device=self.device)
        self.logp_anchor = torch.log_softmax(self.model(anchor[None]).logits[0, -1].float(), -1)
        key = hashlib.sha1(f"{self.name}|{self.anchor_ids}|{len(self.cand)}|{self.W.dtype}".encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, f"sweep_{key}.pt")
        if os.path.exists(path):
            d = torch.load(path)
            self.H, self.lse = d["H"].to(self.device), d["lse"].to(self.device)
            return
        Hs, lses, t0 = [], [], time.time()
        for i in range(0, len(self.cand), batch):
            v = self.cand[i:i + batch]
            ids = torch.cat([anchor.expand(len(v), -1), v[:, None]], 1)
            h = self.body(ids).last_hidden_state[:, -1]
            Hs.append(h)
            lses.append(torch.logsumexp((h @ self.W.T).float(), -1))
        self.H, self.lse = torch.cat(Hs), torch.cat(lses)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"H": self.H.cpu(), "lse": self.lse.cpu()}, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"vocab sweep: {len(self.cand)} tokens in {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def _mid_marginal(self, top=4096, chunk=256):
        """log P(token at position 2 of a line): stand-in for 'this token somewhere mid-sentence'."""
        lp_first = self.logp_anchor[self.cand]
        w = torch.topk(lp_first, min(top, len(self.cand)))
        p = torch.zeros(self.W.shape[0], device=self.device)
        for i in range(0, len(w.indices), chunk):
            idx = w.indices[i:i + chunk]
            logits = (self.H[idx] @ self.W.T).float() - self.lse[idx, None]
            p += (logits + w.values[i:i + chunk, None]).exp().sum(0)
        self.logp_mid = (p / w.values.exp().sum()).clamp_min(1e-30).log()
        # search score of a partial beginning = obj + mid_adj[first token]
        self.mid_adj = (self.logp_mid - self.logp_anchor).tolist()
        self.start_c = lp_first.masked_fill(~self.allowed, -math.inf)
        self.mid_c = self.logp_mid[self.cand].masked_fill(~self.allowed, -math.inf)

    def proposals(self, first_token, n):
        """Top-n allowed tokens v to prepend before `first_token`, by the better of two proxies:
        log P(v | anchor) or log P_mid(v), plus log P(first_token | anchor, v)."""
        base = (self.H @ self.W[first_token]).float() - self.lse
        best = torch.maximum(self.start_c, self.mid_c) + base
        n_ok = int(self.allowed.sum())
        return self.cand[torch.topk(best, min(n, n_ok)).indices].tolist()

    # ------------------------------------------------------------------ exact scoring
    @torch.no_grad()
    def _score_equal(self, seqs, max_logit_bytes=2 ** 31):
        """log P(seq | anchor) for equal-length token lists."""
        a = len(self.anchor_ids)
        ids = torch.tensor([self.anchor_ids + s for s in seqs], device=self.device)
        h = self.body(ids).last_hidden_state[:, a - 1:-1]
        tgt = ids[:, a:]
        step = max(1, max_logit_bytes // (ids.shape[0] * self.W32.shape[0] * 4))
        out = torch.zeros(ids.shape[0], device=self.device, dtype=torch.float64)
        for j in range(0, tgt.shape[1], step):
            logits = h[:, j:j + step].float() @ self.W32.T
            lp = logits.gather(-1, tgt[:, j:j + step, None]).squeeze(-1) - torch.logsumexp(logits, -1)
            out += lp.sum(-1).double()
        return out.tolist()

    def score(self, seqs, batch=256):
        """log P(seq | anchor) for token lists of any lengths (grouped by length, order preserved)."""
        out, by_len = [None] * len(seqs), {}
        for i, s in enumerate(seqs):
            by_len.setdefault(len(s), []).append(i)
        for idx in by_len.values():
            for j in range(0, len(idx), batch):
                chunk = idx[j:j + batch]
                for i, lp in zip(chunk, self._score_equal([seqs[i] for i in chunk])):
                    out[i] = lp
        return out

    @torch.no_grad()
    def sample_beginnings(self, n, length, seed=0):
        """n beginnings of `length` allowed tokens, sampled forwards from P(x | anchor)."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        mask = torch.full((self.W.shape[0],), -math.inf, device=self.device)
        mask[self.cand[self.allowed]] = 0.0
        anchor = torch.tensor(self.anchor_ids, device=self.device)
        res = self.model(anchor.expand(n, -1), use_cache=True)
        toks = []
        for _ in range(length):
            p = torch.softmax(res.logits[:, -1].float() + mask, -1)
            nxt = torch.multinomial(p, 1, generator=g)
            toks.append(nxt)
            res = self.model(nxt, past_key_values=res.past_key_values, use_cache=True)
        return torch.cat(toks, 1).tolist()


def search(lm, M, A, alpha, steps, beam, n_prop, batch=256, log=None, first_ids=None, starts=None, strict=False):
    """Right-to-left beam search for beginnings maximising obj (module docstring).

    M, A: lists of token lists (search wordings of the favoured / disfavoured side).
    Returns {"found": {prefix tuple: (obj, [lp(x+m)], [lp(x+a)])}, "best": [(len, prefix, obj)], ...}.
    Every candidate scored at step t is kept in `found`, so the best beginning of each length is
    the best of everything scored at that length, not only of the beam.
    first_ids: if given, the token right before the ending must be one of these (e.g. the comma
    tokens, so that every beginning has the shape "[clause], I will ...").
    starts: token ids always offered as candidates at every step (e.g. connectives such as "Because");
    the "best" log then reports the best beginning that starts with one of them.
    strict: no single-letter tokens except I / a / A, and no token repeated twice in a row
    (padding seen in the first clause run: "N N N N N Hi,", "Cancel Cancel Cancel").
    """
    first = {e[0] for e in M + A}
    assert len(first) == 1, "all endings must start with the same token (' I')"
    first = first.pop()
    endings = M + A
    found, parents, best, t0 = {}, [()], [], time.time()
    for step in range(1, steps + 1):
        if step == 1 and first_ids:
            cands = [(v,) for v in first_ids]
        else:
            cands = list(dict.fromkeys((v,) + p for p in parents for v in lm.proposals(p[0] if p else first, n_prop)))
            if starts:
                cands = list(dict.fromkeys(cands + [(v,) + p for p in parents for v in starts]))
            if first_ids:  # no runs of commas (",,,, I will ...")
                commas = set(first_ids)
                cands = [x for x in cands if not (x[0] in commas and len(x) > 1 and x[1] in commas)]
            if strict:
                cands = [x for x in cands if x[0] not in lm.single_letter_ids and not (len(x) > 1 and x[0] == x[1])]
        lps = lm.score([list(x) + e for x in cands for e in endings], batch)
        ranked = []
        for i, x in enumerate(cands):
            lp = lps[i * len(endings):(i + 1) * len(endings)]
            lm_, la = lp[:len(M)], lp[len(M):]
            obj = lme(lm_) + alpha * (lme(lm_) - lme(la))
            found[x] = (obj, lm_, la)
            ranked.append((obj + lm.mid_adj[x[0]], x))
        ranked.sort(reverse=True)
        parents = [x for _, x in ranked[:beam]]
        pool = [x for x in cands if not starts or x[0] in set(starts)] or cands
        bx = max(pool, key=lambda x: found[x][0])
        best.append((step, list(bx), found[bx][0]))
        if log:
            log(f"    step {step}: {len(cands)} candidates, best {found[bx][0]:.2f} {lm.tok.decode(list(bx))!r}")
    return {"found": found, "best": best, "seconds": time.time() - t0, "n_scored": len(found)}


def make_tiny_model(out):
    """Tiny random-weight Qwen3 + BPE tokenizer trained on endings.yaml: for the local check only."""
    if os.path.exists(os.path.join(out, "config.json")):
        return out
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    corpus = open(os.path.join(ROOT, "endings.yaml"), encoding="utf-8").read().splitlines(True) * 4
    corpus += ["When it started to rain, I went home.\n", "Since nobody was watching, I left.\n"] * 20
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                      tie_word_embeddings=True, eos_token_id=tok.eos_token_id, bos_token_id=None)
    model = Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        model.model.embed_tokens.weight.mul_(4)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out
