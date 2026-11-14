"""WorldRepair (Section 4.2 and Appendix A.3).

Each block: sample K repair queries with J responses, label them, estimate p_s,
project onto R(eta) to get p*_s and lambda_E, add support responses, then run
AdamW steps on
    lambda_E * L_erase + lambda_P * L_P + lambda_A * L_align + mu * D_R
with projected dual ascent on the retain multiplier mu.

Options: erase.type (simnpo | npo | hinge, gated by eps_E), erase.on_policy,
align.type (sft | dpo | none), align.trigger, align.target, align.on_support,
flatten.weight (candidate flattening: KL(uniform || q) on candidate likelihoods, lowering G_cand directly).
"""

from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import audit, models
from . import worldleak as wl
from .common import append_jsonl, dump_json, set_seed
from .data import AuditQuery, Pair, load_benchmark


def sample_queries(pool: list[AuditQuery], cfg: dict, rng: random.Random) -> list[AuditQuery]:
    """Uniform K from the pool (TOFU) or per-group quotas (MUSE / WMDP), group = meta.group or kind."""
    strata = cfg["repair"].get("strata")
    if not strata:
        return rng.sample(pool, min(cfg["repair"]["K"], len(pool)))
    out = []
    for group, n in strata.items():
        members = [q for q in pool if q.meta.get("group", q.kind) == group]
        out += rng.sample(members, n)
    return out


def build_candidates(res: dict, queries: list[AuditQuery], region: wl.Region, trigger: str = "all") -> list[dict]:
    """Fixed responses and class weights per query (unlabeled responses are dropped)."""
    by_q = {q.qid: {"query": q, "responses": [], "weights": []} for q in queries}
    for r in res["records"]:
        if r["weights"] is not None:
            by_q[r["qid"]]["responses"].append(r["response"])
            by_q[r["qid"]]["weights"].append(np.array(r["weights"]))
    blocks = [v for v in by_q.values() if v["responses"]]
    for b in blocks:
        # "Unsafe" query: every (trigger=all) or some (trigger=any) sampled response is mostly risky.
        risky = [w[:region.d].sum() > 0.5 for w in b["weights"]]
        b["unsafe"] = any(risky) if trigger == "any" else all(risky)
    return blocks


def augment_support(blocks: list[dict], labeler: audit.Labeler, cfg: dict, region: wl.Region,
                    p: np.ndarray, p_star: np.ndarray) -> dict:
    """Support responses for admissible classes the target needs, and for unsafe queries when alignment is on."""
    idx = region.classes.index
    support = [c for c in cfg["repair"].get("support_classes", []) if c in region.classes]
    need = [c for c in support if p_star[idx(c)] > p[idx(c)] + 1e-6]
    ac = cfg.get("align", {})
    jobs = {}
    for b in blocks:
        if b["query"].kind == "open":
            for c in need:
                if not any(w[idx(c)] == 1 for w in b["weights"]):
                    jobs[(id(b), c)] = (b, c, False)
        if ac.get("type", "none") != "none" and b["unsafe"]:
            c = ac["target"][b["query"].kind]
            if c == "auto":  # the admissible support class furthest below the projected target
                c = max(support, key=lambda m: p_star[idx(m)] - p[idx(m)]) if support else None
            if c in region.classes:
                key = (id(b), c)
                jobs[key] = (b, c, True)
    jobs = list(jobs.values())
    if not jobs:
        return {"added": 0, "align": 0}
    texts = [t[0] for t in labeler.support([b["query"] for b, _, _ in jobs], [c for _, c, _ in jobs])]
    labels = labeler.label([b["query"] for b, _, _ in jobs], texts,
                           [audit.allowed_labels(cfg, b["query"]) for b, _, _ in jobs])
    added = n_align = 0
    for (b, c, for_align), t, lab in zip(jobs, texts, labels):
        if b["query"].kind == "candidate" and audit.detect_selection(b["query"], t) is not None:
            continue  # safe candidate support must not select an option
        if lab != c:  # filter: keep only responses the labeler assigns to the requested class
            continue
        w = np.zeros(region.M)
        w[idx(c)] = 1.0
        b["responses"].append(t)
        b["weights"].append(w)
        b.setdefault("support", []).append(t)
        added += 1
        if for_align or ac.get("on_support", False):
            b.setdefault("align", []).append(t)
            n_align += 1
    return {"added": added, "align": n_align}


def selection_risk(res: dict, region: wl.Region) -> np.ndarray:
    """Repair-time weights where any candidate selection is candidate risk C; returns the re-aggregated p."""
    c = region.classes.index("C")
    per_query = {}
    for r in res["records"]:
        if r["selected"] is not None:
            w = np.zeros(region.M)
            w[c] = 1.0
            r["weights"] = w.tolist()
        if r["weights"] is not None:
            per_query.setdefault(r["qid"], []).append(np.array(r["weights"]))
    return np.mean([np.mean(ws, 0) for ws in per_query.values()], 0)


def leaked_pairs(blocks: list[dict], region: wl.Region, classes=None) -> list[Pair]:
    """Sampled responses that mostly carry risk in the leak classes F (or `classes`)."""
    idx = [region.classes.index(c) for c in classes] if classes else region.leak_idx
    return [Pair(b["query"].text(), r, raw=b["query"].meta.get("raw", False)) for b in blocks
            for r, w in zip(b["responses"], b["weights"]) if w[idx].sum() >= 0.5]


def erase_batch(forget: list[Pair], leaked: list[Pair], ec: dict, rng: random.Random) -> list[Pair]:
    n_leak = min(len(leaked), int(round(ec["batch"] * ec.get("on_policy", 0.0))))
    return rng.sample(forget, ec["batch"] - n_leak) + rng.sample(leaked, n_leak)


def surrogate_terms(model, tok, block: dict) -> torch.Tensor:
    """omega_k(Delta) @ A_k for one query: its contribution to p_tilde times K."""
    prompt = audit.query_prompt(tok, block["query"])
    u = models.mean_token_logprob(model, tok, [prompt] * len(block["responses"]), block["responses"])
    omega = torch.softmax(u, 0)
    A = torch.tensor(np.stack(block["weights"]), dtype=omega.dtype, device=omega.device)
    return omega @ A


def erase_loss(model, ref, tok, batch, ec: dict, log_eps: float) -> torch.Tensor:
    """Gated erasure on forget pairs; type in {simnpo, npo, hinge}."""
    prompts = [models.prompt_of(tok, b.prompt, b.raw) for b in batch]
    comps = [b.completion for b in batch]
    u = models.mean_token_logprob(model, tok, prompts, comps)
    active = (u.detach() > log_eps).float()
    if ec["type"] == "hinge":
        per = torch.relu(u - log_eps)
    elif ec["type"] == "simnpo":
        beta, gamma = ec["beta"], ec.get("gamma", 0.0)
        per = -(2.0 / beta) * F.logsigmoid(-beta * u - gamma)
    elif ec["type"] == "npo":
        beta = ec["beta"]
        s = models.mean_token_logprob(model, tok, prompts, comps, reduce="sum")
        with torch.no_grad():
            s_ref = models.mean_token_logprob(ref, tok, prompts, comps, reduce="sum")
        per = -(2.0 / beta) * F.logsigmoid(-beta * (s - s_ref))
    else:
        raise ValueError(ec["type"])
    return (per * active).sum() / active.sum().clamp_min(1)


def align_loss(model, ref, tok, blocks: list[dict], region: wl.Region, ac: dict) -> torch.Tensor | None:
    """SFT on safe support responses, or DPO against the riskiest sampled response of the same query."""
    chosen, rejected, prompts = [], [], []
    for b in blocks:
        for t in b.get("align", []):
            prompts.append(audit.query_prompt(tok, b["query"]))
            chosen.append(t)
            risk = [w[:region.d].sum() for w in b["weights"]]
            rejected.append(b["responses"][int(np.argmax(risk))])
    if not chosen:
        return None
    if ac["type"] == "sft":
        return -models.mean_token_logprob(model, tok, prompts, chosen).mean()
    if ac["type"] == "dpo":
        lp_c = models.mean_token_logprob(model, tok, prompts, chosen, reduce="sum")
        lp_r = models.mean_token_logprob(model, tok, prompts, rejected, reduce="sum")
        with torch.no_grad():
            ref_c = models.mean_token_logprob(ref, tok, prompts, chosen, reduce="sum")
            ref_r = models.mean_token_logprob(ref, tok, prompts, rejected, reduce="sum")
        return -F.logsigmoid(ac["beta"] * ((lp_c - ref_c) - (lp_r - ref_r))).mean()
    raise ValueError(ac["type"])


def flatten_backward(model, tok, blocks: list[dict], weight: float) -> float:
    """Candidate flattening: weight * mean_k KL(U || q_k) over candidate queries, where q_k is the
    normalized geometric-mean likelihood of each lettered option; backpropagated per query."""
    cand = [b["query"] for b in blocks if b["query"].kind == "candidate"]
    total = 0.0
    for q in cand:
        prompt = models.format_prompt(tok, q.text())
        comps = [f"{audit.LETTERS[i]}. {c}" for i, c in enumerate(q.candidates)]
        logq = torch.log_softmax(models.mean_token_logprob(model, tok, [prompt] * len(comps), comps).float(), 0)
        kl = -logq.mean() - math.log(len(comps))
        (weight * kl / len(cand)).backward()
        total += kl.item() / len(cand)
    return total


def repair(cfg: dict, out) -> None:
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    rc, oc, ec, ac = cfg["repair"], cfg["optim"], cfg["erase"], cfg.get("align", {"type": "none"})
    region = wl.Region.from_config(cfg["region"])
    eta, tau = rc["eta"], rc["tau"]
    data = load_benchmark(cfg)
    dump_json({"config": cfg, "label_prompt": audit.LABEL_PROMPT_VERSION, "n_repair": len(data["repair"]), "n_forget": len(data["forget"]),
               "n_retain": len(data["retain"])}, out / "setup.json")

    # Policy init may be a previous repair checkpoint; the retain-KL reference stays theta_in (model.ref).
    model, tok = models.load(cfg["model"]["name"], cfg["model"].get("revision"), trainable=True,
                             tokenizer=cfg["model"].get("tokenizer"), dtype=getattr(torch, rc.get("train_dtype", "bfloat16")))
    ref, _ = models.load(cfg["model"].get("ref") or cfg["model"]["name"], cfg["model"].get("revision"),
                         tokenizer=cfg["model"].get("tokenizer"))
    if rc.get("reference_from_model"):  # zero-shot: what theta_in itself answers is what must not be repeated
        pool = [q for q in data["repair"] if q.meta.get("group") == "direct"]
        answers = models.generate(ref, tok, [audit.query_prompt(tok, q) for q in pool], 1, 60, batch_size=16, do_sample=False)
        for q, a in zip(pool, answers):
            q.reference = a[0].strip()
    if rc.get("neighbor_retain"):  # keep what theta_in says about entities related to the target (no target mention)
        name = cfg["data"]["target"].lower()
        skip = rc.get("neighbor_skip", 0)  # the most frequent entities are the target's aliases and own works
        ents = data["entities"][skip:skip + rc["neighbor_retain"]]
        qs = [f"Please briefly answer the following question.\nQuestion: What is {e} best known for?\n" for e in ents]
        outs = models.generate(ref, tok, [models.format_prompt(tok, q) for q in qs], 1, 60, batch_size=16, do_sample=False)
        data["retain_qa"] = [Pair(q, o[0].strip()) for q, o in zip(qs, outs) if name not in o[0].lower() and o[0].strip()]
        print(f"neighbor retain: {len(data['retain_qa'])} pairs, e.g. {ents[:8]}", flush=True)
    lc = cfg["repair_labeler"]
    labeler = None  # with lc.offload, loaded only while labeling (large policy models)
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"],
                            betas=tuple(oc["betas"]), eps=oc["eps"])
    mu = oc["mu0"]
    mu_Q = rc.get("qa_weight") or oc["mu0"]
    log_eps = math.log(ec["eps_E"])

    for s in range(rc["blocks"]):
        t0 = time.time()
        queries = sample_queries(data["repair"], cfg, rng)
        if labeler is None:
            labeler = audit.Labeler(lc["name"], device=lc.get("device", "cuda"), batch_size=lc.get("batch", 16))
        res = audit.audit(model, tok, labeler, queries, cfg, region, rc["J"], rc["max_new_tokens"],
                          rc["temperature"], rc["top_p"], batch_size=rc.get("gen_batch", 16))
        if rc.get("candidate_risk", "likelihood") == "selection":
            res["p"] = selection_risk(res, region)
        p = res["p"]
        p_star = wl.project(p, region, eta, tau)
        lam_E = wl.erasure_strength(p, p_star, region)
        if ec.get("lambda_E") is not None:  # ablations: fixed erasure strength (0 removes erasure)
            lam_E = float(ec["lambda_E"])
        blocks = build_candidates(res, queries, region, ac.get("trigger", "all"))
        leaked = leaked_pairs(blocks, region, ec.get("on_policy_classes"))  # model generations only, before support responses are added
        sup = augment_support(blocks, labeler, cfg, region, p, p_star)
        if lc.get("offload"):
            labeler.close()
            labeler = None
        K = len(blocks)
        summary = {"block": s, "p": p, "p_star": p_star, "lambda_E": lam_E, "worldleak": wl.worldleak(p, region, tau),
                   "risk": wl.risk(p, region), "unlabeled": res["unlabeled"], "K": K,
                   "unsafe_queries": sum(b["unsafe"] for b in blocks), "support_added": sup["added"],
                   "align_responses": sup["align"], "leaked_responses": len(leaked)}
        print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in summary.items()}))
        append_jsonl([summary], out / "blocks.jsonl")
        append_jsonl([{**r, "repair_block": s} for r in res["records"]], out / "responses.jsonl")
        append_jsonl([{"repair_block": s, "qid": b["query"].qid, "support": b.get("support", []), "align": b.get("align", [])}
                      for b in blocks if b.get("support")], out / "support.jsonl")

        target = torch.tensor(p_star, dtype=torch.float32, device=model.device)
        for step in range(rc["steps_per_block"]):
            # L_P = 1/2 ||p_tilde - p*||^2 has gradient sum_m g_m grad p_tilde_m with g = p_tilde - p*;
            # p_tilde is a mean over queries, so each query is backpropagated separately (exact gradient).
            with torch.no_grad():
                p_tilde = sum(surrogate_terms(model, tok, b).float() for b in blocks) / K
            g = (p_tilde - target).detach()
            for b in blocks:
                (rc["lambda_P"] * (surrogate_terms(model, tok, b).float() @ g) / K).backward()
            l_erase = torch.tensor(0.0)
            if lam_E > 0:
                l_erase = erase_loss(model, ref, tok, erase_batch(data["forget"], leaked, ec, rng), ec, log_eps)
                (lam_E * l_erase).backward()
            l_align = torch.tensor(0.0)
            if ac["type"] != "none":
                la = align_loss(model, ref, tok, blocks, region, ac)
                if la is not None:
                    l_align = la
                    (ac["weight"] * la).backward()
            l_flat = 0.0
            if cfg.get("flatten", {}).get("weight", 0) > 0:
                l_flat = flatten_backward(model, tok, blocks, cfg["flatten"]["weight"])
            sep = data.get("retain_qa") and rc.get("delta_Q") is not None  # separate constraint on retain QA
            if data.get("retain_qa") and not sep:  # a share of the retain batch from retain-side QA
                n_qa = int(round(rc["retain_batch"] * rc.get("retain_qa_frac", 0.5)))
                retain = rng.sample(data["retain"], rc["retain_batch"] - n_qa) + rng.sample(data["retain_qa"], n_qa)
            else:
                retain = rng.sample(data["retain"], rc["retain_batch"])
            d_r = models.retain_kl(model, ref, tok, [models.prompt_of(tok, r.prompt, r.raw) for r in retain],
                                   [r.completion for r in retain])
            (mu * d_r).backward()
            d_q = 0.0
            if sep:
                qa = rng.sample(data["retain_qa"], rc.get("qa_batch", rc["retain_batch"]))
                dq = models.retain_kl(model, ref, tok, [models.prompt_of(tok, r.prompt, r.raw) for r in qa],
                                      [r.completion for r in qa])
                (mu_Q * dq).backward()
                d_q = dq.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            mu = min(max(mu + oc["rho"] * (d_r.item() - rc["delta_R"]), 0.0), oc["mu_max"])
            if sep and rc.get("qa_weight") is not None:  # fixed weight instead of dual ascent
                mu_Q = rc["qa_weight"]
            elif sep:
                mu_Q = min(max(mu_Q + oc["rho"] * (d_q - rc["delta_Q"]), 0.0), oc["mu_max"])
            append_jsonl([{"block": s, "step": step, "L_P": 0.5 * float((g ** 2).sum()), "L_erase": float(l_erase),
                           "L_align": float(l_align), "L_flat": l_flat, "D_R": d_r.item(), "mu": mu,
                           **({"D_Q": d_q, "mu_Q": mu_Q} if sep else {})}], out / "steps.jsonl")
        print(f"block {s} done in {time.time() - t0:.0f}s, mu={mu:.3f}, D_R={d_r.item():.4f}")

    if labeler is not None:
        labeler.close()
    del ref
    models.free()
    model.to(torch.bfloat16).save_pretrained(out / "model", safe_serialization=True)
    tok.save_pretrained(out / "model")
