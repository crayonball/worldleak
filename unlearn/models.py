"""Model loading, chat formatting, sampling, and likelihood / KL utilities."""

from __future__ import annotations

import gc

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(name: str, revision: str | None = None, device: str = "cuda", trainable: bool = False,
         tokenizer: str | None = None, dtype=torch.bfloat16):
    """tokenizer: fallback tokenizer repo for checkpoints that ship without tokenizer files.
    dtype: weight precision. Small AdamW steps mostly round away on bf16 weights; train in float32 to avoid that."""
    try:
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
    except (OSError, ValueError):
        if tokenizer is None:
            raise
        tok = AutoTokenizer.from_pretrained(tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype).to(device)
    model.train(trainable)
    model.requires_grad_(trainable)
    if trainable:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model, tok


def free() -> None:
    """Release CUDA memory after the caller has dropped its model references."""
    gc.collect()
    torch.cuda.empty_cache()


def prompt_of(tok, text: str, raw: bool = False) -> str:
    """Raw text (pretraining-style continuation) is used as is; everything else gets the chat / QA format."""
    return text if raw else format_prompt(tok, text)


def format_prompt(tok, prompt: str, system: str | None = None) -> str:
    """Chat template when available (Instruct / Zephyr); plain QA format for base models."""
    if tok.chat_template:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"Question: {prompt}\nAnswer:"


@torch.no_grad()
def generate(model, tok, prompts: list[str], n: int, max_new_tokens: int,
             temperature: float = 0.7, top_p: float = 0.9, batch_size: int = 16,
             do_sample: bool = True) -> list[list[str]]:
    """n samples per prompt; prompts are already chat-formatted."""
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    out: list[list[str]] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None, top_p=top_p if do_sample else None,
            num_return_sequences=n, pad_token_id=tok.pad_token_id,
        )
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out.extend(texts[j * n:(j + 1) * n] for j in range(len(chunk)))
    model.train(was_training)
    model.config.use_cache = not was_training
    return out


def _encode_pairs(tok, prompts: list[str], completions: list[str], max_len: int, device):
    """Right-padded prompt+completion ids with a mask over completion tokens only."""
    ids, masks = [], []
    for p, c in zip(prompts, completions):
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        c_ids = tok(c, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        seq = (p_ids + c_ids)[:max_len]
        m = ([0] * len(p_ids) + [1] * len(c_ids))[:max_len]
        ids.append(seq)
        masks.append(m)
    L = max(map(len, ids))
    input_ids = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(ids), L), dtype=torch.long)
    comp = torch.zeros((len(ids), L), dtype=torch.float)
    for i, (s, m) in enumerate(zip(ids, masks)):
        input_ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        comp[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return input_ids.to(device), attn.to(device), comp.to(device)


def mean_token_logprob(model, tok, prompts: list[str], completions: list[str], max_len: int = 1024,
                       reduce: str = "mean") -> torch.Tensor:
    """u = (1/|y|) sum_l log P(y_l | q, y_<l) (reduce="sum" drops the 1/|y|); differentiable with grads on."""
    input_ids, attn, comp = _encode_pairs(tok, prompts, completions, max_len, model.device)
    logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float()
    lp = torch.gather(F.log_softmax(logits, -1), 2, input_ids[:, 1:, None]).squeeze(-1)
    mask = comp[:, 1:]
    total = (lp * mask).sum(1)
    return total if reduce == "sum" else total / mask.sum(1).clamp_min(1)


def retain_kl(model, ref_model, tok, prompts: list[str], completions: list[str], max_len: int = 512,
              ref_prompts: list[str] | None = None) -> torch.Tensor:
    """D_R = E_r KL(P_in(.|r) || P_Delta(.|r)), averaged over retain completion positions.
    ref_prompts: prompts for the reference when they differ from the policy's (in-context unlearning)."""
    input_ids, attn, comp = _encode_pairs(tok, prompts, completions, max_len, model.device)
    lp = F.log_softmax(model(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float(), -1)
    if ref_prompts is None:
        with torch.no_grad():
            ref_lp = F.log_softmax(ref_model(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float(), -1)
        kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1)
        mask = comp[:, 1:]
        return ((kl * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    r_ids, r_attn, r_comp = _encode_pairs(tok, ref_prompts, completions, max_len, model.device)
    with torch.no_grad():
        ref_lp = F.log_softmax(ref_model(input_ids=r_ids, attention_mask=r_attn).logits[:, :-1].float(), -1)
    out = []
    for i in range(len(completions)):  # same completion tokens at different offsets
        a, b = lp[i][comp[i, 1:].bool()], ref_lp[i][r_comp[i, 1:].bool()]
        assert len(a) == len(b), "completion truncated differently"
        out.append((b.exp() * (b - a)).sum(-1).mean())
    return torch.stack(out).mean()
