"""Label-free teacher: the unmasked base model's own greedy outputs, and a
sparse top-k cache of its full distribution at every output position.

Caching once removes the per-epoch teacher forward (~30% of step compute in
the naive loop) and is exact up to top-k truncation.
"""

import torch
import torch.nn.functional as F

from .data import collate, prompt_batch


@torch.no_grad()
def generate_targets(model, tok, examples, hooks, max_new_tokens, bs=8):
    """Greedy unmasked outputs become each example's distillation target."""
    hooks.mode = "off"
    device = next(model.parameters()).device
    tok.padding_side = "left"
    for i in range(0, len(examples), bs):
        chunk = examples[i: i + bs]
        ids, attn, P = prompt_batch(chunk, tok.pad_token_id, device)
        gen = _generate(model, ids, attn, max_new_tokens, tok.pad_token_id)
        for j, e in enumerate(chunk):
            out = gen[j, P:]
            if tok.eos_token_id in out:
                out = out[: (out == tok.eos_token_id).nonzero()[0, 0] + 1]
            e["out_ids"] = out.tolist()
            e["out_text"] = tok.decode(out, skip_special_tokens=True).strip()
    return examples


def _generate(model, ids, attn, max_new_tokens, pad_id):
    kw = dict(input_ids=ids, attention_mask=attn,
              max_new_tokens=max_new_tokens, do_sample=False,
              pad_token_id=pad_id)
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return model.generate(**kw)
    return model.generate(**kw)


@torch.no_grad()
def cache_distributions(model, hooks, batches, topk):
    """One forward pass per batch; store renormalized top-k teacher probs at
    output positions."""
    hooks.mode = "off"
    cache = []
    for ids, labs, attn, pos, P, O in batches:
        if hasattr(model, "disable_adapter"):
            ctx = model.disable_adapter()
        else:
            ctx = torch.no_grad()
        with ctx:
            logits = model(input_ids=ids, attention_mask=attn,
                           position_ids=pos, logits_to_keep=O + 1).logits
        m = labs[:, P:] != -100
        probs = F.softmax(logits[:, :-1][m].float(), -1)
        p, ix = torch.topk(probs, min(topk, probs.shape[-1]), dim=-1)
        cache.append((p / p.sum(-1, keepdim=True), ix))
    return cache


def sparse_kl(student_logits, labs, P, tp, ti):
    """Forward KL(teacher || student) on the cached top-k support."""
    m = labs[:, P:] != -100
    s_lp = F.log_softmax(student_logits[:, :-1][m].float(), -1)
    return (tp * (tp.clamp_min(1e-9).log() - s_lp.gather(-1, ti))).sum(-1).mean()
