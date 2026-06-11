"""Label-free teacher: the unmasked base model's own greedy outputs, and a
sparse top-k cache of its full distribution at every output position.

Caching once removes the per-epoch teacher forward (~30% of step compute in
the naive loop). The cache keeps the top-k probabilities UN-renormalized
plus the residual off-support mass, so the KL can be computed as a proper
binned KL (zero at student == teacher). Renormalizing instead — as early
versions did — makes the loss read -log(coverage) at the optimum and puts a
positive gradient on every off-support token, which on diffuse-output tasks
actively pushes the student to sharpen onto the support; that bias was the
dominant source of unmasked drift on the JSON run.
"""

import torch
import torch.nn.functional as F

from .data import prompt_batch


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
    """One forward pass per batch; store top-k teacher probs (un-renormalized)
    and the residual off-support mass at output positions."""
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
        r = (1.0 - p.sum(-1)).clamp_min(0.0)
        cache.append((p, ix, r))
    return cache


def out_logprobs(logits, labs, P):
    """Float32 log-softmax over the output positions, shared by the KL and
    CE terms so the full-vocab softmax is materialized once per step."""
    m = labs[:, P:] != -100
    return F.log_softmax(logits[:, :-1][m].float(), -1), m


def sparse_kl(s_lp, tp, ti, tr):
    """Binned forward KL(teacher || student): top-k support plus one residual
    bucket. Zero iff the student matches the teacher on the support and in
    total off-support mass. Both log arguments are clamped: late in training
    the student can put ~all mass on the support and 1 - s_mass underflows
    to <= 0 in float32."""
    s_at = s_lp.gather(-1, ti)
    support = (tp * (tp.clamp_min(1e-9).log() - s_at)).sum(-1)
    s_mass = s_at.exp().sum(-1)
    bucket = tr * (tr.clamp_min(1e-9).log()
                   - (1.0 - s_mass).clamp_min(1e-9).log())
    return (support + bucket).mean()
