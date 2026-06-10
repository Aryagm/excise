"""Behavior probes: does the masked model reproduce the teacher's exact
output? Self-match is strict (verbatim), label-free, and task-agnostic.

Probes exist because distribution-level KL is miscalibrated at high
sparsity — it can read healthy while generation quality collapses. Floors
must be decided by generating, not by losses.
"""

import torch

from .data import prompt_batch
from .teacher import _generate


@torch.no_grad()
def self_match(model, tok, examples, hooks, mode, mask=None, bs=8,
               max_new_tokens=128, use_adapter=True):
    hooks.mode, hooks.mask = mode, mask
    device = next(model.parameters()).device
    tok.padding_side = "left"
    was_training = model.training
    model.eval()
    hit = 0
    for i in range(0, len(examples), bs):
        chunk = examples[i: i + bs]
        ids, attn, P = prompt_batch(chunk, tok.pad_token_id, device)
        if use_adapter or not hasattr(model, "disable_adapter"):
            gen = model.generate(input_ids=ids, attention_mask=attn,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        else:
            gen = _generate(model, ids, attn, max_new_tokens,
                            tok.pad_token_id)
        for j, e in enumerate(chunk):
            txt = tok.decode(gen[j, P:], skip_special_tokens=True).strip()
            hit += txt == e["out_text"]
    hooks.mode, hooks.mask = "off", None
    if was_training:
        model.train()
    return hit / len(examples)
