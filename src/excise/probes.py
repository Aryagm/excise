"""Behavior probes: does the masked model reproduce the teacher's exact
output? Self-match is strict (verbatim), label-free, and task-agnostic.

Probes exist because distribution-level KL is miscalibrated at high
sparsity — it can read healthy while generation quality collapses. Floors
must be decided by generating, not by losses.

Two implementations:

- `self_match` free-runs greedy decoding and compares decoded text. Used
  for final receipts and for non-exact match scopes.
- `tf_self_match` is the teacher-forced shortcut for scope="exact": a
  greedy rollout reproduces the cached target iff, under teacher forcing on
  that target, the argmax at every output position equals the target token
  (induction on the shared prefix). One batched forward instead of up to
  max_new_tokens sequential decode steps — cheap enough to probe a dev
  split AND an unmasked drift check at every probe step.
"""

import torch

from .data import prompt_batch
from .teacher import _generate


def _norm(text: str, scope: str) -> str:
    text = text.strip()
    return text.split("\n")[0].strip() if scope == "first_line" else text


@torch.no_grad()
def self_match(model, tok, examples, hooks, mode, mask=None, bs=8,
               max_new_tokens=128, use_adapter=True, scope="exact"):
    if hooks is not None:
        hooks.mode, hooks.mask = mode, mask
    device = next(model.parameters()).device
    tok.padding_side = "left"
    was_training = model.training
    model.eval()
    # Sorting by target length keeps the per-chunk generation cap tight.
    examples = sorted(examples, key=lambda e: len(e["out_ids"]))
    hit = 0
    for i in range(0, len(examples), bs):
        chunk = examples[i: i + bs]
        ids, attn, P = prompt_batch(chunk, tok.pad_token_id, device)
        # A faithful rollout never needs more steps than the longest target
        # in the chunk (targets ending in eos stop themselves earlier).
        cap = min(max_new_tokens, max(len(e["out_ids"]) for e in chunk))
        if use_adapter or not hasattr(model, "disable_adapter"):
            gen = model.generate(input_ids=ids, attention_mask=attn,
                                 max_new_tokens=cap,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        else:
            gen = _generate(model, ids, attn, cap, tok.pad_token_id)
        for j, e in enumerate(chunk):
            txt = tok.decode(gen[j, P:], skip_special_tokens=True)
            hit += _norm(txt, scope) == _norm(e["out_text"], scope)
    if hooks is not None:
        hooks.mode, hooks.mask = "off", None
    if was_training:
        model.train()
    return hit / len(examples)


@torch.no_grad()
def tf_self_match(model, batches, hooks, mode, mask=None) -> float:
    """Fraction of examples whose greedy rollout would reproduce the cached
    target verbatim, computed by teacher-forced argmax agreement. Equivalent
    to `self_match(..., scope="exact")` up to batched-decode numeric noise;
    `batches` are pre-collated (ids, labs, attn, pos, P, O) tuples."""
    hooks.mode, hooks.mask = mode, mask
    was_training = model.training
    model.eval()
    hit, n = 0, 0
    for ids, labs, attn, pos, P, O in batches:
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                       logits_to_keep=O + 1).logits
        lab = labs[:, P:]
        ok = (logits.argmax(-1)[:, :-1] == lab) | (lab == -100)
        hit += ok.all(-1).sum().item()
        n += lab.shape[0]
    hooks.mode, hooks.mask = "off", None
    if was_training:
        model.train()
    return hit / n
