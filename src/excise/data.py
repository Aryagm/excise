"""Prompt loading and batching.

Collate layout: prompts left-padded so every output starts at the same
column, outputs right-padded. This lets training use ``logits_to_keep`` and
never materialize full-vocabulary logits over long prompts.
"""

import json
from pathlib import Path

import torch


def load_prompts(source) -> list[str]:
    """Accepts a list of strings, a .txt path (one prompt per line), or a
    .jsonl path with {"prompt": ...} rows."""
    if isinstance(source, (list, tuple)):
        return [str(p) for p in source]
    path = Path(source)
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return [r["prompt"] for r in rows]
    return [l for l in path.read_text().splitlines() if l.strip()]


def collate(batch: list[dict], pad_id: int, device) -> tuple:
    """batch rows: {"prompt_ids": [...], "out_ids": [...]}"""
    P = max(len(b["prompt_ids"]) for b in batch)
    O = max(len(b["out_ids"]) for b in batch)
    n = len(batch)
    ids = torch.full((n, P + O), pad_id, dtype=torch.long)
    labs = torch.full((n, P + O), -100, dtype=torch.long)
    attn = torch.zeros((n, P + O), dtype=torch.long)
    for i, b in enumerate(batch):
        p, o = b["prompt_ids"], b["out_ids"]
        ids[i, P - len(p): P] = torch.tensor(p)
        ids[i, P: P + len(o)] = torch.tensor(o)
        labs[i, P: P + len(o)] = torch.tensor(o)
        attn[i, P - len(p): P + len(o)] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    return (ids.to(device), labs.to(device), attn.to(device), pos.to(device),
            P, O)


def prompt_batch(examples: list[dict], pad_id: int, device) -> tuple:
    """Left-padded prompt-only batch for generation."""
    P = max(len(e["prompt_ids"]) for e in examples)
    ids = torch.full((len(examples), P), pad_id, dtype=torch.long)
    attn = torch.zeros((len(examples), P), dtype=torch.long)
    for j, e in enumerate(examples):
        ids[j, P - len(e["prompt_ids"]):] = torch.tensor(e["prompt_ids"])
        attn[j, P - len(e["prompt_ids"]):] = 1
    return ids.to(device), attn.to(device), P
