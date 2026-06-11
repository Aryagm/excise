"""fc iteration 2: attack the train/held generalization gap with data.

fc4b (v0.1) trained on ONE BFCL split capped at 400 rows -> 320 train
prompts, ~62 epochs, ~20pt train/held gap, unmasked drift to 86.25%.
Diagnosis: memorization + a guardrail that could not see held-out drift.

This run: every single-turn BFCL v4 split (~2k rows), the multi-tool loader
fix (v0.1 silently kept only function[0] on multi-tool rows), distractor-
tool augmentation (1-2 unrelated schemas shuffled into each prompt, so the
kept channels must carry tool SELECTION, not just argument filling — free,
because the teacher is label-free), plus the v0.2 anchor guardrail.

v0.1 reference: held 77.5% @ 50% channels, 76.3% @ 40%; unmasked 86.25%.
"""

import json
import random
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from excise import ExtractConfig, extract

MODEL = "Qwen/Qwen3-4B"
BASE = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
        "berkeley-function-call-leaderboard/bfcl_eval/data/")
SPLITS = ["BFCL_v4_simple_python.json", "BFCL_v4_simple_java.json",
          "BFCL_v4_simple_javascript.json", "BFCL_v4_live_simple.json",
          "BFCL_v4_multiple.json", "BFCL_v4_live_multiple.json"]


def wrap(fn):
    params = json.loads(json.dumps(fn.get("parameters", {}))
                        .replace('"dict"', '"object"'))
    return {"type": "function", "function": {
        "name": fn["name"], "description": fn.get("description", ""),
        "parameters": params}}


def main(outdir):
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(0)

    rows = []
    for split in SPLITS:
        path = Path(split)
        if not path.exists():
            try:
                urllib.request.urlretrieve(BASE + split, path)
            except Exception as e:                      # noqa: BLE001
                print(f"[fc] skip {split}: {e}")
                continue
        n0 = len(rows)
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        print(f"[fc] {split}: +{len(rows) - n0} rows")

    # global tool pool for distractor augmentation
    pool = []
    for r in rows:
        fns = r["function"] if isinstance(r["function"], list) else [r["function"]]
        pool.extend(wrap(fn) for fn in fns)

    prompts = []
    for r in rows:
        fns = r["function"] if isinstance(r["function"], list) else [r["function"]]
        tools = [wrap(fn) for fn in fns]            # full list (v0.1 kept [0])
        names = {t["function"]["name"] for t in tools}
        distractors = []
        while len(distractors) < rng.randint(1, 2):
            cand = pool[rng.randrange(len(pool))]
            if cand["function"]["name"] not in names:
                distractors.append(cand)
                names.add(cand["function"]["name"])
        tools = tools + distractors
        rng.shuffle(tools)
        q = r["question"]
        while isinstance(q, list):
            q = q[0]
        msgs = [q] if isinstance(q, dict) else [{"role": "user",
                                                 "content": str(q)}]
        try:
            text = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                           add_generation_prompt=True)
        prompts.append(text)
    print(f"[fc] {len(prompts)} prompts built (pool={len(pool)} tools)")

    from datasets import load_dataset
    ws = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="test")
    anchors = [t.strip() for t in ws["text"] if len(t.split()) > 60][:48]
    for g in ["Explain why the sky is blue.", "Write a haiku about autumn.",
              "What is the capital of Australia?",
              "Summarize Romeo and Juliet in two sentences.",
              "Give three tips for learning a new language.",
              "Describe how photosynthesis works."]:
        anchors.append(tok.apply_chat_template(
            [{"role": "user", "content": g}], tokenize=False,
            add_generation_prompt=True))

    t0 = time.time()
    result = extract(
        MODEL, prompts=prompts, tokenizer=tok,
        config=ExtractConfig(batch_size=8, probe_below=0.7,
                             eval_batch_size=16, anchor_texts=anchors),
    )
    out = result.save(outdir)
    (out / "report.txt").write_text(result.report())
    r = result.receipts
    summary = {
        "task": "fc_v02", "n_prompts": len(prompts),
        "floor": result.floor, "floor_reason": result.floor_reason,
        "frontier": result.frontier,
        "probe_base": r["probe_base"],
        "base_self_match": r["base_self_match"],
        "unmasked_self_match": r["unmasked_self_match"],
        "random_mask_self_match": r["random_mask_self_match"],
        "steps": r["steps"], "wall_minutes": r["wall_minutes"],
        "vocab_support_size": r["vocab_support_size"],
        "total_minutes": (time.time() - t0) / 60,
        "v01_reference": {"held_at_50": 0.775, "held_at_40": 0.763,
                          "unmasked": 0.8625, "train_prompts": 320},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("==== SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/fc_v02")
    main(ap.parse_args().out)
