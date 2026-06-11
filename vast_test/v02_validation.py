"""v0.2 GPU validation battery.

Three stages, run with --task {arith,json,all}:

  arith  — dogfood regression: 2-digit addition on Qwen2.5-Math-1.5B with
           the same config as the validated v0.1 battery. Compares floor /
           recovery / drift, then exercises export_sliced with vocabulary
           pruning and benchmarks decode throughput (base vs sliced vs
           sliced+vocab-pruned).
           v0.1 reference: floor 2.94%, held ~95% at floor, 1.54B -> 0.42B.

  json   — the v0.1 failure case: structured JSON extraction on
           Qwen2.5-1.5B-Instruct. v0.1 reference: floor 33.9%, held 90% at
           floor, UNMASKED DRIFT to 55.8% self-match (guardrail blind on
           held-out + missing in polish + renormalized-KL sharpening bias).
           Reruns with the bucketed KL, wikitext anchor guardrail, polish
           guardrail, and probe_base-calibrated dev-split floor detection.

Results land in out/<task>/ as receipts.json + report.txt + summary.json.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from excise import ExtractConfig, extract, param_count


def save_all(result, out_dir, summary):
    out = result.save(out_dir)
    (out / "report.txt").write_text(result.report())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("==== SUMMARY", json.dumps(summary))
    print(result.report())
    return out


@torch.no_grad()
def bench_decode(model, tok_or_ids, new_tokens=64, batch_sizes=(1, 8, 64),
                 pad_id=0):
    """Greedy decode throughput (tokens/sec), warmed up, generate-loop only.
    `tok_or_ids` is a [1, P] prompt id tensor (already remapped if needed)."""
    model.eval()
    device = next(model.parameters()).device
    res = {}
    for bs in batch_sizes:
        ids = tok_or_ids.repeat(bs, 1).to(device)
        attn = torch.ones_like(ids)
        kw = dict(input_ids=ids, attention_mask=attn,
                  max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                  do_sample=False, pad_token_id=pad_id)
        model.generate(**kw)                       # warmup
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            model.generate(**kw)
        torch.cuda.synchronize()
        res[f"bs{bs}"] = round(3 * bs * new_tokens / (time.time() - t0), 1)
    return res


def run_arith(outdir, max_steps=3000, seed=42, min_target=0.008):
    from transformers import AutoTokenizer
    prompts = [f"{a} + {b} =" for a in range(10, 100) for b in range(10, 100)]
    t0 = time.time()
    result = extract(
        "Qwen/Qwen2.5-Math-1.5B",
        prompts=prompts,
        config=ExtractConfig(max_new_tokens=4, batch_size=64, kl_budget=0.02,
                             probe_below=0.12, max_prompt_tokens=32,
                             probe_n=128, max_steps=max_steps, seed=seed,
                             min_target=min_target),
    )
    r = result.receipts
    summary = {
        "task": "arith",
        "floor": result.floor, "floor_reason": result.floor_reason,
        "frontier": result.frontier,
        "probe_base": r["probe_base"],
        "base_self_match": r["base_self_match"],
        "unmasked_self_match": r["unmasked_self_match"],
        "random_mask_self_match": r["random_mask_self_match"],
        "steps": r["steps"], "wall_minutes": r["wall_minutes"],
        "vocab_support_size": r["vocab_support_size"],
    }

    # --- export: slice, then vocab-prune; verify both; bench all three
    tok = result._tok
    bench_ids = tok("12 + 34 =", return_tensors="pt")["input_ids"]
    pad = tok.pad_token_id

    before = param_count(result._model)
    sliced = result.export_sliced(prune_vocabulary=True, verify_n=128)
    after = param_count(sliced)
    summary.update({
        "params_before": before, "params_after_slice_prune": after,
        "sliced_self_match": result.receipts.get("sliced_self_match"),
        "pruned_self_match": result.receipts.get("pruned_self_match"),
    })

    keep = torch.as_tensor(result.vocab_support)
    old2new = torch.full((len(tok),), -1, dtype=torch.long)
    old2new[keep] = torch.arange(len(keep))
    pruned_pad = int(old2new[pad]) if old2new[pad] >= 0 else 0
    summary["bench_sliced_pruned"] = bench_decode(
        sliced, old2new[bench_ids], pad_id=pruned_pad)

    out = result.save(outdir)              # receipts + sliced artifact
    (out / "report.txt").write_text(result.report())
    del sliced, result
    torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B", dtype=torch.bfloat16).cuda()
    summary["params_base"] = param_count(base)
    summary["bench_base"] = bench_decode(base, bench_ids, pad_id=pad)
    del base
    torch.cuda.empty_cache()
    summary["total_minutes"] = (time.time() - t0) / 60
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("==== SUMMARY", json.dumps(summary))


def run_json(outdir, diverse=False):
    from transformers import AutoTokenizer
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)

    NAMES = ["John", "Maria", "Wei", "Aisha", "Carlos", "Yuki", "Omar",
             "Elena", "Raj", "Sofia", "Liam", "Nadia", "Pedro", "Hana",
             "Igor", "Zara"]
    CITIES = ["Paris", "Tokyo", "Lagos", "Lima", "Oslo", "Delhi", "Cairo",
              "Seoul", "Quito", "Porto", "Hanoi", "Perth"]
    JOBS = ["teacher", "engineer", "nurse", "chef", "pilot", "farmer",
            "artist", "lawyer"]
    if diverse:
        # diversity test: same capability, 5x the prompts, varied surface
        # forms — names/cities/jobs pools widened, several sentence
        # templates with distractor clauses, varied instruction phrasings.
        NAMES += ["Anya Petrova", "Jamal", "Chen Wei", "Lucia", "Tomás",
                  "Ingrid", "Kofi", "Mei-Ling", "Dmitri", "Fatima",
                  "Oluwaseun", "Birgit", "Ravi", "Esperanza", "Kenji",
                  "Astrid", "Mateo", "Priya", "Sven", "Amara", "Hugo",
                  "Noor", "Kasper", "Imani"]
        CITIES += ["Reykjavik", "Montevideo", "Kathmandu", "Windhoek",
                   "Tbilisi", "Da Nang", "Cusco", "Galway", "Sapporo",
                   "Bergen", "Marrakesh", "Valparaiso", "Tallinn",
                   "Chiang Mai", "Antigua", "Brno", "Kampala", "Yerevan"]
        JOBS += ["software developer", "midwife", "carpenter", "barista",
                 "geologist", "translator", "electrician", "librarian",
                 "fisherman", "architect", "paramedic", "violinist"]
    TEMPLATES = [
        "{n} is a {a}-year-old {j} living in {c}.",
        "{n}, {a}, works as a {j} in {c}.",
        "At {a} years old, {n} has built a career as a {j}; home is {c}.",
        "{n} moved to {c} years ago and now works there as a {j}. "
        "{n} just turned {a}.",
        "Meet {n}: {j} by trade, {c} resident, age {a}.",
        "Despite the long hours, {n} ({a}) still loves being a {j}. "
        "Friends visiting {c} often stay over.",
        "The {j} who fixed our problem was {n} from {c} — {a} years old, "
        "apparently.",
        "{n} has lived in {c} since 2019, works as a {j}, and recently "
        "celebrated turning {a}.",
    ]
    INSTRUCTIONS = [
        "Extract name, age, job, and city as a JSON object. "
        "Reply with only the JSON.",
        "Return a JSON object with keys name, age, job, city. "
        "Output only JSON.",
        "Pull out the person's name, age, job and city. "
        "Answer as a single JSON object, nothing else.",
        "From the text below, produce JSON with fields name, age, "
        "job, city. JSON only.",
    ]
    rng = random.Random(0)
    prompts = []
    n_prompts = 3000 if diverse else 600
    for _ in range(n_prompts):
        n, c, j = rng.choice(NAMES), rng.choice(CITIES), rng.choice(JOBS)
        a = rng.randint(19, 79)
        tmpl = rng.choice(TEMPLATES) if diverse else TEMPLATES[0]
        instr = rng.choice(INSTRUCTIONS) if diverse else INSTRUCTIONS[0]
        text = tmpl.format(n=n, a=a, j=j, c=c)
        msgs = [{"role": "user", "content": f"{instr}\n\n{text}"}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))

    # off-task anchors for the guardrail: plain encyclopedic text + a few
    # generic chat-formatted instructions
    from datasets import load_dataset
    ws = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="test")
    anchors = [t.strip() for t in ws["text"] if len(t.split()) > 60][:48]
    generic = ["Explain why the sky is blue.",
               "Write a haiku about autumn.",
               "What is the capital of Australia?",
               "Summarize the plot of Romeo and Juliet in two sentences.",
               "Give three tips for learning a new language.",
               "What does HTTP stand for?",
               "Describe how photosynthesis works.",
               "List four prime numbers between 10 and 30."]
    for gtext in generic:
        msgs = [{"role": "user", "content": gtext}]
        anchors.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))

    result = extract(
        MODEL,
        prompts=prompts,
        tokenizer=tok,
        config=ExtractConfig(max_new_tokens=64, batch_size=24,
                             kl_budget=0.025, probe_below=0.6,
                             max_prompt_tokens=128,
                             anchor_texts=anchors),
    )
    r = result.receipts
    summary = {
        "task": "json_diverse" if diverse else "json",
        "n_prompts": n_prompts,
        "floor": result.floor, "floor_reason": result.floor_reason,
        "frontier": result.frontier,
        "probe_base": r["probe_base"],
        "base_self_match": r["base_self_match"],
        "unmasked_self_match": r["unmasked_self_match"],
        "random_mask_self_match": r["random_mask_self_match"],
        "steps": r["steps"], "wall_minutes": r["wall_minutes"],
        "vocab_support_size": r["vocab_support_size"],
        "guardrail_trace_tail": r["guardrail_trace"][-8:],
        "v01_reference": {"floor": 0.339, "held_at_floor": 0.90,
                          "unmasked_drift": 0.558},
    }
    before = param_count(result._model)
    sliced = result.export_sliced(prune_vocabulary=True, verify_n=128)
    summary.update({
        "params_before": before,
        "params_after_slice_prune": param_count(sliced),
        "sliced_self_match": result.receipts.get("sliced_self_match"),
        "pruned_self_match": result.receipts.get("pruned_self_match"),
    })
    save_all(result, outdir, summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",
                    choices=["arith", "arith-long", "arith-deep", "json",
                             "json-diverse", "all"],
                    default="all")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.task in ("arith", "all"):
        run_arith(Path(args.out) / "arith_v02", seed=args.seed)
    if args.task == "arith-long":
        run_arith(Path(args.out) / f"arith_v02_long_s{args.seed}",
                  max_steps=6000, seed=args.seed)
    if args.task == "arith-deep":
        # below the previous min_target floor: where does the capability
        # actually stop compressing?
        run_arith(Path(args.out) / f"arith_v02_deep_s{args.seed}",
                  max_steps=9000, seed=args.seed, min_target=0.002)
    if args.task in ("json", "all"):
        run_json(Path(args.out) / "json_v02")
    if args.task == "json-diverse":
        run_json(Path(args.out) / "json_diverse", diverse=True)
