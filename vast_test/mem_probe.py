"""Replicate run_json's extract() stages with CUDA memory logging to find
what holds memory at the attribution OOM. Run on the instance."""

import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from excise.arch import find_mlps
from excise.config import ExtractConfig
from excise.data import collate
from excise.hooks import GateHooks
from excise.teacher import cache_distributions, generate_targets, out_logprobs


def mem(tag):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    print(f"[mem] {tag:40s} alloc={a:6.2f}GiB reserved={r:6.2f}GiB",
          flush=True)


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = ExtractConfig(max_new_tokens=64, batch_size=24, max_prompt_tokens=128)

NAMES = ["John", "Maria", "Wei", "Aisha"]
rng = random.Random(0)
prompts = []
for i in range(600):
    n = rng.choice(NAMES)
    text = f"{n} is a {rng.randint(19, 79)}-year-old teacher in Lima."
    msgs = [{"role": "user",
             "content": "Extract name, age, job, and city as a JSON object. "
                        f"Reply with only the JSON.\n\n{text}"}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True))

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).cuda()
mem("model loaded")
model = get_peft_model(model, LoraConfig(
    r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
    target_modules="all-linear", task_type="CAUSAL_LM"))
mem("peft wrapped")

mlp_map = find_mlps(model)
hooks = GateHooks(mlp_map)
examples = []
for p in prompts:
    ids = tok(p, add_special_tokens=False)["input_ids"]
    if 0 < len(ids) <= cfg.max_prompt_tokens:
        examples.append({"prompt": p, "prompt_ids": ids})
generate_targets(model, tok, examples, hooks, cfg.max_new_tokens,
                 bs=cfg.batch_size)
examples = [e for e in examples if e["out_ids"]]
mem("targets generated")

train = examples[:416]
train.sort(key=lambda e: len(e["prompt_ids"]))
batches = [collate(train[i: i + cfg.batch_size], tok.pad_token_id, "cuda")
           for i in range(0, len(train), cfg.batch_size)]
mem("batches collated")
cache = cache_distributions(model, hooks, batches, cfg.teacher_topk)
mem("teacher cached")

# ---- attribution replica with per-iteration logging
import math
emb = model.get_input_embeddings().weight
emb.requires_grad_(True)
scores = torch.zeros(mlp_map.n_layers, mlp_map.d_ff, dtype=torch.float32,
                     device="cuda")
n_b = max(1, math.ceil(cfg.attr_examples / cfg.batch_size))
sel = batches[:: max(1, len(batches) // n_b)][:n_b]
print(f"attr: {len(sel)} batches, shapes "
      f"{[tuple(b[0].shape) for b in sel[:4]]}", flush=True)
for it, (ids, labs, attn, pos, P, O) in enumerate(sel):
    micro = max(1, min(ids.shape[0], 2048 // max(1, ids.shape[1])))
    sl = slice(0, micro)
    hooks.mode, hooks.captured = "capture", {}
    with model.disable_adapter(), torch.enable_grad():
        logits = model(input_ids=ids[sl], attention_mask=attn[sl],
                       position_ids=pos[sl], logits_to_keep=O + 1).logits
        mem(f"attr it{it} fwd done (micro={micro}, seq={ids.shape[1]})")
        lp, m = out_logprobs(logits, labs[sl], P)
        (-lp.gather(-1, labs[sl][:, P:][m].unsqueeze(-1)).mean()).backward()
    mem(f"attr it{it} bwd done")
    for li, x in hooks.captured.items():
        if x.grad is not None:
            scores[li] += (x.grad.float() * x.float()).abs().sum((0, 1))
    model.zero_grad(set_to_none=True)
    hooks.captured = {}
    mem(f"attr it{it} cleared")
    if it >= 3:
        break
print("OK")
