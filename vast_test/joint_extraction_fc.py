#!/usr/bin/env python3
"""4B function-calling verification: label-free joint capability extraction.

Tests whether the v2.1 joint method (validated on arithmetic) works on a
real-world structured-generation capability: single-turn function calling
(BFCL v4 simple-python prompts) on Qwen3-4B.

Fully label-free: the teacher is the model's own unmasked greedy tool call.
Recovery = exact self-match (masked output == unmasked base output), which is
STRICTER than the paper's BFCL exact-match metric.

Paper context (Qwen3-8B): raw mask 19.1% recovery at 36.2% MLP; best
post-attribution collimator 84.6% at the same substrate.
"""

import argparse
import json
import math
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B"
DATA_URL = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
            "berkeley-function-call-leaderboard/bfcl_eval/data/"
            "BFCL_v4_simple_python.json")
DATA_PATH = "bfcl_simple.json"
DEVICE = "cuda"
DTYPE = torch.bfloat16

MAX_EXAMPLES = 400
MAX_PROMPT_TOKENS = 1024
MAX_NEW_TOKENS = 128
HELD_FRAC = 0.20

MAX_STEPS = 2500
WARMUP_STEPS = 100
BATCH_SIZE = 8
LORA_LR = 1e-4
GATE_LR = 5e-2
DUAL_LR = 0.1
KL_BUDGET = 0.025
TARGET_DECAY = 0.994
MIN_TARGET = 0.03
GUARDRAIL_EVERY = 4
GUARDRAIL_W = 1.0
POLISH_STEPS = 100
PROBE_EVERY = 150
PROBE_BELOW = 0.70        # fn-calling substrate expected much larger than math
PROBE_N = 64
PROBE_TOL = 0.08          # floor when self-match < 1 - tol, twice in a row
CE_WEIGHT = 0.05
TEACHER_TOPK = 128
EVAL_BUDGETS = [0.50, 0.40, 0.30, 0.20]

# ---------------------------------------------------------------- gates

class Gates(torch.nn.Module):
    GAMMA, ZETA, BETA = -0.1, 1.1, 2.0 / 3.0

    def __init__(self, n_layers, d_ff):
        super().__init__()
        self.la = torch.nn.Parameter(torch.full((n_layers, d_ff), 3.0))

    def sample(self, li):
        la = self.la[li]
        u = torch.rand_like(la).clamp_(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (-u).log1p() + la) / self.BETA)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def p_open(self):
        shift = self.BETA * math.log(-self.GAMMA / self.ZETA)
        return torch.sigmoid(self.la - shift)

    def expected_open(self):
        return self.p_open().mean()

    def topk_mask(self, k_total):
        score = self.la.detach().flatten()
        idx = torch.topk(score, k_total).indices
        mask = torch.zeros_like(score)
        mask[idx] = 1.0
        return mask.view_as(self.la)


class MLPState:
    def __init__(self):
        self.mode = "off"
        self.mask = None
        self.gates = None
        self.sampled = None   # per-layer gate samples, drawn OUTSIDE the
                              # checkpointed region (in-region RNG breaks
                              # gradient-checkpoint recomputation)


def patch_mlps(model, state):
    layers = model.base_model.model.model.layers
    for li, layer in enumerate(layers):
        mlp = layer.mlp

        def fwd(x, mlp=mlp, li=li):
            inter = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
            if state.mode == "sample":
                inter = inter * state.sampled[li].to(inter.dtype)
            elif state.mode == "mask":
                inter = inter * state.mask[li].to(inter.dtype)
            return mlp.down_proj(inter)

        mlp.forward = fwd
    return len(layers)

# ---------------------------------------------------------------- data

def load_bfcl(tok):
    if not Path(DATA_PATH).exists():
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[:MAX_EXAMPLES]
    prompts = []
    for r in rows:
        fn = r["function"][0] if isinstance(r["function"], list) else r["function"]
        params = json.loads(json.dumps(fn.get("parameters", {}))
                            .replace('"dict"', '"object"'))
        tool = {"type": "function",
                "function": {"name": fn["name"],
                             "description": fn.get("description", ""),
                             "parameters": params}}
        q = r["question"]
        while isinstance(q, list):
            q = q[0]
        msgs = [q] if isinstance(q, dict) else [{"role": "user", "content": str(q)}]
        try:
            text = tok.apply_chat_template(msgs, tools=[tool], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tools=[tool], tokenize=False,
                                           add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= MAX_PROMPT_TOKENS:
            prompts.append({"id": r["id"], "prompt_ids": ids})
    return prompts


def collate(batch, pad_id):
    """Left-pad prompts so outputs start aligned; right-pad outputs."""
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
    return (ids.to(DEVICE), labs.to(DEVICE), attn.to(DEVICE), pos.to(DEVICE),
            P, O)

# ------------------------------------------------------------ teacher

@torch.no_grad()
def teacher_generate(model, tok, examples, state, bs=8):
    """Greedy unmasked base outputs = the label-free distillation target."""
    state.mode = "off"
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(examples), bs):
        chunk = examples[i: i + bs]
        P = max(len(e["prompt_ids"]) for e in chunk)
        ids = torch.full((len(chunk), P), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), P), dtype=torch.long)
        for j, e in enumerate(chunk):
            ids[j, P - len(e["prompt_ids"]):] = torch.tensor(e["prompt_ids"])
            attn[j, P - len(e["prompt_ids"]):] = 1
        with model.disable_adapter():
            gen = model.generate(input_ids=ids.to(DEVICE),
                                 attention_mask=attn.to(DEVICE),
                                 max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, e in enumerate(chunk):
            out = gen[j, P:]
            if tok.eos_token_id in out:
                out = out[: (out == tok.eos_token_id).nonzero()[0, 0] + 1]
            outs.append(out.tolist())
    for e, o in zip(examples, outs):
        e["out_ids"] = o
        e["out_text"] = tok.decode(o, skip_special_tokens=True).strip()
    return examples


@torch.no_grad()
def cache_teacher(model, state, batches):
    state.mode = "off"
    cache = []
    for ids, labs, attn, pos, P, O in batches:
        with model.disable_adapter():
            logits = model(input_ids=ids, attention_mask=attn,
                           position_ids=pos, logits_to_keep=O + 1).logits
        m = labs[:, P:] != -100                       # logits[:, t] predicts P+t
        probs = F.softmax(logits[:, :-1][m].float(), -1)
        p, ix = torch.topk(probs, TEACHER_TOPK, dim=-1)
        cache.append(((p / p.sum(-1, keepdim=True)), ix))
    return cache


def masked_kl(logits, labs, P, O, tp, ti):
    m = labs[:, P:] != -100
    s_lp = F.log_softmax(logits[:, :-1][m].float(), -1)
    return (tp * (tp.clamp_min(1e-9).log() - s_lp.gather(-1, ti))).sum(-1).mean()

# ---------------------------------------------------------- evaluation

@torch.no_grad()
def self_match(model, tok, examples, state, mode, mask=None, adapter=True, bs=8):
    state.mode, state.mask = mode, mask
    tok.padding_side = "left"
    model.eval()
    hit = 0
    for i in range(0, len(examples), bs):
        chunk = examples[i: i + bs]
        P = max(len(e["prompt_ids"]) for e in chunk)
        ids = torch.full((len(chunk), P), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), P), dtype=torch.long)
        for j, e in enumerate(chunk):
            ids[j, P - len(e["prompt_ids"]):] = torch.tensor(e["prompt_ids"])
            attn[j, P - len(e["prompt_ids"]):] = 1
        kw = dict(input_ids=ids.to(DEVICE), attention_mask=attn.to(DEVICE),
                  max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                  pad_token_id=tok.pad_token_id)
        if adapter:
            gen = model.generate(**kw)
        else:
            with model.disable_adapter():
                gen = model.generate(**kw)
        for j, e in enumerate(chunk):
            txt = tok.decode(gen[j, P:], skip_special_tokens=True).strip()
            if txt == e["out_text"]:
                hit += 1
    state.mode, state.mask = "off", None
    return hit / len(examples)

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fc4b")
    ap.add_argument("--seed", type=int, default=42)
    cfg = ap.parse_args()
    t0 = time.time()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, attn_implementation="sdpa").to(DEVICE)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    state = MLPState()
    n_layers = patch_mlps(model, state)
    d_ff = model.config.intermediate_size
    n_channels = n_layers * d_ff

    examples = load_bfcl(tok)
    print(f"[setup] {len(examples)} prompts; {n_layers}x{d_ff}="
          f"{n_channels} channels", flush=True)

    print("[teacher] generating unmasked greedy tool calls...", flush=True)
    examples = teacher_generate(model, tok, examples, state)
    has_call = sum("<tool_call>" in e["out_text"] for e in examples)
    print(f"[teacher] {has_call}/{len(examples)} outputs contain <tool_call>",
          flush=True)

    rng = random.Random(42)
    order = list(range(len(examples)))
    rng.shuffle(order)
    n_held = int(len(examples) * HELD_FRAC)
    held = [examples[i] for i in order[:n_held]]
    train = [examples[i] for i in order[n_held:]]
    train.sort(key=lambda e: len(e["prompt_ids"]))     # length-bucketed batches
    probe_set = train[:: max(1, len(train) // PROBE_N)][:PROBE_N]
    print(f"[setup] train={len(train)} held={len(held)}", flush=True)

    batches = [collate(train[i: i + BATCH_SIZE], tok.pad_token_id)
               for i in range(0, len(train), BATCH_SIZE)]
    cache = cache_teacher(model, state, batches)
    print(f"[teacher] cached {len(batches)} batches top{TEACHER_TOPK}", flush=True)

    gates = Gates(n_layers, d_ff).to(DEVICE)
    state.gates = gates
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": LORA_LR},
                             {"params": gates.parameters(), "lr": GATE_LR}],
                            weight_decay=0.0)

    lam, ema_kl, probe_dry = 0.0, 0.0, 0
    target = gates.expected_open().item()
    floor_reason = "max_steps"
    bi = 0
    border = list(range(len(batches)))
    model.train()

    for step in range(1, MAX_STEPS + 1):
        if bi == 0:
            rng.shuffle(border)
        ids, labs, attn, pos, P, O = batches[border[bi]]
        tp, ti = cache[border[bi]]
        bi = (bi + 1) % len(batches)

        # gate samples drawn here, once per step: the checkpointed layer
        # forwards must be deterministic for recomputation to match
        state.sampled = [gates.sample(li) for li in range(n_layers)]
        state.mode = "sample"
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                       logits_to_keep=O + 1).logits
        kl = masked_kl(logits, labs, P, O, tp, ti)
        m = labs[:, P:] != -100
        ce = -F.log_softmax(logits[:, :-1][m].float(), -1).gather(
            -1, labs[:, P:][m].unsqueeze(-1)).mean()
        loss = kl + CE_WEIGHT * ce + lam * gates.expected_open()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        state.mode = "off"
        state.sampled = None

        if step % GUARDRAIL_EVERY == 0:
            g_logits = model(input_ids=ids, attention_mask=attn,
                             position_ids=pos, logits_to_keep=O + 1).logits
            (GUARDRAIL_W * masked_kl(g_logits, labs, P, O, tp, ti)).backward()

        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()

        ema_kl = 0.9 * ema_kl + 0.1 * kl.item()
        open_now = gates.expected_open().item()
        if step > WARMUP_STEPS and ema_kl < KL_BUDGET and (open_now - target) < 0.05:
            target = max(MIN_TARGET, target * TARGET_DECAY)
        lam = max(0.0, lam + DUAL_LR * (open_now - target))

        if step % 50 == 0:
            print(f"[train] step={step} kl={kl.item():.4f} ema={ema_kl:.4f} "
                  f"open={open_now:.4f} target={target:.4f} lam={lam:.2f}",
                  flush=True)

        if open_now < PROBE_BELOW and step % PROBE_EVERY == 0:
            k_now = max(1, int(round(open_now * n_channels)))
            pmask = gates.topk_mask(k_now).to(DEVICE)
            pacc = self_match(model, tok, probe_set, state, "mask", pmask)
            model.train()
            probe_dry = probe_dry + 1 if pacc < 1.0 - PROBE_TOL else 0
            print(f"[probe] step={step} open={open_now:.4f} "
                  f"self_match={pacc:.4f} dry={probe_dry}", flush=True)
            if probe_dry >= 2:
                floor_reason = "probe"
                print(f"[floor] probe floor at step {step} open={open_now:.4f}",
                      flush=True)
                break
        if target <= MIN_TARGET and open_now <= MIN_TARGET * 1.5:
            floor_reason = "min_target"
            break

    floor_frac = gates.expected_open().item()
    k_floor = max(1, int(round(floor_frac * n_channels)))
    mask = gates.topk_mask(k_floor).to(DEVICE)
    for p in gates.parameters():
        p.requires_grad_(False)
    print(f"[polish] {POLISH_STEPS} steps at floor={floor_frac:.4f}", flush=True)
    for _ in range(POLISH_STEPS):
        if bi == 0:
            rng.shuffle(border)
        ids, labs, attn, pos, P, O = batches[border[bi]]
        tp, ti = cache[border[bi]]
        bi = (bi + 1) % len(batches)
        state.mode, state.mask = "mask", mask
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                       logits_to_keep=O + 1).logits
        kl = masked_kl(logits, labs, P, O, tp, ti)
        opt.zero_grad(set_to_none=True)
        kl.backward()
        state.mode, state.mask = "off", None
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()

    results = {"name": cfg.name, "model": MODEL_NAME, "seed": cfg.seed,
               "n_prompts": len(examples), "teacher_tool_calls": has_call,
               "floor_frac": floor_frac, "floor_reason": floor_reason,
               "evals": {}}
    budgets = sorted(set(EVAL_BUDGETS + [round(floor_frac, 4)]), reverse=True)
    for frac in budgets:
        k = max(1, int(round(frac * n_channels)))
        mk = gates.topk_mask(k).to(DEVICE)
        acc = self_match(model, tok, held, state, "mask", mk, adapter=True)
        results["evals"][f"{frac:.4f}"] = acc
        print(f"[eval@{frac:.0%}] held self-match={acc:.4f}", flush=True)

    unmasked = self_match(model, tok, held, state, "off", adapter=True)
    results["unmasked_self_match"] = unmasked
    results["wall_minutes"] = (time.time() - t0) / 60
    print(f"[guardrail] unmasked self-match={unmasked:.4f}", flush=True)

    np.savez(f"gates_{cfg.name}.npz", la=gates.la.detach().cpu().numpy())
    with open(f"results_{cfg.name}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {cfg.name}: {results['wall_minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
