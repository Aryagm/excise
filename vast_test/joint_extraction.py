#!/usr/bin/env python3
"""Bridge experiment: one-shot joint gate+collimator capability extraction.

Tests whether a single training run -- hard-concrete L0 gates on MLP
intermediate channels trained jointly with a rank-32 KL-LoRA collimator,
teacher = frozen unmasked base model -- can match PRISM's staged
collimate->attribute->mask pipeline on its own arithmetic testbed
(Qwen2.5-Math-1.5B, 2-digit addition).

Paper ground truth: raw attribution mask ~29% recovery at ~5.75% MLP;
staged pre-attribution collimation 91.33% at ~5.05% MLP.
Success here: >=85% recovery at ~5% from one joint run, no attribution stage.
"""

import json
import math
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16

TRAIN_STEPS = 2400
BATCH_SIZE = 64
LORA_LR = 1e-4
GATE_LR = 5e-2
CE_WEIGHT = 0.05
TARGET_FINAL_OPEN = 0.04          # anneal target: 4% of channels
ANNEAL_START, ANNEAL_END = 200, 2000
DUAL_LR = 1e-3                    # lagrangian step for sparsity penalty
EVAL_BUDGETS = [0.20, 0.10, 0.05]  # harden + eval when open fraction crosses these
HELDOUT_FRAC = 0.10
ATTR_EXAMPLES = 512

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------- data

def build_pairs():
    pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_held = int(len(pairs) * HELDOUT_FRAC)
    return pairs[n_held:], pairs[:n_held]


def prompt_of(a, b):
    return f"{a} + {b} ="


def answer_of(a, b):
    return f" {a + b}"


def encode_batch(tok, batch):
    rows = []
    for a, b in batch:
        p_ids = tok(prompt_of(a, b), add_special_tokens=False)["input_ids"]
        a_ids = tok(answer_of(a, b), add_special_tokens=False)["input_ids"]
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        rows.append((ids, labels))
    maxlen = max(len(r[0]) for r in rows)
    pad = tok.pad_token_id
    input_ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
    labels = torch.full((len(rows), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), maxlen), dtype=torch.long)
    for i, (ids, labs) in enumerate(rows):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(labs)] = torch.tensor(labs)
        attn[i, : len(ids)] = 1
    return input_ids.to(DEVICE), labels.to(DEVICE), attn.to(DEVICE)

# ------------------------------------------------------- gates + patching

class HardConcreteGates(torch.nn.Module):
    GAMMA, ZETA, BETA = -0.1, 1.1, 2.0 / 3.0

    def __init__(self, n_layers, d_ff):
        super().__init__()
        # init ~open: sigmoid(3.0) = 0.95
        self.log_alpha = torch.nn.Parameter(torch.full((n_layers, d_ff), 3.0))

    def sample(self, layer):
        la = self.log_alpha[layer]
        u = torch.rand_like(la).clamp_(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (-u).log1p() + la) / self.BETA)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def expected_open(self):
        return torch.sigmoid(
            self.log_alpha - self.BETA * math.log(-self.GAMMA / self.ZETA)
        )

    def topk_mask(self, k_total):
        flat = self.log_alpha.detach().flatten()
        idx = torch.topk(flat, k_total).indices
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        return mask.view_as(self.log_alpha)


class MLPState:
    """Shared control for the patched MLP forwards.

    mode: 'off' (no gating: teacher / base), 'sample' (training),
          'mask' (binary mask in self.mask, zero-isolation contract).
    """

    def __init__(self):
        self.mode = "off"
        self.mask = None
        self.gates = None
        self.capture = False
        self.captured = {}


def patch_mlps(model, state):
    layers = model.base_model.model.model.layers
    for li, layer in enumerate(layers):
        mlp = layer.mlp

        def fwd(x, mlp=mlp, li=li):
            inter = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
            if state.capture:
                inter.retain_grad()
                state.captured[li] = inter
            if state.mode == "sample":
                inter = inter * state.gates.sample(li).to(inter.dtype)
            elif state.mode == "mask":
                inter = inter * state.mask[li].to(inter.dtype)
            return mlp.down_proj(inter)

        mlp.forward = fwd
    return len(layers)

# ------------------------------------------------------------- evaluation

INT_RE = re.compile(r"\s*(\d+)")


@torch.no_grad()
def evaluate(model, tok, pairs, state, mode, mask=None, adapter=True, bs=256):
    state.mode, state.mask = mode, mask
    tok.padding_side = "left"
    correct = 0
    ctx = torch.no_grad()
    model.eval()
    for i in range(0, len(pairs), bs):
        chunk = pairs[i : i + bs]
        enc = tok([prompt_of(a, b) for a, b in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(DEVICE)
        if adapter:
            out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        else:
            with model.disable_adapter():
                out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        for (a, b), g in zip(chunk, gen):
            m = INT_RE.match(g)
            if m and int(m.group(1)) == a + b:
                correct += 1
    state.mode, state.mask = "off", None
    return correct / len(pairs)


@torch.no_grad()
def surrogate_kl(model, tok, pairs, state, mode, mask=None, adapter=True, n=128):
    """Mean forward KL(base || masked-student) on answer tokens (dev slice)."""
    state.mode, state.mask = "off", None
    input_ids, labels, attn = encode_batch(tok, pairs[:n])
    with model.disable_adapter():
        t_logits = model(input_ids=input_ids, attention_mask=attn).logits
    state.mode, state.mask = mode, mask
    if adapter:
        s_logits = model(input_ids=input_ids, attention_mask=attn).logits
    else:
        with model.disable_adapter():
            s_logits = model(input_ids=input_ids, attention_mask=attn).logits
    state.mode, state.mask = "off", None
    sl, tl = s_logits[:, :-1], t_logits[:, :-1]
    m = labels[:, 1:] != -100
    tp = F.softmax(tl[m].float(), -1)
    kl = (tp * (tp.clamp_min(1e-9).log() - F.log_softmax(sl[m].float(), -1))).sum(-1)
    return kl.mean().item()

# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    results = {"model": MODEL_NAME, "config": {
        "steps": TRAIN_STEPS, "batch": BATCH_SIZE, "lora_lr": LORA_LR,
        "gate_lr": GATE_LR, "ce_weight": CE_WEIGHT,
        "target_open": TARGET_FINAL_OPEN}}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, attn_implementation="sdpa").to(DEVICE)

    lora = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    state = MLPState()
    n_layers = patch_mlps(model, state)
    d_ff = model.config.intermediate_size
    n_channels = n_layers * d_ff
    print(f"[setup] {n_layers} layers x {d_ff} = {n_channels} MLP channels", flush=True)

    gates = HardConcreteGates(n_layers, d_ff).to(DEVICE)
    state.gates = gates

    train_pairs, held_pairs = build_pairs()
    print(f"[setup] train={len(train_pairs)} held={len(held_pairs)}", flush=True)

    # ---- base accuracy
    base_acc = evaluate(model, tok, held_pairs, state, "off", adapter=False)
    print(f"[base] accuracy={base_acc:.4f}", flush=True)
    results["base_acc"] = base_acc

    # ---- random mask baseline @5%
    k5 = int(0.05 * n_channels)
    g = torch.Generator().manual_seed(SEED)
    rand_idx = torch.randperm(n_channels, generator=g)[:k5]
    rand_mask = torch.zeros(n_channels)
    rand_mask[rand_idx] = 1.0
    rand_mask = rand_mask.view(n_layers, d_ff).to(DEVICE)
    rand_acc = evaluate(model, tok, held_pairs, state, "mask", rand_mask, adapter=False)
    print(f"[random@5%] acc={rand_acc:.4f} recovery={rand_acc/base_acc:.4f}", flush=True)
    results["random_5pct"] = {"acc": rand_acc, "recovery": rand_acc / base_acc}

    # ---- grad x act attribution baseline (proxy for ReLP)
    print("[attr] computing grad x act attribution...", flush=True)
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)
    scores = torch.zeros(n_layers, d_ff, device=DEVICE, dtype=torch.float32)
    model.eval()
    attr_pairs = train_pairs[:ATTR_EXAMPLES]
    for i in range(0, len(attr_pairs), 32):
        input_ids, labels, attn = encode_batch(tok, attr_pairs[i : i + 32])
        state.capture, state.captured = True, {}
        state.mode = "off"
        with model.disable_adapter(), torch.enable_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            sl, lab = logits[:, :-1], labels[:, 1:]
            m = lab != -100
            lp = F.log_softmax(sl[m].float(), -1)
            loss = -lp.gather(-1, lab[m].unsqueeze(-1)).mean()
            loss.backward()
        for li, inter in state.captured.items():
            if inter.grad is not None:
                scores[li] += (inter.grad.float() * inter.float()).abs().sum((0, 1))
        model.zero_grad(set_to_none=True)
        state.capture, state.captured = False, {}
    emb.requires_grad_(False)

    flat = scores.flatten()
    for frac in (0.05, 0.25):
        k = int(frac * n_channels)
        idx = torch.topk(flat, k).indices
        amask = torch.zeros(n_channels, device=DEVICE)
        amask[idx] = 1.0
        amask = amask.view(n_layers, d_ff)
        acc = evaluate(model, tok, held_pairs, state, "mask", amask, adapter=False)
        print(f"[rawmask@{frac:.0%}] acc={acc:.4f} recovery={acc/base_acc:.4f}", flush=True)
        results[f"rawmask_{int(frac*100)}pct"] = {"acc": acc, "recovery": acc / base_acc}

    # ---- joint training
    print("[joint] starting joint gate+collimator training...", flush=True)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": LORA_LR},
        {"params": gates.parameters(), "lr": GATE_LR},
    ], weight_decay=0.0)

    lam = 0.0
    checkpoints = {}
    pending = sorted(EVAL_BUDGETS, reverse=True)
    order = list(range(len(train_pairs)))
    rng = random.Random(SEED + 1)
    ptr = len(order)
    model.train()

    for step in range(1, TRAIN_STEPS + 1):
        if ptr + BATCH_SIZE > len(order):
            rng.shuffle(order)
            ptr = 0
        batch = [train_pairs[j] for j in order[ptr : ptr + BATCH_SIZE]]
        ptr += BATCH_SIZE
        input_ids, labels, attn = encode_batch(tok, batch)

        # teacher: frozen base, no gates
        state.mode = "off"
        with torch.no_grad(), model.disable_adapter():
            t_logits = model(input_ids=input_ids, attention_mask=attn).logits

        # student: adapter + sampled gates
        state.mode = "sample"
        s_logits = model(input_ids=input_ids, attention_mask=attn).logits
        state.mode = "off"

        sl, tl, lab = s_logits[:, :-1], t_logits[:, :-1], labels[:, 1:]
        m = lab != -100
        s_lp = F.log_softmax(sl[m].float(), -1)
        t_p = F.softmax(tl[m].float(), -1)
        kl = (t_p * (t_p.clamp_min(1e-9).log() - s_lp)).sum(-1).mean()
        ce = -s_lp.gather(-1, lab[m].unsqueeze(-1)).mean()

        open_frac = gates.expected_open().mean()
        loss = kl + CE_WEIGHT * ce + lam * open_frac
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()

        # sparsity target schedule (cosine 1.0 -> TARGET_FINAL_OPEN)
        if step < ANNEAL_START:
            target = 1.0
        elif step > ANNEAL_END:
            target = TARGET_FINAL_OPEN
        else:
            t = (step - ANNEAL_START) / (ANNEAL_END - ANNEAL_START)
            target = TARGET_FINAL_OPEN + (1 - TARGET_FINAL_OPEN) * 0.5 * (1 + math.cos(math.pi * t))
        lam = max(0.0, lam + DUAL_LR * (open_frac.item() - target) * 100)

        if step % 100 == 0:
            print(f"[joint] step={step} kl={kl.item():.4f} ce={ce.item():.4f} "
                  f"open={open_frac.item():.4f} target={target:.4f} lam={lam:.3f}",
                  flush=True)

        # checkpoint evals as the open fraction crosses each budget
        if pending and open_frac.item() <= pending[0]:
            frac = pending.pop(0)
            k = int(frac * n_channels)
            mask = gates.topk_mask(k).to(DEVICE)
            acc = evaluate(model, tok, held_pairs, state, "mask", mask, adapter=True)
            skl = surrogate_kl(model, tok, held_pairs, state, "mask", mask, adapter=True)
            checkpoints[f"{frac:.2f}"] = {
                "step": step, "acc": acc, "recovery": acc / base_acc,
                "surrogate_kl": skl}
            print(f"[ckpt@{frac:.0%}] step={step} acc={acc:.4f} "
                  f"recovery={acc/base_acc:.4f} kl={skl:.4f}", flush=True)
            model.train()

    # ---- final evals
    final_open = gates.expected_open().mean().item()
    print(f"[final] expected open fraction={final_open:.4f}", flush=True)

    finals = {}
    for frac in (0.05, 0.04, 0.03):
        k = int(frac * n_channels)
        mask = gates.topk_mask(k).to(DEVICE)
        acc = evaluate(model, tok, held_pairs, state, "mask", mask, adapter=True)
        skl = surrogate_kl(model, tok, held_pairs, state, "mask", mask, adapter=True)
        finals[f"{frac:.2f}"] = {"acc": acc, "recovery": acc / base_acc,
                                 "surrogate_kl": skl}
        print(f"[final@{frac:.0%}] acc={acc:.4f} recovery={acc/base_acc:.4f} "
              f"kl={skl:.4f}", flush=True)

    # behavior preservation: collimated model, unmasked
    unmasked_acc = evaluate(model, tok, held_pairs, state, "off", adapter=True)
    print(f"[guardrail] collimated unmasked acc={unmasked_acc:.4f}", flush=True)

    results["joint_checkpoints"] = checkpoints
    results["joint_final"] = finals
    results["collimated_unmasked_acc"] = unmasked_acc
    results["final_open_frac"] = final_open
    results["wall_minutes"] = (time.time() - t0) / 60

    np.savez("gates_final.npz", log_alpha=gates.log_alpha.detach().cpu().numpy())
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    print(f"[done] {results['wall_minutes']:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
