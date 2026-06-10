#!/usr/bin/env python3
"""v2: lean one-shot joint capability extraction.

Improvements over v1:
  - Teacher distributions precomputed ONCE (top-k sparse cache) instead of
    recomputed every epoch (~30% step compute removed).
  - Dataset pre-tokenized and GPU-resident (v1 ran the GPU at 44% util).
  - Attribution warm-start: grad x act scores bias gate init ordering.
  - Adaptive KL-budget annealing: gates close as fast as behavior allows,
    run terminates at the discovered floor (no preset schedule/budget).
  - Unmasked-KL guardrail (paper eq.4 term c): collimated model stays a
    valid unmasked model.
  - Hierarchical gates: per-layer x per-channel (enables whole-layer drops).
  - Mask-polish phase: brief adapter-only tuning under the hardened binary
    mask to remove the sampled-gate train/eval mismatch.
  - Generation evals only at the end; surrogate KL used during training.

Usage: python joint_extraction_v2.py --name default --seed 42 \
         [--ce-weight 0.05] [--lora-scope all|mlp] [--base-acc 0.9753]
"""

import argparse
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
DEVICE = "cuda"
DTYPE = torch.bfloat16

MAX_STEPS = 3000
WARMUP_STEPS = 100
BATCH_SIZE = 64
LORA_LR = 1e-4
GATE_LR = 5e-2
LAYER_GATE_LR = 1e-2
DUAL_LR = 0.1
KL_BUDGET = 0.02          # adaptive anneal: close gates while EMA-KL < this
TARGET_DECAY = 0.993      # per-step multiplicative target decay when KL ok
MIN_TARGET = 0.008
STALL_STEPS = 400         # stop if open fraction stops moving this long
GUARDRAIL_EVERY = 4
GUARDRAIL_W = 1.0
POLISH_STEPS = 150
PROBE_EVERY = 150         # generation probe cadence once open < PROBE_BELOW
PROBE_BELOW = 0.12
PROBE_N = 128
PROBE_TOL = 0.03          # floor = probe acc drops >3pts below base, twice
TEACHER_TOPK = 128
ATTR_EXAMPLES = 256
HELDOUT_FRAC = 0.10
EVAL_BUDGETS = [0.05, 0.03]

# ---------------------------------------------------------------- data

def build_pairs(seed=42):
    pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    rng = random.Random(42)  # split fixed across seeds for comparability
    rng.shuffle(pairs)
    n_held = int(len(pairs) * HELDOUT_FRAC)
    return pairs[n_held:], pairs[:n_held]


def prompt_of(a, b):
    return f"{a} + {b} ="


def pretokenize(tok, pairs):
    """Tokenize all examples once; return GPU-resident padded tensors."""
    rows = []
    for a, b in pairs:
        p = tok(prompt_of(a, b), add_special_tokens=False)["input_ids"]
        ans = tok(f" {a + b}", add_special_tokens=False)["input_ids"]
        rows.append((p + ans, [-100] * len(p) + ans))
    maxlen = max(len(r[0]) for r in rows)
    pad = tok.pad_token_id
    ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
    labs = torch.full((len(rows), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), maxlen), dtype=torch.long)
    for i, (r, l) in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        labs[i, : len(l)] = torch.tensor(l)
        attn[i, : len(r)] = 1
    return ids.to(DEVICE), labs.to(DEVICE), attn.to(DEVICE)

# ------------------------------------------------------- gates + patching

class HierGates(torch.nn.Module):
    GAMMA, ZETA, BETA = -0.1, 1.1, 2.0 / 3.0

    def __init__(self, n_layers, d_ff, channel_init):
        super().__init__()
        self.ch = torch.nn.Parameter(channel_init)              # [L, d_ff]
        self.layer = torch.nn.Parameter(torch.full((n_layers,), 4.0))

    def _sample(self, la):
        u = torch.rand_like(la).clamp_(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (-u).log1p() + la) / self.BETA)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def sample(self, li):
        return self._sample(self.ch[li]) * self._sample(self.layer[li : li + 1])

    def p_open(self):
        shift = self.BETA * math.log(-self.GAMMA / self.ZETA)
        return torch.sigmoid(self.ch - shift) * torch.sigmoid(
            self.layer - shift).unsqueeze(1)

    def expected_open(self):
        return self.p_open().mean()

    def topk_mask(self, k_total):
        score = self.p_open().detach().flatten()
        idx = torch.topk(score, k_total).indices
        mask = torch.zeros_like(score)
        mask[idx] = 1.0
        return mask.view_as(self.ch)


class MLPState:
    def __init__(self):
        self.mode = "off"      # off | sample | mask
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

# ------------------------------------------------------- teacher cache

@torch.no_grad()
def cache_teacher(model, state, ids, labs, attn, bs=256):
    """One pass over the data; store top-k teacher probs at answer positions."""
    state.mode = "off"
    n, T = ids.shape
    ans_mask = labs[:, 1:] != -100                       # predicts token t+1
    n_ans = ans_mask.sum().item()
    top_p = torch.empty(n_ans, TEACHER_TOPK, device=DEVICE, dtype=torch.float32)
    top_i = torch.empty(n_ans, TEACHER_TOPK, device=DEVICE, dtype=torch.long)
    row_of = torch.full((n, T - 1), -1, dtype=torch.long, device=DEVICE)
    row_of[ans_mask] = torch.arange(n_ans, device=DEVICE)
    ptr = 0
    for i in range(0, n, bs):
        sl = slice(i, i + bs)
        with model.disable_adapter():
            logits = model(input_ids=ids[sl], attention_mask=attn[sl]).logits
        m = ans_mask[sl]
        probs = F.softmax(logits[:, :-1][m].float(), -1)
        p, ix = torch.topk(probs, TEACHER_TOPK, dim=-1)
        k = p.shape[0]
        top_p[ptr : ptr + k] = p / p.sum(-1, keepdim=True)   # renormalize support
        top_i[ptr : ptr + k] = ix
        ptr += k
    return top_p, top_i, row_of, ans_mask


def sparse_kl(student_logits, batch_idx, top_p, top_i, row_of, ans_mask):
    """Forward KL(teacher || student) on cached top-k teacher support."""
    m = ans_mask[batch_idx]                              # [B, T-1]
    rows = row_of[batch_idx][m]                          # [N_ans]
    s_lp = F.log_softmax(student_logits[:, :-1][m].float(), -1)
    tp, ti = top_p[rows], top_i[rows]
    s_at = s_lp.gather(-1, ti)
    return (tp * (tp.clamp_min(1e-9).log() - s_at)).sum(-1).mean()

# ------------------------------------------------------- attribution

def attribution_scores(model, state, ids, labs, attn, n_layers, d_ff):
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)
    scores = torch.zeros(n_layers, d_ff, device=DEVICE, dtype=torch.float32)
    for i in range(0, ATTR_EXAMPLES, 32):
        sl = slice(i, i + 32)
        state.capture, state.captured, state.mode = True, {}, "off"
        with model.disable_adapter(), torch.enable_grad():
            logits = model(input_ids=ids[sl], attention_mask=attn[sl]).logits
            lab = labs[sl][:, 1:]
            m = lab != -100
            lp = F.log_softmax(logits[:, :-1][m].float(), -1)
            (-lp.gather(-1, lab[m].unsqueeze(-1)).mean()).backward()
        for li, inter in state.captured.items():
            if inter.grad is not None:
                scores[li] += (inter.grad.float() * inter.float()).abs().sum((0, 1))
        model.zero_grad(set_to_none=True)
        state.capture, state.captured = False, {}
    emb.requires_grad_(False)
    return scores

# ------------------------------------------------------- evaluation

INT_RE = re.compile(r"\s*(\d+)")


@torch.no_grad()
def gen_accuracy(model, tok, pairs, state, mode, mask=None, adapter=True, bs=512):
    state.mode, state.mask = mode, mask
    tok.padding_side = "left"
    model.eval()
    correct = 0
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
        for (a, b), g in zip(chunk, tok.batch_decode(
                out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            m = INT_RE.match(g)
            if m and int(m.group(1)) == a + b:
                correct += 1
    state.mode, state.mask = "off", None
    return correct / len(pairs)

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce-weight", type=float, default=0.05)
    ap.add_argument("--lora-scope", choices=["all", "mlp"], default="all")
    ap.add_argument("--base-acc", type=float, default=None)
    ap.add_argument("--slice-test", action="store_true",
                    help="after extraction: merge adapter, physically delete "
                         "masked MLP channels, verify equivalence, benchmark "
                         "params/VRAM/throughput vs the full model")
    ap.add_argument("--layer-gates", action="store_true",
                    help="enable hierarchical layer gates (off: v2.1 default; "
                         "v2.0 layer gates closed 16/28 layers and broke recovery)")
    cfg = ap.parse_args()

    t0 = time.time()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, attn_implementation="sdpa").to(DEVICE)

    targets = (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                "down_proj"] if cfg.lora_scope == "all"
               else ["gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=targets, task_type="CAUSAL_LM"))

    state = MLPState()
    n_layers = patch_mlps(model, state)
    d_ff = model.config.intermediate_size
    n_channels = n_layers * d_ff

    train_pairs, held_pairs = build_pairs()
    tr_ids, tr_labs, tr_attn = pretokenize(tok, train_pairs)
    print(f"[setup] {cfg.name} seed={cfg.seed} channels={n_channels} "
          f"train={len(train_pairs)} held={len(held_pairs)}", flush=True)

    base_acc = cfg.base_acc or gen_accuracy(
        model, tok, held_pairs, state, "off", adapter=False)
    print(f"[base] acc={base_acc:.4f}", flush=True)

    # teacher cache: one pass, ~30 batches
    top_p, top_i, row_of, ans_mask = cache_teacher(
        model, state, tr_ids, tr_labs, tr_attn)
    print(f"[teacher] cached {top_p.shape[0]} answer positions "
          f"top{TEACHER_TOPK}", flush=True)

    # attribution warm-start: bias gate init by grad x act percentile
    scores = attribution_scores(model, state, tr_ids, tr_labs, tr_attn,
                                n_layers, d_ff)
    pct = scores.flatten().argsort().argsort().float() / (n_channels - 1)
    ch_init = (2.0 + 3.0 * pct).view(n_layers, d_ff)
    gates = HierGates(n_layers, d_ff, ch_init).to(DEVICE)
    state.gates = gates
    if not cfg.layer_gates:
        gates.layer.data.fill_(8.0)        # p_open ~ 1: layer gating inert
        gates.layer.requires_grad_(False)
    print(f"[attr] warm-start init open={gates.expected_open().item():.4f} "
          f"layer_gates={cfg.layer_gates}", flush=True)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": LORA_LR},
              {"params": [gates.ch], "lr": GATE_LR}]
    if cfg.layer_gates:
        groups.append({"params": [gates.layer], "lr": LAYER_GATE_LR})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    # generation-probe floor detection: KL alone is miscalibrated at the deep
    # end (v2.0 descended to 2.2% open with KL "healthy" while true recovery
    # fell to 62%). Probe on train pairs so the held-out eval stays untouched.
    probe_pairs = train_pairs[:PROBE_N]
    probe_base = gen_accuracy(model, tok, probe_pairs, state, "off",
                              adapter=False, bs=PROBE_N)
    print(f"[probe] base={probe_base:.4f} on {PROBE_N} train pairs", flush=True)

    lam, ema_kl = 0.0, 0.0
    target = gates.expected_open().item()
    open_hist = []
    probe_dry = 0
    floor_reason = "max_steps"
    n_train = tr_ids.shape[0]
    model.train()
    step = 0

    for step in range(1, MAX_STEPS + 1):
        idx = torch.randint(0, n_train, (BATCH_SIZE,), device=DEVICE)
        state.mode = "sample"
        logits = model(input_ids=tr_ids[idx], attention_mask=tr_attn[idx]).logits
        state.mode = "off"
        kl = sparse_kl(logits, idx, top_p, top_i, row_of, ans_mask)

        lab = tr_labs[idx][:, 1:]
        m = lab != -100
        ce = -F.log_softmax(logits[:, :-1][m].float(), -1).gather(
            -1, lab[m].unsqueeze(-1)).mean()

        loss = kl + cfg.ce_weight * ce + lam * gates.expected_open()

        if step % GUARDRAIL_EVERY == 0:   # unmasked student stays near base
            g_logits = model(input_ids=tr_ids[idx],
                             attention_mask=tr_attn[idx]).logits
            loss = loss + GUARDRAIL_W * sparse_kl(
                g_logits, idx, top_p, top_i, row_of, ans_mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()

        ema_kl = 0.9 * ema_kl + 0.1 * kl.item()
        open_now = gates.expected_open().item()
        open_hist.append(open_now)

        # adaptive anneal: lower the target only while behavior holds and
        # the gates are keeping up with the target
        if step > WARMUP_STEPS and ema_kl < KL_BUDGET and (open_now - target) < 0.05:
            target = max(MIN_TARGET, target * TARGET_DECAY)
        lam = max(0.0, lam + DUAL_LR * (open_now - target))

        if step % 100 == 0:
            print(f"[train] step={step} kl={kl.item():.4f} ema={ema_kl:.4f} "
                  f"open={open_now:.4f} target={target:.4f} lam={lam:.2f}",
                  flush=True)

        if open_now < PROBE_BELOW and step % PROBE_EVERY == 0:
            k_now = max(1, int(round(open_now * n_channels)))
            pmask = gates.topk_mask(k_now).to(DEVICE)
            pacc = gen_accuracy(model, tok, probe_pairs, state, "mask", pmask,
                                adapter=True, bs=PROBE_N)
            model.train()
            probe_dry = probe_dry + 1 if pacc < probe_base - PROBE_TOL else 0
            print(f"[probe] step={step} open={open_now:.4f} acc={pacc:.4f} "
                  f"(base {probe_base:.4f}) dry={probe_dry}", flush=True)
            if probe_dry >= 2:
                floor_reason = "probe"
                print(f"[floor] probe floor at step {step} "
                      f"open={open_now:.4f}", flush=True)
                break

        if target <= MIN_TARGET and open_now <= MIN_TARGET * 1.5:
            floor_reason = "min_target"
            print(f"[floor] hit MIN_TARGET at step {step}", flush=True)
            break
        if (len(open_hist) > STALL_STEPS
                and open_hist[-STALL_STEPS] - open_now < 0.002
                and ema_kl >= KL_BUDGET):
            floor_reason = "stall"
            print(f"[floor] stalled at step {step} open={open_now:.4f}", flush=True)
            break

    floor_frac = gates.expected_open().item()
    k_floor = max(1, int(round(floor_frac * n_channels)))

    # polish: freeze hardened mask, brief adapter-only tuning under it
    mask = gates.topk_mask(k_floor).to(DEVICE)
    for p in gates.parameters():
        p.requires_grad_(False)
    print(f"[polish] {POLISH_STEPS} steps at floor={floor_frac:.4f} "
          f"(k={k_floor})", flush=True)
    for _ in range(POLISH_STEPS):
        idx = torch.randint(0, n_train, (BATCH_SIZE,), device=DEVICE)
        state.mode, state.mask = "mask", mask
        logits = model(input_ids=tr_ids[idx], attention_mask=tr_attn[idx]).logits
        state.mode, state.mask = "off", None
        kl = sparse_kl(logits, idx, top_p, top_i, row_of, ans_mask)
        lab = tr_labs[idx][:, 1:]
        m = lab != -100
        ce = -F.log_softmax(logits[:, :-1][m].float(), -1).gather(
            -1, lab[m].unsqueeze(-1)).mean()
        opt.zero_grad(set_to_none=True)
        (kl + cfg.ce_weight * ce).backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()

    # final generation evals
    results = {"name": cfg.name, "seed": cfg.seed,
               "config": {"ce_weight": cfg.ce_weight, "lora_scope": cfg.lora_scope,
                          "layer_gates": cfg.layer_gates},
               "base_acc": base_acc, "floor_frac": floor_frac,
               "floor_reason": floor_reason,
               "steps_used": step, "evals": {}}
    budgets = sorted(set(EVAL_BUDGETS + [floor_frac]), reverse=True)
    for frac in budgets:
        k = max(1, int(round(frac * n_channels)))
        mk = gates.topk_mask(k).to(DEVICE)
        acc = gen_accuracy(model, tok, held_pairs, state, "mask", mk, adapter=True)
        results["evals"][f"{frac:.4f}"] = {"acc": acc, "recovery": acc / base_acc}
        print(f"[eval@{frac:.2%}] acc={acc:.4f} recovery={acc/base_acc:.4f}",
              flush=True)

    layer_open = (torch.sigmoid(gates.layer) > 0.5).sum().item()
    unmasked = gen_accuracy(model, tok, held_pairs, state, "off", adapter=True)
    results["unmasked_acc"] = unmasked
    results["layers_open"] = layer_open

    if cfg.slice_test:
        import gc
        smask = gates.topk_mask(k_floor)
        scaffold_acc = gen_accuracy(model, tok, held_pairs, state, "mask",
                                    smask.to(DEVICE), adapter=True)
        model = model.merge_and_unload()       # fold adapter into base weights
        p_before = sum(p.numel() for p in model.parameters())

        def timed_acc():
            torch.cuda.synchronize()
            t = time.time()
            acc = gen_accuracy(model, tok, held_pairs, state, "off", adapter=True)
            torch.cuda.synchronize()
            return acc, time.time() - t

        full_acc, full_t = timed_acc()         # merged, unsliced reference

        for li, layer in enumerate(model.model.layers):
            keep = smask[li].nonzero(as_tuple=True)[0].to(DEVICE)
            mlp = layer.mlp
            mlp.gate_proj.weight = torch.nn.Parameter(
                mlp.gate_proj.weight.data[keep].clone())
            mlp.gate_proj.out_features = len(keep)
            mlp.up_proj.weight = torch.nn.Parameter(
                mlp.up_proj.weight.data[keep].clone())
            mlp.up_proj.out_features = len(keep)
            mlp.down_proj.weight = torch.nn.Parameter(
                mlp.down_proj.weight.data[:, keep].clone())
            mlp.down_proj.in_features = len(keep)
        gc.collect()
        torch.cuda.empty_cache()
        p_after = sum(p.numel() for p in model.parameters())
        vram = torch.cuda.memory_allocated() / 2**30

        sliced_acc, sliced_t = timed_acc()
        results["slice"] = {
            "scaffold_acc": scaffold_acc,
            "merged_full_acc": full_acc,
            "sliced_acc": sliced_acc,
            "params_before": p_before, "params_after": p_after,
            "param_ratio": p_before / p_after,
            "gen_seconds_full": full_t, "gen_seconds_sliced": sliced_t,
            "speedup": full_t / sliced_t,
            "vram_sliced_gb": vram,
        }
        print(f"[slice] scaffold={scaffold_acc:.4f} sliced={sliced_acc:.4f} "
              f"(equiv check) | params {p_before/1e9:.2f}B -> "
              f"{p_after/1e9:.2f}B ({p_before/p_after:.2f}x) | "
              f"gen {full_t:.1f}s -> {sliced_t:.1f}s "
              f"({full_t/sliced_t:.2f}x) | vram {vram:.2f}GB", flush=True)

    results["wall_minutes"] = (time.time() - t0) / 60
    print(f"[guardrail] unmasked acc={unmasked:.4f} (base {base_acc:.4f}); "
          f"layers open {layer_open}/{n_layers}", flush=True)

    np.savez(f"gates_{cfg.name}.npz", ch=gates.ch.detach().cpu().numpy(),
             layer=gates.layer.detach().cpu().numpy())
    with open(f"results_{cfg.name}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {cfg.name}: {results['wall_minutes']:.1f} min", flush=True)
    if cfg.base_acc is None:
        with open("base_acc.txt", "w") as f:
            f.write(str(base_acc))


if __name__ == "__main__":
    main()
