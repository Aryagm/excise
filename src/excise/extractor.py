"""The extraction loop: joint gate + adapter training with an adaptive
sparsity controller, behavior-probe floor detection, and a guardrail that
keeps the unmasked model anchored to the base."""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from .arch import find_mlps
from .config import ExtractConfig
from .data import collate, load_prompts
from .export import prune_vocab, slice_model
from .gates import HardConcreteGates
from .hooks import GateHooks
from .probes import self_match, tf_self_match
from .teacher import (cache_distributions, generate_targets, out_logprobs,
                      sparse_kl)


class ExtractionResult:
    def __init__(self, model, tok, gates, hooks, mlp_map, config,
                 floor: float, floor_reason: str, frontier: list,
                 receipts: dict, held: list, vocab_support: list):
        self._model = model
        self._tok = tok
        self._gates = gates
        self._hooks = hooks
        self._mlp_map = mlp_map
        self.config = config
        self.floor = floor
        self.floor_reason = floor_reason
        self.frontier = frontier          # [(budget_frac, held_self_match)]
        self.receipts = receipts
        self._held = held
        self.vocab_support = vocab_support
        self._sliced = False
        self._vocab_keep = None

    def save(self, out_dir: str):
        """Persist mask, adapter (or the sliced model, after
        export_sliced()), config, and receipts."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "gates.npz",
                 log_alpha=self._gates.log_alpha.detach().cpu().numpy())
        k = max(1, int(round(self.floor * self._mlp_map.n_channels)))
        np.savez(out / "mask_floor.npz",
                 mask=self._gates.topk_mask(k).cpu().numpy())
        (out / "config.json").write_text(json.dumps(self.config.to_dict(),
                                                    indent=2))
        (out / "receipts.json").write_text(json.dumps(self.receipts, indent=2))
        (out / "frontier.json").write_text(json.dumps(self.frontier, indent=2))
        (out / "vocab_support.json").write_text(
            json.dumps(self.vocab_support))
        if hasattr(self._model, "save_pretrained"):
            if self._sliced:
                self._model.save_pretrained(str(out / "sliced"))
                if hasattr(self._tok, "save_pretrained"):
                    self._tok.save_pretrained(str(out / "sliced"))
            else:
                self._model.save_pretrained(str(out / "adapter"))
        return out

    def export_sliced(self, budget: float | None = None,
                      prune_vocabulary: bool = False, verify_n: int = 64):
        """Merge the adapter and physically delete masked channels.
        Destructive: the result's model becomes the sliced model. Returns a
        plain transformers model. Slicing is mathematically equivalent to
        zero-isolation masking; `verify_n > 0` re-checks that equivalence on
        held data and records it in the receipts. With `prune_vocabulary`,
        the embedding/lm_head rows are additionally cut to the capability's
        token support (the pruned model speaks a remapped id space; see
        export.prune_vocab)."""
        budget = budget or self.floor
        k = max(1, int(round(budget * self._mlp_map.n_channels)))
        mask = self._gates.topk_mask(k)
        model = self._model
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        slice_model(model, mask)
        # The gate hooks reference deleted channels; they must not survive.
        self._hooks.remove()
        self._model = model
        self._sliced = True
        if verify_n:
            n = min(len(self._held), verify_n)
            acc = self_match(model, self._tok, self._held[:n], None, "off",
                             bs=self._eval_bs(),
                             max_new_tokens=self.config.max_new_tokens,
                             scope=self.config.match_scope)
            self.receipts["sliced_self_match"] = acc
        if prune_vocabulary:
            old2new = prune_vocab(model, self.vocab_support)
            self._vocab_keep = torch.as_tensor(self.vocab_support)
            if verify_n:
                n = min(len(self._held), verify_n)
                acc = self._pruned_self_match(self._held[:n], old2new)
                self.receipts["pruned_self_match"] = acc
        return model

    def _eval_bs(self):
        return self.config.eval_batch_size or self.config.batch_size

    @torch.no_grad()
    def _pruned_self_match(self, examples, old2new) -> float:
        """Self-match through the vocab-remapped id space."""
        from .data import prompt_batch
        tok, model = self._tok, self._model
        device = next(model.parameters()).device
        old2new = old2new.to(device)
        keep = self._vocab_keep.to(device)
        tok.padding_side = "left"
        model.eval()
        bs = self._eval_bs()
        hit = 0
        for i in range(0, len(examples), bs):
            chunk = examples[i: i + bs]
            ids, attn, P = prompt_batch(chunk, tok.pad_token_id, device)
            gen = model.generate(
                input_ids=old2new[ids], attention_mask=attn,
                max_new_tokens=min(self.config.max_new_tokens,
                                   max(len(e["out_ids"]) for e in chunk)),
                do_sample=False,
                pad_token_id=int(old2new[tok.pad_token_id]))
            back = keep[gen[:, P:]]
            for j, e in enumerate(chunk):
                txt = tok.decode(back[j], skip_special_tokens=True).strip()
                ref = e["out_text"].strip()
                if self.config.match_scope == "first_line":
                    txt, ref = txt.split("\n")[0], ref.split("\n")[0]
                hit += txt.strip() == ref.strip()
        return hit / len(examples)

    def masked_eval(self, budget: float) -> float:
        """Held-out self-match at an arbitrary budget."""
        if self._sliced:
            raise RuntimeError(
                "model has been sliced; the gate hooks are gone and masked "
                "evals are no longer meaningful. Evaluate the sliced model "
                "directly (receipts['sliced_self_match']).")
        k = max(1, int(round(budget * self._mlp_map.n_channels)))
        mask = self._gates.topk_mask(k).to(
            next(self._model.parameters()).device)
        return self_match(self._model, self._tok, self._held, self._hooks,
                          "mask", mask, bs=self._eval_bs(),
                          max_new_tokens=self.config.max_new_tokens,
                          scope=self.config.match_scope)

    def report(self) -> str:
        r = self.receipts
        n = r.get("held_n", 0)

        def ci(p):
            if not n:
                return ""
            return f" ± {1.96 * math.sqrt(max(p * (1 - p), 0) / n):.1%}"

        lines = [
            "excise extraction report",
            "=" * 40,
            f"channels:            {self._mlp_map.n_channels}",
            f"floor:               {self.floor:.2%} ({self.floor_reason})",
            f"frontier (held-out self-match, n={n}):",
        ]
        for b, acc in self.frontier:
            lines.append(f"  keep {b:>6.2%}  ->  {acc:.1%}{ci(acc)}")
        lines += [
            f"base decode noise:   {r['base_self_match']:.1%} self-match "
            "(adapter off; ceiling for everything below)",
            f"unmasked drift:      {r['unmasked_self_match']:.1%} self-match "
            "(should be near the base line)",
            f"random-mask control: {r['random_mask_self_match']:.1%} "
            "(layer-profile-matched; should be near 0%)",
            f"probe trace:         {len(r['probe_trace'])} probes, "
            f"min {min((p['self_match'] for p in r['probe_trace']), default=1.0):.1%}",
            f"wall time:           {r['wall_minutes']:.1f} min",
        ]
        if "sliced_self_match" in r:
            lines.append(f"sliced self-match:   {r['sliced_self_match']:.1%} "
                         "(physical slice == mask check)")
        if "pruned_self_match" in r:
            lines.append(f"pruned self-match:   {r['pruned_self_match']:.1%} "
                         "(after vocabulary pruning)")
        return "\n".join(lines)


def _attribution_init(model, hooks, batches, mlp_map, cfg, device, log):
    """Warm-start gate init from |grad x act| attribution percentiles on the
    base model (adapter disabled), as in the validated research pipeline.
    Returns a [n_layers, d_ff] log-alpha init spread over 2.0..5.0."""
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)
    scores = torch.zeros(mlp_map.n_layers, mlp_map.d_ff, dtype=torch.float32,
                         device=device)
    n_b = max(1, math.ceil(cfg.attr_examples / cfg.batch_size))
    sel = batches[:: max(1, len(batches) // n_b)][:n_b]
    # Attribution retains every layer's down-proj input AND its grad, with
    # gradient checkpointing off (captured tensors would not survive
    # recomputation) — memory scales with tokens x d_ff x layers, so bound
    # the per-backward slice by TOKEN budget, not example count.
    budget = 1024 if sum(p.numel() for p in model.parameters()) > 3e9 else 2048
    for ids, labs, attn, pos, P, O in sel:
        micro = max(1, min(ids.shape[0], budget // max(1, ids.shape[1])))
        sl = slice(0, micro)
        hooks.mode, hooks.captured = "capture", {}
        ctx = (model.disable_adapter() if hasattr(model, "disable_adapter")
               else torch.enable_grad())
        with ctx, torch.enable_grad():
            logits = model(input_ids=ids[sl], attention_mask=attn[sl],
                           position_ids=pos[sl],
                           logits_to_keep=O + 1).logits
            lp, m = out_logprobs(logits, labs[sl], P)
            (-lp.gather(-1, labs[sl][:, P:][m].unsqueeze(-1)).mean()
             ).backward()
        for li, x in hooks.captured.items():
            if x.grad is not None:
                scores[li] += (x.grad.float() * x.float()).abs().sum((0, 1))
        model.zero_grad(set_to_none=True)
        hooks.captured = {}
    hooks.mode = "off"
    emb.requires_grad_(False)
    if scores.sum() == 0:
        log("[excise] attribution produced no signal; constant gate init")
        return None
    pct = (scores.flatten().argsort().argsort().float()
           / (scores.numel() - 1))
    log("[excise] attribution warm-start over "
        f"{sum(min(b[0].shape[0], micro or b[0].shape[0]) for b in sel)} "
        "examples")
    return (2.0 + 3.0 * pct).view(mlp_map.n_layers, mlp_map.d_ff)


def extract(model_or_name, prompts, tokenizer=None,
            config: ExtractConfig | None = None, device=None,
            log=print) -> ExtractionResult:
    """Extract the capability exercised by `prompts` into a sparse channel
    substrate. Label-free: the model's own greedy outputs are the target."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = config or ExtractConfig()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    t0 = time.time()

    if isinstance(model_or_name, str):
        tok = tokenizer or AutoTokenizer.from_pretrained(model_or_name)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_or_name,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
    else:
        model, tok = model_or_name, tokenizer
        if tok is None:
            raise ValueError("pass tokenizer= when passing a model object")
        device = device or next(model.parameters()).device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    n_params = sum(p.numel() for p in model.parameters())
    use_ckpt = (cfg.gradient_checkpointing if cfg.gradient_checkpointing
                is not None else n_params > 3e9)

    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none", target_modules="all-linear", task_type="CAUSAL_LM"))

    mlp_map = find_mlps(model)
    hooks = GateHooks(mlp_map)
    eval_bs = cfg.eval_batch_size or cfg.batch_size
    log(f"[excise] {mlp_map.n_layers} layers x {mlp_map.d_ff} = "
        f"{mlp_map.n_channels} channels | ckpt={use_ckpt}")

    # data: tokenize, generate label-free targets, split
    raw = load_prompts(prompts)
    examples = []
    for p in raw:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if 0 < len(ids) <= cfg.max_prompt_tokens:
            examples.append({"prompt": p, "prompt_ids": ids})
    log(f"[excise] {len(examples)}/{len(raw)} prompts within token limit")
    generate_targets(model, tok, examples, hooks, cfg.max_new_tokens,
                     bs=eval_bs)
    examples = [e for e in examples if e["out_ids"]]

    rng = random.Random(cfg.seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    n_held = max(1, int(len(examples) * cfg.held_frac))
    held = [examples[i] for i in order[:n_held]]
    train = [examples[i] for i in order[n_held:]]

    # Probe split: floor detection must run on prompts the gradient never
    # sees, or the stopper measures memorization and overshoots whenever
    # there is a train/held gap. Falls back to in-train probes (with a
    # warning) when the prompt set is too small to give probes away.
    if cfg.probe_holdout and len(train) >= 4 * cfg.probe_n:
        probe_set, train = train[:cfg.probe_n], train[cfg.probe_n:]
        probe_src = "dev"
    else:
        stride = max(1, len(train) // cfg.probe_n)
        probe_set = train[::stride][:cfg.probe_n]
        probe_src = "train"
        if cfg.probe_holdout:
            log("[excise] WARNING: too few prompts for a probe holdout; "
                "probing on train data (floor may read optimistic)")
    train.sort(key=lambda e: len(e["prompt_ids"]))
    log(f"[excise] train={len(train)} held={len(held)} "
        f"probe={len(probe_set)} ({probe_src})")

    batches = [collate(train[i: i + cfg.batch_size], tok.pad_token_id, device)
               for i in range(0, len(train), cfg.batch_size)]
    cache = cache_distributions(model, hooks, batches, cfg.teacher_topk)

    probe_set = sorted(probe_set, key=lambda e: len(e["prompt_ids"]))
    probe_batches = [collate(probe_set[i: i + cfg.batch_size],
                             tok.pad_token_id, device)
                     for i in range(0, len(probe_set), cfg.batch_size)]

    # Anchor batches: off-task text the guardrail rotates through, so the
    # unmasked model stays anchored beyond the prompts it is fitting.
    anchor_batches, anchor_cache = [], []
    if cfg.anchor_texts:
        anchors = []
        for t in cfg.anchor_texts:
            ids = tok(str(t), add_special_tokens=False)["input_ids"]
            ids = ids[: cfg.anchor_max_tokens]
            if len(ids) >= 2:
                anchors.append({"prompt_ids": ids[:1], "out_ids": ids[1:]})
        anchors.sort(key=lambda e: len(e["out_ids"]))
        # Anchor sequences are longer than train outputs; size their batches
        # to the same output-position budget so the guardrail's fp32
        # log-softmax transient stays no larger than a train step's.
        a_bs = max(1, (cfg.batch_size * cfg.max_new_tokens)
                   // cfg.anchor_max_tokens)
        anchor_batches = [collate(anchors[i: i + a_bs],
                                  tok.pad_token_id, device)
                          for i in range(0, len(anchors), a_bs)]
        anchor_cache = cache_distributions(model, hooks, anchor_batches,
                                           cfg.teacher_topk)
        log(f"[excise] guardrail anchors: {len(anchors)} texts in "
            f"{len(anchor_batches)} batches")

    # vocabulary support of the capability (for prune_vocab at export)
    support = set()
    for e in examples:
        support.update(e["prompt_ids"])
        support.update(e["out_ids"])
    for tp, ti, tr in cache:
        support.update(torch.unique(ti).cpu().tolist())
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        t = getattr(tok, attr, None)
        if t is not None:
            support.add(int(t))
    vocab_support = sorted(support)

    gate_init = cfg.gate_init
    if cfg.attr_warmstart:
        init = _attribution_init(model, hooks, batches, mlp_map, cfg,
                                 device, log)
        if init is not None:
            gate_init = init
    if use_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    gates = HardConcreteGates(mlp_map.n_layers, mlp_map.d_ff,
                              gate_init).to(device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg.lora_lr,
         "weight_decay": cfg.lora_weight_decay},
        {"params": gates.parameters(), "lr": cfg.gate_lr,
         "weight_decay": 0.0},
    ])

    # Decode-noise baseline: even the unmodified model does not reproduce
    # its own batched greedy targets perfectly (bf16 batch effects). The
    # probe stopper compares against the unmasked model AT THE SAME STEP,
    # never against an assumed 1.0.
    probe_base = tf_self_match(model, probe_batches, hooks, "off")
    log(f"[excise] probe_base={probe_base:.4f} (unmasked, teacher-forced)")

    # ---- adaptive descent
    lam, ema_kl, probe_dry = 0.0, 0.0, 0
    target = gates.expected_open().item()
    floor_reason = "max_steps"
    probe_trace, guardrail_trace = [], []
    last_pass = None                       # (open, log_alpha) rollback point
    g_int, g_count, ai = cfg.guardrail_every, 0, 0
    bi, border = 0, list(range(len(batches)))
    model.train()
    step = 0
    for step in range(1, cfg.max_steps + 1):
        if bi == 0:
            rng.shuffle(border)
        ids, labs, attn, pos, P, O = batches[border[bi]]
        tp, ti, tr = cache[border[bi]]
        bi = (bi + 1) % len(batches)

        hooks.sampled = gates.sample_all()
        hooks.mode = "sample"
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                       logits_to_keep=O + 1).logits
        s_lp, m = out_logprobs(logits, labs, P)
        kl = sparse_kl(s_lp, tp, ti, tr)
        ce = -s_lp.gather(-1, labs[:, P:][m].unsqueeze(-1)).mean()
        loss = kl + cfg.ce_weight * ce + lam * gates.expected_open()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        hooks.mode = "off"
        hooks.sampled = None

        if step % g_int == 0:
            g_count += 1
            use_anchor = anchor_batches and g_count % 2 == 0
            if use_anchor:
                g_ids, g_labs, g_attn, g_pos, g_P, g_O = anchor_batches[ai]
                g_tp, g_ti, g_tr = anchor_cache[ai]
                ai = (ai + 1) % len(anchor_batches)
            else:
                g_ids, g_labs, g_attn, g_pos, g_P, g_O = ids, labs, attn, pos, P, O
                g_tp, g_ti, g_tr = tp, ti, tr
            g_logits = model(input_ids=g_ids, attention_mask=g_attn,
                             position_ids=g_pos,
                             logits_to_keep=g_O + 1).logits
            g_slp, _ = out_logprobs(g_logits, g_labs, g_P)
            g_kl = sparse_kl(g_slp, g_tp, g_ti, g_tr)
            (cfg.guardrail_weight * g_kl).backward()
            guardrail_trace.append(
                {"step": step, "src": "anchor" if use_anchor else "train",
                 "kl": g_kl.item()})
            # Self-tune the cadence from the ANCHOR KL only: the train-batch
            # KL reads near-zero precisely when drift is off-distribution.
            if use_anchor:
                if g_kl.item() < 0.25 * cfg.kl_budget:
                    g_int = min(g_int * 2, 16)
                elif g_kl.item() > cfg.kl_budget:
                    g_int = max(g_int // 2, 2)

        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()

        ema_kl = 0.9 * ema_kl + 0.1 * kl.item()
        open_now = gates.expected_open().item()
        if (step > cfg.warmup_steps and ema_kl < cfg.kl_budget
                and (open_now - target) < 0.05):
            target = max(cfg.min_target, target * cfg.target_decay)
        lam = max(0.0, lam + cfg.dual_lr * (open_now - target))

        if step % 100 == 0:
            log(f"[excise] step={step} kl={kl.item():.4f} open={open_now:.4f} "
                f"target={target:.4f} g_int={g_int}")

        if open_now < cfg.probe_below and step % cfg.probe_every == 0:
            k_now = max(1, int(round(open_now * mlp_map.n_channels)))
            pmask = gates.topk_mask(k_now).to(device)
            pacc = tf_self_match(model, probe_batches, hooks, "mask", pmask)
            punm = tf_self_match(model, probe_batches, hooks, "off")
            probe_trace.append({"step": step, "open": open_now,
                                "self_match": pacc, "unmasked": punm})
            if pacc < punm - cfg.probe_tol:
                probe_dry += 1
            else:
                probe_dry = 0
                last_pass = (open_now, gates.log_alpha.detach().clone())
            log(f"[excise] probe step={step} open={open_now:.4f} "
                f"match={pacc:.4f} unmasked={punm:.4f} dry={probe_dry}")
            if probe_dry >= 2:
                floor_reason = "probe"
                break
        if target <= cfg.min_target and open_now <= cfg.min_target * 1.5:
            floor_reason = "min_target"
            break

    if floor_reason == "probe" and last_pass is not None:
        # Roll back to the last sparsity level that still passed, instead
        # of freezing the (already failing) level the stopper caught.
        open_back, alpha_back = last_pass
        gates.log_alpha.data.copy_(alpha_back)
        log(f"[excise] rolled gates back to last passing probe "
            f"(open={open_back:.4f})")

    floor = gates.expected_open().item()
    k_floor = max(1, int(round(floor * mlp_map.n_channels)))
    log(f"[excise] floor={floor:.4f} ({floor_reason}) at step {step}")

    # ---- polish under the hardened mask (guardrail and CE stay active: a
    # polish phase without the guardrail is exactly where the JSON run
    # drifted to 55.8% unmasked self-match)
    mask = gates.topk_mask(k_floor).to(device)
    for p in gates.parameters():
        p.requires_grad_(False)
    for pstep in range(1, cfg.polish_steps + 1):
        if bi == 0:
            rng.shuffle(border)
        ids, labs, attn, pos, P, O = batches[border[bi]]
        tp, ti, tr = cache[border[bi]]
        bi = (bi + 1) % len(batches)
        hooks.mode, hooks.mask = "mask", mask
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                       logits_to_keep=O + 1).logits
        s_lp, m = out_logprobs(logits, labs, P)
        kl = sparse_kl(s_lp, tp, ti, tr)
        ce = -s_lp.gather(-1, labs[:, P:][m].unsqueeze(-1)).mean()
        opt.zero_grad(set_to_none=True)
        (kl + cfg.ce_weight * ce).backward()
        hooks.mode, hooks.mask = "off", None
        if pstep % g_int == 0:
            g_count += 1
            use_anchor = anchor_batches and g_count % 2 == 0
            if use_anchor:
                g_ids, g_labs, g_attn, g_pos, g_P, g_O = anchor_batches[ai]
                g_tp, g_ti, g_tr = anchor_cache[ai]
                ai = (ai + 1) % len(anchor_batches)
            else:
                g_ids, g_labs, g_attn, g_pos, g_P, g_O = ids, labs, attn, pos, P, O
                g_tp, g_ti, g_tr = tp, ti, tr
            g_logits = model(input_ids=g_ids, attention_mask=g_attn,
                             position_ids=g_pos,
                             logits_to_keep=g_O + 1).logits
            g_slp, _ = out_logprobs(g_logits, g_labs, g_P)
            g_kl = sparse_kl(g_slp, g_tp, g_ti, g_tr)
            (cfg.guardrail_weight * g_kl).backward()
            guardrail_trace.append(
                {"step": step + pstep,
                 "src": "anchor" if use_anchor else "train",
                 "kl": g_kl.item()})
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()

    # ---- frontier + receipts
    frontier = []
    for b in sorted(set(list(cfg.eval_budgets) + [round(floor, 4)]),
                    reverse=True):
        if b < floor * 0.99:
            continue
        k = max(1, int(round(b * mlp_map.n_channels)))
        acc = self_match(model, tok, held, hooks, "mask",
                         gates.topk_mask(k).to(device), bs=eval_bs,
                         max_new_tokens=cfg.max_new_tokens,
                         scope=cfg.match_scope)
        frontier.append((b, acc))
        log(f"[excise] eval@{b:.2%} held self-match={acc:.4f}")

    # Layer-profile-matched random control: same number of channels kept in
    # each layer as the floor mask, chosen at random — a stronger null than
    # a global random draw.
    g = torch.Generator().manual_seed(cfg.seed)
    floor_mask = gates.topk_mask(k_floor)
    rand_mask = torch.zeros_like(floor_mask)
    for li in range(mlp_map.n_layers):
        k_li = int(floor_mask[li].sum())
        if k_li:
            idx = torch.randperm(mlp_map.d_ff, generator=g)[:k_li]
            rand_mask[li, idx] = 1.0
    rand_acc = self_match(model, tok, held, hooks, "mask",
                          rand_mask.to(device), bs=eval_bs,
                          max_new_tokens=cfg.max_new_tokens,
                          scope=cfg.match_scope)
    unmasked = self_match(model, tok, held, hooks, "off", bs=eval_bs,
                          max_new_tokens=cfg.max_new_tokens,
                          scope=cfg.match_scope)
    base = self_match(model, tok, held, hooks, "off", bs=eval_bs,
                      max_new_tokens=cfg.max_new_tokens,
                      use_adapter=False, scope=cfg.match_scope)
    receipts = {
        "n_channels": mlp_map.n_channels,
        "floor": floor, "floor_reason": floor_reason, "steps": step,
        "probe_src": probe_src,
        "probe_base": probe_base,
        "base_self_match": base,
        "unmasked_self_match": unmasked,
        "random_mask_self_match": rand_acc,
        "held_n": len(held),
        "probe_trace": probe_trace,
        "guardrail_trace": guardrail_trace,
        "vocab_support_size": len(vocab_support),
        "wall_minutes": (time.time() - t0) / 60,
    }
    log(f"[excise] base={base:.4f} unmasked={unmasked:.4f} "
        f"random-control={rand_acc:.4f} "
        f"({receipts['wall_minutes']:.1f} min)")
    return ExtractionResult(model, tok, gates, hooks, mlp_map, cfg, floor,
                            floor_reason, frontier, receipts, held,
                            vocab_support)
