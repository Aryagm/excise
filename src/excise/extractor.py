"""The extraction loop: joint gate + adapter training with an adaptive
sparsity controller, behavior-probe floor detection, and a guardrail that
keeps the unmasked model anchored to the base."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from .arch import find_mlps
from .config import ExtractConfig
from .data import collate, load_prompts
from .export import slice_model
from .gates import HardConcreteGates
from .hooks import GateHooks
from .probes import self_match
from .teacher import cache_distributions, generate_targets, sparse_kl


class ExtractionResult:
    def __init__(self, model, tok, gates, hooks, mlp_map, config,
                 floor: float, floor_reason: str, frontier: list,
                 receipts: dict, held: list):
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
        self._sliced = False

    def save(self, out_dir: str):
        """Persist mask, adapter, config, and receipts. Call before
        export_sliced()."""
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
        if hasattr(self._model, "save_pretrained") and not self._sliced:
            self._model.save_pretrained(str(out / "adapter"))
        return out

    def export_sliced(self, budget: float | None = None):
        """Merge the adapter and physically delete masked channels.
        Destructive: the result's model becomes the sliced model. Returns a
        plain transformers model. Slicing is mathematically equivalent to
        zero-isolation masking, so fidelity matches the masked evals."""
        budget = budget or self.floor
        k = max(1, int(round(budget * self._mlp_map.n_channels)))
        mask = self._gates.topk_mask(k)
        model = self._model
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        slice_model(model, mask)
        self._model = model
        self._sliced = True
        return model

    def masked_eval(self, budget: float) -> float:
        """Held-out self-match at an arbitrary budget."""
        k = max(1, int(round(budget * self._mlp_map.n_channels)))
        mask = self._gates.topk_mask(k).to(
            next(self._model.parameters()).device)
        return self_match(self._model, self._tok, self._held, self._hooks,
                          "mask", mask,
                          max_new_tokens=self.config.max_new_tokens)

    def report(self) -> str:
        r = self.receipts
        lines = [
            "excise extraction report",
            "=" * 40,
            f"channels:            {self._mlp_map.n_channels}",
            f"floor:               {self.floor:.2%} ({self.floor_reason})",
            "frontier (held-out self-match):",
        ]
        for b, acc in self.frontier:
            lines.append(f"  keep {b:>6.2%}  ->  {acc:.1%}")
        lines += [
            f"unmasked drift:      {r['unmasked_self_match']:.1%} self-match "
            "(should be near 100%)",
            f"random-mask control: {r['random_mask_self_match']:.1%} "
            "(should be near 0%)",
            f"probe trace:         {len(r['probe_trace'])} probes, "
            f"min {min((p['self_match'] for p in r['probe_trace']), default=1.0):.1%}",
            f"wall time:           {r['wall_minutes']:.1f} min",
        ]
        return "\n".join(lines)


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
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        bias="none", target_modules="all-linear", task_type="CAUSAL_LM"))
    if use_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    mlp_map = find_mlps(model)
    hooks = GateHooks(mlp_map)
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
                     bs=cfg.batch_size)
    examples = [e for e in examples if e["out_ids"]]

    rng = random.Random(cfg.seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    n_held = max(1, int(len(examples) * cfg.held_frac))
    held = [examples[i] for i in order[:n_held]]
    train = [examples[i] for i in order[n_held:]]
    train.sort(key=lambda e: len(e["prompt_ids"]))
    stride = max(1, len(train) // cfg.probe_n)
    probe_set = train[::stride][:cfg.probe_n]
    log(f"[excise] train={len(train)} held={len(held)} probe={len(probe_set)}")

    batches = [collate(train[i: i + cfg.batch_size], tok.pad_token_id, device)
               for i in range(0, len(train), cfg.batch_size)]
    cache = cache_distributions(model, hooks, batches, cfg.teacher_topk)

    gates = HardConcreteGates(mlp_map.n_layers, mlp_map.d_ff,
                              cfg.gate_init).to(device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg.lora_lr},
        {"params": gates.parameters(), "lr": cfg.gate_lr},
    ], weight_decay=0.0)

    # ---- adaptive descent
    lam, ema_kl, probe_dry = 0.0, 0.0, 0
    target = gates.expected_open().item()
    floor_reason = "max_steps"
    probe_trace = []
    bi, border = 0, list(range(len(batches)))
    model.train()
    step = 0
    for step in range(1, cfg.max_steps + 1):
        if bi == 0:
            rng.shuffle(border)
        ids, labs, attn, pos, P, O = batches[border[bi]]
        tp, ti = cache[border[bi]]
        bi = (bi + 1) % len(batches)

        hooks.sampled = gates.sample_all()
        hooks.mode = "sample"
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                       logits_to_keep=O + 1).logits
        kl = sparse_kl(logits, labs, P, tp, ti)
        m = labs[:, P:] != -100
        ce = -F.log_softmax(logits[:, :-1][m].float(), -1).gather(
            -1, labs[:, P:][m].unsqueeze(-1)).mean()
        loss = kl + cfg.ce_weight * ce + lam * gates.expected_open()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        hooks.mode = "off"
        hooks.sampled = None

        if step % cfg.guardrail_every == 0:
            g_logits = model(input_ids=ids, attention_mask=attn,
                             position_ids=pos, logits_to_keep=O + 1).logits
            (cfg.guardrail_weight * sparse_kl(g_logits, labs, P, tp, ti)
             ).backward()

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
                f"target={target:.4f}")

        if open_now < cfg.probe_below and step % cfg.probe_every == 0:
            k_now = max(1, int(round(open_now * mlp_map.n_channels)))
            pmask = gates.topk_mask(k_now).to(device)
            pacc = self_match(model, tok, probe_set, hooks, "mask", pmask,
                              bs=cfg.batch_size,
                              max_new_tokens=cfg.max_new_tokens)
            probe_trace.append({"step": step, "open": open_now,
                                "self_match": pacc})
            probe_dry = probe_dry + 1 if pacc < 1.0 - cfg.probe_tol else 0
            log(f"[excise] probe step={step} open={open_now:.4f} "
                f"match={pacc:.4f} dry={probe_dry}")
            if probe_dry >= 2:
                floor_reason = "probe"
                break
        if target <= cfg.min_target and open_now <= cfg.min_target * 1.5:
            floor_reason = "min_target"
            break

    floor = gates.expected_open().item()
    k_floor = max(1, int(round(floor * mlp_map.n_channels)))
    log(f"[excise] floor={floor:.4f} ({floor_reason}) at step {step}")

    # ---- polish under the hardened mask
    mask = gates.topk_mask(k_floor).to(device)
    for p in gates.parameters():
        p.requires_grad_(False)
    for _ in range(cfg.polish_steps):
        if bi == 0:
            rng.shuffle(border)
        ids, labs, attn, pos, P, O = batches[border[bi]]
        tp, ti = cache[border[bi]]
        bi = (bi + 1) % len(batches)
        hooks.mode, hooks.mask = "mask", mask
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                       logits_to_keep=O + 1).logits
        kl = sparse_kl(logits, labs, P, tp, ti)
        opt.zero_grad(set_to_none=True)
        kl.backward()
        hooks.mode, hooks.mask = "off", None
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
                         gates.topk_mask(k).to(device), bs=cfg.batch_size,
                         max_new_tokens=cfg.max_new_tokens)
        frontier.append((b, acc))
        log(f"[excise] eval@{b:.2%} held self-match={acc:.4f}")

    g = torch.Generator().manual_seed(cfg.seed)
    rand_mask = torch.zeros(mlp_map.n_channels)
    rand_mask[torch.randperm(mlp_map.n_channels, generator=g)[:k_floor]] = 1.0
    rand_acc = self_match(model, tok, held[: min(len(held), 64)], hooks,
                          "mask",
                          rand_mask.view(mlp_map.n_layers, -1).to(device),
                          bs=cfg.batch_size,
                          max_new_tokens=cfg.max_new_tokens)
    unmasked = self_match(model, tok, held, hooks, "off",
                          bs=cfg.batch_size,
                          max_new_tokens=cfg.max_new_tokens)
    receipts = {
        "n_channels": mlp_map.n_channels,
        "floor": floor, "floor_reason": floor_reason, "steps": step,
        "unmasked_self_match": unmasked,
        "random_mask_self_match": rand_acc,
        "probe_trace": probe_trace,
        "wall_minutes": (time.time() - t0) / 60,
    }
    log(f"[excise] unmasked={unmasked:.4f} random-control={rand_acc:.4f} "
        f"({receipts['wall_minutes']:.1f} min)")
    return ExtractionResult(model, tok, gates, hooks, mlp_map, cfg, floor,
                            floor_reason, frontier, receipts, held)
