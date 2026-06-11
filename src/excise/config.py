"""Extraction configuration. Defaults are the values validated in the
arithmetic battery (3 seeds) and the Qwen3-4B function-calling run, updated
for the v0.2 calibration fixes (bucketed KL, probe_base-relative floor
detection, attribution warm-start)."""

from dataclasses import dataclass, field, asdict


@dataclass
class ExtractConfig:
    # adapter
    lora_r: int = 32
    lora_alpha: int = 32
    lora_lr: float = 1e-4
    # Dropout + weight decay regularize the adapter toward the base model
    # (LoRA delta = 0 IS the base model, so decay anchors in parameter
    # space for free). Worth raising (~0.05 / 0.01) on small prompt sets
    # (<1k), where r=32 all-linear has far more capacity than data.
    lora_dropout: float = 0.0
    lora_weight_decay: float = 0.0

    # gates
    gate_lr: float = 5e-2
    gate_init: float = 3.0
    # Warm-start gate init from |grad x act| attribution percentiles
    # (init spread 2.0..5.0), matching the validated research pipeline.
    attr_warmstart: bool = True
    attr_examples: int = 256

    # training
    max_steps: int = 3000
    warmup_steps: int = 100
    batch_size: int = 8
    ce_weight: float = 0.05
    teacher_topk: int = 128
    max_new_tokens: int = 128
    max_prompt_tokens: int = 1024
    gradient_checkpointing: bool | None = None   # None = auto (>3B params)
    seed: int = 42

    # adaptive sparsity controller
    kl_budget: float = 0.025
    target_decay: float = 0.993
    min_target: float = 0.008
    dual_lr: float = 0.1

    # behavior probes (floor detection). Distribution-level KL alone is
    # miscalibrated at high sparsity; generation probes are load-bearing.
    # Probes run teacher-forced (see probes.tf_self_match), are compared
    # against the unmasked model measured at the same step (so adapter
    # drift is not misread as masking damage), and on probe-stop the gates
    # roll back to the last passing probe snapshot.
    # probe_tol is the tolerated drop below the MEASURED unmasked baseline
    # (a stricter denominator than the 1.0 v0.1 assumed). With probe_n=64
    # verbatim probes, binomial noise alone is ~4 points; do not set the
    # tolerance inside the noise band.
    probe_every: int = 150
    probe_below: float = 0.70
    probe_n: int = 64
    probe_tol: float = 0.08
    # Carve the probe set out of train (excluded from gradient batches) when
    # there is enough data, so the floor is decided on non-memorized
    # prompts. Falls back to in-train probes below 4x probe_n examples.
    probe_holdout: bool = True

    # guardrail: keep the unmasked model anchored to the base. With
    # anchor_texts set, the guardrail alternates between the current train
    # batch and a rotating batch of off-task text — the train batch alone
    # can only anchor points the adapter is already fitting, which is why
    # drift shows up on held-out data first. The cadence self-tunes from
    # the measured anchor KL (only when anchors exist).
    guardrail_every: int = 4
    guardrail_weight: float = 1.0
    anchor_texts: list | None = None
    anchor_max_tokens: int = 128

    # fidelity metric: "exact" compares full decoded outputs verbatim;
    # "first_line" compares only up to the first newline — use for tasks
    # where the answer ends early and the tail of the continuation is
    # unconstrained (e.g. "12 + 34 =" -> "46\n<anything>")
    match_scope: str = "exact"

    # finishing. Polish keeps the guardrail and CE terms active (a polish
    # phase without the guardrail is where the JSON run drifted). Each
    # off-floor frontier budget additionally gets a brief polish of its own
    # from the floor-polished snapshot before evaluation — without it the
    # floor polish specializes the adapter and off-floor budgets read
    # artificially low (set 0 to restore the raw post-polish frontier).
    polish_steps: int = 150
    frontier_polish_steps: int = 60
    eval_budgets: tuple = (0.5, 0.3, 0.2, 0.1, 0.05)
    held_frac: float = 0.2
    # generation batch size for probes/evals (no grads, so it can be much
    # larger than the training batch); None = 4x batch_size
    eval_batch_size: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["eval_budgets"] = list(d["eval_budgets"])
        if d["anchor_texts"]:
            d["anchor_texts"] = f"<{len(d['anchor_texts'])} texts>"
        return d
