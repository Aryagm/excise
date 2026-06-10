"""Extraction configuration. Defaults are the values validated in the
arithmetic battery (3 seeds) and the Qwen3-4B function-calling run."""

from dataclasses import dataclass, field, asdict


@dataclass
class ExtractConfig:
    # adapter
    lora_r: int = 32
    lora_alpha: int = 32
    lora_lr: float = 1e-4

    # gates
    gate_lr: float = 5e-2
    gate_init: float = 3.0

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
    probe_every: int = 150
    probe_below: float = 0.70
    probe_n: int = 64
    probe_tol: float = 0.08

    # guardrail: keep the unmasked model anchored to the base
    guardrail_every: int = 4
    guardrail_weight: float = 1.0

    # fidelity metric: "exact" compares full decoded outputs verbatim;
    # "first_line" compares only up to the first newline — use for tasks
    # where the answer ends early and the tail of the continuation is
    # unconstrained (e.g. "12 + 34 =" -> "46\n<anything>")
    match_scope: str = "exact"

    # finishing
    polish_steps: int = 150
    eval_budgets: tuple = (0.5, 0.3, 0.2, 0.1, 0.05)
    held_frac: float = 0.2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["eval_budgets"] = list(d["eval_budgets"])
        return d
