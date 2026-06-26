# excise v0.2.0

Extract one capability from an open LLM into a smaller, deployable model.
One command. No labels: the model teaches itself.

## Highlights

- **One-shot extraction**: learnable channel gates and a small adapter train
  jointly in a single run; an adaptive controller reports the stopping point
  it finds, paced by behavior probes rather than loss curves.
- **Calibrated receipts**: dev-split probes are scored against the unmasked
  model measured at the same step, with exit reasons recorded for probe,
  lower-bound, and step-cap stops.
- **Fixed sparse-KL guardrail**: binned KL keeps the residual mass, off-task
  anchor text catches unmasked drift, and the guardrail remains active during
  polish.
- **Real export**: `export_sliced()` physically deletes masked channels, and
  vocabulary pruning compounds the memory reduction when the capability uses
  a small token support.

## Measured (single RTX 4090)

| Capability | Model | Kept | Fidelity | Exported |
|---|---|---|---|---|
| 2-digit arithmetic | Qwen2.5-Math-1.5B | 0.7% of MLP | 91% verbatim match | 1.54B -> 0.17B |
| Arithmetic, few-shot | SmolLM2-1.7B | 7.4% of MLP | 97.2% verbatim match | 1.75B -> 0.59B |
| JSON extraction | Qwen2.5-1.5B-Instruct | 1.9% of MLP | 97.2% verbatim match | 1.58B -> 0.21B |
| Function calling (BFCL) | Qwen3-4B | 40% of MLP | ~76% verbatim match | ~2.4B projected |

The JSON result is the v0.2 validation story: the old sparse-KL cache and
guardrail-free polish path produced catastrophic unmasked drift, while v0.2's
binned KL, off-task anchors, and guarded polish cut drift to under two points
on the 3,000-prompt diverse run.

## Install

```bash
pip install "excise @ git+https://github.com/Aryagm/excise.git@v0.2.0"
```

## Known limitations

- Memory shrinks unconditionally; latency gains depend on model size,
  batching, and runtime compilation.
- Reported floors are controller stopping points. Lower-bound and step-cap
  exits are upper bounds on the behavioral-collapse minimum.
- Sliced checkpoints have heterogeneous layer widths and should be loaded with
  `excise.load_sliced(dir)`.
- Verbatim self-match undercounts semantic fidelity.

Method, results, and negative results: see the paper in `paper/`.
Built on the capability-extraction framing of Mishra & Pagare's PRISM.
