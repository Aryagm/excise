# excise v0.1.0

Extract one capability from an open LLM into a smaller, deployable model.
One command. No training data — the model teaches itself.

## Highlights

- **One-shot extraction**: learnable channel gates + a small adapter train
  jointly in a single run; an adaptive controller finds the minimal
  substrate automatically, paced by generation probes (not loss curves).
- **Label-free**: you provide prompts; the frozen model's own outputs are
  the target.
- **Real export**: `export_sliced()` physically deletes masked channels —
  verified numerically equivalent to masking.
- **Receipts**: every run reports a random-mask control, unmasked-drift
  check, and the full probe trace.

## Measured (single RTX 4090)

| Capability | Model | Kept | Fidelity | Exported |
|---|---|---|---|---|
| Arithmetic | Qwen2.5-Math-1.5B | 2.9% of MLP | 97% | 1.54B → 0.42B |
| Arithmetic (few-shot) | SmolLM2-1.7B | 7.4% | 97.2% | 1.75B → 0.59B |
| JSON extraction | Qwen2.5-1.5B-Instruct | 33.9% | 90.0% | 1.58B → 0.78B |
| Function calling (BFCL) | Qwen3-4B | 40% | ~76% | ~2.4B projected |

## Install

```bash
pip install excise
```

## Known limitations (honest list)

- Memory shrinks unconditionally; latency gains depend on workload (short
  large-batch generation barely speeds up).
- Polish phase lacks the guardrail term — receipts flag the resulting
  unmasked drift on some tasks; fix slated for v0.2.
- Sliced checkpoints need re-slice-on-load (heterogeneous layer widths).
- Verbatim self-match undercounts semantic fidelity.

Method, results, and negative results: see the paper in `paper/`.
Built on the capability-extraction framing of Mishra & Pagare's PRISM.
