# excise

Extract one capability from an open LLM into a smaller, deployable model.
One command. No training data — the model teaches itself.

```python
from excise import extract

result = extract("Qwen/Qwen2.5-Math-1.5B", prompts=my_prompts)  # just prompts
print(result.report())     # frontier, floor, integrity receipts
result.save("substrate/")  # mask + adapter + receipts
small = result.export_sliced()   # physically smaller model
```

## What it does

Most of a language model is not needed for any single behavior. `excise`
finds the minimal set of MLP channels that carries one capability — the one
your prompts exercise — and trains a small adapter that concentrates the
behavior into those channels, **jointly, in a single run**. The rest of the
network can then be deleted.

Measured results (single RTX 4090, details in the paper):

| Capability | Model | Kept channels | Fidelity | Exported size |
|---|---|---|---|---|
| 2-digit arithmetic | Qwen2.5-Math-1.5B | **2.9%** of MLP | 97% of base accuracy | **1.54B → 0.42B params (3.7×)** |
| Function calling (BFCL) | Qwen3-4B | 40% of MLP | ~76% verbatim output match | 4.0B → ~2.4B projected |

- **Label-free.** The target is the model's own unmasked output distribution.
  You provide prompts; nothing else.
- **The whole size-fidelity frontier from one run.** An adaptive controller
  closes channels as fast as behavior allows and stops at the floor itself.
- **Behavior probes, not loss curves.** Distribution-level KL reads healthy
  while generation quality collapses at high sparsity (we measured a 35-point
  silent failure). The controller decides by actually generating.
- **Receipts.** Every extraction reports a random-mask control, unmasked-model
  drift, and the full probe trace. If your task was trivially easy or the
  extraction is invalid, the report says so.
- **Real export.** `export_sliced()` deletes the dead weights. Slicing is
  mathematically equivalent to masking for gated MLPs — fidelity is identical
  to the masked evaluation (verified to one eval example in 810).

## Honest caveats

- **Memory shrinks unconditionally; speed depends on workload.** MLP slicing
  pays most in long-decode, small-batch settings and on larger models. On a
  1.5B model with 4-token outputs at batch 512 we measured only 1.11×
  generation speedup despite 3.7× fewer parameters (attention and the
  vocabulary projection dominate there).
- The extracted model is a specialist. Out-of-capability behavior degrades —
  that's the point — so route accordingly.
- Capabilities differ in how small they can go: narrow skills (arithmetic)
  compress to a few percent; broad ones (function calling) need a substantial
  fraction of channels.
- Sliced checkpoints have heterogeneous layer widths: persist with
  `torch.save(model.state_dict())` plus the saved mask (reload = load base,
  re-slice, load state dict).

## Install

```bash
pip install excise          # PyPI release pending; for now:
pip install git+https://github.com/e3group/excise
```

## CLI

```bash
excise extract --model Qwen/Qwen2.5-Math-1.5B --prompts prompts.txt \
    --out substrate/ --slice
```

## How it works (short version)

Hard-concrete L0 gates on every MLP intermediate channel train jointly with a
rank-32 LoRA adapter under a forward-KL leash to the frozen base model.
A Lagrangian controller lowers the open-channel target whenever the KL stays
under budget; generation probes veto the descent when behavior degrades; a
guardrail term keeps the *unmasked* model anchored to the base, so the
adapter can't cheat by becoming a different model. After the floor is found,
a brief polish under the hardened binary mask removes the stochastic-gate
train/eval mismatch.

Builds on the capability-extraction framing of Mishra & Pagare's
[PRISM](https://github.com/e-xperiments/prism-capability-extraction) — the
extraction contract and recovery metric come from their work; credit to them
for posing the problem precisely. On their arithmetic benchmark, one
`excise` run matches or beats their staged pipeline's best hand-tuned result
at every budget.

## License

MIT
