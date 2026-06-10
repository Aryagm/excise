"""Extract 2-digit addition from Qwen2.5-Math-1.5B.

Expected (RTX 4090, ~12 min): floor around 3% of MLP channels, held-out
self-match >95% at 5%, sliced export ~1.54B -> ~0.42B params.
"""

import torch

from excise import ExtractConfig, extract, param_count

prompts = [f"{a} + {b} =" for a in range(10, 100) for b in range(10, 100)]

result = extract(
    "Qwen/Qwen2.5-Math-1.5B",
    prompts=prompts,
    config=ExtractConfig(max_new_tokens=4, batch_size=64, kl_budget=0.02,
                         probe_below=0.12, max_prompt_tokens=32),
)
print(result.report())
result.save("out/arithmetic")

before = param_count(result._model)
sliced = result.export_sliced()
print(f"sliced: {before/1e9:.2f}B -> {param_count(sliced)/1e9:.2f}B params")
torch.save(sliced.state_dict(), "out/arithmetic/sliced_state_dict.pt")
