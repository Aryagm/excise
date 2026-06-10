"""Breadth: Llama-architecture model (SmolLM2-1.7B, LlamaForCausalLM).

Few-shot 2-digit addition; first_line matching since the answer ends at the
first newline.
"""

import torch

from excise import ExtractConfig, extract, param_count

FEWSHOT = "12 + 34 = 46\n8 + 71 = 79\n"
prompts = [f"{FEWSHOT}{a} + {b} =" for a in range(10, 100)
           for b in range(10, 100)]

result = extract(
    "HuggingFaceTB/SmolLM2-1.7B",
    prompts=prompts[::2],            # 4050 prompts is plenty
    config=ExtractConfig(max_new_tokens=4, batch_size=48, kl_budget=0.02,
                         probe_below=0.25, max_prompt_tokens=48,
                         match_scope="first_line"),
)
print(result.report())
result.save("out/smollm_arithmetic")
before = param_count(result._model)
sliced = result.export_sliced()
print(f"sliced: {before/1e9:.2f}B -> {param_count(sliced)/1e9:.2f}B params")
