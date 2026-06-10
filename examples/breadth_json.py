"""Breadth: a practitioner capability — structured JSON extraction
(Qwen2.5-1.5B-Instruct). Prompts are chat-formatted before being passed in;
excise itself is task- and template-agnostic.
"""

import random

import torch
from transformers import AutoTokenizer

from excise import ExtractConfig, extract, param_count

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

NAMES = ["John", "Maria", "Wei", "Aisha", "Carlos", "Yuki", "Omar", "Elena",
         "Raj", "Sofia", "Liam", "Nadia", "Pedro", "Hana", "Igor", "Zara"]
CITIES = ["Paris", "Tokyo", "Lagos", "Lima", "Oslo", "Delhi", "Cairo",
          "Seoul", "Quito", "Porto", "Hanoi", "Perth"]
JOBS = ["teacher", "engineer", "nurse", "chef", "pilot", "farmer", "artist",
        "lawyer"]

rng = random.Random(0)
prompts = []
for _ in range(600):
    n, c, j = rng.choice(NAMES), rng.choice(CITIES), rng.choice(JOBS)
    a = rng.randint(19, 79)
    text = (f"{n} is a {a}-year-old {j} living in {c}.")
    msgs = [{"role": "user",
             "content": "Extract name, age, job, and city as a JSON object. "
                        f"Reply with only the JSON.\n\n{text}"}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True))

result = extract(
    MODEL,
    prompts=prompts,
    tokenizer=tok,
    config=ExtractConfig(max_new_tokens=64, batch_size=24, kl_budget=0.025,
                         probe_below=0.6, max_prompt_tokens=128),
)
print(result.report())
result.save("out/json_extraction")
before = param_count(result._model)
sliced = result.export_sliced()
print(f"sliced: {before/1e9:.2f}B -> {param_count(sliced)/1e9:.2f}B params")
