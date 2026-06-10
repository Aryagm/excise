"""Extract single-turn function calling from Qwen3-4B using BFCL prompts.

Label-free: targets are the model's own greedy tool calls. ~60 min on a
24GB GPU (gradient checkpointing auto-enables above 3B params).

Known limitation (measured): with only ~320 training prompts the adapter
drifts on unseen prompts (unmasked self-match ~86%). More diverse prompts
raise every number; see the receipts.
"""

import json
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from excise import ExtractConfig, extract

MODEL = "Qwen/Qwen3-4B"
DATA_URL = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
            "berkeley-function-call-leaderboard/bfcl_eval/data/"
            "BFCL_v4_simple_python.json")

path = Path("bfcl_simple.json")
if not path.exists():
    urllib.request.urlretrieve(DATA_URL, path)

tok = AutoTokenizer.from_pretrained(MODEL)
prompts = []
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    fn = r["function"][0] if isinstance(r["function"], list) else r["function"]
    params = json.loads(json.dumps(fn.get("parameters", {}))
                        .replace('"dict"', '"object"'))
    tool = {"type": "function", "function": {
        "name": fn["name"], "description": fn.get("description", ""),
        "parameters": params}}
    q = r["question"]
    while isinstance(q, list):
        q = q[0]
    msgs = [q] if isinstance(q, dict) else [{"role": "user", "content": str(q)}]
    prompts.append(tok.apply_chat_template(
        msgs, tools=[tool], tokenize=False, add_generation_prompt=True,
        enable_thinking=False))

result = extract(MODEL, prompts=prompts, tokenizer=tok,
                 config=ExtractConfig(batch_size=8, probe_below=0.7))
print(result.report())
result.save("out/function_calling")
