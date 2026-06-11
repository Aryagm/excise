"""v0.1-code arithmetic baseline: run the UNMODIFIED v0.1 library (checked
out to /root/substrate_v01/src, first on sys.path) on the exact dogfood
config, so v0.2 changes are compared against ground truth rather than
memory. Saves the same receipts/summary layout as v02_validation."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/substrate_v01/src")

import excise  # noqa: E402
from excise import ExtractConfig, extract, param_count  # noqa: E402

assert "substrate_v01" in excise.__file__, excise.__file__
print("using", excise.__file__, excise.__version__)

prompts = [f"{a} + {b} =" for a in range(10, 100) for b in range(10, 100)]
t0 = time.time()
result = extract(
    "Qwen/Qwen2.5-Math-1.5B",
    prompts=prompts,
    config=ExtractConfig(max_new_tokens=4, batch_size=64, kl_budget=0.02,
                         probe_below=0.12, max_prompt_tokens=32),
)
out = result.save("/root/out/arith_v01_baseline")
(Path(out) / "report.txt").write_text(result.report())
r = result.receipts
summary = {
    "task": "arith_v01_baseline",
    "floor": result.floor, "floor_reason": result.floor_reason,
    "frontier": result.frontier,
    "unmasked_self_match": r["unmasked_self_match"],
    "random_mask_self_match": r["random_mask_self_match"],
    "steps": r["steps"], "wall_minutes": r["wall_minutes"],
    "total_minutes": (time.time() - t0) / 60,
    "params_before": param_count(result._model),
}
sliced = result.export_sliced()
summary["params_after_slice"] = param_count(sliced)
(Path(out) / "summary.json").write_text(json.dumps(summary, indent=2))
print("==== SUMMARY", json.dumps(summary))
