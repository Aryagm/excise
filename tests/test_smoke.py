"""End-to-end smoke test on a tiny random model, CPU, ~1 minute.

A random model has no capability to preserve, so recovery numbers are
meaningless here — what this verifies is mechanics: the pipeline runs, the
controller descends, artifacts save, and (the meaningful assertion) the
sliced model is numerically equivalent to the masked model.
"""

import string

import pytest
import torch

from excise import ExtractConfig, extract, param_count


class CharTokenizer:
    """Minimal whitespace-free char tokenizer satisfying the API surface
    excise uses (call, decode, pad/eos ids, padding_side)."""

    def __init__(self):
        chars = string.ascii_lowercase + string.digits + " +=."
        self.vocab = {c: i + 2 for i, c in enumerate(chars)}
        self.inv = {i: c for c, i in self.vocab.items()}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"

    def __call__(self, text, add_special_tokens=False, **kw):
        if isinstance(text, str):
            return {"input_ids": [self.vocab.get(c, 2) for c in text]}
        raise NotImplementedError

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.inv.get(i, "") for i in ids
                       if i not in (self.pad_token_id, self.eos_token_id))


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, AutoModelForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    model = AutoModelForCausalLM.from_config(cfg)
    model.generation_config.pad_token_id = 0
    return model, CharTokenizer()


def test_end_to_end(tiny, tmp_path):
    model, tok = tiny
    prompts = [f"{a} + {b} =" for a in range(4, 20) for b in range(4, 8)]
    cfg = ExtractConfig(max_steps=40, warmup_steps=5, batch_size=8,
                        polish_steps=5, probe_every=10, probe_n=8,
                        max_new_tokens=4, teacher_topk=16,
                        eval_budgets=(0.5,), target_decay=0.9,
                        kl_budget=10.0,  # random model: let it descend
                        gradient_checkpointing=False)
    result = extract(model, prompts, tokenizer=tok, config=cfg, device="cpu",
                     log=lambda *a: None)

    assert 0 < result.floor < 1
    assert result.frontier and all(0 <= a <= 1 for _, a in result.frontier)
    assert "probe_trace" in result.receipts

    out = result.save(tmp_path / "artifact")
    assert (out / "gates.npz").exists()
    assert (out / "receipts.json").exists()
    assert (out / "adapter").exists()

    # the meaningful equivalence assertion: masked forward == sliced forward
    budget = max(result.floor, 0.3)
    k = max(1, int(round(budget * result._mlp_map.n_channels)))
    mask = result._gates.topk_mask(k)
    ids = torch.tensor([[5, 6, 7, 8]])
    result._hooks.mode, result._hooks.mask = "mask", mask
    with torch.no_grad():
        masked_logits = result._model(input_ids=ids).logits
    result._hooks.mode, result._hooks.mask = "off", None

    before = param_count(result._model)
    sliced = result.export_sliced(budget=budget)
    after = param_count(sliced)
    assert after < before
    with torch.no_grad():
        sliced_logits = sliced(input_ids=ids).logits
    assert torch.allclose(masked_logits, sliced_logits, atol=1e-4), \
        "sliced model must be numerically equivalent to masked model"
