import math

import pytest
import torch

from excise.arch import find_mlps
from excise.gates import HardConcreteGates


def tiny_model(arch="qwen2"):
    from transformers import AutoConfig, AutoModelForCausalLM
    kw = dict(vocab_size=128, hidden_size=32, intermediate_size=64,
              num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=256)
    if arch == "qwen2":
        from transformers import Qwen2Config
        cfg = Qwen2Config(**kw)
    elif arch == "llama":
        from transformers import LlamaConfig
        cfg = LlamaConfig(**kw)
    else:
        raise ValueError(arch)
    return AutoModelForCausalLM.from_config(cfg)


@pytest.mark.parametrize("arch", ["qwen2", "llama"])
def test_arch_registry(arch):
    m = tiny_model(arch)
    mm = find_mlps(m)
    assert mm.n_layers == 2
    assert mm.d_ff == 64
    assert mm.n_channels == 128


def test_gates_topk_and_open():
    g = HardConcreteGates(2, 8, init=3.0)
    assert 0.9 < g.expected_open().item() <= 1.0
    g.log_alpha.data[0, :4] = -10.0
    mask = g.topk_mask(12)
    assert mask.sum() == 12
    assert mask[0, :4].sum() == 0          # lowest scores excluded
    s = g.sample_all()
    assert len(s) == 2 and s[0].shape == (8,)
    assert (s[0] >= 0).all() and (s[0] <= 1).all()


def test_gates_grad_flows():
    g = HardConcreteGates(1, 4, init=0.0)
    s = g.sample_all()[0]
    (s.sum()).backward()
    assert g.log_alpha.grad is not None
