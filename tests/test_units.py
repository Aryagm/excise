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


def test_gates_tensor_init():
    init = torch.linspace(2.0, 5.0, 16).view(2, 8)
    g = HardConcreteGates(2, 8, init=init)
    assert torch.allclose(g.log_alpha.data, init)
    with pytest.raises(ValueError):
        HardConcreteGates(2, 4, init=init)


def _toy_cache(vocab=50, positions=16, topk=8, seed=0):
    """Random teacher distribution + its top-k cache, as cache_distributions
    would store it (un-renormalized probs + residual mass)."""
    gen = torch.Generator().manual_seed(seed)
    t_logits = torch.randn(positions, vocab, generator=gen) * 2
    t_probs = torch.softmax(t_logits, -1)
    tp, ti = torch.topk(t_probs, topk, dim=-1)
    tr = (1.0 - tp.sum(-1)).clamp_min(0.0)
    return t_logits, tp, ti, tr


def test_sparse_kl_zero_at_teacher():
    """The binned KL must be ~0 when the student equals the teacher —
    the renormalized variant reads -log(coverage) there instead."""
    from excise.teacher import sparse_kl
    t_logits, tp, ti, tr = _toy_cache()
    s_lp = torch.log_softmax(t_logits, -1)
    loss = sparse_kl(s_lp, tp, ti, tr)
    assert abs(loss.item()) < 1e-5
    # and the old renormalized form is biased exactly by -log(coverage)
    tp_hat = tp / tp.sum(-1, keepdim=True)
    old = (tp_hat * (tp_hat.clamp_min(1e-9).log()
                     - s_lp.gather(-1, ti))).sum(-1).mean()
    expected_bias = (-tp.sum(-1).log()).mean()
    assert abs(old.item() - expected_bias.item()) < 1e-5
    assert old.item() > 0.01


def test_sparse_kl_positive_and_finite_when_student_sharpens():
    from excise.teacher import sparse_kl
    t_logits, tp, ti, tr = _toy_cache()
    s_lp = torch.log_softmax(t_logits * 3, -1)      # sharper student
    loss = sparse_kl(s_lp, tp, ti, tr)
    assert loss.item() > 0
    # degenerate: student puts ~all mass on one support token; the residual
    # bucket's log(1 - s_mass) must not produce nan/inf
    hard = torch.full_like(t_logits, -1e4)
    hard[torch.arange(len(ti)), ti[:, 0]] = 0.0
    loss = sparse_kl(torch.log_softmax(hard, -1), tp, ti, tr)
    assert torch.isfinite(loss)


def test_prune_vocab_logit_equivalence():
    """Pruned model logits over kept ids must equal the original model's
    logits at those ids, for remapped inputs."""
    from excise.export import prune_vocab
    torch.manual_seed(0)
    m = tiny_model("qwen2")
    m.eval()
    ids = torch.tensor([[5, 9, 17, 33, 2]])
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    keep = sorted({0, 1, 2, 5, 9, 17, 33, 40, 41, 99})
    old2new = prune_vocab(m, keep)
    assert m.get_input_embeddings().weight.shape[0] == len(keep)
    assert m.config.vocab_size == len(keep)
    with torch.no_grad():
        out = m(input_ids=old2new[ids]).logits
    keep_t = torch.tensor(keep)
    assert torch.allclose(ref[..., keep_t], out, atol=1e-5)


def test_load_sliced_roundtrip(tmp_path):
    from excise.export import load_sliced, slice_model
    torch.manual_seed(0)
    m = tiny_model("qwen2")
    m.eval()
    mask = torch.zeros(2, 64)
    mask[0, :40] = 1.0
    mask[1, 10:30] = 1.0
    slice_model(m, mask)
    ids = torch.tensor([[5, 9, 17, 33]])
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    m.save_pretrained(tmp_path / "sliced")
    m2 = load_sliced(tmp_path / "sliced")
    m2.eval()
    with torch.no_grad():
        out = m2(input_ids=ids).logits
    assert torch.allclose(ref, out, atol=1e-5)


def test_slice_model_rejects_empty_layer():
    from excise.export import slice_model
    m = tiny_model("qwen2")
    mask = torch.zeros(2, 64)
    mask[0, :8] = 1.0                      # layer 1 keeps nothing
    with pytest.raises(ValueError, match="zero channels"):
        slice_model(m, mask)
