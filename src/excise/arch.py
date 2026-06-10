"""Architecture registry: locate the gated-MLP down-projections.

The unit of extraction is the MLP intermediate channel — the input dimension
of each block's down-projection. Anything shaped like
``down_proj(act(gate(x)) * up(x))`` works: Qwen2/3, Llama, Mistral, Gemma,
and most decoder-only models in transformers.
"""

import re

import torch

def _is_linear_like(module) -> bool:
    """nn.Linear, or a PEFT LoRA wrapper around one (hooking the wrapper
    gates both the base path and the adapter path)."""
    return isinstance(module, torch.nn.Linear) or (
        hasattr(module, "base_layer")
        and isinstance(module.base_layer, torch.nn.Linear))


_DOWN_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\.mlp\.down_proj$"),
    re.compile(r"\.layers\.(\d+)\.feed_forward\.w2$"),
    re.compile(r"\.h\.(\d+)\.mlp\.c_proj$"),
]


class MLPMap:
    """Per-layer handles to the MLP down-projection and its siblings."""

    def __init__(self, down_projs: list, mlp_parents: list):
        self.down_projs = down_projs        # nn.Linear per layer
        self.mlp_parents = mlp_parents      # owning MLP module per layer

    @property
    def n_layers(self) -> int:
        return len(self.down_projs)

    @property
    def d_ff(self) -> int:
        return self.down_projs[0].in_features

    @property
    def n_channels(self) -> int:
        return sum(dp.in_features for dp in self.down_projs)


def find_mlps(model: torch.nn.Module) -> MLPMap:
    """Locate every decoder layer's down-projection, in layer order."""
    found = {}
    parents = {}
    by_name = dict(model.named_modules())
    for name, module in by_name.items():
        for pat in _DOWN_PATTERNS:
            m = pat.search(name)
            if m and _is_linear_like(module):
                li = int(m.group(1))
                found[li] = module
                parents[li] = by_name[name.rsplit(".", 1)[0]]
    if not found:
        raise ValueError(
            "Could not locate MLP down-projections. excise supports "
            "decoder-only models with '<...>.layers.<i>.mlp.down_proj' "
            "(Qwen, Llama, Mistral, Gemma) or '.feed_forward.w2' layouts. "
            f"Model type: {type(model).__name__}")
    order = sorted(found)
    if order != list(range(len(order))):
        raise ValueError(f"Non-contiguous layer indices found: {order}")
    d_ffs = {found[i].in_features for i in order}
    if len(d_ffs) != 1:
        raise ValueError(f"Per-layer d_ff differs ({d_ffs}); already sliced?")
    return MLPMap([found[i] for i in order], [parents[i] for i in order])
