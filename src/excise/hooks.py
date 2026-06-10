"""Gate application via forward pre-hooks on each down-projection.

Hooks survive `generate`, PEFT wrapping, and weight surgery; nothing about
the model's own code is replaced.
"""

import torch

from .arch import MLPMap


class GateHooks:
    """Multiplies the down-projection input by a per-channel gate.

    mode:
      "off"    — no-op (base model behavior)
      "sample" — multiply by `self.sampled[li]` (training; pre-sampled)
      "mask"   — multiply by `self.mask[li]` (binary, zero-isolation)
    """

    def __init__(self, mlp_map: MLPMap):
        self.mode = "off"
        self.mask = None          # [n_layers, d_ff] tensor
        self.sampled = None       # list of per-layer tensors
        self._handles = []
        for li, dp in enumerate(mlp_map.down_projs):
            self._handles.append(
                dp.register_forward_pre_hook(self._make_hook(li)))

    def _make_hook(self, li: int):
        def hook(module, args):
            if self.mode == "off":
                return None
            x = args[0]
            if self.mode == "sample":
                g = self.sampled[li]
            else:
                g = self.mask[li]
            return (x * g.to(x.dtype),) + args[1:]
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
