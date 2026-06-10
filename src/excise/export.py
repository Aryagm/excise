"""Physical slicing: delete masked channels from the weights.

For gated MLPs this is exactly equivalent to zero-isolation masking — a
zeroed channel contributes nothing to the down-projection, a deleted one
contributes nothing by absence. Verified empirically: sliced accuracy
matched masked-scaffold accuracy to within one eval example.
"""

import torch

from .arch import find_mlps


def slice_model(model: torch.nn.Module, mask: torch.Tensor) -> torch.nn.Module:
    """Delete channels where mask[layer, channel] == 0, in place.

    Slices every Linear in each MLP whose output dim equals d_ff (gate/up
    projections) and the down-projection's input dim.
    """
    mlp_map = find_mlps(model)
    d_ff = mlp_map.d_ff
    if mask.shape != (mlp_map.n_layers, d_ff):
        raise ValueError(f"mask shape {tuple(mask.shape)} != "
                         f"{(mlp_map.n_layers, d_ff)}")
    with torch.no_grad():
        for li in range(mlp_map.n_layers):
            keep = mask[li].nonzero(as_tuple=True)[0].to(
                mlp_map.down_projs[li].weight.device)
            parent = mlp_map.mlp_parents[li]
            for child in parent.children():
                if not isinstance(child, torch.nn.Linear):
                    continue
                if child.out_features == d_ff:          # gate / up
                    child.weight = torch.nn.Parameter(
                        child.weight.data[keep].clone())
                    if child.bias is not None:
                        child.bias = torch.nn.Parameter(
                            child.bias.data[keep].clone())
                    child.out_features = len(keep)
                elif child.in_features == d_ff:         # down
                    child.weight = torch.nn.Parameter(
                        child.weight.data[:, keep].clone())
                    child.in_features = len(keep)
    # Note: per-layer widths are now heterogeneous. The model works
    # in-memory and via torch.save(state_dict); round-tripping through
    # save_pretrained/from_pretrained needs the saved mask (load the base
    # model, re-slice with the mask, then load the state dict).
    widths = [mlp_map.down_projs[li].in_features
              for li in range(mlp_map.n_layers)]
    setattr(model.config, "excise_layer_widths", widths)
    return model


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
