"""Physical slicing: delete masked channels from the weights.

For gated MLPs this is exactly equivalent to zero-isolation masking — a
zeroed channel contributes nothing to the down-projection, a deleted one
contributes nothing by absence. Verified empirically: sliced accuracy
matched masked-scaffold accuracy to within one eval example.

Beyond MLP channels, `prune_vocab` slices the embedding/lm_head rows down
to the capability's token support. For a narrow capability the vocabulary
is the dominant remaining cost (55% of the parameters of the sliced 1.5B
arithmetic model), so this is where most of the remaining size lives.
"""

import json
from pathlib import Path

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
    if (mask.sum(-1) == 0).any():
        empty = (mask.sum(-1) == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"mask keeps zero channels in layers {empty}; a Linear cannot "
            "have zero features. Keep at least one channel per layer "
            "(top-k masks are global, so very low budgets can empty a "
            "layer) or drop the layer explicitly.")
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
    widths = [mlp_map.down_projs[li].in_features
              for li in range(mlp_map.n_layers)]
    setattr(model.config, "excise_layer_widths", widths)
    return model


def prune_vocab(model: torch.nn.Module, keep_ids) -> torch.Tensor:
    """Slice the embedding (and lm_head, if untied) rows down to `keep_ids`,
    in place. Returns an old->new id map (LongTensor of size old_vocab, -1
    for dropped ids).

    The pruned model speaks a remapped id space: encode with the original
    tokenizer then map through the returned table (or
    `config.excise_vocab_keep`, which inverts it). Keep at minimum every id
    that appears in the capability's prompts and teacher outputs, plus
    eos/pad — `ExtractionResult.vocab_support()` collects exactly that, plus
    every id in the teacher's top-k cache.
    """
    keep = torch.as_tensor(sorted(set(int(i) for i in keep_ids)),
                           dtype=torch.long)
    embed = model.get_input_embeddings()
    old_vocab = embed.weight.shape[0]
    if keep.max() >= old_vocab:
        raise ValueError("keep_ids contains ids outside the vocabulary")
    head = model.get_output_embeddings()
    tied = head is not None and head.weight is embed.weight
    with torch.no_grad():
        new_w = embed.weight.data[keep.to(embed.weight.device)].clone()
        embed.weight = torch.nn.Parameter(new_w)
        embed.num_embeddings = len(keep)
        if head is not None:
            if tied:
                head.weight = embed.weight
            else:
                head.weight = torch.nn.Parameter(
                    head.weight.data[keep.to(head.weight.device)].clone())
            head.out_features = len(keep)
    old2new = torch.full((old_vocab,), -1, dtype=torch.long)
    old2new[keep] = torch.arange(len(keep))

    def _remap(v):
        return int(old2new[v]) if v is not None and old2new[v] >= 0 else None

    cfg = model.config
    cfg.vocab_size = len(keep)
    setattr(cfg, "excise_vocab_keep", keep.tolist())
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if getattr(cfg, attr, None) is not None:
            setattr(cfg, attr, _remap(getattr(cfg, attr)))
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
            v = getattr(gen, attr, None)
            if isinstance(v, list):
                setattr(gen, attr, [m for m in (_remap(x) for x in v)
                                    if m is not None])
            elif v is not None:
                setattr(gen, attr, _remap(v))
    return old2new


def load_sliced(path, device=None, dtype=None):
    """Round-trip loader for a sliced artifact saved with save_pretrained.

    transformers cannot construct heterogeneous per-layer MLP widths from a
    config, so: build a skeleton with intermediate_size=1, resize each
    layer's Linears to the widths recorded in `excise_layer_widths`, then
    load the saved weights. Never use ignore_mismatched_sizes=True instead —
    it silently REINITIALIZES every mismatched weight and produces a model
    that loads cleanly but is garbage.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    path = Path(path)
    config = AutoConfig.from_pretrained(path)
    widths = getattr(config, "excise_layer_widths", None)
    if widths is None:
        raise ValueError(f"{path} has no excise_layer_widths; not a sliced "
                         "excise artifact")
    config.intermediate_size = 1
    model = AutoModelForCausalLM.from_config(config)
    mlp_map = find_mlps(model)
    for li, parent in enumerate(mlp_map.mlp_parents):
        for name, child in list(parent.named_children()):
            if not isinstance(child, torch.nn.Linear):
                continue
            if child.out_features == 1:                  # gate / up
                new = torch.nn.Linear(child.in_features, widths[li],
                                      bias=child.bias is not None)
            elif child.in_features == 1:                 # down
                new = torch.nn.Linear(widths[li], child.out_features,
                                      bias=child.bias is not None)
            else:
                continue
            setattr(parent, name, new)
    config.intermediate_size = max(widths)

    state = {}
    index = path / "model.safetensors.index.json"
    from safetensors.torch import load_file
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        for shard in shards:
            state.update(load_file(path / shard))
    else:
        state = load_file(path / "model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False,
                                                assign=True)
    tied = getattr(config, "tie_word_embeddings", False)
    missing = [k for k in missing if not (tied and k.endswith("lm_head.weight"))]
    if missing or unexpected:
        raise RuntimeError(f"sliced checkpoint mismatch: missing={missing} "
                           f"unexpected={unexpected}")
    if tied:
        model.tie_weights()
    if dtype is not None:
        model = model.to(dtype)
    if device is not None:
        model = model.to(device)
    return model


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
