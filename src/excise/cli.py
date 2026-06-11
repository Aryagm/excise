"""CLI: excise extract --model NAME --prompts FILE --out DIR"""

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="excise",
        description="Extract one capability from an LLM into a smaller "
                    "model. Label-free.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--model", required=True)
    ex.add_argument("--prompts", required=True,
                    help=".txt (one per line) or .jsonl with {'prompt': ...}")
    ex.add_argument("--out", required=True)
    ex.add_argument("--slice", action="store_true",
                    help="also export the physically sliced model "
                         "(reload it with excise.load_sliced)")
    ex.add_argument("--prune-vocab", action="store_true",
                    help="with --slice: additionally prune the embedding/"
                         "lm_head to the capability's token support")
    ex.add_argument("--max-steps", type=int, default=None)
    ex.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from .config import ExtractConfig
    from .extractor import extract

    cfg = ExtractConfig(seed=args.seed)
    if args.max_steps:
        cfg.max_steps = args.max_steps
    result = extract(args.model, args.prompts, config=cfg)
    out = result.save(args.out)
    print(result.report())
    if args.slice:
        from .export import param_count
        before = param_count(result._model)
        sliced = result.export_sliced(prune_vocabulary=args.prune_vocab)
        after = param_count(sliced)
        result.save(args.out)          # now also writes out/sliced
        print(result.report())
        print(f"sliced: {before/1e9:.2f}B -> {after/1e9:.2f}B params "
              f"({before/after:.2f}x), saved to {out / 'sliced'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
