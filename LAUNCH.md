# Launch checklist

Everything below is prepared; items marked ☐ need Arya's action (accounts,
publishing — things only you can/should do).

## Repo

- ☐ Create the GitHub repo (paper + README reference `github.com/Aryagm/excise`;
  edit both if the org/name differs) and `git push`.
- ☐ Confirm the package name `excise` (PyPI name is free as of 2026-06-09).
  Renaming = `pyproject.toml`, `src/excise/`, README, paper, this file.
- ☑ MIT license, CI workflow (CPU tests), .gitignore.

- ☐ After flipping the repo public, restore the CI badge at the top of
  README.md (badges on private repos don't render):
  `<a href="https://github.com/Aryagm/excise/actions"><img src="https://github.com/Aryagm/excise/actions/workflows/test.yml/badge.svg" alt="tests"></a>`

## Library

- ☑ Core, tests (5/5), CLI, examples (arithmetic, function calling, SmolLM2,
  JSON extraction).
- ☑ GPU-validated end-to-end (dogfood + breadth runs; see receipts in
  vast_test/).
- ☐ `pip install build twine && python -m build && twine upload dist/*`
  (or set up PyPI trusted publishing from the repo).

## Paper

- ☑ Compiles (`tectonic excise.tex`), 4 figures regenerate from committed
  result JSONs (`python paper/make_figures.py`).
- ☑ Bib entries cross-checked against the PRISM paper's bibliography and
  known venues; final eyeball before submission recommended.
- ☐ Pick the workshop (NeurIPS/ICLR efficiency or interpretability tracks)
  and reformat margins/style file as required.
- ☐ Decide author list / affiliation.
- Suggested courtesy: email the PRISM authors (Mishra & Pagare) with the
  preprint before public launch — we answer their stated open question on
  their benchmark.

## Announcement thread (draft)

1/ We built `excise`: point it at an open LLM + a pile of prompts, get back
a *smaller model* that does that one thing. No labels, no pipeline, one
command. pip install excise 🧵

2/ A 1.5B math model does 2-digit addition through just 2.9% of its MLP
channels. excise finds that floor automatically — and deletes the rest:
1.54B → 0.42B params, 97% of the skill intact. [frontier figure]

3/ It's label-free. The model teaches itself: the target is its own output
distribution. We extracted function calling from Qwen3-4B using only BFCL
prompts — zero gold answers. [fc figure]

4/ The surprising lesson: loss curves LIE at high sparsity. Our first
controller watched the distillation KL — it stayed "healthy" while real
accuracy collapsed 35 points. The fix: probes that actually *generate*.
[miscalibration figure]

5/ Every extraction ships with receipts: random-mask control, drift check,
probe trace. If your extraction is invalid, the report tells you — not
your users.

6/ Honest caveat: memory shrinks unconditionally (3.7× here); speedup
depends on workload — we measured 1.11× on short-batch generation because
attention + lm_head dominate there. Long decode on bigger models is where
slicing pays in latency.

7/ Built on the capability-extraction framing of PRISM (Mishra & Pagare) —
their contract, our method: what took a 4-stage pipeline is now one run
that beats it on its own benchmark. Paper + code: [links]

8/ Total compute for every result in the paper: under $5 on one rented
RTX 4090. Capability extraction is now a commodity operation. What would
you excise?

## Costs to date

GPU total ≈ $6 (incl. failures and breadth runs); instance 40296201 still
running at $0.40/hr — destroy after breadth runs are confirmed + artifacts
pulled (needs Arya's approval per standing instruction).
