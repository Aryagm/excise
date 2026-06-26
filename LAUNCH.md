# Launch checklist

Everything below is prepared; items marked ☐ need Arya's action (accounts,
publishing — things only you can/should do).

## Repo

- ☐ Create the GitHub repo (paper + README reference `github.com/Aryagm/excise`;
  edit both if the org/name differs) and `git push`.
- ☐ PyPI release is deferred until account access is restored. For launch,
  README and release notes use pip's direct GitHub install path.
- ☑ MIT license, CI workflow (CPU tests), .gitignore.

- ☐ After flipping the repo public, restore the CI badge at the top of
  README.md (badges on private repos don't render):
  `<a href="https://github.com/Aryagm/excise/actions"><img src="https://github.com/Aryagm/excise/actions/workflows/test.yml/badge.svg" alt="tests"></a>`

## Library

- ☑ Core, tests (5/5), CLI, examples (arithmetic, function calling, SmolLM2,
  JSON extraction).
- ☑ GPU-validated end-to-end (dogfood + breadth runs; see receipts in
  vast_test/).
- ☐ When PyPI access is restored:
  `pip install build twine && python -m build && twine upload dist/*`
  (or set up trusted publishing from the repo).

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

## Release-day command block (run in order)

```bash
# 1. Flip public
gh repo edit Aryagm/excise --visibility public

# 2. Restore CI badge in README (markdown above), commit, push.

# 3. Tag + GitHub release
git tag v0.2.0 && git push origin v0.2.0
gh release create v0.2.0 --title "excise v0.2.0" \
    --notes-file .github/release-notes-v0.2.0.md

# 4. Post the thread (below), then Show HN + r/MachineLearning [P]
```

PRISM-authors email draft: `release_private/prism_authors_email.md`
(gitignored — never committed). Send a few days BEFORE release day.

## Announcement thread (draft)

1/ We built `excise`: point it at an open LLM + a pile of prompts, get back
a *smaller model* that does that one thing. No labels, no pipeline, one
command. Install from GitHub with pip. 🧵

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
