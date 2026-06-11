#!/usr/bin/env python3
"""Generate all paper figures from the experiment artifacts in vast_test/.
Styled for print: serif/STIX typography, restrained palette, subtle grids."""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent / "vast_test"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})
RED, GRAY, GOLD, LGRAY = "#a63232", "#6e6e6e", "#b8932f", "#c9c9c9"
INK = "#222222"


def load(name):
    return json.loads((ROOT / name).read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


# ------------------------------------------------------------- method
def fig_method():
    fig, ax = plt.subplots(figsize=(6.8, 2.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, lines, accent=False, dashed=False):
        fc = "#fdf5f4" if accent else "#f7f7f7"
        ec = RED if accent else GRAY
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.45,rounding_size=1.1",
            facecolor=fc, edgecolor=ec, linewidth=1.0,
            linestyle="--" if dashed else "-"))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="center",
                fontsize=8.2, weight="bold",
                color=RED if accent else INK)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 7.6 - 4.2 * i, ln, ha="center",
                    va="center", fontsize=7.0, color=INK)

    def arrow(x1, y1, x2, y2, label=None, dy=2.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.9, color=GRAY,
                                     shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2, max(y1, y2) + dy, label, ha="center",
                    fontsize=6.6, color=GRAY, style="italic")

    box(1, 9, 17, 17, "frozen base $M$",
        ["greedy outputs $\\hat{y}$", "top-$k$ dists,", "cached once"])
    box(29, 9, 21, 17, "joint student", ["LoRA $\\theta$ (rank 32)",
        "hard-concrete gates $z$", "binned KL + anchored guardrail"],
        accent=True)
    box(60, 9, 22, 17, "adaptive controller",
        ["target $\\downarrow$ while KL ok",
         "probes vs unmasked baseline",
         "2 strikes $\\Rightarrow$ floor + rollback"])
    box(88, 16.5, 11.5, 9.5, "frontier", ["+ receipts"])
    box(88, 3.5, 11.5, 9.5, "sliced", ["$\\equiv$ masked"])

    arrow(19, 17.5, 28, 17.5, "prompts only", 11.0)
    arrow(51, 17.5, 59, 17.5)
    arrow(83, 19.5, 87, 21)
    arrow(83, 15.5, 87, 8.5)
    ax.text(39.5, 4.0, "one training run", ha="center", fontsize=7.2,
            color=RED, style="italic")
    save(fig, "method")


# ------------------------------------------------------------ frontier
def fig_frontier():
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    prism = [(5.75, 29.0, "raw mask", (7, -2)),
             (90.61, 99.53, "raw, broad", (-58, -10)),
             (5.05, 91.33, "staged collimation", (10, -3)),
             (4.65, 90.6, "refined MVC", (-14, -16))]
    ax.scatter([p[0] for p in prism], [p[1] for p in prism], s=46, marker="s",
               facecolor="white", edgecolor=GRAY, linewidth=1.2,
               label="PRISM (staged pipeline)", zorder=3)
    for x, y, t, off in prism:
        ax.annotate(t, (x, y), textcoords="offset points", xytext=off,
                    fontsize=7, color=GRAY)
    v02 = []
    for s in (42, 43, 44):
        d = json.loads((ROOT / "library_runs" / "v02" /
                        f"arith_v02_long_s{s}" / "summary.json").read_text())
        v02.append((d["floor"] * 100, d["frontier"][-1][1] * 100))
    deep = json.loads((ROOT / "library_runs" / "v02" / "arith_v02_deep_s42" /
                       "summary.json").read_text())
    v02.append((deep["floor"] * 100, deep["frontier"][-1][1] * 100))
    ax.scatter([p[0] for p in v02], [p[1] for p in v02], s=56, marker="D",
               color=RED, edgecolor="white", linewidth=0.8, zorder=5,
               label="excise (automatic floors)")
    ax.annotate("found automatically, one\nlabel-free run: 0.7–1.2%\n"
                "at 91% — still descending", xy=(0.73, 90.3),
                xytext=(1.05, 52), fontsize=7.2, color=RED, ha="left",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkB=5))
    ax.set_xscale("log")
    ax.set_xticks([0.7, 1.2, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["0.7", "1.2", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("MLP channels kept (%)")
    ax.set_ylabel("fidelity (%)")
    ax.set_xlim(0.6, 120)
    ax.set_ylim(0, 112)
    ax.axhline(100, color=LGRAY, lw=0.7, ls=":", zorder=1)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "frontier")


# ------------------------------------------------------------- fc
def fig_fc():
    log = (ROOT / "logs" / "fc.log").read_text()
    probes = [(float(m.group(1)) * 100, float(m.group(2)) * 100)
              for m in re.finditer(
                  r"\[probe\] step=\d+ open=([\d.]+) self_match=([\d.]+)",
                  log)]
    r = load("results_fc4b.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    xs, ys = zip(*sorted(probes))
    a1.plot(xs, ys, "-", color=RED, lw=1.3, zorder=3)
    a1.scatter(xs, ys, s=22, color=RED, edgecolor="white", linewidth=0.6,
               zorder=4)
    a1.axhline(100, color=LGRAY, lw=0.7, ls=":")
    a1.set_xlabel("MLP channels open (%)")
    a1.set_ylabel("exact self-match (%)")
    a1.set_title("(a) probe trace during descent (train prompts)",
                 fontsize=8.2)
    a1.invert_xaxis()
    a1.set_ylim(0, 108)

    budgets = sorted((float(k) * 100, v * 100) for k, v in r["evals"].items())
    bars = a2.bar([f"{b:.0f}" for b, _ in budgets], [v for _, v in budgets],
                  color=RED, width=0.6, zorder=3)
    for b, (_, v) in zip(bars, budgets):
        a2.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v + 2),
                    ha="center", fontsize=7, color=INK)
    a2.axhline(r["unmasked_self_match"] * 100, color=GRAY, lw=1.0, ls="--",
               label=f"unmasked ceiling "
                     f"({r['unmasked_self_match']*100:.0f}%)", zorder=4)
    a2.set_xlabel("MLP channels kept (%)")
    a2.set_title("(b) held-out evaluation", fontsize=8.2)
    a2.set_ylim(0, 108)
    a2.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.2)
    save(fig, "function_calling")


# ------------------------------------------------- miscalibration
def fig_miscalibration():
    bad_log = (ROOT / "logs" / "v20_failed" / "battery.log").read_text()
    start = bad_log.find("RUN: --name s42")
    end = bad_log.find("RUN: --name s43")
    bad_log = bad_log[start: end if end > start else len(bad_log)]
    bad = json.loads(
        (ROOT / "logs" / "v20_failed" / "results_s42.json").read_text())
    kls = [(float(m.group(1)), float(m.group(2)))
           for m in re.finditer(
               r"\[train\] step=\d+ kl=[\d.]+ ema=([\d.]+) open=([\d.]+)",
               bad_log)]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    opens = [o * 100 for _, o in kls]
    emas = [k for k, _ in kls]
    ax.plot(opens, emas, "-", color=GOLD, lw=1.4,
            label="EMA distillation KL (left)")
    ax.axhline(0.02, color=GRAY, lw=0.8, ls="--", label="KL budget (left)")
    ax.invert_xaxis()
    ax.set_xlabel("MLP channels open (%)")
    ax.set_ylabel("EMA KL")
    ax.set_ylim(0, 0.08)
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines.right.set_visible(True)
    ax2.spines.right.set_linewidth(0.7)
    evals = sorted((float(k) * 100, v["recovery"] * 100)
                   for k, v in bad["evals"].items())
    ax2.scatter([e[0] for e in evals], [e[1] for e in evals], s=64,
                marker="X", color=RED, edgecolor="white", linewidth=0.6,
                label="true recovery (right)", zorder=5)
    ax2.set_ylabel("recovery (%)", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.set_ylim(0, 112)
    ax2.annotate("KL under budget,\nrecovery collapsed to 62–66%",
                 xy=(5.0, 65), xytext=(72, 48), fontsize=7.2, color=RED,
                 ha="left",
                 arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                 shrinkB=6))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left",
              fontsize=7.2)
    save(fig, "miscalibration")


# ------------------------------------------------------- ablations
def fig_ablations():
    runs = [("seed 42", "s42"), ("seed 43", "s43"), ("seed 44", "s44"),
            ("MLP-only\nadapter", "mlponly"), ("label-free\n($\\beta_{ce}{=}0$)", "ce0")]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    labels, recs, floors = [], [], []
    for label, name in runs:
        r = load(f"results_{name}.json")
        labels.append(label)
        recs.append(r["evals"]["0.0500"]["recovery"] * 100)
        floors.append(r["floor_frac"] * 100)
    bars = ax.bar(labels, recs, color=[RED] * 3 + ["#c47a7a"] * 2, width=0.62,
                  zorder=3)
    for b, f, rec in zip(bars, floors, recs):
        ax.annotate(f"{rec:.1f}", (b.get_x() + b.get_width() / 2, rec + 1.5),
                    ha="center", fontsize=7.2, color=INK)
        ax.annotate(f"floor {f:.1f}%",
                    (b.get_x() + b.get_width() / 2, 8), ha="center",
                    fontsize=6.8, color="white", weight="bold")
    ax.axhline(91.33, color=GRAY, lw=1.0, ls="--")
    ax.annotate("PRISM staged pipeline (91.3%)", xy=(4.42, 91.3),
                xytext=(4.42, 110), fontsize=7.2, color=GRAY, ha="right",
                va="top")
    ax.axhline(100, color=LGRAY, lw=0.7, ls=":")
    ax.set_ylabel("recovery @ 5% channels (%)")
    ax.set_ylim(0, 118)
    save(fig, "ablations")


# ------------------------------------------------------- v0.2 (json)
def fig_v02():
    r = json.loads((ROOT / "library_runs" / "v02" / "json_v02" /
                    "receipts.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    # (a) guardrail KL: train batches read near-zero while anchor batches
    # expose the drift the train-only guardrail cannot see
    tr = [(g["step"], g["kl"]) for g in r["guardrail_trace"]
          if g["src"] == "train"]
    an = [(g["step"], g["kl"]) for g in r["guardrail_trace"]
          if g["src"] == "anchor"]
    a1.plot(*zip(*tr), "-", color=GRAY, lw=1.0,
            label="train batches (what v0.1 saw)")
    a1.plot(*zip(*an), "-", color=RED, lw=1.2,
            label="off-task anchors (where drift lives)")
    a1.set_yscale("log")
    a1.set_xlabel("step")
    a1.set_ylabel("guardrail KL to base")
    a1.set_title("(a) the same adapter, two guardrail views", fontsize=8.2)
    a1.legend(frameon=False, loc="upper right", fontsize=6.8)

    # (b) probe trace: masked vs unmasked-at-same-step, with the rollback
    opens = [p["open"] * 100 for p in r["probe_trace"]]
    masked = [p["self_match"] * 100 for p in r["probe_trace"]]
    unm = [p["unmasked"] * 100 for p in r["probe_trace"]]
    a2.plot(opens, unm, "--", color=GRAY, lw=1.1, marker="o", ms=3.5,
            label="unmasked, same step (baseline)")
    a2.plot(opens, masked, "-", color=RED, lw=1.3, marker="o", ms=3.5,
            label="masked probe")
    floor = r["floor"] * 100
    a2.axvline(floor, color=GOLD, lw=1.1, ls=":")
    a2.annotate(f"floor {floor:.1f}%\n(rolled back to\nlast passing probe)",
                xy=(floor, 38), xytext=(floor + 9, 22), fontsize=6.8,
                color=INK,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.8))
    a2.invert_xaxis()
    a2.set_xlabel("MLP channels open (%)")
    a2.set_ylabel("exact self-match (%)")
    a2.set_ylim(0, 108)
    a2.set_title("(b) floor detection against a measured baseline",
                 fontsize=8.2)
    a2.legend(frameon=False, loc="lower left", fontsize=6.8)
    fig.tight_layout(w_pad=2.2)
    save(fig, "v02_calibration")


# ----------------------------------------------------- data diversity
def fig_diversity():
    t = json.loads((ROOT / "library_runs" / "v02" / "json_v021_polished" /
                    "summary.json").read_text())
    d = json.loads((ROOT / "library_runs" / "v02" / "json_diverse" /
                    "summary.json").read_text())

    def row(s):
        floor = s["floor"] * 100
        fid = dict((round(b, 4), a) for b, a in s["frontier"])[
            round(s["floor"], 4)] * 100
        drift = (s["base_self_match"] - s["unmasked_self_match"]) * 100
        return floor, fid, drift

    panels = [("channel floor (%)", 0, "log"),
              ("fidelity at floor (%)", 1, None),
              ("unmasked drift (pts)", 2, None)]
    vals = {"600 prompts,\none template": row(t),
            "3,000 prompts,\nvaried": row(d)}
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.9))
    for ax, (title, i, scale) in zip(axes, panels):
        labels = list(vals)
        ys = [vals[l][i] for l in labels]
        bars = ax.bar(labels, ys, color=[GRAY, RED], width=0.55, zorder=3)
        for b, v in zip(bars, ys):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2,
                                     v * (1.06 if scale else 1) + (0 if scale else 1)),
                        ha="center", fontsize=7.6, color=INK)
        if scale:
            ax.set_yscale("log")
            ax.set_ylim(1, 60)
        else:
            ax.set_ylim(0, max(ys) * 1.2)
        ax.set_title(title, fontsize=8.2)
        ax.tick_params(axis="x", labelsize=7.2)
    fig.tight_layout(w_pad=2.0)
    save(fig, "diversity")


# ------------------------------------------------- parameter composition
def fig_params():
    # Qwen2.5-Math-1.5B architecture: 28 layers, hidden 1536, d_ff 8960,
    # vocab 151,936 (tied embeddings). Component math from the config;
    # totals match the measured param counts in the run receipts.
    deep = json.loads((ROOT / "library_runs" / "v02" / "arith_v02_deep_s42" /
                       "summary.json").read_text())
    base_total = 1543.7
    mlp0 = 3 * 1536 * 8960 * 28 / 1e6
    emb0 = 151936 * 1536 / 1e6
    rest = base_total - mlp0 - emb0
    after_total = deep["params_after_slice_prune"] / 1e6
    mlp1 = mlp0 * deep["floor"]
    emb1 = 1746 * 1536 / 1e6
    rest1 = after_total - mlp1 - emb1

    fig, ax = plt.subplots(figsize=(6.0, 1.75))
    rows = [("base 1.54B", [mlp0, emb0, rest]),
            (f"after excise {after_total/1000:.2f}B",
             [mlp1, emb1, rest1])]
    colors = [RED, GOLD, GRAY]
    names = ["MLP channels", "vocabulary (embed + head)",
             "attention + norms"]
    for yi, (label, parts) in enumerate(rows[::-1]):
        x = 0
        for p, c in zip(parts, colors):
            ax.barh(yi, p, left=x, height=0.55, color=c, zorder=3,
                    edgecolor="white", linewidth=0.5)
            x += p
        ax.text(-18, yi, label, ha="right", va="center", fontsize=8.4,
                color=INK)
    ax.annotate("extraction removes 99.3% of MLP channels and 98.9% of\n"
                "the vocabulary; attention (untouched) is 94% of what remains",
                xy=(after_total + 8, 0), xytext=(360, -0.05), fontsize=7.4,
                color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7,
                                shrinkB=4))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, names, frameon=False, fontsize=7.2, loc="lower right",
              ncol=3, bbox_to_anchor=(1.0, -0.42))
    ax.set_xlim(0, 1600)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("parameters (millions)", fontsize=8)
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", visible=False)
    save(fig, "params")


if __name__ == "__main__":
    fig_method()
    fig_frontier()
    fig_fc()
    fig_miscalibration()
    fig_ablations()
    fig_v02()
    fig_diversity()
    fig_params()
    print("figures ->", OUT)
