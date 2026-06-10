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

    box(1, 9, 19, 17, "frozen base $M$",
        ["greedy outputs $\\hat{y}$", "top-$k$ dists, cached once"])
    box(27, 9, 23, 17, "joint student", ["LoRA $\\theta$ (rank 32)",
        "hard-concrete gates $z$", "KL leash + guardrail"], accent=True)
    box(57, 9, 21, 17, "adaptive controller",
        ["target $\\downarrow$ while KL $\\leq$ budget",
         "probes generate & match;", "2 strikes $\\Rightarrow$ floor"])
    box(84, 16.5, 15, 9.5, "frontier", ["+ receipts"])
    box(84, 3.5, 15, 9.5, "sliced model", ["$\\equiv$ masked"])

    arrow(20, 17.5, 27, 17.5, "prompts only", 3.2)
    arrow(50, 17.5, 57, 17.5)
    arrow(78, 19.5, 84, 21)
    arrow(78, 15.5, 84, 8.5)
    ax.text(38.5, 4.2, "one training run", ha="center", fontsize=7.2,
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
    xs, ys = [], []
    for s in ("s42", "s43", "s44"):
        r = load(f"results_{s}.json")
        for k, v in r["evals"].items():
            xs.append(float(k) * 100)
            ys.append(v["recovery"] * 100)
    ax.scatter(xs, ys, s=46, marker="o", color=RED, edgecolor="white",
               linewidth=0.7, zorder=4,
               label="excise (one joint run; 3 seeds)")
    ax.annotate("automatic floors:\n2.9–4.1%", xy=(2.95, 90),
                xytext=(2.15, 66), fontsize=7.2, color=RED, ha="left",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkB=5))
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("MLP channels kept (%)")
    ax.set_ylabel("recovery (% of base accuracy)")
    ax.set_xlim(2, 120)
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


if __name__ == "__main__":
    fig_method()
    fig_frontier()
    fig_fc()
    fig_miscalibration()
    fig_ablations()
    print("figures ->", OUT)
