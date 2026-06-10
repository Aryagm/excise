#!/usr/bin/env python3
"""README charts: bolder and simpler than the paper figures."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
})
RED, DARK, GRAY, LIGHT = "#c0392b", "#2c3e50", "#95a5a6", "#ecf0f1"


def hero():
    rows = [  # (label, full B, sliced B, fidelity text, projected)
        ("Arithmetic\nQwen2.5-Math-1.5B", 1.54, 0.42, "97% of the skill kept", False),
        ("Arithmetic (few-shot)\nSmolLM2-1.7B", 1.75, 0.59, "97% output fidelity", False),
        ("JSON extraction\nQwen2.5-1.5B-Instruct", 1.58, 0.78, "90% output fidelity", False),
        ("Function calling\nQwen3-4B", 4.02, 2.40, "~76% output fidelity", True),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    y = list(range(len(rows)))[::-1]
    for yi, (label, full, sliced, fid, proj) in zip(y, rows):
        ax.barh(yi + 0.18, full, height=0.32, color=LIGHT, edgecolor=GRAY,
                linewidth=0.8, zorder=2)
        ax.barh(yi - 0.18, sliced, height=0.32,
                color=RED if not proj else "#d98477", zorder=3)
        ax.text(full + 0.06, yi + 0.18, f"{full:.1f}B", va="center",
                fontsize=11, color=GRAY, weight="bold")
        ax.text(sliced + 0.06, yi - 0.18,
                f"{sliced:.1f}B  ·  {fid}" + ("  (projected)" if proj else ""),
                va="center", fontsize=11, color=DARK, weight="bold")
        ax.text(-0.08, yi, label, va="center", ha="right", fontsize=10.5,
                color=DARK, linespacing=1.3)
    ax.text(0.02, max(y) + 0.78, "full model", fontsize=10, color=GRAY,
            weight="bold")
    ax.text(0.02, max(y) + 0.44, "after excise", fontsize=10, color=RED,
            weight="bold")
    ax.set_xlim(0, 5.4)
    ax.set_ylim(-0.65, max(y) + 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("one capability, a fraction of the model",
                 fontsize=15, color=DARK, weight="bold", pad=14, loc="left")
    fig.savefig(OUT / "hero.png")
    plt.close(fig)


def frontier():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.spines.left.set_visible(True)
    prism = [(5.75, 29.0), (90.61, 99.53), (5.05, 91.33), (4.65, 90.6)]
    ax.scatter([p[0] for p in prism], [p[1] for p in prism], s=130,
               marker="s", facecolor="white", edgecolor=GRAY, linewidth=2,
               zorder=3, label="published 4-stage pipeline (PRISM)")
    xs, ys = [], []
    for s in ("s42", "s43", "s44"):
        r = json.loads((ROOT / "vast_test" / f"results_{s}.json").read_text())
        for k, v in r["evals"].items():
            xs.append(float(k) * 100)
            ys.append(v["recovery"] * 100)
    ax.scatter(xs, ys, s=150, marker="o", color=RED, edgecolor="white",
               linewidth=1.5, zorder=4, label="excise — one run, 12 minutes")
    ax.annotate("attribution alone:\nskill collapses", xy=(5.75, 29),
                xytext=(11, 38), fontsize=11, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
    ax.annotate("hand-tuned best:\n91% kept at 5%", xy=(5.05, 91.3),
                xytext=(11, 72), fontsize=11, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
    ax.annotate("excise: 97–101% kept,\nfloor found automatically",
                xy=(3.0, 97.5), xytext=(2.2, 116), fontsize=12, color=RED,
                weight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 30, 100])
    ax.set_xticklabels(["3%", "5%", "10%", "30%", "100%"], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlabel("how much of the model's MLP you keep", fontsize=12,
                  color=DARK)
    ax.set_ylabel("how much of the skill survives (%)", fontsize=12,
                  color=DARK)
    ax.axhline(100, color=GRAY, lw=0.8, ls=":")
    ax.set_xlim(2, 130)
    ax.set_ylim(0, 132)
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "frontier.png")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    frontier()
    print("assets ->", OUT)
