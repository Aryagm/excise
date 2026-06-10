#!/usr/bin/env python3
"""Generate all paper figures from the experiment artifacts in vast_test/."""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "vast_test"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
C_OURS, C_PRISM, C_BAD, C_GRey = "#b13d3d", "#7a7a7a", "#c8a13a", "#bbbbbb"


def load(name):
    return json.loads((ROOT / name).read_text())


# ---------------------------------------------------------------- fig 1
def fig_frontier():
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    # PRISM (from the paper, Qwen2.5-Math-1.5B 2-digit addition)
    prism = [(5.75, 29.0, "raw mask"), (90.61, 99.53, "raw, broad"),
             (5.05, 91.33, "staged collimation"), (4.65, 90.6, "refined MVC")]
    ax.scatter([p[0] for p in prism], [p[1] for p in prism], s=42, marker="s",
               color=C_PRISM, label="PRISM (staged pipeline)", zorder=3)
    for x, y, t in prism:
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(6, -3),
                    fontsize=7, color=C_PRISM)
    # ours: three seeds, evals
    xs, ys = [], []
    for s in ("s42", "s43", "s44"):
        r = load(f"results_{s}.json")
        for k, v in r["evals"].items():
            xs.append(float(k) * 100)
            ys.append(v["recovery"] * 100)
    ax.scatter(xs, ys, s=42, marker="o", color=C_OURS, zorder=4,
               label="excise (one joint run, 3 seeds)")
    ax.set_xscale("log")
    ax.set_xlabel("MLP channels kept (%)")
    ax.set_ylabel("recovery (% of base accuracy)")
    ax.set_xlim(2, 110)
    ax.set_ylim(0, 110)
    ax.axhline(100, color=C_GRey, lw=0.6, ls=":")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.savefig(OUT / "frontier.pdf")
    fig.savefig(OUT / "frontier.png")
    print("frontier: ours", sorted(zip(xs, ys)))


# ---------------------------------------------------------------- fig 2
def fig_fc():
    log = (ROOT / "logs" / "fc.log").read_text()
    probes = [(float(m.group(1)) * 100, float(m.group(2)) * 100)
              for m in re.finditer(
                  r"\[probe\] step=\d+ open=([\d.]+) self_match=([\d.]+)", log)]
    r = load("results_fc4b.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    xs, ys = zip(*sorted(probes))
    a1.plot(xs, ys, "o-", color=C_OURS, ms=4, lw=1.2,
            label="probe (train prompts)")
    a1.axhline(100, color=C_GRey, lw=0.6, ls=":")
    a1.set_xlabel("MLP channels open (%)")
    a1.set_ylabel("exact self-match (%)")
    a1.set_title("descent probe trace", fontsize=9)
    a1.invert_xaxis()
    a1.set_ylim(0, 105)
    a1.legend(fontsize=7.5, frameon=False, loc="lower left")

    budgets = sorted((float(k) * 100, v * 100) for k, v in r["evals"].items())
    a2.bar([f"{b:.0f}%" for b, _ in budgets], [v for _, v in budgets],
           color=C_OURS, width=0.62)
    a2.axhline(r["unmasked_self_match"] * 100, color=C_PRISM, lw=1.0, ls="--",
               label=f"unmasked ceiling ({r['unmasked_self_match']*100:.0f}%)")
    a2.set_xlabel("MLP channels kept")
    a2.set_ylabel("held-out exact self-match (%)")
    a2.set_title("held-out evaluation", fontsize=9)
    a2.set_ylim(0, 105)
    a2.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "function_calling.pdf")
    fig.savefig(OUT / "function_calling.png")
    print("fc: probes", len(probes), "| held", budgets)


# ---------------------------------------------------------------- fig 3
def fig_miscalibration():
    """v2.0: KL stayed under budget while true recovery collapsed."""
    bad_log = (ROOT / "logs" / "v20_failed" / "battery.log").read_text()
    bad = load("../vast_test/logs/v20_failed/results_s42.json".replace(
        "../vast_test/", ""))
    kls = [(float(m.group(1)), float(m.group(2)))
           for m in re.finditer(
               r"\[train\] step=\d+ kl=[\d.]+ ema=([\d.]+) "
               r"open=([\d.]+)", bad_log)]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    opens = [o * 100 for _, o in kls]
    emas = [k for k, _ in kls]
    ax.plot(opens, emas, "-", color=C_BAD, lw=1.4,
            label="EMA distillation KL (looks healthy)")
    ax.axhline(0.02, color=C_GRey, lw=0.8, ls="--", label="KL budget")
    ax.invert_xaxis()
    ax.set_xlabel("MLP channels open (%)")
    ax.set_ylabel("EMA KL")
    ax2 = ax.twinx()
    ax2.spines.right.set_visible(True)
    evals = sorted((float(k) * 100, v["recovery"] * 100)
                   for k, v in bad["evals"].items())
    ax2.scatter([e[0] for e in evals], [e[1] for e in evals], s=60, marker="x",
                color=C_OURS, label="true recovery (collapsed)")
    ax2.set_ylabel("recovery (%)", color=C_OURS)
    ax2.set_ylim(0, 110)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left")
    fig.savefig(OUT / "miscalibration.pdf")
    fig.savefig(OUT / "miscalibration.png")
    print("miscalibration: final evals", evals)


# ---------------------------------------------------------------- fig 4
def fig_ablations():
    runs = [("seed 42", "s42"), ("seed 43", "s43"), ("seed 44", "s44"),
            ("MLP-only\nadapter", "mlponly"), ("label-free\n(CE=0)", "ce0")]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    labels, recs, floors = [], [], []
    for label, name in runs:
        r = load(f"results_{name}.json")
        rec5 = r["evals"].get("0.0500", {}).get("recovery")
        labels.append(label)
        recs.append(rec5 * 100)
        floors.append(r["floor_frac"] * 100)
    bars = ax.bar(labels, recs, color=C_OURS, width=0.6)
    for b, f in zip(bars, floors):
        ax.annotate(f"floor\n{f:.1f}%", (b.get_x() + b.get_width() / 2, 12),
                    ha="center", fontsize=7, color="white")
    ax.axhline(91.33, color=C_PRISM, lw=1.0, ls="--",
               label="PRISM staged (91.3%)")
    ax.axhline(100, color=C_GRey, lw=0.6, ls=":")
    ax.set_ylabel("recovery @ 5% channels (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.savefig(OUT / "ablations.pdf")
    fig.savefig(OUT / "ablations.png")
    print("ablations:", dict(zip(labels, zip(recs, floors))))


if __name__ == "__main__":
    fig_frontier()
    fig_fc()
    fig_miscalibration()
    fig_ablations()
    print("figures ->", OUT)
