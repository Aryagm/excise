#!/usr/bin/env python3
"""Generate all paper figures from the experiment artifacts in vast_test/.
Styled for print: serif/STIX typography, restrained palette, subtle grids."""

import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
try:
    from adjustText import adjust_text
except ImportError as exc:
    raise SystemExit(
        "paper/make_figures.py needs the paper figure extras: "
        "pip install '.[paper]'"
    ) from exc

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
    "axes.edgecolor": "#1f2328",
    "axes.labelcolor": "#1f2328",
    "axes.linewidth": 0.75,
    "xtick.color": "#1f2328",
    "ytick.color": "#1f2328",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid": True,
    "grid.alpha": 1.0,
    "grid.color": "#e7e9ee",
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})
INK = "#1f2328"
MUTED = "#667085"
GRID = "#e7e9ee"
PANEL = "#f8fafc"
EXCISE = "#c43c39"
EXCISE_DARK = "#9f2f2d"
EXCISE_LIGHT = "#fff3f1"
EXCISE_PALE = "#d98882"
PRISM = "#416f8f"
FLOOR = "#b68a2a"
CONTROL = "#8a8f98"

# Backwards-compatible aliases used throughout the figure code.
RED, GRAY, GOLD, LGRAY = EXCISE, MUTED, FLOOR, GRID


def load(name):
    return json.loads((ROOT / name).read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


# ------------------------------------------------------------- method
def fig_method():
    fig, ax = plt.subplots(figsize=(6.8, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, eyebrow, title, lines, accent=False):
        fc = EXCISE_LIGHT if accent else PANEL
        ec = EXCISE_DARK if accent else CONTROL
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.45,rounding_size=1.1",
            facecolor=fc, edgecolor=ec, linewidth=1.15))
        ax.text(x + 1.1, y + h - 2.2, eyebrow, ha="left", va="center",
                fontsize=5.7, color=EXCISE_DARK if accent else MUTED,
                weight="bold")
        ax.text(x + w / 2, y + h - 5.0, title, ha="center", va="center",
                fontsize=8.1, weight="bold",
                color=EXCISE_DARK if accent else INK)
        for i, ln in enumerate(lines):
            if len(lines) == 1:
                line_y = y + 1.9
            elif len(lines) > 3:
                line_y = y + h - 7.3 - 2.75 * i
            else:
                line_y = y + h - 8.2 - 3.55 * i
            body_size = 6.3 if len(lines) == 1 else 6.8
            ax.text(x + w / 2, line_y, ln, ha="center", va="center",
                    fontsize=body_size, color=INK)

    def arrow(x1, y1, x2, y2, label=None, dy=2.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=9.5,
                                     linewidth=1.0, color=MUTED,
                                     shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2, max(y1, y2) + dy, label, ha="center",
                    fontsize=6.3, color=MUTED, style="italic")

    ax.add_patch(FancyBboxPatch(
        (23.4, 5.8), 57.2, 22.6,
        boxstyle="round,pad=0.25,rounding_size=1.4",
        facecolor="#fff8f6", edgecolor="#f0c7c1", linewidth=0.7,
        linestyle=":"))
    ax.text(52.0, 4.0, "one training run: selection and relocation happen together",
            ha="center", va="center", fontsize=6.7, color=EXCISE_DARK,
            style="italic")

    box(1.5, 10.0, 18.0, 17.2, "PROMPTS ONLY", "teacher cache",
        ["frozen base $M$", "greedy outputs $\\hat{y}$",
         "top-$k$ + residual mass"])
    box(26.0, 10.0, 23.0, 17.2, "TRAINED", "joint student",
        ["LoRA adapter $\\theta$", "hard-concrete gates $z$",
         "masked KL objective", "unmasked guardrail"],
        accent=True)
    box(57.0, 10.0, 21.5, 17.2, "STOPPING RULE", "controller",
        ["lower target while KL ok", "dev probes vs unmasked",
         "rollback after 2 strikes"])
    box(86.0, 18.8, 12.5, 9.2, "OUTPUT", "frontier", ["+ receipts"])
    box(86.0, 5.4, 12.5, 9.2, "OUTPUT", "sliced", ["$\\equiv$ mask"])

    arrow(20.5, 18.6, 25.2, 18.6, "targets", 4.2)
    arrow(49.8, 18.6, 56.1, 18.6, "probe", 2.4)
    arrow(79.4, 20.6, 85.0, 23.2)
    arrow(79.4, 16.3, 85.0, 10.8)
    save(fig, "method")


# ------------------------------------------------------------ frontier
def fig_frontier():
    fig = plt.figure(figsize=(5.15, 3.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.65, 0.78], hspace=0.38)
    ax = fig.add_subplot(gs[0])
    tab = fig.add_subplot(gs[1])

    v02_runs = []
    for s in (42, 43, 44):
        d = json.loads((ROOT / "library_runs" / "v02" /
                        f"arith_v02_long_s{s}" / "summary.json").read_text())
        v02_runs.append((d["floor"] * 100, d["frontier"][-1][1] * 100))
    deep = json.loads((ROOT / "library_runs" / "v02" / "arith_v02_deep_s42" /
                       "summary.json").read_text())
    v02_x = sum(p[0] for p in v02_runs) / len(v02_runs)
    v02_y = sum(p[1] for p in v02_runs) / len(v02_runs)
    v02_min = min(p[1] for p in v02_runs)
    v02_max = max(p[1] for p in v02_runs)
    deep_pt = (deep["floor"] * 100, deep["frontier"][-1][1] * 100)
    old_pt = (7.6, 89.0)
    prism = [(5.75, 29.0), (90.61, 99.53), (5.05, 91.33), (4.65, 90.6)]

    def display_offset(point, dx_pt=0, dy_pt=0):
        x_disp, y_disp = ax.transData.transform(point)
        scale = fig.dpi / 72.0
        return ax.transData.inverted().transform(
            (x_disp + dx_pt * scale, y_disp + dy_pt * scale))

    def log_path(p0, p1, t):
        log_x = (1 - t) * math.log10(p0[0]) + t * math.log10(p1[0])
        return 10 ** log_x, (1 - t) * p0[1] + t * p1[1]

    ax.set_xscale("log")
    ax.set_xlim(0.6, 120)
    ax.set_ylim(0, 112)
    ax.set_xticks([0.7, 1.2, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["0.7", "1.2", "2", "5", "10", "20", "50", "100"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.axvspan(0.65, 1.3, color=EXCISE_LIGHT, zorder=0)
    ax.add_patch(FancyArrowPatch(
        old_pt, (v02_x, v02_y), arrowstyle="-|>", mutation_scale=12,
        lw=1.55, color=RED, shrinkA=9, shrinkB=12,
        connectionstyle="arc3,rad=-0.12", zorder=2))
    ax.add_patch(FancyArrowPatch(
        log_path((v02_x, v02_y), deep_pt, 0.30),
        log_path((v02_x, v02_y), deep_pt, 0.72),
        arrowstyle="-|>", mutation_scale=9,
        lw=1.0, color=EXCISE_DARK, shrinkA=0, shrinkB=0,
        connectionstyle="arc3,rad=0.08", zorder=2))
    ax.scatter([old_pt[0]], [old_pt[1]], s=76, color=CONTROL,
               edgecolor="white", linewidth=1.0, zorder=5)
    ax.scatter([v02_x], [v02_y], s=86, marker="D", color=RED,
               edgecolor="white", linewidth=1.0, zorder=6)
    ax.scatter([deep_pt[0]], [deep_pt[1]], s=76, marker="D",
               color=EXCISE_DARK, edgecolor="white", linewidth=1.0, zorder=7)
    ax.scatter([p[0] for p in prism], [p[1] for p in prism], s=48, marker="s",
               facecolor="white", edgecolor=PRISM, linewidth=1.15, zorder=5)

    labels = [
        (old_pt, "v0.1\n7.6%, 89.0%", CONTROL, 10, -18, "left"),
        ((v02_x, v02_y), f"v0.2 aggregate\n1.2%, {v02_y:.1f}%",
         RED, 16, 44, "left"),
        (deep_pt, "extended\n0.71%, 91.1%", EXCISE_DARK, 8, -10, "left"),
        ((5.75, 29.0), "raw mask\n5.75%, 29.0%", PRISM, 12, -12, "left"),
        ((90.61, 99.53), "raw, broad\n90.61%, 99.5%", PRISM, -14, -14,
         "right"),
        ((5.05, 91.33), "staged collimation\n5.05%, 91.3%", PRISM,
         12, 20, "left"),
        ((4.65, 90.6), "refined MVC\n4.65%, 90.6%", PRISM, -18, -22,
         "right"),
    ]
    texts, target_x, target_y = [], [], []
    for point, label, color, dx, dy, ha in labels:
        x, y = display_offset(point, dx, dy)
        texts.append(ax.text(x, y, label, ha=ha, va="center",
                             fontsize=6.5 if color != INK else 7.1,
                             color=color, zorder=8))
        target_x.append(point[0])
        target_y.append(point[1])
    adjust_text(
        texts, ax=ax, target_x=target_x, target_y=target_y,
        x=[old_pt[0], v02_x, deep_pt[0], *[p[0] for p in prism]],
        y=[old_pt[1], v02_y, deep_pt[1], *[p[1] for p in prism]],
        expand=(1.08, 1.16), force_text=(0.08, 0.16),
        force_static=(0.08, 0.12), force_pull=(0.01, 0.02),
        max_move=(7, 7), iter_lim=250, min_arrow_len=7,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.45,
                        shrinkA=2, shrinkB=3))
    ax.annotate("6.3x lower floor\nwith higher recovery",
                xy=log_path(old_pt, (v02_x, v02_y), 0.58),
                xytext=(0, -46), textcoords="offset points",
                ha="center", va="top", fontsize=7.1, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.55,
                                shrinkA=2, shrinkB=3))
    ax.set_xlabel("MLP channels kept at controller exit, % (log scale)",
                  labelpad=5)
    ax.set_ylabel("fidelity / reported recovery (%)", labelpad=5)
    ax.axhline(100, color=LGRAY, lw=0.7, ls=":", zorder=1)

    tab.axis("off")
    tab.set_xlim(0, 1)
    tab.set_ylim(0, 1)
    rows = [
        ("floor", "7.6%", "1.2%", "6.3x lower", RED),
        ("held-out self-match", "89.0%",
         f"{v02_min:.1f}-{v02_max:.1f}%", "+2.5 pts mean", RED),
        ("physical export", "0.48B", "0.17B",
         "2.8x fewer parameters", EXCISE_DARK),
    ]
    for y, (name, old, new, note, color) in zip([0.76, 0.47, 0.18], rows):
        tab.plot([0.0, 1.0], [y - 0.13, y - 0.13], color=GRID, lw=0.7)
        tab.text(0.02, y, name, ha="left", va="center", fontsize=7.35,
                 color=INK, weight="bold")
        tab.text(0.40, y, old, ha="right", va="center", fontsize=8.0,
                 color=CONTROL)
        tab.text(0.49, y, r"$\rightarrow$", ha="center", va="center",
                 fontsize=9.0, color=MUTED)
        tab.text(0.56, y, new, ha="left", va="center", fontsize=8.55,
                 color=color, weight="bold")
        tab.text(0.98, y, note, ha="right", va="center", fontsize=7.05,
                 color=MUTED)
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
    a2.axhline(r["unmasked_self_match"] * 100, color=CONTROL, lw=1.0, ls="--",
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
    ax.axhline(0.02, color=CONTROL, lw=0.8, ls="--", label="KL budget (left)")
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
            ("MLP-only\nadapter", "mlponly"), ("no hard-token CE\n($\\beta_{ce}{=}0$)", "ce0")]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    labels, recs, floors = [], [], []
    for label, name in runs:
        r = load(f"results_{name}.json")
        labels.append(label)
        recs.append(r["evals"]["0.0500"]["recovery"] * 100)
        floors.append(r["floor_frac"] * 100)
    bars = ax.bar(labels, recs, color=[RED] * 3 + [EXCISE_PALE] * 2, width=0.62,
                  zorder=3)
    for b, f, rec in zip(bars, floors, recs):
        ax.annotate(f"{rec:.1f}", (b.get_x() + b.get_width() / 2, rec + 1.5),
                    ha="center", fontsize=7.2, color=INK)
        ax.annotate(f"floor {f:.1f}%",
                    (b.get_x() + b.get_width() / 2, 8), ha="center",
                    fontsize=6.8, color="white", weight="bold")
    ax.axhline(91.33, color=PRISM, lw=1.0, ls="--")
    ax.annotate("PRISM staged pipeline (91.3%)", xy=(4.42, 91.3),
                xytext=(4.42, 110), fontsize=7.2, color=PRISM, ha="right",
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
    a1.plot(*zip(*tr), "-", color=CONTROL, lw=1.0,
            label="train batches")
    a1.plot(*zip(*an), "-", color=RED, lw=1.2,
            label="off-task anchors")
    a1.set_yscale("log")
    a1.set_xlabel("step")
    a1.set_ylabel("guardrail KL to base")
    a1.set_title("(a) the same adapter, two guardrail views", fontsize=8.2)
    a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              ncol=2, fontsize=6.4)

    # (b) probe trace: masked vs unmasked-at-same-step, with the rollback
    opens = [p["open"] * 100 for p in r["probe_trace"]]
    masked = [p["self_match"] * 100 for p in r["probe_trace"]]
    unm = [p["unmasked"] * 100 for p in r["probe_trace"]]
    a2.plot(opens, unm, "--", color=CONTROL, lw=1.1, marker="o", ms=3.5,
            label="unmasked, same step (baseline)")
    a2.plot(opens, masked, "-", color=RED, lw=1.3, marker="o", ms=3.5,
            label="masked probe")
    floor = r["floor"] * 100
    a2.axvline(floor, color=GOLD, lw=1.1, ls=":")
    a2.annotate(f"floor {floor:.1f}%\n(rolled back to\nlast passing probe)",
                xy=(floor, 40), xytext=(floor + 13, 48), fontsize=6.8,
                color=INK, ha="right", va="center",
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
        bars = ax.bar(labels, ys, color=[CONTROL, RED], width=0.55, zorder=3)
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
    colors = [RED, GOLD, CONTROL]
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
                arrowprops=dict(arrowstyle="-", color=CONTROL, lw=0.7,
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
