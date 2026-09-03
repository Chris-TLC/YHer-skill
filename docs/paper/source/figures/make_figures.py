"""Generate all paper figures from verified data. Run with .venv-pub python."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/Users/mac/Desktop/项目文件夹/Tools/PROJECT_HANDOFF/researchwrite/arxiv-paper-20260902/figures"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})
C = {"blue": "#4C72B0", "orange": "#DD8452", "green": "#55A868", "red": "#C44E52",
     "gray": "#8C8C8C", "purple": "#8172B3", "teal": "#64B5CD"}

# ---------- Fig 1: System overview (session flow) ----------
fig, ax = plt.subplots(figsize=(5.7, 3.0))
ax.axis("off")
steps = [
    ("1. Freeze", "Three disjoint item families (diagnostic / practice / held-out)", C["blue"]),
    ("2. Serve", "Server-side scoring; fail-closed before response", C["teal"]),
    ("3. Select", "EIG adaptive selection; four-state belief; prerequisite descent", C["green"]),
    ("4. Teach", "Checkpoint anchored to verified standard solutions", C["purple"]),
    ("5. Recommend", "Signed video-resource trails (track map); propensity snapshots", C["orange"]),
    ("6. Verify", "Held-out check on two unseen families; FSRS stability", C["red"]),
]
row_h = 0.135
y = 1.0
for i, (tag, desc, col) in enumerate(steps):
    y -= row_h
    box = FancyBboxPatch((0.02, y), 0.96, row_h - 0.018, boxstyle="round,pad=0.008",
                         fc=col, ec="none", alpha=0.9)
    ax.add_patch(box)
    ax.text(0.05, y + (row_h - 0.018)/2, f"{tag}   {desc}",
            ha="left", va="center", fontsize=7.2, color="white")
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, y - 0.024), xytext=(0.5, y - 0.002),
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.1))
ax.text(0.5, 0.015, "All six stages read the same evidence ledger; no stage can invent a fact.",
        ha="center", fontsize=6.6, style="italic", color="0.35")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
plt.savefig(f"{OUT}/fig1_system_flow.pdf"); plt.close()

# ---------- Fig 2: Quality funnel 6083 → 3329 → 2526 → 1202 ----------
stages = ["Raw slices", "Structured\n(v4)", "R5 pool", "Serviceable"]
counts = [6083, 3329, 2526, 1202]
yields = ["6,083", "3,329\n(54.7\%)", "2,526\n(75.9\%)", "1,202\n(47.6\%)"]
fig, ax = plt.subplots(figsize=(3.5, 2.4))
colors = [C["blue"], C["teal"], C["green"], C["orange"]]
bars = ax.bar(stages, counts, color=colors, width=0.6)
for b, c, y in zip(bars, counts, yields):
    ax.text(b.get_x() + b.get_width()/2, c * 1.05, y, ha="center",
            fontsize=6.6, va="bottom")
ax.set_yscale("log")
ax.set_ylabel("Items (log scale)")
ax.set_title("Quality funnel: 6,083 slices to 1,202 serviceable items")
ax.set_ylim(500, 30000)
ax.tick_params(axis="x", labelsize=7)
plt.savefig(f"{OUT}/fig2_funnel.pdf"); plt.close()

# ---------- Fig 3: Generation routes ----------
routes = ["From-scratch\n(2 identical rounds)", "Style transfer\n(adversarial)",
          "Style transfer\n(fair conditions)", "Figure-anchored\nreskin"]
vals = [87, 65, 60, 26/29*100]
cols = [C["gray"], C["orange"], C["teal"], C["purple"]]
fig, ax = plt.subplots(figsize=(3.8, 2.4))
bars = ax.barh(routes, vals, color=cols, height=0.55, edgecolor="0.25", lw=0.6)
for b, v in zip(bars, vals):
    ax.text(v + 1.5, b.get_y() + b.get_height()/2, f"{v:.0f}%", va="center", fontsize=7.2)
ax.axvline(100, color="0.35", ls="--", lw=0.9)
ax.text(100, 3.55, "human parity", fontsize=6.8, color="0.35", ha="center")
ax.set_xlabel("Distinguishability by blind expert (%)")
ax.set_xlim(0, 112)
ax.set_title("Generation verdicts by route (¥13.87)")
ax.tick_params(axis="y", labelsize=7)
plt.savefig(f"{OUT}/fig3_generation.pdf"); plt.close()

# ---------- Fig 4: Audit verdicts ----------
fig, ax = plt.subplots(figsize=(3.4, 2.0))
cats = ["Keep", "Modify", "Replace", "Downgrade"]
ns = [4, 10, 5, 2]
cols = [C["green"], C["orange"], C["red"], C["gray"]]
bars = ax.bar(cats, ns, color=cols, width=0.55)
for b, n in zip(bars, ns):
    ax.text(b.get_x() + b.get_width()/2, n + 0.25, str(n), ha="center", fontsize=8)
ax.set_ylabel("Components")
ax.set_title("21 engine components, final verdicts")
ax.set_ylim(0, 12)
plt.savefig(f"{OUT}/fig4_audit_verdicts.pdf"); plt.close()

# ---------- Fig 5: The hardest audit numbers (dual panel) ----------
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2))
# (a) diagnostic measurement error rates
ax = axes[0]
labels = ["Four-state\nclassification\nceiling (12 items)", "Old stopping\nrule false-stop\nrate"]
vals = [61.5, 10.3]
errs = [[7.5, 0.0], [7.5, 0.0]]
bars = ax.bar(labels, vals, color=[C["orange"], C["red"]], width=0.5,
              yerr=errs, capsize=3, error_kw=dict(lw=0.8))
for b, v, e in zip(bars, vals, errs):
    ax.text(b.get_x() + b.get_width()/2, v + e[0] + 6, f"{v:.1f}%", ha="center", fontsize=7)
ax.text(0, 88, "simulated range 54\u201369%", fontsize=6.4, ha="center", color="0.3")
ax.set_ylabel("Rate (%)")
ax.set_ylim(0, 100)
ax.set_title("(a) Measurement error rates", fontsize=8)
ax.tick_params(axis="x", labelsize=6.4)
# (b) memory stability in days (log scale)
ax = axes[1]
ax.bar(["Project S value\nafter 10 reviews\n(3-day rhythm)", "FSRS-4.5\nreference"], [4608, 73],
       color=[C["gray"], C["green"]], width=0.5)
ax.set_yscale("log")
ax.set_ylabel("Stability S (days, log)")
ax.set_ylim(12, 30000)
ax.set_title("(b) Memory stability: hand-set vs damped", fontsize=8)
ax.text(0, 7200, "4,608", fontsize=7.2, ha="center")
ax.text(1, 120, "73", fontsize=7.2, ha="center")
ax.tick_params(axis="x", labelsize=6.4)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_audit_numbers.pdf"); plt.close()

# ---------- Fig 6: Costs ----------
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2), gridspec_kw=dict(wspace=0.42))
# (a) absolute costs
items = ["Demo QA\n(205 events)", "Batch 13 recovery\n(105 assets)", "Five-generation\nrounds", "Full-pool VL audit\n(2,526 items)"]
costs = [1.12, 2.53, 13.87, 17.93]
axes[0].barh(items, costs, color=[C["green"], C["teal"], C["orange"], C["blue"]], height=0.6)
axes[0].set_xscale("log")
axes[0].set_xlabel("Total cost (¥, log scale)")
axes[0].set_title("(a) Absolute costs", fontsize=8)
axes[0].tick_params(axis="y", labelsize=6.6)
for i, c in enumerate(costs):
    axes[0].text(c * 1.15, i, f"¥{c:,.2f}", fontsize=6.8, va="center")
axes[0].set_xlim(0.6, 60)
# (b) unit costs (per item or per asset, log scale)
units = [17.93/2526, 2.53/105, 13.87/2526]
labels_u = ["VL audit\nper item", "Recovery\nper asset", "Gen rounds\nper bank item"]
axes[1].bar(labels_u, units, color=[C["blue"], C["teal"], C["orange"]], width=0.55)
axes[1].set_yscale("log")
axes[1].set_ylabel("Cost per unit (¥, log)")
axes[1].set_title("(b) Unit costs", fontsize=8)
axes[1].tick_params(axis="x", labelsize=6.4)
for i, u in enumerate(units):
    axes[1].text(i, u * 1.2, f"¥{u:.4f}", fontsize=6.4, ha="center")
axes[1].set_ylim(0.0005, 0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_costs.pdf"); plt.close()

print("figures done")
