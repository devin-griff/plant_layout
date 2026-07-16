"""Regenerates the formulation schematic (formulation.png).

Illustrates disjunct Y^1 ("i is left of j") of the decoupled model: two
blocks with the rectilinear CENTER-TO-CENTER distances labeled: dx_{i,j}
horizontal, dy_{i,j} vertical, measured between the block centers (the
process-plant layout literature convention). Keep this in sync with the
model in app.py; re-run after any notation change so the image never goes
stale again.

    python formulation_schematic.py    # writes formulation.png next to this file

The formulation notebook ("plant layout.ipynb") references this PNG by
relative path (images/formulation.png), which GitHub's notebook preview
renders; cell attachments do not. No re-embedding needed: the notebook
picks up a regenerated PNG automatically.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(10, 5.2))

# Block i upper-left, block j lower-right (i is left of AND above j).
i_x, i_y, i_w, i_h = 1.2, 3.4, 2.4, 1.4   # x[1.2, 3.6], y[3.4, 4.8]
j_x, j_y, j_w, j_h = 4.6, 1.2, 2.6, 1.4   # x[4.6, 7.2], y[1.2, 2.6]
ci_x, ci_y = i_x + i_w / 2, i_y + i_h / 2
cj_x, cj_y = j_x + j_w / 2, j_y + j_h / 2

for bx, by, bw, bh in [(i_x, i_y, i_w, i_h), (j_x, j_y, j_w, j_h)]:
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, lw=1.8,
                           edgecolor="black"))

# Labels sit off-center so the dotted center guides don't cross them.
ax.text(ci_x - 0.62, ci_y + 0.36, r"$i$", ha="center", va="center",
        fontsize=22, style="italic")
ax.text(cj_x - 0.68, cj_y + 0.36, r"$j$", ha="center", va="center",
        fontsize=22, style="italic")

# Center marks and dotted guides through each center: the distances are
# center-to-center, so the guides run through the block centers.
for cx, cy in [(ci_x, ci_y), (cj_x, cj_y)]:
    ax.plot([cx], [cy], marker="o", ms=6, color="black")
for cx in (ci_x, cj_x):
    ax.plot([cx, cx], [0.9, 5.15], ls=":", color="black", lw=1.9)
for cy in (ci_y, cj_y):
    ax.plot([0.7, 8.0], [cy, cy], ls=":", color="black", lw=1.9)

# dx: horizontal center distance, drawn above the blocks.
ay = 4.95
ax.annotate("", xy=(cj_x, ay), xytext=(ci_x, ay),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
ax.text((ci_x + cj_x) / 2, ay + 0.12, r"$dx_{i,j}$", ha="center", va="bottom",
        fontsize=18)

# dy: vertical center distance, drawn to the right of block j.
axx = 7.6
ax.annotate("", xy=(axx, ci_y), xytext=(axx, cj_y),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
ax.text(axx + 0.12, (ci_y + cj_y) / 2, r"$dy_{i,j}$", ha="left",
        va="center", fontsize=18)

# Active-disjunct indicator, in the open band between the blocks.
ax.text(4.15, 3.0, r"$Y^{1}_{i,j} = 1$", ha="center", va="center",
        fontsize=20)

ax.set_xlim(0.5, 8.4)
ax.set_ylim(0.8, 5.6)
ax.set_aspect("equal")
ax.axis("off")

out = Path(__file__).parent / "formulation.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", out)
