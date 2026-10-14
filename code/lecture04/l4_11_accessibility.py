"""l4_11_accessibility.py -- colorblind-friendliness and accessibility: palette comparison, grayscale check, and contrast computation.

Accessibility is not "nice to have": about 8% of men have some degree of color-vision deficiency,
and grayscale printing also erases color differences. The script compares three palettes, runs a
grayscale simulation, computes the WCAG contrast between text and background, and lists the checks.
"""
import numpy as np  # numeric computation
import pandas as pd  # data tables
import matplotlib.pyplot as plt  # comparison figures
import matplotlib.colors as mcolors  # color-space conversion (to grayscale)
from _common_l4 import OKABE, PALETTE, DAT, save, dump  # reuse the shared helpers and the two palettes

counts = pd.read_csv(DAT / "l4_category_counts.csv")  # category counts (from l4_05)
labels = counts["category"].tolist()  # category names
values = counts["count"].tolist()  # count per category

# ---------------------------------------------------------------- 1) three palettes
palettes = {  # the three palettes to compare
    "matplotlib default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"],
    "SZU theme colors": PALETTE[:len(labels)],  # this course's brand palette
    "Okabe-Ito (colorblind-friendly)": OKABE[:len(labels)],  # a widely used colorblind-friendly palette
}  # end of the three-palette dictionary
def to_gray(hex_color):  # convert a color to grayscale brightness
    """Compute perceived brightness with ITU-R BT.601 weights (0-1)."""  # function contract
    r, g, b = mcolors.to_rgb(hex_color)  # convert to RGB in 0-1
    return 0.299 * r + 0.587 * g + 0.114 * b  # weighted sum to get the brightness
def contrast_ratio(fg, bg):  # compute the WCAG contrast
    """Return the contrast between foreground and background (1-21; 4.5+ passes for body text)."""  # function contract
    l1, l2 = to_gray(fg), to_gray(bg)  # compute both brightnesses
    hi, lo = max(l1, l2), min(l1, l2)  # brighter and darker
    return (hi + 0.05) / (lo + 0.05)  # WCAG formula
report_rows = []  # accessibility check table
for name, pal in palettes.items():  # evaluate each palette
    grays = [to_gray(c) for c in pal[:len(labels)]]  # convert each color to grayscale
    gaps = np.diff(np.sort(grays))  # adjacent grayscale gaps after sorting
    min_gap = float(gaps.min()) if len(gaps) else 0.0  # smallest grayscale gap (larger = easier to tell apart)
    contrasts = [contrast_ratio(c, "#FFFFFF") for c in pal[:len(labels)]]  # contrast against a white background
    report_rows.append({"palette": name,  # palette name
                        "min_gray_gap": round(min_gap, 3),  # grayscale separability
                        "min_contrast_vs_white": round(min(contrasts), 2),  # lower bound of contrast
                        "body_text_pass(>=4.5)": bool(min(contrasts) >= 4.5),  # WCAG body-text threshold
                        "usable_as_text_color": bool(min(contrasts) >= 4.5)})  # whether it works as text color
report = pd.DataFrame(report_rows)  # summarize into a table
dump(report, "l4_accessibility_report.csv")  # write the check table to disk
print(report.to_string(index=False))  # print to the terminal

# ---------------------------------------------------------------- 2) figure: palettes, grayscale, and contrast
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), gridspec_kw={"width_ratios": [1.7, 1.2, 1.1]})  # three cells: palette grayscale comparison / grayscale brightness distribution / contrast matrix
y_pos = np.arange(len(palettes))[::-1]  # one row per palette
for i, (name, pal) in enumerate(palettes.items()):  # cell 1: color blocks + grayscale blocks
    for j, c in enumerate(pal[:len(labels)]):  # draw a square per color
        axes[0].add_patch(plt.Rectangle((j, y_pos[i] + 0.35), 0.9, 0.3,  # color square
                                        color=c, transform=axes[0].transData))  # color
        gray = to_gray(c)  # grayscale of this color
        axes[0].add_patch(plt.Rectangle((j, y_pos[i] - 0.05), 0.9, 0.3,  # grayscale square
                                        color=(gray, gray, gray), transform=axes[0].transData))  # grayscale value
axes[0].set_xlim(-0.2, len(labels) + 0.2); axes[0].set_ylim(-0.3, len(palettes) - 0.3)  # coordinate range
axes[0].set_xticks([j + 0.45 for j in range(len(labels))], labels, rotation=25, fontsize=8)  # category labels
axes[0].set_yticks(y_pos + 0.3, list(palettes.keys()), fontsize=8.6)  # palette names
axes[0].grid(False)  # a swatch chart does not need a grid
axes[0].set_title("Top: original colors; bottom: grayscale simulation -- if grayscale can't separate them, colorblind readers are misled")  # subplot title
x = np.arange(len(labels))  # x coordinates
width = 0.26  # bar width
for k, (name, pal) in enumerate(palettes.items()):  # cell 2: grayscale-gap comparison
    grays = [to_gray(c) for c in pal[:len(labels)]]  # grayscale of each color
    axes[1].bar(x + (k - 1) * width, grays, width=width,  # grouped bars
                color=pal[:len(labels)], alpha=0.9, label=name)  # fill with the color itself
axes[1].set_xticks(x, labels, rotation=25, fontsize=8)  # category labels
axes[1].set_ylabel("perceived brightness (0-1)")  # axis label
axes[1].set_title("The more spread out the grayscale brightness, the easier to tell apart")  # subplot title
axes[1].legend(fontsize=7.6)  # legend
bg_list = ["#FFFFFF", "#F7F3F2", "#1E2630"]  # three backgrounds
fg_list = ["#8A0C3C", "#66717D", "#C9A227", "#2E6F8E"]  # four foregrounds
matrix = np.array([[contrast_ratio(f, b) for b in bg_list] for f in fg_list])  # contrast matrix
im = axes[2].imshow(matrix, cmap="RdPu", vmin=1, vmax=21)  # cell 3: contrast heatmap
axes[2].set_xticks(range(len(bg_list)), ["White", "Light bg", "Dark bg"], fontsize=8.6)  # background labels
axes[2].set_yticks(range(len(fg_list)), fg_list, fontsize=8.6)  # foreground labels
for i in range(matrix.shape[0]):  # write the contrast value in each cell
    for j in range(matrix.shape[1]):  # double loop
        axes[2].text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8.6,  # write the value in the contrast matrix
                     color="white" if matrix[i, j] > 10 else "#1E2630")  # value label
axes[2].grid(False)  # a heatmap does not need a grid
axes[2].set_title("WCAG contrast (>=4.5 passes)")  # subplot title
plt.colorbar(im, ax=axes[2], fraction=0.046)  # colorbar
plt.tight_layout()  # adjust the three cells
save(fig, "fig_palette_accessibility.png")  # save: accessibility comparison

# ---------------------------------------------------------------- 3) checklist
CHECKLIST = [  # accessibility self-check list
    "Choose a colorblind-friendly palette (Okabe-Ito / viridis, etc.); avoid red-green contrast",  # check 1: palette choice
    "Add redundant encoding beyond color (shape, line style, pattern, direct labels)",  # check 2: redundant encoding
    "Still distinguishable after grayscale printing (checked in the second panel here)",  # check 3: grayscale separability
    "Text-vs-background contrast >= 4.5 (>= 3.0 for large text)",  # check 4: contrast threshold
    "Font size: >= 18pt body text on a projector, >= 12pt chart annotations",  # check 5: minimum font sizes
    "Don't convey up/down or good/bad with color alone (red-green is the worst)",  # check 6: don't use color alone for good/bad
    "Provide alt text for every chart (one sentence stating the conclusion)",  # check 7: provide alt text
    "Interactive charts should offer keyboard-accessible filtering and zoom",  # check 8: keyboard accessibility
]  # end of the checklist
dump(pd.DataFrame({"check_item": CHECKLIST}), "l4_accessibility_checklist.csv")  # write the checklist to disk
print("checklist length:", len(CHECKLIST))  # print the checklist size
print("gray-level gap of a red-green pair (example):", round(abs(to_gray("#8A0C3C") - to_gray("#2E6F8E")), 3))  # example value
