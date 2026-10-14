"""_common_l4.py -- shared helpers for the Lecture 4 (Data Description and Visualization) examples (every line has an inline comment).

Unify fonts, palettes, and plotting parameters so the 13 figures look stylistically
consistent and legible in the slides; fix the random seed so the samples and figures
are exactly reproducible on every run.
"""
import json  # read/write JSON (geographic data and metadata)
import matplotlib  # pick the backend before importing the other plotting modules
matplotlib.use("Agg")  # headless backend: renders figures without a display
import matplotlib.pyplot as plt  # figure and axes interface
import numpy as np  # numeric arrays and random numbers
import pandas as pd  # tabular data structures
from pathlib import Path  # cross-platform path handling

SEED = 20260417  # fixed random seed shared across the lecture
RNG = np.random.default_rng(SEED)  # shared random generator: results are reproducible
ROOT = Path(__file__).resolve().parents[2]  # .../website (two levels up from code/lecture04)
FIG = ROOT / "figures" / "lecture04"  # figure output directory
DAT = ROOT / "data" / "lecture04"  # sample data output directory
INT = FIG / "interactive"  # plotly interactive HTML output directory
for _d in (FIG, DAT, INT):  # make sure each directory exists
    _d.mkdir(parents=True, exist_ok=True)  # create it automatically if missing

# ---------------------------------------------------------------- shared visual spec
FONT = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # Chinese font priority (Windows ships the first two)
PALETTE = ["#8A0C3C", "#C9A227", "#66717D", "#2E6F8E", "#711426",  # categorical palette: SZU lychee red + gold + gray + blue
           "#A6426B", "#4F7942", "#B5651D"]  # plus four more distinguishable colors
SEQ = "RdPu"  # sequential palette (single-hue ramp, good for a choropleth)
DIV = "RdBu_r"  # diverging palette (differences around a central baseline)
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",  # Okabe-Ito colorblind-friendly palette (8 colors)
         "#0072B2", "#D55E00", "#CC79A7", "#000000"]  # each color stays distinguishable in grayscale

plt.rcParams["font.sans-serif"] = FONT  # global Chinese font
plt.rcParams["axes.unicode_minus"] = False  # render the minus sign instead of a box
plt.rcParams["figure.dpi"] = 140  # interactive preview resolution
plt.rcParams["savefig.dpi"] = 200  # save resolution: sharper on a projector
plt.rcParams["savefig.bbox"] = "tight"  # trim extra white margins when saving
plt.rcParams["axes.grid"] = True  # turn on a light grid to make values easier to read
plt.rcParams["grid.alpha"] = 0.25  # grid transparency
plt.rcParams["axes.titlesize"] = 12.5  # subplot title font size
plt.rcParams["axes.labelsize"] = 11  # axis label font size
plt.rcParams["xtick.labelsize"] = 9.5  # x-axis tick font size
plt.rcParams["ytick.labelsize"] = 9.5  # y-axis tick font size
plt.rcParams["legend.fontsize"] = 9  # legend font size
plt.rcParams["figure.titlesize"] = 14  # figure title font size
plt.rcParams["axes.spines.top"] = False  # drop the top spine to reduce chart junk
plt.rcParams["axes.spines.right"] = False  # drop the right spine


def style_axes(ax, title=None, xlabel=None, ylabel=None):  # uniformly set the title and axis labels of one subplot
    """Apply the shared style to one subplot: title, axis labels, grid."""  # function contract
    if title:  # only set the title if provided
        ax.set_title(title)  # set the subplot title
    if xlabel:  # only set the x label if provided
        ax.set_xlabel(xlabel)  # set the x-axis label
    if ylabel:  # only set the y label if provided
        ax.set_ylabel(ylabel)  # set the y-axis label
    return ax  # return the same object for chaining


def annotated(ax, text, loc="upper left"):  # add an in-figure annotation (key to a self-contained caption)
    """Add one line of explanatory text in one of the four corners."""  # function contract
    xy = {"upper left": (0.02, 0.96), "upper right": (0.98, 0.96),  # the two upper positions
          "lower left": (0.02, 0.06), "lower right": (0.98, 0.06)}.get(loc, (0.02, 0.96))  # the two lower positions; unknown ones fall back to upper left
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, va="top",  # place the text in axes coordinates
            ha="left" if "left" in loc else "right", fontsize=8.6,  # left/right alignment
            color="#1E2630", bbox=dict(boxstyle="round,pad=0.25",  # add a light box behind the text for readability
                                       fc="#F7F3F2", ec="#D6CDD0", lw=0.6))  # background and border
    return ax  # return the same object


def save(fig, name):  # save the figure and free memory
    """Save the figure to website/figures/lecture04 and close the graphics object."""  # function contract
    path = FIG / name  # build the full path
    fig.savefig(path, facecolor="white")  # save on a white background for easy slide insertion
    plt.close(fig)  # close the figure to free memory
    print("[figure]", path)  # print the artifact path


def dump(df, name):  # write the sample table into the data directory
    """Write the DataFrame as UTF-8 CSV into website/data/lecture04."""  # function contract
    path = DAT / name  # build the full path
    df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel opens it cleanly
    print("[data]", path)  # print the artifact path


def load_geojson(name):  # read the (optional) geographic boundary file
    """Read GeoJSON from the data directory; return None if the file is missing so the caller can degrade."""  # function contract
    path = DAT / name  # build the path
    if not path.exists():  # file missing
        print("[geojson] not found, fall back to a symbol map:", path)  # warn that it will fall back to a symbol map
        return None  # return an empty value
    with open(path, "r", encoding="utf-8") as fh:  # read as UTF-8
        return json.load(fh)  # return the parsed dictionary


def lorenz(values):  # compute the Lorenz curve (measures inequality)
    """Return the cumulative population share and cumulative value share for the Lorenz curve and Gini coefficient."""  # function contract
    x = np.sort(np.asarray(values, dtype=float))  # sort ascending
    cum = np.cumsum(x) / x.sum()  # cumulative share
    return np.insert(np.arange(1, len(x) + 1) / len(x), 0, 0), np.insert(cum, 0, 0)  # pad both ends with 0
