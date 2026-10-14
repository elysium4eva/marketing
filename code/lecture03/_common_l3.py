"""_common_l3.py -- shared helpers for the Lecture 3 (data preparation) examples (every line carries a comment).

A fixed random seed keeps the samples and figures identical on every run, which is the
precondition for reproducible lecture examples.
"""
import matplotlib                                                  # pick the plotting backend first, then import the rest of the plotting modules
matplotlib.use("Agg")                                              # headless backend: figures render even without a display
import matplotlib.pyplot as plt                                    # figure and axes interface
import numpy as np                                                 # numeric arrays and random numbers
import pandas as pd                                                # tabular data structures
from pathlib import Path                                           # cross-platform path handling

SEED = 20260317                                                    # fixed random seed shared by every script
RNG = np.random.default_rng(SEED)                                  # shared random generator: keeps results reproducible
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels up from code/lecture03)
FIG = ROOT / "figures" / "lecture03"                               # figure output directory for Lecture 3
DAT = ROOT / "data" / "lecture03"                                  # sample data output directory for Lecture 3
FIG.mkdir(parents=True, exist_ok=True)                             # create the directory if it does not exist
DAT.mkdir(parents=True, exist_ok=True)                             # create the directory if it does not exist

plt.rcParams["figure.dpi"] = 140                                   # interactive preview resolution
plt.rcParams["savefig.dpi"] = 200                                  # saved resolution: sharper when projected
plt.rcParams["savefig.bbox"] = "tight"                             # trim the excess white margin when saving
plt.rcParams["axes.grid"] = True                                   # light grid on, so values are easier to read
plt.rcParams["grid.alpha"] = 0.25                                  # grid opacity, so it does not dominate the figure
plt.rcParams["axes.titlesize"] = 13                                # subplot title size (readable when projected)
plt.rcParams["axes.labelsize"] = 11                                # axis label font size
plt.rcParams["xtick.labelsize"] = 10                               # x tick label font size
plt.rcParams["ytick.labelsize"] = 10                               # y tick label font size
plt.rcParams["legend.fontsize"] = 9.2                              # legend font size
plt.rcParams["font.size"] = 10                                     # default size for all remaining text


def save(fig, name):                                               # save one figure and free its memory
    """Save a figure into website/figures/lecture03 and close the canvas."""   # function contract
    path = FIG / name                                              # build the full output path
    fig.savefig(path, facecolor="white")                           # save on a white background, ready to paste into slides
    plt.close(fig)                                                 # close the figure to free memory
    print("[figure]", path)                                        # print the artifact path for inspection


def dump(df, name):                                                # write a sample table into the data directory
    """Write a DataFrame as UTF-8 CSV into website/data/lecture03."""   # function contract
    path = DAT / name                                              # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig so Excel renders non-ASCII text correctly
    print("[data]", path)                                          # print the artifact path for inspection


def money_to_float(series):                                        # turn messy amount strings into numbers
    """Strip currency symbols, thousands separators and spaces, then convert to float."""   # function contract
    cleaned = (series.astype(str)                                  # cast to string first so regex can be applied
               .str.replace(r"[^\d.\-]", "", regex=True)           # keep only digits, decimal point and minus sign
               .replace({"": np.nan}))                             # empty strings become real missing values
    return pd.to_numeric(cleaned, errors="coerce")                 # anything unparsable also becomes NaN
