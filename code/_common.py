"""_common.py -- Shared helpers: fonts, output folders, and save utilities (every line is commented)."""  # module docstring: states the purpose
import matplotlib                                                    # import the main matplotlib package
matplotlib.use("Agg")                                                # use the headless backend so figures render without a display
import matplotlib.pyplot as plt                                      # import the plotting interface as plt
import numpy as np                                                   # import the numerical computing library as np
from pathlib import Path                                             # import the cross-platform path utility

RNG = np.random.default_rng(20260917)                                # fixed random seed so every run gives the same result
FIG = Path(__file__).resolve().parents[1] / "figures"                # figure output folder: website/figures under the repo
DAT = Path(__file__).resolve().parents[1] / "data"                   # data output folder: website/data under the repo
FIG.mkdir(parents=True, exist_ok=True)                               # create the figure folder if missing (including parents)
DAT.mkdir(parents=True, exist_ok=True)                               # create the data folder if missing (including parents)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # CJK font fallback list (the first two ship with Windows)
plt.rcParams["font.monospace"] = ["Consolas", "Microsoft YaHei", "DejaVu Sans Mono"]  # also add CJK fonts to monospace to avoid missing glyphs
plt.rcParams["axes.unicode_minus"] = False                           # render axis minus signs correctly instead of boxes
plt.rcParams["figure.dpi"] = 140                                     # default on-screen/saved resolution
plt.rcParams["savefig.dpi"] = 200                                    # resolution when saving figures (sharper for slides)
plt.rcParams["savefig.bbox"] = "tight"                               # trim extra whitespace when saving
plt.rcParams["axes.grid"] = True                                     # enable the grid by default for easier reading
plt.rcParams["grid.alpha"] = 0.25                                    # grid opacity, kept subtle
plt.rcParams["axes.titlesize"] = 13                                  # subplot title size (prioritize projector readability)
plt.rcParams["axes.labelsize"] = 11                                  # axis label size
plt.rcParams["xtick.labelsize"] = 10                                 # x tick label size
plt.rcParams["ytick.labelsize"] = 10                                 # y tick label size
plt.rcParams["legend.fontsize"] = 9.2                                # legend font size


def save(fig, name):                                                 # define the save helper: takes a figure and a file name
    """Save the figure into the FIG folder and close it to avoid memory leaks."""  # function description
    path = FIG / name                                                # build the full output path
    fig.savefig(path, facecolor="white")                             # save as PNG on a white background
    plt.close(fig)                                                   # close the figure to free memory
    print("[figure]", path)                                          # print the output path for checking


def dump_csv(df, name):                                              # define the CSV dump helper
    """Write a DataFrame into the DAT folder (UTF-8-SIG so Excel opens CJK text)."""  # function description
    path = DAT / name                                                # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")               # write the CSV without the index, Excel-friendly
    print("[data]", path)                                            # print the output path
