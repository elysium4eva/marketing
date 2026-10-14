"""l3_09_panel_balance.py -- balanced and unbalanced panels, gaps and interpolation.

The panel example from the slides becomes real data here: a balanced panel with T=11 and n=2, a
"nominally balanced" panel where each firm has a different T_i, and a panel with block missingness.
The script diagnoses the structure first, then interpolates within each unit and marks every imputed
cell so that the patching stays visible.
"""
import numpy as np                                                 # random walks to generate the panel values
import pandas as pd                                                # panel reshaping and interpolation
import matplotlib.pyplot as plt                                    # panel heat maps
from _common_l3 import RNG, save, dump                             # reuse the shared seed, figure and dump helpers

MONTHS = list(range(1, 12))                                        # T = 11 periods, matching the slides

# ---------------------------------------------------------------- 1) balanced panel: n=2, T=11
def make_series(n, start):                                         # generate one firm-level time series
    """A random walk with its own level and trend, one value per period."""   # function contract
    level = start + np.cumsum(RNG.normal(0.6, 1.1, n))             # accumulate random changes period by period
    return np.round(level, 2)                                      # round for readability
balanced = pd.DataFrame({"firm_id": np.repeat(["F1", "F2"], len(MONTHS)),   # two firms
                         "month": MONTHS * 2,                      # both firms have all 11 periods
                         "revenue": np.concatenate([make_series(len(MONTHS), 100),   # the F1 series
                                                    make_series(len(MONTHS), 130)])})   # the F2 series
dump(balanced, "l3_panel_balanced.csv")                            # write the balanced panel
wide_balanced = balanced.pivot(index="firm_id", columns="month", values="revenue")   # firm x month matrix
print("balanced panel shape (n x T):", wide_balanced.shape)        # print the shape: n=2, T=11

# ---------------------------------------------------------------- 2) unbalanced panel: a different T_i per firm
spec = {"F1": 9, "F2": 11, "F3": 8}                                # T1=9, T2=11, T3=8, n=3
frames = []                                                        # one data frame per firm
for i, (firm, periods) in enumerate(spec.items()):                 # generate a series of a different length for each firm
    frames.append(pd.DataFrame({"firm_id": firm,                   # firm identifier
                                "month": list(range(1, periods + 1)),   # this firm's own time range
                                "revenue": make_series(periods, 90 + 25 * i)}))   # the value series
unbalanced = pd.concat(frames, ignore_index=True)                  # concatenate into the unbalanced panel
unbalanced.loc[(unbalanced["firm_id"] == "F2") &                   # plant one more block of missingness
               (unbalanced["month"].isin([4, 5])), "revenue"] = np.nan   # F2 has no observations in months 4 and 5
dump(unbalanced, "l3_panel_unbalanced.csv")                        # write the unbalanced panel
wide_unbalanced = unbalanced.pivot(index="firm_id", columns="month", values="revenue")   # the matrix with the gaps
coverage = (unbalanced.assign(observed=unbalanced["revenue"].notna())   # coverage statistics per firm
            .groupby("firm_id")["observed"].agg(periods="count", observed="sum"))   # expected periods / observed periods
coverage["T_i"] = [spec[f] for f in coverage.index]                # the declared period count per firm
coverage["coverage_pct"] = (100 * coverage["observed"] / coverage["periods"]).round(1)   # coverage rate
coverage["balanced"] = coverage["periods"] == coverage["periods"].max()   # whether every firm has the same period count
print("panel coverage table:\n", coverage.to_string())             # print the structural diagnosis
dump(coverage.reset_index(), "l3_panel_coverage.csv")              # write the coverage table

# ---------------------------------------------------------------- 3) fill the gaps and flag them
repaired = unbalanced.sort_values(["firm_id", "month"]).copy()     # sort first so interpolation follows time order
repaired["revenue_interpolated"] = (repaired.groupby("firm_id")["revenue"]   # group: interpolate within each unit
                                    .transform(lambda s: s.interpolate(limit_direction="both")))   # fill in both directions
repaired["filled_flag"] = repaired["revenue"].isna() & repaired["revenue_interpolated"].notna()   # flag the cells that were filled
repaired["revenue"] = repaired["revenue_interpolated"]             # promote the filled values to the main column
dump(repaired, "l3_panel_repaired.csv")                            # write the repaired panel
wide_repaired = repaired.pivot(index="firm_id", columns="month", values="revenue")   # the repaired matrix
still_missing = int(wide_repaired.isna().sum().sum())              # how many cells interpolation still cannot fill
print("cells still missing after interpolation:", still_missing)   # interpolation cannot invent a period out of nothing
print("imputed cells:", int(repaired["filled_flag"].sum()))        # the number of cells actually filled

# ---------------------------------------------------------------- figure A: the balanced panel
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))                # line chart + heat map
for firm in balanced["firm_id"].unique():                          # one line per firm
    sub = balanced[balanced["firm_id"] == firm]                    # fetch this firm's data across the periods
    axes[0].plot(sub["month"], sub["revenue"], "-o", ms=3.5, lw=1.4, label=firm)   # line chart
axes[0].set_title("balanced panel: n = 2, T = 11")                 # subplot title
axes[0].set_xlabel("month"); axes[0].set_ylabel("revenue")         # axis meaning
axes[0].legend(title="firm")                                       # legend
im = axes[1].imshow(wide_balanced.values, cmap="RdPu", aspect="auto")   # firm x month heat map
axes[1].set_xticks(range(len(MONTHS)), MONTHS, fontsize=8)         # x axis: months
axes[1].set_yticks(range(wide_balanced.shape[0]), wide_balanced.index, fontsize=10)   # y axis: firms
axes[1].set_title("heat map: every cell is observed")              # subplot title
axes[1].grid(False)                                                # a heat map does not need a grid
plt.colorbar(im, ax=axes[1], fraction=0.046)                       # value colour bar
plt.tight_layout()                                                 # adjust the two panels
save(fig, "fig_panel_balanced.png")                                # save: figure for the balanced-panel slide

# ---------------------------------------------------------------- figure B: before and after repairing
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                # two heat maps + coverage
for ax, matrix, title in [(axes[0], wide_unbalanced, "before: gaps and shorter T_i"),   # the original structure
                          (axes[1], wide_repaired, "after: interpolated (marked cells)")]:   # after repairing
    ax.imshow(matrix.values, cmap="RdPu", aspect="auto")           # heat map
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, fontsize=8)   # x axis: months
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=10)   # y axis: firms
    ax.set_title(title, fontsize=10.5)                             # subplot title
    ax.grid(False)                                                 # a heat map does not need a grid
for _, row in repaired[repaired["filled_flag"]].iterrows():        # circle every imputed cell
    axes[1].scatter(row["month"] - 1, list(wide_repaired.index).index(row["firm_id"]),   # the cell coordinates
                    marker="o", s=90, facecolors="none", edgecolors="#1E2630", lw=1.2)   # hollow circle
axes[2].bar(coverage.index, coverage["coverage_pct"], color="#8A0C3C", alpha=0.85)   # coverage per firm
axes[2].axhline(100, color="#C9A227", ls="--", lw=1.2, label="full coverage")   # the 100% reference line
axes[2].set_ylim(0, 115)                                           # leave room for the reference line
axes[2].set_title("coverage per firm (%)")                         # subplot title
axes[2].legend(fontsize=8.5)                                       # legend
plt.tight_layout()                                                 # adjust the three panels
save(fig, "fig_panel_heatmap.png")                                 # save: figure for the panel-repair slide
