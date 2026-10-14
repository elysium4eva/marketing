"""l3_06_outliers_iqr_z_iforest.py -- three ways to detect outliers and three ways to treat them.

The sample mixes a log-normal "normal salary" with three kinds of anomaly: two data-entry errors,
one real but extreme value and a small unusual group. The script compares the IQR rule, the Z-score
rule and Isolation Forest, then shows how dropping, winsorising and keeping affect the mean, the
median and the standard deviation.
"""
import numpy as np                                                  # random sampling and quantiles
import pandas as pd                                                 # tables and descriptive statistics
import matplotlib.pyplot as plt                                     # flag comparison figure
from sklearn.ensemble import IsolationForest                        # multivariate anomaly detection
from _common_l3 import RNG, save, dump                              # reuse the shared seed, figure and dump helpers

# ---------------------------------------------------------------- 1) a sample with a known composition
n_core, n_err, n_extreme, n_group = 400, 2, 1, 8                    # size of the four sub-samples
core = RNG.lognormal(mean=10.4, sigma=0.45, size=n_core)            # normal salaries (yuan)
errors = np.array([-3200.0, 999999.0])                              # two data-entry errors
extreme = np.array([480000.0])                                      # one real but extreme salary
group = RNG.normal(loc=250000, scale=4000, size=n_group)            # a small group of unusual records
values = np.concatenate([core, errors, extreme, group])             # concatenate into a single salary column
kind = (["core"] * n_core + ["entry_error"] * n_err                 # record the true data-generating process
        + ["legitimate_extreme"] * n_extreme + ["unusual_group"] * n_group)   # for teaching comparison only
df = pd.DataFrame({"salary": np.round(values, 2), "true_kind": kind})   # the table used for the analysis
dump(df, "l3_income_outliers.csv")                                  # write the sample

# ---------------------------------------------------------------- 2) three detection rules
q1, q3 = df["salary"].quantile([.25, .75])                          # quartiles of the observed column
iqr = q3 - q1                                                       # interquartile range
df["iqr_outlier"] = (df["salary"] < q1 - 1.5 * iqr) | (df["salary"] > q3 + 1.5 * iqr)   # boxplot rule
z = (df["salary"] - df["salary"].mean()) / df["salary"].std(ddof=0)   # classic Z score
df["z_outlier"] = z.abs() > 3                                       # |z| > 3 rule
mad = (df["salary"] - df["salary"].median()).abs().median()         # median absolute deviation
robust_z = 0.6745 * (df["salary"] - df["salary"].median()) / mad    # robust Z (based on MAD)
df["robust_z_outlier"] = robust_z.abs() > 3.5                       # threshold for the robust version
iso = IsolationForest(contamination=0.03, random_state=42)          # expect roughly 3% to be anomalous
df["iforest_outlier"] = iso.fit_predict(df[["salary"]]) == -1       # -1 means flagged as an outlier
flags = ["iqr_outlier", "z_outlier", "robust_z_outlier", "iforest_outlier"]   # the four rule columns
print("flagged counts:", {c: int(df[c].sum()) for c in flags})      # how many rows each rule flags
print("rules agree on all flags:", int((df[flags].sum(axis=1) == len(flags)).sum()))   # rows flagged by all four rules
print("extreme value flagged by IQR/Z:", bool(df.loc[df["true_kind"] == "legitimate_extreme", "iqr_outlier"].iloc[0]))   # the genuine extreme value
print("entry errors flagged by IQR:", int(df.loc[df["true_kind"] == "entry_error", "iqr_outlier"].sum()))   # the data-entry errors

# ---------------------------------------------------------------- 3) three treatments
keep = df["salary"]                                                 # option A: keep everything
drop = df.loc[~df["iqr_outlier"], "salary"]                         # option B: drop the IQR outliers
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr                       # upper and lower winsorising bounds
wins = df["salary"].clip(lower=lower, upper=upper)                  # option C: winsorise to the bounds
summary = pd.DataFrame({"treatment": ["keep", "drop (IQR)", "winsorise (IQR)"],   # the three treatments
                        "n": [len(keep), len(drop), len(wins)],     # sample size retained
                        "mean": [keep.mean(), drop.mean(), wins.mean()],   # mean
                        "median": [keep.median(), drop.median(), wins.median()],   # median (robust)
                        "std": [keep.std(), drop.std(), wins.std()]}).round(1)   # standard deviation
print(summary.to_string(index=False))                               # print the trade-off table in the terminal
dump(summary, "l3_outlier_treatment.csv")                           # write the comparison table

# ---------------------------------------------------------------- figure: flags and treatments
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # scatter / boxplot / rule counts
axes[0].scatter(np.arange(len(df)), df["salary"], s=10,             # scatter of salary against the row index
                c=np.where(df["iqr_outlier"], "#8A0C3C", "#B9BFC7"))   # red marks the rows flagged by IQR
axes[0].axhline(upper, color="#C9A227", lw=1.2, ls="--", label="IQR fence")   # upper bound
axes[0].axhline(lower, color="#C9A227", lw=1.2, ls="--")            # lower bound
axes[0].set_yscale("symlog")                                        # symlog makes both the errors and the extremes visible
axes[0].set_title("flagged points (IQR rule)")                      # subplot title
axes[0].set_xlabel("row index"); axes[0].set_ylabel("salary (yuan)")   # axis meaning
axes[0].legend()                                                    # legend: the boundary lines
axes[1].boxplot([keep, drop, wins], patch_artist=True,              # distribution comparison across the treatments
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))     # shade the box for readability
axes[1].set_xticklabels(["keep", "drop", "winsorise"])              # x tick labels
axes[1].set_title("distribution after treatment")                   # subplot title
axes[2].bar([c.replace("_outlier", "") for c in flags],             # rule names
            [int(df[c].sum()) for c in flags], color="#66717D", alpha=0.9)   # rows flagged per rule
axes[2].set_title("rows flagged per rule")                          # subplot title
axes[2].tick_params(axis="x", rotation=20)                          # rotate the labels
for i, c in enumerate(flags):                                       # annotate the value on top of each bar
    axes[2].text(i, int(df[c].sum()), str(int(df[c].sum())), ha="center", va="bottom", fontsize=9)   # value label
plt.tight_layout()                                                  # adjust the three panels
save(fig, "fig_outliers.png")                                       # save: figure for the outliers slide
