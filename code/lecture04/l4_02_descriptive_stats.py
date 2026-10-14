"""l4_02_descriptive_stats.py -- central tendency, spread, and distribution shape: compute them, then plot them.

The script first reproduces the original course's "5, 19, 24, 62, 91, 100 -> range 95" example,
then uses a right-skewed income sample to contrast the robustness of mean/median/mode and
variance/MAD/IQR, and finally plots a side-by-side view of skewness and kurtosis (heavy/thin tails).
"""
import numpy as np  # numeric computation and distribution construction
import pandas as pd  # statistics and tables
import matplotlib.pyplot as plt  # descriptive-statistics figures
from scipy import stats  # mode and kernel density (for descriptive statistics only)
from _common_l4 import RNG, save, dump, style_axes, annotated  # reuse the shared and caption helpers

# ---------------------------------------------------------------- numeric example from the original course
DEMO = [5, 19, 24, 62, 91, 100]  # the six observations given in the original course
demo_range = max(DEMO) - min(DEMO)  # range = max - min
print("Original-course example:", DEMO, "-> range =", demo_range, "(should be 95)")  # print and self-check
assert demo_range == 95, "The original-course example should have range 95"  # assert: matches the original course

# ---------------------------------------------------------------- a right-skewed income sample
N = 1000  # sample size
income = RNG.lognormal(mean=2.6, sigma=0.55, size=N)  # monthly income (thousand CNY): typically right-skewed
outliers = np.array([98.0, 132.5, 210.0])  # manually add three high-income outliers
income_all = np.concatenate([income, outliers])  # merge into a sample "with outliers"
df = pd.DataFrame({"income_k": np.round(income_all, 2)})  # assemble into a table
df["is_outlier"] = df["income_k"] > 60  # flag the high-income points (for teaching only)
dump(df, "l4_income_skewed.csv")  # write the sample to disk

def describe_series(s):  # compute descriptive statistics
    """Return central tendency, spread, and shape metrics, covering every statistic required by the slides."""  # function contract
    return {  # compute each item
        "n": int(s.notna().sum()),  # effective sample size
        "mean": float(s.mean()),  # mean
        "median": float(s.median()),  # median
        "mode": float(s.round(1).mode().iloc[0]),  # mode (most common value when binned to 0.1)
        "std": float(s.std(ddof=1)),  # standard deviation
        "var": float(s.var(ddof=1)),  # variance
        "min": float(s.min()),  # minimum
        "max": float(s.max()),  # maximum
        "range": float(s.max() - s.min()),  # range
        "q1": float(s.quantile(0.25)),  # first quartile
        "q3": float(s.quantile(0.75)),  # third quartile
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),  # interquartile range
        "mad": float((s - s.median()).abs().median()),  # median absolute deviation (robust spread)
        "skew": float(s.skew()),  # skewness
        "kurt": float(s.kurt()),  # excess kurtosis
    }  # end of the dictionary returned by describe_series
rows = []  # three comparison groups
rows.append({"sample": "no outliers", **describe_series(df.loc[~df["is_outlier"], "income_k"])})  # clean sample
rows.append({"sample": "with outliers", **describe_series(df["income_k"])})  # full sample
rows.append({"sample": "original-course example", **describe_series(pd.Series(DEMO).astype(float))})  # original-course data
table = pd.DataFrame(rows)  # summarize into a table
table = table.round(3)  # keep three decimal places
print(table.to_string(index=False))  # print to the terminal
dump(table, "l4_descriptive_table.csv")  # write the descriptive-statistics table to disk

# robustness demo: add outliers one at a time and watch the mean and median shift
steps = np.arange(0, 11)  # add 0-10 outliers
base = df.loc[~df["is_outlier"], "income_k"].values  # base sample
means, medians = [], []  # record the two measures of central tendency
for k in steps:  # compute each scenario
    sample = np.concatenate([base, np.full(k, 60.0 + 15.0 * k)])  # each added outlier is higher than the last
    means.append(np.mean(sample))  # mean
    medians.append(np.median(sample))  # median
d_mean = (means[-1] - means[0]) / means[0] * 100  # relative shift of the mean (%)
d_med = (medians[-1] - medians[0]) / medians[0] * 100  # relative shift of the median (%)
print(f"After adding 10 outliers: mean shifts {d_mean:+.1f}%, median shifts {d_med:+.1f}%")  # print the robustness conclusion

# ---------------------------------------------------------------- figure A: central tendency and robustness
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))  # distribution / robustness / metric comparison
axes[0].hist(df["income_k"], bins=40, color="#8A0C3C", alpha=0.8)  # income distribution histogram
axes[0].axvline(df["income_k"].mean(), color="#C9A227", lw=1.8,  # mean line (dragged by outliers)
                label=f"mean {df['income_k'].mean():.1f}")  # legend with the value
axes[0].axvline(df["income_k"].median(), color="#2E6F8E", lw=1.8,  # median line (robust)
                label=f"median {df['income_k'].median():.1f}")  # legend with the value
style_axes(axes[0], "Income distribution: mean vs median", "monthly income (thousand CNY)", "count")  # shared style
axes[0].legend()  # show the legend
axes[1].plot(steps, means, "-o", ms=4, color="#C9A227", label="mean")  # mean vs number of outliers
axes[1].plot(steps, medians, "-s", ms=4, color="#2E6F8E", label="median")  # median barely moves
style_axes(axes[1], "Effect of outliers on central tendency", "number of outliers added", "monthly income (thousand CNY)")  # shared style
axes[1].legend()  # legend
annotated(axes[1], f"mean {d_mean:+.1f}% / median {d_med:+.1f}%", "upper left")  # in-figure annotation
metrics = ["mean", "median", "std", "iqr", "mad"]  # pick five common metrics
clean = table.loc[table["sample"] == "no outliers", metrics].iloc[0]  # values for the clean sample
dirty = table.loc[table["sample"] == "with outliers", metrics].iloc[0]  # values for the sample with outliers
x = np.arange(len(metrics))  # bar-chart x coordinates
axes[2].bar(x - 0.2, clean.values, width=0.4, label="no outliers", color="#2E6F8E")  # left bars
axes[2].bar(x + 0.2, dirty.values, width=0.4, label="with outliers", color="#8A0C3C")  # right bars
axes[2].set_xticks(x, ["mean", "median", "std", "IQR", "MAD"])  # metric names
style_axes(axes[2], "Robustness: which metric do outliers drag", "metric", "value")  # shared style
axes[2].legend()  # legend
plt.tight_layout()  # adjust the three subplots
save(fig, "fig_central_tendency.png")  # save: one-page figure for central tendency

# ---------------------------------------------------------------- figure B: spread and shape
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.7))  # same mean different variance / spread metrics / skewness / kurtosis
grid = np.linspace(-4, 4, 400)  # x coordinates for the density curves
for sd, c in zip([0.5, 1.0, 1.8], ["#2E6F8E", "#C9A227", "#8A0C3C"]):  # three different standard deviations
    axes[0].plot(grid, stats.norm.pdf(grid, 0, sd), lw=1.8, color=c, label=f"σ = {sd}")  # same mean, different spread
style_axes(axes[0], "Same mean, different spread", "value", "probability density")  # shared style
axes[0].legend()  # legend
disp = table.loc[table["sample"] == "with outliers", ["std", "iqr", "mad"]].iloc[0]  # three spread metrics
axes[1].bar(["std", "IQR", "MAD"], disp.values,  # spread-metric comparison
            color=["#8A0C3C", "#C9A227", "#2E6F8E"], alpha=0.9)  # three colors
style_axes(axes[1], "Three spread metrics (with outliers, thousand CNY)", "metric", "value")  # shared style
skew_sets = [("left-skewed", -4.0, "#2E6F8E"), ("symmetric", 0.0, "#66717D"), ("right-skewed", 4.0, "#8A0C3C")]  # three skewness levels
for name, a, c in skew_sets:  # plot three densities with different skewness
    axes[2].plot(grid, stats.skewnorm.pdf(grid, a), lw=1.8, color=c, label=f"{name} (skew {a:+.0f})")  # three density curves with different skewness
style_axes(axes[2], "Skewness: which way the tail points", "value", "probability density")  # shared style
axes[2].legend(fontsize=8.4)  # legend
kurt_sets = [("heavy-tailed leptokurtic", 2.0), ("normal mesokurtic", 0.0), ("thin-tailed platykurtic", -0.9)]  # three kurtosis levels
t = np.linspace(-5, 5, 400)  # x coordinates for the kurtosis comparison
axes[3].plot(t, stats.norm.pdf(t), lw=1.6, color="#66717D", label="standard normal")  # reference curve
axes[3].plot(t, stats.t.pdf(t, 3), lw=1.8, color="#8A0C3C", label="heavy-tailed t(3)")  # heavy tail: extremes more frequent
axes[3].plot(t, stats.uniform.pdf(t, -1.7, 3.4), lw=1.8, color="#2E6F8E", label="thin-tailed uniform")  # thin tail
style_axes(axes[3], "Kurtosis: how heavy the tails are", "value", "probability density")  # shared style
axes[3].legend(fontsize=8.4)  # legend
plt.tight_layout()  # adjust the four subplots
save(fig, "fig_dispersion_shape.png")  # save: one-page figure for spread and shape
