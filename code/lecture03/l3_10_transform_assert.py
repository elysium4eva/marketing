"""l3_10_transform_assert.py -- several transforms that tame a right-skewed distribution, plus a schema assertion report.

Income-like variables are usually right-skewed and a few extreme values dominate models and tests.
The script compares the usual transforms, quantifies their effect on skewness and kurtosis, and
finishes with a lightweight (pandera-style) schema check that gates publication -- which is what a
production pipeline would do.
"""
import numpy as np                                                 # logs and descriptive statistics
import pandas as pd                                                # tables and quantile binning
import matplotlib.pyplot as plt                                    # distribution comparison figure
from _common_l3 import RNG, save, dump                             # reuse the shared seed, figure and dump helpers

# ---------------------------------------------------------------- 1) a right-skewed variable
N = 900                                                            # number of firms
revenue = RNG.lognormal(mean=4.6, sigma=0.9, size=N)               # strongly right-skewed revenue
region = RNG.choice(["East", "Central", "West"], N, p=[0.5, 0.3, 0.2])   # a categorical column
raw = pd.DataFrame({"firm_id": [f"F{i:04d}" for i in range(N)],    # firm key
                    "region": region,                              # region (categorical)
                    "revenue": np.round(revenue, 2)})              # revenue (right-skewed numeric)
dump(raw, "l3_income_skewed.csv")                                  # write the raw sample

# ---------------------------------------------------------------- 2) five transforms
transformed = raw.copy()                                           # transform on a copy so the raw table is kept
transformed["log1p"] = np.log1p(transformed["revenue"])            # log(1+x): handles zeros and compresses the right tail
transformed["sqrt"] = np.sqrt(transformed["revenue"])              # square root: gentler than the log
transformed["zscore"] = ((transformed["revenue"] - transformed["revenue"].mean())   # centre first
                         / transformed["revenue"].std(ddof=0))     # then scale: mean 0, standard deviation 1
transformed["minmax"] = ((transformed["revenue"] - transformed["revenue"].min())   # shift so it starts at 0
                         / (transformed["revenue"].max() - transformed["revenue"].min()))   # scale into [0,1]
transformed["quartile"] = pd.qcut(transformed["revenue"], 4, labels=["Q1", "Q2", "Q3", "Q4"])   # equal-frequency bins
transformed["decile"] = pd.qcut(transformed["revenue"], 10, labels=False)   # deciles (integer codes)
summary = pd.DataFrame([                                           # shape comparison before and after the transforms
    {"variable": name, "mean": round(transformed[name].mean(), 3),   # location
     "sd": round(transformed[name].std(), 3),                      # spread
     "skew": round(transformed[name].skew(), 3),                   # skewness: 0 means symmetric
     "kurtosis": round(transformed[name].kurt(), 3)}               # excess kurtosis: 0 is close to normal
    for name in ["revenue", "log1p", "sqrt", "zscore", "minmax"]])   # five comparable columns
print(summary.to_string(index=False))                              # print the effect of each transform
dump(summary, "l3_transform_summary.csv")                          # write the comparison table
print("quartile counts:", transformed["quartile"].value_counts().sort_index().to_dict())   # counts per equal-frequency bin
print("decile means (revenue):\n",                                 # deciles are commonly used for business reporting
      transformed.groupby("decile")["revenue"].mean().round(1).to_string())   # mean revenue per decile

# ---------------------------------------------------------------- 3) schema assertions before publishing
RULES = {                                                          # a minimal schema contract
    "firm_id": {"dtype": "object", "unique": True, "regex": r"^F\d{4}$"},   # the key: unique and formatted
    "region": {"dtype": "object", "allowed": ["East", "Central", "West"]},   # categorical: closed value domain
    "revenue": {"dtype": "float64", "min": 0.0, "max": 1_000_000.0, "nullable": False},   # numeric range
    "log1p": {"dtype": "float64", "min": 0.0, "nullable": False},   # the derived column is non-negative
}                                                                  # end of the contract definition
def check(table, rules):                                           # run the contract rule by rule and collect the violations
    """Return a list of violations instead of stopping at the first problem."""   # function contract
    report = []                                                    # one row per rule
    for column, spec in rules.items():                             # walk the declared columns
        present = column in table.columns                          # a missing column is itself a violation
        report.append({"column": column, "rule": "present",        # record the presence check
                       "ok": present, "detail": "" if present else "missing column"})   # state the reason when it fails
        if not present:                                            # skip the remaining rules if the column is absent
            continue                                               # move on to the next column
        series = table[column]                                     # the column to check
        if spec.get("unique"):                                     # uniqueness rule
            dup = int(series.duplicated().sum())                   # number of duplicates
            report.append({"column": column, "rule": "unique", "ok": dup == 0,   # the uniqueness result
                           "detail": f"{dup} duplicates"})         # state the duplicate count
        if spec.get("regex"):                                      # identifier-format rule
            bad = int((~series.astype(str).str.match(spec["regex"])).sum())   # values that break the format
            report.append({"column": column, "rule": "regex", "ok": bad == 0,   # the format result
                           "detail": f"{bad} non-conforming"})     # state the non-conforming count
        if spec.get("allowed"):                                    # value-domain rule
            extra = sorted(set(series.dropna()) - set(spec["allowed"]))   # unexpected categories that appeared
            report.append({"column": column, "rule": "allowed", "ok": not extra,   # the value-domain result
                           "detail": f"unexpected: {extra}"})      # state the out-of-domain values
        if spec.get("min") is not None:                            # lower-bound rule
            below = int((series < spec["min"]).sum())              # values below the lower bound
            report.append({"column": column, "rule": "min", "ok": below == 0,   # the lower-bound result
                           "detail": f"{below} below {spec['min']}"})   # state the count out of bounds
        if spec.get("max") is not None:                            # upper-bound rule
            above = int((series > spec["max"]).sum())              # values above the upper bound
            report.append({"column": column, "rule": "max", "ok": above == 0,   # the upper-bound result
                           "detail": f"{above} above {spec['max']}"})   # state the count out of bounds
        if spec.get("nullable") is False:                          # not-null rule
            miss = int(series.isna().sum())                        # number of missing values
            report.append({"column": column, "rule": "not-null", "ok": miss == 0,   # the not-null result
                           "detail": f"{miss} missing"})           # state the missing count
    return pd.DataFrame(report)                                    # return the detail table for the slides
report = check(transformed, RULES)                                 # run the checks on the transformed table
print("schema checks passed:", int(report["ok"].sum()), "of", len(report))   # print how many checks passed
print(report[~report["ok"]].to_string(index=False))                # print only the failing entries
dump(report, "l3_schema_report.csv")                               # write the validation report
assert report["ok"].all(), "schema contract failed: fix the table before publishing it"   # stop the pipeline when something fails

# ---------------------------------------------------------------- figure: before and after the transforms
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                # raw / transformed / shape metrics
axes[0].hist(transformed["revenue"], bins=40, color="#8A0C3C", alpha=0.85)   # raw distribution: strongly right-skewed
axes[0].set_title(f"raw revenue (skew = {transformed['revenue'].skew():.2f})")   # put the skewness in the title
axes[0].set_xlabel("revenue")                                      # meaning of the x axis
axes[1].hist(transformed["log1p"], bins=40, color="#C9A227", alpha=0.9)   # after log1p
axes[1].set_title(f"log1p (skew = {transformed['log1p'].skew():.2f})")   # put the skewness in the title
axes[1].set_xlabel("log(1 + revenue)")                             # meaning of the x axis
metrics = summary.set_index("variable")[["skew", "kurtosis"]]      # shape metrics per transform
x = np.arange(len(metrics))                                        # x coordinates for the bar chart
axes[2].bar(x - 0.2, metrics["skew"], width=0.4, label="skew", color="#8A0C3C")   # skewness comparison
axes[2].bar(x + 0.2, metrics["kurtosis"], width=0.4, label="kurtosis", color="#66717D")   # kurtosis comparison
axes[2].axhline(0, color="#1E2630", lw=1.0)                        # zero line: symmetric and close to normal
axes[2].set_xticks(x, metrics.index, rotation=20, fontsize=8.5)    # transform names
axes[2].set_title("shape metrics after transformation")            # subplot title
axes[2].legend(fontsize=8.5)                                       # legend
plt.tight_layout()                                                 # adjust the three panels
save(fig, "fig_transform_scale.png")                               # save: figure for the transform slide
