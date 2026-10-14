"""l3_08_imputation_compare.py -- five imputation strategies, scored against a known truth.

Income is first deleted under an MAR mechanism, then restored with five methods: mean, median,
regression, KNN and MICE (iterative imputation). Because the complete data is known, the script can
evaluate each strategy on everything from the distribution and correlations down to the downstream
regression coefficient -- which is the hands-on test of whether imputation made things worse.
"""
import numpy as np                                                 # linear algebra and descriptive statistics
import pandas as pd                                                # tables and missing-data handling
import matplotlib.pyplot as plt                                    # distribution comparison figure
from sklearn.experimental import enable_iterative_imputer          # explicitly enable the experimental MICE imputer first
from sklearn.impute import KNNImputer, IterativeImputer            # the two imputers: KNN and MICE
from sklearn.linear_model import LinearRegression                  # regression imputation and the downstream model
from _common_l3 import RNG, save, dump                             # reuse the shared seed, figure and dump helpers

# ---------------------------------------------------------------- 1) build the complete data first, then damage it
N = 700                                                            # number of surveyed employees
age = RNG.integers(23, 58, N)                                      # age (years)
hours = np.round(RNG.normal(42, 6, N).clip(20, 65), 1)             # weekly working hours
edu = RNG.choice([0, 1, 2], N, p=[0.4, 0.42, 0.18])                # education (coded 0/1/2)
income = (12 + 0.30 * hours + 0.55 * edu + 0.06 * age              # the true data-generating process
          + RNG.normal(0, 2.4, N))                                 # plus individual noise
complete = pd.DataFrame({"age": age, "hours": hours, "edu": edu,   # covariates
                         "income": np.round(income, 2)})           # the income that will be made missing
dump(complete, "l3_income_complete.csv")                           # write the complete reference table
truth_mean, truth_sd = complete["income"].mean(), complete["income"].std()   # reference mean and standard deviation
truth_corr = complete["income"].corr(complete["hours"])            # reference correlation
coef_truth = LinearRegression().fit(complete[["hours"]], complete["income"]).coef_[0]   # reference regression coefficient
print(f"truth: mean={truth_mean:.2f} sd={truth_sd:.2f} corr(hours)={truth_corr:.3f} slope={coef_truth:.3f}")   # print the benchmark

# ---------------------------------------------------------------- 2) delete income under an MAR mechanism
p_missing = np.where(edu == 2, 0.55, np.where(edu == 1, 0.30, 0.10))   # the higher the education, the less they disclose
mask = RNG.random(N) < p_missing                                   # missingness mask
damaged = complete.copy()                                          # work on a copy
damaged.loc[mask, "income"] = np.nan                               # income becomes missing
dump(damaged, "l3_income_missing.csv")                             # write the damaged table
print("missing rate in income:", round(100 * damaged['income'].isna().mean(), 1), "%")   # print the missing rate

# ---------------------------------------------------------------- 3) five ways to fill the gaps
filled = {"mean": damaged["income"].fillna(damaged["income"].mean()),   # global mean imputation
          "median": damaged["income"].fillna(damaged["income"].median()),   # global median imputation
          "drop": None}                                            # complete-case analysis (no imputation)
features = ["hours", "edu", "age"]                                 # observed variables used for prediction
known = damaged.dropna(subset=["income"])                          # rows used to fit the model
reg = LinearRegression().fit(known[features], known["income"])     # the regression imputation model
reg_pred = reg.predict(damaged.loc[damaged["income"].isna(), features])   # predict for the missing rows
reg_series = damaged["income"].copy()                              # copy the column so it can be filled
reg_series.loc[damaged["income"].isna()] = reg_pred                # fill with the predictions
filled["regression"] = reg_series                                  # register the strategy
knn = KNNImputer(n_neighbors=5).fit_transform(damaged[features + ["income"]])   # KNN imputation
filled["knn"] = pd.Series(knn[:, -1], index=damaged.index)         # take the income column
mice = IterativeImputer(max_iter=10, random_state=42).fit_transform(damaged[features + ["income"]])   # MICE
filled["mice"] = pd.Series(mice[:, -1], index=damaged.index)       # the iterative imputation result
complete_case = damaged.dropna(subset=["income"])                  # the complete-case-only option

# ---------------------------------------------------------------- 4) score each strategy
rows = []                                                          # one row per strategy
for name, series in filled.items():                                # evaluate one by one
    data = complete_case if series is None else damaged.assign(income=series)   # pick the table to analyse
    mean = data["income"].mean()                                   # mean estimate
    sd = data["income"].std()                                      # spread estimate
    corr = data["income"].corr(data["hours"])                      # correlation with working hours
    slope = LinearRegression().fit(data[["hours"]], data["income"]).coef_[0]   # downstream regression coefficient
    rows.append({"strategy": name, "n_used": int(data["income"].notna().sum()),   # rows actually used in the analysis
                 "mean": round(mean, 2), "mean_bias": round(mean - truth_mean, 2),   # bias of the mean
                 "sd": round(sd, 2), "sd_bias": round(sd - truth_sd, 2),   # bias of the standard deviation
                 "corr_hours": round(corr, 3), "corr_error": round(corr - truth_corr, 3),   # correlation error
                 "slope": round(slope, 3), "slope_error": round(slope - coef_truth, 3)})   # regression-coefficient error
scores = pd.DataFrame(rows)                                        # summary comparison table
print(scores.to_string(index=False))                               # print to the terminal
dump(scores, "l3_imputation_metrics.csv")                          # write the comparison table
strategy_means = pd.DataFrame({"strategy": ["complete_data"] + list(filled),   # truth + each strategy
                               "mean_income": [truth_mean] + [mean for mean in scores["mean"]]})   # the mean series
dump(strategy_means, "l3_imputation_means.csv")                    # write the small table used for the figure

# ---------------------------------------------------------------- figure: distribution and error
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                # density / boxplot / error
bins = np.linspace(complete["income"].min(), complete["income"].max(), 30)   # shared bins for comparison
axes[0].hist(complete["income"], bins=bins, density=True, color="#1E2630",   # the truth as the reference
             alpha=0.35, label="complete (truth)")                 # legend: the complete data
for name, color in zip(["mean", "regression", "mice"], ["#66717D", "#C9A227", "#8A0C3C"]):
    axes[0].hist(filled[name], bins=bins, density=True, histtype="step",   # distribution of the three fill methods
                 lw=1.6, color=color, label=name)                  # one curve each
axes[0].set_title("income distribution after imputation")          # subplot title
axes[0].set_xlabel("income (10k yuan)")                            # meaning of the x axis
axes[0].legend(fontsize=8.5)                                       # compact legend
box_data = [complete["income"]] + [filled[k] for k in ["mean", "median", "regression", "knn", "mice"]]   # the series to compare
axes[1].boxplot(box_data, patch_artist=True,                       # spread comparison across the strategies
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))    # shade the box
axes[1].set_xticklabels(["truth", "mean", "median", "regr", "knn", "mice"], fontsize=8.5)   # labels
axes[1].set_title("spread is the first casualty")                  # subplot title
x = np.arange(len(scores))                                         # x coordinates for the bar chart
axes[2].bar(x - 0.2, scores["mean_bias"], width=0.4, label="mean bias", color="#8A0C3C")   # bias in the mean
axes[2].bar(x + 0.2, scores["sd_bias"], width=0.4, label="spread bias", color="#C9A227")   # bias in the spread
axes[2].axhline(0, color="#1E2630", lw=1.0)                        # the zero line means no error
axes[2].set_xticks(x, scores["strategy"], rotation=20, fontsize=8.5)   # strategy names
axes[2].set_title("bias vs the complete-data truth")               # subplot title
axes[2].set_ylabel("income units")                                 # both bars share the same unit
axes[2].legend(fontsize=8.5)                                       # compact legend
plt.tight_layout()                                                 # adjust the three panels
save(fig, "fig_imputation_compare.png")                            # save: figure for the imputation slide
