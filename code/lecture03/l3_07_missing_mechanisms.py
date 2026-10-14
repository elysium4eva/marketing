"""l3_07_missing_mechanisms.py -- generate MCAR / MAR / MNAR missingness and quantify the bias it creates.

The same complete population is damaged in three different ways. Because the truth is known, the
script can show directly how each mechanism distorts a simple estimate (income per capita): MCAR is
essentially unbiased, MAR can be corrected with the right model, and MNAR cannot be fixed without
making an assumption.
"""
import numpy as np                                                 # random sampling and bias computation
import pandas as pd                                                # tables and missing-data tools
import matplotlib.pyplot as plt                                    # missingness matrix figure
from _common_l3 import RNG, save, dump                             # reuse the shared seed, figure and dump helpers

# ---------------------------------------------------------------- 1) a complete population
N = 800                                                            # number of surveyed households
education = RNG.choice(["high_school", "bachelor", "master"], N,   # education drives income
                       p=[0.45, 0.40, 0.15])                       # a realistic education mix
age = RNG.integers(22, 60, N)                                      # working age (years)
base = {"high_school": 8.6, "bachelor": 9.6, "master": 10.2}       # log-income intercept per education level
mu = np.array([base[e] for e in education]) + 0.012 * (age - 22)   # income rises with age
income = RNG.lognormal(mean=mu, sigma=0.30)                        # complete and known monthly income (yuan)
complete = pd.DataFrame({"household_id": [f"H{i:04d}" for i in range(N)],   # household key
                         "education": education, "age": age,       # fully observed covariates
                         "income": np.round(income, 2)})           # the variable that will be made missing
dump(complete, "l3_income_complete_l3_07.csv")                     # write the complete sample
true_mean = complete["income"].mean()                              # the actual quantity to estimate: mean income
print(f"true mean income (complete data): {true_mean:.2f} yuan/month")   # the benchmark for comparison

# ---------------------------------------------------------------- 2) the three missingness mechanisms
mcar_mask = RNG.random(N) < 0.30                                   # MCAR: every row has the same 30% chance
mar_p = np.where(education == "master", 0.55,                      # MAR: missingness depends on education,
                 np.where(education == "bachelor", 0.30, 0.12))    # and education is a fully observed variable
mar_mask = RNG.random(N) < mar_p                                   # draw the MAR missingness by probability
mnar_p = np.clip((income - income.min()) / (income.max() - income.min()), 0, 1) ** 2   # the richer you are, the less you report
mnar_mask = RNG.random(N) < 0.15 + 0.55 * mnar_p                   # MNAR: higher incomes are more likely to be missing
mechanisms = {"MCAR": mcar_mask, "MAR": mar_mask, "MNAR": mnar_mask}   # the three damaged versions
rows, panels, matrices = [], {}, {}                                # containers for the report, the samples and the figures
for name, mask in mechanisms.items():                              # build one table per mechanism
    damaged = complete.copy()                                      # the complete rows stay untouched
    damaged.loc[mask, "income"] = np.nan                           # blank out income only
    naive_mean = damaged["income"].mean()                          # estimate the mean from the observed rows
    rows.append({"mechanism": name, "missing_rate_pct": round(100 * mask.mean(), 1),   # missing share
                 "observed_mean": round(naive_mean, 2),            # naive estimate
                 "bias_vs_truth": round(naive_mean - true_mean, 2),   # absolute gap versus the truth
                 "bias_pct": round(100 * (naive_mean - true_mean) / true_mean, 1)})   # relative bias
    panels[name] = damaged                                         # kept for CSV export
    matrices[name] = damaged[["income", "age", "education"]].isna().astype(int)   # 0/1 matrix for the figure
bias = pd.DataFrame(rows)                                          # one row per mechanism
print(bias.to_string(index=False))                                 # print the bias table in the terminal
dump(bias, "l3_missing_bias.csv")                                  # write the bias table
for name, damaged in panels.items():                               # save each of the three damaged versions
    dump(damaged, f"l3_income_missing_{name}.csv")                 # for example l3_income_missing_MNAR.csv

# ---------------------------------------------------------------- 3) a panel sample with block missingness
n_ids, n_periods = 6, 12                                           # a small firm panel
grid = pd.DataFrame([(f"F{i:02d}", t) for i in range(1, n_ids + 1)   # the complete (firm, month) grid
                     for t in range(1, n_periods + 1)],            # twelve months for every firm
                    columns=["firm_id", "month"])                  # column names
grid["revenue"] = np.round(RNG.lognormal(6.0, 0.35, len(grid)), 1)   # one revenue value per firm-month
drop_firm = grid["firm_id"].isin(["F03", "F05"]) & grid["month"].isin([3, 4, 5])   # a whole block disappears
drop_tail = grid["firm_id"].eq("F06") & grid["month"].gt(9)        # one firm exits the sample early
grid.loc[drop_firm | drop_tail, "revenue"] = np.nan                # the panel is now unbalanced
dump(grid, "l3_missing_panel_sample.csv")                          # write the panel with block missingness (distinct from the l3_09 panel)
coverage = (grid.assign(observed=grid["revenue"].notna())          # coverage per firm
            .groupby("firm_id")["observed"].agg(["sum", "count"]))   # observed months / total months
coverage["coverage_pct"] = (100 * coverage["sum"] / coverage["count"]).round(1)   # coverage (%)
print("panel coverage:\n", coverage.to_string())                   # print the coverage table in the terminal
dump(coverage.reset_index(), "l3_missing_panel_coverage.csv")      # write the coverage table (avoid clashing with l3_09)

# ---------------------------------------------------------------- figure: missingness matrix and bias
fig = plt.figure(figsize=(11.0, 4.2))                              # custom layout: two matrices + one bar chart
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.5])        # three blocks: left, middle and right
for i, name in enumerate(["MCAR", "MAR"]):                         # draw the first two mechanisms
    ax = fig.add_subplot(gs[0, i])                                 # one axes per mechanism
    ax.imshow(matrices[name].T.values, cmap="Greys", aspect="auto", vmin=0, vmax=1)   # 1 = missing (dark)
    ax.set_yticks(range(3), ["income", "age", "education"], fontsize=9)   # y axis: variables
    ax.set_xlabel("households")                                    # x axis: households
    ax.set_title(f"{name}: missing pattern "                       # include the missing share in the title
                 f"({100 * mechanisms[name].mean():.0f}% of income missing)", fontsize=10.5)   # the share value
    ax.grid(False)                                                 # the matrix does not need a grid
ax = fig.add_subplot(gs[0, 2])                                     # right panel: bias of the naive mean
colors = ["#66717D", "#C9A227", "#8A0C3C"]                         # grey / gold / red for the three mechanisms
ax.bar(bias["mechanism"], bias["bias_pct"], color=colors, alpha=0.9)   # relative-bias bar chart
ax.axhline(0, color="#1E2630", lw=1.0)                             # the zero line means unbiased
ax.set_title("bias of the naive mean income (%)", fontsize=10.5)   # subplot title
for i, v in enumerate(bias["bias_pct"]):                           # annotate the value on each bar
    ax.text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)   # value label
plt.tight_layout()                                                 # adjust the layout
save(fig, "fig_missing_matrix.png")                                # save: figure for the missing-data slide
