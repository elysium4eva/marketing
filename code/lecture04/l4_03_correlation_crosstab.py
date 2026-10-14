"""l4_03_correlation_crosstab.py -- correlation and contingency tables: from statistics to graphics.

The script uses "trade gravity"-style synthetic data to reproduce the original course's conclusion:
when the range of X is truncated (for example, keeping only a short stretch with small distances),
both Pearson and Spearman correlations weaken noticeably.
It also uses the survey sample to demonstrate contingency tables (crosstabs) and conditional means.
"""
import numpy as np  # numeric computation
import pandas as pd  # tables and crosstabs
import matplotlib.pyplot as plt  # scatter and comparison figures
from scipy.stats import chi2_contingency  # chi-square test for contingency tables
from _common_l4 import RNG, DAT, save, dump, style_axes, annotated  # reuse the shared and caption helpers

# ---------------------------------------------------------------- 1) synthesize "trade pair" data (gravity-model style)
N = 220  # number of country pairs
gdp_x = RNG.lognormal(mean=26.0, sigma=1.0, size=N)  # exporter GDP (USD, right-skewed)
gdp_m = RNG.lognormal(mean=26.3, sigma=1.0, size=N)  # importer GDP
dist = RNG.lognormal(mean=8.6, sigma=0.7, size=N)  # bilateral distance (km)
trade = (0.9 * np.log(gdp_x) + 0.85 * np.log(gdp_m)  # gravity model: trade rises with the size of both sides
         - 1.15 * np.log(dist) + RNG.normal(0, 0.55, N))  # falls with distance + random noise
trade_value = np.exp(trade + 6.0)  # convert back to USD scale
pairs = pd.DataFrame({"exporter": [f"E{i:03d}" for i in range(1, N + 1)],  # exporter id
                      "importer": [f"I{i:03d}" for i in range(1, N + 1)],  # importer id
                      "distance_km": np.round(dist, 0),  # bilateral distance
                      "gdp_exporter": np.round(gdp_x, 0),  # exporter GDP
                      "gdp_importer": np.round(gdp_m, 0),  # importer GDP
                      "trade_usd": np.round(trade_value, 0)})  # trade value
dump(pairs, "l4_trade_pairs.csv")  # write the sample to disk

log_trade = np.log(pairs["trade_usd"])  # take logs: symmetrize a right-skewed variable
log_dist = np.log(pairs["distance_km"])  # take logs of distance as well
r_full = np.corrcoef(log_dist, log_trade)[0, 1]  # Pearson correlation (full sample)
rho_full = pd.Series(log_dist).corr(pd.Series(log_trade), method="spearman")  # Spearman (full sample)
# restricted range: keep only the 25% with the smallest distance, mimicking "study only nearby trade partners"
cut = np.quantile(log_dist, 0.25)  # 25% quantile as the cutoff point
mask = log_dist <= cut  # truncation mask
r_restricted = np.corrcoef(log_dist[mask], log_trade[mask])[0, 1]  # Pearson after restriction
rho_restricted = pd.Series(log_dist[mask]).corr(pd.Series(log_trade[mask]), method="spearman")  # Spearman after restriction
print(f"full sample Pearson={r_full:.3f} Spearman={rho_full:.3f}")  # print the full-sample correlations
print(f"restricted range Pearson={r_restricted:.3f} Spearman={rho_restricted:.3f}")  # print the correlations after restriction

# ---------------------------------------------------------------- 2) correlation matrix
analyze = pairs[["gdp_exporter", "gdp_importer", "distance_km", "trade_usd"]].copy()  # select the analysis columns
analyze["log_trade"] = log_trade  # add log trade value
analyze["log_distance"] = log_dist  # add log distance
corr_pearson = analyze.corr(method="pearson").round(3)  # Pearson correlation matrix
corr_spearman = analyze.corr(method="spearman").round(3)  # Spearman correlation matrix
dump(corr_pearson.reset_index(names="variable"), "l4_correlation_pearson.csv")  # write the Pearson matrix to disk
dump(corr_spearman.reset_index(names="variable"), "l4_correlation_spearman.csv")  # write the Spearman matrix to disk
print("Pearson matrix (first 3 rows):\n", corr_pearson.head(3).to_string())  # print a snippet to the terminal

# ---------------------------------------------------------------- 3) contingency table and conditional means (using the L4 survey sample)
survey = pd.read_csv(DAT / "l4_survey_raw.csv")  # read the survey sample produced by l4_01
survey["income_band"] = pd.cut(survey["income_k"],  # bin income into three tiers
                               bins=[0, 8, 16, 100],  # bin edges (thousand CNY)
                               labels=["low income", "middle income", "high income"])  # tier names
ct = pd.crosstab(survey["income_band"], survey["channel"],  # crosstab: income tier x channel
                 margins=True, margins_name="Total")  # with marginal totals
ct_pct = pd.crosstab(survey["income_band"], survey["channel"], normalize="index").round(3)  # row percentages
cond = survey.groupby("income_band", observed=True)["satisfaction"].agg(["mean", "median", "count"]).round(2)  # conditional stats by income tier (mean/median/n)
print("income tier x channel (row %):\n", ct_pct.to_string())  # print the row percentages
dump(ct.reset_index(), "l4_crosstab_income_channel.csv")  # write the crosstab to disk
dump(cond.reset_index(), "l4_conditional_satisfaction.csv")  # write the conditional means to disk
# chi-square and Cramér's V: an intuitive measure of contingency-table strength
core = ct.iloc[:-1, :-1]  # core table after dropping the marginal totals
chi2, p, dof, expected = chi2_contingency(core)  # chi-square test: are they independent
n_obs = core.values.sum()  # sample size
cramers_v = float(np.sqrt(chi2 / (n_obs * (min(core.shape) - 1))))  # Cramér's V: association strength from 0 to 1
print(f"chi2={chi2:.1f}  p={p:.3f}  Cramér's V={cramers_v:.3f}")  # print the association strength

# ---------------------------------------------------------------- figure: effect of restricted range on correlation
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))  # full sample / restricted range / coefficient comparison
axes[0].scatter(log_dist, log_trade, s=14, color="#8A0C3C", alpha=0.55)  # full-sample scatter
axes[0].axvline(cut, color="#66717D", ls="--", lw=1.2, label="25% quantile cutoff line")  # mark the cutoff position
z = np.polyfit(log_dist, log_trade, 1)  # full-sample regression line
xs = np.linspace(log_dist.min(), log_dist.max(), 50)  # line x coordinates
axes[0].plot(xs, np.polyval(z, xs), color="#2E6F8E", lw=1.8)  # draw the regression line
style_axes(axes[0], "Full sample: distance vs trade value", "ln(distance)", "ln(trade value)")  # shared style
annotated(axes[0], f"Pearson {r_full:.2f}\nSpearman {rho_full:.2f}")  # in-figure annotation of the correlations
axes[0].legend(fontsize=8.4)  # legend
axes[1].scatter(log_dist[mask], log_trade[mask], s=16, color="#2E6F8E", alpha=0.7)  # restricted-range scatter
z2 = np.polyfit(log_dist[mask], log_trade[mask], 1)  # restricted-sample regression line
xs2 = np.linspace(log_dist[mask].min(), log_dist[mask].max(), 50)  # line x coordinates
axes[1].plot(xs2, np.polyval(z2, xs2), color="#8A0C3C", lw=1.8)  # draw the regression line
style_axes(axes[1], "Restricted range: only the closest 25% of partners", "ln(distance)", "ln(trade value)")  # shared style
annotated(axes[1], f"Pearson {r_restricted:.2f}\nSpearman {rho_restricted:.2f}")  # in-figure annotation
bars = ["Pearson full", "Pearson restricted", "Spearman full", "Spearman restricted"]  # four comparison bars
vals = [r_full, r_restricted, rho_full, rho_restricted]  # corresponding values (all negative in this example)
vals_abs = [abs(v) for v in vals]  # take absolute values for the bars (the y-axis is "correlation strength")
axes[2].bar(bars, vals_abs, color=["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"], alpha=0.9)  # bar comparison
axes[2].set_ylim(0, max(vals_abs) * 1.3)  # y-axis starts at 0, leaving room for labels
style_axes(axes[2], "Correlation strength shrinks after restricting the range", "", "absolute correlation |r|")  # shared style
for i, v in enumerate(vals_abs):  # annotate values on top of the bars
    axes[2].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5)  # value label
axes[2].tick_params(axis="x", labelsize=8.0)  # shrink the axis-label font to avoid clashing with the margins
plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.17, wspace=0.30)  # set margins by hand (more controllable than tight_layout)
save(fig, "fig_restricted_range.png")  # save: one-page figure for correlation and truncation
