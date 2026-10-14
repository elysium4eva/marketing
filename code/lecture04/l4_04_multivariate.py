"""l4_04_multivariate.py -- multivariate: correlation heatmaps, grouped scatter, small multiples, and conditional distributions.

A firm panel (60 firms x 8 years) answers "what should we look at with multiple variables":
first pairwise correlations (heatmap), then whether the relationship changes across groups
(grouped scatter and conditional distributions), and finally small multiples to lay trends
out by industry -- cramming ten colors into one chart is the most common mistake.
"""
import numpy as np  # numeric computation
import pandas as pd  # panel data handling
import matplotlib.pyplot as plt  # four-panel figure
from _common_l4 import RNG, save, dump  # reuse the shared helpers

N_FIRM, N_YEAR = 60, 8  # 60 firms, 8 years
INDUSTRIES = ["Electronics", "Pharmaceuticals", "Machinery", "Consumer"]  # four industries
firms = pd.DataFrame({  # firm-level static attributes
    "firm_id": [f"F{i:03d}" for i in range(1, N_FIRM + 1)],  # firm id
    "industry": RNG.choice(INDUSTRIES, N_FIRM),  # industry
    "size": RNG.choice(["Small", "Medium", "Large"], N_FIRM, p=[0.45, 0.35, 0.2]),  # size tier
    "base_rev": RNG.lognormal(mean=2.2, sigma=0.6, size=N_FIRM),  # initial revenue (100 million CNY)
    "growth": RNG.normal(0.09, 0.05, N_FIRM),  # firm-level growth rate
    "leverage": RNG.uniform(0.15, 0.75, N_FIRM).round(3),  # debt-to-asset ratio
})  # end of the firm static-attributes table
rows = []  # rows of the long panel table
for _, f in firms.iterrows():  # expand years for each firm
    for year in range(2018, 2018 + N_YEAR):  # 8 years
        rev = f["base_rev"] * (1 + f["growth"]) ** (year - 2018) * np.exp(RNG.normal(0, 0.06))  # revenue evolution
        roa = (0.075 - 0.05 * (f["leverage"] - 0.45) + RNG.normal(0, 0.012))  # return on assets: lower when leverage is high
        rows.append({"firm_id": f["firm_id"], "industry": f["industry"],  # id and industry
                     "size": f["size"], "year": year,  # size and year
                     "revenue": round(float(rev), 3),  # revenue
                     "roa": round(float(roa), 4),  # return on assets
                     "leverage": f["leverage"]})  # leverage (firm level, constant across years)
panel = pd.DataFrame(rows)  # assemble into a panel
dump(panel, "l4_firm_panel.csv")  # write the sample to disk

numeric_cols = ["revenue", "roa", "leverage"]  # numeric columns used in the correlation analysis
corr = panel[numeric_cols].corr().round(3)  # three-variable correlation matrix
print("Correlation matrix:\n", corr.to_string())  # print to the terminal
cond = (panel.groupby(["industry", "size"])["roa"].agg(["mean", "count"]).round(4).reset_index())  # conditional means by industry x size
dump(cond, "l4_conditional_roa.csv")  # write the conditional means to disk
print("Conditional means (industry x size, first 6 rows):\n", cond.head(6).to_string(index=False))  # print a snippet

# ---------------------------------------------------------------- figure: four multivariate views
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.7))  # heatmap / grouped scatter / small multiples / conditional distribution
im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)  # panel 1: correlation heatmap (diverging palette)
axes[0].set_xticks(range(len(numeric_cols)), ["Revenue", "ROA", "Leverage"], fontsize=9)  # x-axis labels
axes[0].set_yticks(range(len(numeric_cols)), ["Revenue", "ROA", "Leverage"], fontsize=9)  # y-axis labels
axes[0].grid(False)  # a heatmap does not need a grid
for i in range(len(numeric_cols)):  # write a value in each cell
    for j in range(len(numeric_cols)):  # double loop over the matrix
        axes[0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8.6,  # write the correlation in each heatmap cell
                     color="white" if abs(corr.values[i, j]) > 0.6 else "#1E2630")  # white text on dark cells
axes[0].set_title("Correlation matrix (diverging palette)")  # subplot title
plt.colorbar(im, ax=axes[0], fraction=0.046)  # colorbar
for ind, c in zip(INDUSTRIES, ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):  # color by industry
    sub = panel[panel["industry"] == ind]  # take this industry's subsample
    axes[1].scatter(sub["leverage"], sub["roa"], s=12, color=c, alpha=0.5, label=ind)  # panel 2: grouped scatter
axes[1].axhline(panel["roa"].mean(), color="#1E2630", lw=1.0, ls="--")  # overall-mean reference line
axes[1].set_title("Does grouping change the relationship")  # subplot title
axes[1].set_xlabel("debt-to-asset ratio"); axes[1].set_ylabel("ROA")  # axis labels
axes[1].legend(fontsize=8.2, ncol=2)  # two-column legend
for k, ind in enumerate(INDUSTRIES):  # small multiples: one panel per industry
    sub = panel[panel["industry"] == ind].groupby("year")["revenue"].mean()  # industry yearly mean
    axes[2].plot(sub.index, sub.values, "-o", ms=3, lw=1.5,  # draw the industry trend line
                 color=["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"][k], label=ind)  # color and legend
axes[2].set_title("Small multiples: industry revenue trends")  # subplot title
axes[2].set_xlabel("year"); axes[2].set_ylabel("average revenue (100 million CNY)")  # axis labels
axes[2].legend(fontsize=8.2)  # legend
data = [panel.loc[panel["industry"] == ind, "roa"].values for ind in INDUSTRIES]  # ROA by industry
bp = axes[3].boxplot(data, patch_artist=True)  # panel 4: conditional distribution (boxplot)
axes[3].set_xticklabels(INDUSTRIES)  # group labels (use set_xticklabels from 3.9 on)
for patch, c in zip(bp["boxes"], ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):  # color the boxes
    patch.set_facecolor(c); patch.set_alpha(0.55)  # set the fill color and transparency
axes[3].set_title("Conditional distribution: ROA by industry")  # subplot title
axes[3].set_ylabel("ROA")  # axis label
plt.tight_layout()  # adjust the four subplots
save(fig, "fig_multivariate.png")  # save: one-page multivariate figure
