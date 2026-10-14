"""l4_12_report_pipeline.py -- report pipeline: produce statistical tables, figures, and captions together.

Assemble the results of the previous steps in "report order": descriptive-statistics table ->
group comparison table -> figure -> caption text, all written to data/lecture04/report/ so they can
be copied straight into a report, with each caption spelling out the data source, sample size,
definitions, and uncertainty -- exactly what "self-contained captions" require.
"""
import numpy as np  # numeric computation
import pandas as pd  # table handling
import matplotlib.pyplot as plt  # report figures
from _common_l4 import DAT, save, dump  # reuse the shared helpers

REPORT = DAT / "report"  # report-materials directory
REPORT.mkdir(parents=True, exist_ok=True)  # create it if missing

# ---------------------------------------------------------------- 1) read in the samples generated earlier
survey = pd.read_csv(DAT / "l4_survey_raw.csv")  # survey sample (l4_01)
ab = pd.read_csv(DAT / "l4_ab_test.csv")  # experiment data (l4_09)
unc = pd.read_csv(DAT / "l4_uncertainty_summary.csv")  # uncertainty summary (l4_10)

# ---------------------------------------------------------------- 2) table 1: descriptive statistics
desc = survey[["age", "income_k", "satisfaction", "spend_k"]].describe().T.round(3)  # common descriptive statistics
desc["missing_rate"] = survey[["age", "income_k", "satisfaction", "spend_k"]].isna().mean().round(4)  # add the missing rate
desc = desc.reset_index(names="variable")  # turn the variable name into a column
desc.to_csv(REPORT / "table1_descriptive.csv", index=False, encoding="utf-8-sig")  # write table 1 to disk
print("[data]", REPORT / "table1_descriptive.csv")  # print the artifact path

# ---------------------------------------------------------------- 3) table 2: group comparison (channel x satisfaction)
by_channel = (survey.groupby("channel")  # group by channel
              .agg(n=("respondent_id", "count"),  # sample size per group
                   satisfaction_mean=("satisfaction", "mean"),  # mean satisfaction
                   income_median=("income_k", "median"),  # median income (robust)
                   repurchase_rate=("revisit", "mean"))  # repurchase rate
              .round(3).reset_index())  # keep three decimal places
by_channel.to_csv(REPORT / "table2_channel.csv", index=False, encoding="utf-8-sig")  # write table 2 to disk
print("[data]", REPORT / "table2_channel.csv")  # print the artifact path

# ---------------------------------------------------------------- 4) figure: one-page report main figure (four panels)
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.6))  # four panels in the report main figure
axes[0].bar(by_channel["channel"], by_channel["satisfaction_mean"],  # (1) satisfaction by channel
            color="#8A0C3C", alpha=0.88)  # single color to avoid color noise
axes[0].set_ylim(2.5, 5.0)  # focus on the difference range and annotate the starting point
axes[0].set_title("Mean satisfaction by channel"); axes[0].set_ylabel("satisfaction (1-5)")  # title and axis label
axes[0].text(0.02, 4.9, "y-axis starts at 2.5, only to compare differences", fontsize=7.8, color="#66717D", va="top")  # state the truncation explicitly
axes[1].hist(survey["income_k"].dropna(), bins=30, color="#2E6F8E", alpha=0.85)  # (2) income distribution
axes[1].axvline(survey["income_k"].median(), color="#C9A227", lw=1.6, label="median")  # median line
axes[1].set_title("Income distribution (12 missing, not imputed)")  # title states how missing values were handled
axes[1].set_xlabel("monthly income (thousand CNY)"); axes[1].legend(fontsize=8)  # axis label and legend
rate = ab.groupby("group")["converted"].mean() * 100  # (3) experiment conversion rate
se = np.sqrt((rate / 100) * (1 - rate / 100) / ab.groupby("group").size()) * 100 * 1.96  # 95% error
axes[2].bar(rate.index, rate.values, yerr=se.values, capsize=5,  # bar chart with error bars
            color=["#66717D", "#8A0C3C"], alpha=0.9)  # two groups, two colors
axes[2].set_ylim(0, 20); axes[2].set_title("A/B conversion rate (with 95% error bars)")  # title
axes[2].set_ylabel("conversion %")  # axis label
axes[2].text(0.02, 19, f"n={len(ab):,}", fontsize=8, color="#66717D", va="top")  # annotate the sample size
ci = unc[unc["metric"].str.contains("order value")]  # (4) interval comparison for the order value
ypos = np.arange(len(ci))  # y coordinates
axes[3].errorbar(ci["point_estimate"], ypos,  # point estimate + interval
                 xerr=[ci["point_estimate"] - ci["ci_lower"], ci["ci_upper"] - ci["point_estimate"]],  # left/right errors
                 fmt="o", capsize=4, color="#711426")  # markers and caps
axes[3].set_yticks(ypos, ci["metric"], fontsize=8.4)  # group-name labels
axes[3].set_title("Order-value point estimate and 95% interval"); axes[3].set_xlabel("CNY")  # title and axis label
plt.tight_layout()  # adjust the four panels
save(fig, "fig_report_panel.png")  # save: report main figure

# ---------------------------------------------------------------- 5) captions: the self-contained style
captions = [  # every caption can be understood on its own
    "Figure 1  Mean satisfaction by channel (data: l4_survey_raw.csv, n=500; satisfaction on a 1-5 scale; "  # figure 1 caption (data source, sample size, definitions)
    "5 missing values kept as missing with no imputation; the y-axis starts at 2.5, only to compare differences across channels).",  # figure 1 caption, continued
    "Figure 2  Respondent monthly income distribution (n=500; income in thousand CNY, right-skewed; 12 sensitive-item values missing and not imputed; "  # figure 2 caption
    "the median marks the center in the chart).",  # figure 2 caption, continued
    "Figure 3  A/B experiment conversion-rate comparison (n=2400, randomly split in half; error bars are 95% confidence intervals; "  # figure 3 caption
    "conclusion: 'the treatment group shows a hint of improvement, but its interval overlaps the control group's, so a larger sample is needed to decide').",  # figure 3 caption, continued (states the uncertainty conclusion)
    "Figure 4  Point estimate and 95% bootstrap confidence interval of converters' order value (B=2000 resamples; "  # figure 4 caption
    "order value is right-skewed, so the median is reported as well).",  # figure 4 caption, continued
]  # end of the caption list
(REPORT / "captions.md").write_text("# Captions (self-contained style example)\n\n" + "\n\n".join(captions), encoding="utf-8")
print("[data]", REPORT / "captions.md")  # print the artifact path
print("report pack:", sorted(p.name for p in REPORT.iterdir()))  # list the report materials
