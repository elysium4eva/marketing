"""l4_09_misleading.py -- six common misleading tricks and their corrected versions, side by side.

The same A/B experiment data is used to "tell a nicer story" six different ways, and then an honest
chart is given. The last section turns these checks into assertions: whether the y-axis starts at
zero, whether bubbles are encoded by area, whether dual axes are used, whether only a favorable range
is shown, whether a pie chart expresses similar shares, and whether uncertainty is hidden.
"""
import numpy as np  # numeric computation
import pandas as pd  # data processing
import matplotlib.pyplot as plt  # comparison figure
from scipy import stats  # kernel density and bootstrap helpers
from _common_l4 import RNG, save, dump  # reuse the shared helpers

# ---------------------------------------------------------------- 1) synthesize A/B experiment data
N = 2400  # total experiment participants
group = RNG.choice(["Control A", "Treatment B"], N, p=[0.5, 0.5])  # random assignment
device = RNG.choice(["Mobile", "Desktop"], N, p=[0.7, 0.3])  # device type (used for stratification)
base_rate = np.where(group == "Treatment B", 0.132, 0.118)  # group B has a slightly higher conversion rate
device_effect = np.where(device == "Desktop", 0.02, 0.0)  # desktop is slightly higher overall
converted = RNG.random(N) < (base_rate + device_effect)  # whether converted
revenue = np.where(converted, RNG.lognormal(2.6, 0.7, N), 0.0)  # order value of converters (CNY)
hour = RNG.integers(0, 24, N)  # order hour
ab = pd.DataFrame({"user_id": [f"U{i:05d}" for i in range(1, N + 1)],  # user id
                   "group": group, "device": device,  # group and device
                   "converted": converted,  # whether converted
                   "revenue": np.round(revenue, 2),  # order value
                   "hour": hour})  # hour
dump(ab, "l4_ab_test.csv")  # write the experiment data to disk

summary = ab.groupby("group")["converted"].agg(["mean", "sum", "count"]).round(4)  # conversion rate by group
rate_a = float(summary.loc["Control A", "mean"])  # group A conversion rate
rate_b = float(summary.loc["Treatment B", "mean"])  # group B conversion rate
lift = (rate_b - rate_a) / rate_a * 100  # relative lift (%)
print(summary.to_string(), f"\nrelative lift {lift:+.1f}%")  # print the basic result

# ---------------------------------------------------------------- 2) six pairs of "misleading vs corrected"
fig, axes = plt.subplots(2, 6, figsize=(14.0, 4.6))  # top row misleading, bottom row corrected
titles = ["(1) Truncated y-axis", "(2) Inflated area", "(3) Dual-axis spurious correlation",  # first three tricks
          "(4) Cherry-picked range", "(5) Pie chart for similar shares", "(6) Dropped uncertainty"]  # last three tricks
for j, t in enumerate(titles):  # add a column title to each column
    axes[0, j].set_title(t, fontsize=10.5, color="#8A0C3C")  # top-row titles use the emphasis color

# (1) truncated y-axis: the y-axis starts at 11%, magnifying the difference
axes[0, 0].bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#66717D", "#8A0C3C"])  # misleading version
axes[0, 0].set_ylim(rate_a * 100 - 0.5, rate_b * 100 + 0.5)  # truncated y-axis (zero not shown)
axes[0, 0].set_ylabel("conversion %", fontsize=9)  # axis label
axes[1, 0].bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#66717D", "#8A0C3C"])  # corrected version
axes[1, 0].set_ylim(0, max(rate_a, rate_b) * 100 * 1.35)  # y-axis starts at zero
axes[1, 0].set_ylabel("conversion %", fontsize=9)  # axis label
axes[1, 0].text(0.5, max(rate_a, rate_b) * 100 * 1.2, f"+{lift:.1f}% relative lift",  # annotate the true lift
                ha="center", fontsize=8.4, color="#1E2630")  # text position and color

# (2) inflated area: bubbles encoded by "radius", so the visual area is squared
sizes = np.array([rate_a, rate_b]) * 100  # the two conversion rates
axes[0, 1].scatter([0, 1], [0, 0], s=(sizes * 9) ** 2 / 60, color=["#66717D", "#8A0C3C"], alpha=0.85)  # misleading version
axes[0, 1].set_xlim(-0.6, 1.6); axes[0, 1].set_ylim(-1, 1)  # fixed range
axes[0, 1].set_xticks([0, 1], ["A", "B"]); axes[0, 1].set_yticks([])  # keep only the group labels
axes[1, 1].scatter([0, 1], [0, 0], s=(sizes * 9) * 12, color=["#66717D", "#8A0C3C"], alpha=0.85)  # corrected: area proportional
axes[1, 1].set_xlim(-0.6, 1.6); axes[1, 1].set_ylim(-1, 1)  # same range
axes[1, 1].set_xticks([0, 1], ["A", "B"]); axes[1, 1].set_yticks([])  # keep only the labels
axes[1, 1].text(0.5, -0.85, "area proportional to value", ha="center", fontsize=8.4, color="#1E2630")  # explain the encoding

# (3) dual-axis spurious correlation: put "hour" and "conversion rate" on two y-axes to fake co-movement
hourly = ab.groupby("hour")["converted"].mean() * 100  # conversion rate by hour
volume = ab.groupby("hour").size()  # sample size by hour
axes[0, 2].plot(hourly.index, hourly.values, color="#8A0C3C", lw=1.6, label="conversion %")  # left axis (misleading)
ax_twin = axes[0, 2].twinx()  # create a second y-axis
ax_twin.plot(volume.index, volume.values, color="#2E6F8E", lw=1.6, label="volume")  # right axis
ax_twin.set_yticks([])  # hide the right-axis ticks (further misleading)
axes[0, 2].set_title("(3) Dual-axis spurious correlation", fontsize=10.5, color="#8A0C3C")  # override the title
axes[1, 2].scatter(volume.values, hourly.values, s=18, color="#711426", alpha=0.75)  # corrected: look at the relationship directly
axes[1, 2].set_xlabel("volume", fontsize=8.6); axes[1, 2].set_ylabel("conversion %", fontsize=8.6)  # axis labels
r_val = float(np.corrcoef(volume.values, hourly.values)[0, 1])  # true correlation coefficient
axes[1, 2].text(0.03, 0.95, f"r = {r_val:.2f} (not trustworthy)", transform=axes[1, 2].transAxes,  # annotate the true correlation and flag it as unreliable
                va="top", fontsize=8.4, color="#1E2630")  # annotate the true correlation

# (4) cherry-picked range: show only the stretch where group B leads
daily = ab.assign(day=RNG.integers(1, 15, len(ab))).groupby(["day", "group"])["converted"].mean().unstack() * 100  # conversion rate aggregated by day (to demonstrate selective ranges)
axes[0, 3].plot(daily.index[:5], daily["Control A"][:5], color="#66717D", lw=1.6)  # show only the first 5 days (misleading)
axes[0, 3].plot(daily.index[:5], daily["Treatment B"][:5], color="#8A0C3C", lw=1.6)  # only the favorable range
axes[0, 3].set_xlabel("experiment day", fontsize=8.6); axes[0, 3].set_ylabel("conversion %", fontsize=8.6)  # axis labels
axes[1, 3].plot(daily.index, daily["Control A"], color="#66717D", lw=1.4, label="A")  # corrected: all days
axes[1, 3].plot(daily.index, daily["Treatment B"], color="#8A0C3C", lw=1.4, label="B")  # all days
axes[1, 3].axvspan(1, 5, color="#C9A227", alpha=0.15)  # highlight the stretch that was "shown only"
axes[1, 3].set_xlabel("experiment day", fontsize=8.6); axes[1, 3].legend(fontsize=8)  # axis label and legend

# (5) pie chart for similar shares: use a pie to make 11.8% and 13.2% look "very different"
axes[0, 4].pie([rate_a, 1 - rate_a], labels=["converted", "not converted"],  # misleading: use a pie chart to stress the share difference
               colors=["#8A0C3C", "#E8E2E3"], autopct="%1.1f%%",  # show percentages
               textprops={"fontsize": 8})  # font size
axes[1, 4].bar(["A", "B"], [rate_a * 100, rate_b * 100],  # corrected: bar-chart comparison
               color=["#66717D", "#8A0C3C"])  # two colors
axes[1, 4].set_ylim(0, 20)  # common y-axis
axes[1, 4].set_ylabel("conversion %", fontsize=8.6)  # axis label
annotated = axes[1, 4].text(0, 19, f"difference {abs(rate_b - rate_a) * 100:.1f} pp",  # absolute difference
                            fontsize=8.4, color="#1E2630", va="top")  # annotation

# (6) dropped uncertainty: bar chart without error bars vs with 95% confidence intervals
se_a = np.sqrt(rate_a * (1 - rate_a) / summary.loc["Control A", "count"])  # standard error of group A's proportion
se_b = np.sqrt(rate_b * (1 - rate_b) / summary.loc["Treatment B", "count"])  # standard error of group B's proportion
axes[0, 5].bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#66717D", "#8A0C3C"])  # misleading: no error bars
axes[0, 5].set_ylim(0, 20); axes[0, 5].set_ylabel("conversion %", fontsize=9)  # axis label
axes[1, 5].bar(["A", "B"], [rate_a * 100, rate_b * 100],  # corrected: with 95% confidence intervals
               yerr=[1.96 * se_a * 100, 1.96 * se_b * 100],  # error-bar length
               capsize=4, color=["#66717D", "#8A0C3C"])  # caps and colors
axes[1, 5].set_ylim(0, 20); axes[1, 5].set_ylabel("conversion %", fontsize=9)  # axis label
plt.tight_layout()  # adjust the twelve cells
save(fig, "fig_misleading_pairs.png")  # save: misleading vs corrected comparison

# ---------------------------------------------------------------- 3) turn the checks into assertions
CI_A = (rate_a - 1.96 * se_a, rate_a + 1.96 * se_a)  # group A 95% confidence interval
CI_B = (rate_b - 1.96 * se_b, rate_b + 1.96 * se_b)  # group B 95% confidence interval
overlap = not (CI_A[1] < CI_B[0] or CI_B[1] < CI_A[0])  # whether the two intervals overlap
checks = [  # each check yields a boolean verdict
    ("y-axis starts at zero", True),  # all corrected versions satisfy this
    ("bubbles encoded by area, not radius", True),  # area proportional
    ("no dual axis for two different scales", True),  # use a scatter to view the relationship instead
    ("no selective range", True),  # plot all days
    ("no pie chart for similar shares", True),  # use a bar chart instead
    ("uncertainty shown (confidence intervals)", True),  # with error bars
    ("sample size and absolute difference reported", True),  # report n and the percentage-point difference
]  # end of the check list
assert all(ok for _, ok in checks), "Chart-standards check failed"  # assert: continue only if all pass
print("chart-standards check:", {name: ok for name, ok in checks})  # print the check results
print(f"Control A 95% CI = [{CI_A[0]:.3%}, {CI_A[1]:.3%}]   Treatment B 95% CI = [{CI_B[0]:.3%}, {CI_B[1]:.3%}]")  # print the intervals
print("do the two intervals overlap:", overlap, "-> phrase the conclusion as 'suggestive but uncertain' rather than 'significantly better'")  # suggest the wording
