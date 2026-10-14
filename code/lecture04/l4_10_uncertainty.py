"""l4_10_uncertainty.py -- how to show uncertainty: error bars, bootstrap intervals, confidence bands, and fan forecasts.

"Not showing uncertainty" is itself a form of misleading. Using a bootstrap, the script estimates
intervals for the conversion rate and the order value from the same A/B data, demonstrates four
common ways to express them, and saves the results as a table for easy inclusion in reports.
"""
import numpy as np  # numeric computation and resampling
import pandas as pd  # data processing
import matplotlib.pyplot as plt  # four views
from _common_l4 import RNG, DAT, save, dump  # reuse the shared helpers

ab = pd.read_csv(DAT / "l4_ab_test.csv")  # read the A/B experiment data (from l4_09)
B = 2000  # number of bootstrap repetitions
def bootstrap_ci(values, stat=np.mean, level=0.95, boot=B):  # generic bootstrap-interval function
    """Resample a given sample and return the point estimate and percentile confidence interval."""  # function contract
    values = np.asarray(values, dtype=float)  # convert to a float array
    idx = RNG.integers(0, len(values), size=(boot, len(values)))  # draw resample indices of the same size each time
    stats_ = stat(values[idx], axis=1)  # compute the statistic for each resample
    lo, hi = np.percentile(stats_, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])  # percentile interval
    return float(stat(values)), float(lo), float(hi), stats_  # return the point estimate, interval, and distribution

rows = []  # summary table
boot_store = {}  # keep the bootstrap distributions for plotting
for g in ["Control A", "Treatment B"]:  # compute for each group
    sub = ab[ab["group"] == g]  # take this group's sample
    m, lo, hi, dist = bootstrap_ci(sub["converted"])  # bootstrap interval for the conversion rate
    rows.append({"metric": f"{g} · conversion rate", "point_estimate": round(m, 4),  # record the point estimate
                 "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),  # record the confidence interval
                 "n": int(len(sub))})  # record the sample size
    boot_store[g] = (m, lo, hi, dist)  # stash the bootstrap distribution
    rev = sub.loc[sub["converted"], "revenue"]  # take only the converters' order values
    m2, lo2, hi2, dist2 = bootstrap_ci(rev)  # bootstrap interval for the order value
    rows.append({"metric": f"{g} · order value", "point_estimate": round(m2, 2),  # record the order-value point estimate
                 "ci_lower": round(lo2, 2), "ci_upper": round(hi2, 2),  # order-value interval
                 "n": int(len(rev))})  # order-value sample size
    boot_store[g + " · order value"] = (m2, lo2, hi2, dist2)  # stash it
diff_dist = boot_store["Treatment B"][3] - boot_store["Control A"][3]  # bootstrap distribution of the group difference
d_point, d_lo, d_hi = float(diff_dist.mean()), float(np.percentile(diff_dist, 2.5)), float(np.percentile(diff_dist, 97.5))  # mean and 95% percentile interval of the difference distribution
rows.append({"metric": "conversion-rate difference (B-A)", "point_estimate": round(d_point, 4),  # difference point estimate
             "ci_lower": round(d_lo, 4), "ci_upper": round(d_hi, 4),  # difference interval
             "n": int(len(ab))})  # total sample size
unc = pd.DataFrame(rows)  # summarize into a table
dump(unc, "l4_uncertainty_summary.csv")  # write the uncertainty summary to disk
print(unc.to_string(index=False))  # print to the terminal
crosses_zero = d_lo < 0 < d_hi  # whether the difference interval crosses 0
print(f"difference 95% CI = [{d_lo:+.4f}, {d_hi:+.4f}], crosses 0: {crosses_zero}")  # print the conclusion hint

# ---------------------------------------------------------------- figure: four views of uncertainty
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.7))  # error bars / bootstrap distribution / confidence band / fan forecast
mA, loA, hiA, _ = boot_store["Control A"]  # group A conversion rate
mB, loB, hiB, _ = boot_store["Treatment B"]  # group B conversion rate
axes[0].errorbar([0, 1], [mA * 100, mB * 100],  # point estimate + error bars
                 yerr=[[(mA - loA) * 100, (mB - loB) * 100],  # lower error
                       [(hiA - mA) * 100, (hiB - mB) * 100]],  # upper error
                 fmt="o", capsize=5, color="#8A0C3C", ms=7)  # markers and caps
axes[0].set_xticks([0, 1], ["Control A", "Treatment B"])  # group labels
axes[0].set_ylim(0, max(hiA, hiB) * 100 * 1.25)  # leave room
axes[0].set_title("(1) Point estimate + 95% error bars")  # subplot title
axes[0].set_ylabel("conversion %")  # axis label
axes[1].hist(diff_dist * 100, bins=40, color="#2E6F8E", alpha=0.85)  # (2) bootstrap distribution of the difference
axes[1].axvline(0, color="#1E2630", lw=1.4, ls="--")  # zero line
axes[1].axvline(d_lo * 100, color="#C9A227", lw=1.2)  # interval lower bound
axes[1].axvline(d_hi * 100, color="#C9A227", lw=1.2, label="95% interval")  # interval upper bound
axes[1].set_title("(2) Bootstrap: distribution of the difference")  # subplot title
axes[1].set_xlabel("conversion-rate difference (pp)"); axes[1].legend(fontsize=8.4)  # axis label and legend
daily = ab.assign(day=RNG.integers(1, 21, len(ab))).groupby(["day", "group"])["converted"].mean().unstack()  # conversion rate aggregated by day (for the confidence band)
band_lo, band_hi = [], []  # (3) per-day bootstrap band
for d in daily.index:  # compute the interval for each day
    v = daily.loc[d, "Treatment B"]  # group B conversion rate that day
    m, lo, hi, _ = bootstrap_ci(ab.loc[(ab["group"] == "Treatment B"), "converted"], level=0.8)  # estimate the interval from the full sample
    band_lo.append(max(lo * 100, 0)); band_hi.append(hi * 100)  # record the lower and upper bounds (as percentages)
axes[2].plot(daily.index, daily["Treatment B"] * 100, "-o", ms=3, color="#8A0C3C", lw=1.4)  # line
axes[2].fill_between(daily.index, band_lo, band_hi, color="#8A0C3C", alpha=0.18, label="80% interval")  # confidence band
axes[2].set_title("(3) Time series + confidence band")  # subplot title
axes[2].set_xlabel("experiment day"); axes[2].set_ylabel("conversion %")  # axis labels
axes[2].legend(fontsize=8.4)  # legend
months = np.arange(1, 25)  # (4) forecast for the next 24 periods
center = 12.0 + 0.18 * months  # central forecast (slowly rising)
width = 0.9 + 0.42 * months  # forecast uncertainty widens over time
axes[3].plot(months, center, color="#8A0C3C", lw=1.8, label="central forecast")  # central line
axes[3].fill_between(months, center - width, center + width, color="#C9A227", alpha=0.25, label="80% prediction interval")  # 80% band
axes[3].fill_between(months, center - 2 * width, center + 2 * width, color="#C9A227", alpha=0.15, label="95% prediction interval")  # 95% band
axes[3].set_title("(4) Fan forecast: more uncertain further out")  # subplot title
axes[3].set_xlabel("future period n"); axes[3].set_ylabel("metric value")  # axis labels
axes[3].legend(fontsize=8.2)  # legend
plt.tight_layout()  # adjust the four cells
save(fig, "fig_uncertainty.png")  # save: four views of uncertainty
