"""01_datatype_numeric.py -- numeric data (continuous vs discrete): sample generation + visualization."""  # module comment
import numpy as np                                                    # numerical computing
import pandas as pd                                                   # tabular data handling
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, save, dump_csv                               # reuse shared helpers (seed / save / dump)

# ---------------------------------------------------------------- 1) generate samples
n = 800                                                               # sample size for the continuous variable: 800 trading days
mu, sigma = 0.0004, 0.018                                             # mean and volatility of daily returns (log returns)
ret = RNG.normal(mu, sigma, n)                                        # continuous sample: daily log returns (roughly normal)
volume = RNG.lognormal(mean=14.0, sigma=0.5, size=n)                  # continuous sample: trading volume (lognormal, right-skewed)
orders = RNG.poisson(lam=12, size=n)                                  # discrete sample: daily order count (Poisson counts)

df = pd.DataFrame({"day": np.arange(1, n + 1),                       # assemble into one table: which trading day
                   "log_return": ret,                                 # continuous var (1): daily log return
                   "volume": volume,                                  # continuous var (2): trading volume (right-skewed, long tail)
                   "orders": orders})                                 # discrete var: order count (non-negative integer)
dump_csv(df, "numeric_samples.csv")                                   # write the samples into the repo data/ folder

# ---------------------------------------------------------------- 2) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                   # one row, three columns: histogram / box plot / count bar chart
axes[0].hist(ret, bins=40, color="#8A0C3C", alpha=0.85)               # left: histogram of a continuous variable (shows the distribution shape)
axes[0].set_title("Continuous: daily log return distribution")        # set the left title
axes[0].set_xlabel("Return")                                          # set the left x-axis meaning
axes[0].set_ylabel("Frequency")                                       # set the left y-axis meaning
axes[0].axvline(ret.mean(), color="#C9A227", lw=1.6,                 # draw a mean reference line (gold)
                label="mean %.4f" % ret.mean())                       # show the mean value in the legend
axes[0].legend(fontsize=8)                                            # show the legend

axes[1].boxplot([ret, np.log(volume)], showfliers=True,               # middle: box plot comparing two continuous variables
                patch_artist=True,                                    # allow a custom box fill color
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))       # box color and opacity
axes[1].set_xticklabels(["Return", "log Volume"])                     # add readable labels to the two box groups
axes[1].set_title("Continuous: box plot (outliers)")                  # set the middle title

axes[2].bar(np.arange(1, 26), [np.sum(orders == k) for k in range(1, 26)],  # right: discrete count distribution (1-25 orders)
            color="#66717D", width=0.8)                               # gray bars show the frequency of each value
axes[2].set_title("Discrete: daily order count (counts)")             # set the right title
axes[2].set_xlabel("Order count (integer values)")                    # set the right x-axis meaning
plt.tight_layout()                                                    # auto-adjust subplot spacing to avoid overlap
save(fig, "fig_numeric.png")                                          # save the figure to website/figures/
